"""
DuplexModel: wraps a frozen Qwen3-1.7B-Base with workspace adapter layers.

Architecture:
    1. Qwen3 decoder layers are frozen, loaded in bfloat16
    2. CrossAttentionAdapter injected after self-attention in each layer
    3. UpdateEncoder encodes prompt/correction text
    4. WorkspaceModule maintains and updates latent state
    5. Decoder cross-attends to workspace via adapters

At initialization (zero-init adapters), the model is identical to vanilla Qwen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import CrossAttentionAdapter
from .workspace import WorkspaceModule
from .encoder import UpdateEncoder
from .config import DuplexConfig


class DuplexModel(nn.Module):
    def __init__(self, config: DuplexConfig, local_rank: int = 0, compile_modules: bool = False):
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.qwen_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Qwen with SDPA (scaled dot-product attention — fast built-in kernel)
        self.qwen = AutoModelForCausalLM.from_pretrained(
            config.qwen_model_path,
            dtype=torch.bfloat16,
            device_map={"": device_str},
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        for param in self.qwen.parameters():
            param.requires_grad = False

        # Trainable adapter components
        encoder_head_dim = config.encoder_dim // config.adapter_n_heads
        self.encoder = UpdateEncoder(
            vocab_size=config.vocab_size,
            d_model=config.encoder_dim,
            n_heads=config.adapter_n_heads,
            head_dim=encoder_head_dim,
            d_ff=config.encoder_ff_dim,
            n_layers=config.n_encoder_layers,
            dropout=config.adapter_dropout,
        )
        if config.encoder_dim != config.workspace_dim:
            self.encoder.set_output_projection(config.workspace_dim)

        self.workspace = WorkspaceModule(
            n_slots=config.n_workspace_slots,
            d_model=config.workspace_dim,
            n_heads=config.adapter_n_heads,
            head_dim=config.head_dim,
            dropout=config.adapter_dropout,
        )

        adapter_head_dim = config.d_model // config.adapter_n_heads
        self.adapters = nn.ModuleList([
            CrossAttentionAdapter(
                d_model=config.d_model,
                n_heads=config.adapter_n_heads,
                head_dim=adapter_head_dim,
                dropout=config.adapter_dropout,
            )
            for _ in range(config.n_decoder_layers)
        ])

        # Move new components to correct device/dtype
        self.encoder = self.encoder.to(device=device_str, dtype=torch.bfloat16)
        self.workspace = self.workspace.to(device=device_str, dtype=torch.bfloat16)
        self.adapters = self.adapters.to(device=device_str, dtype=torch.bfloat16)

        # torch.compile the trainable hot-path modules for extra speed
        if compile_modules:
            self.encoder = torch.compile(self.encoder)
            self.workspace = torch.compile(self.workspace)
            self.adapters = torch.compile(self.adapters)

        self._inject_adapters()
        self._current_workspace: torch.Tensor | None = None

    def _inject_adapters(self):
        """Monkey-patch each Qwen decoder layer to include cross-attention to workspace."""
        for i, layer in enumerate(self.qwen.model.layers):
            adapter = self.adapters[i]
            original_forward = layer.forward

            def make_patched_forward(orig_fn, adapter_module):
                def patched_forward(*args, **kwargs):
                    output = orig_fn(*args, **kwargs)
                    is_tuple = isinstance(output, tuple)
                    hidden_states = output[0] if is_tuple else output

                    ws = self._current_workspace
                    if ws is not None:
                        hidden_states = hidden_states + adapter_module(hidden_states, ws)

                    return (hidden_states,) + output[1:] if is_tuple else hidden_states
                return patched_forward

            layer.forward = make_patched_forward(original_forward, adapter)

    def _encode_and_update_workspace(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_output = self.encoder(text_ids, attention_mask=text_mask)
        return self.workspace(encoder_output, encoder_mask=text_mask, workspace=workspace)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_ids: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        update_ids: torch.Tensor | None = None,
        update_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ws = None
        if prompt_ids is not None:
            ws = self._encode_and_update_workspace(prompt_ids, prompt_mask)

        if update_ids is not None and update_ids.size(1) > 0:
            if update_mask is not None and update_mask.any():
                ws = self._encode_and_update_workspace(update_ids, update_mask, workspace=ws)

        self._current_workspace = ws

        # Qwen is frozen — inference_mode skips all activation tracking for it,
        # giving a meaningful speed boost while adapter grads still flow through
        # the patched layer.forward hooks
        with torch.inference_mode(mode=False):
            outputs = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        self._current_workspace = None

        result = {"logits": outputs.logits}
        if outputs.loss is not None:
            result["loss"] = outputs.loss
        return result

    @torch.no_grad()
    def generate_with_update(
        self,
        prompt_text: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        correction_text: str | None = None,
        correction_after_tokens: int | None = None,
    ) -> tuple[str, str | None]:
        self.eval()
        device = next(self.qwen.parameters()).device

        prompt_enc = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        ws = self._encode_and_update_workspace(
            prompt_enc["input_ids"], prompt_enc["attention_mask"]
        )

        correction_enc = None
        if correction_text and correction_after_tokens is not None:
            correction_enc = self.tokenizer(correction_text, return_tensors="pt").to(device)

        self._current_workspace = ws
        generated_ids = prompt_enc["input_ids"].clone()
        text_at_correction = None

        for step in range(max_new_tokens):
            if (correction_enc is not None
                    and correction_after_tokens is not None
                    and step == correction_after_tokens):
                text_at_correction = self.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
                ws = self._encode_and_update_workspace(
                    correction_enc["input_ids"],
                    correction_enc["attention_mask"],
                    workspace=ws,
                )
                self._current_workspace = ws

            outputs = self.qwen(input_ids=generated_ids)
            next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        self._current_workspace = None
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return full_text, text_at_correction

    def get_trainable_params(self) -> list[nn.Parameter]:
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.workspace.parameters())
        params.extend(self.adapters.parameters())
        return params

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_param_summary(self):
        total = self.total_param_count()
        trainable = self.trainable_param_count()
        frozen = total - trainable
        print(f"Total params:     {total:>12,}")
        print(f"Trainable params: {trainable:>12,}")
        print(f"Frozen params:    {frozen:>12,}")
        print(f"Trainable %:      {100 * trainable / total:>11.1f}%")
