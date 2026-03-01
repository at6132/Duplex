"""
DuplexModel v2: Frozen Qwen3-1.7B with dual-path adapter injection.

Architecture:
    1. Qwen3 decoder layers are frozen, loaded in bfloat16
    2. UpdateEncoder encodes prompt/correction text into per-token states
    3. WorkspaceModule compresses prompt context into latent slots (high-level)
    4. CrossAttentionAdapter v2 cross-attends to BOTH workspace slots AND
       per-token correction states (entity-level detail)
    5. Flamingo-style tanh(alpha) gating per layer for stable scaling
    6. Special tokens (<|REVISE_START|>, <|REVISE_END|>, <|INSERT|>) for
       retroactive revision

At initialization all gates are zero -> model is identical to vanilla Qwen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import CrossAttentionAdapter
from .workspace import WorkspaceModule
from .encoder import UpdateEncoder
from .config import DuplexConfig, SPECIAL_TOKENS


class DuplexModel(nn.Module):
    def __init__(self, config: DuplexConfig, local_rank: int = 0):
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

        # Tokenizer + special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.qwen_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        new_tokens = list(SPECIAL_TOKENS.values())
        n_added = self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

        self.revise_start_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["revise_start"])
        self.revise_end_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["revise_end"])
        self.insert_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["insert"])

        # Load Qwen
        self.qwen = AutoModelForCausalLM.from_pretrained(
            config.qwen_model_path,
            dtype=torch.bfloat16,
            device_map={"": device_str},
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        # Resize embeddings for special tokens
        if n_added > 0:
            self.qwen.resize_token_embeddings(len(self.tokenizer))

        for param in self.qwen.parameters():
            param.requires_grad = False

        # Encoder
        encoder_head_dim = config.encoder_dim // config.adapter_n_heads
        self.encoder = UpdateEncoder(
            vocab_size=len(self.tokenizer),
            d_model=config.encoder_dim,
            n_heads=config.adapter_n_heads,
            head_dim=encoder_head_dim,
            d_ff=config.encoder_ff_dim,
            n_layers=config.n_encoder_layers,
            dropout=config.adapter_dropout,
        )
        if config.encoder_dim != config.workspace_dim:
            self.encoder.set_output_projection(config.workspace_dim)

        # Workspace (high-level context slots)
        self.workspace = WorkspaceModule(
            n_slots=config.n_workspace_slots,
            d_model=config.workspace_dim,
            n_heads=config.adapter_n_heads,
            head_dim=config.head_dim,
            dropout=config.adapter_dropout,
        )

        # Adapters (dual-path cross-attention)
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

        # Flamingo-style tanh gates: one learnable scalar per layer, init to 0
        self.adapter_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(config.n_decoder_layers)
        ])

        # Move to device
        self.encoder = self.encoder.to(device=device_str, dtype=torch.bfloat16)
        self.workspace = self.workspace.to(device=device_str, dtype=torch.bfloat16)
        self.adapters = self.adapters.to(device=device_str, dtype=torch.bfloat16)
        self.adapter_gates = self.adapter_gates.to(device=device_str, dtype=torch.bfloat16)

        self._inject_adapters()

        # Runtime state for adapter hooks
        self._current_workspace: torch.Tensor | None = None
        self._current_correction_tokens: torch.Tensor | None = None
        self._current_correction_mask: torch.Tensor | None = None

    def _inject_adapters(self):
        """Monkey-patch each Qwen layer with dual-path cross-attention + tanh gating."""
        for i, layer in enumerate(self.qwen.model.layers):
            adapter = self.adapters[i]
            gate = self.adapter_gates[i]
            original_forward = layer.forward

            def make_patched_forward(orig_fn, adapter_module, gate_param):
                def patched_forward(*args, **kwargs):
                    output = orig_fn(*args, **kwargs)
                    is_tuple = isinstance(output, tuple)
                    hidden_states = output[0] if is_tuple else output

                    ws = self._current_workspace
                    if ws is not None:
                        adapter_out = adapter_module(
                            hidden_states, ws,
                            self._current_correction_tokens,
                            self._current_correction_mask,
                        )
                        hidden_states = hidden_states + torch.tanh(gate_param).to(hidden_states.dtype) * adapter_out

                    return (hidden_states,) + output[1:] if is_tuple else hidden_states
                return patched_forward

            layer.forward = make_patched_forward(original_forward, adapter, gate)

    def _encode_and_update_workspace(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text, update workspace, return (workspace, per_token_states)."""
        encoder_output = self.encoder(text_ids, attention_mask=text_mask)
        ws = self.workspace(encoder_output, encoder_mask=text_mask, workspace=workspace)
        return ws, encoder_output

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
        corr_tokens = None
        corr_mask = None

        if prompt_ids is not None:
            ws, _ = self._encode_and_update_workspace(prompt_ids, prompt_mask)

        if update_ids is not None and update_ids.size(1) > 0:
            if update_mask is not None and update_mask.any():
                corr_encoded = self.encoder(update_ids, attention_mask=update_mask)
                ws = self.workspace(corr_encoded, encoder_mask=update_mask, workspace=ws)
                corr_tokens = corr_encoded
                corr_mask = update_mask

        self._current_workspace = ws
        self._current_correction_tokens = corr_tokens
        self._current_correction_mask = corr_mask

        with torch.inference_mode(mode=False):
            outputs = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        self._current_workspace = None
        self._current_correction_tokens = None
        self._current_correction_mask = None

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
        ws, _ = self._encode_and_update_workspace(
            prompt_enc["input_ids"], prompt_enc["attention_mask"]
        )

        correction_enc = None
        if correction_text and correction_after_tokens is not None:
            correction_enc = self.tokenizer(correction_text, return_tensors="pt").to(device)

        self._current_workspace = ws
        self._current_correction_tokens = None
        self._current_correction_mask = None
        generated_ids = prompt_enc["input_ids"].clone()
        text_at_correction = None

        for step in range(max_new_tokens):
            if (correction_enc is not None
                    and correction_after_tokens is not None
                    and step == correction_after_tokens):
                text_at_correction = self.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
                corr_encoded = self.encoder(
                    correction_enc["input_ids"],
                    attention_mask=correction_enc["attention_mask"],
                )
                ws = self.workspace(
                    corr_encoded,
                    encoder_mask=correction_enc["attention_mask"],
                    workspace=ws,
                )
                self._current_workspace = ws
                self._current_correction_tokens = corr_encoded
                self._current_correction_mask = correction_enc["attention_mask"]

            outputs = self.qwen(input_ids=generated_ids)
            next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        self._current_workspace = None
        self._current_correction_tokens = None
        self._current_correction_mask = None
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        return full_text, text_at_correction

    INJECTION_VISUAL = (
        "\n\n"
        "=== DUPLEX RECEIVING INJECTION -- Workspace updating now ===\n"
        "=== (No stop. No restart. Continuing with new information.) ===\n\n"
    )

    @torch.no_grad()
    def generate_with_update_streaming(
        self,
        prompt_text: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        correction_text: str | None = None,
        correction_after_tokens: int | None = None,
    ):
        """Stream generation with mid-stream correction injection."""
        self.eval()
        device = next(self.qwen.parameters()).device

        prompt_enc = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        ws, _ = self._encode_and_update_workspace(
            prompt_enc["input_ids"], prompt_enc["attention_mask"]
        )

        correction_enc = None
        if correction_text and correction_after_tokens is not None:
            correction_enc = self.tokenizer(correction_text, return_tensors="pt").to(device)

        self._current_workspace = ws
        self._current_correction_tokens = None
        self._current_correction_mask = None
        generated_ids = prompt_enc["input_ids"].clone()

        for step in range(max_new_tokens):
            if (correction_enc is not None
                    and correction_after_tokens is not None
                    and step == correction_after_tokens):
                text_so_far = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                yield text_so_far + self.INJECTION_VISUAL

                corr_encoded = self.encoder(
                    correction_enc["input_ids"],
                    attention_mask=correction_enc["attention_mask"],
                )
                ws = self.workspace(
                    corr_encoded,
                    encoder_mask=correction_enc["attention_mask"],
                    workspace=ws,
                )
                self._current_workspace = ws
                self._current_correction_tokens = corr_encoded
                self._current_correction_mask = correction_enc["attention_mask"]

            outputs = self.qwen(input_ids=generated_ids)
            next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            full = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            yield full

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        self._current_workspace = None
        self._current_correction_tokens = None
        self._current_correction_mask = None

    def get_trainable_params(self) -> list[nn.Parameter]:
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.workspace.parameters())
        params.extend(self.adapters.parameters())
        params.extend(self.adapter_gates)
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
        gate_vals = [torch.tanh(g).item() for g in self.adapter_gates]
        print(f"Gate values:      min={min(gate_vals):.4f}, max={max(gate_vals):.4f}, mean={sum(gate_vals)/len(gate_vals):.4f}")
