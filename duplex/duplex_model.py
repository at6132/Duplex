"""
DuplexModel: wraps a frozen Qwen3-1.7B-Base with workspace adapter layers.

Architecture:
    1. Qwen3 decoder layers are frozen (optionally 4-bit quantized)
    2. CrossAttentionAdapter injected after self-attention in each layer
    3. UpdateEncoder encodes prompt/correction text
    4. WorkspaceModule maintains and updates latent state
    5. Decoder cross-attends to workspace via adapters

At initialization (zero-init adapters), the model is identical to vanilla Qwen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from functools import partial

from .adapter import CrossAttentionAdapter
from .workspace import WorkspaceModule
from .encoder import UpdateEncoder
from .config import DuplexConfig


class DuplexModel(nn.Module):
    def __init__(self, config: DuplexConfig):
        super().__init__()
        self.config = config

        # Load Qwen3
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.qwen_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if config.quantize_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.qwen = AutoModelForCausalLM.from_pretrained(
            config.qwen_model_path,
            quantization_config=quant_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Freeze all Qwen parameters
        for param in self.qwen.parameters():
            param.requires_grad = False

        # New trainable components
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
        # Project encoder output to workspace dim if sizes differ
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

        # Move new components to the same device/dtype as Qwen
        device = next(self.qwen.parameters()).device
        self.encoder = self.encoder.to(device=device, dtype=torch.bfloat16)
        self.workspace = self.workspace.to(device=device, dtype=torch.bfloat16)
        self.adapters = self.adapters.to(device=device, dtype=torch.bfloat16)

        # Inject adapters into Qwen's decoder layers
        self._inject_adapters()

        # Storage for current workspace state during generation
        self._current_workspace: torch.Tensor | None = None

    def _inject_adapters(self):
        """Monkey-patch each Qwen decoder layer to include cross-attention to workspace."""
        decoder_layers = self.qwen.model.layers

        for i, layer in enumerate(decoder_layers):
            adapter = self.adapters[i]
            original_forward = layer.forward

            def make_patched_forward(orig_fn, adapter_module):
                def patched_forward(*args, **kwargs):
                    output = orig_fn(*args, **kwargs)

                    # Qwen layers return a tuple: (hidden_states, ...)
                    is_tuple = isinstance(output, tuple)
                    if is_tuple:
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    # Apply cross-attention adapter if workspace is available
                    ws = self._current_workspace
                    if ws is not None:
                        adapter_out = adapter_module(hidden_states, ws)
                        hidden_states = hidden_states + adapter_out

                    if is_tuple:
                        return (hidden_states,) + output[1:]
                    return hidden_states

                return patched_forward

            layer.forward = make_patched_forward(original_forward, adapter)

    def _encode_and_update_workspace(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode text and update workspace."""
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
        """
        Training forward pass.

        Args:
            input_ids: (B, T) decoder input tokens (output to predict)
            attention_mask: (B, T) decoder attention mask
            prompt_ids: (B, T_p) prompt tokens for encoder
            prompt_mask: (B, T_p) prompt attention mask
            update_ids: (B, T_u) correction tokens for encoder (optional)
            update_mask: (B, T_u) correction attention mask (optional)
            labels: (B, T) target labels for loss computation

        Returns:
            dict with 'loss', 'logits'
        """
        # Initialize workspace from prompt
        ws = None
        if prompt_ids is not None:
            ws = self._encode_and_update_workspace(prompt_ids, prompt_mask)

        # Update workspace with correction if provided
        if update_ids is not None and update_ids.size(1) > 0:
            if update_mask is not None and update_mask.any():
                ws = self._encode_and_update_workspace(update_ids, update_mask, workspace=ws)

        # Set workspace for adapter layers
        self._current_workspace = ws

        # Forward through Qwen
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Clear workspace
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
        """
        Generate text with optional mid-stream workspace update.

        Args:
            prompt_text: initial prompt
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            correction_text: correction to inject mid-stream
            correction_after_tokens: inject correction after this many generated tokens

        Returns:
            (full_generated_text, text_at_correction_point_or_None)
        """
        self.eval()
        device = next(self.qwen.parameters()).device

        # Encode prompt and init workspace
        prompt_enc = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        ws = self._encode_and_update_workspace(
            prompt_enc["input_ids"], prompt_enc["attention_mask"]
        )

        # Prepare correction if provided
        correction_enc = None
        if correction_text and correction_after_tokens is not None:
            correction_enc = self.tokenizer(correction_text, return_tensors="pt").to(device)

        self._current_workspace = ws

        # Start generation
        generated_ids = prompt_enc["input_ids"].clone()
        text_at_correction = None

        for step in range(max_new_tokens):
            # Inject correction at the specified step
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
        """Return only the trainable (non-frozen) parameters."""
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
