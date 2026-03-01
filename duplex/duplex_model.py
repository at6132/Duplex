"""
DuplexModel v3: Frozen Qwen3-1.7B with prefix-based conditioning.

Architecture:
    1. Qwen3 decoder layers are frozen, loaded in bfloat16
    2. UpdateEncoder encodes prompt/correction text into per-token states
    3. WorkspaceModule compresses context into N latent slots
    4. Workspace slots are PREPENDED as soft tokens to Qwen's input embeddings
    5. When correction arrives, correction token embeddings are also prepended
    6. Qwen's own self-attention naturally conditions on the prefix — no
       monkey-patching, no hidden-state perturbation, no cascade corruption.
    7. Special tokens (<|REVISE_START|>, <|REVISE_END|>, <|INSERT|>) for
       retroactive revision

Why prefix instead of cross-attention adapters:
    Cross-attention residuals added to every (or every Nth) layer cause
    cascading perturbation during autoregressive generation. Even small
    per-layer noise compounds across 28 layers, producing gibberish.
    Prefix tokens go through the backbone's own self-attention — the model
    already knows how to condition on input tokens, so this is inherently
    stable and leverages pretrained attention patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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

        # Load Qwen (frozen)
        self.qwen = AutoModelForCausalLM.from_pretrained(
            config.qwen_model_path,
            dtype=torch.bfloat16,
            device_map={"": device_str},
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        if n_added > 0:
            self.qwen.resize_token_embeddings(len(self.tokenizer))

        for param in self.qwen.parameters():
            param.requires_grad = False

        # Encoder (processes prompt/correction text → per-token states)
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

        # Workspace (compresses encoder output into N prefix slots)
        self.workspace = WorkspaceModule(
            n_slots=config.n_workspace_slots,
            d_model=config.workspace_dim,
            n_heads=config.adapter_n_heads,
            head_dim=config.head_dim,
            dropout=config.adapter_dropout,
        )

        # Move trainable modules to device
        self.encoder = self.encoder.to(device=device_str, dtype=torch.bfloat16)
        self.workspace = self.workspace.to(device=device_str, dtype=torch.bfloat16)

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

    def _build_prefix(
        self,
        workspace: torch.Tensor | None,
        correction_tokens: torch.Tensor | None = None,
        correction_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Build prefix tensor from workspace slots + optional correction tokens.

        Returns (B, N_prefix, D) or None if no workspace.
        """
        if workspace is None:
            return None

        parts = [workspace]

        if correction_tokens is not None:
            if correction_mask is not None:
                # Zero out padded positions so they don't contribute
                correction_tokens = correction_tokens * correction_mask.unsqueeze(-1).to(correction_tokens.dtype)
            parts.append(correction_tokens)

        return torch.cat(parts, dim=1) if len(parts) > 1 else workspace

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

        prefix = self._build_prefix(ws, corr_tokens, corr_mask)

        # Get input embeddings from Qwen's embedding layer
        input_embeds = self.qwen.model.embed_tokens(input_ids)

        if prefix is not None:
            B = input_ids.size(0)
            n_prefix = prefix.size(1)

            # Prepend prefix to input embeddings
            combined_embeds = torch.cat([prefix, input_embeds], dim=1)

            # Extend attention mask
            if attention_mask is not None:
                prefix_mask = torch.ones(B, n_prefix, device=input_ids.device, dtype=attention_mask.dtype)
                combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                combined_mask = None

            # Extend labels (prefix positions are masked with -100)
            if labels is not None:
                prefix_labels = torch.full(
                    (B, n_prefix), -100, device=labels.device, dtype=labels.dtype
                )
                combined_labels = torch.cat([prefix_labels, labels], dim=1)
            else:
                combined_labels = None
        else:
            combined_embeds = input_embeds
            combined_mask = attention_mask
            combined_labels = labels

        with torch.inference_mode(mode=False):
            outputs = self.qwen(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                labels=combined_labels,
            )

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

        prefix = self._build_prefix(ws)
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
                prefix = self._build_prefix(ws, corr_encoded, correction_enc["attention_mask"])

            input_embeds = self.qwen.model.embed_tokens(generated_ids)
            combined = torch.cat([prefix, input_embeds], dim=1)
            outputs = self.qwen(inputs_embeds=combined)

            # Logits for the last real token (not prefix)
            next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

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

        prefix = self._build_prefix(ws)
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
                prefix = self._build_prefix(ws, corr_encoded, correction_enc["attention_mask"])

            input_embeds = self.qwen.model.embed_tokens(generated_ids)
            combined = torch.cat([prefix, input_embeds], dim=1)
            outputs = self.qwen(inputs_embeds=combined)

            next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            full = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            yield full

            if next_token.item() == self.tokenizer.eos_token_id:
                break

    def get_trainable_params(self) -> list[nn.Parameter]:
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.workspace.parameters())
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
        print(f"Prefix slots:     {self.config.n_workspace_slots}")
        print(f"Architecture:     prefix conditioning (no cross-attention adapters)")
