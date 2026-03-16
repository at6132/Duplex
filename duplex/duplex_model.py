"""
Duplex-1.4-1.7B: Frozen Qwen3-1.7B with DEEP prefix conditioning (P-Tuning v2).

Architecture:
    1. Qwen3 decoder layers are frozen, loaded in bfloat16
    2. UpdateEncoder encodes prompt/correction text into per-token states
    3. WorkspaceModule compresses context into N latent slots
    4. DeepPrefixEncoder projects workspace slots into per-layer K/V pairs
    5. K/V pairs are injected at EVERY Qwen layer via past_key_values
    6. Qwen's attention at each layer directly sees the workspace prefix
    7. Revision markers use existing Qwen tokens ([[REVISE:, ]])

Why deep prefix instead of shallow (input-only) prefix:
    Shallow prefix prepends workspace slots only to input embeddings. The signal
    must survive 28 frozen attention layers — by the deeper layers, it's diluted
    to nothing. This is a known failure mode for 1B-3B models (P-Tuning v2 paper).
    Deep prefix injects fresh K/V at every layer, giving direct influence at
    every depth. Uses past_key_values — clean, no monkey-patching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from peft import get_peft_model, LoraConfig, TaskType

from .workspace import WorkspaceModule
from .encoder import UpdateEncoder
from .config import DuplexConfig, REVISION_MARKERS


class DeepPrefixEncoder(nn.Module):
    """Projects workspace slots into per-layer K/V pairs for all Qwen layers.

    Uses independent projections per layer (not a shared MLP), which gives each
    layer direct control over its prefix representation and trains more stably.

    Input:  (B, n_prefix, workspace_dim)  — workspace slots
    Output: DynamicCache with (B, n_kv_heads, n_prefix, head_dim) K/V per layer
    """

    def __init__(
        self,
        n_prefix: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        input_dim: int,
    ):
        super().__init__()
        self.n_prefix = n_prefix
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.kv_dim = n_kv_heads * head_dim

        self.kv_projs = nn.ModuleList([
            nn.Linear(input_dim, 2 * self.kv_dim, bias=False)
            for _ in range(n_layers)
        ])

    def forward(self, prefix_embeds: torch.Tensor) -> DynamicCache:
        B, P, _ = prefix_embeds.shape

        cache = DynamicCache()
        for layer_idx in range(self.n_layers):
            kv = self.kv_projs[layer_idx](prefix_embeds)
            k, v = kv.chunk(2, dim=-1)
            k = k.view(B, P, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
            v = v.view(B, P, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
            cache.update(k, v, layer_idx)

        return cache


class DuplexModel(nn.Module):
    def __init__(self, config: DuplexConfig, local_rank: int = 0):
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.qwen_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.revision_markers = REVISION_MARKERS

        self.qwen = AutoModelForCausalLM.from_pretrained(
            config.qwen_model_path,
            dtype=torch.bfloat16,
            device_map={"": device_str},
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        for param in self.qwen.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        self.qwen = get_peft_model(self.qwen, lora_config)

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

        self.workspace = WorkspaceModule(
            n_slots=config.n_workspace_slots,
            d_model=config.workspace_dim,
            n_heads=config.adapter_n_heads,
            head_dim=config.head_dim,
            dropout=config.adapter_dropout,
        )

        self.deep_prefix = DeepPrefixEncoder(
            n_prefix=config.n_workspace_slots,
            n_layers=config.n_decoder_layers,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            input_dim=config.workspace_dim,
        )

        self.encoder = self.encoder.to(device=device_str, dtype=torch.bfloat16)
        self.workspace = self.workspace.to(device=device_str, dtype=torch.bfloat16)
        self.deep_prefix = self.deep_prefix.to(device=device_str, dtype=torch.bfloat16)

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

    def _build_deep_cache(self, workspace: torch.Tensor) -> DynamicCache:
        """Build per-layer K/V cache from workspace slots."""
        return self.deep_prefix(workspace)

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
            ws, _ = self._encode_and_update_workspace(prompt_ids, prompt_mask)

        if update_ids is not None and update_ids.size(1) > 0:
            if update_mask is not None and update_mask.any():
                corr_encoded = self.encoder(update_ids, attention_mask=update_mask)
                ws = self.workspace(corr_encoded, encoder_mask=update_mask, workspace=ws)

        if ws is not None:
            B = input_ids.size(0)
            n_prefix = ws.size(1)

            prefix_cache = self._build_deep_cache(ws)

            if attention_mask is not None:
                prefix_mask = torch.ones(
                    B, n_prefix, device=input_ids.device, dtype=attention_mask.dtype
                )
                extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                extended_mask = None

            with torch.inference_mode(mode=False):
                outputs = self.qwen(
                    input_ids=input_ids,
                    attention_mask=extended_mask,
                    past_key_values=prefix_cache,
                    labels=labels,
                )
        else:
            with torch.inference_mode(mode=False):
                outputs = self.qwen(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

        result = {"logits": outputs.logits}
        if outputs.loss is not None:
            result["loss"] = outputs.loss
        return result

    GENERIC_INSTRUCTION = "Respond to the following request."

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

        generic_enc = self.tokenizer(
            self.GENERIC_INSTRUCTION, return_tensors="pt"
        ).to(device)
        generated_ids = generic_enc["input_ids"].clone()
        generic_len = generated_ids.size(1)
        n_prefix = ws.size(1)
        text_at_correction = None

        for step in range(max_new_tokens):
            if (correction_enc is not None
                    and correction_after_tokens is not None
                    and step == correction_after_tokens):
                text_at_correction = self._decode_response(generated_ids[0], generic_len)
                corr_encoded = self.encoder(
                    correction_enc["input_ids"],
                    attention_mask=correction_enc["attention_mask"],
                )
                ws = self.workspace(
                    corr_encoded,
                    encoder_mask=correction_enc["attention_mask"],
                    workspace=ws,
                )
                n_prefix = ws.size(1)

            prefix_cache = self._build_deep_cache(ws)
            seq_len = generated_ids.size(1)
            prefix_mask = torch.ones(1, n_prefix, device=device, dtype=torch.long)
            text_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)
            attn_mask = torch.cat([prefix_mask, text_mask], dim=1)

            outputs = self.qwen(
                input_ids=generated_ids,
                attention_mask=attn_mask,
                past_key_values=prefix_cache,
            )

            next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        response = self._decode_response(generated_ids[0], generic_len)
        return response, text_at_correction

    def _decode_response(self, token_ids: torch.Tensor, skip_prefix_len: int) -> str:
        """Decode tokens, stripping the generic instruction prefix."""
        response_ids = token_ids[skip_prefix_len:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=False).strip()

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

        generic_enc = self.tokenizer(
            self.GENERIC_INSTRUCTION, return_tensors="pt"
        ).to(device)
        generated_ids = generic_enc["input_ids"].clone()
        generic_len = generated_ids.size(1)
        n_prefix = ws.size(1)

        for step in range(max_new_tokens):
            if (correction_enc is not None
                    and correction_after_tokens is not None
                    and step == correction_after_tokens):
                text_so_far = self._decode_response(generated_ids[0], generic_len)
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
                n_prefix = ws.size(1)

            prefix_cache = self._build_deep_cache(ws)
            seq_len = generated_ids.size(1)
            prefix_mask = torch.ones(1, n_prefix, device=device, dtype=torch.long)
            text_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)
            attn_mask = torch.cat([prefix_mask, text_mask], dim=1)

            outputs = self.qwen(
                input_ids=generated_ids,
                attention_mask=attn_mask,
                past_key_values=prefix_cache,
            )

            next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            response = self._decode_response(generated_ids[0], generic_len)
            yield response

            if next_token.item() == self.tokenizer.eos_token_id:
                break

    def get_trainable_params(self) -> list[nn.Parameter]:
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.workspace.parameters())
        params.extend(self.deep_prefix.parameters())
        for name, p in self.qwen.named_parameters():
            if p.requires_grad:
                params.append(p)
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
        lora_params = sum(p.numel() for n, p in self.qwen.named_parameters() if p.requires_grad)
        print(f"LoRA params:      {lora_params:>12,}")
        print(f"Architecture:     deep prefix + LoRA (Q/V, r=16)")
