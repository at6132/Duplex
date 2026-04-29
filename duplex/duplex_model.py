"""
Duplex-1.5-4B: Gemma 3 4B-IT with deep prefix conditioning + LoRA.

Architecture:
    1. Gemma 3 4B-IT loaded in 4-bit (BitsAndBytes) — fits in 12GB VRAM
    2. LoRA (r=16) on Q/V projections — teaches Gemma to attend to prefix K/V
    3. UpdateEncoder uses Gemma's pretrained embeddings (frozen) — already knows English
    4. WorkspaceModule compresses encoder output into 32 latent slots
    5. DeepPrefixEncoder projects workspace into per-layer K/V (injected via past_key_values)
    6. Gradient checkpointing enabled for VRAM efficiency

Key improvements over v1.4 (Qwen 1.7B):
    - Instruction-tuned backbone (Gemma already knows prompt-following)
    - 4B params (2.3x larger, more capacity for prefix conditioning)
    - Pretrained embeddings in encoder (no more random init)
    - Runs locally on RTX 5070 12GB — no cloud costs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

from .workspace import WorkspaceModule
from .encoder import UpdateEncoder
from .config import DuplexConfig, REVISION_MARKERS



class DuplexModel(nn.Module):
    def __init__(self, config: DuplexConfig, local_rank: int = 0):
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.revision_markers = REVISION_MARKERS

        # Load backbone in 4-bit for VRAM efficiency
        load_kwargs = {"trust_remote_code": True, "device_map": {"": device_str}}
        if config.quantize_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config
        else:
            load_kwargs["dtype"] = torch.bfloat16

        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.model_path, **load_kwargs
        )
        self.backbone.gradient_checkpointing_enable()

        for param in self.backbone.parameters():
            param.requires_grad = False

        # LoRA on Q/V — teaches backbone to attend to deep prefix K/V
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        # Encoder uses backbone's pretrained embeddings (frozen)
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
        # Find embed_tokens through the model hierarchy (handles PEFT wrapper + Gemma3 multimodal)
        embed_layer = None
        for name, module in self.backbone.named_modules():
            if name.endswith('embed_tokens') and isinstance(module, torch.nn.Embedding):
                embed_layer = module
                break
        if embed_layer is None:
            raise RuntimeError("Could not find embed_tokens in backbone")
        self.encoder.use_external_embeddings(embed_layer)
        if config.encoder_dim != config.workspace_dim:
            self.encoder.set_output_projection(config.workspace_dim)

        self.workspace = WorkspaceModule(
            n_slots=config.n_workspace_slots,
            d_model=config.workspace_dim,
            n_heads=config.adapter_n_heads,
            head_dim=min(config.head_dim, config.workspace_dim // config.adapter_n_heads),
            dropout=config.adapter_dropout,
        )

        # Project workspace (2560d) to backbone embed dim for input-level prefix.
        # With LoRA teaching the backbone to attend to prefix + instruction-tuned
        # Gemma already knowing prompt-following, input-level prefix is sufficient.
        backbone_embed_dim = embed_layer.embedding_dim
        self.deep_prefix_to_embed = nn.Linear(config.workspace_dim, backbone_embed_dim, bias=False)

        self.encoder = self.encoder.to(device=device_str, dtype=torch.bfloat16)
        self.workspace = self.workspace.to(device=device_str, dtype=torch.bfloat16)
        self.deep_prefix_to_embed = self.deep_prefix_to_embed.to(device=device_str, dtype=torch.bfloat16)

    def _encode_and_update_workspace(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.encoder(text_ids, attention_mask=text_mask)
        ws = self.workspace(encoder_output, encoder_mask=text_mask, workspace=workspace)
        return ws, encoder_output

    def _build_prefix_embeds(self, workspace: torch.Tensor) -> torch.Tensor:
        """Project workspace slots to backbone embedding space."""
        return self.deep_prefix_to_embed(workspace)

    def _get_embed_layer(self):
        """Get the backbone's token embedding layer through any wrappers."""
        for name, module in self.backbone.named_modules():
            if name.endswith('embed_tokens') and isinstance(module, torch.nn.Embedding):
                return module
        raise RuntimeError("Could not find embed_tokens")

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

            # Project workspace to backbone's embedding dim and prepend as soft tokens
            prefix = self.deep_prefix_to_embed(ws)
            embed_layer = self._get_embed_layer()
            with torch.no_grad():
                input_embeds = embed_layer(input_ids)
            input_embeds = input_embeds.to(prefix.dtype)
            combined_embeds = torch.cat([prefix, input_embeds], dim=1)

            if attention_mask is not None:
                prefix_mask = torch.ones(B, n_prefix, device=input_ids.device, dtype=attention_mask.dtype)
                combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                combined_mask = None

            if labels is not None:
                prefix_labels = torch.full((B, n_prefix), -100, device=labels.device, dtype=labels.dtype)
                combined_labels = torch.cat([prefix_labels, labels], dim=1)
            else:
                combined_labels = None

            outputs = self.backbone(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                labels=combined_labels,
            )
        else:
            outputs = self.backbone(
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
        device = next(self.backbone.parameters()).device

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
        text_at_correction = None

        embed_layer = self._get_embed_layer()

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

            prefix = self._build_prefix_embeds(ws)
            input_embeds = embed_layer(generated_ids).to(prefix.dtype)
            combined = torch.cat([prefix, input_embeds], dim=1)
            outputs = self.backbone(inputs_embeds=combined)

            next_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        response = self._decode_response(generated_ids[0], generic_len)
        return response, text_at_correction

    def _decode_response(self, token_ids: torch.Tensor, skip_prefix_len: int) -> str:
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
        self.eval()
        device = next(self.backbone.parameters()).device

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

        embed_layer = self._get_embed_layer()

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

            prefix = self._build_prefix_embeds(ws)
            input_embeds = embed_layer(generated_ids).to(prefix.dtype)
            combined = torch.cat([prefix, input_embeds], dim=1)
            outputs = self.backbone(inputs_embeds=combined)

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
        for p in self.encoder.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.workspace.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.deep_prefix_to_embed.parameters():
            if p.requires_grad:
                params.append(p)
        for name, p in self.backbone.named_parameters():
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
        lora_params = sum(p.numel() for n, p in self.backbone.named_parameters() if p.requires_grad)
        print(f"Total params:     {total:>12,}")
        print(f"Trainable params: {trainable:>12,}")
        print(f"Frozen params:    {frozen:>12,}")
        print(f"Trainable %:      {100 * trainable / total:>11.1f}%")
        print(f"LoRA params:      {lora_params:>12,}")
        print(f"Prefix slots:     {self.config.n_workspace_slots}")
        print(f"Architecture:     Duplex-1.5 | deep prefix + LoRA | Gemma 3 4B-IT (4-bit)")
