import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    TransformerBlock,
    SinusoidalPositionalEncoding,
    make_causal_mask,
)
from .update_encoder import UpdateEncoder
from .workspace import WorkspaceModule


class WorkspaceGenerator(nn.Module):
    """
    Full experimental model: workspace-conditioned generator.

    Architecture:
        1. UpdateEncoder encodes prompt/correction text
        2. WorkspaceModule maintains latent reasoning state
        3. Decoder generates tokens conditioned on workspace via cross-attention

    Training flow:
        - Encode prompt -> init workspace -> teacher-force prefix
        - Encode correction -> update workspace -> teacher-force continuation
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_decoder_layers: int = 4,
        n_encoder_layers: int = 2,
        n_workspace_slots: int = 16,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        self.encoder = UpdateEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_encoder_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            token_emb=self.token_emb,
        )

        self.workspace = WorkspaceModule(
            n_slots=n_workspace_slots,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.decoder_layers = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout,
                has_cross_attention=True,
            )
            for _ in range(n_decoder_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def _decode(
        self,
        token_ids: torch.Tensor,
        workspace_state: torch.Tensor,
    ) -> torch.Tensor:
        """Run decoder over token_ids, cross-attending to workspace_state."""
        B, T = token_ids.shape
        causal_mask = make_causal_mask(T, token_ids.device)
        pad_mask = (token_ids != self.pad_id).unsqueeze(1).unsqueeze(2).float()
        attn_mask = causal_mask * pad_mask

        x = self.token_emb(token_ids)
        x = self.pos_enc(x)

        for layer in self.decoder_layers:
            x = layer(x, self_attn_mask=attn_mask, cross_kv=workspace_state)

        x = self.ln_final(x)
        return self.head(x)

    def _make_pad_mask(self, ids: torch.Tensor) -> torch.Tensor:
        return (ids != self.pad_id).unsqueeze(1).unsqueeze(2).float()

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prefix_ids: torch.Tensor,
        update_ids: torch.Tensor,
        continuation_ids: torch.Tensor,
        prefix_loss_mask: torch.Tensor | None = None,
        continuation_loss_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full training forward pass with two-stage workspace update.

        Args:
            prompt_ids: (B, T_p) initial prompt
            prefix_ids: (B, T_pre) output prefix tokens (teacher-forced)
            update_ids: (B, T_u) correction/update text
            continuation_ids: (B, T_cont) revised continuation tokens (teacher-forced)
            prefix_loss_mask: (B, T_pre) optional mask for prefix loss
            continuation_loss_mask: (B, T_cont) optional mask for continuation loss

        Returns:
            dict with 'loss', 'prefix_logits', 'continuation_logits', 'workspace_delta'
        """
        # Stage 1: encode prompt and init workspace
        prompt_pad_mask = self._make_pad_mask(prompt_ids)
        prompt_encoded = self.encoder(prompt_ids, pad_mask=prompt_pad_mask)
        ws = self.workspace(prompt_encoded, encoder_pad_mask=prompt_pad_mask)

        # Stage 2: decode prefix conditioned on initial workspace
        prefix_logits = self._decode(prefix_ids, ws)

        # Stage 3: encode update and update workspace
        ws_before_update = ws.detach().clone()
        if update_ids.size(1) > 0 and (update_ids != self.pad_id).any():
            update_pad_mask = self._make_pad_mask(update_ids)
            update_encoded = self.encoder(update_ids, pad_mask=update_pad_mask)
            ws = self.workspace(update_encoded, encoder_pad_mask=update_pad_mask, workspace=ws)

        workspace_delta = (ws - ws_before_update).norm(dim=-1).mean()

        # Stage 4: decode continuation conditioned on updated workspace
        continuation_logits = self._decode(continuation_ids, ws)

        # Compute loss on both segments
        total_loss = torch.tensor(0.0, device=prompt_ids.device)
        n_segments = 0

        if prefix_ids.size(1) > 1:
            prefix_loss = self._segment_loss(
                prefix_logits, prefix_ids, prefix_loss_mask
            )
            total_loss = total_loss + prefix_loss
            n_segments += 1

        if continuation_ids.size(1) > 1:
            cont_loss = self._segment_loss(
                continuation_logits, continuation_ids, continuation_loss_mask
            )
            total_loss = total_loss + cont_loss
            n_segments += 1

        if n_segments > 0:
            total_loss = total_loss / n_segments

        return {
            "loss": total_loss,
            "prefix_logits": prefix_logits,
            "continuation_logits": continuation_logits,
            "workspace_delta": workspace_delta,
        }

    def _segment_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Cross-entropy loss on a token segment with optional masking."""
        B = logits.size(0)
        shift_logits = logits[:, :-1].contiguous()
        shift_targets = targets[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
            ignore_index=self.pad_id,
            reduction="none",
        ).view(B, -1)

        if loss_mask is not None:
            shift_mask = loss_mask[:, 1:].contiguous()
            return (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
        return loss.mean()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        eos_id: int = 2,
        update_ids: torch.Tensor | None = None,
        update_after_step: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Autoregressive generation with optional mid-stream workspace update.

        Args:
            prompt_ids: (B, T) initial prompt tokens
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            eos_id: end of sequence token
            update_ids: (B, T_u) correction tokens, injected at update_after_step
            update_after_step: generation step at which to inject the update

        Returns:
            (generated_ids, workspace_snapshots)
        """
        self.eval()

        prompt_pad_mask = self._make_pad_mask(prompt_ids)
        prompt_encoded = self.encoder(prompt_ids, pad_mask=prompt_pad_mask)
        ws = self.workspace(prompt_encoded, encoder_pad_mask=prompt_pad_mask)

        workspace_snapshots = [ws.clone()]

        B = prompt_ids.size(0)
        bos = torch.full((B, 1), 1, dtype=torch.long, device=prompt_ids.device)
        generated = bos

        for step in range(max_new_tokens):
            if update_ids is not None and update_after_step is not None and step == update_after_step:
                update_pad_mask = self._make_pad_mask(update_ids)
                update_encoded = self.encoder(update_ids, pad_mask=update_pad_mask)
                ws = self.workspace(update_encoded, encoder_pad_mask=update_pad_mask, workspace=ws)
                workspace_snapshots.append(ws.clone())

            logits = self._decode(generated, ws)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return generated, workspace_snapshots
