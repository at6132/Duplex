import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    TransformerBlock,
    SinusoidalPositionalEncoding,
    make_causal_mask,
)


class BaselineDecoder(nn.Module):
    """
    Standard decoder-only autoregressive transformer.

    Trained on serialized interruption sequences:
        <BOS> prompt <SEP> output_prefix <UPDATE> correction <SEP> continuation <EOS>

    Serves as the control condition for the full-duplex experiments.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, has_cross_attention=False)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

    def forward(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) token IDs
            loss_mask: (B, T) binary mask, 1 where loss should be computed
            targets: (B, T) target token IDs (typically input_ids shifted)

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, T = input_ids.shape

        causal_mask = make_causal_mask(T, input_ids.device)
        pad_mask = (input_ids != self.pad_id).unsqueeze(1).unsqueeze(2).float()
        attn_mask = causal_mask * pad_mask

        x = self.token_emb(input_ids)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, self_attn_mask=attn_mask)

        x = self.ln_final(x)
        logits = self.head(x)

        result = {"logits": logits}

        if targets is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_targets = targets[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=self.pad_id,
                reduction="none",
            )
            loss = loss.view(B, -1)

            if loss_mask is not None:
                shift_mask = loss_mask[:, 1:].contiguous()
                loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
            else:
                loss = loss.mean()

            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressive generation from a prompt."""
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            if generated.size(1) >= 512:
                break

            out = self.forward(generated)
            next_logits = out["logits"][:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return generated
