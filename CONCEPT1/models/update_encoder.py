import torch
import torch.nn as nn

from .components import TransformerBlock, SinusoidalPositionalEncoding


class UpdateEncoder(nn.Module):
    """
    Lightweight transformer encoder for prompt/correction text.
    Uses bidirectional self-attention (no causal mask).
    Shares token embeddings with the generator.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        token_emb: nn.Embedding | None = None,
    ):
        super().__init__()
        self.token_emb = token_emb  # shared with generator; set externally if None
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, has_cross_attention=False)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token IDs for prompt or correction text
            pad_mask: (B, 1, 1, T) with 1=real, 0=pad

        Returns:
            (B, T, D) encoded representations
        """
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)

        # Bidirectional: pad_mask only (no causal), shape (B, 1, T, T)
        if pad_mask is not None:
            pm = pad_mask.squeeze(2)  # (B, 1, T)
            attn_mask = pm.unsqueeze(-1) * pm.unsqueeze(-2)  # (B, 1, T, T)
        else:
            attn_mask = None

        for layer in self.layers:
            x = layer(x, self_attn_mask=attn_mask)

        return self.ln_final(x)
