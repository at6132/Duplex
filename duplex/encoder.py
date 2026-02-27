"""
Update Encoder: 4-layer bidirectional transformer encoder that encodes
prompt/correction text into representations for workspace update.
Uses its own embeddings (Qwen's are frozen and not shared).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, d_ff: int, dropout: float):
        super().__init__()
        self.inner_dim = n_heads * head_dim

        self.ln1 = nn.RMSNorm(d_model, eps=1e-6)
        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        self.ln2 = nn.RMSNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask: (B, T) -> (B, 1, 1, T) for broadcasting
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        x = x + self.o_proj(out)

        h = self.ln2(x)
        x = x + self.ffn(h)

        return x


class UpdateEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        d_model: int = 2048,
        n_heads: int = 16,
        head_dim: int = 128,
        d_ff: int = 6144,
        n_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        # Sinusoidal positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, head_dim, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.RMSNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        # Optional projection to a different output dim (e.g. d_model=1024 -> workspace_dim=2048)
        self.output_proj = None

    def set_output_projection(self, target_dim: int):
        """Add a linear projection from encoder dim to target dim."""
        d_model = self.embed.embedding_dim
        if d_model != target_dim:
            self.output_proj = nn.Linear(d_model, target_dim, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token IDs
            attention_mask: (B, T), 1=real 0=pad

        Returns:
            (B, T, D) encoded representations
        """
        x = self.embed(input_ids)
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=attention_mask)

        x = self.ln_final(x)
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x
