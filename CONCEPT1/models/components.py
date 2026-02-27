import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        T_k = key.size(1)

        Q = self.w_q(query).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)  # all-masked rows (padding) produce NaN; zero them out
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional cross-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        has_cross_attention: bool = False,
    ):
        super().__init__()
        self.has_cross_attention = has_cross_attention

        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        if has_cross_attention:
            self.ln_cross = nn.LayerNorm(d_model)
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_kv: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.self_attn(h, h, h, mask=self_attn_mask)

        if self.has_cross_attention and cross_kv is not None:
            h = self.ln_cross(x)
            x = x + self.cross_attn(h, cross_kv, cross_kv, mask=cross_attn_mask)

        h = self.ln2(x)
        x = x + self.ff(h)
        return x


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Returns a (1, 1, T, T) causal mask where 1 = attend, 0 = mask."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def make_pad_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Returns a (B, 1, 1, T) mask where 1 = real token, 0 = padding."""
    return (ids != pad_id).unsqueeze(1).unsqueeze(2).float()
