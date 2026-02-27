"""
CrossAttentionAdapter: zero-initialized cross-attention module inserted into
each Qwen decoder layer. At initialization, output projection is all zeros
so the adapter is a no-op and the model behaves exactly like vanilla Qwen.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionAdapter(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        head_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim

        self.ln = nn.RMSNorm(d_model, eps=1e-6)

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Zero-init output projection so adapter is a no-op at start
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D) from the decoder self-attention output
            workspace: (B, N_slots, D) current workspace state

        Returns:
            (B, T, D) residual to add to hidden_states
        """
        B, T, _ = hidden_states.shape
        N = workspace.size(1)

        h = self.ln(hidden_states)

        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(workspace).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(workspace).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        return self.o_proj(out)
