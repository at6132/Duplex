"""
CrossAttentionAdapter v2: Dual-path cross-attention.

Path 1: workspace slots (32 x D) — high-level context conditioning
Path 2: per-token correction states (T_corr x D) — entity-level detail

KV = concat(workspace_slots, correction_token_states) when correction is
available, otherwise just workspace_slots.

Output projection is zero-initialized so adapter is a no-op at start.
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

        self.ln_q = nn.RMSNorm(d_model, eps=1e-6)
        self.ln_kv = nn.RMSNorm(d_model, eps=1e-6)

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        workspace: torch.Tensor,
        correction_tokens: torch.Tensor | None = None,
        correction_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D) from the decoder layer
            workspace: (B, N_slots, D) workspace state
            correction_tokens: (B, T_corr, D) per-token encoder states (optional)
            correction_mask: (B, T_corr) 1=real 0=pad (optional)

        Returns:
            (B, T, D) residual to add to hidden_states
        """
        B, T, _ = hidden_states.shape

        # Build KV context: workspace + optional correction tokens
        if correction_tokens is not None:
            kv_context = torch.cat([workspace, correction_tokens], dim=1)
            N_ws = workspace.size(1)
            N_corr = correction_tokens.size(1)
            # Mask: workspace slots are always valid, correction may have padding
            ws_mask = torch.ones(B, N_ws, device=hidden_states.device, dtype=hidden_states.dtype)
            if correction_mask is not None:
                kv_mask = torch.cat([ws_mask, correction_mask.to(hidden_states.dtype)], dim=1)
            else:
                corr_mask = torch.ones(B, N_corr, device=hidden_states.device, dtype=hidden_states.dtype)
                kv_mask = torch.cat([ws_mask, corr_mask], dim=1)
        else:
            kv_context = workspace
            kv_mask = None

        N_kv = kv_context.size(1)

        h = self.ln_q(hidden_states)
        kv = self.ln_kv(kv_context)

        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, N_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, N_kv, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if kv_mask is not None:
            mask = kv_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N_kv)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        return self.o_proj(out)
