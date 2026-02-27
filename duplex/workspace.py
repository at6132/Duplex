"""
Workspace module: persistent latent slots with gated cross-attention update.
Scaled up from CONCEPT1 (16x256) to (32x2048) to match Qwen3-1.7B dimensions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkspaceModule(nn.Module):
    def __init__(
        self,
        n_slots: int = 32,
        d_model: int = 2048,
        n_heads: int = 16,
        head_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim

        self.init_workspace = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)

        # Cross-attention: workspace attends to encoder output
        self.ln_w = nn.RMSNorm(d_model, eps=1e-6)
        self.ln_e = nn.RMSNorm(d_model, eps=1e-6)

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.attn_out = nn.Linear(self.inner_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Gated update
        self.delta_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.gate_linear = nn.Linear(d_model * 2, d_model)

    def get_initial_workspace(self, batch_size: int) -> torch.Tensor:
        return self.init_workspace.expand(batch_size, -1, -1).clone()

    def update(
        self,
        workspace: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Gated update: workspace attends to encoder output, selectively modifies slots.

        Args:
            workspace: (B, N_slots, D)
            encoder_output: (B, T_enc, D)
            encoder_mask: (B, T_enc), 1=real 0=pad

        Returns:
            updated workspace: (B, N_slots, D)
        """
        B = workspace.size(0)
        N = self.n_slots
        T_enc = encoder_output.size(1)

        w = self.ln_w(workspace)
        e = self.ln_e(encoder_output)

        q = self.q_proj(w).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(e).view(B, T_enc, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(e).view(B, T_enc, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if encoder_mask is not None:
            mask = encoder_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_enc)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)
        attn = self.dropout(attn)

        attended = (attn @ v).transpose(1, 2).contiguous().view(B, N, self.inner_dim)
        attended = self.attn_out(attended)

        delta = self.delta_mlp(attended)
        gate = torch.sigmoid(self.gate_linear(torch.cat([workspace, attended], dim=-1)))

        return workspace + gate * delta

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = encoder_output.size(0)
        if workspace is None:
            workspace = self.get_initial_workspace(B).to(
                device=encoder_output.device, dtype=encoder_output.dtype
            )
        return self.update(workspace, encoder_output, encoder_mask)
