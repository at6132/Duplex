import torch
import torch.nn as nn

from .components import MultiHeadAttention


class WorkspaceModule(nn.Module):
    """
    Persistent latent workspace with gated update mechanism.

    The workspace is a set of N learnable slots of dimension D that represent
    the model's current task/reasoning state. Updates from encoded text
    selectively modify slots through a gated cross-attention mechanism.
    """

    def __init__(
        self,
        n_slots: int = 16,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model

        self.init_workspace = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln_workspace = nn.LayerNorm(d_model)
        self.ln_encoder = nn.LayerNorm(d_model)

        self.delta_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.gate_linear = nn.Linear(d_model * 2, d_model)

    def get_initial_workspace(self, batch_size: int) -> torch.Tensor:
        return self.init_workspace.expand(batch_size, -1, -1)

    def update(
        self,
        workspace: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Gated update: workspace attends to encoder output, then selectively
        modifies its slots.

        Args:
            workspace: (B, N_slots, D)
            encoder_output: (B, T_enc, D)
            encoder_pad_mask: (B, 1, 1, T_enc), 1=real 0=pad

        Returns:
            updated workspace: (B, N_slots, D)
        """
        w_normed = self.ln_workspace(workspace)
        e_normed = self.ln_encoder(encoder_output)

        cross_attn_mask = None
        if encoder_pad_mask is not None:
            # Expand to (B, 1, N_slots, T_enc) for cross-attention
            cross_attn_mask = encoder_pad_mask.expand(-1, -1, self.n_slots, -1)

        attended = self.cross_attn(w_normed, e_normed, e_normed, mask=cross_attn_mask)

        delta = self.delta_mlp(attended)
        gate = torch.sigmoid(self.gate_linear(torch.cat([workspace, attended], dim=-1)))

        return workspace + gate * delta

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_pad_mask: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Initialize or update workspace from encoder output.

        If workspace is None, starts from learnable init and applies first update.
        If workspace is provided, applies incremental update.
        """
        B = encoder_output.size(0)
        if workspace is None:
            workspace = self.get_initial_workspace(B)
        return self.update(workspace, encoder_output, encoder_pad_mask)
