from __future__ import annotations

import torch
from torch import nn


class RewardNetwork(nn.Module):
    """Predict task reward from predicted and goal latents."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = 2 * latent_dim
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_normalization(
        self, mean: torch.Tensor, scale: torch.Tensor
    ) -> None:
        self.input_mean.copy_(mean)
        self.input_scale.copy_(scale.clamp_min(1e-5))

    def forward(
        self, predicted_latent: torch.Tensor, goal_latent: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat([predicted_latent, goal_latent], dim=-1)
        features = (features - self.input_mean) / self.input_scale
        return self.network(features).squeeze(-1)
