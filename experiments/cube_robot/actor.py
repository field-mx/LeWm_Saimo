from __future__ import annotations

import torch
from torch import nn


class GaussianActor(nn.Module):
    """Goal-conditioned, bounded Gaussian policy for the single-cube task."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        hidden = self.trunk(observation)
        mean = self.mean_head(hidden)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.distribution(observation).mean)

    def sample(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action)
        log_prob -= torch.log(1.0 - action.square() + 1e-6)
        return action, log_prob.sum(dim=-1)

    def log_prob(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        bounded_action = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = torch.atanh(bounded_action)
        distribution = self.distribution(observation)
        log_prob = distribution.log_prob(raw_action)
        log_prob -= torch.log(1.0 - bounded_action.square() + 1e-6)
        return log_prob.sum(dim=-1)


class ValueCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),  
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)
