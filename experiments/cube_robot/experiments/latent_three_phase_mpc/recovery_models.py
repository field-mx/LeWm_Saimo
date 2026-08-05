from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def state_one_hot(state: torch.Tensor, count: int) -> torch.Tensor:
    return nn.functional.one_hot(state.long(), num_classes=count).float()


def goal_features(
    previous: torch.Tensor,
    current: torch.Tensor,
    goal: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([previous, current, goal, current - goal], dim=-1)


def control_features(
    previous: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    goal: torch.Tensor,
    state: torch.Tensor,
    state_count: int,
) -> torch.Tensor:
    return torch.cat(
        [
            previous,
            current,
            target,
            goal,
            current - target,
            current - goal,
            state_one_hot(state, state_count),
        ],
        dim=-1,
    )


class VisualSemanticNet(nn.Module):
    """Predict RGB-only manipulation predicates used by the recovery FSM."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        predicate_count: int,
    ) -> None:
        super().__init__()
        self.predicate_count = predicate_count
        self.network = _mlp(
            4 * latent_dim,
            hidden_dim,
            predicate_count,
        )

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(goal_features(previous, current, goal))


class RecoveryTargetNet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        state_count: int,
    ) -> None:
        super().__init__()
        self.state_count = state_count
        self.network = _mlp(
            4 * latent_dim + state_count,
            hidden_dim,
            latent_dim,
        )

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        goal: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                goal_features(previous, current, goal),
                state_one_hot(state, self.state_count),
            ],
            dim=-1,
        )
        return self.network(features)


class RecoveryBlockActor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_block: int,
        hidden_dim: int,
        state_count: int,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_block = action_block
        self.output_dim = action_dim * action_block
        self.state_count = state_count
        self.network = _mlp(
            6 * latent_dim + state_count,
            hidden_dim,
            self.output_dim,
        )

    def mean(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        target: torch.Tensor,
        goal: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        features = control_features(
            previous,
            current,
            target,
            goal,
            state,
            self.state_count,
        )
        return torch.tanh(self.network(features))


class RecoveryValueNet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        state_count: int,
    ) -> None:
        super().__init__()
        self.state_count = state_count
        self.network = _mlp(
            6 * latent_dim + state_count,
            hidden_dim,
            1,
        )

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        target: torch.Tensor,
        goal: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        features = control_features(
            previous,
            current,
            target,
            goal,
            state,
            self.state_count,
        )
        return self.network(features).squeeze(-1)


class RecoveryValueEnsemble(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        state_count: int,
        ensemble_size: int,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            [
                RecoveryValueNet(latent_dim, hidden_dim, state_count)
                for _ in range(ensemble_size)
            ]
        )

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        target: torch.Tensor,
        goal: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                member(previous, current, target, goal, state)
                for member in self.members
            ],
            dim=0,
        )
