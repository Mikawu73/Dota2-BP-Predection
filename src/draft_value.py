from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from src.draft_mlp import DraftDataset, DraftVocabulary


class DraftValueDataset(Dataset):
    """Factual Pick decisions with the acting side's eventual result."""

    def __init__(
        self,
        samples_path: Path,
        split: str,
        vocabulary: DraftVocabulary,
        recency_half_life_days: float | None = None,
    ):
        self.samples = DraftDataset(
            samples_path,
            split,
            vocabulary,
            recency_half_life_days=recency_half_life_days,
            include_outcome_label=True,
        )
        self.indices = torch.nonzero(self.samples.action_types == 1, as_tuple=False).flatten()

    def __len__(self) -> int:
        return self.indices.numel()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return self.samples[self.indices[index].item()]


class DraftValueModel(nn.Module):
    def __init__(
        self,
        hero_count: int,
        patch_count: int,
        max_action_order: int,
        team_count: int,
        hidden_sizes: tuple[int, int] = (256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hero_count = hero_count
        self.patch_embedding = nn.Embedding(patch_count + 1, 8, padding_idx=0)
        self.order_embedding = nn.Embedding(max_action_order + 1, 8)
        self.side_embedding = nn.Embedding(2, 4)
        self.team_embedding = nn.Embedding(team_count + 1, 32, padding_idx=0)
        self.lineup_hero_embedding = nn.Embedding(hero_count + 1, 32, padding_idx=hero_count)
        self.lineup_phi = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.lineup_rho = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.candidate_embedding = nn.Embedding(hero_count, 32)

        input_size = 4 * hero_count + 8 + 8 + 4 + 64 + 128 + 32
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], 1),
        )

    def encode_history(self, heroes: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(heroes.shape[0], 4, self.hero_count, device=heroes.device)
        valid = heroes >= 0
        batches = torch.arange(heroes.shape[0], device=heroes.device).unsqueeze(1).expand_as(heroes)
        state[batches[valid], categories[valid].long(), heroes[valid].long()] = 1.0
        return state

    def encode_lineups(self, history: torch.Tensor, acting_sides: torch.Tensor) -> torch.Tensor:
        hero_features = self.lineup_phi(self.lineup_hero_embedding.weight[:self.hero_count])
        picked = history[:, :2]
        lineups = self.lineup_rho(torch.matmul(picked, hero_features) / picked.sum(2, keepdim=True).clamp_min(1))
        batches = torch.arange(history.shape[0], device=history.device)
        return torch.cat([
            lineups[batches, acting_sides], lineups[batches, 1 - acting_sides],
        ], dim=1)

    def forward(
        self,
        history_heroes: torch.Tensor,
        history_categories: torch.Tensor,
        patch_indices: torch.Tensor,
        action_orders: torch.Tensor,
        acting_sides: torch.Tensor,
        team_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        history = self.encode_history(history_heroes, history_categories)
        features = torch.cat([
            history.flatten(1),
            self.patch_embedding(patch_indices),
            self.order_embedding(action_orders),
            self.side_embedding(acting_sides),
            self.team_embedding(team_indices).flatten(1),
            self.encode_lineups(history, acting_sides),
            self.candidate_embedding(candidate_indices),
        ], dim=1)
        return self.network(features).squeeze(1)

    def score_candidates(
        self,
        history_heroes: torch.Tensor,
        history_categories: torch.Tensor,
        patch_indices: torch.Tensor,
        action_orders: torch.Tensor,
        acting_sides: torch.Tensor,
        team_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return win probabilities for candidate heroes for one or more repeated states."""
        return torch.sigmoid(self(
            history_heroes, history_categories, patch_indices, action_orders,
            acting_sides, team_indices, candidate_indices,
        ))
