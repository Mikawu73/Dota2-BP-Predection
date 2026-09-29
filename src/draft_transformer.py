from __future__ import annotations

import torch
from torch import nn


class DraftTransformer(nn.Module):
    def __init__(
        self,
        hero_count: int,
        patch_count: int,
        max_action_order: int,
        team_count: int = 0,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.hero_count = hero_count
        self.max_action_order = max_action_order
        self.hero_embedding = nn.Embedding(hero_count + 1, d_model, padding_idx=hero_count)
        self.category_embedding = nn.Embedding(5, d_model, padding_idx=4)
        self.position_embedding = nn.Embedding(max_action_order + 1, d_model)
        self.patch_embedding = nn.Embedding(patch_count + 1, d_model, padding_idx=0)
        self.order_embedding = nn.Embedding(max_action_order + 1, d_model)
        self.side_embedding = nn.Embedding(2, d_model)
        self.action_embedding = nn.Embedding(2, d_model)
        self.team_embedding = nn.Embedding(team_count + 1, 32, padding_idx=0) if team_count else None
        self.team_projection = nn.Linear(64, d_model) if team_count else None
        self.context_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.context_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, hero_count)

    def forward(
        self,
        history_heroes: torch.Tensor,
        history_categories: torch.Tensor,
        patch_indices: torch.Tensor,
        action_orders: torch.Tensor,
        acting_sides: torch.Tensor,
        action_types: torch.Tensor,
        team_indices: torch.Tensor | None = None,
        roster_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del roster_indices
        batch_size, history_length = history_heroes.shape
        if history_length > self.max_action_order:
            raise ValueError("History is longer than max_action_order")

        padding = history_heroes < 0
        safe_heroes = history_heroes.long().masked_fill(padding, self.hero_count)
        safe_categories = history_categories.long().masked_fill(padding, 4)
        positions = torch.arange(history_length, device=history_heroes.device)
        history_tokens = (
            self.hero_embedding(safe_heroes)
            + self.category_embedding(safe_categories)
            + self.position_embedding(positions).unsqueeze(0)
        )

        context = (
            self.context_token.expand(batch_size, -1, -1).squeeze(1)
            + self.patch_embedding(patch_indices)
            + self.order_embedding(action_orders)
            + self.side_embedding(acting_sides)
            + self.action_embedding(action_types)
        )
        if self.team_embedding is not None:
            if team_indices is None:
                raise ValueError("team_indices are required for this model")
            context = context + self.team_projection(self.team_embedding(team_indices).flatten(1))

        tokens = torch.cat([context.unsqueeze(1), history_tokens], dim=1)
        context_padding = torch.zeros((batch_size, 1), dtype=torch.bool, device=padding.device)
        encoded = self.encoder(tokens, src_key_padding_mask=torch.cat([context_padding, padding], dim=1))
        logits = self.output_head(self.output_norm(encoded[:, 0]))

        used_heroes = torch.zeros(batch_size, self.hero_count, dtype=torch.bool, device=history_heroes.device)
        if history_length:
            batch_indices = torch.arange(batch_size, device=history_heroes.device).unsqueeze(1).expand_as(safe_heroes)
            used_heroes[batch_indices[~padding], safe_heroes[~padding]] = True
        return logits.masked_fill(used_heroes, -1e9), used_heroes
