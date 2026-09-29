from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import Dataset


DATA_COLUMNS = [
    "match_id", "patch_id", "radiant_team_id", "dire_team_id",
    "action_order", "acting_side", "is_pick",
    "previous_hero_ids", "previous_is_pick", "previous_team_sides", "target_hero_id",
]


@dataclass(frozen=True)
class DraftVocabulary:
    hero_ids: tuple[int, ...]
    patch_ids: tuple[int, ...]
    team_ids: tuple[int, ...] = ()
    player_ids: tuple[int, ...] = ()

    @property
    def hero_to_index(self) -> dict[int, int]:
        return {hero_id: index for index, hero_id in enumerate(self.hero_ids)}

    @property
    def patch_to_index(self) -> dict[int, int]:
        return {patch_id: index + 1 for index, patch_id in enumerate(self.patch_ids)}

    @property
    def team_to_index(self) -> dict[int, int]:
        return {team_id: index + 1 for index, team_id in enumerate(self.team_ids)}

    @property
    def player_to_index(self) -> dict[int, int]:
        return {player_id: index + 1 for index, player_id in enumerate(self.player_ids)}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_vocabulary(
    samples_path: Path,
    rosters_path: Path | None = None,
    include_teams: bool = False,
    include_players: bool = False,
) -> DraftVocabulary:
    table = pq.read_table(
        samples_path,
        columns=["patch_id", "target_hero_id", "radiant_team_id", "dire_team_id"],
        filters=[("split", "=", "train")],
    )
    team_ids: tuple[int, ...] = ()
    if include_teams:
        team_ids = tuple(sorted(set(table["radiant_team_id"].to_pylist()) | set(table["dire_team_id"].to_pylist())))
    player_ids: tuple[int, ...] = ()
    if include_players:
        if rosters_path is None:
            raise ValueError("rosters_path is required when include_players=True")
        rosters = pq.read_table(
            rosters_path,
            columns=["radiant_player_ids", "dire_player_ids"],
            filters=[("split", "=", "train")],
        )
        players = {
            player_id
            for column in ("radiant_player_ids", "dire_player_ids")
            for roster in rosters[column].to_pylist()
            for player_id in roster
        }
        player_ids = tuple(sorted(players))
    return DraftVocabulary(
        hero_ids=tuple(sorted(set(table["target_hero_id"].to_pylist()))),
        patch_ids=tuple(sorted(set(table["patch_id"].to_pylist()))),
        team_ids=team_ids,
        player_ids=player_ids,
    )


def history_category(is_pick: bool, side: str) -> int:
    side_offset = 0 if side == "radiant" else 1
    return side_offset if is_pick else 2 + side_offset


class DraftDataset(Dataset):
    def __init__(
        self,
        samples_path: Path,
        split: str,
        vocabulary: DraftVocabulary,
        rosters_path: Path | None = None,
        recency_half_life_days: float | None = None,
        include_outcome_label: bool = False,
        include_role_label: bool = False,
    ):
        if recency_half_life_days is not None and recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")
        columns = DATA_COLUMNS + (["start_time"] if recency_half_life_days is not None else [])
        if include_outcome_label:
            columns.append("acting_side_win")
        if include_role_label:
            columns.append("target_position")
        table = pq.read_table(samples_path, columns=columns, filters=[("split", "=", split)])
        self.split = split
        self.include_outcome_label = include_outcome_label
        self.include_role_label = include_role_label
        self.vocabulary = vocabulary
        hero_to_index = vocabulary.hero_to_index
        patch_to_index = vocabulary.patch_to_index
        team_to_index = vocabulary.team_to_index
        player_to_index = vocabulary.player_to_index
        row_count = table.num_rows

        history_lengths = [len(value) for value in table["previous_hero_ids"].to_pylist()]
        max_history = max(history_lengths, default=0)
        self.history_heroes = torch.full((row_count, max_history), -1, dtype=torch.int16)
        self.history_categories = torch.full((row_count, max_history), -1, dtype=torch.int8)

        raw_heroes = table["previous_hero_ids"]
        raw_picks = table["previous_is_pick"]
        raw_sides = table["previous_team_sides"]
        for row_index in range(row_count):
            heroes = raw_heroes[row_index].as_py()
            picks = raw_picks[row_index].as_py()
            sides = raw_sides[row_index].as_py()
            if not (len(heroes) == len(picks) == len(sides)):
                raise ValueError(f"Inconsistent history lengths at {split} row {row_index}")
            encoded_heroes = [hero_to_index[hero_id] for hero_id in heroes]
            encoded_categories = [history_category(is_pick, side) for is_pick, side in zip(picks, sides)]
            length = len(encoded_heroes)
            if length:
                self.history_heroes[row_index, :length] = torch.tensor(encoded_heroes, dtype=torch.int16)
                self.history_categories[row_index, :length] = torch.tensor(encoded_categories, dtype=torch.int8)

        targets = [hero_to_index.get(hero_id, -1) for hero_id in table["target_hero_id"].to_pylist()]
        if any(target < 0 for target in targets):
            raise ValueError(f"{split} contains target heroes absent from the training vocabulary")
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.outcome_targets = torch.tensor(
            table["acting_side_win"].to_pylist() if include_outcome_label else [False] * row_count,
            dtype=torch.float32,
        )
        self.role_targets = torch.tensor(
            [position - 1 if position is not None else -1 for position in table["target_position"].to_pylist()]
            if include_role_label else [-1] * row_count,
            dtype=torch.long,
        )
        raw_patch_ids = table["patch_id"].to_pylist()
        self.raw_patch_ids = torch.tensor(raw_patch_ids, dtype=torch.long)
        self.patch_indices = torch.tensor(
            [patch_to_index.get(patch_id, 0) for patch_id in raw_patch_ids], dtype=torch.long
        )
        self.action_orders = torch.tensor(table["action_order"].to_pylist(), dtype=torch.long)
        self.acting_sides = torch.tensor(
            [0 if side == "radiant" else 1 for side in table["acting_side"].to_pylist()], dtype=torch.long
        )
        self.action_types = torch.tensor(
            [1 if is_pick else 0 for is_pick in table["is_pick"].to_pylist()], dtype=torch.long
        )
        self.sample_weights = torch.ones(row_count, dtype=torch.float32)
        if split == "train" and recency_half_life_days is not None:
            start_times = table["start_time"].to_pylist()
            reference_time = max(start_times)
            seconds_per_half_life = recency_half_life_days * 86_400
            self.sample_weights = torch.tensor([
                math.exp(-math.log(2) * (reference_time - start_time) / seconds_per_half_life)
                for start_time in start_times
            ], dtype=torch.float32)
        radiant_team_ids = table["radiant_team_id"].to_pylist()
        dire_team_ids = table["dire_team_id"].to_pylist()
        sides = table["acting_side"].to_pylist()
        self.team_indices = torch.tensor([
            [
                team_to_index.get(radiant if side == "radiant" else dire, 0),
                team_to_index.get(dire if side == "radiant" else radiant, 0),
            ]
            for radiant, dire, side in zip(radiant_team_ids, dire_team_ids, sides)
        ], dtype=torch.long)

        self.roster_indices = torch.zeros((row_count, 2, 5), dtype=torch.long)
        if vocabulary.player_ids:
            if rosters_path is None:
                raise ValueError("rosters_path is required for a player vocabulary")
            roster_table = pq.read_table(
                rosters_path,
                columns=["match_id", "radiant_player_ids", "dire_player_ids"],
                filters=[("split", "=", split)],
            )
            rosters = {
                match_id: (radiant, dire)
                for match_id, radiant, dire in zip(
                    roster_table["match_id"].to_pylist(),
                    roster_table["radiant_player_ids"].to_pylist(),
                    roster_table["dire_player_ids"].to_pylist(),
                )
            }
            encoded_rosters = []
            for match_id, side in zip(table["match_id"].to_pylist(), sides):
                radiant, dire = rosters[match_id]
                acting, opponent = (radiant, dire) if side == "radiant" else (dire, radiant)
                encoded_rosters.append([
                    [player_to_index.get(player_id, 0) for player_id in acting],
                    [player_to_index.get(player_id, 0) for player_id in opponent],
                ])
            self.roster_indices = torch.tensor(encoded_rosters, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        items = (
            self.history_heroes[index], self.history_categories[index], self.patch_indices[index],
            self.action_orders[index], self.acting_sides[index], self.action_types[index],
            self.team_indices[index], self.roster_indices[index], self.targets[index],
            self.sample_weights[index],
        )
        if self.include_outcome_label:
            items += (self.outcome_targets[index],)
        if self.include_role_label:
            items += (self.role_targets[index],)
        return items


class DraftContextMLP(nn.Module):
    def __init__(
        self,
        hero_count: int,
        patch_count: int,
        max_action_order: int,
        team_count: int = 0,
        player_count: int = 0,
        hidden_sizes: tuple[int, int] = (256, 128),
        dropout: float = 0.2,
        split_action_heads: bool = False,
        lineup_encoder: bool = False,
        lineup_attention: bool = False,
        role_coverage: bool = False,
        candidate_scorer: bool = False,
        sequence_encoder: bool = False,
        outcome_auxiliary: bool = False,
        role_auxiliary: bool = False,
    ):
        super().__init__()
        if lineup_encoder and lineup_attention:
            raise ValueError("DeepSets and attention lineup encoders are mutually exclusive")
        if split_action_heads and candidate_scorer:
            raise ValueError("Split action heads and candidate scorer are mutually exclusive")
        self.hero_count = hero_count
        self.split_action_heads = split_action_heads
        self.lineup_encoder = lineup_encoder
        self.lineup_attention = lineup_attention
        self.role_coverage = role_coverage
        self.candidate_scorer = candidate_scorer
        self.sequence_encoder = sequence_encoder
        self.outcome_auxiliary = outcome_auxiliary
        self.role_auxiliary = role_auxiliary
        self.register_buffer(
            "hero_role_probabilities", torch.full((hero_count, 5), 0.2), persistent=role_coverage,
        )
        # Index 0 represents a patch absent from the training vocabulary.
        self.patch_embedding = nn.Embedding(patch_count + 1, 8, padding_idx=0)
        self.order_embedding = nn.Embedding(max_action_order + 1, 8)
        self.side_embedding = nn.Embedding(2, 4)
        self.action_embedding = nn.Embedding(2, 4)
        self.team_embedding = nn.Embedding(team_count + 1, 32, padding_idx=0) if team_count else None
        self.player_embedding = nn.Embedding(player_count + 1, 32, padding_idx=0) if player_count else None
        if lineup_encoder or lineup_attention:
            self.lineup_hero_embedding = nn.Embedding(hero_count + 1, 32, padding_idx=hero_count)
        else:
            self.lineup_hero_embedding = None
        if lineup_encoder:
            self.lineup_phi = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
            self.lineup_rho = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        else:
            self.lineup_phi = None
            self.lineup_rho = None
        if lineup_attention:
            self.lineup_side_embedding = nn.Embedding(2, 32)
            self.lineup_query_tokens = nn.Parameter(torch.empty(1, 2, 32))
            nn.init.normal_(self.lineup_query_tokens, std=0.02)
            attention_layer = nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=128, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=False,
            )
            self.lineup_attention_encoder = nn.TransformerEncoder(
                attention_layer, num_layers=1, enable_nested_tensor=False,
            )
            self.lineup_attention_projection = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        else:
            self.lineup_side_embedding = None
            self.lineup_query_tokens = None
            self.lineup_attention_encoder = None
            self.lineup_attention_projection = None
        if sequence_encoder:
            self.sequence_hero_embedding = nn.Embedding(hero_count + 1, 32, padding_idx=hero_count)
            self.sequence_category_embedding = nn.Embedding(4, 32)
            self.sequence_position_embedding = nn.Embedding(max_action_order + 2, 32)
            self.sequence_cls_token = nn.Parameter(torch.empty(1, 1, 32))
            nn.init.normal_(self.sequence_cls_token, std=0.02)
            sequence_layer = nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=128, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=False,
            )
            self.sequence_transformer = nn.TransformerEncoder(
                sequence_layer, num_layers=1, enable_nested_tensor=False,
            )
        else:
            self.sequence_hero_embedding = None
            self.sequence_category_embedding = None
            self.sequence_position_embedding = None
            self.sequence_cls_token = None
            self.sequence_transformer = None
        input_size = 4 * hero_count + 8 + 8 + 4 + 4
        if self.team_embedding is not None:
            input_size += 64
        if self.player_embedding is not None:
            input_size += 64
        if self.lineup_encoder or self.lineup_attention:
            input_size += 128
        if self.role_coverage:
            input_size += 10
        if self.sequence_encoder:
            input_size += 32
        hidden_layers: list[nn.Module] = [
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if candidate_scorer:
            self.network = nn.Sequential(*hidden_layers)
            self.candidate_context_projection = nn.Linear(hidden_sizes[1], 64)
            self.candidate_hero_embedding = nn.Embedding(hero_count, 64)
            self.candidate_bias = nn.Parameter(torch.zeros(hero_count))
            self.pick_head = None
            self.ban_head = None
        elif split_action_heads:
            self.network = nn.Sequential(*hidden_layers)
            self.pick_head = nn.Linear(hidden_sizes[1], hero_count)
            self.ban_head = nn.Linear(hidden_sizes[1], hero_count)
            self.hero_head = None
        elif outcome_auxiliary or role_auxiliary:
            self.network = nn.Sequential(*hidden_layers)
            self.hero_head = nn.Linear(hidden_sizes[1], hero_count)
            self.pick_head = None
            self.ban_head = None
            self.candidate_context_projection = None
            self.candidate_hero_embedding = None
            self.candidate_bias = None
        else:
            # Preserve the original module layout and checkpoint keys.
            self.network = nn.Sequential(*hidden_layers, nn.Linear(hidden_sizes[1], hero_count))
            self.pick_head = None
            self.ban_head = None
            self.candidate_context_projection = None
            self.candidate_hero_embedding = None
            self.candidate_bias = None
            self.hero_head = None
        self.outcome_head = nn.Linear(hidden_sizes[1], 1) if outcome_auxiliary else None
        self.role_head = nn.Linear(hidden_sizes[1], 5) if role_auxiliary else None

    def encode_history(self, heroes: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        batch_size = heroes.shape[0]
        state = torch.zeros(batch_size, 4, self.hero_count, device=heroes.device)
        valid = heroes >= 0
        batch_indices = torch.arange(batch_size, device=heroes.device).unsqueeze(1).expand_as(heroes)
        state[batch_indices[valid], categories[valid].long(), heroes[valid].long()] = 1.0
        return state

    def encode_lineups(
        self,
        history: torch.Tensor,
        acting_sides: torch.Tensor,
    ) -> torch.Tensor:
        if not self.lineup_encoder:
            raise RuntimeError("Lineup encoder is not enabled")
        hero_features = self.lineup_phi(self.lineup_hero_embedding.weight[:self.hero_count])
        picked_heroes = history[:, :2]
        summed = torch.matmul(picked_heroes, hero_features)
        counts = picked_heroes.sum(dim=2, keepdim=True).clamp_min(1)
        lineups = self.lineup_rho(summed / counts)
        batch_indices = torch.arange(history.shape[0], device=history.device)
        acting = lineups[batch_indices, acting_sides]
        opponent = lineups[batch_indices, 1 - acting_sides]
        return torch.cat([acting, opponent], dim=1)

    def encode_lineups_with_attention(
        self,
        heroes: torch.Tensor,
        categories: torch.Tensor,
        acting_sides: torch.Tensor,
    ) -> torch.Tensor:
        if not self.lineup_attention:
            raise RuntimeError("Attention lineup encoder is not enabled")
        picks = (categories == 0) | (categories == 1)
        safe_heroes = heroes.long().masked_fill(~picks, self.hero_count)
        absolute_sides = categories.long().clamp(0, 1)
        relative_sides = absolute_sides ^ acting_sides.unsqueeze(1)
        hero_tokens = (
            self.lineup_hero_embedding(safe_heroes)
            + self.lineup_side_embedding(relative_sides)
        )
        queries = self.lineup_query_tokens.expand(heroes.shape[0], -1, -1)
        tokens = torch.cat([queries, hero_tokens], dim=1)
        query_mask = torch.zeros((heroes.shape[0], 2), dtype=torch.bool, device=heroes.device)
        padding_mask = torch.cat([query_mask, ~picks], dim=1)
        encoded = self.lineup_attention_encoder(tokens, src_key_padding_mask=padding_mask)
        lineup_queries = self.lineup_attention_projection(encoded[:, :2])
        return lineup_queries.flatten(1)

    def set_hero_role_probabilities(self, probabilities: torch.Tensor) -> None:
        if probabilities.shape != self.hero_role_probabilities.shape:
            raise ValueError(
                f"Expected role probabilities with shape {tuple(self.hero_role_probabilities.shape)}, "
                f"got {tuple(probabilities.shape)}"
            )
        self.hero_role_probabilities.copy_(probabilities)

    def encode_role_coverage(self, history: torch.Tensor, acting_sides: torch.Tensor) -> torch.Tensor:
        picked_heroes = history[:, :2]
        log_uncovered = torch.log1p(-self.hero_role_probabilities.clamp(max=1 - 1e-6))
        coverage = 1.0 - torch.exp(torch.matmul(picked_heroes, log_uncovered))
        batch_indices = torch.arange(history.shape[0], device=history.device)
        acting = coverage[batch_indices, acting_sides]
        opponent = coverage[batch_indices, 1 - acting_sides]
        return torch.cat([acting, opponent], dim=1)

    def encode_sequence(self, heroes: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        if not self.sequence_encoder:
            raise RuntimeError("Sequence encoder is not enabled")
        padding = heroes < 0
        safe_heroes = heroes.long().masked_fill(padding, self.hero_count)
        safe_categories = categories.long().masked_fill(padding, 0)
        positions = torch.arange(heroes.shape[1], device=heroes.device) + 1
        tokens = (
            self.sequence_hero_embedding(safe_heroes)
            + self.sequence_category_embedding(safe_categories)
            + self.sequence_position_embedding(positions).unsqueeze(0)
        )
        cls = self.sequence_cls_token.expand(heroes.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        cls_padding = torch.zeros((heroes.shape[0], 1), dtype=torch.bool, device=heroes.device)
        encoded = self.sequence_transformer(
            tokens, src_key_padding_mask=torch.cat([cls_padding, padding], dim=1)
        )
        return encoded[:, 0]

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
        return_auxiliary: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        history = self.encode_history(history_heroes, history_categories)
        feature_parts = [
            history.flatten(1),
            self.patch_embedding(patch_indices),
            self.order_embedding(action_orders),
            self.side_embedding(acting_sides),
            self.action_embedding(action_types),
        ]
        if self.team_embedding is not None:
            if team_indices is None:
                raise ValueError("team_indices are required for this model")
            feature_parts.append(self.team_embedding(team_indices).flatten(1))
        if self.player_embedding is not None:
            if roster_indices is None:
                raise ValueError("roster_indices are required for this model")
            embedded_players = self.player_embedding(roster_indices)
            known_players = (roster_indices != 0).unsqueeze(-1)
            roster_sums = (embedded_players * known_players).sum(dim=2)
            roster_counts = known_players.sum(dim=2).clamp_min(1)
            feature_parts.append((roster_sums / roster_counts).flatten(1))
        if self.lineup_encoder:
            feature_parts.append(self.encode_lineups(history, acting_sides))
        if self.lineup_attention:
            feature_parts.append(
                self.encode_lineups_with_attention(history_heroes, history_categories, acting_sides)
            )
        if self.role_coverage:
            feature_parts.append(self.encode_role_coverage(history, acting_sides))
        if self.sequence_encoder:
            feature_parts.append(self.encode_sequence(history_heroes, history_categories))
        features = torch.cat(feature_parts, dim=1)
        encoded = self.network(features)
        if self.candidate_scorer:
            query = self.candidate_context_projection(encoded)
            logits = (
                torch.matmul(query, self.candidate_hero_embedding.weight.T) / math.sqrt(query.shape[1])
                + self.candidate_bias
            )
        elif self.split_action_heads:
            pick_logits = self.pick_head(encoded)
            ban_logits = self.ban_head(encoded)
            logits = torch.where(action_types.unsqueeze(1).bool(), pick_logits, ban_logits)
        elif self.outcome_auxiliary or self.role_auxiliary:
            logits = self.hero_head(encoded)
        else:
            logits = encoded
        used_heroes = history.amax(dim=1).bool()
        masked_logits = logits.masked_fill(used_heroes, -1e9)
        if return_auxiliary:
            auxiliary: dict[str, torch.Tensor] = {}
            if self.outcome_head is not None:
                auxiliary["outcome"] = self.outcome_head(encoded).squeeze(1)
            if self.role_head is not None:
                auxiliary["role"] = self.role_head(encoded)
            if not auxiliary:
                raise RuntimeError("No auxiliary head is enabled")
            return masked_logits, used_heroes, auxiliary
        return masked_logits, used_heroes
