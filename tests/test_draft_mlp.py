from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.draft_mlp import DraftContextMLP, DraftDataset, build_vocabulary, history_category
from scripts.train_draft_mlp import pick_result_weights


def test_dataset_vocabulary_unknown_patch_and_used_hero_mask(tmp_path: Path):
    path = tmp_path / "samples.parquet"
    pq.write_table(pa.Table.from_pylist([
        {
            "match_id": 1, "split": "train", "patch_id": 58,
            "radiant_team_id": 100, "dire_team_id": 200,
            "action_order": 0, "acting_side": "radiant",
            "is_pick": False, "previous_hero_ids": [], "previous_is_pick": [],
            "previous_team_sides": [], "target_hero_id": 10,
        },
        {
            "match_id": 1, "split": "train", "patch_id": 58,
            "radiant_team_id": 100, "dire_team_id": 200,
            "action_order": 1, "acting_side": "dire",
            "is_pick": True, "previous_hero_ids": [10], "previous_is_pick": [False],
            "previous_team_sides": ["radiant"], "target_hero_id": 20,
        },
        {
            "match_id": 2, "split": "validation", "patch_id": 59,
            "radiant_team_id": 300, "dire_team_id": 400,
            "action_order": 1, "acting_side": "dire",
            "is_pick": True, "previous_hero_ids": [10], "previous_is_pick": [False],
            "previous_team_sides": ["radiant"], "target_hero_id": 20,
        },
    ]), path)

    vocabulary = build_vocabulary(path)
    assert vocabulary.hero_ids == (10, 20)
    validation = DraftDataset(path, "validation", vocabulary)
    assert validation.patch_indices.tolist() == [0]
    assert history_category(False, "radiant") == 2

    model = DraftContextMLP(hero_count=2, patch_count=1, max_action_order=1, hidden_sizes=(8, 4), dropout=0)
    assert torch.count_nonzero(model.patch_embedding.weight[0]).item() == 0
    batch = [tensor.unsqueeze(0) for tensor in validation[0][:-2]]
    logits, used = model(*batch)
    assert used.tolist() == [[True, False]]
    assert logits[0, 0].item() < -1e8
    assert torch.isfinite(logits[0, 1])


def test_team_and_roster_vocabularies_use_train_only(tmp_path: Path):
    samples_path = tmp_path / "samples.parquet"
    rosters_path = tmp_path / "rosters.parquet"
    sample_rows = []
    for split, match_id, radiant_team, dire_team in (
        ("train", 1, 100, 200), ("validation", 2, 300, 400)
    ):
        sample_rows.append({
            "match_id": match_id, "split": split, "patch_id": 58,
            "radiant_team_id": radiant_team, "dire_team_id": dire_team,
            "action_order": 0, "acting_side": "radiant", "is_pick": True,
            "previous_hero_ids": [], "previous_is_pick": [], "previous_team_sides": [],
            "target_hero_id": 10,
        })
    pq.write_table(pa.Table.from_pylist(sample_rows), samples_path)
    pq.write_table(pa.Table.from_pylist([
        {"match_id": 1, "split": "train", "radiant_player_ids": [1, 2, 3, 4, 5], "dire_player_ids": [6, 7, 8, 9, 10]},
        {"match_id": 2, "split": "validation", "radiant_player_ids": [11, 12, 13, 14, 15], "dire_player_ids": [16, 17, 18, 19, 20]},
    ]), rosters_path)

    vocabulary = build_vocabulary(samples_path, rosters_path, include_teams=True, include_players=True)
    assert vocabulary.team_ids == (100, 200)
    assert vocabulary.player_ids == tuple(range(1, 11))
    validation = DraftDataset(samples_path, "validation", vocabulary, rosters_path)
    assert validation.team_indices.tolist() == [[0, 0]]
    assert validation.roster_indices.tolist() == [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]

    model = DraftContextMLP(
        hero_count=1, patch_count=1, max_action_order=0,
        team_count=len(vocabulary.team_ids), player_count=len(vocabulary.player_ids),
        hidden_sizes=(8, 4), dropout=0,
    )
    batch = [tensor.unsqueeze(0) for tensor in validation[0][:-2]]
    logits, _ = model(*batch)
    assert logits.shape == (1, 1)


def test_split_action_heads_route_pick_and_ban_rows():
    model = DraftContextMLP(
        hero_count=3, patch_count=1, max_action_order=0,
        hidden_sizes=(8, 4), dropout=0, split_action_heads=True,
    )
    with torch.no_grad():
        model.pick_head.weight.zero_()
        model.pick_head.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
        model.ban_head.weight.zero_()
        model.ban_head.bias.copy_(torch.tensor([4.0, 5.0, 6.0]))

    logits, _ = model(
        torch.empty((2, 0), dtype=torch.int16),
        torch.empty((2, 0), dtype=torch.int8),
        torch.ones(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([1, 0], dtype=torch.long),
    )
    assert logits.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_lineup_encoder_ignores_bans_and_is_order_invariant():
    torch.manual_seed(1)
    model = DraftContextMLP(
        hero_count=4, patch_count=1, max_action_order=3,
        hidden_sizes=(8, 4), dropout=0, lineup_encoder=True,
    )
    heroes = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.int16)
    categories = torch.tensor([[0, 0, 2], [0, 0, 3]], dtype=torch.int8)
    acting_sides = torch.zeros(2, dtype=torch.long)

    encoded = model.encode_lineups(model.encode_history(heroes, categories), acting_sides)

    assert encoded.shape == (2, 128)
    assert torch.allclose(encoded[0], encoded[1])


def test_recency_weights_halve_after_one_half_life(tmp_path: Path):
    path = tmp_path / "samples.parquet"
    base = {
        "split": "train", "patch_id": 58,
        "radiant_team_id": 100, "dire_team_id": 200,
        "action_order": 0, "acting_side": "radiant", "is_pick": True,
        "previous_hero_ids": [], "previous_is_pick": [], "previous_team_sides": [],
        "target_hero_id": 10,
    }
    pq.write_table(pa.Table.from_pylist([
        {**base, "match_id": 1, "start_time": 1_000_000},
        {**base, "match_id": 2, "start_time": 1_086_400},
    ]), path)
    vocabulary = build_vocabulary(path)

    dataset = DraftDataset(path, "train", vocabulary, recency_half_life_days=1)

    assert torch.allclose(dataset.sample_weights, torch.tensor([0.5, 1.0]))


def test_lineup_attention_ignores_bans_and_pick_order():
    torch.manual_seed(1)
    model = DraftContextMLP(
        hero_count=5, patch_count=1, max_action_order=3,
        hidden_sizes=(8, 4), dropout=0, lineup_attention=True,
    )
    model.eval()
    heroes = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.int16)
    categories = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.int8)
    acting_sides = torch.zeros(2, dtype=torch.long)

    encoded = model.encode_lineups_with_attention(heroes, categories, acting_sides)

    assert encoded.shape == (2, 128)
    assert torch.allclose(encoded[0], encoded[1], atol=1e-6)


def test_role_coverage_uses_picks_and_relative_sides():
    model = DraftContextMLP(
        hero_count=3, patch_count=1, max_action_order=2,
        hidden_sizes=(8, 4), dropout=0, role_coverage=True,
    )
    model.set_hero_role_probabilities(torch.tensor([
        [0.9, 0.025, 0.025, 0.025, 0.025],
        [0.025, 0.025, 0.025, 0.025, 0.9],
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ]))
    history = model.encode_history(
        torch.tensor([[0, 1, 2]], dtype=torch.int16),
        torch.tensor([[0, 1, 2]], dtype=torch.int8),
    )
    coverage = model.encode_role_coverage(history, torch.tensor([0]))

    assert coverage.shape == (1, 10)
    assert torch.isclose(coverage[0, 0], torch.tensor(0.9))
    assert torch.isclose(coverage[0, 9], torch.tensor(0.9))
    assert coverage[0, 1] < 0.03


def test_candidate_scorer_outputs_one_score_per_hero_and_masks_used_heroes():
    model = DraftContextMLP(
        hero_count=4, patch_count=1, max_action_order=2,
        hidden_sizes=(8, 4), dropout=0, lineup_encoder=True, candidate_scorer=True,
    )
    logits, used = model(
        torch.tensor([[0, 1]], dtype=torch.int16),
        torch.tensor([[0, 2]], dtype=torch.int8),
        torch.tensor([1]), torch.tensor([2]), torch.tensor([0]), torch.tensor([1]),
    )

    assert logits.shape == (1, 4)
    assert used.tolist() == [[True, True, False, False]]
    assert logits[0, 0] < -1e8 and logits[0, 1] < -1e8
    assert torch.isfinite(logits[0, 2:]).all()


def test_sequence_encoder_preserves_action_order_and_handles_empty_history():
    torch.manual_seed(5)
    model = DraftContextMLP(
        hero_count=4, patch_count=1, max_action_order=3,
        hidden_sizes=(8, 4), dropout=0, sequence_encoder=True,
    )
    model.eval()
    heroes = torch.tensor([[0, 1, -1], [1, 0, -1], [-1, -1, -1]], dtype=torch.int16)
    categories = torch.tensor([[0, 1, -1], [1, 0, -1], [-1, -1, -1]], dtype=torch.int8)

    encoded = model.encode_sequence(heroes, categories)

    assert encoded.shape == (3, 32)
    assert not torch.allclose(encoded[0], encoded[1])
    assert torch.isfinite(encoded).all()


def test_outcome_auxiliary_dataset_and_head(tmp_path: Path):
    path = tmp_path / "samples.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "match_id": 1, "split": "train", "patch_id": 58,
        "radiant_team_id": 100, "dire_team_id": 200,
        "action_order": 0, "acting_side": "radiant", "is_pick": True,
        "previous_hero_ids": [], "previous_is_pick": [], "previous_team_sides": [],
        "target_hero_id": 10, "acting_side_win": True,
    }]), path)
    vocabulary = build_vocabulary(path)
    dataset = DraftDataset(path, "train", vocabulary, include_outcome_label=True)
    assert dataset[0][-1].item() == 1.0

    model = DraftContextMLP(
        hero_count=1, patch_count=1, max_action_order=0,
        hidden_sizes=(8, 4), dropout=0, outcome_auxiliary=True,
    )
    inputs = [tensor.unsqueeze(0) for tensor in dataset[0][:-3]]
    logits, used, auxiliary = model(*inputs, return_auxiliary=True)
    assert logits.shape == (1, 1)
    assert used.shape == (1, 1)
    assert auxiliary["outcome"].shape == (1,)
    assert torch.isfinite(auxiliary["outcome"]).all()


def test_role_auxiliary_uses_pick_labels_only(tmp_path: Path):
    path = tmp_path / "samples.parquet"
    base = {
        "match_id": 1, "split": "train", "patch_id": 58,
        "radiant_team_id": 100, "dire_team_id": 200,
        "previous_hero_ids": [], "previous_is_pick": [], "previous_team_sides": [],
        "target_hero_id": 10,
    }
    pq.write_table(pa.Table.from_pylist([
        {**base, "action_order": 0, "acting_side": "radiant", "is_pick": True, "target_position": 5},
        {**base, "action_order": 1, "acting_side": "dire", "is_pick": False, "target_position": None},
    ]), path)
    vocabulary = build_vocabulary(path)
    dataset = DraftDataset(path, "train", vocabulary, include_role_label=True)
    assert [dataset[index][-1].item() for index in range(2)] == [4, -1]

    model = DraftContextMLP(
        hero_count=1, patch_count=1, max_action_order=1,
        hidden_sizes=(8, 4), dropout=0, role_auxiliary=True,
    )
    inputs = [tensor.unsqueeze(0) for tensor in dataset[0][:-3]]
    logits, _, auxiliary = model(*inputs, return_auxiliary=True)
    assert logits.shape == (1, 1)
    assert auxiliary["role"].shape == (1, 5)


def test_pick_result_weights_leave_bans_unchanged():
    weights = pick_result_weights(
        torch.tensor([1, 1, 0, 0]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        winning_pick_weight=1.5,
        losing_pick_weight=0.5,
    )
    assert weights.tolist() == [1.5, 0.5, 1.0, 1.0]
