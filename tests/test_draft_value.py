from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts.train_draft_value import binary_auc, train_value_model
from src.draft_mlp import build_vocabulary
from src.draft_value import DraftValueDataset, DraftValueModel


def _write_samples(path: Path) -> None:
    base = {
        "match_id": 1, "split": "train", "start_time": 1000, "patch_id": 58,
        "radiant_team_id": 100, "dire_team_id": 200,
        "previous_hero_ids": [], "previous_is_pick": [], "previous_team_sides": [],
    }
    pq.write_table(pa.Table.from_pylist([
        {**base, "action_order": 0, "acting_side": "radiant", "is_pick": False,
         "target_hero_id": 10, "acting_side_win": True},
        {**base, "action_order": 1, "acting_side": "dire", "is_pick": True,
         "target_hero_id": 20, "acting_side_win": False},
    ]), path)


def test_value_dataset_keeps_only_factual_picks(tmp_path: Path):
    path = tmp_path / "samples.parquet"
    _write_samples(path)
    vocabulary = build_vocabulary(path, include_teams=True)
    dataset = DraftValueDataset(path, "train", vocabulary)

    assert len(dataset) == 1
    assert dataset[0][8].item() == vocabulary.hero_to_index[20]
    assert dataset[0][10].item() == 0.0


def test_value_model_scores_candidate_heroes():
    torch.manual_seed(3)
    model = DraftValueModel(hero_count=3, patch_count=1, max_action_order=2, team_count=2)
    logits = model(
        torch.tensor([[0, -1], [0, -1]], dtype=torch.int16),
        torch.tensor([[0, -1], [0, -1]], dtype=torch.int8),
        torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([0, 0]),
        torch.tensor([[1, 2], [1, 2]]), torch.tensor([1, 2]),
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert logits[0] != logits[1]


def test_binary_auc_handles_order_and_ties():
    assert binary_auc([0, 1], [0.1, 0.9]) == 1.0
    assert binary_auc([0, 1], [0.5, 0.5]) == 0.5


def test_value_training_writes_checkpoint_and_summary(tmp_path: Path):
    samples = tmp_path / "samples.parquet"
    output = tmp_path / "value"
    rows = []
    for split, match_offset in (("train", 0), ("validation", 10)):
        for index, won in enumerate((False, True)):
            rows.append({
                "match_id": match_offset + index, "split": split,
                "start_time": 1000 + match_offset + index, "patch_id": 58,
                "radiant_team_id": 100, "dire_team_id": 200,
                "action_order": index, "acting_side": "radiant", "is_pick": True,
                "previous_hero_ids": [], "previous_is_pick": [], "previous_team_sides": [],
                "target_hero_id": 10 + index, "acting_side_win": won,
            })
    pq.write_table(pa.Table.from_pylist(rows), samples)

    summary = train_value_model(
        samples, output, epochs=1, batch_size=2, accumulation_steps=1,
        recency_half_life_days=None,
    )

    assert summary["best_epoch"] == 1
    assert summary["best_validation_metrics"]["samples"] == 2
    assert (output / "draft_value_best.pt").is_file()
    assert (output / "draft_value_training_summary.json").is_file()
