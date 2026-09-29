from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_dataset_splits import build_splits
from scripts.build_draft_samples import build_draft_samples


def test_chronological_splits_and_draft_history(tmp_path: Path):
    eligible_path = tmp_path / "eligible.parquet"
    splits_path = tmp_path / "dataset_splits.parquet"
    drafts_path = tmp_path / "drafts.parquet"
    matches_path = tmp_path / "matches.parquet"
    positions_path = tmp_path / "positions.parquet"
    samples_path = tmp_path / "draft_samples.parquet"

    pq.write_table(pa.Table.from_pylist([
        {"match_id": 3, "start_time": 300, "patch_id": 1, "radiant_team_id": 10, "dire_team_id": 20},
        {"match_id": 1, "start_time": 100, "patch_id": 1, "radiant_team_id": 10, "dire_team_id": 20},
        {"match_id": 2, "start_time": 200, "patch_id": 1, "radiant_team_id": 10, "dire_team_id": 20},
    ]), eligible_path)
    build_splits(eligible_path, splits_path)

    splits = pq.read_table(splits_path).to_pylist()
    assert [(row["match_id"], row["split"]) for row in splits] == [
        (1, "train"), (2, "validation"), (3, "test")
    ]

    actions = []
    for match_id in (1, 2, 3):
        actions.extend([
            {"match_id": match_id, "order": 0, "team_side": "radiant", "is_pick": False, "action_type": "ban", "hero_id": 10 + match_id},
            {"match_id": match_id, "order": 1, "team_side": "dire", "is_pick": True, "action_type": "pick", "hero_id": 20 + match_id},
        ])
    pq.write_table(pa.Table.from_pylist(actions), drafts_path)
    pq.write_table(pa.Table.from_pylist([
        {"match_id": match_id, "radiant_win": match_id != 2}
        for match_id in (1, 2, 3)
    ]), matches_path)
    pq.write_table(pa.Table.from_pylist([
        {"match_id": match_id, "hero_id": 20 + match_id, "position": match_id + 1}
        for match_id in (1, 2, 3)
    ]), positions_path)
    summary = build_draft_samples(
        splits_path, drafts_path, matches_path, positions_path, samples_path, batch_size=2,
    )

    samples = pq.read_table(samples_path).to_pylist()
    assert summary["samples_written"] == 6
    assert all(len(row["previous_hero_ids"]) == row["action_order"] for row in samples)
    second = next(row for row in samples if row["match_id"] == 1 and row["action_order"] == 1)
    assert second["previous_hero_ids"] == [11]
    assert second["target_hero_id"] == 21
    assert second["acting_side_win"] is False
    assert second["target_position"] == 2
