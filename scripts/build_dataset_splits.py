from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SPLIT_SCHEMA = pa.schema([
    ("match_id", pa.int64()),
    ("start_time", pa.int64()),
    ("patch_id", pa.int64()),
    ("radiant_team_id", pa.int64()),
    ("dire_team_id", pa.int64()),
    ("split", pa.string()),
    ("chronological_rank", pa.int64()),
])


def _atomic_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(table, temp, compression="zstd")
    temp.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def build_splits(input_path: Path, output_path: Path, train_ratio: float = 0.70, validation_ratio: float = 0.15) -> dict[str, object]:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Ratios must be positive and leave a positive test split")
    rows = pq.read_table(input_path, columns=[
        "match_id", "start_time", "patch_id", "radiant_team_id", "dire_team_id"
    ]).to_pylist()
    if len(rows) < 3:
        raise ValueError("At least three eligible matches are required")
    rows.sort(key=lambda row: (row["start_time"], row["match_id"]))

    train_end = min(max(1, int(len(rows) * train_ratio)), len(rows) - 2)
    validation_end = max(train_end + 1, int(len(rows) * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, len(rows) - 1)
    output_rows = []
    for rank, row in enumerate(rows):
        split = "train" if rank < train_end else "validation" if rank < validation_end else "test"
        output_rows.append({
            "match_id": row["match_id"],
            "start_time": row["start_time"],
            "patch_id": row["patch_id"],
            "radiant_team_id": row["radiant_team_id"],
            "dire_team_id": row["dire_team_id"],
            "split": split,
            "chronological_rank": rank,
        })
    _atomic_parquet(pa.Table.from_pylist(output_rows, schema=SPLIT_SCHEMA), output_path)

    split_summary: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_file": str(input_path.resolve()),
        "output_file": str(output_path.resolve()),
        "total_matches": len(output_rows),
        "ratios": {"train": train_ratio, "validation": validation_ratio, "test": 1 - train_ratio - validation_ratio},
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        selected = [row for row in output_rows if row["split"] == split]
        split_summary["splits"][split] = {
            "matches": len(selected),
            "earliest_start_time": selected[0]["start_time"],
            "latest_start_time": selected[-1]["start_time"],
        }
    _atomic_json(output_path.with_name("dataset_splits_summary.json"), split_summary)
    return split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create chronological train/validation/test splits")
    parser.add_argument("--input", type=Path, default=Path("data/processed/profile/eligible_matches.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/model/dataset_splits.parquet"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    args = parser.parse_args()
    print(json.dumps(build_splits(args.input, args.output, args.train_ratio, args.validation_ratio), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
