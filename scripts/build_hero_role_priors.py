from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.roles import POSITIONS, infer_team_positions


POSITION_SCHEMA = pa.schema([
    ("match_id", pa.int64()), ("split", pa.string()), ("side", pa.string()),
    ("player_slot", pa.int64()), ("account_id", pa.int64()), ("hero_id", pa.int64()),
    ("position", pa.int64()), ("position_score", pa.float64()),
    ("assignment_confidence", pa.float64()), ("resource_percentile", pa.float64()),
])
PRIOR_SCHEMA = pa.schema([
    ("hero_id", pa.int64()), ("train_appearances", pa.int64()),
    *[(f"position_{position}_count", pa.int64()) for position in POSITIONS],
    *[(f"position_{position}_probability", pa.float64()) for position in POSITIONS],
])


def _write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


def build_role_priors(players_path: Path, splits_path: Path, output_dir: Path, smoothing: float = 5.0) -> dict:
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    players = pq.read_table(players_path).to_pylist()
    splits_table = pq.read_table(splits_path, columns=["match_id", "split"])
    split_by_match = dict(zip(splits_table["match_id"].to_pylist(), splits_table["split"].to_pylist()))
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for player in players:
        split = split_by_match.get(player["match_id"])
        if split is not None and player["side"] in {"radiant", "dire"}:
            grouped[(player["match_id"], player["side"])].append(player)

    inferred = []
    skipped = 0
    for (match_id, side), team_players in grouped.items():
        if len(team_players) != 5:
            skipped += 1
            continue
        split = split_by_match[match_id]
        for row in infer_team_positions(team_players):
            inferred.append({key: row[key] for key in (
                "match_id", "player_slot", "account_id", "hero_id", "position",
                "position_score", "assignment_confidence", "resource_percentile",
            )} | {"split": split, "side": side})

    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for row in inferred:
        if row["split"] == "train":
            counts[row["hero_id"]][row["position"]] += 1
    prior_rows = []
    for hero_id in sorted(counts):
        appearances = sum(counts[hero_id].values())
        denominator = appearances + smoothing
        row = {"hero_id": hero_id, "train_appearances": appearances}
        for position in POSITIONS:
            row[f"position_{position}_count"] = counts[hero_id][position]
            row[f"position_{position}_probability"] = (
                counts[hero_id][position] + smoothing / len(POSITIONS)
            ) / denominator
        prior_rows.append(row)

    _write(pa.Table.from_pylist(inferred, schema=POSITION_SCHEMA), output_dir / "player_positions.parquet")
    _write(pa.Table.from_pylist(prior_rows, schema=PRIOR_SCHEMA), output_dir / "hero_role_priors.parquet")
    confidence = [row["assignment_confidence"] for row in inferred]
    split_counts = Counter(row["split"] for row in inferred)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": "one-to-one assignment using lane compatibility, roaming, and within-team GPM/XPM/last-hit rank",
        "leakage_control": "hero role priors are aggregated from train matches only",
        "smoothing": smoothing,
        "eligible_team_matches": len(grouped), "skipped_non_five_player_teams": skipped,
        "inferred_player_rows_by_split": dict(sorted(split_counts.items())),
        "heroes_with_train_priors": len(prior_rows),
        "assignment_confidence_mean": sum(confidence) / len(confidence),
        "assignment_confidence_min": min(confidence),
        "outputs": {
            "player_positions": str((output_dir / "player_positions.parquet").resolve()),
            "hero_role_priors": str((output_dir / "hero_role_priors.parquet").resolve()),
        },
    }
    summary_path = output_dir / "hero_role_priors_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer player positions and build train-only hero role priors")
    parser.add_argument("--players", type=Path, default=Path("data/processed/test/match_players.parquet"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/model/dataset_splits.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/model"))
    parser.add_argument("--smoothing", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(build_role_priors(args.players, args.splits, args.output_dir, args.smoothing), indent=2))


if __name__ == "__main__":
    main()
