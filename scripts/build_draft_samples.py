from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


SAMPLE_SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("match_id", pa.int64()),
    ("start_time", pa.int64()),
    ("patch_id", pa.int64()),
    ("radiant_team_id", pa.int64()),
    ("dire_team_id", pa.int64()),
    ("split", pa.string()),
    ("action_order", pa.int64()),
    ("acting_side", pa.string()),
    ("is_pick", pa.bool_()),
    ("action_type", pa.string()),
    ("previous_hero_ids", pa.list_(pa.int64())),
    ("previous_is_pick", pa.list_(pa.bool_())),
    ("previous_team_sides", pa.list_(pa.string())),
    ("target_hero_id", pa.int64()),
    ("acting_side_win", pa.bool_()),
    ("target_position", pa.int64()),
])


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def build_draft_samples(
    splits_path: Path,
    drafts_path: Path,
    matches_path: Path,
    positions_path: Path,
    output_path: Path,
    batch_size: int = 10_000,
) -> dict[str, object]:
    for path in (splits_path, drafts_path, matches_path, positions_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    connection = duckdb.connect()
    writer = pq.ParquetWriter(temp, SAMPLE_SCHEMA, compression="zstd")
    counts: Counter[str] = Counter()
    samples_written = 0
    try:
        cursor = connection.execute(f"""
            SELECT s.match_id, s.start_time, s.patch_id,
                   s.radiant_team_id, s.dire_team_id, s.split,
                   d."order", d.team_side, d.is_pick, d.action_type, d.hero_id,
                   CASE WHEN d.team_side = 'radiant' THEN m.radiant_win ELSE NOT m.radiant_win END,
                   CASE WHEN d.is_pick THEN pp.position ELSE NULL END
            FROM read_parquet({_sql_path(splits_path)}) s
            JOIN read_parquet({_sql_path(drafts_path)}) d USING (match_id)
            JOIN read_parquet({_sql_path(matches_path)}) m USING (match_id)
            LEFT JOIN read_parquet({_sql_path(positions_path)}) pp
              ON d.match_id = pp.match_id AND d.hero_id = pp.hero_id
            ORDER BY s.start_time, s.match_id, d."order"
        """)
        current_match: int | None = None
        previous_heroes: list[int] = []
        previous_picks: list[bool] = []
        previous_sides: list[str] = []
        buffer: list[dict[str, object]] = []

        with tqdm(desc="draft samples", unit="sample") as progress:
            while rows := cursor.fetchmany(batch_size):
                for match_id, start_time, patch_id, radiant_id, dire_id, split, order, side, is_pick, action_type, hero_id, acting_side_win, target_position in rows:
                    if match_id != current_match:
                        current_match = match_id
                        previous_heroes = []
                        previous_picks = []
                        previous_sides = []
                    buffer.append({
                        "sample_id": f"{match_id}:{order}",
                        "match_id": match_id,
                        "start_time": start_time,
                        "patch_id": patch_id,
                        "radiant_team_id": radiant_id,
                        "dire_team_id": dire_id,
                        "split": split,
                        "action_order": order,
                        "acting_side": side,
                        "is_pick": is_pick,
                        "action_type": action_type,
                        "previous_hero_ids": list(previous_heroes),
                        "previous_is_pick": list(previous_picks),
                        "previous_team_sides": list(previous_sides),
                        "target_hero_id": hero_id,
                        "acting_side_win": acting_side_win,
                        "target_position": target_position,
                    })
                    previous_heroes.append(hero_id)
                    previous_picks.append(is_pick)
                    previous_sides.append(side)
                    counts[split] += 1
                    samples_written += 1
                    if len(buffer) >= batch_size:
                        writer.write_table(pa.Table.from_pylist(buffer, schema=SAMPLE_SCHEMA))
                        progress.update(len(buffer))
                        buffer.clear()
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=SAMPLE_SCHEMA))
                progress.update(len(buffer))
    except Exception:
        writer.close()
        temp.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        temp.replace(output_path)
    finally:
        connection.close()

    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "splits_file": str(splits_path.resolve()),
        "drafts_file": str(drafts_path.resolve()),
        "matches_file": str(matches_path.resolve()),
        "positions_file": str(positions_path.resolve()),
        "output_file": str(output_path.resolve()),
        "samples_written": samples_written,
        "samples_by_split": dict(counts),
        "target": "hero_id for the current pick/ban action",
        "history_rule": "each sample contains only actions with a smaller action_order from the same match",
    }
    summary_path = output_path.with_name("draft_samples_summary.json")
    summary_temp = summary_path.with_suffix(summary_path.suffix + f".{os.getpid()}.tmp")
    summary_temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_temp.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand complete drafts into next-action prediction samples")
    parser.add_argument("--splits", type=Path, default=Path("data/processed/model/dataset_splits.parquet"))
    parser.add_argument("--drafts", type=Path, default=Path("data/processed/test/draft_actions.parquet"))
    parser.add_argument("--matches", type=Path, default=Path("data/processed/test/matches.parquet"))
    parser.add_argument("--positions", type=Path, default=Path("data/processed/model/player_positions.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/model/draft_samples.parquet"))
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(
        build_draft_samples(
            args.splits, args.drafts, args.matches, args.positions, args.output, args.batch_size,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
