from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import duckdb


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def build_match_rosters(splits_path: Path, players_path: Path, output_path: Path) -> dict[str, object]:
    for path in (splits_path, players_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    connection = duckdb.connect()
    try:
        splits = _sql_path(splits_path)
        players = _sql_path(players_path)
        connection.execute(f"""
            COPY (
                SELECT s.match_id, s.start_time, s.patch_id, s.split,
                       list(p.account_id ORDER BY p.player_slot)
                           FILTER (WHERE p.side = 'radiant') AS radiant_player_ids,
                       list(p.account_id ORDER BY p.player_slot)
                           FILTER (WHERE p.side = 'dire') AS dire_player_ids
                FROM read_parquet({splits}) s
                JOIN read_parquet({players}) p USING (match_id)
                GROUP BY s.match_id, s.start_time, s.patch_id, s.split
                HAVING count(*) FILTER (WHERE p.side = 'radiant' AND p.account_id IS NOT NULL) = 5
                   AND count(*) FILTER (WHERE p.side = 'dire' AND p.account_id IS NOT NULL) = 5
                ORDER BY s.start_time, s.match_id
            ) TO {_sql_path(temp)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        roster_count, split_count = connection.execute(f"""
            SELECT
                (SELECT count(*) FROM read_parquet({_sql_path(temp)})),
                (SELECT count(*) FROM read_parquet({splits}))
        """).fetchone()
    finally:
        connection.close()
    if roster_count != split_count:
        temp.unlink(missing_ok=True)
        raise ValueError(f"Complete rosters found for {roster_count} of {split_count} split matches")
    temp.replace(output_path)

    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "splits_file": str(splits_path.resolve()),
        "players_file": str(players_path.resolve()),
        "output_file": str(output_path.resolve()),
        "matches_written": roster_count,
        "players_per_match": 10,
    }
    summary_path = output_path.with_name("match_rosters_summary.json")
    summary_temp = summary_path.with_suffix(summary_path.suffix + f".{os.getpid()}.tmp")
    summary_temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_temp.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build five-player rosters for eligible matches")
    parser.add_argument("--splits", type=Path, default=Path("data/processed/model/dataset_splits.parquet"))
    parser.add_argument("--players", type=Path, default=Path("data/processed/test/match_players.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/model/match_rosters.parquet"))
    args = parser.parse_args()
    print(json.dumps(build_match_rosters(args.splits, args.players, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
