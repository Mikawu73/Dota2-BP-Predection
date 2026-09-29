from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import duckdb


REQUIRED_FILES = (
    "matches.parquet",
    "match_players.parquet",
    "draft_actions.parquet",
    "normalization_manifest.parquet",
)


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _copy_parquet(connection: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    connection.execute(f"COPY ({query}) TO {_sql_path(temp)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    temp.replace(path)


def profile_dataset(input_dir: Path, output_dir: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing normalized files in {input_dir}: {', '.join(missing)}")

    connection = duckdb.connect()
    try:
        for name in REQUIRED_FILES:
            table_name = name.removesuffix(".parquet")
            connection.execute(
                f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet({_sql_path(input_dir / name)})"
            )

        connection.execute("""
            CREATE TEMP VIEW player_stats AS
            SELECT match_id,
                   count(*) AS player_count,
                   count(*) FILTER (WHERE hero_id IS NULL) AS missing_player_hero_ids,
                   count(*) FILTER (WHERE side NOT IN ('radiant', 'dire')) AS unknown_player_sides
            FROM match_players GROUP BY match_id
        """)
        connection.execute("""
            CREATE TEMP VIEW draft_stats AS
            SELECT match_id,
                   count(*) AS draft_action_count,
                   count(*) FILTER (WHERE is_pick) AS pick_count,
                   count(*) FILTER (WHERE is_pick = FALSE) AS ban_count,
                   count(*) FILTER (WHERE hero_id IS NULL) AS missing_draft_hero_ids,
                   count(*) FILTER (WHERE team_side NOT IN ('radiant', 'dire')) AS unknown_draft_teams,
                   count(DISTINCT "order") AS distinct_order_count,
                   min("order") AS min_order,
                   max("order") AS max_order
            FROM draft_actions GROUP BY match_id
        """)
        connection.execute("""
            CREATE TEMP VIEW assessed_matches AS
            SELECT m.*,
                   coalesce(p.player_count, 0) AS player_count,
                   coalesce(p.missing_player_hero_ids, 0) AS missing_player_hero_ids,
                   coalesce(p.unknown_player_sides, 0) AS unknown_player_sides,
                   coalesce(d.draft_action_count, 0) AS draft_action_count,
                   coalesce(d.pick_count, 0) AS pick_count,
                   coalesce(d.ban_count, 0) AS ban_count,
                   coalesce(d.missing_draft_hero_ids, 0) AS missing_draft_hero_ids,
                   coalesce(d.unknown_draft_teams, 0) AS unknown_draft_teams,
                   coalesce(d.distinct_order_count, 0) AS distinct_order_count,
                   d.min_order,
                   d.max_order,
                   nm.status,
                   nm.error_count,
                   nm.draft_picks_match_players,
                   (
                       m.match_id IS NOT NULL
                       AND m.start_time IS NOT NULL
                       AND m.patch_id IS NOT NULL
                       AND m.radiant_win IS NOT NULL
                       AND m.radiant_team_id IS NOT NULL
                       AND m.dire_team_id IS NOT NULL
                       AND m.radiant_team_id <> m.dire_team_id
                       AND coalesce(p.player_count, 0) = 10
                       AND coalesce(p.missing_player_hero_ids, 0) = 0
                       AND coalesce(p.unknown_player_sides, 0) = 0
                       AND coalesce(d.draft_action_count, 0) > 0
                       AND coalesce(d.pick_count, 0) = 10
                       AND coalesce(d.missing_draft_hero_ids, 0) = 0
                       AND coalesce(d.unknown_draft_teams, 0) = 0
                       AND d.distinct_order_count = d.draft_action_count
                       AND d.min_order = 0
                       AND d.max_order = d.draft_action_count - 1
                       AND nm.error_count = 0
                       AND nm.draft_picks_match_players = TRUE
                   ) AS is_eligible
            FROM matches m
            LEFT JOIN player_stats p USING (match_id)
            LEFT JOIN draft_stats d USING (match_id)
            LEFT JOIN normalization_manifest nm USING (match_id)
        """)

        eligible_query = """
            SELECT match_id, start_time, start_datetime_utc, patch_id,
                   radiant_team_id, dire_team_id, radiant_win,
                   player_count, draft_action_count, pick_count, ban_count
            FROM assessed_matches WHERE is_eligible
            ORDER BY start_time, match_id
        """
        _copy_parquet(connection, eligible_query, output_dir / "eligible_matches.parquet")
        _copy_parquet(connection, """
            SELECT patch_id,
                   count(*) AS total_matches,
                   count(*) FILTER (WHERE is_eligible) AS eligible_matches,
                   min(start_time) AS earliest_start_time,
                   max(start_time) AS latest_start_time
            FROM assessed_matches GROUP BY patch_id ORDER BY patch_id
        """, output_dir / "patch_profile.parquet")
        _copy_parquet(connection, """
            SELECT draft_action_count, pick_count, ban_count,
                   count(*) AS match_count,
                   count(*) FILTER (WHERE is_eligible) AS eligible_match_count
            FROM assessed_matches GROUP BY ALL
            ORDER BY draft_action_count, pick_count, ban_count
        """, output_dir / "draft_length_profile.parquet")
        _copy_parquet(connection, """
            SELECT hero_id,
                   count(*) FILTER (WHERE is_pick) AS pick_count,
                   count(*) FILTER (WHERE is_pick = FALSE) AS ban_count,
                   count(*) AS total_actions
            FROM draft_actions WHERE hero_id IS NOT NULL
            GROUP BY hero_id ORDER BY total_actions DESC, hero_id
        """, output_dir / "hero_frequency.parquet")
        _copy_parquet(connection, """
            WITH appearances AS (
                SELECT radiant_team_id AS team_id, match_id FROM matches WHERE radiant_team_id IS NOT NULL
                UNION ALL
                SELECT dire_team_id AS team_id, match_id FROM matches WHERE dire_team_id IS NOT NULL
            )
            SELECT team_id, count(DISTINCT match_id) AS match_count
            FROM appearances GROUP BY team_id ORDER BY match_count DESC, team_id
        """, output_dir / "team_frequency.parquet")

        total, eligible, earliest, latest = connection.execute("""
            SELECT count(*), count(*) FILTER (WHERE is_eligible),
                   min(start_time), max(start_time)
            FROM assessed_matches
        """).fetchone()
        reason_queries = {
            "missing_start_time": "start_time IS NULL",
            "missing_patch_id": "patch_id IS NULL",
            "missing_result": "radiant_win IS NULL",
            "invalid_team_ids": "radiant_team_id IS NULL OR dire_team_id IS NULL OR radiant_team_id = dire_team_id",
            "not_10_players": "player_count <> 10",
            "invalid_player_heroes_or_sides": "missing_player_hero_ids > 0 OR unknown_player_sides > 0",
            "missing_draft": "draft_action_count = 0",
            "not_10_picks": "pick_count <> 10",
            "invalid_draft_heroes_or_teams": "missing_draft_hero_ids > 0 OR unknown_draft_teams > 0",
            "invalid_draft_order": "distinct_order_count <> draft_action_count OR min_order <> 0 OR max_order <> draft_action_count - 1",
            "draft_does_not_match_players": "draft_picks_match_players IS DISTINCT FROM TRUE",
            "normalization_errors": "error_count IS NULL OR error_count > 0",
        }
        exclusion_counts = {
            name: connection.execute(f"SELECT count(*) FROM assessed_matches WHERE {condition}").fetchone()[0]
            for name, condition in reason_queries.items()
        }
        report: dict[str, object] = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "input_directory": str(input_dir.resolve()),
            "total_matches": total,
            "eligible_matches": eligible,
            "excluded_matches": total - eligible,
            "eligible_percentage": round(eligible * 100 / total, 3) if total else 0.0,
            "earliest_match_utc": datetime.fromtimestamp(earliest, UTC).isoformat() if earliest else None,
            "latest_match_utc": datetime.fromtimestamp(latest, UTC).isoformat() if latest else None,
            "exclusion_counts_non_exclusive": exclusion_counts,
            "eligibility_rule": "10 players, 10 picks, valid continuous draft, known teams/version/result, and draft picks match final heroes",
        }
        _write_json(output_dir / "dataset_profile.json", report)
        return report
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile normalized Dota 2 draft data")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/profile"))
    args = parser.parse_args()
    print(json.dumps(profile_dataset(args.input_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
