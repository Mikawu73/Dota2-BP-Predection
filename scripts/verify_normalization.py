from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/database/dota2.duckdb"))
    args = parser.parse_args()
    connection = duckdb.connect(str(args.database), read_only=True)
    connection.execute("SET TimeZone='UTC'")
    for table in ("test_matches", "test_match_players", "test_draft_actions", "test_normalization_manifest"):
        print(f"\n{table}: rows={connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]}")
        print("columns:", connection.execute(f"DESCRIBE {table}").fetchall())
        query = f"SELECT * FROM {table} ORDER BY 1 LIMIT 5"
        if table == "test_matches":
            query = "SELECT match_id,start_time,CAST(start_datetime_utc AS VARCHAR),patch_id,radiant_win,radiant_team_id,dire_team_id FROM test_matches ORDER BY match_id LIMIT 5"
        elif table == "test_normalization_manifest":
            query = "SELECT * EXCLUDE(normalized_at_utc),CAST(normalized_at_utc AS VARCHAR) FROM test_normalization_manifest ORDER BY source_file LIMIT 5"
        print("first_5:", connection.execute(query).fetchall())
    print("\nplayers_by_match:", connection.execute("""
        SELECT match_id, COUNT(*) AS player_count,
               SUM(CASE WHEN side = 'radiant' THEN 1 ELSE 0 END) AS radiant_players,
               SUM(CASE WHEN side = 'dire' THEN 1 ELSE 0 END) AS dire_players
        FROM test_match_players GROUP BY match_id ORDER BY match_id
    """).fetchall())
    print("\ndraft_by_match:", connection.execute("""
        SELECT match_id, COUNT(*) AS action_count,
               SUM(CASE WHEN is_pick THEN 1 ELSE 0 END) AS pick_count,
               SUM(CASE WHEN is_pick = FALSE THEN 1 ELSE 0 END) AS ban_count
        FROM test_draft_actions GROUP BY match_id ORDER BY match_id
    """).fetchall())
    print("\nnon_success:", connection.execute("""
        SELECT match_id, status, warning_messages
        FROM test_normalization_manifest WHERE status <> 'success'
    """).fetchall())
    connection.close()


if __name__ == "__main__":
    main()
