from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset splits and expanded draft samples")
    parser.add_argument("--profile-dir", type=Path, default=Path("data/processed/profile"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/processed/model"))
    args = parser.parse_args()
    eligible = (args.profile_dir / "eligible_matches.parquet").resolve().as_posix()
    splits = (args.model_dir / "dataset_splits.parquet").resolve().as_posix()
    samples = (args.model_dir / "draft_samples.parquet").resolve().as_posix()

    connection = duckdb.connect()
    try:
        split_counts = connection.execute(f"""
            SELECT split, count(*) FROM read_parquet('{splits}')
            GROUP BY split ORDER BY split
        """).fetchall()
        sample_count, unique_ids, bad_histories = connection.execute(f"""
            SELECT count(*), count(DISTINCT sample_id),
                   count(*) FILTER (WHERE len(previous_hero_ids) <> action_order)
            FROM read_parquet('{samples}')
        """).fetchone()
        expected_actions = connection.execute(f"""
            SELECT sum(draft_action_count) FROM read_parquet('{eligible}')
        """).fetchone()[0]
        boundaries = connection.execute(f"""
            SELECT split, min(start_time), max(start_time)
            FROM read_parquet('{splits}') GROUP BY split
        """).fetchall()
    finally:
        connection.close()

    assert sample_count == expected_actions, (sample_count, expected_actions)
    assert unique_ids == sample_count, (unique_ids, sample_count)
    assert bad_histories == 0, bad_histories
    ranges = {split: (minimum, maximum) for split, minimum, maximum in boundaries}
    assert ranges["train"][1] <= ranges["validation"][0] <= ranges["test"][0], ranges
    print("split_counts:", split_counts)
    print("samples:", sample_count)
    print("unique_sample_ids:", unique_ids)
    print("bad_history_lengths:", bad_histories)
    print("time_ranges:", ranges)


if __name__ == "__main__":
    main()
