from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


METHODS = ("global", "context", "team", "roster", "combined")
PREDICTION_SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("match_id", pa.int64()),
    ("split", pa.string()),
    ("patch_id", pa.int64()),
    ("action_order", pa.int64()),
    ("action_type", pa.string()),
    ("acting_side", pa.string()),
    ("target_hero_id", pa.int64()),
    *[(f"{method}_rank", pa.int64()) for method in METHODS],
])


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _probabilities(counter: Counter[int], hero_ids: list[int]) -> list[float]:
    total = sum(counter.values())
    if not total:
        return [0.0] * len(hero_ids)
    return [counter[hero_id] / total for hero_id in hero_ids]


def _blend(parts: Iterable[tuple[float, list[float]]], fallback: list[float]) -> list[float]:
    active = [(weight, vector) for weight, vector in parts if any(vector)]
    if not active:
        return fallback
    total_weight = sum(weight for weight, _ in active)
    return [sum(weight * vector[index] for weight, vector in active) / total_weight for index in range(len(fallback))]


def _rank(target: int, used: set[int], scores: list[float], hero_ids: list[int], hero_index: dict[int, int]) -> int:
    target_index = hero_index.get(target)
    if target_index is None:
        return len(hero_ids) + 1
    target_score = scores[target_index]
    return 1 + sum(
        hero_id not in used
        and (score > target_score or (score == target_score and hero_id < target))
        for hero_id, score in zip(hero_ids, scores)
        if hero_id != target
    )


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def evaluate_baselines(
    samples_path: Path,
    rosters_path: Path,
    players_path: Path,
    output_dir: Path,
    batch_size: int = 10_000,
) -> dict[str, object]:
    for path in (samples_path, rosters_path, players_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    samples_sql = _sql_path(samples_path)
    rosters_sql = _sql_path(rosters_path)
    players_sql = _sql_path(players_path)

    global_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    phase_counts: defaultdict[tuple[int, str, str], Counter[int]] = defaultdict(Counter)
    context_counts: defaultdict[tuple[int, int, str, str], Counter[int]] = defaultdict(Counter)
    team_counts: defaultdict[tuple[int, str], Counter[int]] = defaultdict(Counter)
    team_patch_counts: defaultdict[tuple[int, int, str], Counter[int]] = defaultdict(Counter)
    training_rows = connection.execute(f"""
        SELECT patch_id, action_order, action_type, acting_side,
               target_hero_id, radiant_team_id, dire_team_id
        FROM read_parquet({samples_sql}) WHERE split = 'train'
    """).fetchall()
    for patch, order, action_type, side, hero_id, radiant_id, dire_id in training_rows:
        acting_team = radiant_id if side == "radiant" else dire_id
        global_counts[action_type][hero_id] += 1
        phase_counts[(order, action_type, side)][hero_id] += 1
        context_counts[(patch, order, action_type, side)][hero_id] += 1
        team_counts[(acting_team, action_type)][hero_id] += 1
        team_patch_counts[(acting_team, patch, action_type)][hero_id] += 1
    del training_rows

    hero_ids = sorted({hero_id for counter in global_counts.values() for hero_id in counter})
    hero_index = {hero_id: index for index, hero_id in enumerate(hero_ids)}
    global_vectors = {action: _probabilities(counter, hero_ids) for action, counter in global_counts.items()}
    zero_vector = [0.0] * len(hero_ids)

    player_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
    player_patch_counts: defaultdict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for account_id, patch_id, hero_id, matches in connection.execute(f"""
        SELECT p.account_id, s.patch_id, p.hero_id, count(*)
        FROM read_parquet({players_sql}) p
        JOIN (SELECT match_id, patch_id FROM read_parquet({rosters_sql}) WHERE split = 'train') s USING (match_id)
        WHERE p.account_id IS NOT NULL AND p.hero_id IS NOT NULL
        GROUP BY p.account_id, s.patch_id, p.hero_id
    """).fetchall():
        player_counts[account_id][hero_id] += matches
        player_patch_counts[(account_id, patch_id)][hero_id] += matches

    rosters = {
        match_id: {"radiant": list(radiant), "dire": list(dire)}
        for match_id, radiant, dire in connection.execute(f"""
            SELECT match_id, radiant_player_ids, dire_player_ids FROM read_parquet({rosters_sql})
        """).fetchall()
    }

    context_cache: dict[tuple[int, int, str, str], list[float]] = {}
    team_cache: dict[tuple[int, int, str], list[float]] = {}
    roster_cache: dict[tuple[int, int, str], list[float]] = {}

    def context_vector(patch: int, order: int, action: str, side: str) -> list[float]:
        key = (patch, order, action, side)
        if key not in context_cache:
            exact = _probabilities(context_counts[key], hero_ids)
            phase = _probabilities(phase_counts[(order, action, side)], hero_ids)
            context_cache[key] = _blend(
                ((0.65, exact), (0.25, phase), (0.10, global_vectors[action]))
                if sum(context_counts[key].values()) >= 20
                else ((0.8, phase), (0.2, global_vectors[action])),
                global_vectors[action],
            )
        return context_cache[key]

    def team_vector(team_id: int, patch: int, action: str) -> list[float]:
        key = (team_id, patch, action)
        if key not in team_cache:
            overall = _probabilities(team_counts[(team_id, action)], hero_ids)
            patch_counter = team_patch_counts[(team_id, patch, action)]
            exact = _probabilities(patch_counter, hero_ids)
            team_cache[key] = _blend(
                ((0.7, exact), (0.3, overall)) if sum(patch_counter.values()) >= 10 else ((1.0, overall),),
                zero_vector,
            )
        return team_cache[key]

    def roster_vector(match_id: int, patch: int, roster_side: str) -> list[float]:
        key = (match_id, patch, roster_side)
        if key not in roster_cache:
            player_vectors: list[list[float]] = []
            for account_id in rosters[match_id][roster_side]:
                overall = _probabilities(player_counts[account_id], hero_ids)
                patch_counter = player_patch_counts[(account_id, patch)]
                exact = _probabilities(patch_counter, hero_ids)
                player_vectors.append(_blend(
                    ((0.7, exact), (0.3, overall)) if sum(patch_counter.values()) >= 5 else ((1.0, overall),),
                    zero_vector,
                ))
            scores = []
            for index in range(len(hero_ids)):
                preferences = sorted((vector[index] for vector in player_vectors), reverse=True)
                scores.append(preferences[0] + 0.25 * preferences[1])
            total = sum(scores)
            roster_cache[key] = [score / total for score in scores] if total else zero_vector
        return roster_cache[key]

    metric_totals: defaultdict[tuple[str, str, str, str], list[float]] = defaultdict(lambda: [0, 0, 0, 0, 0.0])
    unseen_targets: Counter[str] = Counter()
    predictions_path = output_dir / "statistical_baseline_predictions.parquet"
    predictions_temp = predictions_path.with_suffix(predictions_path.suffix + f".{os.getpid()}.tmp")
    writer = pq.ParquetWriter(predictions_temp, PREDICTION_SCHEMA, compression="zstd")
    prediction_buffer: list[dict[str, object]] = []

    def record(method: str, split: str, action: str, order: int, patch: int, rank: int) -> None:
        for scope, value in (("overall", "all"), ("action_type", action), ("action_order", str(order)), ("patch", str(patch))):
            totals = metric_totals[(method, split, scope, value)]
            totals[0] += 1
            totals[1] += rank <= 1
            totals[2] += rank <= 3
            totals[3] += rank <= 5
            totals[4] += 1 / rank if rank <= len(hero_ids) else 0.0

    try:
        cursor = connection.execute(f"""
            SELECT sample_id, match_id, split, patch_id, action_order,
                   action_type, acting_side, target_hero_id, previous_hero_ids,
                   radiant_team_id, dire_team_id
            FROM read_parquet({samples_sql})
            WHERE split IN ('validation', 'test')
            ORDER BY split, start_time, match_id, action_order
        """)
        with tqdm(desc="evaluate baselines", unit="sample") as progress:
            while rows := cursor.fetchmany(batch_size):
                for sample_id, match_id, split, patch, order, action, side, target, previous, radiant_id, dire_id in rows:
                    acting_team = radiant_id if side == "radiant" else dire_id
                    roster_side = side if action == "pick" else "dire" if side == "radiant" else "radiant"
                    global_score = global_vectors[action]
                    context_score = context_vector(patch, order, action, side)
                    team_component = team_vector(acting_team, patch, action)
                    roster_component = roster_vector(match_id, patch, roster_side)
                    scores = {
                        "global": global_score,
                        "context": context_score,
                        "team": _blend(((0.5, context_score), (0.5, team_component)), context_score),
                        "roster": _blend(((0.5, context_score), (0.5, roster_component)), context_score),
                        "combined": _blend(
                            ((0.30, context_score), (0.20, team_component), (0.50, roster_component))
                            if action == "pick"
                            else ((0.25, context_score), (0.65, team_component), (0.10, roster_component)),
                            context_score,
                        ),
                    }
                    used = set(previous)
                    prediction = {
                        "sample_id": sample_id,
                        "match_id": match_id,
                        "split": split,
                        "patch_id": patch,
                        "action_order": order,
                        "action_type": action,
                        "acting_side": side,
                        "target_hero_id": target,
                    }
                    if target not in hero_index:
                        unseen_targets[split] += 1
                    for method, method_scores in scores.items():
                        rank = _rank(target, used, method_scores, hero_ids, hero_index)
                        prediction[f"{method}_rank"] = rank
                        record(method, split, action, order, patch, rank)
                    prediction_buffer.append(prediction)
                    if len(prediction_buffer) >= batch_size:
                        writer.write_table(pa.Table.from_pylist(prediction_buffer, schema=PREDICTION_SCHEMA))
                        progress.update(len(prediction_buffer))
                        prediction_buffer.clear()
            if prediction_buffer:
                writer.write_table(pa.Table.from_pylist(prediction_buffer, schema=PREDICTION_SCHEMA))
                progress.update(len(prediction_buffer))
    except Exception:
        writer.close()
        predictions_temp.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        predictions_temp.replace(predictions_path)
    finally:
        connection.close()

    metric_rows = []
    for (method, split, scope, value), totals in sorted(metric_totals.items()):
        count, top1, top3, top5, reciprocal_rank = totals
        metric_rows.append({
            "method": method,
            "split": split,
            "scope": scope,
            "scope_value": value,
            "sample_count": int(count),
            "top1_accuracy": top1 / count,
            "top3_accuracy": top3 / count,
            "top5_accuracy": top5 / count,
            "mrr": reciprocal_rank / count,
        })
    metrics_path = output_dir / "statistical_baseline_metrics.parquet"
    metrics_temp = metrics_path.with_suffix(metrics_path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(metric_rows), metrics_temp, compression="zstd")
    metrics_temp.replace(metrics_path)

    overall = {
        split: {
            method: next({
                "top1_accuracy": row["top1_accuracy"],
                "top3_accuracy": row["top3_accuracy"],
                "top5_accuracy": row["top5_accuracy"],
                "mrr": row["mrr"],
            } for row in metric_rows if row["method"] == method and row["split"] == split and row["scope"] == "overall")
            for method in METHODS
        }
        for split in ("validation", "test")
    }
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "training_split": "train only",
        "candidate_heroes": len(hero_ids),
        "unseen_targets": dict(unseen_targets),
        "methods": {
            "global": "pick/ban global popularity",
            "context": "global plus action phase, with patch-specific frequency when available",
            "team": "context plus acting team's historical pick/ban preference",
            "roster": "context plus acting roster for picks or opponent roster for bans",
            "combined": "validation-selected action-specific mixture: roster-heavy picks and team-heavy bans",
        },
        "overall_metrics": overall,
    }
    _write_json_atomic(output_dir / "statistical_baseline_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate statistical Dota 2 draft baselines")
    parser.add_argument("--samples", type=Path, default=Path("data/processed/model/draft_samples.parquet"))
    parser.add_argument("--rosters", type=Path, default=Path("data/processed/model/match_rosters.parquet"))
    parser.add_argument("--players", type=Path, default=Path("data/processed/test/match_players.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/baselines"))
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(evaluate_baselines(args.samples, args.rosters, args.players, args.output_dir, args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
