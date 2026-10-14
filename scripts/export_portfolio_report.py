from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUNS = (
    "decay_current_none", "decay_current_60", "decay_current_120", "decay_current_240",
    "model_b_plus_recency60_seed7", "model_b_plus_recency60_seed2026",
    "model_b_plus_recency60_sequence", "model_b_plus_recency60_sequence_seed7",
    "model_b_plus_recency60_sequence_seed2026",
    "model_b_plus_recency60_sequence_multitask",
    "model_b_plus_recency60_sequence_role_multitask", "model_b_plus_outcome_weighted",
    "draft_value",
)
PUBLIC_FIELDS = (
    "generated_at_utc", "model", "seed", "device", "train_samples", "validation_samples",
    "training_patches", "batch_size", "accumulation_steps", "effective_batch_size",
    "initial_learning_rate", "scheduler_patience", "early_stopping_patience",
    "recency_half_life_days", "sequence_encoder", "lineup_encoder", "lineup_attention",
    "role_coverage", "outcome_auxiliary", "outcome_loss_weight", "role_auxiliary",
    "role_loss_weight", "outcome_weighted_pick_loss", "winning_pick_weight",
    "losing_pick_weight", "best_epoch", "best_validation_metrics", "validation_constant_baseline",
)


def read_source(root: Path, relative: str) -> tuple[dict, dict]:
    path = root / relative
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), {
        "path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
    }


def build_report(root: Path) -> dict:
    profile, profile_source = read_source(root, "profile/dataset_profile.json")
    splits, splits_source = read_source(root, "model/dataset_splits_summary.json")
    samples, samples_source = read_source(root, "model/draft_samples_summary.json")
    runs = []
    for name in RUNS:
        filename = "draft_value_training_summary.json" if name == "draft_value" else "model_b_training_summary.json"
        summary, source = read_source(root, f"neural/{name}/{filename}")
        expected = (
            {"train": splits["splits"]["train"]["matches"] * 10,
             "validation": splits["splits"]["validation"]["matches"] * 10}
            if name == "draft_value" else samples["samples_by_split"]
        )
        for split in ("train", "validation"):
            if summary[f"{split}_samples"] != expected[split]:
                raise ValueError(f"Sample count mismatch for {name}: {split}")
        runs.append({
            "run": name, "source": source,
            **{key: summary[key] for key in PUBLIC_FIELDS if key in summary},
        })
    return {
        "schema_version": 1,
        "evaluation_split": "validation",
        "provenance_note": "Source paths are relative to data/processed. Hashes identify summary files, not training datasets. Matching counts do not prove identical data.",
        "comparison_note": "Historical statistical baseline and old model_comparison.json are excluded: their data snapshot predates this cohort. Seed 42 original B+ did not record physical batch size; later runs used 512 x 4 accumulation. These are exploratory comparisons, not a fully controlled multi-seed benchmark.",
        "dataset": {
            "sources": [profile_source, splits_source, samples_source],
            "raw_matches": profile["total_matches"],
            "eligible_matches": profile["eligible_matches"],
            "excluded_matches": profile["excluded_matches"],
            "earliest_match_utc": profile["earliest_match_utc"],
            "latest_match_utc": profile["latest_match_utc"],
            "match_splits": splits["splits"],
            "samples": samples["samples_written"],
            "sample_splits": samples["samples_by_split"],
        },
        "runs": runs,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Recorded Validation Results", "",
        "Generated with `python -m scripts.export_portfolio_report`. See [results.json](results.json) for configuration and source-summary hashes.", "",
        report["comparison_note"], "",
        "These are validation model-selection results, not a final test benchmark. A single-seed maximum is not an estimate of typical performance.", "",
        "| Run | Seed | Best epoch | Top-5 (%) | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        metrics = run["best_validation_metrics"]
        if "top5_accuracy" in metrics:
            lines.append(f"| {run['run']} | {run['seed']} | {run['best_epoch']} | {100 * metrics['top5_accuracy']:.4f} | {metrics['mrr']:.4f} |")
    value = next(run for run in report["runs"] if run["run"] == "draft_value")
    metrics = value["best_validation_metrics"]
    lines.extend([
        "", "## Value Model", "",
        f"Best epoch {value['best_epoch']}: AUC {metrics['auc']:.4f}, log loss {metrics['log_loss']:.5f}, Brier {metrics['brier_score']:.5f}.",
        "Evaluated on actual historical Picks. This does not establish the benefit of choosing a different hero.", "",
        "## Interpretation", "",
        "- B+ with 60-day decay is the current behavior-model candidate, pending final test evaluation.",
        "- Ordered Transformer results vary by seed; there is no established improvement in the mean.",
        "- Outcome and position auxiliary tasks have only single-seed evidence in this cohort.",
        "- Outcome-weighted imitation is not a win-rate estimator.",
        "- Value-model validation loss worsened after epoch 1. Candidate contribution has not been isolated from team and draft-context effects.",
        "- Rows from the same match are dependent. Future uncertainty estimates should resample matches, not individual actions.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a public, path-sanitized snapshot of recorded experiments")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    report = build_report(args.processed_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "results.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Exported {len(report['runs'])} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
