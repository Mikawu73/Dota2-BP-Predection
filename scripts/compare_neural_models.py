from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def compare_models(neural_dir: Path, baseline_summary: Path, output_path: Path) -> dict[str, object]:
    models = {}
    candidates = {
        "model_a": neural_dir / "model_a" / "model_a_training_summary.json",
        "model_b": neural_dir / "model_b" / "model_b_training_summary.json",
        "model_c": neural_dir / "model_c" / "model_c_training_summary.json",
        "model_b_tuned_1": neural_dir / "model_b_tuned_1" / "model_b_training_summary.json",
        "model_b_split_heads": neural_dir / "model_b_split_heads" / "model_b_training_summary.json",
        "model_b_plus": neural_dir / "model_b_plus" / "model_b_training_summary.json",
        "model_b_plus_recency120": neural_dir / "model_b_plus_recency120" / "model_b_training_summary.json",
        "model_b_plus_recency120_attention": neural_dir / "model_b_plus_recency120_attention" / "model_b_training_summary.json",
        "model_b_plus_recency120_roles": neural_dir / "model_b_plus_recency120_roles" / "model_b_training_summary.json",
        "model_d": neural_dir / "model_d" / "model_d_training_summary.json",
    }
    for name, path in candidates.items():
        if name in {
            "model_b_tuned_1", "model_b_split_heads", "model_b_plus",
            "model_b_plus_recency120", "model_d",
            "model_b_plus_recency120_attention",
            "model_b_plus_recency120_roles",
        } and not path.is_file():
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        models[name] = {
            "description": summary["model"],
            "best_epoch": summary["best_epoch"],
            **summary["best_validation_metrics"],
        }
    baseline = json.loads(baseline_summary.read_text(encoding="utf-8"))["overall_metrics"]["validation"]["combined"]
    best_model = max(models, key=lambda name: models[name]["top5_accuracy"])
    report: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_split": "validation",
        "statistical_combined": baseline,
        "models": models,
        "selected_model": best_model,
        "selected_model_top5_gain_over_statistical": models[best_model]["top5_accuracy"] - baseline["top5_accuracy"],
        "test_evaluated": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare validation metrics for draft MLP variants")
    parser.add_argument("--neural-dir", type=Path, default=Path("data/processed/neural"))
    parser.add_argument("--baseline-summary", type=Path, default=Path("data/processed/baselines/statistical_baseline_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/neural/model_comparison.json"))
    args = parser.parse_args()
    print(json.dumps(compare_models(args.neural_dir, args.baseline_summary, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
