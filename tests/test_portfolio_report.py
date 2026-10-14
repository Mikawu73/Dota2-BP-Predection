import json
from pathlib import Path

import pytest

from scripts.export_portfolio_report import RUNS, build_report, render_markdown


def write_sources(root: Path) -> None:
    sources = {
        "profile/dataset_profile.json": {
            "total_matches": 4, "eligible_matches": 3, "excluded_matches": 1,
            "earliest_match_utc": "2026-01-01", "latest_match_utc": "2026-01-03",
        },
        "model/dataset_splits_summary.json": {
            "splits": {split: {"matches": 1} for split in ("train", "validation", "test")},
        },
        "model/draft_samples_summary.json": {
            "samples_written": 72,
            "samples_by_split": {split: 24 for split in ("train", "validation", "test")},
        },
    }
    for name in RUNS:
        value = name == "draft_value"
        filename = "draft_value_training_summary.json" if value else "model_b_training_summary.json"
        sources[f"neural/{name}/{filename}"] = {
            "model": name, "seed": 42, "best_epoch": 1,
            "train_samples": 10 if value else 24,
            "validation_samples": 10 if value else 24,
            "checkpoint": "C:/private/user/model.pt",
            "unexpected_private_field": "DO_NOT_EXPORT",
            "best_validation_metrics": (
                {"auc": 0.55, "log_loss": 0.69, "brier_score": 0.25}
                if value else {"top5_accuracy": 0.31, "mrr": 0.22}
            ),
        }
    for relative, data in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")


def test_report_exports_only_public_fields_and_source_hashes(tmp_path):
    write_sources(tmp_path)
    report = build_report(tmp_path)
    serialized = json.dumps(report)
    assert "C:/private" not in serialized
    assert "DO_NOT_EXPORT" not in serialized
    assert len(report["runs"]) == len(RUNS)
    assert all(len(run["source"]["sha256"]) == 64 for run in report["runs"])
    assert "31.0000" in render_markdown(report)


def test_report_rejects_mismatched_sample_counts(tmp_path):
    write_sources(tmp_path)
    path = tmp_path / "neural/decay_current_60/model_b_training_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["validation_samples"] = 23
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Sample count mismatch"):
        build_report(tmp_path)
