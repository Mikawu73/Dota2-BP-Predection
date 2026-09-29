from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.draft_mlp import DraftContextMLP, DraftDataset, DraftVocabulary


def evaluate_by_patch(checkpoint_path: Path, samples_path: Path, split: str, batch_size: int) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocabulary = DraftVocabulary(**checkpoint["vocabulary"])
    dataset = DraftDataset(samples_path, split, vocabulary)
    model = DraftContextMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "top1": 0, "top3": 0, "top5": 0, "rr": 0.0}
    )
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            history, categories, patches, orders, sides, actions, teams, rosters, targets, _ = batch
            logits, _ = model(history, categories, patches, orders, sides, actions, teams, rosters)
            target_scores = logits.gather(1, targets.unsqueeze(1))
            ranks = 1 + (logits > target_scores).sum(dim=1)
            batch_start = sum(group["count"] for group in totals.values())
            raw_patches = dataset.raw_patch_ids[batch_start:batch_start + len(patches)]
            for raw_patch in raw_patches.unique():
                mask = raw_patches == raw_patch
                key = str(raw_patch.item())
                group = totals[key]
                group["count"] += mask.sum().item()
                group["top1"] += (ranks[mask] <= 1).sum().item()
                group["top3"] += (ranks[mask] <= 3).sum().item()
                group["top5"] += (ranks[mask] <= 5).sum().item()
                group["rr"] += (1.0 / ranks[mask].float()).sum().item()
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "split": split,
        "training_patch_ids": list(vocabulary.patch_ids),
        "metrics_by_patch": {
            patch: {
                "samples": int(values["count"]),
                "top1_accuracy": values["top1"] / values["count"],
                "top3_accuracy": values["top3"] / values["count"],
                "top5_accuracy": values["top5"] / values["count"],
                "mrr": values["rr"] / values["count"],
            }
            for patch, values in totals.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an MLP checkpoint separately by patch")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=Path("data/processed/model/draft_samples.parquet"))
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(evaluate_by_patch(args.checkpoint, args.samples, args.split, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
