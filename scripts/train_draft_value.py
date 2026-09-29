from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.draft_mlp import build_vocabulary
from src.draft_value import DraftValueDataset, DraftValueModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def binary_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def run_epoch(
    model: DraftValueModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    accumulation_steps: int = 1,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    loss_sum = 0.0
    weight_sum = 0.0
    count = 0
    correct = 0
    brier_sum = 0.0
    labels: list[int] = []
    scores: list[float] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        if training:
            optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(tqdm(
            loader, desc="train value" if training else "validation value", unit="batch", leave=False,
        )):
            items = [item.to(device) for item in batch]
            histories, categories, patches, orders, sides, _, teams, _, candidates, weights, outcomes = items
            logits = model(histories, categories, patches, orders, sides, teams, candidates)
            losses = loss_function(logits, outcomes)
            loss = (losses * weights).sum() / weights.sum()
            if training:
                group_start = (batch_index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(loader) - group_start)
                (loss / group_size).backward()
                if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            probabilities = torch.sigmoid(logits)
            loss_sum += (losses * weights).sum().item()
            weight_sum += weights.sum().item()
            count += outcomes.numel()
            correct += ((probabilities >= 0.5) == outcomes.bool()).sum().item()
            brier_sum += ((probabilities - outcomes) ** 2).sum().item()
            labels.extend(outcomes.int().cpu().tolist())
            scores.extend(probabilities.cpu().tolist())
    return {
        "log_loss": loss_sum / weight_sum,
        "accuracy": correct / count,
        "auc": binary_auc(labels, scores),
        "brier_score": brier_sum / count,
        "samples": count,
        "positive_rate": sum(labels) / count,
    }


def train_value_model(
    samples_path: Path,
    output_dir: Path,
    epochs: int = 40,
    batch_size: int = 512,
    accumulation_steps: int = 4,
    learning_rate: float = 1e-3,
    patience: int = 5,
    scheduler_patience: int = 2,
    recency_half_life_days: float | None = 60,
    seed: int = 42,
    device_name: str = "auto",
) -> dict[str, Any]:
    if not samples_path.is_file():
        raise FileNotFoundError(samples_path)
    if min(epochs, batch_size, accumulation_steps, patience, scheduler_patience) <= 0:
        raise ValueError("training counts and patience values must be positive")
    set_seed(seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    vocabulary = build_vocabulary(samples_path, include_teams=True)
    print("Encoding factual train Picks...")
    train_dataset = DraftValueDataset(
        samples_path, "train", vocabulary, recency_half_life_days,
    )
    print("Encoding factual validation Picks...")
    validation_dataset = DraftValueDataset(samples_path, "validation", vocabulary)
    max_order = max(
        train_dataset.samples.action_orders.max().item(),
        validation_dataset.samples.action_orders.max().item(),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = DraftValueModel(
        len(vocabulary.hero_ids), len(vocabulary.patch_ids), max_order, len(vocabulary.team_ids),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience, min_lr=1e-5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "draft_value_best.pt"
    checkpoint_temp = checkpoint_path.with_suffix(checkpoint_path.suffix + f".{os.getpid()}.tmp")
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, accumulation_steps)
        validation_metrics = run_epoch(model, validation_loader, device)
        current_learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(validation_metrics["log_loss"])
        row = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        print(json.dumps(row))
        if validation_metrics["log_loss"] < best_loss:
            best_loss = validation_metrics["log_loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocabulary": vocabulary.to_dict(),
                "model_config": {
                    "hero_count": len(vocabulary.hero_ids),
                    "patch_count": len(vocabulary.patch_ids),
                    "max_action_order": max_order,
                    "team_count": len(vocabulary.team_ids),
                    "hidden_sizes": [256, 128],
                    "dropout": 0.2,
                },
                "epoch": epoch,
                "validation_metrics": validation_metrics,
            }, checkpoint_temp)
            checkpoint_temp.replace(checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": "Factual Pick value model: B+ context + candidate hero -> acting-side win probability",
        "limitation": "Validation measures factual outcome prediction, not counterfactual recommendation quality.",
        "device": str(device),
        "seed": seed,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "recency_half_life_days": recency_half_life_days,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "early_stopping_metric": "validation log_loss",
        "best_epoch": best_epoch,
        "best_validation_metrics": next(row["validation"] for row in history if row["epoch"] == best_epoch),
        "validation_constant_baseline": {
            "prediction": validation_dataset.samples.outcome_targets[validation_dataset.indices].mean().item(),
            "auc": 0.5,
            "log_loss": -sum(
                probability * math.log(max(probability, 1e-12))
                for probability in (
                    validation_dataset.samples.outcome_targets[validation_dataset.indices].mean().item(),
                    1 - validation_dataset.samples.outcome_targets[validation_dataset.indices].mean().item(),
                )
            ),
            "brier_score": (
                validation_dataset.samples.outcome_targets[validation_dataset.indices].mean().item()
                * (1 - validation_dataset.samples.outcome_targets[validation_dataset.indices].mean().item())
            ),
        },
        "checkpoint": str(checkpoint_path.resolve()),
        "history": history,
    }
    summary_path = output_dir / "draft_value_training_summary.json"
    summary_temp = summary_path.with_suffix(summary_path.suffix + f".{os.getpid()}.tmp")
    summary_temp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_temp.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a factual Dota 2 Pick value model")
    parser.add_argument("--samples", type=Path, default=Path("data/processed/model/draft_samples.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/neural/draft_value"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--recency-half-life-days", type=float, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    summary = train_value_model(
        args.samples, args.output_dir, args.epochs, args.batch_size, args.accumulation_steps,
        args.learning_rate, args.patience, args.scheduler_patience,
        args.recency_half_life_days, args.seed, args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
