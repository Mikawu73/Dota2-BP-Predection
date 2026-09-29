from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from scripts.train_draft_mlp import run_epoch, set_seed
from src.draft_mlp import DraftDataset, build_vocabulary
from src.draft_transformer import DraftTransformer


def train_model_d(
    samples_path: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 2048,
    accumulation_steps: int = 1,
    learning_rate: float = 1e-3,
    patience: int = 5,
    scheduler_patience: int = 2,
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
    if device.type == "cpu":
        torch.set_num_threads(8)
    vocabulary = build_vocabulary(samples_path, include_teams=True)
    print("Encoding train split...")
    train_dataset = DraftDataset(samples_path, "train", vocabulary)
    print("Encoding validation split...")
    validation_dataset = DraftDataset(samples_path, "validation", vocabulary)
    max_order = max(train_dataset.action_orders.max().item(), validation_dataset.action_orders.max().item())
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model_config = {
        "hero_count": len(vocabulary.hero_ids),
        "patch_count": len(vocabulary.patch_ids),
        "max_action_order": max_order,
        "team_count": len(vocabulary.team_ids),
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 128,
        "dropout": 0.2,
    }
    model = DraftTransformer(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=scheduler_patience, min_lr=1e-5
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_d_best.pt"
    checkpoint_temp = checkpoint_path.with_suffix(checkpoint_path.suffix + f".{os.getpid()}.tmp")
    history: list[dict[str, Any]] = []
    best_top5 = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, accumulation_steps)
        validation_metrics = run_epoch(model, validation_loader, device)
        current_learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(validation_metrics["top5_accuracy"])
        row = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if validation_metrics["top5_accuracy"] > best_top5:
            best_top5 = validation_metrics["top5_accuracy"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocabulary": vocabulary.to_dict(),
                "model_config": model_config,
                "epoch": epoch,
                "validation_metrics": validation_metrics,
                "training_config": {
                    "batch_size": batch_size,
                    "accumulation_steps": accumulation_steps,
                    "effective_batch_size": batch_size * accumulation_steps,
                    "initial_learning_rate": learning_rate,
                    "scheduler_patience": scheduler_patience,
                    "early_stopping_patience": patience,
                    "seed": seed,
                },
            }, checkpoint_temp)
            checkpoint_temp.replace(checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": "Model D: BP action sequence + team/context Transformer",
        "device": str(device),
        "seed": seed,
        "model_config": model_config,
        "initial_learning_rate": learning_rate,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "scheduler_patience": scheduler_patience,
        "early_stopping_patience": patience,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "candidate_heroes": len(vocabulary.hero_ids),
        "training_teams": len(vocabulary.team_ids),
        "training_patches": list(vocabulary.patch_ids),
        "best_epoch": best_epoch,
        "best_validation_metrics": next(row["validation"] for row in history if row["epoch"] == best_epoch),
        "checkpoint": str(checkpoint_path.resolve()),
        "history": history,
    }
    summary_path = output_dir / "model_d_training_summary.json"
    summary_temp = summary_path.with_suffix(summary_path.suffix + f".{os.getpid()}.tmp")
    summary_temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_temp.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model D draft Transformer")
    parser.add_argument("--samples", type=Path, default=Path("data/processed/model/draft_samples.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/neural/model_d"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    summary = train_model_d(
        args.samples, args.output_dir, args.epochs, args.batch_size, args.accumulation_steps, args.learning_rate,
        args.patience, args.scheduler_patience, args.seed, args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
