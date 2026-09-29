from __future__ import annotations

import argparse
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pyarrow.parquet as pq

from src.draft_mlp import DraftContextMLP, DraftDataset, build_vocabulary


def load_role_probabilities(path: Path, hero_ids: tuple[int, ...]) -> torch.Tensor:
    columns = ["hero_id"] + [f"position_{position}_probability" for position in range(1, 6)]
    table = pq.read_table(path, columns=columns)
    rows = {row["hero_id"]: row for row in table.to_pylist()}
    return torch.tensor([
        [rows[hero_id][column] for column in columns[1:]] if hero_id in rows else [0.2] * 5
        for hero_id in hero_ids
    ], dtype=torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pick_result_weights(
    actions: torch.Tensor,
    outcome_targets: torch.Tensor,
    winning_pick_weight: float,
    losing_pick_weight: float,
) -> torch.Tensor:
    pick_weights = torch.where(
        outcome_targets.bool(), winning_pick_weight, losing_pick_weight,
    )
    return torch.where(actions.bool(), pick_weights, torch.ones_like(outcome_targets))


def run_epoch(
    model: DraftContextMLP,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    accumulation_steps: int = 1,
    outcome_auxiliary: bool = False,
    outcome_loss_weight: float = 0.2,
    role_auxiliary: bool = False,
    role_loss_weight: float = 0.2,
    outcome_weighted_pick_loss: bool = False,
    winning_pick_weight: float = 1.5,
    losing_pick_weight: float = 0.5,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_function = nn.CrossEntropyLoss(reduction="none")
    outcome_loss_function = nn.BCEWithLogitsLoss(reduction="none")
    role_loss_function = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
    totals = {"loss": 0.0, "loss_weight": 0.0, "count": 0, "top1": 0, "top3": 0, "top5": 0, "rr": 0.0}
    outcome_loss_sum = 0.0
    outcome_correct = 0
    role_loss_sum = 0.0
    role_correct = 0
    role_count = 0
    rank_groups = {
        "pick": [0, 0], "ban": [0, 0],
        "winning_pick": [0, 0], "losing_pick": [0, 0],
    }
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        if training:
            optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(
            tqdm(loader, desc="train" if training else "validation", unit="batch", leave=False)
        ):
            items = [
                item.to(device) for item in batch
            ]
            history_heroes, history_categories, patches, orders, sides, actions, teams, rosters, targets, weights = items[:10]
            label_index = 10
            has_outcome_label = outcome_auxiliary or outcome_weighted_pick_loss
            outcome_targets = items[label_index] if has_outcome_label else None
            label_index += int(has_outcome_label)
            role_targets = items[label_index] if role_auxiliary else None
            if outcome_auxiliary or role_auxiliary:
                logits, _, auxiliary = model(
                    history_heroes, history_categories, patches, orders, sides, actions, teams, rosters,
                    return_auxiliary=True,
                )
            else:
                logits, _ = model(history_heroes, history_categories, patches, orders, sides, actions, teams, rosters)
            losses = loss_function(logits, targets)
            optimization_weights = weights
            if training and outcome_weighted_pick_loss:
                result_weights = pick_result_weights(
                    actions, outcome_targets, winning_pick_weight, losing_pick_weight,
                )
                optimization_weights = weights * result_weights
            loss = (losses * optimization_weights).sum() / optimization_weights.sum()
            if outcome_auxiliary:
                outcome_logits = auxiliary["outcome"]
                outcome_losses = outcome_loss_function(outcome_logits, outcome_targets)
                loss = loss + outcome_loss_weight * (outcome_losses * weights).sum() / weights.sum()
            if role_auxiliary:
                role_logits = auxiliary["role"]
                role_losses = role_loss_function(role_logits, role_targets)
                role_mask = role_targets >= 0
                role_weight = weights * role_mask
                loss = loss + role_loss_weight * (role_losses * role_weight).sum() / role_weight.sum().clamp_min(1)
            if training:
                group_start = (batch_index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(loader) - group_start)
                (loss / group_size).backward()
                if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = targets.shape[0]
            target_scores = logits.gather(1, targets.unsqueeze(1))
            ranks = 1 + (logits > target_scores).sum(dim=1)
            totals["loss"] += (losses * weights).sum().item()
            totals["loss_weight"] += weights.sum().item()
            totals["count"] += batch_size
            totals["top1"] += (ranks <= 1).sum().item()
            totals["top3"] += (ranks <= 3).sum().item()
            totals["top5"] += (ranks <= 5).sum().item()
            totals["rr"] += (1.0 / ranks.float()).sum().item()
            top5 = ranks <= 5
            pick_mask = actions.bool()
            for name, mask in (("pick", pick_mask), ("ban", ~pick_mask)):
                rank_groups[name][0] += mask.sum().item()
                rank_groups[name][1] += top5[mask].sum().item()
            if has_outcome_label:
                for name, mask in (
                    ("winning_pick", pick_mask & outcome_targets.bool()),
                    ("losing_pick", pick_mask & ~outcome_targets.bool()),
                ):
                    rank_groups[name][0] += mask.sum().item()
                    rank_groups[name][1] += top5[mask].sum().item()
            if outcome_auxiliary:
                outcome_loss_sum += outcome_losses.sum().item()
                outcome_correct += ((outcome_logits >= 0) == outcome_targets.bool()).sum().item()
            if role_auxiliary:
                role_loss_sum += role_losses[role_mask].sum().item()
                role_correct += (role_logits.argmax(1)[role_mask] == role_targets[role_mask]).sum().item()
                role_count += role_mask.sum().item()
    count = totals["count"]
    metrics = {
        "loss": totals["loss"] / totals["loss_weight"],
        "top1_accuracy": totals["top1"] / count,
        "top3_accuracy": totals["top3"] / count,
        "top5_accuracy": totals["top5"] / count,
        "mrr": totals["rr"] / count,
    }
    for name, (group_count, group_top5) in rank_groups.items():
        if group_count:
            metrics[f"{name}_top5_accuracy"] = group_top5 / group_count
            metrics[f"{name}_samples"] = group_count
    if outcome_auxiliary:
        metrics["outcome_loss"] = outcome_loss_sum / count
        metrics["outcome_accuracy"] = outcome_correct / count
    if role_auxiliary:
        metrics["role_loss"] = role_loss_sum / role_count
        metrics["role_accuracy"] = role_correct / role_count
        metrics["role_samples"] = role_count
    return metrics


def train_model(
    samples_path: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    patience: int = 3,
    seed: int = 42,
    device_name: str = "auto",
    model_variant: str = "a",
    rosters_path: Path = Path("data/processed/model/match_rosters.parquet"),
    scheduler_patience: int = 2,
    split_action_heads: bool = False,
    lineup_encoder: bool = False,
    recency_half_life_days: float | None = None,
    lineup_attention: bool = False,
    role_coverage: bool = False,
    role_priors_path: Path = Path("data/processed/model/hero_role_priors.parquet"),
    candidate_scorer: bool = False,
    sequence_encoder: bool = False,
    accumulation_steps: int = 1,
    outcome_auxiliary: bool = False,
    outcome_loss_weight: float = 0.2,
    role_auxiliary: bool = False,
    role_loss_weight: float = 0.2,
    outcome_weighted_pick_loss: bool = False,
    winning_pick_weight: float = 1.5,
    losing_pick_weight: float = 0.5,
) -> dict[str, Any]:
    if not samples_path.is_file():
        raise FileNotFoundError(samples_path)
    if epochs <= 0 or batch_size <= 0 or patience <= 0 or scheduler_patience <= 0 or accumulation_steps <= 0:
        raise ValueError("epochs, batch_size, patience, scheduler_patience, and accumulation_steps must be positive")
    if outcome_loss_weight < 0 or role_loss_weight < 0:
        raise ValueError("auxiliary loss weights must be non-negative")
    if winning_pick_weight <= 0 or losing_pick_weight <= 0:
        raise ValueError("pick result weights must be positive")
    set_seed(seed)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    include_teams = model_variant in {"b", "c"}
    include_players = model_variant == "c"
    if include_players and not rosters_path.is_file():
        raise FileNotFoundError(rosters_path)
    if role_coverage and not role_priors_path.is_file():
        raise FileNotFoundError(role_priors_path)
    vocabulary = build_vocabulary(samples_path, rosters_path, include_teams, include_players)
    print("Encoding train split...")
    train_dataset = DraftDataset(
        samples_path, "train", vocabulary, rosters_path if include_players else None,
        recency_half_life_days,
        outcome_auxiliary or outcome_weighted_pick_loss,
        role_auxiliary,
    )
    print("Encoding validation split...")
    validation_dataset = DraftDataset(
        samples_path, "validation", vocabulary, rosters_path if include_players else None,
        include_outcome_label=outcome_auxiliary or outcome_weighted_pick_loss,
        include_role_label=role_auxiliary,
    )
    max_order = max(train_dataset.action_orders.max().item(), validation_dataset.action_orders.max().item())

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = DraftContextMLP(
        len(vocabulary.hero_ids), len(vocabulary.patch_ids), max_order,
        team_count=len(vocabulary.team_ids), player_count=len(vocabulary.player_ids),
        split_action_heads=split_action_heads,
        lineup_encoder=lineup_encoder,
        lineup_attention=lineup_attention,
        role_coverage=role_coverage,
        candidate_scorer=candidate_scorer,
        sequence_encoder=sequence_encoder,
        outcome_auxiliary=outcome_auxiliary,
        role_auxiliary=role_auxiliary,
    ).to(device)
    if role_coverage:
        model.set_hero_role_probabilities(load_role_probabilities(role_priors_path, vocabulary.hero_ids).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=scheduler_patience, min_lr=1e-5
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"model_{model_variant}"
    checkpoint_path = output_dir / f"{model_name}_best.pt"
    checkpoint_temp = checkpoint_path.with_suffix(checkpoint_path.suffix + f".{os.getpid()}.tmp")
    history: list[dict[str, Any]] = []
    best_top5 = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer, accumulation_steps,
            outcome_auxiliary, outcome_loss_weight,
            role_auxiliary, role_loss_weight,
            outcome_weighted_pick_loss, winning_pick_weight, losing_pick_weight,
        )
        validation_metrics = run_epoch(
            model, validation_loader, device,
            outcome_auxiliary=outcome_auxiliary, outcome_loss_weight=outcome_loss_weight,
            role_auxiliary=role_auxiliary, role_loss_weight=role_loss_weight,
            outcome_weighted_pick_loss=outcome_weighted_pick_loss,
            winning_pick_weight=winning_pick_weight, losing_pick_weight=losing_pick_weight,
        )
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
                "model_config": {
                    "hero_count": len(vocabulary.hero_ids),
                    "patch_count": len(vocabulary.patch_ids),
                    "max_action_order": max_order,
                    "team_count": len(vocabulary.team_ids),
                    "player_count": len(vocabulary.player_ids),
                    "hidden_sizes": [256, 128],
                    "dropout": 0.2,
                    "split_action_heads": split_action_heads,
                    "lineup_encoder": lineup_encoder,
                    "lineup_attention": lineup_attention,
                    "role_coverage": role_coverage,
                    "candidate_scorer": candidate_scorer,
                    "sequence_encoder": sequence_encoder,
                    "outcome_auxiliary": outcome_auxiliary,
                    "role_auxiliary": role_auxiliary,
                },
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
                    "split_action_heads": split_action_heads,
                    "lineup_encoder": lineup_encoder,
                    "recency_half_life_days": recency_half_life_days,
                    "lineup_attention": lineup_attention,
                    "role_coverage": role_coverage,
                    "role_priors_path": str(role_priors_path.resolve()) if role_coverage else None,
                    "candidate_scorer": candidate_scorer,
                    "sequence_encoder": sequence_encoder,
                    "outcome_auxiliary": outcome_auxiliary,
                    "outcome_loss_weight": outcome_loss_weight,
                    "role_auxiliary": role_auxiliary,
                    "role_loss_weight": role_loss_weight,
                    "outcome_weighted_pick_loss": outcome_weighted_pick_loss,
                    "winning_pick_weight": winning_pick_weight,
                    "losing_pick_weight": losing_pick_weight,
                },
            }, checkpoint_temp)
            checkpoint_temp.replace(checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model_description = {
        "a": "Model A: BP state + patch/action embeddings + MLP",
        "b": "Model B: Model A + acting/opponent team embeddings",
        "c": "Model C: Model B + acting/opponent roster embeddings",
    }[model_variant]
    if lineup_encoder:
        model_description += " + DeepSets lineup encoder"
    if lineup_attention:
        model_description += " + Set Self-Attention lineup encoder"
    if split_action_heads:
        model_description += " + separate pick/ban heads"
    if role_coverage:
        model_description += " + soft hero-position coverage"
    if candidate_scorer:
        model_description += " + candidate-conditioned hero scorer"
    if sequence_encoder:
        model_description += " + ordered BP Transformer encoder"
    if outcome_auxiliary:
        model_description += " + acting-side win auxiliary task"
    if role_auxiliary:
        model_description += " + picked-hero position auxiliary task"
    if outcome_weighted_pick_loss:
        model_description += " + outcome-weighted pick loss"
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": model_description,
        "device": str(device),
        "seed": seed,
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
        "training_players": len(vocabulary.player_ids),
        "training_patches": list(vocabulary.patch_ids),
        "split_action_heads": split_action_heads,
        "lineup_encoder": lineup_encoder,
        "lineup_attention": lineup_attention,
        "recency_half_life_days": recency_half_life_days,
        "role_coverage": role_coverage,
        "role_priors_path": str(role_priors_path.resolve()) if role_coverage else None,
        "candidate_scorer": candidate_scorer,
        "sequence_encoder": sequence_encoder,
        "outcome_auxiliary": outcome_auxiliary,
        "outcome_loss_weight": outcome_loss_weight if outcome_auxiliary else None,
        "role_auxiliary": role_auxiliary,
        "role_loss_weight": role_loss_weight if role_auxiliary else None,
        "outcome_weighted_pick_loss": outcome_weighted_pick_loss,
        "winning_pick_weight": winning_pick_weight if outcome_weighted_pick_loss else None,
        "losing_pick_weight": losing_pick_weight if outcome_weighted_pick_loss else None,
        "training_weight_summary": {
            "minimum": train_dataset.sample_weights.min().item(),
            "maximum": train_dataset.sample_weights.max().item(),
            "mean": train_dataset.sample_weights.mean().item(),
            "effective_sample_size": (
                train_dataset.sample_weights.sum().item() ** 2
                / train_dataset.sample_weights.square().sum().item()
            ),
        },
        "best_epoch": best_epoch,
        "best_validation_metrics": next(row["validation"] for row in history if row["epoch"] == best_epoch),
        "checkpoint": str(checkpoint_path.resolve()),
        "history": history,
    }
    summary_path = output_dir / f"{model_name}_training_summary.json"
    summary_temp = summary_path.with_suffix(summary_path.suffix + f".{os.getpid()}.tmp")
    summary_temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_temp.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dota 2 draft MLP variants A, B, or C")
    parser.add_argument("--samples", type=Path, default=Path("data/processed/model/draft_samples.parquet"))
    parser.add_argument("--rosters", type=Path, default=Path("data/processed/model/match_rosters.parquet"))
    parser.add_argument("--model", dest="model_variant", choices=("a", "b", "c"), default="a")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument(
        "--split-action-heads", action="store_true",
        help="Use separate hero classifiers for pick and ban actions",
    )
    lineup_group = parser.add_mutually_exclusive_group()
    lineup_group.add_argument(
        "--lineup-encoder", action="store_true",
        help="Add DeepSets encodings of acting and opponent picked-hero lineups",
    )
    lineup_group.add_argument(
        "--lineup-attention", action="store_true",
        help="Encode acting and opponent picked heroes with Set Self-Attention",
    )
    parser.add_argument(
        "--recency-half-life-days", type=float,
        help="Exponentially down-weight older train samples by this half-life",
    )
    parser.add_argument("--role-coverage", action="store_true", help="Add soft position coverage from train-only hero role priors")
    parser.add_argument("--role-priors", type=Path, default=Path("data/processed/model/hero_role_priors.parquet"))
    parser.add_argument(
        "--candidate-scorer", action="store_true",
        help="Score each candidate hero by matching its embedding to the draft context",
    )
    parser.add_argument(
        "--sequence-encoder", action="store_true",
        help="Add an ordered Transformer encoding of all previous draft actions",
    )
    parser.add_argument(
        "--outcome-auxiliary", action="store_true",
        help="Jointly predict whether the acting side wins as an auxiliary task",
    )
    parser.add_argument("--outcome-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--role-auxiliary", action="store_true",
        help="Predict the selected hero's inferred position for pick rows",
    )
    parser.add_argument("--role-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--outcome-weighted-pick-loss", action="store_true",
        help="Up-weight picks from wins and down-weight picks from losses during training",
    )
    parser.add_argument("--winning-pick-weight", type=float, default=1.5)
    parser.add_argument("--losing-pick-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    output_dir = args.output_dir or Path(f"data/processed/neural/model_{args.model_variant}")
    summary = train_model(
        args.samples, output_dir, args.epochs, args.batch_size, args.learning_rate,
        args.patience, args.seed, args.device, args.model_variant, args.rosters, args.scheduler_patience,
        args.split_action_heads,
        args.lineup_encoder,
        args.recency_half_life_days,
        args.lineup_attention,
        args.role_coverage,
        args.role_priors,
        args.candidate_scorer,
        args.sequence_encoder,
        args.accumulation_steps,
        args.outcome_auxiliary,
        args.outcome_loss_weight,
        args.role_auxiliary,
        args.role_loss_weight,
        args.outcome_weighted_pick_loss,
        args.winning_pick_weight,
        args.losing_pick_weight,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
