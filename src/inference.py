from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from src.draft_mlp import DraftContextMLP, DraftVocabulary, history_category


DEFAULT_CHECKPOINT = Path("data/processed/neural/decay_current_60/model_b_best.pt")


def integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class DraftPredictor:
    """Single-state inference using the saved training vocabulary and configuration."""

    def __init__(self, checkpoint: Path):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = saved["model_config"]
        self.vocabulary = DraftVocabulary(**{
            key: tuple(value) for key, value in saved["vocabulary"].items()
        })
        for key, values in (
            ("hero_count", self.vocabulary.hero_ids),
            ("patch_count", self.vocabulary.patch_ids),
            ("team_count", self.vocabulary.team_ids),
            ("player_count", self.vocabulary.player_ids),
        ):
            if self.config.get(key, 0) != len(values) or len(set(values)) != len(values):
                raise ValueError(f"Invalid checkpoint vocabulary: {key}")
        if not self.vocabulary.hero_ids:
            raise ValueError("Checkpoint has no candidate heroes")
        if self.vocabulary.player_ids:
            raise ValueError("Player-roster checkpoints are not supported by this inference interface")
        self.model = DraftContextMLP(**self.config)
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        self.model.eval()
        self.hero_index = self.vocabulary.hero_to_index
        self.patch_index = self.vocabulary.patch_to_index
        self.team_index = self.vocabulary.team_to_index
        self.max_order = self.config["max_action_order"]
        self.checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.epoch = saved.get("epoch")

    def metadata(self) -> dict:
        return {
            "model": "DraftContextMLP",
            "checkpoint_sha256": self.checkpoint_hash,
            "epoch": self.epoch,
            "model_config": self.config,
            "training_patch_ids": list(self.vocabulary.patch_ids),
            "team_ids": list(self.vocabulary.team_ids),
            "hero_ids": list(self.vocabulary.hero_ids),
            "max_actions": self.max_order + 1,
            "task": "next_action_prediction_not_win_probability",
        }

    def encode(self, state: dict) -> tuple[list[torch.Tensor], list[str], int]:
        if not isinstance(state, dict):
            raise ValueError("Input must be a JSON object")
        allowed = {"patch_id", "radiant_team_id", "dire_team_id", "acting_side", "is_pick", "history", "top_k"}
        if set(state) - allowed:
            raise ValueError("Unexpected fields: " + ", ".join(sorted(set(state) - allowed)))
        patch = integer(state.get("patch_id"), "patch_id")
        side = state.get("acting_side")
        if side not in ("radiant", "dire"):
            raise ValueError("acting_side must be radiant or dire")
        if type(state.get("is_pick")) is not bool:
            raise ValueError("is_pick must be a boolean")
        top_k = integer(state.get("top_k", 5), "top_k")
        if top_k > 20:
            raise ValueError("top_k must be <= 20")
        history = state.get("history")
        if not isinstance(history, list) or len(history) > self.max_order + 1:
            raise ValueError(f"history must be a list of at most {self.max_order + 1} actions")
        complete = len(history) == self.max_order + 1
        heroes, categories = [], []
        pick_counts = {"radiant": 0, "dire": 0}
        for index, action in enumerate(history):
            if not isinstance(action, dict) or set(action) != {"hero_id", "side", "is_pick"}:
                raise ValueError(f"history[{index}] requires exactly hero_id, side, is_pick")
            hero = integer(action["hero_id"], f"history[{index}].hero_id")
            if hero not in self.hero_index:
                raise ValueError(f"Hero {hero} is absent from the checkpoint vocabulary")
            encoded = self.hero_index[hero]
            if encoded in heroes:
                raise ValueError(f"Hero {hero} has already been picked or banned")
            if action["side"] not in ("radiant", "dire") or type(action["is_pick"]) is not bool:
                raise ValueError(f"Invalid side or is_pick at history[{index}]")
            if action["is_pick"]:
                pick_counts[action["side"]] += 1
            heroes.append(encoded)
            categories.append(history_category(action["is_pick"], action["side"]))
        if max(pick_counts.values()) > 5 or (not complete and state["is_pick"] and pick_counts[side] >= 5):
            raise ValueError("A side cannot pick more than five heroes")
        if not complete and len(heroes) >= len(self.hero_index):
            raise ValueError("No available heroes remain")
        warnings = []
        patch_encoded = self.patch_index.get(patch, 0)
        if patch_encoded == 0:
            warnings.append(f"Patch {patch} was not seen during training; using the unknown patch embedding.")
        teams = []
        raw_teams = []
        for name in ("radiant", "dire"):
            team = state.get(f"{name}_team_id")
            if team is not None:
                integer(team, f"{name}_team_id")
            raw_teams.append(team)
            teams.append(self.team_index.get(team, 0))
            if self.team_index and teams[-1] == 0:
                warnings.append(f"{name.title()} team is unspecified or unseen; using the unknown team embedding.")
        if raw_teams[0] is not None and raw_teams[0] == raw_teams[1]:
            raise ValueError("Radiant and Dire must have different team IDs")
        if side == "dire":
            teams.reverse()
        tensors = [
            torch.tensor([heroes], dtype=torch.int16),
            torch.tensor([categories], dtype=torch.int8),
            torch.tensor([patch_encoded], dtype=torch.long),
            torch.tensor([len(history)], dtype=torch.long),
            torch.tensor([0 if side == "radiant" else 1], dtype=torch.long),
            torch.tensor([int(state["is_pick"])], dtype=torch.long),
            torch.tensor([teams], dtype=torch.long),
            torch.zeros((1, 2, 5), dtype=torch.long),
        ]
        return tensors, warnings, top_k

    @torch.inference_mode()
    def predict(self, state: dict) -> dict:
        tensors, warnings, top_k = self.encode(state)
        if len(state["history"]) == self.max_order + 1:
            return {
                "status": "complete", "predictions": [], "warnings": warnings,
                "checkpoint_sha256": self.checkpoint_hash,
            }
        logits, used = self.model(*tensors)
        available = (~used[0]).nonzero().flatten()
        scores = logits[0, available]
        if not torch.isfinite(scores).all():
            raise ValueError("Model produced non-finite scores")
        probabilities = scores.softmax(dim=0)
        order = torch.argsort(probabilities, descending=True, stable=True)[:top_k]
        return {
            "action_order": len(state["history"]),
            "acting_side": state["acting_side"],
            "is_pick": state["is_pick"],
            "available_heroes": len(available),
            "predictions": [
                {"hero_id": self.vocabulary.hero_ids[available[i].item()],
                 "probability": probabilities[i].item()}
                for i in order
            ],
            "warnings": warnings,
            "probability_kind": "uncalibrated_next_action_softmax_not_win_probability",
            "checkpoint_sha256": self.checkpoint_hash,
        }
