from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.draft_mlp import DraftContextMLP, DraftDataset, DraftVocabulary
from src.inference import DraftPredictor


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(42)
    vocabulary = DraftVocabulary(tuple(range(1, 31)), (58, 59), (100, 200))
    config = {
        "hero_count": 30, "patch_count": 2, "team_count": 2,
        "player_count": 0, "max_action_order": 23, "hidden_sizes": [16, 8],
        "dropout": 0.2, "lineup_encoder": True,
    }
    model = DraftContextMLP(**config)
    path = tmp_path / "model.pt"
    torch.save({"model_config": config, "vocabulary": vocabulary.to_dict(),
                "model_state_dict": model.state_dict(), "epoch": 1}, path)
    return path


@pytest.fixture
def state():
    return {
        "patch_id": 59, "radiant_team_id": 100, "dire_team_id": 200,
        "acting_side": "dire", "is_pick": True, "top_k": 5,
        "history": [
            {"hero_id": 1, "side": "radiant", "is_pick": False},
            {"hero_id": 2, "side": "dire", "is_pick": True},
        ],
    }


def test_encoding_matches_training_dataset(checkpoint, state, tmp_path):
    predictor = DraftPredictor(checkpoint)
    row = {
        "match_id": 10, "split": "validation", "patch_id": state["patch_id"],
        "radiant_team_id": 100, "dire_team_id": 200, "action_order": 2,
        "acting_side": state["acting_side"], "is_pick": state["is_pick"],
        "previous_hero_ids": [1, 2], "previous_is_pick": [False, True],
        "previous_team_sides": ["radiant", "dire"], "target_hero_id": 3,
    }
    path = tmp_path / "samples.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    dataset = DraftDataset(path, "validation", predictor.vocabulary)
    tensors, warnings, _ = predictor.encode(state)
    assert warnings == []
    for expected, actual in zip(dataset[0][:8], tensors):
        assert torch.equal(expected.unsqueeze(0), actual)
    assert tensors[6].tolist() == [[2, 1]]


def test_ranking_is_deterministic_masked_and_normalized(checkpoint, state):
    predictor = DraftPredictor(checkpoint)
    result = predictor.predict(state)
    assert result == predictor.predict(state)
    assert len(result["predictions"]) == 5
    assert not ({1, 2} & {item["hero_id"] for item in result["predictions"]})
    assert result["available_heroes"] == 28
    assert len(result["checkpoint_sha256"]) == 64
    scores = [row["probability"] for row in result["predictions"]]
    assert scores == sorted(scores, reverse=True)
    tensors, _, _ = predictor.encode(state)
    with torch.inference_mode():
        logits, _ = predictor.model(*tensors)
    probabilities = logits.softmax(dim=-1)[0]
    for item in result["predictions"]:
        assert item["probability"] == pytest.approx(probabilities[item["hero_id"] - 1].item())
    assert sum(scores) < 1  # Top-k is not separately renormalized.


def test_empty_history_and_unknown_context(checkpoint, state):
    predictor = DraftPredictor(checkpoint)
    state.update(history=[], patch_id=60, radiant_team_id=None, dire_team_id=999)
    tensors, warnings, _ = predictor.encode(state)
    assert tensors[0].shape == (1, 0)
    assert tensors[2].tolist() == [0]
    assert tensors[6].tolist() == [[0, 0]]
    assert len(warnings) == 3
    assert len(predictor.predict(state)["predictions"]) == 5


@pytest.mark.parametrize("change, message", [
    ({"patch_id": True}, "patch_id"),
    ({"patch_id": 1.5}, "patch_id"),
    ({"is_pick": 1}, "boolean"),
    ({"acting_side": "other"}, "acting_side"),
    ({"top_k": 21}, "top_k"),
    ({"top_k": 0}, "top_k"),
    ({"radiant_team_id": 200}, "different team"),
    ({"target_hero_id": 3}, "Unexpected fields"),
    ({"history": None}, "history"),
    ({"history": [{"hero_id": 999, "side": "dire", "is_pick": True}]}, "absent"),
    ({"history": [{"hero_id": 1, "side": "dire", "is_pick": "true"}]}, "Invalid"),
])
def test_invalid_states_rejected(checkpoint, state, change, message):
    state.update(change)
    with pytest.raises(ValueError, match=message):
        DraftPredictor(checkpoint).predict(state)


def test_duplicates_and_pick_limits(checkpoint, state):
    predictor = DraftPredictor(checkpoint)
    state["history"].append(deepcopy(state["history"][0]))
    with pytest.raises(ValueError, match="already"):
        predictor.predict(state)
    state["history"] = [{"hero_id": i, "side": "dire", "is_pick": True} for i in range(1, 6)]
    with pytest.raises(ValueError, match="five heroes"):
        predictor.predict(state)
    state["is_pick"] = False
    assert predictor.predict(state)["available_heroes"] == 25


def test_last_action_and_complete_draft(checkpoint, state):
    predictor = DraftPredictor(checkpoint)
    # Freeform sandbox checks uniqueness and pick limits, not patch-specific CM schedules.
    state["history"] = [{"hero_id": i, "side": "dire", "is_pick": False} for i in range(1, 24)]
    result = predictor.predict(state)
    assert result["action_order"] == 23
    state["history"].append({"hero_id": 24, "side": "radiant", "is_pick": True})
    assert predictor.predict(state)["status"] == "complete"
    state["history"].append({"hero_id": 25, "side": "radiant", "is_pick": True})
    with pytest.raises(ValueError, match="at most 24"):
        predictor.predict(state)


def test_corrupt_vocabulary_fails_closed(checkpoint):
    saved = torch.load(checkpoint, weights_only=True)
    saved["vocabulary"]["hero_ids"] = (1, 1)
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="vocabulary"):
        DraftPredictor(checkpoint)


def test_cli_works_without_parquet_files(checkpoint, state, tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.predict_draft", "--checkpoint", str(checkpoint), "--input", str(path)],
        capture_output=True, text=True, check=True,
    )
    assert len(json.loads(result.stdout)["predictions"]) == 5
    path.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.predict_draft", "--checkpoint", str(checkpoint), "--input", str(path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "patch_id" in result.stderr
