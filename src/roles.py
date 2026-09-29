from __future__ import annotations

from itertools import permutations
from typing import Any


POSITIONS = (1, 2, 3, 4, 5)


def _resource_percentiles(players: list[dict[str, Any]]) -> list[float]:
    values = []
    for player in players:
        metrics = [player.get("gold_per_min"), player.get("xp_per_min"), player.get("last_hits")]
        values.append(sum(float(value or 0) for value in metrics) / len(metrics))
    order = sorted(range(len(values)), key=values.__getitem__)
    percentiles = [0.5] * len(values)
    for rank, index in enumerate(order):
        percentiles[index] = rank / max(len(values) - 1, 1)
    return percentiles


def position_scores(player: dict[str, Any], resource_percentile: float) -> tuple[float, ...]:
    """Score a player's compatibility with positions 1-5.

    OpenDota lane_role is a lane label (safe/mid/off/jungle), not a farm-position
    label. Resource percentile separates cores and supports that shared a lane.
    """
    lane_role = player.get("lane_role")
    roaming = player.get("is_roaming") is True
    core = resource_percentile
    support = 1.0 - resource_percentile
    scores = [2.2 * core, 2.0 * core, 1.1 * core + 0.5, 1.7 * support, 2.0 * support]
    if lane_role == 1:  # safe lane: usually position 1 + 5
        scores[0] += 2.4
        scores[4] += 2.0
    elif lane_role == 2:  # mid
        scores[1] += 3.3
    elif lane_role == 3:  # off lane: usually position 3 + 4
        scores[2] += 2.5
        scores[3] += 1.8
    elif lane_role == 4:  # jungle/unknown lane assignment
        scores[3] += 1.5
    if roaming:
        scores[3] += 1.4
        scores[4] += 0.4
        scores[0] -= 0.5
        scores[1] -= 0.5
    return tuple(scores)


def infer_team_positions(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(players) != 5:
        raise ValueError(f"Expected five players, got {len(players)}")
    percentiles = _resource_percentiles(players)
    score_matrix = [position_scores(player, percentile) for player, percentile in zip(players, percentiles)]
    assignments = []
    for positions in permutations(POSITIONS):
        total = sum(score_matrix[index][position - 1] for index, position in enumerate(positions))
        assignments.append((total, positions))
    assignments.sort(reverse=True)
    best_score, best_positions = assignments[0]
    margin = best_score - assignments[1][0]
    confidence = margin / max(abs(best_score), 1.0)
    return [
        {
            **player,
            "position": position,
            "position_score": score_matrix[index][position - 1],
            "assignment_confidence": confidence,
            "resource_percentile": percentiles[index],
        }
        for index, (player, position) in enumerate(zip(players, best_positions))
    ]
