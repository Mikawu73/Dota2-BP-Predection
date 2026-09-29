from scripts.evaluate_statistical_baselines import _blend, _probabilities, _rank
from collections import Counter


def test_probability_blending_and_used_hero_mask():
    heroes = [1, 2, 3]
    probabilities = _probabilities(Counter({1: 1, 2: 3}), heroes)
    assert probabilities == [0.25, 0.75, 0.0]
    assert _blend(((0.5, probabilities), (0.5, [0.0, 0.0, 1.0])), probabilities) == [0.125, 0.375, 0.5]

    # Hero 2 has the highest score, but it is already used and must be masked.
    assert _rank(3, {2}, [0.1, 0.9, 0.5], heroes, {1: 0, 2: 1, 3: 2}) == 1
