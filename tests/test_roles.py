from src.roles import infer_team_positions


def test_position_assignment_is_unique_and_uses_lane_and_farm():
    players = [
        {"hero_id": 1, "lane_role": 1, "is_roaming": False, "gold_per_min": 700, "xp_per_min": 700, "last_hits": 350},
        {"hero_id": 2, "lane_role": 2, "is_roaming": False, "gold_per_min": 650, "xp_per_min": 750, "last_hits": 300},
        {"hero_id": 3, "lane_role": 3, "is_roaming": False, "gold_per_min": 500, "xp_per_min": 600, "last_hits": 220},
        {"hero_id": 4, "lane_role": 3, "is_roaming": True, "gold_per_min": 350, "xp_per_min": 450, "last_hits": 80},
        {"hero_id": 5, "lane_role": 1, "is_roaming": False, "gold_per_min": 280, "xp_per_min": 380, "last_hits": 35},
    ]
    inferred = infer_team_positions(players)
    assert sorted(row["position"] for row in inferred) == [1, 2, 3, 4, 5]
    assert {row["hero_id"]: row["position"] for row in inferred} == {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    assert all(row["assignment_confidence"] > 0 for row in inferred)


def test_position_assignment_requires_five_players():
    try:
        infer_team_positions([{}])
    except ValueError as error:
        assert "five" in str(error)
    else:
        raise AssertionError("Expected ValueError")
