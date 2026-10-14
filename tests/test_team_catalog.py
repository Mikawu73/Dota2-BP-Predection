import json

from scripts.build_team_catalog import build_catalog


def test_latest_names_aliases_fallback_and_conflicts(tmp_path):
    rows = [
        {"start_time": 100, "match_id": 1, "radiant_team_id": 10,
         "radiant_team": {"team_id": 10, "name": " Old  Name "},
         "dire_team_id": 20, "dire_name": "Same Name"},
        {"start_time": 200, "match_id": 2, "radiant_team_id": 10,
         "radiant_team": {"team_id": 10, "name": "New Name"},
         "dire_team": {"team_id": 30, "name": "Same Name"}},
        {"start_time": 300, "radiant_team_id": 10,
         "radiant_team": {"team_id": 999, "name": "Wrong Mapping"},
         "dire_team_id": True, "dire_name": "Invalid ID"},
        {"start_time": 0, "radiant_team_id": 10, "radiant_name": "Older Name"},
        {"radiant_team_id": 40, "radiant_team": {"name": "  "}, "radiant_name": "Fallback"},
    ]
    for i, row in enumerate(rows):
        (tmp_path / f"{i}.json").write_text(json.dumps(row), encoding="utf-8")
    (tmp_path / "broken.json").write_text("invalid", encoding="utf-8")
    (tmp_path / "array.json").write_text("[]", encoding="utf-8")
    teams, counts = build_catalog(tmp_path)
    assert teams == [
        {"id": 10, "name": "New Name", "aliases": ["Old Name", "Older Name"]},
        {"id": 20, "name": "Same Name", "aliases": []},
        {"id": 30, "name": "Same Name", "aliases": []},
        {"id": 40, "name": "Fallback", "aliases": []},
    ]
    assert counts == {"files_read": 5, "invalid_files": 2, "id_conflicts": 1}
    assert build_catalog(tmp_path)[0] == teams
