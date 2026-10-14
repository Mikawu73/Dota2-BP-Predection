from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def positive_int(value) -> bool:
    return type(value) is int and value > 0


def build_catalog(input_dir: Path, progress: bool = False) -> tuple[list[dict], dict]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    teams: dict[int, dict] = {}
    counts = {"files_read": 0, "invalid_files": 0, "id_conflicts": 0}
    files = sorted(input_dir.glob("*.json"))
    for path in tqdm(files, desc="Team names", unit="match", disable=not progress):
        try:
            match = json.loads(path.read_bytes())
        except (OSError, ValueError):
            counts["invalid_files"] += 1
            continue
        if not isinstance(match, dict):
            counts["invalid_files"] += 1
            continue
        counts["files_read"] += 1
        timestamp = match.get("start_time")
        match_id = match.get("match_id")
        timestamp = timestamp if positive_int(timestamp) else 0
        match_id = match_id if positive_int(match_id) else 0
        for side in ("radiant", "dire"):
            nested = match.get(f"{side}_team")
            nested = nested if isinstance(nested, dict) else {}
            team_id = match.get(f"{side}_team_id")
            nested_id = nested.get("team_id")
            if positive_int(team_id) and positive_int(nested_id) and team_id != nested_id:
                counts["id_conflicts"] += 1
                continue
            if not positive_int(team_id):
                team_id = nested_id
            if not positive_int(team_id):
                continue
            candidates = (nested.get("name"), match.get(f"{side}_name"))
            name = next((" ".join(value.split()) for value in candidates
                         if isinstance(value, str) and value.strip()), None)
            if name is None:
                continue
            # A renamed team keeps its ID. Display the latest observed name deterministically.
            rank = (timestamp, match_id, name)
            record = teams.setdefault(team_id, {"rank": rank, "name": name, "names": set()})
            record["names"].add(name)
            if rank > record["rank"]:
                record.update(rank=rank, name=name)
    catalog = [
        {"id": team_id, "name": record["name"],
         "aliases": sorted(record["names"] - {record["name"]})}
        for team_id, record in sorted(teams.items())
    ]
    return catalog, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build display-only team names from local match JSON")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/match_details"))
    parser.add_argument("--output", type=Path, default=Path("src/web/teams.json"))
    args = parser.parse_args()
    catalog, counts = build_catalog(args.input_dir, progress=True)
    if not catalog:
        parser.error("No named teams found; existing output was not changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({**counts, "teams": len(catalog), "output": str(args.output)}))


if __name__ == "__main__":
    main()
