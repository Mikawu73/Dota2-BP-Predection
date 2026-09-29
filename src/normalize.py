from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .utils import atomic_write_json
from .validation import ValidationResult, validate_normalized_match

MATCH_COLUMNS = ["match_id", "start_time", "start_datetime_utc", "patch_id", "radiant_win", "radiant_team_id", "dire_team_id"]
PLAYER_COLUMNS = [
    "match_id", "player_slot", "account_id", "professional_name_raw", "side", "hero_id", "win",
    "lane_role", "is_roaming", "gold_per_min", "xp_per_min", "last_hits",
]
DRAFT_COLUMNS = ["match_id", "order", "is_pick", "action_type", "hero_id", "team", "team_side"]
MANIFEST_COLUMNS = ["source_file", "match_id", "status", "player_count", "draft_action_count", "pick_count", "ban_count", "missing_account_id_count", "missing_hero_id_count", "unknown_player_side_count", "unknown_draft_team_count", "duplicate_player_slot_count", "duplicate_draft_order_count", "draft_picks_match_players", "warning_count", "error_count", "warning_messages", "error_messages", "normalized_at_utc"]

SCHEMAS = {
    "matches": pa.schema([("match_id", pa.int64()), ("start_time", pa.int64()), ("start_datetime_utc", pa.timestamp("us", tz="UTC")), ("patch_id", pa.int64()), ("radiant_win", pa.bool_()), ("radiant_team_id", pa.int64()), ("dire_team_id", pa.int64())]),
    "match_players": pa.schema([
        ("match_id", pa.int64()), ("player_slot", pa.int64()), ("account_id", pa.int64()),
        ("professional_name_raw", pa.string()), ("side", pa.string()), ("hero_id", pa.int64()),
        ("win", pa.int64()), ("lane_role", pa.int64()), ("is_roaming", pa.bool_()),
        ("gold_per_min", pa.int64()), ("xp_per_min", pa.int64()), ("last_hits", pa.int64()),
    ]),
    "draft_actions": pa.schema([("match_id", pa.int64()), ("order", pa.int64()), ("is_pick", pa.bool_()), ("action_type", pa.string()), ("hero_id", pa.int64()), ("team", pa.int64()), ("team_side", pa.string())]),
    "normalization_manifest": pa.schema([(name, pa.int64() if name in {"match_id", "player_count", "draft_action_count", "pick_count", "ban_count", "missing_account_id_count", "missing_hero_id_count", "unknown_player_side_count", "unknown_draft_team_count", "duplicate_player_slot_count", "duplicate_draft_order_count", "warning_count", "error_count"} else pa.bool_() if name == "draft_picks_match_players" else pa.timestamp("us", tz="UTC") if name == "normalized_at_utc" else pa.string()) for name in MANIFEST_COLUMNS]),
}


def nullable_int(value: Any, *, positive: bool = False) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if not positive or parsed > 0 else None


def nullable_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def infer_player_side(is_radiant: bool | None, player_slot: int | None) -> str:
    if isinstance(is_radiant, bool):
        return "radiant" if is_radiant else "dire"
    slot = nullable_int(player_slot)
    if slot is None:
        return "unknown"
    return "radiant" if slot < 128 else "dire"


def derive_player_win(side: str, radiant_win: bool | None) -> int | None:
    if radiant_win is None or side not in {"radiant", "dire"}:
        return None
    return int(radiant_win == (side == "radiant"))


def _utc_datetime(value: Any) -> datetime | None:
    timestamp = nullable_int(value)
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def normalize_match(raw_match: dict[str, Any]) -> dict[str, Any]:
    return {
        "match_id": nullable_int(raw_match.get("match_id"), positive=True),
        "start_time": nullable_int(raw_match.get("start_time")),
        "start_datetime_utc": _utc_datetime(raw_match.get("start_time")),
        "patch_id": nullable_int(raw_match.get("patch")),
        "radiant_win": nullable_bool(raw_match.get("radiant_win")),
        "radiant_team_id": nullable_int(raw_match.get("radiant_team_id"), positive=True),
        "dire_team_id": nullable_int(raw_match.get("dire_team_id"), positive=True),
    }


def normalize_players(raw_match: dict[str, Any]) -> list[dict[str, Any]]:
    match = normalize_match(raw_match)
    raw_players = raw_match.get("players")
    if not isinstance(raw_players, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw in raw_players:
        if not isinstance(raw, dict):
            continue
        slot = nullable_int(raw.get("player_slot"))
        side = infer_player_side(raw.get("isRadiant"), slot)
        account_id = nullable_int(raw.get("account_id"), positive=True)
        name = raw.get("name")
        rows.append({
            "match_id": match["match_id"], "player_slot": slot, "account_id": account_id,
            "professional_name_raw": name.strip() if isinstance(name, str) else None,
            "side": side, "hero_id": nullable_int(raw.get("hero_id"), positive=True),
            "win": derive_player_win(side, match["radiant_win"]),
            "lane_role": nullable_int(raw.get("lane_role"), positive=True),
            "is_roaming": nullable_bool(raw.get("is_roaming")),
            "gold_per_min": nullable_int(raw.get("gold_per_min")),
            "xp_per_min": nullable_int(raw.get("xp_per_min")),
            "last_hits": nullable_int(raw.get("last_hits")),
        })
    return rows


def normalize_draft_actions(raw_match: dict[str, Any]) -> list[dict[str, Any]]:
    match_id = nullable_int(raw_match.get("match_id"), positive=True)
    raw_actions = raw_match.get("picks_bans")
    if not isinstance(raw_actions, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw in raw_actions:
        if not isinstance(raw, dict):
            continue
        is_pick = nullable_bool(raw.get("is_pick")); team = nullable_int(raw.get("team"))
        rows.append({
            "match_id": match_id, "order": nullable_int(raw.get("order")), "is_pick": is_pick,
            "action_type": "pick" if is_pick is True else "ban" if is_pick is False else "unknown",
            "hero_id": nullable_int(raw.get("hero_id"), positive=True), "team": team,
            "team_side": "radiant" if team == 0 else "dire" if team == 1 else "unknown",
        })
    return rows


# Compatibility aliases for the collection pipeline.
normalize_draft = normalize_draft_actions


def side_from_player_slot(slot: int | None) -> str:
    return infer_player_side(None, slot)


def side_from_draft_team(value: Any) -> str | None:
    return {0: "radiant", 1: "dire"}.get(nullable_int(value))


def player_win(radiant_win: bool | None, side: str | None) -> bool | None:
    value = derive_player_win(side or "unknown", radiant_win)
    return bool(value) if value is not None else None


def _source_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.glob("*.json") if not p.name.startswith(".") and "summary" not in p.name.lower() and not p.name.endswith(".tmp"))


def _manifest(path: Path, match_id: int | None, result: ValidationResult, players: list[dict[str, Any]], drafts: list[dict[str, Any]], status: str | None = None) -> dict[str, Any]:
    return {
        "source_file": str(path), "match_id": match_id, "status": status or ("partial" if result.warnings else "success"),
        "player_count": len(players), "draft_action_count": len(drafts),
        "pick_count": sum(a["is_pick"] is True for a in drafts), "ban_count": sum(a["is_pick"] is False for a in drafts),
        "missing_account_id_count": sum(p["account_id"] is None for p in players),
        "missing_hero_id_count": sum(p["hero_id"] is None for p in players) + sum(a["hero_id"] is None for a in drafts),
        "unknown_player_side_count": sum(p["side"] == "unknown" for p in players),
        "unknown_draft_team_count": sum(a["team_side"] == "unknown" for a in drafts),
        "duplicate_player_slot_count": result.duplicate_player_slots,
        "duplicate_draft_order_count": result.duplicate_draft_orders,
        "draft_picks_match_players": result.draft_picks_match_players,
        "warning_count": len(result.warnings), "error_count": len(result.errors),
        "warning_messages": json.dumps(result.warnings, ensure_ascii=False), "error_messages": json.dumps(result.errors, ensure_ascii=False),
        "normalized_at_utc": datetime.now(UTC),
    }


def _table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


def _write_parquet_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(table, temp)
    temp.replace(path)


def _write_database(database: Path, tables: dict[str, pa.Table]) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    try:
        connection.execute("SET TimeZone='UTC'")
        connection.execute("BEGIN")
        for name, table in tables.items():
            view = f"incoming_{name}"; connection.register(view, table)
            connection.execute(f"CREATE OR REPLACE TABLE test_{name} AS SELECT * FROM {view}")
            connection.unregister(view)
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def run_normalization(input_dir: Path, output_dir: Path, database: Path, *, limit: int | None = None, match_id: int | None = None, parquet: bool = True, write_database: bool = True) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = _source_files(input_dir)
    if match_id is not None:
        files = [p for p in files if p.stem == str(match_id)]
    if limit is not None:
        files = files[:limit]
    matches: list[dict[str, Any]] = []; players: list[dict[str, Any]] = []; drafts: list[dict[str, Any]] = []; manifests: list[dict[str, Any]] = []
    for path in tqdm(files, desc="normalize", unit="match"):
        before = path.read_bytes()
        try:
            raw = json.loads(before.decode("utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("top-level JSON is not an object")
            match = normalize_match(raw)
            if match["match_id"] is None:
                raise ValueError("invalid match_id")
            match_players = normalize_players(raw); match_drafts = normalize_draft_actions(raw)
            result = validate_normalized_match(match, match_players, match_drafts, raw, path)
            matches.append(match); players.extend(match_players); drafts.extend(match_drafts)
            manifests.append(_manifest(path, match["match_id"], result, match_players, match_drafts))
        except Exception as exc:
            failed = ValidationResult(errors=[str(exc)])
            manifests.append(_manifest(path, None, failed, [], [], "failed"))
            logging.exception("Failed to normalize %s", path)
        if path.read_bytes() != before:
            raise RuntimeError(f"Raw file changed during normalization: {path}")
    matches.sort(key=lambda row: row["match_id"]); players.sort(key=lambda row: (row["match_id"], row["player_slot"] if row["player_slot"] is not None else 999)); drafts.sort(key=lambda row: (row["match_id"], row["order"] if row["order"] is not None else 999))
    tables = {"matches": _table(matches, SCHEMAS["matches"]), "match_players": _table(players, SCHEMAS["match_players"]), "draft_actions": _table(drafts, SCHEMAS["draft_actions"]), "normalization_manifest": _table(manifests, SCHEMAS["normalization_manifest"])}
    if parquet:
        for name, table in tables.items(): _write_parquet_atomic(table, output_dir / f"{name}.parquet")
    if write_database: _write_database(database, tables)
    counts = Counter(row["status"] for row in manifests)
    per_players = Counter(p["match_id"] for p in players)
    per_sides: dict[int, Counter[str]] = {}
    for player in players:
        per_sides.setdefault(player["match_id"], Counter())[player["side"]] += 1

    per_drafts = Counter(a["match_id"] for a in drafts)
    per_picks = Counter(a["match_id"] for a in drafts if a["is_pick"] is True)
    orders_by_match: dict[int, list[int]] = {}
    for action in drafts:
        if action["order"] is not None:
            orders_by_match.setdefault(action["match_id"], []).append(action["order"])
    valid_orders = {
        mid
        for mid, count in per_drafts.items()
        if sorted(orders_by_match.get(mid, [])) == list(range(count))
    }
    comparable = [m for m in manifests if m["draft_picks_match_players"] is not None]
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(), "input_directory": str(input_dir.resolve()), "files_discovered": len(files),
        "files_success": counts["success"], "files_partial": counts["partial"], "files_failed": counts["failed"],
        "matches_written": len(matches), "match_players_written": len(players), "draft_actions_written": len(drafts),
        "matches_with_10_players": sum(v == 10 for v in per_players.values()), "matches_with_5v5_sides": sum(v["radiant"] == 5 and v["dire"] == 5 for v in per_sides.values()),
        "matches_with_24_draft_actions": sum(v == 24 for v in per_drafts.values()), "matches_with_10_picks": sum(v == 10 for v in per_picks.values()),
        "matches_with_valid_draft_order": len(valid_orders), "matches_where_draft_matches_players": sum(m["draft_picks_match_players"] is True for m in comparable),
        "missing_account_ids": sum(p["account_id"] is None for p in players), "missing_player_hero_ids": sum(p["hero_id"] is None for p in players), "missing_draft_hero_ids": sum(a["hero_id"] is None for a in drafts),
        "missing_lane_roles": sum(p["lane_role"] is None for p in players), "missing_is_roaming": sum(p["is_roaming"] is None for p in players),
        "missing_gold_per_min": sum(p["gold_per_min"] is None for p in players), "missing_xp_per_min": sum(p["xp_per_min"] is None for p in players),
        "missing_last_hits": sum(p["last_hits"] is None for p in players),
        "unknown_player_sides": sum(p["side"] == "unknown" for p in players), "unknown_draft_teams": sum(a["team_side"] == "unknown" for a in drafts),
        "duplicate_player_slots": sum(m["duplicate_player_slot_count"] for m in manifests), "duplicate_draft_orders": sum(m["duplicate_draft_order_count"] for m in manifests),
        "confirmed_team_mapping": "0=radiant,1=dire" if comparable and all(m["draft_picks_match_players"] is True for m in comparable) else "unconfirmed",
        "anomalous_match_ids": [m["match_id"] for m in manifests if m["status"] != "success"],
    }
    atomic_write_json(output_dir / "normalization_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize OpenDota match detail JSON without network access")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/match_details")); parser.add_argument("--output-dir", type=Path, default=Path("data/processed/test")); parser.add_argument("--database", type=Path, default=Path("data/database/dota2.duckdb"))
    parser.add_argument("--force", action="store_true", help="Accepted for explicit full rebuilds (rebuild is always idempotent)"); parser.add_argument("--limit", type=int); parser.add_argument("--match-id", type=int); parser.add_argument("--no-parquet", action="store_true"); parser.add_argument("--no-database", action="store_true"); parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_parquet and args.no_database:
        raise SystemExit("--no-parquet and --no-database cannot be used together")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
    summary = run_normalization(args.input_dir, args.output_dir, args.database, limit=args.limit, match_id=args.match_id, parquet=not args.no_parquet, write_database=not args.no_database)
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0 if not summary["files_failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
