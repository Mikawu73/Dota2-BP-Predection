from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duplicate_player_slots: int = 0
    duplicate_draft_orders: int = 0
    draft_picks_match_players: bool | None = None


def _duplicates(values: list[Any]) -> int:
    return sum(count - 1 for value, count in Counter(values).items() if value is not None and count > 1)


def validate_normalized_match(match: dict[str, Any], players: list[dict[str, Any]], draft_actions: list[dict[str, Any]], raw_match: dict[str, Any] | None = None, source_path: Path | None = None) -> ValidationResult:
    result = ValidationResult()
    if match.get("match_id") is None:
        result.errors.append("invalid match_id")
        return result
    if source_path and source_path.stem.isdigit() and int(source_path.stem) != match["match_id"]:
        result.warnings.append(f"filename match_id {source_path.stem} differs from JSON {match['match_id']}")
    if match.get("start_datetime_utc") is None: result.warnings.append("missing or invalid start_time")
    if match.get("patch_id") is None: result.warnings.append("missing patch_id")
    if match.get("radiant_win") is None: result.warnings.append("missing radiant_win")
    if len(players) != 10: result.warnings.append(f"expected 10 players, got {len(players)}")
    if not draft_actions: result.warnings.append("missing picks_bans")
    result.duplicate_player_slots = _duplicates([p.get("player_slot") for p in players])
    result.duplicate_draft_orders = _duplicates([a.get("order") for a in draft_actions])
    if result.duplicate_player_slots: result.warnings.append("duplicate player_slot")
    if result.duplicate_draft_orders: result.warnings.append("duplicate draft order")
    orders = sorted(a["order"] for a in draft_actions if a.get("order") is not None)
    if orders and orders != list(range(len(orders))): result.warnings.append("draft order is not continuous from 0")
    if any(p.get("account_id") is None for p in players): result.warnings.append("missing account_id")
    if any(p.get("hero_id") is None for p in players): result.warnings.append("missing player hero_id")
    if any(a.get("hero_id") is None for a in draft_actions): result.warnings.append("missing draft hero_id")
    if any(p.get("side") == "unknown" for p in players): result.warnings.append("unknown player side")
    if any(a.get("team_side") == "unknown" for a in draft_actions): result.warnings.append("unknown draft team")
    player_heroes = [p["hero_id"] for p in players if p.get("hero_id") is not None]
    if _duplicates(player_heroes): result.warnings.append("duplicate player hero_id")
    draft_heroes = [a["hero_id"] for a in draft_actions if a.get("hero_id") is not None]
    if _duplicates(draft_heroes): result.warnings.append("duplicate hero_id in draft")
    picked = {a["hero_id"] for a in draft_actions if a.get("is_pick") is True and a.get("hero_id") is not None}
    banned = {a["hero_id"] for a in draft_actions if a.get("is_pick") is False and a.get("hero_id") is not None}
    if picked & banned: result.warnings.append(f"heroes both picked and banned: {sorted(picked & banned)}")
    if raw_match:
        raw_players = raw_match.get("players") if isinstance(raw_match.get("players"), list) else []
        for raw, normalized in zip(raw_players, players):
            if isinstance(raw.get("isRadiant"), bool) and isinstance(raw.get("player_slot"), int):
                slot_side = "radiant" if raw["player_slot"] < 128 else "dire"
                if normalized["side"] != slot_side: result.warnings.append(f"isRadiant/player_slot conflict at slot {raw['player_slot']}")
            raw_win = raw.get("win")
            if raw_win in (0, 1) and normalized["win"] is not None and raw_win != normalized["win"]: result.warnings.append(f"derived win mismatch at slot {raw.get('player_slot')}")
    if players and draft_actions:
        rp = {p["hero_id"] for p in players if p["side"] == "radiant" and p["hero_id"] is not None}; dp = {p["hero_id"] for p in players if p["side"] == "dire" and p["hero_id"] is not None}
        rd = {a["hero_id"] for a in draft_actions if a["is_pick"] is True and a["team"] == 0 and a["hero_id"] is not None}; dd = {a["hero_id"] for a in draft_actions if a["is_pick"] is True and a["team"] == 1 and a["hero_id"] is not None}
        result.draft_picks_match_players = rp == rd and dp == dd
        if not result.draft_picks_match_players: result.warnings.append(f"draft/player heroes differ radiant={sorted(rp ^ rd)} dire={sorted(dp ^ dd)}")
    return result
