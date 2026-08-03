from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from .normalize import derive_player_win, infer_player_side, nullable_int, normalize_draft_actions
from .utils import atomic_write_json, utc_iso


class Storage:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.details_dir = data_dir / "raw/match_details"
        self.pages_dir = data_dir / "raw/pro_matches"
        (data_dir / "database").mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(data_dir / "database/dota.duckdb"))
        self._schema()

    def _schema(self) -> None:
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS matches(match_id BIGINT PRIMARY KEY,start_time BIGINT,start_datetime_utc VARCHAR,duration INTEGER,patch INTEGER,game_mode INTEGER,lobby_type INTEGER,league_id BIGINT,league_name VARCHAR,radiant_team_id BIGINT,radiant_team_name VARCHAR,dire_team_id BIGINT,dire_team_name VARCHAR,radiant_win BOOLEAN,radiant_score INTEGER,dire_score INTEGER,first_blood_time INTEGER,has_complete_players BOOLEAN,has_picks_bans BOOLEAN);
        CREATE TABLE IF NOT EXISTS match_players(match_id BIGINT,account_id BIGINT,personaname VARCHAR,name VARCHAR,team_id BIGINT,team_name VARCHAR,side VARCHAR,player_slot INTEGER,hero_id INTEGER,kills INTEGER,deaths INTEGER,assists INTEGER,gold_per_min INTEGER,xp_per_min INTEGER,last_hits INTEGER,denies INTEGER,hero_damage INTEGER,tower_damage INTEGER,hero_healing INTEGER,level INTEGER,win BOOLEAN,PRIMARY KEY(match_id,player_slot));
        CREATE TABLE IF NOT EXISTS draft_actions(match_id BIGINT,action_order INTEGER,is_pick BOOLEAN,action_type VARCHAR,hero_id INTEGER,team INTEGER,team_side VARCHAR,raw_team_value INTEGER,PRIMARY KEY(match_id,action_order));
        CREATE TABLE IF NOT EXISTS download_manifest(match_id BIGINT PRIMARY KEY,start_time BIGINT,download_status VARCHAR,http_status INTEGER,attempt_count INTEGER,downloaded_at VARCHAR,raw_file_path VARCHAR,has_players BOOLEAN,has_picks_bans BOOLEAN,error_message VARCHAR);
        CREATE TABLE IF NOT EXISTS heroes(hero_id INTEGER PRIMARY KEY,internal_name VARCHAR,localized_name VARCHAR,primary_attr VARCHAR,attack_type VARCHAR,roles VARCHAR);
        """)

    def raw_path(self, match_id: int) -> Path:
        return self.details_dir / f"{match_id}.json"

    def save_page(self, page: int, cursor: int | None, value: Any) -> Path:
        path = self.pages_dir / f"page_{page:05d}_cursor_{cursor or 'latest'}.json"
        atomic_write_json(path, value)
        return path

    def save_detail(self, match_id: int, value: Any) -> Path:
        path = self.raw_path(match_id)
        atomic_write_json(path, value)
        return path

    def record_manifest(self, match_id: int, start_time: int | None, status: str, path: Path | None = None, http_status: int | None = None, attempts: int = 1, error: str | None = None, detail: dict[str, Any] | None = None) -> None:
        players = detail.get("players") if detail else None
        draft = detail.get("picks_bans") if detail else None
        self.db.execute("INSERT OR REPLACE INTO download_manifest VALUES (?,?,?,?,?,?,?,?,?,?)", [match_id,start_time,status,http_status,attempts,datetime.now(UTC).isoformat(),str(path) if path else None,bool(players),bool(draft),error])

    def normalize_match(self, match: dict[str, Any]) -> None:
        mid = int(match["match_id"])
        players = []
        for player in match.get("players") or []:
            side = infer_player_side(player.get("isRadiant"), player.get("player_slot"))
            team = (match.get("radiant_team") or {}) if side == "radiant" else (match.get("dire_team") or {}) if side == "dire" else {}
            players.append({"match_id":mid,"account_id":nullable_int(player.get("account_id"),positive=True),"personaname":player.get("personaname"),"name":player.get("name"),"team_id":team.get("team_id"),"team_name":team.get("name"),"side":side,"player_slot":nullable_int(player.get("player_slot")),"hero_id":nullable_int(player.get("hero_id"),positive=True),"kills":player.get("kills"),"deaths":player.get("deaths"),"assists":player.get("assists"),"gold_per_min":player.get("gold_per_min"),"xp_per_min":player.get("xp_per_min"),"last_hits":player.get("last_hits"),"denies":player.get("denies"),"hero_damage":player.get("hero_damage"),"tower_damage":player.get("tower_damage"),"hero_healing":player.get("hero_healing"),"level":player.get("level"),"win":derive_player_win(side,match.get("radiant_win"))})
        draft = [{"match_id":row["match_id"],"action_order":row["order"],"is_pick":row["is_pick"],"action_type":row["action_type"],"hero_id":row["hero_id"],"team":row["team"],"team_side":row["team_side"],"raw_team_value":row["team"]} for row in normalize_draft_actions(match)]
        radiant, dire, league = match.get("radiant_team") or {}, match.get("dire_team") or {}, match.get("league") or {}
        self.db.execute("DELETE FROM match_players WHERE match_id=?", [mid]); self.db.execute("DELETE FROM draft_actions WHERE match_id=?", [mid])
        self.db.execute("INSERT OR REPLACE INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [mid,match.get("start_time"),utc_iso(match.get("start_time")),match.get("duration"),match.get("patch"),match.get("game_mode"),match.get("lobby_type"),match.get("leagueid"),league.get("name"),match.get("radiant_team_id") or radiant.get("team_id"),match.get("radiant_name") or radiant.get("name"),match.get("dire_team_id") or dire.get("team_id"),match.get("dire_name") or dire.get("name"),match.get("radiant_win"),match.get("radiant_score"),match.get("dire_score"),match.get("first_blood_time"),len(players)==10,bool(draft)])
        for row in players:
            self.db.execute("INSERT INTO match_players VALUES (" + ",".join("?" for _ in row) + ")", list(row.values()))
        for row in draft:
            self.db.execute("INSERT INTO draft_actions VALUES (?,?,?,?,?,?,?,?)", list(row.values()))

    def normalize_saved(self, start_ts: int, end_ts: int) -> int:
        count = 0
        for path in self.details_dir.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8")); ts = value.get("start_time")
                if ts is not None and start_ts <= ts < end_ts:
                    self.normalize_match(value); count += 1
            except (ValueError, OSError, KeyError, TypeError):
                continue
        return count

    def save_heroes(self, heroes: dict[str, dict[str, Any]]) -> None:
        for hero in heroes.values():
            self.db.execute("INSERT OR REPLACE INTO heroes VALUES (?,?,?,?,?,?)", [hero.get("id"),hero.get("name"),hero.get("localized_name"),hero.get("primary_attr"),hero.get("attack_type"),json.dumps(hero.get("roles") or [])])
