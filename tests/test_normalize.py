from datetime import UTC

from src.normalize import DRAFT_COLUMNS, MATCH_COLUMNS, PLAYER_COLUMNS, derive_player_win, infer_player_side, normalize_draft_actions, normalize_match, normalize_players
from src.validation import validate_normalized_match


def fixture_match():
    return {"match_id":1,"start_time":1_700_000_000,"patch":55,"radiant_win":True,"radiant_team_id":10,"dire_team_id":20,"players":[{"player_slot":i,"isRadiant":True,"account_id":0 if i==0 else i+1,"name":" Pro ","hero_id":i+1,"win":1} for i in range(5)]+[{"player_slot":128+i,"isRadiant":False,"account_id":10+i,"name":None,"hero_id":6+i,"win":0} for i in range(5)],"picks_bans":[{"order":i,"is_pick":True,"hero_id":i+1,"team":0 if i<5 else 1} for i in range(10)]}


def test_match_columns_and_utc():
    row=normalize_match(fixture_match())
    assert list(row)==MATCH_COLUMNS and row["start_datetime_utc"].tzinfo is UTC


def test_players_rules_and_columns():
    rows=normalize_players(fixture_match())
    assert len(rows)==10 and list(rows[0])==PLAYER_COLUMNS
    assert rows[0]["account_id"] is None and rows[0]["professional_name_raw"]=="Pro"
    assert rows[0]["side"]=="radiant" and rows[5]["side"]=="dire"


def test_side_and_win_rules():
    assert infer_player_side(None,0)=="radiant" and infer_player_side(None,128)=="dire"
    assert infer_player_side(False,0)=="dire"
    assert derive_player_win("radiant",True)==1 and derive_player_win("dire",True)==0
    assert derive_player_win("radiant",False)==0 and derive_player_win("dire",False)==1
    assert derive_player_win("unknown",True) is None and derive_player_win("radiant",None) is None


def test_draft_columns_and_mappings():
    rows=normalize_draft_actions(fixture_match())
    assert list(rows[0])==DRAFT_COLUMNS and rows[0]["action_type"]=="pick"
    assert rows[0]["team_side"]=="radiant" and rows[5]["team_side"]=="dire"
    raw={"match_id":1,"picks_bans":[{"order":0,"team":9,"is_pick":None}]}
    assert normalize_draft_actions(raw)[0]["team_side"]=="unknown"


def test_validation_duplicates_conflict_and_draft_match():
    raw=fixture_match(); match=normalize_match(raw); players=normalize_players(raw); drafts=normalize_draft_actions(raw)
    result=validate_normalized_match(match,players,drafts,raw)
    assert result.draft_picks_match_players is True
    players.append(dict(players[0])); drafts.append(dict(drafts[0]))
    result=validate_normalized_match(match,players,drafts,raw)
    assert result.duplicate_player_slots==1 and result.duplicate_draft_orders==1
    raw["players"][0]["isRadiant"]=False
    result=validate_normalized_match(normalize_match(raw),normalize_players(raw),normalize_draft_actions(raw),raw)
    assert any("conflict" in warning for warning in result.warnings)


def test_missing_nested_arrays_are_safe():
    assert normalize_players({"match_id":1})==[] and normalize_draft_actions({"match_id":1})==[]
