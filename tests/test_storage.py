import json

from src.storage import Storage
from src.utils import atomic_write_json


def test_idempotent_and_temp_ignored(tmp_path):
    storage=Storage(tmp_path); match={"match_id":1,"start_time":100,"radiant_win":True,"players":[{"player_slot":0}],"picks_bans":[]}
    storage.normalize_match(match); storage.normalize_match(match)
    assert storage.db.execute("select count(*) from matches").fetchone()[0]==1
    atomic_write_json(storage.details_dir/"1.json",match)
    (storage.details_dir/"2.json.tmp").write_text(json.dumps({"match_id":2,"start_time":100}))
    assert storage.normalize_saved(0,200)==1

