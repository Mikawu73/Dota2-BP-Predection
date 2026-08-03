from __future__ import annotations

import argparse
import json
import logging
from calendar import monthrange
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .api_client import APIError, OpenDotaClient
from .config import load_settings
from .storage import Storage
from .utils import atomic_write_json, utc_iso


def subtract_months(value: datetime, months: int) -> datetime:
    absolute = value.year * 12 + value.month - 1 - months
    year, month0 = divmod(absolute, 12)
    return value.replace(year=year, month=month0 + 1, day=min(value.day, monthrange(year, month0 + 1)[1]))


def date_range(args: argparse.Namespace) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    if args.start_date or args.end_date:
        end = datetime.combine(date.fromisoformat(args.end_date), time.min, UTC) if args.end_date else now
        start = datetime.combine(date.fromisoformat(args.start_date), time.min, UTC) if args.start_date else subtract_months(end, args.months)
    else:
        end, start = now, subtract_months(now, args.months)
    if start >= end: raise ValueError("start date must be earlier than end date")
    return start, end


def discover(client: OpenDotaClient, storage: Storage, start_ts: int, end_ts: int, details_only: bool, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if details_only:
        pages = sorted(storage.pages_dir.glob("page_*.json")); all_rows = []
        for path in pages: all_rows.extend(json.loads(path.read_text(encoding="utf-8")))
        unique = {int(row["match_id"]): row for row in all_rows if start_ts <= int(row.get("start_time", 0)) < end_ts}
        return list(unique.values()), {"pages_fetched": 0, "pagination_reached_cutoff": None, "source_pages": len(pages)}
    cursor = None; page = 1; matches: dict[int, dict[str, Any]] = {}; reached = False
    while True:
        rows = client.get_pro_matches(cursor); storage.save_page(page, cursor, rows)
        if not rows: break
        for row in rows:
            ts = int(row.get("start_time", 0)); mid = int(row["match_id"])
            if start_ts <= ts < end_ts: matches[mid] = row
        if limit is not None and len(matches) >= limit:
            return list(matches.values()), {"pages_fetched": page, "pagination_reached_cutoff": False, "stopped_at_limit": True, "last_cursor": cursor}
        earliest = min(int(row.get("start_time", 0)) for row in rows)
        if earliest < start_ts: reached = True; break
        next_cursor = min(int(row["match_id"]) for row in rows)
        if cursor == next_cursor: break
        cursor, page = next_cursor, page + 1
    return list(matches.values()), {"pages_fetched": page, "pagination_reached_cutoff": reached, "stopped_at_limit": False, "last_cursor": cursor}


def summary(storage: Storage, requested_start: datetime, requested_end: datetime, discovered: int, downloaded: int, skipped: int, failed: int, pagination: dict[str, Any], client_requests: int) -> dict[str, Any]:
    row = storage.db.execute("SELECT min(start_time),max(start_time),count(*),count(*) FILTER(WHERE has_complete_players),count(*) FILTER(WHERE has_picks_bans),count(*) FILTER(WHERE patch IS NULL),count(*) FILTER(WHERE league_id IS NULL) FROM matches").fetchone()
    missing_accounts = storage.db.execute("SELECT count(*) FROM match_players WHERE account_id IS NULL").fetchone()[0]
    invalid_sides = storage.db.execute("SELECT count(*) FROM draft_actions WHERE team_side IS NULL").fetchone()[0]
    duplicate_orders = storage.db.execute("SELECT count(*) FROM (SELECT match_id,action_order,count(*) c FROM draft_actions GROUP BY 1,2 HAVING c>1)").fetchone()[0]
    result = {"requested_start_date":requested_start.isoformat(),"requested_end_date":requested_end.isoformat(),"actual_earliest_match":utc_iso(row[0]),"actual_latest_match":utc_iso(row[1]),"matches_discovered":discovered,"matches_downloaded":downloaded,"matches_skipped_existing":skipped,"matches_normalized":row[2],"matches_with_10_players":row[3],"matches_with_draft":row[4],"partial_matches":storage.db.execute("SELECT count(*) FROM download_manifest WHERE download_status='partial'").fetchone()[0],"failed_matches":failed,"missing_account_id":missing_accounts,"missing_patch":row[5],"missing_league":row[6],"draft_unknown_team_values":invalid_sides,"duplicate_draft_orders":duplicate_orders,"api_requests":client_requests,**pagination}
    atomic_write_json(storage.data_dir / "processed/collection_summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(); p.add_argument("--months",type=int,default=12); p.add_argument("--start-date"); p.add_argument("--end-date"); p.add_argument("--request-delay",type=float); p.add_argument("--max-retries",type=int); p.add_argument("--force",action="store_true"); p.add_argument("--details-only",action="store_true"); p.add_argument("--normalize-only",action="store_true"); p.add_argument("--limit",type=int); p.add_argument("--log-level",default="INFO"); return p


def main() -> int:
    args=parser().parse_args(); settings=load_settings(); settings.log_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=getattr(logging,args.log_level.upper()),format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler(settings.log_dir/"collection.log",encoding="utf-8")])
    start,end=date_range(args); start_ts,end_ts=int(start.timestamp()),int(end.timestamp()); storage=Storage(settings.data_dir); client=OpenDotaClient(settings,args.request_delay,args.max_retries)
    logging.info("UTC range %s to %s",start.isoformat(),end.isoformat())
    if args.normalize_only:
        count=storage.normalize_saved(start_ts,end_ts); result=summary(storage,start,end,count,0,0,0,{"pagination_reached_cutoff":None},0); logging.info("Summary: %s",json.dumps(result,ensure_ascii=False)); return 0
    matches,pagination=discover(client,storage,start_ts,end_ts,args.details_only,args.limit); matches.sort(key=lambda x:int(x["match_id"]),reverse=True)
    if args.limit is not None: matches=matches[:args.limit]
    downloaded=skipped=failed=0
    try: storage.save_heroes(client.get_heroes())
    except APIError as exc: logging.warning("Hero constants unavailable: %s",exc)
    for item in tqdm(matches,desc="match details"):
        mid=int(item["match_id"]); raw=storage.raw_path(mid)
        if raw.exists() and not args.force:
            skipped+=1
            try: storage.normalize_match(json.loads(raw.read_text(encoding="utf-8")))
            except (ValueError,OSError,KeyError,TypeError) as exc: logging.error("Invalid existing %s: %s",mid,exc)
            continue
        try:
            detail=client.get_match_details(mid); path=storage.save_detail(mid,detail); storage.normalize_match(detail)
            status="success" if len(detail.get("players") or [])==10 and detail.get("picks_bans") else "partial"
            storage.record_manifest(mid,item.get("start_time"),status,path=path,detail=detail); downloaded+=1
        except APIError as exc:
            status="not_found" if exc.status==404 else "failed"; storage.record_manifest(mid,item.get("start_time"),status,http_status=exc.status,attempts=exc.attempts,error=str(exc)); failed+=1; logging.error("Match %s: %s",mid,exc)
    result=summary(storage,start,end,len(matches),downloaded,skipped,failed,pagination,client.request_count); logging.info("Summary: %s",json.dumps(result,ensure_ascii=False)); print(json.dumps(result,ensure_ascii=False,indent=2)); return 0 if not failed else 2


if __name__ == "__main__": raise SystemExit(main())
