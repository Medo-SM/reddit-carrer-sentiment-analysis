import asyncio
import httpx
import json
import os
from datetime import datetime, timezone, timedelta

# --- 1. CONFIGURATION ---
SUBREDDITS = ['datascience', 'cscareerquestions']
START_DATE = datetime(2019, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 1, 31, tzinfo=timezone.utc)
DATA_DIR = "/content/drive/MyDrive/reddit_project"
CHUNK_DAYS = 7

# --- 2. HELPERS ---
async def get_last_ts_id_and_count(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return int(START_DATE.timestamp()), None, 0
    line_count = 0
    last_line = None
    with open(path, 'rb') as f:
        for line in f:
            line_count += 1
            last_line = line
    if last_line is None:
        return int(START_DATE.timestamp()), None, 0
    last_item = json.loads(last_line)
    return last_item['created_utc'], last_item['id'], line_count

# --- 3. CORE SCRAPER ---
async def scrape_reddit():
    os.makedirs(DATA_DIR, exist_ok=True)
    async with httpx.AsyncClient(timeout=60.0) as client:
        for sub in SUBREDDITS:
            for mode in ['submission', 'comment']:
                file_path = f"{DATA_DIR}/{sub}_{mode}s.jsonl"
                current_ts, last_id, total_count = await get_last_ts_id_and_count(file_path)
                stop_ts = int((END_DATE + timedelta(days=1)).timestamp())
                last_printed_month = datetime.fromtimestamp(current_ts, timezone.utc).month
                print(f"[*] Starting r/{sub} {mode}s...")
                
                try:
                    while current_ts < stop_ts:
                        next_ts = min(current_ts + (CHUNK_DAYS * 86400), stop_ts)
                        url = f"https://api.pullpush.io/reddit/search/{mode}/"
                        params = {"subreddit": sub, "after": current_ts, "before": next_ts, "size": 100, "sort": "asc"}
                        print(f"[DEBUG] Requesting {datetime.fromtimestamp(current_ts, timezone.utc)} to {datetime.fromtimestamp(next_ts, timezone.utc)}")
                        
                        try:
                            resp = await client.get(url, params=params)
                            if resp.status_code == 200:
                                data = resp.json().get('data', [])
                                if data:
                                    start_index = 0
                                    if last_id is not None and data[0]['created_utc'] == current_ts and data[0]['id'] == last_id:
                                        start_index = 1
                                        if len(data) == 1:
                                            current_ts = next_ts
                                            continue
                                    with open(file_path, 'a', encoding='utf-8') as f:
                                        for item in data[start_index:]:
                                            if mode == 'comment':
                                                clean = {"id": item.get("id"), "created_utc": item.get("created_utc"), "author": item.get("author"), "content": item.get("body"), "score": item.get("score"), "link_id": item.get("link_id"), "parent_id": item.get("parent_id")}
                                            else:
                                                clean = {"id": item.get("id"), "created_utc": item.get("created_utc"), "author": item.get("author"), "content": item.get("title"), "score": item.get("score")}
                                            f.write(json.dumps(clean) + "\n")
                                    written_items = data[start_index:]
                                    if written_items:
                                        total_count += len(written_items)
                                        current_ts = written_items[-1]['created_utc']
                                        last_id = written_items[-1]['id']
                                    else:
                                        current_ts = next_ts
                                else:
                                    current_ts = next_ts
                            elif resp.status_code == 429:
                                await asyncio.sleep(30)
                                continue
                            current_dt = datetime.fromtimestamp(current_ts, timezone.utc)
                            if current_dt.month != last_printed_month:
                                print(f"[PROGRESS] r/{sub} {mode}s: Completed {current_dt.strftime('%B %Y')} | Total: {total_count:,}")
                                last_printed_month = current_dt.month
                            await asyncio.sleep(0.5)
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(f"[!] Error: {e} \u2013 retrying after 5s")
                            await asyncio.sleep(5)
                    print(f"[\u221a] FINISHED r/{sub} {mode}s.")
                except asyncio.CancelledError:
                    print(f"[!] Scraper for r/{sub} {mode}s was cancelled gracefully.")
                    return

await scrape_reddit()
