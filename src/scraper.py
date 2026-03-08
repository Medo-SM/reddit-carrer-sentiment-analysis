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
    """
    Returns (last_timestamp, last_id, line_count) from the JSONL file.
    If file doesn't exist or is empty, returns (START_TIMESTAMP, None, 0).
    """
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

                # Resume info: last timestamp, last id, total count so far
                current_ts, last_id, total_count = await get_last_ts_id_and_count(file_path)

                stop_ts = int((END_DATE + timedelta(days=1)).timestamp())
                last_printed_month = datetime.fromtimestamp(current_ts, timezone.utc).month

                print(f"[*] Starting r/{sub} {mode}s...")

                while current_ts < stop_ts:
                    next_ts = min(current_ts + (CHUNK_DAYS * 86400), stop_ts)

                    url = f"https://api.pullpush.io/reddit/search/{mode}/"
                    params = {
                        "subreddit": sub,
                        "after": current_ts,          # may include items with timestamp == current_ts
                        "before": next_ts,
                        "size": 100,
                        "sort": "asc"
                    }
                    #Debug Code to see the current data requested from the API
                    #print(f"[DEBUG] Requesting {datetime.fromtimestamp(current_ts, timezone.utc)} to {datetime.fromtimestamp(next_ts, timezone.utc)}")

                    try:
                        resp = await client.get(url, params=params)

                        if resp.status_code == 200:
                            data = resp.json().get('data', [])

                            if data:
                                # --- Deduplication against the last saved item ---
                                # If the first item has the same timestamp and id as last saved, skip it.
                                start_index = 0
                                if last_id is not None and data[0]['created_utc'] == current_ts and data[0]['id'] == last_id:
                                    start_index = 1
                                    # Also update current_ts in case this was the only item
                                    if len(data) == 1:
                                        current_ts = next_ts
                                        continue

                                # Write new items (excluding the possible duplicate)
                                with open(file_path, 'a', encoding='utf-8') as f:
                                    for item in data[start_index:]:
                                          if mode == 'comment':
                                            clean = {
                                            "id": item.get("id"),
                                            "created_utc": item.get("created_utc"),
                                            "author": item.get("author"),
                                            "content": item.get("body"),
                                            "score": item.get("score"),           # <-- ADDED
                                            "link_id": item.get("link_id"),       # submission this comment belongs to
                                            "parent_id": item.get("parent_id")    # immediate parent (comment or submission)
                                            }
                                          else:  # submission
                                            clean = {
                                            "id": item.get("id"),
                                            "created_utc": item.get("created_utc"),
                                            "author": item.get("author"),
                                            "content": item.get("title"),         # submissions use title
                                            "score": item.get("score")            # <-- ADDED
                                             # Optional: add "url", "num_comments", etc.
                                            }
                                          f.write(json.dumps(clean) + "\n")

                                # Update counters and last ID
                                written_items = data[start_index:]
                                if written_items:
                                    total_count += len(written_items)
                                    current_ts = written_items[-1]['created_utc']
                                    last_id = written_items[-1]['id']
                                else:
                                    # No new items were written (only duplicate)
                                    current_ts = next_ts
                            else:
                                # No data in this chunk → move to next chunk
                                current_ts = next_ts

                        elif resp.status_code == 429:
                            await asyncio.sleep(30)
                            continue

                        # --- Progress log (monthly) ---
                        current_dt = datetime.fromtimestamp(current_ts, timezone.utc)
                        if current_dt.month != last_printed_month:
                            month_str = current_dt.strftime('%B %Y')
                            print(f"[PROGRESS] r/{sub} {mode}s: Completed {month_str} | Total: {total_count:,}")
                            last_printed_month = current_dt.month

                        await asyncio.sleep(0.5)  # Polite delay

                    except Exception as e:
                        print(f"[!] Error: {e} – retrying after 5s")
                        await asyncio.sleep(5)

                print(f"[√] FINISHED r/{sub} {mode}s.")

# --- EXECUTION ---
await scrape_reddit()