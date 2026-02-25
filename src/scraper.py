import asyncio
import httpx
import json
import os
from datetime import datetime, timezone

# --- 1. CONFIGURATION ---
SUBREDDITS = ['datascience', 'cscareerquestions']
START_DATE = datetime(2019, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 31, tzinfo=timezone.utc)
DATA_DIR = "reddit_data"
CHUNK_DAYS = 7  # Small chunks to prevent 502/504 Gateway errors

# --- 2. HELPERS ---

async def get_last_ts(path):
    """Checks the last line of the file to resume from where you left off."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return int(START_DATE.timestamp())
    with open(path, 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        line = f.readline().decode()
        return json.loads(line)['created_utc']

# --- 3. CORE SCRAPER ---

async def scrape_reddit():
    os.makedirs(DATA_DIR, exist_ok=True)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for sub in SUBREDDITS:
            for mode in [ 'comment']:
                file_path = f"{DATA_DIR}/{sub}_{mode}s.jsonl"
                current_ts = await get_last_ts(file_path)
                stop_ts = int(END_DATE.timestamp())
                total_count = 0

                # Track the current month to trigger the progress print
                last_printed_month = datetime.fromtimestamp(current_ts, timezone.utc).month

                print(f"[*] Starting r/{sub} {mode}s...")

                while current_ts < stop_ts:
                    next_ts = min(current_ts + (CHUNK_DAYS * 86400), stop_ts)

                    url = f"https://api.pullpush.io/reddit/search/{mode}/"
                    params = {
                        "subreddit": sub,
                        "after": current_ts,
                        "before": next_ts,
                        "size": 100,
                        "sort": "asc"
                    }

                    try:
                        resp = await client.get(url, params=params)

                        if resp.status_code == 200:
                            data = resp.json().get('data', [])
                            if data:
                                with open(file_path, 'a', encoding='utf-8') as f:
                                    for item in data:
                                        clean = {
                                            "id": item.get("id"),
                                            "created_utc": item.get("created_utc"),
                                            "author": item.get("author"),
                                            "content": item.get("selftext") if mode == 'submission' else item.get("body")
                                        }
                                        f.write(json.dumps(clean) + "\n")

                                total_count += len(data)
                                current_ts = data[-1]['created_utc']
                            else:
                                current_ts = next_ts # No data, move window forward

                        elif resp.status_code == 429:
                            await asyncio.sleep(30)
                            continue
                        else:
                            await asyncio.sleep(10)
                            continue

                        # --- MONTHLY PROGRESS LOGIC ---
                        current_dt = datetime.fromtimestamp(current_ts, timezone.utc)
                        if current_dt.month != last_printed_month:
                            month_str = current_dt.strftime('%B %Y')
                            print(f"[PROGRESS] r/{sub} {mode}s: Completed {month_str} | Total: {total_count:,}")
                            last_printed_month = current_dt.month

                        # Essential delay for API stability
                        await asyncio.sleep(1.0)

                    except Exception:
                        await asyncio.sleep(5)

                print(f"[√] FINISHED r/{sub} {mode}s. Final: {total_count:,}")

# --- EXECUTION ---
await scrape_reddit()
