import asyncio
import httpx
import json
import os
from datetime import datetime, timezone

# --- 1. CONFIGURATION ---
SUBREDDITS = ['datascience', 'cscareerquestions']
START_DATE = datetime(2019, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 31, tzinfo=timezone.utc)
DATA_DIR = "/content/drive/MyDrive/reddit_project"
CHUNK_DAYS = 7  

# --- 2. HELPERS ---
async def get_last_ts_and_count(path):
    """Instant Resume: Jumps to the end and counts total lines quickly."""
    if not os.path.exists(path) or os.path.getsize(path) == 0: 
        return int(START_DATE.timestamp()), 0
    
    line_count = 0
    # We still do a quick line count so your 'Total' display is accurate
    with open(path, 'rb') as f:
        for line in f:
            line_count += 1
        # After loop, 'line' is the last line
        last_ts = json.loads(line)['created_utc']
    
    return last_ts, line_count

# --- 3. CORE SCRAPER ---

async def scrape_reddit():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for sub in SUBREDDITS:
            for mode in ['submission','comment']:
                file_path = f"{DATA_DIR}/{sub}_{mode}s.jsonl"
                
                # Resumes with the correct timestamp and the actual total from the file
                current_ts, total_count = await get_last_ts_and_count(file_path)
                
                stop_ts = int(END_DATE.timestamp())
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
                                            "content": item.get("body")
                                        }
                                        f.write(json.dumps(clean) + "\n")
                                
                                total_count += len(data)
                                current_ts = data[-1]['created_utc']
                            else:
                                current_ts = next_ts 
                        
                        elif resp.status_code == 429:
                            await asyncio.sleep(30)
                            continue
                        
                        # --- OLD FORMAT PROGRESS LOG ---
                        current_dt = datetime.fromtimestamp(current_ts, timezone.utc)
                        if current_dt.month != last_printed_month:
                            # Matches your request: Month Name Year | Total: X
                            month_str = current_dt.strftime('%B %Y')
                            print(f"[PROGRESS] r/{sub} {mode}s: Completed {month_str} | Total: {total_count:,}")
                            last_printed_month = current_dt.month

                        await asyncio.sleep(0.5) # Balanced speed

                    except Exception:
                        await asyncio.sleep(5)
                
                print(f"[√] FINISHED r/{sub} {mode}s.")

# --- EXECUTION ---
await scrape_reddit()