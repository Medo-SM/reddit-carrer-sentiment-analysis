import asyncio
import httpx
import json
import os
import time  # [RATE-LIMIT FIX]
from datetime import datetime, timezone, timedelta

# --- 1. CONFIGURATION ---
# Arctic Shift variant of core_scraper.py.
# PullPush now refuses free scraping ("not for agents"); Arctic Shift
# (https://arctic-shift.photon-reddit.com) is a free, unauthenticated
# replacement with the same {"data": [...]} response shape and the same
# fields, and it accepts epoch after/before like PullPush did.
# The scraping logic below is UNCHANGED; only the endpoint, one query param
# name (size -> limit) and the rate-limit constants differ.
SUBREDDITS = ['cscareerquestionsEU', 'datascience', 'dataengineering',
              'cybersecurity', 'SoftwareEngineering', 'cscareerquestions']
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 1, 31, tzinfo=timezone.utc)
DATA_DIR = "/content/drive/MyDrive/reddit_project"
CHUNK_DAYS = 7

# --- [RATE-LIMIT FIX] ---
# PullPush hard-caps at 100 req/hr; Arctic Shift instead rate-limits
# dynamically based on server load. We keep the same rolling-hour limiter but
# tuned to be POLITE on this free community service (~1 request / 10s). The
# scraping logic below is unchanged.
API_BASE = "https://arctic-shift.photon-reddit.com/api"
API_MODE = {"submission": "posts", "comment": "comments"}

RATE_LIMIT_PER_HOUR = 360       # polite ceiling for the free service
MIN_429_WAIT_SEC = 30           # 429 retry sleep fallback
MAX_CONSECUTIVE_429 = 3         # consecutive 429s are treated as a server hard stop
HARD_STOP_WAIT_SEC = 3600       # pause a full hour on a detected hard stop, then resume
REQUEST_DELAY_SEC = 1.0         # pacing between successful requests (was 0.5)
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
MAX_CONSECUTIVE_FAILURES = 10   # cap on consecutive non-200/non-429 responses before pausing
_request_times = []             # rolling-hour request timestamps for _wait_for_slot()

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

# [RATE-LIMIT FIX] Blocks until the rolling hourly budget has a free slot.
# It is passive while under the cap, so the scraping speed is kept; it only
# starts pacing once we approach the RATE_LIMIT_PER_HOUR ceiling.
async def _wait_for_slot():
    while True:
        now = time.time()
        # Keep only requests from the trailing 60 minutes
        _request_times[:] = [t for t in _request_times if now - t < 3600]
        if len(_request_times) < RATE_LIMIT_PER_HOUR:
            break
        await asyncio.sleep(_request_times[0] + 3600 - now)
    _request_times.append(time.time())

# --- 3. CORE SCRAPER ---
async def scrape_reddit():
    os.makedirs(DATA_DIR, exist_ok=True)
    async with httpx.AsyncClient(timeout=60.0, headers={"User-Agent": USER_AGENT}) as client:
        for sub in SUBREDDITS:
            for mode in ['submission', 'comment']:
                file_path = f"{DATA_DIR}/{sub}_{mode}s.jsonl"
                current_ts, last_id, total_count = await get_last_ts_id_and_count(file_path)
                stop_ts = int((END_DATE + timedelta(days=1)).timestamp())
                last_printed_month = datetime.fromtimestamp(current_ts, timezone.utc).month
                print(f"[*] Starting r/{sub} {mode}s...")
                
                try:
                    consecutive_429 = 0  # [RATE-LIMIT FIX] reset per file
                    consecutive_failures = 0  # [HTTP-STATUS FIX] reset per file
                    while current_ts < stop_ts:
                        next_ts = min(current_ts + (CHUNK_DAYS * 86400), stop_ts)
                        url = f"{API_BASE}/{API_MODE[mode]}/search/"
                        params = {"subreddit": sub, "after": current_ts, "before": next_ts, "limit": 100, "sort": "asc"}
                        print(f"[DEBUG] Requesting {datetime.fromtimestamp(current_ts, timezone.utc)} to {datetime.fromtimestamp(next_ts, timezone.utc)}")
                        
                        try:
                            await _wait_for_slot()  # [RATE-LIMIT FIX] enforced pacing
                            resp = await client.get(url, params=params)
                            if resp.status_code == 200:
                                consecutive_429 = 0  # [RATE-LIMIT FIX] healthy response clears the counter
                                consecutive_failures = 0  # [HTTP-STATUS FIX] healthy response clears the counter
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
                                # [RATE-LIMIT FIX] back off, wait, retry; if the
                                # server keeps 429ing, pause a full hour and
                                # auto-resume (the original "never give up" behavior).
                                consecutive_429 += 1
                                if consecutive_429 >= MAX_CONSECUTIVE_429:
                                    print(f"[!] Rate limited {consecutive_429}x \u2013 treating as hard stop; pausing {HARD_STOP_WAIT_SEC}s then resuming")
                                    consecutive_429 = 0
                                    await asyncio.sleep(HARD_STOP_WAIT_SEC)
                                else:
                                    await asyncio.sleep(MIN_429_WAIT_SEC)
                                continue
                            else:
                                # [HTTP-STATUS FIX] Any other status (403 bot-block, 500,
                                # 502/503/504, ...) used to fall through silently and
                                # re-request the same window forever. Report it, back
                                # off, and only keep retrying if the failures are
                                # intermittent; too many in a row pause a full hour.
                                consecutive_failures += 1
                                body = resp.text[:200].replace("\n", " ")
                                print(f"[!] HTTP {resp.status_code} ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}) \u2013 backing off 15s: {body}")
                                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                    print(f"[!] Too many consecutive HTTP errors \u2013 pausing {HARD_STOP_WAIT_SEC}s then resuming")
                                    consecutive_failures = 0
                                    await asyncio.sleep(HARD_STOP_WAIT_SEC)
                                else:
                                    await asyncio.sleep(15)
                                continue
                            current_dt = datetime.fromtimestamp(current_ts, timezone.utc)
                            if current_dt.month != last_printed_month:
                                print(f"[PROGRESS] r/{sub} {mode}s: Completed {current_dt.strftime('%B %Y')} | Total: {total_count:,}")
                                last_printed_month = current_dt.month
                            await asyncio.sleep(REQUEST_DELAY_SEC)
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