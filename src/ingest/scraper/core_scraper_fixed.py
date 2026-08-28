import asyncio
import httpx
import json
import os
import time
from datetime import datetime, timezone, timedelta

# --- 1. CONFIGURATION ---
# Fixed copy of core_scraper.py. Changes:
#   * Strict monotonic cursor (created_utc, id) dedup + guaranteed progress
#     (fixes the infinite loop on same-second windows / repeated page writes).
#   * Robust resume: skips truncated/corrupt trailing lines instead of crashing.
#   * Targeted exception handling + consecutive-failure abort (no endless retry).
#   * Hybrid entrypoint: works as a plain script AND when run inside an active
#     event loop (Colab/Jupyter), via `run_main()` / `await main()`.
#   * PullPush rate-limit compliance (strict 100 req/hr):
#       - HourlyRateLimiter paces requests to RATE_LIMIT_PER_HOUR.
#       - 429 responses honour Retry-After; MAX_CONSECUTIVE_429 hard stops
#         the whole run with a resume hint (no endless 429 spinning).
#   * Crash-safe resume: sidecar <file>.state.json (atomic write) persisted
#     after every window, so any interrupt/kill resumes exactly where it left.
#   * Graceful cancellation: CancelledError + KeyboardInterrupt both persist
#     state and print a clear summary.
SUBREDDITS = ['cscareerquestionsEU', 'datascience', 'dataengineering',
              'cybersecurity', 'SoftwareEngineering', 'cscareerquestions']
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 1, 31, tzinfo=timezone.utc)
DATA_DIR = "/content/drive/MyDrive/reddit_project"
CHUNK_DAYS = 7
MAX_CONSECUTIVE_FAILURES = 10
RATE_LIMIT_PER_HOUR = 90          # safety margin under PullPush's 100/hr hard cap
MIN_429_WAIT_SEC = 60             # fallback wait when no Retry-After header
MAX_CONSECUTIVE_429 = 3           # > this = hard stop -> stop the whole run
STATE_FILE_SUFFIX = ".state.json"

# --- 2. RATE LIMITER ---
class RateLimitHardStop(Exception):
    """Raised after MAX_CONSECUTIVE_429 hard stops; aborts the whole run."""

    def __init__(self, subreddit, mode):
        super().__init__(f"r/{subreddit} {mode}s")
        self.subreddit = subreddit
        self.mode = mode


class HourlyRateLimiter:
    """Enforces a strict rolling-hour request budget (PullPush: 100 req/hr)."""

    def __init__(self, max_per_hour):
        self.max_per_hour = max_per_hour
        self._timestamps = []

    async def acquire(self):
        """Block until an hourly request slot is free, then consume it."""
        while True:
            now = time.time()
            self._timestamps = [t for t in self._timestamps if now - t < 3600.0]
            if len(self._timestamps) < self.max_per_hour:
                break
            wait = self._timestamps[0] + 3600.0 - now
            print(f"[RATE] Hourly budget exhausted ({self.max_per_hour}/hr) – waiting {wait/60:.1f} min")
            await asyncio.sleep(wait)
        self._timestamps.append(time.time())


# --- 3. RESUME STATE ---
def _state_path(file_path):
    return f"{file_path}{STATE_FILE_SUFFIX}"


async def load_state(file_path):
    """Resume cursor from the sidecar file, else derive it from the JSONL tail."""
    state_file = _state_path(file_path)
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state["current_ts"], state.get("last_id"), state.get("total_count", 0)
        except (json.JSONDecodeError, KeyError, OSError):
            print(f"[!] Corrupt state file {state_file}; falling back to scanning the data file.")
    return await get_last_ts_id_and_count(file_path)


def save_state(file_path, current_ts, last_id, total_count):
    """Atomically persist the resume cursor (tmp file + os.replace)."""
    state_file = _state_path(file_path)
    tmp_file = f"{state_file}.tmp"
    payload = {
        "current_ts": current_ts,
        "last_id": last_id,
        "total_count": total_count,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_file, state_file)


# --- 4. HELPERS ---
async def get_last_ts_id_and_count(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return int(START_DATE.timestamp()), None, 0
    line_count = 0
    last_item = None
    with open(path, 'rb') as f:
        for line in f:
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    continue
            except json.JSONDecodeError:
                # Skip truncated/corrupt trailing lines; resume from previous valid record.
                continue
            line_count += 1
            last_item = item
    if last_item is None:
        return int(START_DATE.timestamp()), None, 0
    return last_item['created_utc'], last_item['id'], line_count


# --- 5. CORE SCRAPER ---
async def scrape_reddit():
    os.makedirs(DATA_DIR, exist_ok=True)
    limiter = HourlyRateLimiter(RATE_LIMIT_PER_HOUR)
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            for sub in SUBREDDITS:
                for mode in ['submission', 'comment']:
                    file_path = f"{DATA_DIR}/{sub}_{mode}s.jsonl"
                    current_ts, last_id, total_count = await load_state(file_path)
                    stop_ts = int((END_DATE + timedelta(days=1)).timestamp())
                    last_printed_month = datetime.fromtimestamp(current_ts, timezone.utc).month
                    print(f"[*] Starting r/{sub} {mode}s...")
                    print(f"[*>] Resuming at {datetime.fromtimestamp(current_ts, timezone.utc):%Y-%m-%d %H:%M:%S} UTC | records so far: {total_count:,}")

                    consecutive_failures = 0
                    consecutive_429 = 0

                    while current_ts < stop_ts:
                        next_ts = min(current_ts + (CHUNK_DAYS * 86400), stop_ts)
                        url = f"https://api.pullpush.io/reddit/search/{mode}/"
                        params = {"subreddit": sub, "after": current_ts, "before": next_ts, "size": 100, "sort": "asc"}
                        print(f"[DEBUG] Requesting {datetime.fromtimestamp(current_ts, timezone.utc)} to {datetime.fromtimestamp(next_ts, timezone.utc)}")

                        try:
                            await limiter.acquire()
                            resp = await client.get(url, params=params)

                            if resp.status_code == 429:
                                consecutive_429 += 1
                                retry_after = resp.headers.get("Retry-After")
                                try:
                                    retry_after = int(retry_after)
                                except (TypeError, ValueError):
                                    retry_after = None
                                wait = max(MIN_429_WAIT_SEC, retry_after or 0)
                                print(f"[!] Rate limited (429) {consecutive_429}/{MAX_CONSECUTIVE_429} – waiting {wait}s")
                                if consecutive_429 >= MAX_CONSECUTIVE_429:
                                    raise RateLimitHardStop(sub, mode)
                                await asyncio.sleep(wait)
                                continue

                            resp.raise_for_status()
                            data = resp.json().get('data', [])
                            consecutive_failures = 0
                            consecutive_429 = 0

                            if not data:
                                current_ts = next_ts
                                last_id = None
                                continue

                            # Strict monotonic cursor: only rows strictly newer than
                            # (created_utc, id) are written, so same-second rows are
                            # never duplicated and the window always makes progress.
                            # `""` is used as the id sentinel to keep tuple comparisons
                            # valid even before any id has been recorded.
                            cursor = (current_ts, last_id or "")
                            new_items = [
                                item for item in data
                                if (item.get('created_utc'), item.get('id') or " ") > cursor
                            ]

                            if not new_items:
                                # Nothing strictly new in this window -> skip past it.
                                current_ts = next_ts
                                last_id = None
                                continue

                            with open(file_path, 'a', encoding='utf-8') as f:
                                for item in new_items:
                                    if mode == 'comment':
                                        clean = {"id": item.get("id"), "created_utc": item.get("created_utc"), "author": item.get("author"), "content": item.get("body"), "score": item.get("score"), "link_id": item.get("link_id"), "parent_id": item.get("parent_id")}
                                    else:
                                        clean = {"id": item.get("id"), "created_utc": item.get("created_utc"), "author": item.get("author"), "content": item.get("title"), "score": item.get("score")}
                                    f.write(json.dumps(clean) + "\n")
                                f.flush()

                            total_count += len(new_items)
                            last_ts = new_items[-1]['created_utc']
                            last_id = new_items[-1]['id']

                            if last_ts > current_ts:
                                # Normal advance to the last-written row.
                                current_ts = last_ts
                            else:
                                # Stall guard: all returned rows share `current_ts`
                                # (same-second batch); force progress to the next window.
                                print(f"[PROGRESS] All rows at same timestamp {datetime.fromtimestamp(current_ts, timezone.utc)}; advancing window.")
                                current_ts = next_ts
                                last_id = None

                            current_dt = datetime.fromtimestamp(current_ts, timezone.utc)
                            if current_dt.month != last_printed_month:
                                print(f"[PROGRESS] r/{sub} {mode}s: Completed {current_dt.strftime('%B %Y')} | Total: {total_count:,}")
                                last_printed_month = current_dt.month

                        except asyncio.CancelledError:
                            raise
                        except RateLimitHardStop:
                            raise
                        except (httpx.HTTPError, httpx.TransportError) as e:
                            consecutive_failures += 1
                            print(f"[!] HTTP error: {e} – retrying in 5s ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                            await asyncio.sleep(5)
                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                print(f"[!] Aborting r/{sub} {mode}s after {MAX_CONSECUTIVE_FAILURES} consecutive failures.")
                                break
                        except Exception as e:
                            consecutive_failures += 1
                            print(f"[!] Unexpected error: {e} – retrying in 5s ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                            await asyncio.sleep(5)
                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                print(f"[!] Aborting r/{sub} {mode}s after {MAX_CONSECUTIVE_FAILURES} consecutive failures.")
                                break
                        finally:
                            # Persist the cursor after every iteration so any
                            # interrupt/kill resumes exactly where we stopped.
                            try:
                                save_state(file_path, current_ts, last_id, total_count)
                            except Exception as e:
                                print(f"[!] Failed to save state for {file_path}: {e}")

                    print(f"[\u221a] FINISHED r/{sub} {mode}s.")
        except asyncio.CancelledError:
            print("[!] Scraper cancelled gracefully – state was saved; rerun to resume exactly where it stopped.")
        except RateLimitHardStop as e:
            print(f"[!] Rate-limit hard stop detected ({e}). State was saved; rerun later to resume exactly where it stopped.")


async def main():
    await scrape_reddit()


def run_main():
    """Run the scraper both as a standalone script and inside a running event
    loop (Colab/Jupyter), where ``asyncio.run()`` is not allowed."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No active loop -> plain script behaviour.
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("[!] Interrupted (Ctrl-C) – state was saved; rerun to resume exactly where it stopped.")
        return

    # An active loop already exists (notebook/kernel): patch it so that
    # asyncio.run() can be nested, or fall back to `await main()`.
    try:
        import nest_asyncio
    except ImportError:
        raise RuntimeError(
            "Running inside an active event loop (Jupyter/Colab). "
            "Either call `await main()` in a cell, or `pip install nest_asyncio`."
        )
    nest_asyncio.apply()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[!] Interrupted (Ctrl-C) – state was saved; rerun to resume exactly where it stopped.")


if __name__ == "__main__":
    run_main()