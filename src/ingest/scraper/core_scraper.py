#!/usr/bin/env python3
"""Canonical Reddit scraper — merges logic from core_scraper_fixed.py and
core_scraper_arctic.py into a single, config-driven, CLI-capable module.

Features:
  - Arctic Shift API endpoint (free, unauthenticated)
  - Strict monotonic (created_utc, id) cursor — no duplicate writes
  - Crash-safe sidecar .state.json (atomic write via os.replace)
  - Corrupt-trailing-line tolerant resume
  - HourlyRateLimiter with RateLimitHardStop
  - Stall guard (forces window advance on same-second batches)
  - Non-200/429 HTTP-status handling with backoff + failure cap
  - CLI via argparse; config.yaml values are defaults
  - Python logging instead of print()
  - Works as a plain script AND inside Jupyter/Colab (run_main + nest_asyncio)

Usage:
    python -m src.ingest.scraper.core_scraper
    python -m src.ingest.scraper.core_scraper --subreddit cscareerquestions --mode posts
    python -m src.ingest.scraper.core_scraper --log-level DEBUG
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("scraper")

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config, with CONFIG_PATH env override."""
    path = Path(
        config_path
        or os.environ.get("CONFIG_PATH")
        or _DEFAULT_CONFIG_PATH
    )
    if not path.exists():
        logger.warning("Config file %s not found; using built-in defaults.", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _scraper_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract the scraper sub-config with defaults."""
    defaults = {
        "api_base": "https://arctic-shift.photon-reddit.com/api",
        "data_dir": "data/raw/reddit/subreddits",
        "subreddits": [
            "cscareerquestionsEU", "datascience", "dataengineering",
            "cybersecurity", "SoftwareEngineering", "cscareerquestions",
        ],
        "start_date": "2023-01-01",
        "end_date": "2025-01-31",
        "chunk_days": 7,
        "rate_limit_per_hour":120, 
        "min_429_wait_sec": 30,
        "max_consecutive_429": 3,
        "hard_stop_wait_sec": 3600,
        "request_delay_sec": 5.0,
        "user_agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "max_consecutive_failures": 10,
    }
    merged = {**defaults, **cfg.get("scraper", {})}
    return merged


# ---------------------------------------------------------------------------
# API mode mapping
# ---------------------------------------------------------------------------
API_MODE: dict[str, str] = {"submission": "posts", "comment": "comments"}
STATE_FILE_SUFFIX = ".state.json"

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimitHardStop(Exception):
    """Raised after too many consecutive 429s; aborts the run."""

    def __init__(self, subreddit: str, mode: str) -> None:
        super().__init__(f"r/{subreddit} {mode}s")
        self.subreddit = subreddit
        self.mode = mode


class HourlyRateLimiter:
    """Enforces a strict rolling-hour request budget."""

    def __init__(self, max_per_hour: int) -> None:
        self.max_per_hour = max_per_hour
        self._timestamps: list[float] = []

    async def acquire(self) -> None:
        """Block until an hourly request slot is free, then consume it."""
        while True:
            now = time.time()
            self._timestamps = [t for t in self._timestamps if now - t < 3600.0]
            if len(self._timestamps) < self.max_per_hour:
                break
            wait = self._timestamps[0] + 3600.0 - now
            logger.info(
                "Hourly budget exhausted (%d/hr) — waiting %.1f min",
                self.max_per_hour, wait / 60,
            )
            await asyncio.sleep(wait)
        self._timestamps.append(time.time())


# ---------------------------------------------------------------------------
# Resume state (crash-safe sidecar)
# ---------------------------------------------------------------------------

def _state_path(file_path: str) -> str:
    return f"{file_path}{STATE_FILE_SUFFIX}"


def load_state(
    file_path: str, default_ts: int
) -> tuple[int, str | None, int]:
    """Resume cursor from sidecar, else derive from JSONL tail."""
    state_file = _state_path(file_path)
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            return int(state["current_ts"]), state.get("last_id"), state.get("total_count", 0)
        except (json.JSONDecodeError, KeyError, OSError):
            logger.warning(
                "Corrupt state file %s; falling back to scanning the data file.",
                state_file,
            )
    return get_last_ts_id_and_count(file_path, default_ts)


def save_state(
    file_path: str,
    current_ts: int,
    last_id: str | None,
    total_count: int,
) -> None:
    """Atomically persist the resume cursor (tmp + os.replace)."""
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


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def get_last_ts_id_and_count(
    path: str, default_ts: int
) -> tuple[int, str | None, int]:
    """Read (created_utc, id, line_count) from the last valid line of a JSONL.

    Tolerates corrupt/truncated trailing lines by skipping them.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default_ts, None, 0
    line_count = 0
    last_item: dict[str, Any] | None = None
    with open(path, "rb") as f:
        for line in f:
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    continue
            except json.JSONDecodeError:
                continue
            line_count += 1
            last_item = item
    if last_item is None:
        return default_ts, None, 0
    # Normalize to int: JSON floats (e.g. 1738367876.0) are rejected by the
    # Arctic Shift API as an invalid after/before date.
    return int(last_item["created_utc"]), last_item["id"], line_count


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

async def scrape_reddit(scfg: dict[str, Any]) -> None:
    """Run the scraper for all configured subreddits and modes."""
    data_dir = scfg["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    start_date = datetime.fromisoformat(scfg["start_date"]).replace(tzinfo=timezone.utc)
    end_date = datetime.fromisoformat(scfg["end_date"]).replace(tzinfo=timezone.utc)
    default_ts = int(start_date.timestamp())
    stop_ts = int((end_date + timedelta(days=1)).timestamp())
    chunk_seconds = scfg["chunk_days"] * 86400

    limiter = HourlyRateLimiter(scfg["rate_limit_per_hour"])
    headers = {"User-Agent": scfg["user_agent"]}

    async with httpx.AsyncClient(timeout=60.0, headers=headers) as client:
        try:
            for sub in scfg["subreddits"]:
                for mode in ["submission", "comment"]:
                    file_path = os.path.join(data_dir, f"{sub}_{mode}s.jsonl")
                    current_ts, last_id, total_count = load_state(file_path, default_ts)
                    last_printed_month = datetime.fromtimestamp(
                        current_ts, timezone.utc
                    ).month

                    logger.info(
                        "Starting r/%s %ss — resuming at %s UTC | records: %s",
                        sub, mode,
                        datetime.fromtimestamp(current_ts, timezone.utc).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        f"{total_count:,}",
                    )

                    consecutive_failures = 0
                    consecutive_429 = 0

                    while current_ts < stop_ts:
                        next_ts = min(current_ts + chunk_seconds, stop_ts)
                        url = f"{scfg['api_base']}/{API_MODE[mode]}/search/"
                        params = {
                            "subreddit": sub,
                            "after": int(current_ts),
                            "before": int(next_ts),
                            "limit": 100,
                            "sort": "asc",
                        }
                        logger.debug(
                            "Requesting %s to %s",
                            datetime.fromtimestamp(current_ts, timezone.utc),
                            datetime.fromtimestamp(next_ts, timezone.utc),
                        )

                        try:
                            await limiter.acquire()
                            resp = await client.get(url, params=params)

                            # --- 429 handling ---
                            if resp.status_code == 429:
                                consecutive_429 += 1
                                retry_after: int | None = None
                                try:
                                    retry_after = int(
                                        resp.headers.get("Retry-After", "")
                                    )
                                except (TypeError, ValueError):
                                    pass
                                wait = max(
                                    scfg["min_429_wait_sec"], retry_after or 0
                                )
                                logger.warning(
                                    "Rate limited (429) %d/%d — waiting %ds",
                                    consecutive_429,
                                    scfg["max_consecutive_429"],
                                    wait,
                                )
                                if consecutive_429 >= scfg["max_consecutive_429"]:
                                    raise RateLimitHardStop(sub, mode)
                                await asyncio.sleep(wait)
                                continue

                            # --- non-200/non-429 handling ---
                            if resp.status_code != 200:
                                consecutive_failures += 1
                                body = resp.text[:200].replace("\n", " ")
                                logger.warning(
                                    "HTTP %d (%d/%d) — backing off 15s: %s",
                                    resp.status_code,
                                    consecutive_failures,
                                    scfg["max_consecutive_failures"],
                                    body,
                                )
                                if consecutive_failures >= scfg["max_consecutive_failures"]:
                                    logger.warning(
                                        "Too many consecutive HTTP errors — "
                                        "pausing %ds then resuming",
                                        scfg["hard_stop_wait_sec"],
                                    )
                                    consecutive_failures = 0
                                    await asyncio.sleep(scfg["hard_stop_wait_sec"])
                                else:
                                    await asyncio.sleep(15)
                                continue

                            # --- 200 OK ---
                            data = resp.json().get("data", [])
                            consecutive_failures = 0
                            consecutive_429 = 0

                            if not data:
                                current_ts = next_ts
                                last_id = None
                                continue

                            # Strict monotonic cursor: only rows strictly newer
                            # than (created_utc, id) are written.
                            cursor = (current_ts, last_id or "")
                            new_items = [
                                item
                                for item in data
                                if (
                                    item.get("created_utc"),
                                    item.get("id") or " ",
                                )
                                > cursor
                            ]

                            if not new_items:
                                current_ts = next_ts
                                last_id = None
                                continue

                            with open(file_path, "a", encoding="utf-8") as f:
                                for item in new_items:
                                    if mode == "comment":
                                        clean = {
                                            "id": item.get("id"),
                                            "created_utc": int(item.get("created_utc")),
                                            "author": item.get("author"),
                                            "content": item.get("body"),
                                            "score": item.get("score"),
                                            "link_id": item.get("link_id"),
                                            "parent_id": item.get("parent_id"),
                                        }
                                    else:
                                        clean = {
                                            "id": item.get("id"),
                                            "created_utc": int(item.get("created_utc")),
                                            "author": item.get("author"),
                                            "content": item.get("title"),
                                            "score": item.get("score"),
                                        }
                                    f.write(json.dumps(clean) + "\n")
                                f.flush()

                            total_count += len(new_items)
                            last_ts = int(new_items[-1]["created_utc"])
                            last_id = new_items[-1]["id"]

                            if last_ts > current_ts:
                                current_ts = last_ts
                            else:
                                # Stall guard: all rows share current_ts
                                logger.info(
                                    "All rows at same timestamp %s; advancing window.",
                                    datetime.fromtimestamp(
                                        current_ts, timezone.utc
                                    ),
                                )
                                current_ts = next_ts
                                last_id = None

                            current_dt = datetime.fromtimestamp(
                                current_ts, timezone.utc
                            )
                            if current_dt.month != last_printed_month:
                                logger.info(
                                    "r/%s %ss: Completed %s | Total: %s",
                                    sub, mode,
                                    current_dt.strftime("%B %Y"),
                                    f"{total_count:,}",
                                )
                                last_printed_month = current_dt.month

                            await asyncio.sleep(scfg["request_delay_sec"])

                        except asyncio.CancelledError:
                            raise
                        except RateLimitHardStop:
                            raise
                        except (httpx.HTTPError, httpx.TransportError) as e:
                            consecutive_failures += 1
                            logger.warning(
                                "HTTP error: %s — retrying in 5s (%d/%d)",
                                e, consecutive_failures,
                                scfg["max_consecutive_failures"],
                            )
                            await asyncio.sleep(5)
                            if consecutive_failures >= scfg["max_consecutive_failures"]:
                                logger.error(
                                    "Aborting r/%s %ss after %d consecutive failures.",
                                    sub, mode,
                                    scfg["max_consecutive_failures"],
                                )
                                break
                        except Exception as e:
                            consecutive_failures += 1
                            logger.warning(
                                "Unexpected error: %s — retrying in 5s (%d/%d)",
                                e, consecutive_failures,
                                scfg["max_consecutive_failures"],
                            )
                            await asyncio.sleep(5)
                            if consecutive_failures >= scfg["max_consecutive_failures"]:
                                logger.error(
                                    "Aborting r/%s %ss after %d consecutive failures.",
                                    sub, mode,
                                    scfg["max_consecutive_failures"],
                                )
                                break
                        finally:
                            try:
                                save_state(file_path, current_ts, last_id, total_count)
                            except Exception as e:
                                logger.error("Failed to save state for %s: %s", file_path, e)

                    logger.info("FINISHED r/%s %ss.", sub, mode)

        except asyncio.CancelledError:
            logger.info(
                "Scraper cancelled gracefully — state was saved; "
                "rerun to resume."
            )
        except RateLimitHardStop as e:
            logger.warning(
                "Rate-limit hard stop (%s). State was saved; rerun later to resume.",
                e,
            )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def main(scfg: dict[str, Any] | None = None) -> None:
    if scfg is None:
        scfg = _scraper_cfg(load_config())
    await scrape_reddit(scfg)


def run_main(scfg: dict[str, Any] | None = None) -> None:
    """Run the scraper as a standalone script or inside Jupyter/Colab."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No active loop — plain script.
        try:
            asyncio.run(main(scfg))
        except KeyboardInterrupt:
            logger.info(
                "Interrupted (Ctrl-C) — state was saved; rerun to resume."
            )
        return

    # Active loop exists (notebook/kernel): patch with nest_asyncio.
    try:
        import nest_asyncio
    except ImportError:
        raise RuntimeError(
            "Running inside an active event loop (Jupyter/Colab). "
            "Either call `await main()` in a cell, or `pip install nest_asyncio`."
        )
    nest_asyncio.apply()
    try:
        asyncio.run(main(scfg))
    except KeyboardInterrupt:
        logger.info(
            "Interrupted (Ctrl-C) — state was saved; rerun to resume."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Reddit via the Arctic Shift API.",
    )
    parser.add_argument(
        "--subreddit", "-s",
        nargs="+",
        help="Override subreddit list from config.",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["posts", "comments", "both"],
        default="both",
        help="Scrape posts, comments, or both (default: both).",
    )
    parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD), overrides config.",
    )
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD), overrides config.",
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml (default: config/config.yaml or CONFIG_PATH env).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO or LOG_LEVEL env).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config(args.config)
    scfg = _scraper_cfg(cfg)

    # CLI overrides
    if args.subreddit:
        scfg["subreddits"] = args.subreddit
    if args.start:
        scfg["start_date"] = args.start
    if args.end:
        scfg["end_date"] = args.end

    run_main(scfg)
