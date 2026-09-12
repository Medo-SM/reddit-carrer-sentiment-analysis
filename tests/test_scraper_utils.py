"""Tests for scraper utilities (src/ingest/scraper/core_scraper.py).

Importable helpers are tested directly. `import httpx` is a runtime
dependency of the scraper module, so these tests require httpx installed
(declared in notebooks/requirements.txt).
"""
import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from src.ingest.scraper.core_scraper import (
    HourlyRateLimiter,
    _state_path,
    get_last_ts_id_and_count,
    load_state,
    save_state,
)


class TestHourlyRateLimiter:
    def test_starts_immediately_under_cap(self):
        limiter = HourlyRateLimiter(max_per_hour=5)

        async def run():
            await limiter.acquire()
            assert len(limiter._timestamps) == 1

        asyncio.run(run())

    def test_blocks_when_budget_exhausted(self):
        import time

        limiter = HourlyRateLimiter(max_per_hour=1)
        # A request 1s ago fills the budget for another ~3599s.
        limiter._timestamps = [time.time() - 1]

        async def run():
            real_sleep = asyncio.sleep
            sleep_calls = 0

            async def patched_sleep(delay):
                nonlocal sleep_calls
                sleep_calls += 1
                assert delay > 0  # must not return instantly when budget is full
                # Free the budget so the next loop iteration can acquire.
                limiter._timestamps = []
                await real_sleep(0)

            asyncio.sleep = patched_sleep
            try:
                await limiter.acquire()
                assert sleep_calls == 1
            finally:
                asyncio.sleep = real_sleep

        asyncio.run(run())


class TestStateRoundTrip:
    def test_save_load_round_trip(self, tmp_path):
        path = str(tmp_path / "sub_comments.jsonl")
        save_state(path, current_ts=1600000000, last_id="abc", total_count=42)
        ts, last_id, count = load_state(path, default_ts=0)
        assert ts == 1600000000
        assert last_id == "abc"
        assert count == 42

    def test_state_path_suffix(self, tmp_path):
        assert _state_path(str(tmp_path / "x.jsonl")) == str(
            tmp_path / "x.jsonl.state.json"
        )

    def test_load_missing_sidecar_falls_back_to_file(self, tmp_path):
        jsonl = tmp_path / "data.jsonl"
        jsonl.write_text(
            '{"id": "a", "created_utc": 100}\n'
            '{"id": "b", "created_utc": 200}\n',
            encoding="utf-8",
        )
        ts, last_id, count = load_state(str(jsonl), default_ts=0)
        assert (ts, last_id, count) == (200, "b", 2)

    def test_corrupt_trailing_line_skipped(self, tmp_path):
        jsonl = tmp_path / "data.jsonl"
        jsonl.write_text(
            '{"id": "a", "created_utc": 100}\n'
            '{"id": "b", "created_utc": 200}\n'
            '{"truncated\n',
            encoding="utf-8",
        )
        ts, last_id, count = get_last_ts_id_and_count(str(jsonl), default_ts=0)
        assert (ts, last_id, count) == (200, "b", 2)

    def test_missing_file_returns_default(self, tmp_path):
        ts, last_id, count = get_last_ts_id_and_count(
            str(tmp_path / "nope.jsonl"), default_ts=123
        )
        assert (ts, last_id, count) == (123, None, 0)

    def test_corrupt_state_file_falls_back(self, tmp_path):
        jsonl = tmp_path / "data.jsonl"
        jsonl.write_text('{"id": "a", "created_utc": 100}\n', encoding="utf-8")
        state_file = _state_path(str(jsonl))
        with open(state_file, "w", encoding="utf-8") as f:
            f.write("NOT JSON")
        ts, last_id, count = load_state(str(jsonl), default_ts=0)
        assert (ts, last_id, count) == (100, "a", 1)