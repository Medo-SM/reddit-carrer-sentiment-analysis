"""Tests for the sampling module (src/ingest/sampling.py)."""
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.ingest.sampling import (
    get_parquet_from_path,
    hash_author,
    json_to_df,
    stratified_sample_parquet_to_json,
)

# Sample Parquet data matching the schema produced by zst_to_parquet.py
RECORDS = [
    {"id": "a1", "subreddit": "datascience", "created_utc": 1600000000, "score": 5, "text": "hello world today", "author": "alice"},
    {"id": "a2", "subreddit": "datascience", "created_utc": 1600000001, "score": 3, "text": "another great one", "author": "[deleted]"},
    {"id": "a3", "subreddit": "datascience", "created_utc": 1600000002, "score": 1, "text": "third post here", "author": "bob"},
    {"id": "a0", "subreddit": "datascience", "created_utc": 1400000000, "score": 2, "text": "too old 2014", "author": "old"},
]


@pytest.fixture()
def parquet_dir(tmp_path):
    table = pa.Table.from_pylist(RECORDS)
    pq.write_table(table, tmp_path / "datascience_submissions.parquet")
    return tmp_path


class TestGetParquetFromPath:
    def test_finds_parquet_files(self, parquet_dir):
        files = get_parquet_from_path(parquet_dir)
        assert len(files) == 1
        assert files[0].endswith("datascience_submissions.parquet")

    def test_ignores_non_parquet(self, parquet_dir):
        (parquet_dir / "readme.txt").write_text("hello")
        files = get_parquet_from_path(parquet_dir)
        assert len(files) == 1

    def test_recursive_search(self, parquet_dir):
        nested = parquet_dir / "nested"
        nested.mkdir()
        table = pa.Table.from_pylist(RECORDS[:1])
        pq.write_table(table, nested / "extra.parquet")
        files = get_parquet_from_path(parquet_dir)
        assert len(files) == 2


class TestJsonToDf:
    def test_loads_single_subreddit(self, tmp_path):
        (tmp_path / "datascience_submissions.json").write_text(
            json.dumps(RECORDS), encoding="utf-8"
        )
        df = json_to_df(tmp_path, "datascience", "submissions")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert (df["subreddit"] == "datascience").all()

    def test_invalid_dataset_type(self, tmp_path):
        df = json_to_df(tmp_path, "datascience", "bogus")
        assert df is None

    def test_missing_file_returns_none(self, tmp_path):
        df = json_to_df(tmp_path, "nonexistent", "submissions")
        assert df is None


class TestHashAndStratifiedSample:
    def test_hash_author_salted_and_deterministic(self):
        h1 = hash_author("alice")
        h2 = hash_author("alice")
        assert h1 == h2
        assert len(h1) == 16
        assert h1 != "alice"
        assert hash_author("[deleted]") == "DELETED"
        assert hash_author(None) == "DELETED"

    def test_stratified_sample_filters_window_and_hashes_author(self, parquet_dir, tmp_path):
        files = get_parquet_from_path(parquet_dir)
        out_dir = tmp_path / "samples"
        count = stratified_sample_parquet_to_json(
            files,
            out_dir,
            desired_samples=10,
            seed=42,
            start_date="2018-01-01",
            end_date="2022-12-31",
        )
        assert count == 1
        rows = json.loads((out_dir / "datascience_submissions.json").read_text(encoding="utf-8"))
        ids = {r["id"] for r in rows}
        assert ids == {"a1", "a2", "a3"}  # a0 from 2014 excluded
        assert all("author" not in r and "author_hash" in r for r in rows)
