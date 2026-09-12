"""Tests for the extracted sampling module (src/ingest/sampling.py)."""
import json

import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingest.sampling import (
    clean_code_snippets,
    clean_text,
    get_parquet_from_path,
    json_to_df,
)

# Sample Parquet data matching the schema produced by zst_to_parquet.py
RECORDS = [
    {"id": "a1", "subreddit": "datascience", "created_utc": 1600000000, "score": 5, "text": "hello world"},
    {"id": "a2", "subreddit": "datascience", "created_utc": 1600000001, "score": 3, "text": "another one"},
    {"id": "a3", "subreddit": "datascience", "created_utc": 1600000002, "score": 1, "text": "third"},
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
        assert len(df) == 3
        assert (df["subreddit"] == "datascience").all()

    def test_invalid_dataset_type(self, tmp_path):
        df = json_to_df(tmp_path, "datascience", "bogus")
        assert df is None

    def test_missing_file_returns_none(self, tmp_path):
        df = json_to_df(tmp_path, "nonexistent", "submissions")
        assert df is None