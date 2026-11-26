"""Shared sampling and text-cleaning utilities for ingestion and modeling.

Provides importable utilities for parquet discovery, random and stratified
sampling, JSON loading, privacy hashing, and dual text cleaning:
- Sentiment-safe cleaning (`clean_sentiment_text`, `SentimentTextCleaner`):
  removes structural platform artifacts (HTML entities, URLs, Markdown syntax,
  bot/moderator noise) while preserving casing, negations, emojis, and
  expressive punctuation (`!`, `?`).
- Topic-modeling cleaning (`clean_code_snippets`, `clean_text`, `process_pipeline`):
  removes fenced/inline code blocks, URLs, and non-word symbols (`\\W+`) for
  concise BERTopic c-TF-IDF representation.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


# ---------------------------------------------------------------------------
# Parquet discovery
# ---------------------------------------------------------------------------

def get_parquet_from_path(path: str | Path) -> list[str]:
    """Recursively find all .parquet files under *path* (sorted deterministically)."""
    parquet_files: list[str] = []
    for dir_path, _, files in os.walk(path):
        for f in sorted(files):
            if f.endswith(".parquet"):
                parquet_files.append(os.path.join(dir_path, f))
    return sorted(parquet_files)


# ---------------------------------------------------------------------------
# Privacy & noise helpers
# ---------------------------------------------------------------------------

SALT = "reddit_career_sentiment_2026"

_NOISE_EXACT = {"", "[deleted]", "[removed]"}


def hash_author(author: str | None, salt: str = SALT) -> str:
    """One-way salted SHA-256 hash of a Reddit username (first 16 hex chars)."""
    if not isinstance(author, str) or author in _NOISE_EXACT:
        return "DELETED"
    raw = f"{salt}:{author}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def is_noise_text(text: Any) -> bool:
    """Return True if *text* is null, deleted/removed stub, or bot boilerplate."""
    if not isinstance(text, str):
        return True
    stripped = text.strip()
    if stripped in _NOISE_EXACT:
        return True
    if "I am a bot" in stripped or "[removed by moderator]" in stripped:
        return True
    return False


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_parquet_file_to_json(
    file_paths: list[str],
    output_dir: str | Path,
    desired_samples: int = 30_000,
) -> int:
    """Randomly sample rows from each Parquet file and write to JSON."""
    if pq is None:
        raise ImportError("pyarrow is required to read Parquet files.")
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    file_count = 0
    for fp in file_paths:
        file_name = os.path.splitext(os.path.basename(fp))[0]
        metadata = pq.read_metadata(fp)
        total_samples = metadata.num_rows
        table = pq.read_table(fp)

        if total_samples <= desired_samples:
            selected_samples = table.to_pylist()
        else:
            random_indices = sorted(random.sample(range(total_samples), desired_samples))
            selected_samples = table.take(random_indices).to_pylist()

        out_path = output_dir / f"{file_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(selected_samples, f, ensure_ascii=False, indent=4, default=str)

        print(
            f"Random sampling completed for {file_name}, "
            f"retained {len(selected_samples)} samples."
        )
        file_count += 1

    print(f"Done! Sampled {file_count} files")
    return file_count


def _to_utc_datetime(ts_series: pd.Series) -> pd.Series:
    """Parse numeric epoch seconds or string/datetime timestamps into UTC."""
    if pd.api.types.is_numeric_dtype(ts_series.dtype):
        return pd.to_datetime(ts_series, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(ts_series, utc=True, errors="coerce")


def stratified_sample_parquet_to_json(
    file_paths: list[str],
    output_dir: str | Path,
    desired_samples: int = 30_000,
    seed: int = 42,
    start_date: str = "2018-01-01",
    end_date: str = "2022-12-31",
) -> int:
    """Stratify by ISO week of created_utc within [start_date, end_date], then sample proportionally.

    Streams row groups via PyArrow so multi-million-row Parquet files remain
    memory-safe on constrained hosts.
    """
    if pq is None:
        raise ImportError("pyarrow is required to read Parquet files.")
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    file_count = 0
    for fp in file_paths:
        file_name = os.path.splitext(os.path.basename(fp))[0]
        pf = pq.ParquetFile(fp)
        total_samples = pf.metadata.num_rows

        if "created_utc" not in pf.schema.names:
            table = pf.read()
            selected_samples = table.to_pylist()
        else:
            ts_series = pf.read(["created_utc"])["created_utc"].to_pandas()
            dt_series = _to_utc_datetime(ts_series)
            in_window = (dt_series >= start_ts) & (dt_series <= end_ts)
            valid_indices = np.where(in_window)[0]

            if len(valid_indices) == 0:
                print(f"Warning: {file_name} had 0 rows in [{start_date}, {end_date}]; skipping.")
                continue

            if len(valid_indices) <= desired_samples:
                indices_arr = np.sort(valid_indices.astype(np.int64))
            else:
                dt_valid = dt_series.iloc[valid_indices]
                # Drop tz before Period conversion to avoid pandas UserWarning
                week_bins = dt_valid.dt.tz_convert(None).dt.to_period("W")
                indices_by_group: dict[Any, list[int]] = {}
                for orig_idx, wk in zip(valid_indices, week_bins):
                    indices_by_group.setdefault(wk, []).append(int(orig_idx))

                selected_indices: list[int] = []
                n_pool = len(valid_indices)
                for idx_list in indices_by_group.values():
                    group_len = len(idx_list)
                    prop_n = math.ceil((group_len / n_pool) * desired_samples)
                    n_take = min(group_len, prop_n)
                    if n_take > 0:
                        chosen = rng.choice(idx_list, size=n_take, replace=False)
                        selected_indices.extend(chosen.tolist())

                if len(selected_indices) > desired_samples:
                    selected_indices = rng.choice(
                        selected_indices, size=desired_samples, replace=False
                    ).tolist()

                indices_arr = np.sort(np.array(selected_indices, dtype=np.int64))

            # Stream row groups to materialize only selected rows
            selected_samples = []
            current_row = 0
            for rg_idx in range(pf.num_row_groups):
                rg_len = pf.metadata.row_group(rg_idx).num_rows
                rg_start = current_row
                rg_end = current_row + rg_len

                mask = (indices_arr >= rg_start) & (indices_arr < rg_end)
                rg_indices = indices_arr[mask] - rg_start

                if len(rg_indices) > 0:
                    rg_table = pf.read_row_group(rg_idx)
                    selected_samples.extend(rg_table.take(rg_indices).to_pylist())

                current_row = rg_end

        clean_samples = []
        seen_ids = set()
        for rec in selected_samples:
            rec_id = rec.get("id")
            if rec_id is not None:
                if rec_id in seen_ids:
                    continue
                seen_ids.add(rec_id)

            text_val = rec.get("text", rec.get("selftext", rec.get("body", "")))
            if is_noise_text(text_val) or len(text_val.split()) < 3:
                continue

            if "author" in rec:
                rec["author_hash"] = hash_author(rec["author"])
                del rec["author"]
            else:
                rec.setdefault("author_hash", "UNKNOWN")

            clean_samples.append(rec)

        out_path = output_dir / f"{file_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clean_samples, f, ensure_ascii=False, indent=2, default=str)

        print(
            f"Stratified sampling completed for {file_name}: "
            f"retained {len(clean_samples):,} clean samples from {total_samples:,} rows."
        )
        file_count += 1

    print(f"Done! Memory-safe stratified sampling completed for {file_count} files.")
    return file_count


# ---------------------------------------------------------------------------
# JSON → DataFrame
# ---------------------------------------------------------------------------

def json_to_df(
    json_path: str | Path,
    subreddit: str,
    dataset_type: str = "submissions",
    lines: bool = False,
) -> pd.DataFrame | None:
    """Load sampled JSON file(s) for a single subreddit into a DataFrame."""
    try:
        json_path = Path(json_path)
        valid_dataset_types = ["submissions", "comments"]
        if dataset_type not in valid_dataset_types:
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'. Choose from {valid_dataset_types}"
            )

        selected_files = [
            f for f in json_path.glob(f"{subreddit}_{dataset_type}.json") if f.is_file()
        ]
        if not selected_files:
            raise FileNotFoundError()

        tmp_files_list: list[pd.DataFrame] = []
        for file_path in selected_files:
            tmp_file = pd.read_json(file_path, lines=lines)
            tmp_file["subreddit"] = file_path.stem.split("_")[0]
            tmp_files_list.append(tmp_file)

        return pd.concat(tmp_files_list, ignore_index=True)

    except FileNotFoundError:
        print(f"No files found at the current path: {json_path}")
        return None
    except ValueError as e:
        print(e)
        return None


# ---------------------------------------------------------------------------
# Dual text cleaning: (1) Sentiment-safe, (2) Topic-modeling
# ---------------------------------------------------------------------------

def clean_sentiment_text(text: Any) -> str:
    """Clean Reddit text for sentiment analysis without destroying polarity cues.

    Decodes HTML entities, preserves Markdown link anchor text, removes raw
    URLs and Reddit Markdown markers (`*_~>`#`), and collapses whitespace while
    preserving casing, negations, emojis, and expressive punctuation (`!`, `?`).
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    # Keep anchor text in Markdown links [anchor](url) before stripping raw URLs
    text = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)\S+\)", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[*_~>`#]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SentimentTextCleaner:
    """Object-oriented wrapper for sentiment-safe Reddit text cleaning."""

    def clean_text(self, text: Any) -> str:
        return clean_sentiment_text(text)

    def clean_dataframe(self, df: pd.DataFrame, col: str = "text") -> pd.DataFrame:
        df = df.copy()
        noise_mask = df[col].map(is_noise_text)
        df = df[~noise_mask].copy()
        df["clean_text"] = df[col].map(self.clean_text)
        return df[df["clean_text"] != ""].reset_index(drop=True)


def clean_code_snippets(text: Any) -> str:
    """Remove fenced (``` ... ```) and inline (` ... `) code blocks."""
    if not isinstance(text, str):
        return ""
    no_blocks = re.sub(r"```[^`]*?```", "", text)
    return re.sub(r"`[^`]+?`", "", no_blocks)


def clean_text(text: Any) -> str:
    """Remove URLs and special symbols, normalize whitespace (for topic modeling)."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\W+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def process_pipeline(text: Any) -> str:
    """Run the topic-modeling cleaning pipeline (code snippets → symbol strip)."""
    processed = clean_code_snippets(text)
    return clean_text(processed)
