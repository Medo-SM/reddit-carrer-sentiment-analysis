"""Shared sampling and text-cleaning utilities for the AI Implemented pipeline.

Provides importable utilities for parquet discovery, sampling (random and stratified),
JSON loading, and Reddit text cleaning:

    from src.ingest.sampling import (
        get_parquet_from_path,
        sample_parquet_file_to_json,
        stratified_sample_parquet_to_json,
        json_to_df,
        clean_code_snippets,
        clean_text,
        process_pipeline,
    )
"""
from __future__ import annotations

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
    """Recursively find all .parquet files under *path*."""
    parquet_files: list[str] = []
    for dir_path, _, files in os.walk(path):
        for f in sorted(files):
            if f.endswith(".parquet"):
                parquet_files.append(os.path.join(dir_path, f))
    return parquet_files


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_parquet_file_to_json(
    file_paths: list[str],
    output_dir: str | Path,
    desired_samples: int = 30_000,
) -> int:
    """Sample rows from each Parquet file and write to JSON.

    Args:
        file_paths: List of Parquet file paths.
        output_dir: Directory to write the sampled JSON files.
        desired_samples: Max rows to retain per file.

    Returns:
        Number of files processed.
    """
    if pq is None:
        raise ImportError("pyarrow is required to read Parquet files. Please install pyarrow.")
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
            sampled_table = table.take(random_indices)
            selected_samples = sampled_table.to_pylist()

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


SALT = "reddit_career_sentiment_2026"


def hash_author(author: str | None) -> str:
    """One-way SHA-256 hash of Reddit username with fixed salt for privacy preservation."""
    if not isinstance(author, str) or author in ("[deleted]", "[removed]", ""):
        return "DELETED"
    raw = f"{SALT}:{author}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def stratified_sample_parquet_to_json(
    file_paths: list[str],
    output_dir: str | Path,
    desired_samples: int = 5000,
    seed: int = 42,
) -> int:
    """Stratify by ISO week of created_utc, then sample proportionally per week.

    Preserves temporal distribution across the 2020-2025 window while being
    ultra-light on memory by reading row groups incrementally.

    Args:
        file_paths: List of Parquet file paths.
        output_dir: Directory to write the sampled JSON files.
        desired_samples: Target number of rows to retain per file.
        seed: Random seed for deterministic sampling.

    Returns:
        Number of files processed.
    """
    if pq is None:
        raise ImportError("pyarrow is required to read Parquet files. Please install pyarrow.")
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    file_count = 0
    for fp in file_paths:
        file_name = os.path.splitext(os.path.basename(fp))[0]
        pf = pq.ParquetFile(fp)
        total_samples = pf.metadata.num_rows

        if total_samples <= desired_samples or "created_utc" not in pf.schema.names:
            table = pf.read()
            selected_samples = table.to_pylist()
        else:
            # Memory-safe: read only the created_utc column first (~30MB even for 4M rows)
            ts_series = pf.read(["created_utc"])["created_utc"].to_pandas()
            if pd.api.types.is_numeric_dtype(ts_series.dtype):
                dt_series = pd.to_datetime(ts_series, unit="s", utc=True)
            else:
                dt_series = pd.to_datetime(ts_series, utc=True)

            # Filter to study analysis window (2020-2025) if present
            in_window = (dt_series >= "2020-01-01") & (dt_series <= "2025-12-31")
            valid_indices = np.where(in_window)[0]
            if len(valid_indices) == 0:
                valid_indices = np.arange(total_samples)
                dt_valid = dt_series
            else:
                dt_valid = dt_series.iloc[valid_indices]

            week_bins = dt_valid.dt.to_period("W")
            indices_by_group: dict[Any, list[int]] = {}
            for orig_idx, wk in zip(valid_indices, week_bins):
                indices_by_group.setdefault(wk, []).append(orig_idx)

            selected_indices: list[int] = []
            n_pool = len(valid_indices)
            for wk, idx_list in indices_by_group.items():
                group_len = len(idx_list)
                prop_n = math.ceil((group_len / n_pool) * desired_samples)
                n_take = min(group_len, prop_n)
                if n_take > 0:
                    chosen = rng.choice(idx_list, size=n_take, replace=False)
                    selected_indices.extend(chosen.tolist())

            if len(selected_indices) > desired_samples:
                selected_indices = rng.choice(selected_indices, size=desired_samples, replace=False).tolist()

            indices_arr = np.sort(np.array(selected_indices, dtype=np.int64))

            # Memory-safe: stream row groups to extract only selected indices
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
                    sub_table = rg_table.take(rg_indices)
                    records = sub_table.to_pylist()
                    selected_samples.extend(records)

                current_row = rg_end

        # Data quality filters + privacy hashing
        clean_samples = []
        seen_ids = set()
        for rec in selected_samples:
            rec_id = rec.get("id")
            if rec_id in seen_ids:
                continue
            seen_ids.add(rec_id)

            text_val = rec.get("text", rec.get("selftext", rec.get("body", "")))
            if not isinstance(text_val, str) or text_val in ("[deleted]", "[removed]", ""):
                continue
            if len(text_val.split()) < 3:
                continue

            if "author" in rec:
                rec["author_hash"] = hash_author(rec["author"])
                del rec["author"]
            else:
                rec["author_hash"] = "UNKNOWN"

            clean_samples.append(rec)

        out_path = output_dir / f"{file_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clean_samples, f, ensure_ascii=False, indent=2, default=str)

        print(
            f"Stratified sampling completed for {file_name}: "
            f"retained {len(clean_samples)} clean samples from {total_samples:,} rows."
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
    """Load sampled JSON file(s) for a single subreddit into a DataFrame.

    Args:
        json_path: Directory containing the sampled JSON files.
        subreddit: Subreddit name (e.g. ``"datascience"``).
        dataset_type: ``"submissions"`` or ``"comments"``.
        lines: If True, read as line-delimited JSON.

    Returns:
        DataFrame with an added ``subreddit`` column, or None on error.
    """
    try:
        json_path = Path(json_path)
        valid_dataset_types = ["submissions", "comments"]
        if dataset_type not in valid_dataset_types:
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'. "
                f"Choose from {valid_dataset_types}"
            )

        selected_files = [
            f for f in json_path.glob(f"{subreddit}_{dataset_type}.json")
            if f.is_file()
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
# Text cleaning
# ---------------------------------------------------------------------------

def clean_code_snippets(text: str) -> str:
    """Remove fenced (````` ... `````) and inline (`` ` ... ` ``) code blocks."""
    if not isinstance(text, str):
        return ""
    # Multi-line code blocks
    no_blocks = re.sub(r"```[^`]*?```", "", text)
    # Inline code
    clean = re.sub(r"`[^`]+?`", "", no_blocks)
    return clean


def clean_text(text: str) -> str:
    """Remove URLs and special symbols, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text, flags=re.IGNORECASE)  # Remove links
    text = re.sub(r"\W+", " ", text)      # Remove special symbols
    return text.strip()


def process_pipeline(text: str) -> str:
    """Run the full cleaning pipeline (code snippets → text cleaning)."""
    processed = clean_code_snippets(text)
    processed = clean_text(processed)
    return processed
