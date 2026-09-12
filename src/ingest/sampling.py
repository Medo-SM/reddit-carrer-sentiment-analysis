"""Shared sampling and text-cleaning utilities extracted from notebooks.

All 4 notebooks (1.0–1.3) defined these functions inline. This module
provides a single importable source so notebooks become thin calls:

    from src.ingest.sampling import (
        get_parquet_from_path,
        sample_parquet_file_to_json,
        json_to_df,
        clean_code_snippets,
        clean_text,
        process_pipeline,
    )
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Parquet discovery
# ---------------------------------------------------------------------------

def get_parquet_from_path(path: str | Path) -> list[str]:
    """Recursively find all .parquet files under *path*."""
    parquet_files: list[str] = []
    for dir_path, _, files in os.walk(path):
        for f in files:
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
