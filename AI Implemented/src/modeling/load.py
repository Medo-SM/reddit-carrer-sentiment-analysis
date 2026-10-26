"""Data loading utilities: dtype-aware timestamp parsing and sample loading."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import pandas as pd


def parse_created_utc(col: pd.Series) -> pd.Series:
    """Convert created_utc to datetime regardless of source format.

    - int / float  → epoch seconds  (scraper JSONL, raw parquet)
    - str          → ISO 8601       (sampled JSON via default=str)
    - datetime     → no-op (ensuring UTC timezone)

    Raises a warning on object dtype so format assumptions are documented.
    """
    dtype = col.dtype
    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
        return pd.to_datetime(col, unit="s", utc=True)
    if pd.api.types.is_object_dtype(dtype):
        warnings.warn("parse_created_utc: string column detected — assuming ISO 8601 format.")
        return pd.to_datetime(col, utc=True)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.to_datetime(col, utc=True)
    raise TypeError(f"parse_created_utc: unrecognised dtype {dtype}")


def load_samples(
    samples_dir: str | Path,
    subreddits: Sequence[str],
    dataset_types: Sequence[str] = ("submissions", "comments"),
) -> pd.DataFrame:
    """Load all sampled JSON files for the given subreddits into one DataFrame.

    Replaces the broken json_to_df(path, dataset_type) call (Bug 5 fix: wrong arg order).
    Adds parsed `created_utc_dt` column via parse_created_utc() (Bug 1 fix).
    """
    frames: list[pd.DataFrame] = []
    samples_dir = Path(samples_dir)
    for subreddit in subreddits:
        for dtype in dataset_types:
            path = samples_dir / f"{subreddit}_{dtype}.json"
            if not path.exists():
                print(f"[load_samples] missing: {path} — skipping")
                continue
            df = pd.read_json(path)
            df["subreddit"] = subreddit
            df["dataset_type"] = dtype
            if "created_utc" in df.columns:
                df["created_utc_dt"] = parse_created_utc(df["created_utc"])
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No sample files found in {samples_dir}")
    return pd.concat(frames, ignore_index=True)
