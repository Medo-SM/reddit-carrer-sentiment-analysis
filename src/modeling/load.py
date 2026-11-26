"""Data loading utilities: dtype-aware timestamp parsing, sample loading, and stratified subsetting."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.ingest.sampling import is_noise_text


def parse_created_utc(col: pd.Series) -> pd.Series:
    """Convert created_utc to UTC datetime regardless of source format.

    - int / float  -> epoch seconds  (scraper JSONL, raw parquet)
    - str / object -> ISO 8601       (sampled JSON via default=str)
    - datetime64   -> no-op (normalized to UTC)
    """
    dtype = col.dtype
    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
        return pd.to_datetime(col, unit="s", utc=True)
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        warnings.warn(
            "parse_created_utc: string column detected — assuming ISO 8601 format.",
            UserWarning,
            stacklevel=2,
        )
        return pd.to_datetime(col, utc=True)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.to_datetime(col, utc=True)
    raise TypeError(f"parse_created_utc: unrecognised dtype {dtype}")


def load_samples(
    samples_dir: str | Path,
    subreddits: Sequence[str],
    dataset_types: Sequence[str] = ("submissions", "comments"),
    drop_noise: bool = True,
) -> pd.DataFrame:
    """Load sampled JSON files into a single DataFrame with parsed UTC timestamps."""
    frames: list[pd.DataFrame] = []
    samples_dir = Path(samples_dir)
    for subreddit in subreddits:
        for dtype in dataset_types:
            path = samples_dir / f"{subreddit}_{dtype}.json"
            if not path.exists():
                print(f"[load_samples] missing: {path} — skipping")
                continue
            df = pd.read_json(path)
            if df.empty:
                continue
            df["subreddit"] = subreddit
            df["dataset_type"] = dtype
            if "created_utc" in df.columns:
                df["created_utc_dt"] = parse_created_utc(df["created_utc"])
            if drop_noise:
                text_col = next((c for c in ("text", "body", "selftext", "title") if c in df.columns), None)
                if text_col is not None:
                    df = df[~df[text_col].map(is_noise_text)].copy()
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No sample files found in {samples_dir}")
    return pd.concat(frames, ignore_index=True)


def draw_stratified_subset(
    df: pd.DataFrame,
    n_per_subreddit: int,
    seed: int = 42,
    time_col: str = "created_utc_dt",
) -> pd.DataFrame:
    """Draw a deterministic subset balanced across subreddits and calendar quarters."""
    rng = np.random.RandomState(seed)
    work = df.copy()
    if time_col in work.columns:
        dt = pd.to_datetime(work[time_col], utc=True)
        work["_quarter"] = dt.dt.tz_convert(None).dt.to_period("Q").astype(str)
    else:
        work["_quarter"] = "all"

    chosen_indices: list[int] = []
    for sub, sub_df in work.groupby("subreddit", sort=True):
        if len(sub_df) <= n_per_subreddit:
            chosen_indices.extend(sub_df.index.tolist())
            continue

        quarters = sub_df.groupby("_quarter", sort=True)
        n_q = max(1, len(quarters))
        per_q = max(1, n_per_subreddit // n_q)
        sub_chosen: list[int] = []
        remainder_pool: list[int] = []

        for _, q_df in quarters:
            idx_arr = q_df.index.to_numpy()
            if len(idx_arr) <= per_q:
                sub_chosen.extend(idx_arr.tolist())
            else:
                picked = rng.choice(idx_arr, size=per_q, replace=False).tolist()
                sub_chosen.extend(picked)
                picked_set = set(picked)
                remainder_pool.extend([i for i in idx_arr.tolist() if i not in picked_set])

        if len(sub_chosen) < n_per_subreddit and remainder_pool:
            need = min(n_per_subreddit - len(sub_chosen), len(remainder_pool))
            extra = rng.choice(remainder_pool, size=need, replace=False).tolist()
            sub_chosen.extend(extra)
        elif len(sub_chosen) > n_per_subreddit:
            sub_chosen = rng.choice(sub_chosen, size=n_per_subreddit, replace=False).tolist()

        chosen_indices.extend(sub_chosen)

    result = work.loc[sorted(chosen_indices)].drop(columns=["_quarter"]).reset_index(drop=True)
    return result
