"""Intermediate results cache — embeddings (.npy), BERTopic models, and scored Parquet."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _cache_dir(cfg_cache_dir: str | Path) -> Path:
    p = Path(cfg_cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_embeddings(
    embeddings: np.ndarray,
    key: str,
    cache_dir: str | Path = "data/processed/cache",
) -> Path:
    """Save embeddings array to .npy."""
    path = _cache_dir(cache_dir) / f"embeddings_{key}.npy"
    np.save(path, embeddings)
    print(f"[cache] embeddings saved -> {path}")
    return path


def load_embeddings(
    key: str,
    cache_dir: str | Path = "data/processed/cache",
) -> np.ndarray | None:
    """Return cached embeddings if present, else None."""
    path = _cache_dir(cache_dir) / f"embeddings_{key}.npy"
    if path.exists():
        print(f"[cache] embeddings loaded <- {path}")
        return np.load(path)
    return None


def save_topic_model(
    model,
    key: str,
    cache_dir: str | Path = "data/processed/cache",
) -> Path:
    """Save a fitted BERTopic model."""
    path = _cache_dir(cache_dir) / f"topic_model_{key}"
    model.save(str(path))
    print(f"[cache] topic model saved -> {path}")
    return path


def load_topic_model(
    key: str,
    cache_dir: str | Path = "data/processed/cache",
):
    """Return a cached BERTopic model if present, else None."""
    try:
        from bertopic import BERTopic
    except ImportError:
        print("[cache] BERTopic is not installed.")
        return None

    path = _cache_dir(cache_dir) / f"topic_model_{key}"
    if path.exists():
        print(f"[cache] topic model loaded <- {path}")
        return BERTopic.load(str(path))
    return None


def save_parquet_cache(
    df: pd.DataFrame,
    key: str,
    cache_dir: str | Path = "data/processed/cache",
) -> Path:
    """Save an intermediate DataFrame to Parquet."""
    path = _cache_dir(cache_dir) / f"{key}.parquet"
    df.to_parquet(path, index=False)
    print(f"[cache] dataframe saved -> {path}")
    return path


def load_parquet_cache(
    key: str,
    cache_dir: str | Path = "data/processed/cache",
) -> pd.DataFrame | None:
    """Load an intermediate Parquet DataFrame if present, else None."""
    path = _cache_dir(cache_dir) / f"{key}.parquet"
    if path.exists():
        print(f"[cache] dataframe loaded <- {path}")
        return pd.read_parquet(path)
    return None
