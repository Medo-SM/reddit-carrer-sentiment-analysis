"""Intermediate results cache — embeddings (.npy) and fitted BERTopic models."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _cache_dir(cfg_cache_dir: str | Path) -> Path:
    p = Path(cfg_cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_embeddings(
    embeddings: np.ndarray,
    key: str,
    cache_dir: str | Path = "data/output/cache",
) -> Path:
    """Save embeddings array. Key format: '{subreddit}_{dataset_type}_{n_docs}'."""
    path = _cache_dir(cache_dir) / f"embeddings_{key}.npy"
    np.save(path, embeddings)
    print(f"[cache] embeddings saved → {path}")
    return path


def load_embeddings(
    key: str,
    cache_dir: str | Path = "data/output/cache",
) -> np.ndarray | None:
    """Return cached embeddings if they exist, else None."""
    path = _cache_dir(cache_dir) / f"embeddings_{key}.npy"
    if path.exists():
        print(f"[cache] embeddings loaded ← {path}")
        return np.load(path)
    return None


def save_topic_model(
    model,
    key: str,
    cache_dir: str | Path = "data/output/cache",
) -> Path:
    """Save a fitted BERTopic model."""
    path = _cache_dir(cache_dir) / f"topic_model_{key}"
    model.save(str(path))
    print(f"[cache] topic model saved → {path}")
    return path


def load_topic_model(
    key: str,
    cache_dir: str | Path = "data/output/cache",
):
    """Return a cached BERTopic model if it exists, else None."""
    try:
        from bertopic import BERTopic
    except ImportError:
        print("[cache] BERTopic is not installed.")
        return None

    path = _cache_dir(cache_dir) / f"topic_model_{key}"
    if path.exists():
        print(f"[cache] topic model loaded ← {path}")
        return BERTopic.load(str(path))
    return None
