import tempfile

import numpy as np
import pandas as pd

from src.modeling.cache import (
    load_embeddings,
    load_parquet_cache,
    save_embeddings,
    save_parquet_cache,
)


def test_embeddings_roundtrip():
    emb = np.random.rand(10, 384).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        save_embeddings(emb, key="test_sub_submissions_10", cache_dir=tmp)
        loaded = load_embeddings(key="test_sub_submissions_10", cache_dir=tmp)
    assert loaded is not None
    assert np.allclose(emb, loaded)


def test_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        assert load_embeddings("nonexistent", tmp) is None
        assert load_parquet_cache("nonexistent", tmp) is None


def test_parquet_cache_roundtrip():
    df = pd.DataFrame({"id": ["x1", "x2"], "score": [1, 2]})
    with tempfile.TemporaryDirectory() as tmp:
        save_parquet_cache(df, key="scored_test", cache_dir=tmp)
        loaded = load_parquet_cache(key="scored_test", cache_dir=tmp)
    assert loaded is not None
    assert loaded.equals(df)
