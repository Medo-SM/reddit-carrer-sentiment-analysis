import tempfile
import numpy as np

from src.modeling.cache import load_embeddings, save_embeddings


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
