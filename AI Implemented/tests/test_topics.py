import pytest
from unittest.mock import MagicMock
import pandas as pd

from src.modeling.topics import topic_name_map


def test_topic_name_map_format():
    mock_model = MagicMock()
    mock_model.get_topic_info.return_value = pd.DataFrame({
        "Topic": [-1, 0, 1],
        "Count": [10, 50, 40],
    })
    mock_model.get_topic.side_effect = lambda tid: [("word1", 0.5), ("word2", 0.3), ("word3", 0.2)] if tid != -1 else []

    res = topic_name_map(mock_model)
    assert res[-1] == "Topic -1: outlier"
    assert res[0] == "Topic 0: word1_word2_word3"
    assert res[1] == "Topic 1: word1_word2_word3"


@pytest.mark.slow
def test_fit_finds_topics():
    pytest.importorskip("bertopic")
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer
    from src.modeling.topics import fit_topic_model

    docs = (
        ["got a new software engineering job offer today"] * 40 +
        ["layoffs announced at tech startup severance package"] * 40 +
        ["interview prep leetcode grind data structures"] * 40
    )
    umap_params = {"n_neighbors": 5, "n_components": 2, "min_dist": 0.0, "metric": "cosine"}
    hdbscan_params = {
        "min_cluster_size": 5,
        "min_samples": 2,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": True,
    }

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(docs)
    model, topics = fit_topic_model(docs, embeddings, umap_params, hdbscan_params, seed=42)

    assert len(set(topics) - {-1}) >= 1
    name_map = topic_name_map(model)
    assert all(isinstance(k, int) for k in name_map)
    assert all(isinstance(v, str) for v in name_map.values())
