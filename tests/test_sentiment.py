from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.modeling.sentiment import compare_models, score_roberta, score_vader


def _check_vader_available():
    try:
        import vaderSentiment  # noqa: F401
    except ImportError:
        try:
            import nltk  # noqa: F401
        except ImportError:
            pytest.skip("Neither vaderSentiment nor nltk is installed")


def test_vader_positive():
    _check_vader_available()
    df = score_vader(["I love this job and I got a wonderful promotion!"])
    assert df["vader_label"][0] == "positive"
    assert df["vader_compound"][0] > 0.05


def test_vader_negative():
    _check_vader_available()
    df = score_vader(["I hate being unemployed and this layoff is terrible."])
    assert df["vader_label"][0] == "negative"
    assert df["vader_compound"][0] < -0.05


def test_roberta_output_shape():
    torch = pytest.importorskip("torch")
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    mock_model.return_value.logits = torch.tensor([[0.1, 0.2, 0.7]])

    df = score_roberta(["Great post!"], mock_tokenizer, mock_model, device="cpu", batch_size=1)
    assert df.shape == (1, 3)
    assert list(df.columns) == ["negative", "neutral", "positive"]
    assert np.isclose(df.iloc[0].sum(), 1.0)


def test_compare_models_metrics():
    data = pd.DataFrame({
        "roberta_label": ["positive", "negative", "neutral", "positive"],
        "vader_label":   ["positive", "negative", "positive", "neutral"],
        "positive": [0.9, 0.1, 0.2, 0.8],
        "vader_compound": [0.8, -0.7, 0.6, 0.0],
    })
    res = compare_models(data)
    assert res["agreement_rate"] == 0.5
    assert "cohen_kappa" in res
    assert "macro_f1" in res
    assert res["vader_neutral_pct"] == 25.0
    assert res["roberta_neutral_pct"] == 25.0
    assert len(res["top_disagreements"]) == 2
