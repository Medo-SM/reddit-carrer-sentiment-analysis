import pandas as pd

from src.modeling.text import apply_pipeline, apply_sentiment_cleaning, apply_topic_cleaning


def test_url_removed():
    s = pd.Series(["check out https://example.com for jobs"])
    result = apply_topic_cleaning(s)
    assert "https" not in result[0]
    assert "example" not in result[0]


def test_code_block_removed():
    s = pd.Series(["hello `print(1)` world"])
    result = apply_pipeline(s)
    assert "print" not in result[0]
    assert "hello" in result[0] and "world" in result[0]


def test_none_returns_empty():
    s = pd.Series([None, float("nan")])
    result = apply_pipeline(s)
    assert list(result) == ["", ""]
    assert list(apply_sentiment_cleaning(s)) == ["", ""]


def test_sentiment_cleaning_keeps_cues():
    s = pd.Series(["NOT happy at all!! 😢 Check https://foo.bar"])
    result = apply_sentiment_cleaning(s)
    assert "NOT" in result[0]
    assert "!!" in result[0]
    assert "😢" in result[0]
    assert "https://foo.bar" not in result[0]
