import pandas as pd
from src.modeling.text import apply_pipeline


def test_url_removed():
    s = pd.Series(["check https://reddit.com for info"])
    assert "http" not in apply_pipeline(s)[0]


def test_code_block_removed():
    s = pd.Series(["`print('hello')`"])
    assert "`" not in apply_pipeline(s)[0]


def test_none_returns_empty():
    s = pd.Series([None])
    assert apply_pipeline(s)[0] == ""


def test_empty_string_ok():
    s = pd.Series([""])
    assert apply_pipeline(s)[0] == ""


def test_mixed_content():
    s = pd.Series(["Visit http://example.com — `code` here!"])
    result = apply_pipeline(s)[0]
    assert "http" not in result and "`" not in result
