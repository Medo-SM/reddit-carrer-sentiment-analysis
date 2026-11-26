"""Tests for text cleaning utilities (src/ingest/sampling.py)."""
import pandas as pd

from src.ingest.sampling import (
    SentimentTextCleaner,
    clean_code_snippets,
    clean_sentiment_text,
    clean_text,
    is_noise_text,
    process_pipeline,
)


class TestCleanText:
    def test_removes_urls(self):
        assert clean_text("visit http://example.com now") == "visit now"

    def test_removes_www_urls(self):
        assert clean_text("check www.example.com/page") == "check"

    def test_removes_special_symbols(self):
        result = clean_text("Python! <3 data science?")
        assert "!" not in result
        assert "data" in result

    def test_non_string_input_returns_empty(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    def test_empty_string(self):
        assert clean_text("") == ""


class TestCleanCodeSnippets:
    def test_removes_multiline_code_blocks(self):
        text = "Some text\n```python\nprint('hi')\n```\nafter"
        result = clean_code_snippets(text)
        assert "print" not in result
        assert "Some text" in result
        assert "after" in result

    def test_removes_inline_code(self):
        assert clean_code_snippets("run `pip install x` now") == "run  now"

    def test_non_string_input_returns_empty(self):
        assert clean_code_snippets(None) == ""


class TestProcessPipeline:
    def test_full_pipeline(self):
        text = "I use ```code``` and http://link.com and `x` symbols!"
        result = process_pipeline(text)
        assert "code" not in result
        assert "link" not in result
        assert "symbols" in result

    def test_empty_input(self):
        assert process_pipeline("") == ""


class TestSentimentTextCleaner:
    def test_preserves_punctuation_emojis_and_negation(self):
        raw = "I do NOT like this &amp; hate [these docs](https://example.com)!! **Terrible** 😡"
        cleaned = clean_sentiment_text(raw)
        assert "NOT" in cleaned
        assert "&" in cleaned and "&amp;" not in cleaned
        assert "these docs" in cleaned
        assert "https://example.com" not in cleaned
        assert "!!" in cleaned
        assert "😡" in cleaned
        assert "**" not in cleaned

    def test_noise_filtering(self):
        assert is_noise_text("[deleted]")
        assert is_noise_text("[removed]")
        assert is_noise_text("I am a bot, this action was performed automatically.")
        assert not is_noise_text("Great engineering role!")

    def test_clean_dataframe_drops_noise(self):
        df = pd.DataFrame({"text": ["[deleted]", "I am a bot", "Loved the interview!"]})
        cleaned = SentimentTextCleaner().clean_dataframe(df)
        assert len(cleaned) == 1
        assert cleaned["clean_text"].iloc[0] == "Loved the interview!"
