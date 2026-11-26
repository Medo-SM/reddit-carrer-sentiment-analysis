"""Dual text preprocessing wrappers for sentiment analysis vs. topic modeling."""
from __future__ import annotations

import pandas as pd

from src.ingest.sampling import clean_sentiment_text, process_pipeline


def apply_sentiment_cleaning(series: pd.Series) -> pd.Series:
    """Sentiment-safe cleaning: preserves casing, negations, emojis, and punctuation."""
    return series.fillna("").astype(str).map(clean_sentiment_text)


def apply_topic_cleaning(series: pd.Series) -> pd.Series:
    """Topic-modeling cleaning: strips code blocks, URLs, and special symbols."""
    return series.fillna("").astype(str).map(process_pipeline)


# Backward-compatible alias for topic modeling callers and existing tests.
apply_pipeline = apply_topic_cleaning
