"""Sentiment scoring: RoBERTa (primary) + VADER (cross-check)."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RoBERTa — primary model
# ---------------------------------------------------------------------------

def load_roberta(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
    device: str = "auto",
    local_files_only: bool = False,
):
    """Download/load from cache and return (tokenizer, model, resolved_device)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    model.eval()
    return tokenizer, model, device


def score_roberta(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Batched inference. Returns DataFrame with columns [negative, neutral, positive]."""
    import torch
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kwargs):
            return it

    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa"):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            if hasattr(encoded, "to"):
                encoded = encoded.to(device)
            elif isinstance(encoded, dict):
                encoded = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            results.append(probs)

    if results:
        arr = np.vstack(results)
    else:
        arr = np.empty((0, 3), dtype=np.float32)
    return pd.DataFrame(arr, columns=["negative", "neutral", "positive"])


# ---------------------------------------------------------------------------
# VADER — cross-check only
# ---------------------------------------------------------------------------

def score_vader(texts: list[str], neutral_band: float = 0.05) -> pd.DataFrame:
    """Score with VADER. polarity_label uses neutral band |compound| > neutral_band."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
        except ImportError:
            raise ImportError("vaderSentiment or nltk is required for score_vader")
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kwargs):
            return it

    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for text in tqdm(texts, desc="VADER"):
        score = analyzer.polarity_scores(str(text))
        compound = float(score["compound"])
        if compound > neutral_band:
            label = "positive"
        elif compound < -neutral_band:
            label = "negative"
        else:
            label = "neutral"
        rows.append({"vader_compound": compound, "vader_label": label})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model comparison — RQ1
# ---------------------------------------------------------------------------

def compare_models(df: pd.DataFrame) -> dict:
    """Compute agreement rate and return top-20 disagreement rows.

    Expects columns: roberta_label, vader_label (values: negative/neutral/positive).
    Returns dict with 'agreement_rate' (float) and 'top_disagreements' (DataFrame).
    """
    if "roberta_label" not in df.columns or "vader_label" not in df.columns:
        raise ValueError("DataFrame must contain 'roberta_label' and 'vader_label' columns.")

    agreement = float((df["roberta_label"] == df["vader_label"]).mean())
    disagreements = df[df["roberta_label"] != df["vader_label"]].copy()

    if "positive" in df.columns and "vader_compound" in df.columns:
        disagreements["gap"] = (
            disagreements["positive"] - disagreements["vader_compound"].clip(-1, 1)
        ).abs()
        top20 = disagreements.nlargest(20, "gap")
    else:
        top20 = disagreements.head(20)

    return {"agreement_rate": round(agreement, 4), "top_disagreements": top20}
