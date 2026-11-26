"""Sentiment scoring: RoBERTa (primary) + VADER (cross-check) + RQ1 comparison metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score


# ---------------------------------------------------------------------------
# RoBERTa — primary model
# ---------------------------------------------------------------------------

def load_roberta(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
    device: str = "auto",
    local_files_only: bool = True,
):
    """Load tokenizer and sequence classification model, returning (tokenizer, model, device)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(device)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load '{model_name}' with local_files_only={local_files_only}. "
            "Ensure the model is cached locally or explicitly pass local_files_only=False."
        ) from exc

    model.eval()
    return tokenizer, model, device


def score_roberta(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Batched inference returning DataFrame with columns [negative, neutral, positive]."""
    import torch

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kwargs):
            return it

    if not texts:
        return pd.DataFrame(np.empty((0, 3), dtype=np.float32), columns=["negative", "neutral", "positive"])

    # Sort by length before batching to minimize padding overhead on CPU, then restore order
    order = sorted(range(len(texts)), key=lambda idx: len(texts[idx]))
    ordered_texts = [texts[idx] for idx in order]

    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(ordered_texts), batch_size), desc="RoBERTa"):
            batch = ordered_texts[i : i + batch_size]
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

    arr_sorted = np.vstack(results)
    arr = np.empty_like(arr_sorted)
    arr[order] = arr_sorted
    return pd.DataFrame(arr, columns=["negative", "neutral", "positive"])


# ---------------------------------------------------------------------------
# VADER — cross-check baseline
# ---------------------------------------------------------------------------

def score_vader(texts: list[str], neutral_band: float = 0.05) -> pd.DataFrame:
    """Score texts with VADER; discrete label uses neutral band |compound| <= neutral_band."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise ImportError("vaderSentiment or nltk is required for score_vader") from exc

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
    """Compare RoBERTa and VADER labels with agreement, Cohen's kappa, macro-F1, and disagreements."""
    if "roberta_label" not in df.columns or "vader_label" not in df.columns:
        raise ValueError("DataFrame must contain 'roberta_label' and 'vader_label' columns.")

    valid = df.dropna(subset=["roberta_label", "vader_label"]).copy()
    labels = ["negative", "neutral", "positive"]
    y_ref = valid["roberta_label"].astype(str)
    y_base = valid["vader_label"].astype(str)

    agreement = float((y_ref == y_base).mean()) if len(valid) > 0 else 0.0
    kappa = float(cohen_kappa_score(y_ref, y_base, labels=labels)) if len(valid) > 0 else 0.0
    report = classification_report(
        y_ref, y_base, labels=labels, output_dict=True, zero_division=0
    )
    macro_f1 = float(report["macro avg"]["f1-score"])

    confusion = pd.crosstab(
        valid["vader_label"], valid["roberta_label"], rownames=["vader_label"], colnames=["roberta_label"]
    )

    disagreements = valid[y_ref != y_base].copy()
    if "positive" in valid.columns and "vader_compound" in valid.columns:
        vader_pos_equiv = (disagreements["vader_compound"].clip(-1, 1) + 1.0) / 2.0
        disagreements["gap"] = (disagreements["positive"] - vader_pos_equiv).abs()
        top20 = disagreements.nlargest(20, "gap")
    else:
        top20 = disagreements.head(20)

    return {
        "n_docs": int(len(valid)),
        "n_disagreements": int(len(disagreements)),
        "agreement_rate": round(agreement, 4),
        "cohen_kappa": round(kappa, 4),
        "macro_f1": round(macro_f1, 4),
        "vader_neutral_pct": round(float((y_base == "neutral").mean()) * 100, 2) if len(valid) else 0.0,
        "roberta_neutral_pct": round(float((y_ref == "neutral").mean()) * 100, 2) if len(valid) else 0.0,
        "per_class": report,
        "confusion_matrix": confusion,
        "top_disagreements": top20,
    }
