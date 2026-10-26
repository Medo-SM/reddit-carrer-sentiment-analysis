"""BERTopic wrapper: fit, sweep, and name-map utilities."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

try:
    from src.modeling.tracking import log_experiment
except ImportError:
    try:
        from .tracking import log_experiment
    except Exception:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.modeling.tracking import log_experiment


def fit_topic_model(
    docs: list[str],
    embeddings: np.ndarray,
    umap_params: dict,
    hdbscan_params: dict,
    top_n_words: int = 10,
    seed: int = 42,
):
    """Fit BERTopic with pre-computed embeddings.

    Bug 4 fix: embeddings are computed once on cleaned_text BEFORE this call
    and passed in — BERTopic does NOT re-embed inside fit_transform().
    Bug 3 fix: UMAP random_state set explicitly for determinism.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    u_params = dict(umap_params)
    u_params["random_state"] = seed
    umap_model = UMAP(**u_params)
    hdbscan_model = HDBSCAN(**hdbscan_params)
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=top_n_words,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    return topic_model, np.array(topics)


def topic_name_map(model) -> dict[int, str]:
    """Return {topic_id: 'Topic N: word1_word2_word3'} from model.get_topic_info().

    Bug 2 fix: uses topic_info DataFrame, NOT the per-document topic array.
    """
    info = model.get_topic_info()
    result = {}
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            result[tid] = "Topic -1: outlier"
        else:
            topic_words = model.get_topic(tid)
            if topic_words:
                words = "_".join(w for w, _ in topic_words[:3])
            else:
                words = f"topic_{tid}"
            result[tid] = f"Topic {tid}: {words}"
    return result


def run_hdbscan_sweep(
    docs: list[str],
    embeddings: np.ndarray,
    umap_params: dict,
    hdbscan_base: dict,
    min_cluster_sizes: Sequence[int] = (15, 25, 35, 50),
    min_samples_list: Sequence[int] = (5, 10, 15),
    seed: int = 42,
    experiment_log_dir: str = "data/output/experiments",
) -> pd.DataFrame:
    """Grid sweep over HDBSCAN min_cluster_size and min_samples.

    Evaluates each config on:
    - outlier_pct   : fraction of docs assigned to topic -1 (target < 20%)
    - n_topics      : number of non-outlier topics (expect 10–30 for 45k titles)
    - coherence_cv  : c_v coherence from gensim (higher = better)

    Logs every run to experiment_log.jsonl for systematic comparison.
    """
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
        has_gensim = True
    except ImportError:
        has_gensim = False

    records = []
    tokenized = [doc.split() for doc in docs]
    dictionary = Dictionary(tokenized) if has_gensim else None

    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            params = {**hdbscan_base, "min_cluster_size": mcs, "min_samples": ms}
            model, topics = fit_topic_model(docs, embeddings, umap_params, params, seed=seed)

            n_outliers = int((topics == -1).sum())
            outlier_pct = n_outliers / len(topics) if len(topics) > 0 else 0.0
            info = model.get_topic_info()
            topic_words = [
                [w for w, _ in model.get_topic(t)]
                for t in info["Topic"] if t != -1
            ]
            n_topics = len(topic_words)

            coherence_cv = None
            if has_gensim and dictionary and topic_words:
                try:
                    cm = CoherenceModel(
                        topics=topic_words,
                        texts=tokenized,
                        dictionary=dictionary,
                        coherence="c_v",
                    )
                    coherence_cv = round(float(cm.get_coherence()), 4)
                except Exception as ex:
                    print(f"Warning: CoherenceModel failed: {ex}")
                    coherence_cv = None

            metrics = {
                "outlier_pct": round(outlier_pct, 4),
                "n_topics": n_topics,
                "coherence_cv": coherence_cv,
            }
            log_experiment(
                params={"min_cluster_size": mcs, "min_samples": ms, "seed": seed},
                metrics=metrics,
                output_dir=experiment_log_dir,
            )
            records.append({"min_cluster_size": mcs, "min_samples": ms, **metrics})
            print(f"  mcs={mcs:3d} ms={ms:2d} → topics={n_topics:3d}  outlier={outlier_pct:.1%}  cv={coherence_cv}")

    return pd.DataFrame(records)
