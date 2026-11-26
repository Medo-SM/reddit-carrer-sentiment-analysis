"""BERTopic wrapper: fit, coherence/diversity evaluation, HDBSCAN sweep, and outlier reduction."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.modeling.tracking import log_experiment


def fit_topic_model(
    docs: list[str],
    embeddings: np.ndarray,
    umap_params: dict,
    hdbscan_params: dict,
    top_n_words: int = 10,
    seed: int = 42,
):
    """Fit BERTopic with pre-computed embeddings and deterministic UMAP seed."""
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
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    return topic_model, np.array(topics)


def topic_name_map(model) -> dict[int, str]:
    """Return {topic_id: 'Topic N: word1_word2_word3'} from model.get_topic_info()."""
    info = model.get_topic_info()
    result: dict[int, str] = {}
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


def evaluate_topics(
    docs: list[str],
    model,
    topics: Sequence[int] | np.ndarray,
    top_n: int = 10,
) -> dict[str, float | int | None]:
    """Compute topic count, outlier rate, topic diversity, NPMI, and UMass coherence.

    Uses zero-dependency document-term co-occurrence matrices via scikit-learn/NumPy
    (and optionally computes gensim c_v if gensim happens to be installed).
    """
    topics_arr = np.asarray(topics)
    n_docs = len(topics_arr)
    outlier_pct = float((topics_arr == -1).mean()) if n_docs > 0 else 0.0

    info = model.get_topic_info()
    topic_word_lists: list[list[str]] = []
    for tid in info["Topic"]:
        tid_int = int(tid)
        if tid_int == -1:
            continue
        pairs = model.get_topic(tid_int) or []
        words = [str(w).lower() for w, _ in pairs[:top_n] if w]
        if words:
            topic_word_lists.append(words)

    n_topics = len(topic_word_lists)
    if n_topics == 0 or n_docs == 0:
        return {
            "n_topics": 0,
            "outlier_pct": round(outlier_pct, 4),
            "topic_diversity": 0.0,
            "coherence_npmi": None,
            "coherence_umass": None,
            "coherence_cv": None,
        }

    all_words_flat = [w for words in topic_word_lists for w in words]
    topic_diversity = len(set(all_words_flat)) / len(all_words_flat)

    # Co-occurrence over unique top words only (fast and memory-light)
    vocab = sorted(set(all_words_flat))
    coherence_npmi = None
    coherence_umass = None
    if vocab:
        vectorizer = CountVectorizer(
            vocabulary=vocab,
            binary=True,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        dtm = vectorizer.fit_transform(docs)
        co_mat = (dtm.T @ dtm).toarray().astype(np.float64)
        word_to_idx = vectorizer.vocabulary_

        npmi_scores: list[float] = []
        umass_scores: list[float] = []
        for words in topic_word_lists:
            idxs = [word_to_idx[w] for w in words if w in word_to_idx and co_mat[word_to_idx[w], word_to_idx[w]] > 0]
            if len(idxs) < 2:
                continue
            topic_npmis: list[float] = []
            topic_umass: list[float] = []
            for i in range(1, len(idxs)):
                for j in range(i):
                    wi, wj = idxs[i], idxs[j]
                    df_i = co_mat[wi, wi]
                    df_j = co_mat[wj, wj]
                    df_ij = co_mat[wi, wj]
                    topic_umass.append(math.log((df_ij + 1.0) / max(df_j, 1.0)))
                    if df_ij <= 0:
                        topic_npmis.append(-1.0)
                    else:
                        p_i = df_i / n_docs
                        p_j = df_j / n_docs
                        p_ij = df_ij / n_docs
                        denom = -math.log(min(p_ij, 1.0 - 1e-12))
                        if denom <= 0:
                            topic_npmis.append(0.0)
                        else:
                            pmi = math.log(p_ij / max(p_i * p_j, 1e-12))
                            topic_npmis.append(max(-1.0, min(1.0, pmi / denom)))
            if topic_npmis:
                npmi_scores.append(float(np.mean(topic_npmis)))
            if topic_umass:
                umass_scores.append(float(np.mean(topic_umass)))

        if npmi_scores:
            coherence_npmi = round(float(np.mean(npmi_scores)), 4)
        if umass_scores:
            coherence_umass = round(float(np.mean(umass_scores)), 4)

    # Optional gensim c_v if installed (not required)
    coherence_cv = None
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel

        tokenized = [str(d).lower().split() for d in docs]
        dictionary = Dictionary(tokenized)
        cm = CoherenceModel(
            topics=topic_word_lists,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence_cv = round(float(cm.get_coherence()), 4)
    except Exception:
        coherence_cv = None

    return {
        "n_topics": n_topics,
        "outlier_pct": round(outlier_pct, 4),
        "topic_diversity": round(topic_diversity, 4),
        "coherence_npmi": coherence_npmi,
        "coherence_umass": coherence_umass,
        "coherence_cv": coherence_cv,
    }


def reduce_outliers_safe(
    topic_model,
    docs: list[str],
    topics: Sequence[int] | np.ndarray,
    embeddings: np.ndarray | None = None,
):
    """Reassign outlier (-1) documents via embeddings or c-TF-IDF fallback."""
    topics_list = [int(t) for t in topics]
    if -1 not in topics_list:
        return topic_model, np.array(topics_list)

    try:
        if embeddings is not None:
            new_topics = topic_model.reduce_outliers(
                docs, topics_list, strategy="embeddings", embeddings=embeddings
            )
        else:
            new_topics = topic_model.reduce_outliers(docs, topics_list, strategy="c-tf-idf")
    except Exception:
        new_topics = topic_model.reduce_outliers(docs, topics_list, strategy="c-tf-idf")

    topic_model.update_topics(docs, topics=new_topics)
    return topic_model, np.array(new_topics)


def run_hdbscan_sweep(
    docs: list[str],
    embeddings: np.ndarray,
    umap_params: dict,
    hdbscan_base: dict,
    min_cluster_sizes: Sequence[int] = (15, 25, 35, 50),
    min_samples_list: Sequence[int] = (5, 10, 15),
    umap_n_neighbors_list: Sequence[int] | None = None,
    top_n_words: int = 10,
    seed: int = 42,
    experiment_log_dir: str = "data/processed/experiments",
) -> pd.DataFrame:
    """Grid sweep over HDBSCAN min_cluster_size, min_samples, and optional UMAP n_neighbors."""
    neighbors_grid = list(umap_n_neighbors_list) if umap_n_neighbors_list else [int(umap_params.get("n_neighbors", 15))]
    records = []

    for n_neighbors in neighbors_grid:
        u_params = {**umap_params, "n_neighbors": n_neighbors}
        for mcs in min_cluster_sizes:
            for ms in min_samples_list:
                if ms > mcs:
                    continue
                params = {**hdbscan_base, "min_cluster_size": mcs, "min_samples": ms}
                model, topics = fit_topic_model(
                    docs, embeddings, u_params, params, top_n_words=top_n_words, seed=seed
                )
                metrics = evaluate_topics(docs, model, topics, top_n=top_n_words)

                log_experiment(
                    params={
                        "umap_n_neighbors": n_neighbors,
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                        "seed": seed,
                        "n_docs": len(docs),
                    },
                    metrics=metrics,
                    output_dir=experiment_log_dir,
                )
                row = {
                    "umap_n_neighbors": n_neighbors,
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    **metrics,
                }
                records.append(row)
                print(
                    f"  nn={n_neighbors:2d} mcs={mcs:3d} ms={ms:2d} -> "
                    f"topics={metrics['n_topics']:3d}  outlier={metrics['outlier_pct']:.1%}  "
                    f"npmi={metrics['coherence_npmi']}  div={metrics['topic_diversity']}"
                )

    return pd.DataFrame(records)
