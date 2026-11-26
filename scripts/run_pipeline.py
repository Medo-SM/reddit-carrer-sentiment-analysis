#!/usr/bin/env python3
"""Consolidated End-to-End Pipeline Runner (2018–2022, 7 Subreddits).

Executes Phases 2–5 in the root repository using 100% genuine model outputs
(no synthetic fallback, no hardcoded statistics):
  Phase 2: Stratified ISO-week sampling & temporal coverage audit -> data/samples/2.0/
  Phase 3: BERTopic modeling + HDBSCAN sweep (NPMI/UMass/diversity) -> data/processed/ & data/final/
  Phase 4: Full-sample VADER + 7k stratified genuine RoBERTa + RQ1 validation
  Phase 5: Quarterly aggregation, Mann-Kendall trends, 4-regime statistical tests,
           300-DPI publication figures, and dynamically generated results_report.md
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingest.sampling import stratified_sample_parquet_to_json
from src.modeling import set_seed
from src.modeling.cache import (
    load_embeddings,
    load_parquet_cache,
    save_embeddings,
    save_parquet_cache,
    save_topic_model,
)
from src.modeling.load import draw_stratified_subset, load_samples
from src.modeling.sentiment import compare_models, load_roberta, score_roberta, score_vader
from src.modeling.stats import (
    REGIME_ORDER,
    aggregate_quarterly,
    assign_regime,
    mann_kendall_test,
    regime_comparison,
)
from src.modeling.text import apply_sentiment_cleaning, apply_topic_cleaning
from src.modeling.topics import (
    evaluate_topics,
    fit_topic_model,
    reduce_outliers_safe,
    run_hdbscan_sweep,
    topic_name_map,
)
from src.modeling.tracking import log_experiment


def load_config() -> dict:
    with open(REPO_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def df_to_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def run_phase_2(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 72)
    print("PHASE 2: Stratified Sampling & Temporal Coverage Audit (2018–2022)")
    print("=" * 72)

    raw_dir = (REPO_ROOT / cfg["paths"]["raw_data_dir"]).resolve()
    samples_dir = REPO_ROOT / cfg["paths"]["data_samples_dir"] / "2.0"
    final_dir = REPO_ROOT / cfg["paths"]["final_dir"]
    tables_dir = final_dir / "tables"
    fig_dir = final_dir / "figures"
    for d in (samples_dir, tables_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    subreddits = cfg["analysis"]["subreddits"]
    dataset_types = cfg["analysis"]["dataset_types"]
    start_date = cfg["analysis"]["start_date"]
    end_date = cfg["analysis"]["end_date"]
    sample_size = int(cfg["analysis"]["sample_size"])
    seed = int(cfg["modeling"]["seed"])

    target_files = []
    for sub in subreddits:
        for dtype in dataset_types:
            fp = raw_dir / f"{sub}_{dtype}.parquet"
            if fp.exists():
                target_files.append(str(fp))
            else:
                print(f"Warning: Missing expected parquet file: {fp}")

    existing_jsons = list(samples_dir.glob("*.json"))
    expected_names = {f"{sub}_{dtype}.json" for sub in subreddits for dtype in dataset_types}
    existing_names = {p.name for p in existing_jsons}
    if expected_names.issubset(existing_names):
        print(f"Using existing {len(expected_names)} sample files in {samples_dir}")
    else:
        print(f"Running stratified sampling ({sample_size:,} max rows/file, {start_date}..{end_date})...")
        stratified_sample_parquet_to_json(
            file_paths=target_files,
            output_dir=samples_dir,
            desired_samples=sample_size,
            seed=seed,
            start_date=start_date,
            end_date=end_date,
        )

    df = load_samples(samples_dir=samples_dir, subreddits=subreddits, dataset_types=dataset_types)
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[(df["created_utc_dt"] >= start_ts) & (df["created_utc_dt"] <= end_ts)].copy().reset_index(drop=True)

    min_dt, max_dt = df["created_utc_dt"].min(), df["created_utc_dt"].max()
    assert min_dt.year >= 2018 and max_dt.year <= 2022, f"Unexpected date bounds: {min_dt}..{max_dt}"
    print(f"Loaded {len(df):,} clean sampled rows across {df['subreddit'].nunique()} subreddits ({min_dt.date()} to {max_dt.date()}).")

    df["week"] = df["created_utc_dt"].dt.tz_convert(None).dt.to_period("W").dt.start_time
    weekly_table = df.groupby(["week", "subreddit"]).size().unstack(fill_value=0)
    weekly_table.to_csv(final_dir / "coverage_audit.csv")

    summary = (
        df.groupby(["subreddit", "dataset_type"])
        .agg(n_rows=("id", "count"), min_date=("created_utc_dt", "min"), max_date=("created_utc_dt", "max"))
        .reset_index()
    )
    summary.to_csv(tables_dir / "coverage_summary.csv", index=False)

    plt.figure(figsize=(13, 5.5))
    for sub in sorted(weekly_table.columns):
        plt.plot(weekly_table.index, weekly_table[sub], label=f"r/{sub}", linewidth=1.3, alpha=0.85)
    plt.axvline(pd.Timestamp("2020-03-01"), color="#555555", linestyle="--", linewidth=1.4, label="COVID Freeze (Mar 2020)")
    plt.axvline(pd.Timestamp("2022-06-01"), color="#d9534f", linestyle="--", linewidth=1.4, label="2022 Tech Layoffs (Jun 2022)")
    plt.axvline(pd.Timestamp("2022-11-30"), color="#000000", linestyle=":", linewidth=1.6, label="ChatGPT Launch (Nov 30, 2022)")
    plt.title("Weekly Sampled Discourse Volume Across 7 Tech Subreddits (2018–2022)", fontsize=12)
    plt.xlabel("Week")
    plt.ylabel("Sampled Posts / Comments")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig0_weekly_coverage_audit.png", dpi=300)
    plt.close()

    with open(final_dir / "coverage_audit_report.md", "w", encoding="utf-8") as f:
        f.write("# Data Coverage & Quality Audit (2018–2022)\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write(f"- **Window:** {start_date} to {end_date} (20 complete quarters)\n")
        f.write(f"- **Total Clean Sampled Rows:** {len(df):,}\n")
        f.write(f"- **Observed Date Span:** {min_dt} to {max_dt} (zero 1970 timestamps)\n\n")
        f.write(df_to_md(summary) + "\n")

    return df, summary


def run_phase_3(df: pd.DataFrame, cfg: dict) -> dict:
    print("\n" + "=" * 72)
    print("PHASE 3: BERTopic Modeling & HDBSCAN Sweep (Submission Titles)")
    print("=" * 72)

    mcfg = cfg["modeling"]
    seed = int(mcfg["seed"])
    cache_dir = REPO_ROOT / mcfg["cache_dir"]
    exp_dir = REPO_ROOT / mcfg["experiment_log_dir"]
    final_dir = REPO_ROOT / cfg["paths"]["final_dir"]
    tables_dir = final_dir / "tables"
    fig_dir = final_dir / "figures"

    subs_df = df[df["dataset_type"] == "submissions"].copy().reset_index(drop=True)
    text_col = "text" if "text" in subs_df.columns else "title"
    subs_df["cleaned_text"] = apply_topic_cleaning(subs_df[text_col])
    subs_df = subs_df[subs_df["cleaned_text"].str.split().str.len() >= 3].reset_index(drop=True)

    per_sub = int(mcfg["topic_sample_per_subreddit"])
    subs_df = draw_stratified_subset(subs_df, n_per_subreddit=per_sub, seed=seed)
    docs = subs_df["cleaned_text"].tolist()
    print(f"Topic modeling corpus: {len(docs):,} stratified submissions across {subs_df['subreddit'].nunique()} subreddits.")

    embed_key = f"submissions_7subs_{len(docs)}_seed{seed}"
    embeddings = load_embeddings(embed_key, cache_dir=cache_dir)
    if embeddings is None:
        from sentence_transformers import SentenceTransformer

        print(f"Encoding {len(docs):,} submissions with {mcfg['embedding_model']}...")
        t0 = time.time()
        embedder = SentenceTransformer(
            mcfg["embedding_model"], local_files_only=bool(mcfg.get("local_files_only", True))
        )
        embeddings = embedder.encode(
            docs,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64,
        )
        print(f"Embedding completed in {time.time() - t0:.1f}s")
        save_embeddings(embeddings, embed_key, cache_dir=cache_dir)

    sweep_csv = tables_dir / "topic_sweep_results.csv"
    hdbscan_base = {
        "metric": mcfg["hdbscan"]["metric"],
        "cluster_selection_method": mcfg["hdbscan"]["cluster_selection_method"],
        "prediction_data": bool(mcfg["hdbscan"]["prediction_data"]),
    }

    print("\nRunning HDBSCAN sweep...")
    sweep_df = run_hdbscan_sweep(
        docs=docs,
        embeddings=embeddings,
        umap_params=mcfg["umap"],
        hdbscan_base=hdbscan_base,
        min_cluster_sizes=[25, 40, 60, 80],
        min_samples_list=[5, 10],
        umap_n_neighbors_list=[15],
        top_n_words=int(mcfg["top_n_words"]),
        seed=seed,
        experiment_log_dir=exp_dir,
    )
    sweep_df.to_csv(sweep_csv, index=False)

    # Select best config: prefer 10-40 topics and lowest outlier rate / highest NPMI
    candidates = sweep_df[(sweep_df["n_topics"] >= 10) & (sweep_df["n_topics"] <= 45)]
    if candidates.empty:
        candidates = sweep_df[sweep_df["n_topics"] >= 5]
    if candidates.empty:
        candidates = sweep_df
    low_out = candidates[candidates["outlier_pct"] < 0.25]
    pool = low_out if not low_out.empty else candidates
    best_row = pool.sort_values(
        by=["coherence_npmi", "topic_diversity", "outlier_pct"],
        ascending=[False, False, True],
    ).iloc[0]

    best_mcs = int(best_row["min_cluster_size"])
    best_ms = int(best_row["min_samples"])
    best_nn = int(best_row["umap_n_neighbors"])
    print(
        f"\nSelected best raw config: nn={best_nn}, min_cluster_size={best_mcs}, min_samples={best_ms} "
        f"(topics={int(best_row['n_topics'])}, outlier={best_row['outlier_pct']:.1%}, "
        f"npmi={best_row['coherence_npmi']}, diversity={best_row['topic_diversity']})"
    )

    u_best = {**mcfg["umap"], "n_neighbors": best_nn}
    h_best = {**hdbscan_base, "min_cluster_size": best_mcs, "min_samples": best_ms}
    topic_model, raw_topics = fit_topic_model(
        docs=docs,
        embeddings=embeddings,
        umap_params=u_best,
        hdbscan_params=h_best,
        top_n_words=int(mcfg["top_n_words"]),
        seed=seed,
    )
    raw_metrics = evaluate_topics(docs, topic_model, raw_topics, top_n=int(mcfg["top_n_words"]))

    final_topics = raw_topics
    reduced_metrics = raw_metrics
    used_outlier_reduction = False
    if raw_metrics["outlier_pct"] > 0.20 and raw_metrics["n_topics"] > 1:
        print(
            f"Raw outlier rate {raw_metrics['outlier_pct']:.1%} > 20%; "
            "applying embedding-based outlier reduction..."
        )
        topic_model, final_topics = reduce_outliers_safe(topic_model, docs, raw_topics, embeddings=embeddings)
        reduced_metrics = evaluate_topics(docs, topic_model, final_topics, top_n=int(mcfg["top_n_words"]))
        used_outlier_reduction = True
        log_experiment(
            params={"stage": "post_reduce_outliers", "min_cluster_size": best_mcs, "min_samples": best_ms, "seed": seed},
            metrics=reduced_metrics,
            output_dir=exp_dir,
        )
        print(
            f"Post-reduction metrics: topics={reduced_metrics['n_topics']}, "
            f"outlier={reduced_metrics['outlier_pct']:.1%}, npmi={reduced_metrics['coherence_npmi']}, "
            f"diversity={reduced_metrics['topic_diversity']}"
        )

    name_map = topic_name_map(topic_model)
    subs_df["topic_raw"] = raw_topics
    subs_df["topic"] = final_topics
    subs_df["topic_name"] = subs_df["topic"].map(name_map)
    subs_df["quarter"] = pd.to_datetime(subs_df["created_utc_dt"], utc=True).dt.tz_convert(None).dt.to_period("Q").astype(str)

    save_topic_model(topic_model, "final_submissions_model", cache_dir=cache_dir)
    save_parquet_cache(subs_df, "topics_submissions_df", cache_dir=cache_dir)

    # Export topic summary table
    topic_info = topic_model.get_topic_info().copy()
    topic_info["label"] = topic_info["Topic"].map(name_map)
    topic_info["top_words"] = topic_info["Topic"].apply(
        lambda tid: ", ".join(w for w, _ in (topic_model.get_topic(int(tid)) or [])[:8])
        if int(tid) != -1
        else "OUTLIER"
    )
    topic_info.to_csv(tables_dir / "topics_summary.csv", index=False)

    sub_topic_ct = pd.crosstab(subs_df["subreddit"], subs_df["topic_name"], normalize="index") * 100
    sub_topic_ct.round(2).to_csv(tables_dir / "topic_by_subreddit.csv")

    # Figure 1: 2D UMAP scatter
    from umap import UMAP

    umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=seed)
    coords = umap_2d.fit_transform(embeddings)
    subs_df["x"], subs_df["y"] = coords[:, 0], coords[:, 1]
    df_plot = subs_df[subs_df["topic"] != -1]
    top_tids = df_plot["topic"].value_counts().head(8).index

    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=df_plot[df_plot["topic"].isin(top_tids)],
        x="x",
        y="y",
        hue="topic_name",
        palette="tab10",
        alpha=0.65,
        s=14,
        linewidth=0,
    )
    plt.title(f"Figure 1: 2D UMAP Projection of Top-8 Discovered Career Topics (N={len(subs_df):,})", fontsize=12)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Topics", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_topic_scatter.png", dpi=300)
    plt.close()

    # Figure 2: Quarterly evolution of top-6 topics
    top6_names = [name_map[t] for t in df_plot["topic"].value_counts().head(6).index]
    q_topic = subs_df.groupby(["quarter", "topic_name"]).size().unstack(fill_value=0)
    q_share = q_topic.div(q_topic.sum(axis=1), axis=0) * 100
    q_share_top = q_share[[c for c in top6_names if c in q_share.columns]]

    plt.figure(figsize=(13, 5.5))
    for col in q_share_top.columns:
        plt.plot(q_share_top.index, q_share_top[col], marker="o", markersize=3.5, linewidth=1.7, label=col)
    plt.xticks(rotation=45, ha="right")
    plt.title("Figure 2: Quarterly Evolution of Top-6 Submission Topics (2018Q1–2022Q4)", fontsize=12)
    plt.xlabel("Quarter")
    plt.ylabel("Share of Submissions (%)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_topic_evolution.png", dpi=300)
    plt.close()

    return {
        "n_titles": len(docs),
        "best_params": {"umap_n_neighbors": best_nn, "min_cluster_size": best_mcs, "min_samples": best_ms},
        "raw_metrics": raw_metrics,
        "used_outlier_reduction": used_outlier_reduction,
        "final_metrics": reduced_metrics,
        "top_topics": topic_info[topic_info["Topic"] != -1].head(10)[["Topic", "Count", "label", "top_words"]].to_dict("records"),
    }


def run_phase_4(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print("\n" + "=" * 72)
    print("PHASE 4: Full-Sample VADER + Genuine 7k RoBERTa + RQ1 Validation")
    print("=" * 72)

    mcfg = cfg["modeling"]
    seed = int(mcfg["seed"])
    cache_dir = REPO_ROOT / mcfg["cache_dir"]
    exp_dir = REPO_ROOT / mcfg["experiment_log_dir"]
    final_dir = REPO_ROOT / cfg["paths"]["final_dir"]
    tables_dir = final_dir / "tables"
    fig_dir = final_dir / "figures"

    comments_df = df[df["dataset_type"] == "comments"].copy().reset_index(drop=True)
    text_col = "text" if "text" in comments_df.columns else "body"
    comments_df["cleaned_text"] = apply_sentiment_cleaning(comments_df[text_col])
    comments_df = comments_df[comments_df["cleaned_text"].str.split().str.len() >= 3].reset_index(drop=True)
    print(f"Comments corpus after sentiment-safe cleaning: {len(comments_df):,} comments.")

    # 1. Full-sample VADER (cached)
    vader_full = load_parquet_cache("sentiment_vader_full", cache_dir=cache_dir)
    if vader_full is None or len(vader_full) != len(comments_df):
        print(f"Scoring VADER on full {len(comments_df):,} comments...")
        t0 = time.time()
        v_res = score_vader(comments_df["cleaned_text"].tolist(), neutral_band=0.05)
        comments_df["vader_compound"] = v_res["vader_compound"].values
        comments_df["vader_label"] = v_res["vader_label"].values
        vader_full = comments_df
        save_parquet_cache(vader_full, "sentiment_vader_full", cache_dir=cache_dir)
        print(f"VADER full scoring completed in {time.time() - t0:.1f}s")
    else:
        print(f"Loaded cached full VADER scores ({len(vader_full):,} rows).")

    # 2. Balanced 7,000-comment stratified subset for genuine RoBERTa
    per_sub = int(mcfg["roberta_sample_per_subreddit"])
    roberta_key = f"sentiment_roberta_7subs_{per_sub}per_sub_seed{seed}"
    roberta_df = load_parquet_cache(roberta_key, cache_dir=cache_dir)
    if roberta_df is None:
        roberta_df = draw_stratified_subset(vader_full, n_per_subreddit=per_sub, seed=seed)
        print(f"Running 100% genuine RoBERTa ({mcfg['sentiment_model']}) on {len(roberta_df):,} balanced comments...")
        tokenizer, model, resolved_device = load_roberta(
            model_name=mcfg["sentiment_model"],
            device=mcfg["device"],
            local_files_only=bool(mcfg.get("local_files_only", True)),
        )
        print(f"RoBERTa loaded on device='{resolved_device}', batch_size={mcfg['sentiment_batch_size']}")
        t0 = time.time()
        r_scores = score_roberta(
            texts=roberta_df["cleaned_text"].tolist(),
            tokenizer=tokenizer,
            model=model,
            device=resolved_device,
            batch_size=int(mcfg["sentiment_batch_size"]),
        )
        elapsed = time.time() - t0
        print(f"RoBERTa inference finished in {elapsed:.1f}s ({len(roberta_df) / max(elapsed, 0.1):.1f} docs/s)")

        roberta_df["negative"] = r_scores["negative"].values
        roberta_df["neutral"] = r_scores["neutral"].values
        roberta_df["positive"] = r_scores["positive"].values
        roberta_df["roberta_label"] = r_scores[["negative", "neutral", "positive"]].idxmax(axis=1).values
        assert not roberta_df[["negative", "neutral", "positive", "roberta_label"]].isna().any().any()
        save_parquet_cache(roberta_df, roberta_key, cache_dir=cache_dir)
    else:
        print(f"Loaded cached genuine RoBERTa scores ({len(roberta_df):,} rows).")

    # Also save to canonical processed path
    roberta_df.to_parquet(REPO_ROOT / cfg["paths"]["processed_data_dir"] / "sentiment_df.parquet", index=False)

    # 3. RQ1 model comparison
    comp = compare_models(roberta_df)
    comp["confusion_matrix"].to_csv(tables_dir / "rq1_confusion_matrix.csv")

    per_class_rows = []
    for cls_name in ["negative", "neutral", "positive"]:
        m = comp["per_class"][cls_name]
        per_class_rows.append({
            "class": cls_name,
            "precision_vader_vs_roberta": round(m["precision"], 4),
            "recall_vader_vs_roberta": round(m["recall"], 4),
            "f1_vader_vs_roberta": round(m["f1-score"], 4),
            "support_roberta": int(m["support"]),
        })
    pd.DataFrame(per_class_rows).to_csv(tables_dir / "rq1_per_class_agreement.csv", index=False)

    rq1_summary = {
        "n_docs": comp["n_docs"],
        "n_disagreements": comp["n_disagreements"],
        "agreement_rate": comp["agreement_rate"],
        "cohen_kappa": comp["cohen_kappa"],
        "macro_f1": comp["macro_f1"],
        "vader_neutral_pct": comp["vader_neutral_pct"],
        "roberta_neutral_pct": comp["roberta_neutral_pct"],
    }
    pd.DataFrame([rq1_summary]).to_csv(tables_dir / "rq1_metrics.csv", index=False)

    cols_dis = [c for c in ["subreddit", "created_utc_dt", "vader_compound", "vader_label", "roberta_label", "negative", "neutral", "positive", "gap", "cleaned_text"] if c in comp["top_disagreements"].columns]
    comp["top_disagreements"][cols_dis].to_csv(tables_dir / "rq1_top20_disagreements.csv", index=False)

    log_experiment(
        params={"stage": "rq1_sentiment_validation", "n_roberta": len(roberta_df), "n_vader_full": len(vader_full), "seed": seed},
        metrics={k: v for k, v in rq1_summary.items()},
        output_dir=exp_dir,
    )

    # Figure 3: VADER vs RoBERTa Scatter + Confusion Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    sample_plot = roberta_df.sample(min(2500, len(roberta_df)), random_state=seed)
    sns.scatterplot(
        data=sample_plot,
        x="vader_compound",
        y="positive",
        hue="roberta_label",
        palette={"negative": "#d9534f", "neutral": "#f0ad4e", "positive": "#5cb85c"},
        alpha=0.45,
        s=16,
        ax=ax1,
    )
    ax1.set_title(f"VADER Compound vs. RoBERTa P(positive)\n(Agreement: {comp['agreement_rate']:.1%}, kappa={comp['cohen_kappa']:.3f})", fontsize=10.5)
    ax1.set_xlabel("VADER Compound (-1 to +1)")
    ax1.set_ylabel("RoBERTa P(positive)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ct_pct = pd.crosstab(roberta_df["vader_label"], roberta_df["roberta_label"], normalize="all") * 100
    ct_pct = ct_pct.reindex(index=["negative", "neutral", "positive"], columns=["negative", "neutral", "positive"], fill_value=0.0)
    sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="Blues", cbar=False, linewidths=1, ax=ax2)
    ax2.set_title("Confusion Matrix (% of 7,000 Comments)", fontsize=10.5)
    ax2.set_xlabel("RoBERTa Label (Primary)")
    ax2.set_ylabel("VADER Label (Baseline)")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_vader_roberta_comparison.png", dpi=300)
    plt.close()

    print(
        f"RQ1 Summary: agreement={comp['agreement_rate']:.1%}, kappa={comp['cohen_kappa']:.3f}, "
        f"macro_f1={comp['macro_f1']:.3f}, VADER neutral={comp['vader_neutral_pct']:.1f}%, "
        f"RoBERTa neutral={comp['roberta_neutral_pct']:.1f}%"
    )
    return vader_full, roberta_df, rq1_summary


def run_phase_5(
    vader_full: pd.DataFrame,
    roberta_df: pd.DataFrame,
    topic_summary: dict,
    rq1_summary: dict,
    cov_summary: pd.DataFrame,
    cfg: dict,
):
    print("\n" + "=" * 72)
    print("PHASE 5: Longitudinal Statistics, Regimes, Figures & Report")
    print("=" * 72)

    final_dir = REPO_ROOT / cfg["paths"]["final_dir"]
    tables_dir = final_dir / "tables"
    fig_dir = final_dir / "figures"

    # 1. Quarterly aggregates (RoBERTa 7k & VADER full)
    q_roberta = aggregate_quarterly(roberta_df, label_col="roberta_label")
    q_roberta.to_csv(tables_dir / "quarterly_sentiment_roberta.csv", index=False)

    q_vader = aggregate_quarterly(vader_full, label_col="vader_label")
    q_vader.to_csv(tables_dir / "quarterly_sentiment_vader_full.csv", index=False)

    # 2. Mann-Kendall trends across 20 quarters (2018Q1–2022Q4)
    mk_rows = []
    for sub, grp in q_roberta.groupby("subreddit", sort=True):
        grp_s = grp.sort_values("quarter")
        mk_neg = mann_kendall_test(grp_s["neg_pct"])
        mk_net = mann_kendall_test(grp_s["net_mean"])
        mk_rows.append({
            "subreddit": sub,
            "n_quarters": mk_neg["n"],
            "neg_pct_trend": mk_neg["trend"],
            "neg_pct_z": mk_neg["z"],
            "neg_pct_p": mk_neg["p_value"],
            "neg_pct_sen_slope_per_q": mk_neg["sen_slope"],
            "net_sent_trend": mk_net["trend"],
            "net_sent_p": mk_net["p_value"],
            "net_sent_sen_slope_per_q": mk_net["sen_slope"],
        })
    mk_df = pd.DataFrame(mk_rows)
    mk_df.to_csv(tables_dir / "mann_kendall_trends.csv", index=False)

    # 3. 4-Regime comparison
    reg_roberta = regime_comparison(roberta_df, label_col="roberta_label", n_boot=500, seed=42)
    reg_roberta.to_csv(tables_dir / "regime_comparison_roberta.csv", index=False)

    reg_vader = regime_comparison(vader_full, label_col="vader_label", n_boot=500, seed=42)
    reg_vader.to_csv(tables_dir / "regime_comparison_vader_full.csv", index=False)

    # Subreddit overall distribution
    sub_dist = (
        roberta_df.groupby("subreddit")["roberta_label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )[["negative", "neutral", "positive"]].round(2)
    sub_dist["n_roberta"] = roberta_df.groupby("subreddit").size()
    sub_dist.reset_index().to_csv(tables_dir / "subreddit_sentiment_roberta.csv", index=False)

    # Figure 4: Sentiment by Subreddit
    plt.figure(figsize=(10, 5.2))
    sub_dist[["negative", "neutral", "positive"]].plot(
        kind="bar",
        stacked=True,
        color=["#d9534f", "#f0ad4e", "#5cb85c"],
        width=0.62,
        edgecolor="none",
    )
    plt.title("Figure 4: RoBERTa Sentiment Composition Across 7 Tech Subreddits (2018–2022)", fontsize=12)
    plt.xlabel("Subreddit")
    plt.ylabel("Share of Comments (%)")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_sentiment_by_subreddit.png", dpi=300)
    plt.close()

    # Figure 5: Quarterly Negative Sentiment across 4 Regimes (RoBERTa 7k vs VADER Full)
    q_all_r = q_roberta[q_roberta["subreddit"] == "ALL"].sort_values("quarter")
    q_all_v = q_vader[q_vader["subreddit"] == "ALL"].sort_values("quarter")
    plt.figure(figsize=(12.5, 5.2))
    plt.plot(q_all_r["quarter"], q_all_r["neg_pct"], marker="o", linewidth=2.2, color="#c9302c", label="RoBERTa Negative % (7k balanced)")
    plt.plot(q_all_v["quarter"], q_all_v["neg_pct"], marker="s", linewidth=1.8, linestyle="--", color="#2b5c8f", label=f"VADER Negative % (full {len(vader_full):,})")
    plt.axvspan("2018Q1", "2020Q1", color="#dddddd", alpha=0.35, label="Regime 1: Pre-COVID")
    plt.axvspan("2020Q1", "2020Q4", color="#f0ad4e", alpha=0.18, label="Regime 2: COVID Freeze")
    plt.axvspan("2021Q1", "2022Q2", color="#5cb85c", alpha=0.15, label="Regime 3: Hiring Boom")
    plt.axvspan("2022Q2", "2022Q4", color="#d9534f", alpha=0.18, label="Regime 4: 2022 Layoffs")
    plt.xticks(rotation=45, ha="right")
    plt.title("Figure 5: Quarterly Negative Sentiment Across Macroeconomic Regimes (2018Q1–2022Q4)", fontsize=12)
    plt.xlabel("Quarter")
    plt.ylabel("Negative Comments (%)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_longitudinal_regimes.png", dpi=300)
    plt.close()

    # Figure 6: Subreddit x Regime Negative Sentiment Heatmap
    heat_df = reg_roberta[reg_roberta["subreddit"] != "ALL"].set_index("subreddit")[
        ["pre_covid_neg_pct", "covid_freeze_neg_pct", "hiring_boom_neg_pct", "layoffs_2022_neg_pct"]
    ]
    heat_df.columns = ["Pre-COVID\n(2018–20Q1)", "COVID Freeze\n(2020)", "Hiring Boom\n(2021–22H1)", "2022 Layoffs\n(2022H2)"]
    plt.figure(figsize=(9, 5.2))
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.8)
    plt.title("Figure 6: Negative Sentiment Share (%) by Subreddit & Market Regime (RoBERTa)", fontsize=12)
    plt.ylabel("Subreddit")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_regime_subreddit_heatmap.png", dpi=300)
    plt.close()

    # Dynamically build verified results report
    top_topics_df = pd.DataFrame(topic_summary["top_topics"])
    report_path = final_dir / "results_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Verified Model Results Report (2018–2022, 7 Tech Subreddits)\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write("## 1. Dataset Coverage\n\n")
        f.write(df_to_md(cov_summary) + "\n\n")
        f.write("## 2. Sentiment Model Comparison (RQ1: Genuine RoBERTa vs. VADER on 7,000 Balanced Comments)\n\n")
        f.write(f"- **Primary Model:** `{cfg['modeling']['sentiment_model']}` (no simulated fallback)\n")
        f.write(f"- **Baseline:** VADER (`|compound| <= 0.05` neutral band), also scored across all {len(vader_full):,} comments\n")
        f.write(f"- **Agreement Rate:** {rq1_summary['agreement_rate']:.2%}\n")
        f.write(f"- **Cohen's Kappa:** {rq1_summary['cohen_kappa']:.4f}\n")
        f.write(f"- **Macro-F1 (VADER vs. RoBERTa):** {rq1_summary['macro_f1']:.4f}\n")
        f.write(f"- **Neutral Share:** VADER = {rq1_summary['vader_neutral_pct']:.2f}% vs. RoBERTa = {rq1_summary['roberta_neutral_pct']:.2f}%\n\n")
        f.write("### Subreddit Sentiment Distribution (RoBERTa 7k)\n\n")
        f.write(df_to_md(sub_dist.reset_index()) + "\n\n")
        f.write("## 3. Topic Modeling Results (BERTopic on Stratified Submission Titles)\n\n")
        f.write(f"- **Corpus Size:** {topic_summary['n_titles']:,} cleaned submission titles\n")
        f.write(f"- **Selected HDBSCAN/UMAP Params:** `{json.dumps(topic_summary['best_params'])}`\n")
        f.write(f"- **Raw HDBSCAN Metrics:** `{json.dumps(topic_summary['raw_metrics'])}`\n")
        f.write(f"- **Outlier Reduction Applied:** {topic_summary['used_outlier_reduction']}\n")
        f.write(f"- **Final Topic Metrics:** `{json.dumps(topic_summary['final_metrics'])}`\n\n")
        f.write("### Top-10 Discovered Topics\n\n")
        f.write(df_to_md(top_topics_df) + "\n\n")
        f.write("## 4. Longitudinal & Regime Statistical Tests (2018Q1–2022Q4)\n\n")
        f.write("### Mann-Kendall Quarterly Monotonic Trend Tests (RoBERTa 7k)\n\n")
        f.write(df_to_md(mk_df) + "\n\n")
        f.write("### Four-Regime Negative Sentiment Comparison (RoBERTa 7k)\n\n")
        f.write(df_to_md(reg_roberta) + "\n\n")
        f.write("### Four-Regime Negative Sentiment Comparison (Full-Sample VADER Cross-Check)\n\n")
        f.write(df_to_md(reg_vader) + "\n")

    print(f"Results report written to: {report_path}")


def main():
    t_start = time.time()
    cfg = load_config()
    set_seed(int(cfg["modeling"]["seed"]))

    df, cov_summary = run_phase_2(cfg)
    topic_summary = run_phase_3(df, cfg)
    vader_full, roberta_df, rq1_summary = run_phase_4(df, cfg)
    run_phase_5(vader_full, roberta_df, topic_summary, rq1_summary, cov_summary, cfg)

    print(f"\nALL PHASES COMPLETE in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
