# AI Implementation Plan — Reddit Career Sentiment Analysis

**AI Workspace Root:** `AI Implemented/` (project root — all AI work stays inside this folder)  
**Human Reference Plans:** `plans/research-plan.md`, `plans/modeling-rework.md`, `plans/notebook-architecture.md`, `plans/ToDo.md`  
**Target Venue:** ICWSM 2027 (deadline 15 Jan 2027)  
**Analysis Window:** 2020–2025 | **Break Date:** 30 Nov 2022 (ChatGPT launch) | **Seed:** 42 everywhere

> [!CAUTION]
> The AI implementation is fully self-contained inside `AI Implemented/`. It reads shared data from `data/raw/` and `data/samples/` via config paths but **never writes to or modifies any file outside of `AI Implemented/`**.

> [!NOTE]
> This implementation works with the **6 existing subreddits** that already have parquet files in `data/raw/`. No scraping is performed. `r/ExperiencedDevs` and `r/csMajors` are noted as future additions (RQ3 seniority split) but are out of scope for this implementation pass.

---

## Goal

Implement the complete pipeline for the research project *"Tech Career Sentiment After ChatGPT"* — mirroring the project structure — so the human-authored and AI-authored versions can be compared as parallel references.

The AI-authored version covers, end-to-end:

1. The `src/modeling/` library module (all 6 bugs fixed)
2. Four focused analysis notebooks (`2.0` → `5.0`)
3. Full pytest test suite
4. `config/config.yaml` (AI workspace copy)
5. Updated requirements files

---

## Folder Structure

```
AI Implemented/
│
├── plans/
│   └── IMPLEMENTATION_PLAN.md      ← this file
│
├── config/
│   └── config.yaml                 ← AI copy; corrected HDBSCAN defaults;
│                                      data paths point to shared ../data/ (read-only)
│
├── src/
│   ├── __init__.py
│   ├── ingest/                     ← mirrors root src/ingest/ (no changes to originals)
│   │   ├── __init__.py
│   │   ├── sampling.py             ← copy of root version + stratified sampler variant
│   │   └── zst_to_parquet.py       ← copy, unchanged
│   │
│   └── modeling/                   ← NEW — does not exist in root src/
│       ├── __init__.py             ← set_seed()
│       ├── load.py                 ← parse_created_utc() + load_samples()
│       ├── text.py                 ← apply_pipeline()
│       ├── sentiment.py            ← score_roberta() + score_vader() + compare_models()
│       ├── topics.py               ← fit_topic_model() + topic_name_map() + run_hdbscan_sweep()
│       ├── cache.py                ← save/load embeddings + topic models
│       └── tracking.py             ← log_experiment() → JSONL
│
├── notebooks/
│   ├── 2.0_sampling.ipynb          ← NEW
│   ├── 3.0_topic_modeling.ipynb    ← NEW
│   ├── 4.0_sentiment_analysis.ipynb← NEW
│   └── 5.0_results_and_viz.ipynb  ← NEW
│
├── tests/
│   ├── __init__.py
│   ├── test_text.py
│   ├── test_dates.py
│   ├── test_cache.py
│   ├── test_tracking.py
│   ├── test_sentiment.py
│   └── test_topics.py              ← @pytest.mark.slow
│
├── requirements_cpu.txt            ← adds gensim, transformers, tqdm, vaderSentiment,
│                                      sentence-transformers
├── requirements_gpu.txt            ← same additions, cu build of torch
├── pytest.ini                      ← registers `slow` marker
└── Makefile                        ← test / test-fast / smoke-test / lint / compile / clean
```

---

## Execution Sequence

```
Step 1 → Step 2 → Step 3 → Step 4 → Verification
```

Steps 1, 3, and 4 are independent of each other.  
Notebooks (Step 2) must run strictly in order: 2.0 → 3.0 → 4.0 → 5.0.

---

## Step 1 — `src/modeling/` Module

This module does not exist in the root `src/`. All 6 confirmed bugs from `plans/modeling-rework.md` are fixed here.

### `AI Implemented/config/config.yaml`

The AI workspace config is a self-contained copy. Data paths point **up** to the shared `data/` directory (read-only). All outputs are written inside `AI Implemented/data/output/` only.

```yaml
# AI Implemented workspace config — read-only references to shared data/
paths:
  raw_data_dir: "../data/raw"
  data_samples_dir: "../data/samples"
  ai_output_dir: "data/output"     # AI-only outputs: figures, CSVs, cached models

preprocessing:
  max_len: 256
  lowercase: true

analysis:
  # Only the 6 subreddits with existing parquet files in data/raw/
  # r/ExperiencedDevs and r/csMajors are out of scope for this pass (no raw data yet)
  subreddits:
    - cscareerquestions
    - cscareerquestionsEU
    - datascience
    - dataengineering
    - cybersecurity
    - SoftwareEngineering
  sample_size: 30000
  dataset_types:
    - submissions
    - comments

modeling:
  embedding_model: "all-MiniLM-L6-v2"
  umap:
    n_neighbors: 15
    n_components: 5
    min_dist: 0.0
    metric: cosine
    random_state: 42          # DO NOT change — reproducibility
  hdbscan:
    min_cluster_size: 25      # was 100 in root config → causes Bug 3 (48% outlier rate)
    min_samples: 5
    metric: euclidean
    cluster_selection_method: eom
    prediction_data: true
  top_n_words: 10
  sentiment_model: "cardiffnlp/twitter-roberta-base-sentiment"
  sentiment_batch_size: 64
  device: "auto"              # auto, cuda, cpu, mps
  max_sample_cpu: 5000
  seed: 42
  cache_dir: "data/output/cache"
  experiment_log_dir: "data/output/experiments"
  dataset_types:
    - submissions
    - comments
```

---

### Bug Fix Map

| Bug | Root Cause | Fix Location |
|---|---|---|
| Bug 1 — 1970-01-01 dates | `pd.to_datetime(int_seconds)` treated as nanoseconds | `load.py` → `parse_created_utc()` |
| Bug 2 — wrong legend labels | `topics[topic_id]` indexes the doc-array, not the topic map | `topics.py` → `topic_name_map()` |
| Bug 3 — 48% outlier rate | `HDBSCAN(min_cluster_size=100)` too strict | `topics.py` → `run_hdbscan_sweep()` |
| Bug 4 — double embedding | Pre-computed embeddings not passed to `fit_transform()` | `topics.py` → `fit_topic_model()` |
| Bug 5 — dead code TypeError | `json_to_df(path, dataset_type)` wrong arg order | `load.py` → `load_samples()` replaces it |
| Bug 6 — VADER domain mismatch | VADER tuned for generic social media; weak on career/tech Reddit | `sentiment.py` → RoBERTa primary, VADER cross-check |

---

### `src/modeling/__init__.py`

```python
"""Modeling package — call set_seed(42) at the top of every notebook."""
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and torch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

### `src/modeling/load.py` *(fixes Bug 1 and Bug 5)*

```python
"""Data loading utilities: dtype-aware timestamp parsing and sample loading."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def parse_created_utc(col: pd.Series) -> pd.Series:
    """Convert created_utc to datetime regardless of source format.

    - int / float  → epoch seconds  (scraper JSONL, raw parquet)
    - str          → ISO 8601       (sampled JSON via default=str)
    - datetime     → no-op

    Raises a warning on unexpected dtype so failures are loud, not silent 1970 dates.
    """
    dtype = col.dtype
    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
        return pd.to_datetime(col, unit="s", utc=True)
    if pd.api.types.is_object_dtype(dtype):
        import warnings
        warnings.warn("parse_created_utc: string column detected — assuming ISO 8601.")
        return pd.to_datetime(col, utc=True)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return col  # already correct
    raise TypeError(f"parse_created_utc: unrecognised dtype {dtype}")


def load_samples(
    samples_dir: str | Path,
    subreddits: list[str],
    dataset_types: list[str] = ("submissions", "comments"),
) -> pd.DataFrame:
    """Load all sampled JSON files for the given subreddits into one DataFrame.

    Replaces the broken json_to_df(path, dataset_type) call (Bug 5 — wrong arg order).
    Adds parsed `created_utc_dt` column via parse_created_utc().
    """
    frames = []
    samples_dir = Path(samples_dir)
    for subreddit in subreddits:
        for dtype in dataset_types:
            path = samples_dir / f"{subreddit}_{dtype}.json"
            if not path.exists():
                print(f"[load_samples] missing: {path} — skipping")
                continue
            df = pd.read_json(path)
            df["subreddit"] = subreddit
            df["dataset_type"] = dtype
            df["created_utc_dt"] = parse_created_utc(df["created_utc"])
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No sample files found in {samples_dir}")
    return pd.concat(frames, ignore_index=True)
```

---

### `src/modeling/text.py`

```python
"""Text preprocessing — thin vectorised wrapper over src.ingest.sampling.process_pipeline."""
from __future__ import annotations

import pandas as pd
from src.ingest.sampling import process_pipeline


def apply_pipeline(series: pd.Series) -> pd.Series:
    """Apply the full cleaning pipeline to every string in a Series.

    Handles None / NaN values by returning an empty string.
    """
    return series.fillna("").astype(str).map(process_pipeline)
```

---

### `src/modeling/cache.py`

```python
"""Intermediate results cache — embeddings (.npy) and fitted BERTopic models."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _cache_dir(cfg_cache_dir: str) -> Path:
    p = Path(cfg_cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_embeddings(embeddings: np.ndarray, key: str, cache_dir: str) -> Path:
    """Save embeddings array. Key format: '{subreddit}_{dataset_type}_{n_docs}'."""
    path = _cache_dir(cache_dir) / f"embeddings_{key}.npy"
    np.save(path, embeddings)
    print(f"[cache] embeddings saved → {path}")
    return path


def load_embeddings(key: str, cache_dir: str) -> np.ndarray | None:
    """Return cached embeddings if they exist, else None."""
    path = _cache_dir(cache_dir) / f"embeddings_{key}.npy"
    if path.exists():
        print(f"[cache] embeddings loaded ← {path}")
        return np.load(path)
    return None


def save_topic_model(model, key: str, cache_dir: str) -> Path:
    """Save a fitted BERTopic model."""
    from bertopic import BERTopic  # noqa: F401 — type hint only
    path = _cache_dir(cache_dir) / f"topic_model_{key}"
    model.save(str(path))
    print(f"[cache] topic model saved → {path}")
    return path


def load_topic_model(key: str, cache_dir: str):
    """Return a cached BERTopic model if it exists, else None."""
    from bertopic import BERTopic
    path = _cache_dir(cache_dir) / f"topic_model_{key}"
    if path.exists():
        print(f"[cache] topic model loaded ← {path}")
        return BERTopic.load(str(path))
    return None
```

---

### `src/modeling/tracking.py`

```python
"""Lightweight JSONL experiment log — one line per modeling run."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def log_experiment(
    params: dict,
    metrics: dict,
    output_dir: str = "data/output/experiments",
) -> Path:
    """Append one experiment record to experiment_log.jsonl.

    Args:
        params:  All tunable values (n_docs, dataset_type, hdbscan_*, umap_*, seed, etc.)
        metrics: Outcome values (n_topics, outlier_pct, coherence_cv, agreement_rate, etc.)
        output_dir: Directory for the log file.

    Returns:
        Path to the log file.
    """
    import torch
    import bertopic

    entry = {
        "timestamp": datetime.now().isoformat(),
        "params": params,
        "metrics": metrics,
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "bertopic": bertopic.__version__,
        },
    }
    path = Path(output_dir) / "experiment_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return path
```

---

### `src/modeling/topics.py` *(fixes Bugs 2, 3, 4)*

```python
"""BERTopic wrapper: fit, sweep, and name-map utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

from src.modeling.tracking import log_experiment


def fit_topic_model(
    docs: list[str],
    embeddings: np.ndarray,
    umap_params: dict,
    hdbscan_params: dict,
    top_n_words: int = 10,
    seed: int = 42,
) -> tuple[BERTopic, np.ndarray]:
    """Fit BERTopic with pre-computed embeddings.

    Bug 4 fix: embeddings are computed once on cleaned_text BEFORE this call
    and passed in — BERTopic does NOT re-embed inside fit_transform().
    Bug 3 fix: UMAP random_state set explicitly for determinism.
    """
    umap_model = UMAP(**umap_params, random_state=seed)
    hdbscan_model = HDBSCAN(**hdbscan_params)
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=top_n_words,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    return topic_model, np.array(topics)


def topic_name_map(model: BERTopic) -> dict[int, str]:
    """Return {topic_id: 'Topic N: word1_word2_word3'} from model.get_topic_info().

    Bug 2 fix: uses topic_info DataFrame, NOT the per-document topic array.
    """
    info = model.get_topic_info()
    result = {}
    for _, row in info.iterrows():
        tid = row["Topic"]
        words = "_".join(w for w, _ in model.get_topic(tid)[:3]) if tid != -1 else "outlier"
        result[tid] = f"Topic {tid}: {words}"
    return result


def run_hdbscan_sweep(
    docs: list[str],
    embeddings: np.ndarray,
    umap_params: dict,
    hdbscan_base: dict,
    min_cluster_sizes: list[int] = (15, 25, 35, 50),
    min_samples_list: list[int] = (5, 10, 15),
    seed: int = 42,
    experiment_log_dir: str = "data/output/experiments",
) -> pd.DataFrame:
    """Grid sweep over HDBSCAN min_cluster_size and min_samples.

    Evaluates each config on:
    - outlier_pct   : fraction of docs assigned to topic -1 (target < 20%)
    - n_topics      : number of non-outlier topics (expect 10–30 for 45k titles)
    - coherence_cv  : c_v coherence from gensim (higher = better)

    Logs every run to experiment_log.jsonl for later comparison.
    """
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    records = []
    tokenized = [doc.split() for doc in docs]
    dictionary = Dictionary(tokenized)

    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            params = {**hdbscan_base, "min_cluster_size": mcs, "min_samples": ms}
            model, topics = fit_topic_model(docs, embeddings, umap_params, params, seed=seed)

            n_outliers = int((topics == -1).sum())
            outlier_pct = n_outliers / len(topics)
            topic_words = [
                [w for w, _ in model.get_topic(t)]
                for t in model.get_topic_info()["Topic"] if t != -1
            ]
            n_topics = len(topic_words)

            coherence_cv = None
            if topic_words:
                cm = CoherenceModel(
                    topics=topic_words,
                    texts=tokenized,
                    dictionary=dictionary,
                    coherence="c_v",
                )
                coherence_cv = round(cm.get_coherence(), 4)

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
```

---

### `src/modeling/sentiment.py` *(fixes Bug 6)*

```python
"""Sentiment scoring: RoBERTa (primary) + VADER (cross-check)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# RoBERTa — primary model
# ---------------------------------------------------------------------------

def load_roberta(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
                 device: str = "auto"):
    """Download (first time) and return (tokenizer, model, device_str)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
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
    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa"):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            results.append(probs)
    arr = np.vstack(results)
    return pd.DataFrame(arr, columns=["negative", "neutral", "positive"])


# ---------------------------------------------------------------------------
# VADER — cross-check only
# ---------------------------------------------------------------------------

def score_vader(texts: list[str], neutral_band: float = 0.05) -> pd.DataFrame:
    """Score with VADER. polarity_label uses neutral band |compound| > neutral_band."""
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for text in tqdm(texts, desc="VADER"):
        score = analyzer.polarity_scores(text)
        compound = score["compound"]
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

    Expects columns: roberta_label, vader_label (both in negative/neutral/positive).
    Returns dict with 'agreement_rate' (float) and 'top_disagreements' (DataFrame).
    """
    agreement = (df["roberta_label"] == df["vader_label"]).mean()
    disagreements = df[df["roberta_label"] != df["vader_label"]].copy()
    disagreements["gap"] = (
        df["positive"] - df["vader_compound"].clip(-1, 1)
    ).abs()
    top20 = disagreements.nlargest(20, "gap")
    return {"agreement_rate": round(agreement, 4), "top_disagreements": top20}
```

---

## Step 2 — Four Notebooks

> [!NOTE]
> Notebooks are shipped as `.ipynb` skeletons — all cells contain working code but outputs are empty. Run them in strict order: `2.0 → 3.0 → 4.0 → 5.0`.

### `notebooks/2.0_sampling.ipynb` — Sampling & Coverage Audit

**Purpose:** Produce `data/samples/2.0/` with ~30k stratified rows per subreddit/type file; verify date ranges; produce temporal coverage audit.

| Cell | Code |
|---|---|
| [1] | `import yaml; from src.modeling import set_seed; set_seed(42); cfg = yaml.safe_load(open("../config/config.yaml"))` |
| [2] | `from src.ingest.sampling import get_parquet_from_path; paths = get_parquet_from_path(cfg["paths"]["raw_data_dir"])` |
| [3] | `from src.ingest.sampling import stratified_sample_parquet_to_json` *(new function — see §Step 2 note)* `stratified_sample_parquet_to_json(paths, "data/output/samples/2.0/", desired_samples=30_000)` |
| [4] | `from src.modeling.load import load_samples; df = load_samples("data/output/samples/2.0/", cfg["analysis"]["subreddits"])` |
| [5] | `print(df.groupby("subreddit")[["created_utc_dt"]].agg(["min","max","count"]))` — assert years 2020–2025, not 1970 |
| [6] | Weekly row count per subreddit line plot (temporal coverage audit) |
| [7] | Missing-weeks heatmap — marks June 2023 API blackout |
| [8] | `df.resample("W", on="created_utc_dt").size().to_csv("data/output/coverage_audit.csv")` |

**New function added in `src/ingest/sampling.py` copy:**

```python
def stratified_sample_parquet_to_json(
    file_paths: list[str],
    output_dir: str | Path,
    desired_samples: int = 30_000,
) -> int:
    """Stratify by ISO week of created_utc, then sample proportionally per week.
    Preserves temporal distribution — unlike plain random.sample().
    """
```

---

### `notebooks/3.0_topic_modeling.ipynb` — BERTopic + HDBSCAN Sweep

**Purpose:** Fit stable topic model on submission titles; find best HDBSCAN config; fix Bugs 2, 3, 4.

| Cell | Code summary |
|---|---|
| [1] | Imports, `set_seed(42)`, load config |
| [2] | `load_samples(..., dataset_types=["submissions"])` + `apply_pipeline()` |
| [3] | `load_embeddings(key)` or compute with `SentenceTransformer("all-MiniLM-L6-v2")` + `save_embeddings(key)` |
| [4] | `run_hdbscan_sweep(docs, embeddings, umap_params, hdbscan_base, ...)` |
| [5] | Print sweep results table sorted by `coherence_cv` desc |
| [6] | Select best config (outlier_pct < 20%, max coherence_cv); human review checkpoint |
| [7] | `fit_topic_model(docs, embeddings, best_umap, best_hdbscan)` + `save_topic_model(key)` |
| [8] | `topic_name_map(model)` — print all topic names for human sanity-check |
| [9] | 2D UMAP scatter (colour by topic, labels from `topic_name_map`) — fixes Bug 2 |
| [10] | `plt.savefig("data/output/figures/topic_scatter.png", dpi=300)` |

---

### `notebooks/4.0_sentiment_analysis.ipynb` — RoBERTa + VADER Comparison (RQ1)

**Purpose:** Score sentiment using RoBERTa (primary) and VADER (cross-check); validate model choice for RQ1; produce `sentiment_df.parquet`.

| Cell | Code summary |
|---|---|
| [1] | Imports, `set_seed(42)`, load config |
| [2] | `load_samples(..., dataset_types=["comments"])` — sentiment on comments |
| [3] | `apply_pipeline(df["body"])` → `df["cleaned_text"]` |
| [4] | `score_vader(df["cleaned_text"].tolist())` → merge as `vader_*` columns |
| [5] | `tokenizer, model, device = load_roberta(cfg["modeling"]["sentiment_model"])` then `score_roberta(...)` — loads from cache if available |
| [6] | `df["roberta_label"] = df[["negative","neutral","positive"]].idxmax(axis=1)` |
| [7] | `result = compare_models(df); print(f"Agreement rate: {result['agreement_rate']:.1%}")` |
| [8] | Scatter: VADER compound vs RoBERTa positive probability (RQ1 justification figure) |
| [9] | Table: top-20 disagreements with `cleaned_text` column (shows domain failure modes) |
| [10] | Stacked bar: sentiment distribution by subreddit |
| [11] | Line: weekly RoBERTa sentiment share pre/post 30 Nov 2022 (vertical line at break date) |
| [12] | `df.to_parquet("data/output/sentiment_df.parquet")` |

---

### `notebooks/5.0_results_and_viz.ipynb` — Paper Figures (RQ1–RQ3)

**Purpose:** Generate all 5 publication-ready figures at 300 dpi. Every figure in the paper must come from a cell in this notebook — never exported manually.

| Cell | Figure | Research Question |
|---|---|---|
| [1] | Load `sentiment_df.parquet` + topic model | — |
| [2] | **Figure 1:** Topic distribution by subreddit (stacked bar, top-10 topics) | RQ2 context |
| [3] | **Figure 2:** Topic evolution over time — top-5 topics as line chart, x=week | **RQ2** |
| [4] | **Figure 3:** Weekly RoBERTa sentiment share by specialisation subreddit (4-panel), break line at 30 Nov 2022 | **RQ3a** |
| [5] | **Figure 4:** ITS regression — junior cohort (cscareerquestions + csMajors) vs. senior (ExperiencedDevs) vs. control (cscareerquestionsEU), mark break date | **RQ3b** |
| [6] | **Figure 5:** VADER vs. RoBERTa scatter + agreement rate annotation | **RQ1** |
| [7] | `for fig in figures: fig.savefig(f"data/output/figures/{name}.png", dpi=300, bbox_inches="tight")` |
| [8] | Export summary tables (topic word lists, sentiment counts, ITS coefficients) to `data/output/tables/` as CSV |
| [9] | Print key statistics to cite in paper (n_docs, agreement_rate, break-date sentiment delta, ITS p-values) |

---

## Step 3 — pytest Test Suite

### `tests/test_text.py`

```python
from src.modeling.text import apply_pipeline
import pandas as pd

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
```

### `tests/test_dates.py`

```python
import pandas as pd
from src.modeling.load import parse_created_utc

def test_int_epoch_gives_2023():
    col = pd.Series([1700000000])          # 2023-11-14 in epoch seconds
    result = parse_created_utc(col)
    assert result.dt.year[0] == 2023, f"Expected 2023, got {result.dt.year[0]} (1970 = nanosecond bug)"

def test_float_epoch():
    col = pd.Series([1700000000.0])
    assert parse_created_utc(col).dt.year[0] == 2023

def test_iso_string():
    col = pd.Series(["2023-11-14T10:00:00"])
    assert parse_created_utc(col).dt.year[0] == 2023

def test_datetime_noop():
    col = pd.to_datetime(pd.Series(["2023-11-14"]))
    result = parse_created_utc(col)
    assert result.dt.year[0] == 2023
```

### `tests/test_cache.py`

```python
import numpy as np, tempfile, os
from src.modeling.cache import save_embeddings, load_embeddings

def test_embeddings_roundtrip():
    emb = np.random.rand(10, 384).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        save_embeddings(emb, key="test_sub_submissions_10", cache_dir=tmp)
        loaded = load_embeddings(key="test_sub_submissions_10", cache_dir=tmp)
    assert np.allclose(emb, loaded)

def test_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        assert load_embeddings("nonexistent", tmp) is None
```

### `tests/test_tracking.py`

```python
import json, tempfile
from src.modeling.tracking import log_experiment

def test_log_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = log_experiment({"seed": 42}, {"n_topics": 15}, output_dir=tmp)
        assert path.exists()

def test_log_appends():
    with tempfile.TemporaryDirectory() as tmp:
        log_experiment({"run": 1}, {"outlier_pct": 0.1}, output_dir=tmp)
        log_experiment({"run": 2}, {"outlier_pct": 0.2}, output_dir=tmp)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

def test_log_valid_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = log_experiment({"a": 1}, {"b": 2}, output_dir=tmp)
        line = path.read_text().strip()
        entry = json.loads(line)
        assert "timestamp" in entry and "params" in entry and "metrics" in entry
```

### `tests/test_sentiment.py`

```python
import pandas as pd
from src.modeling.sentiment import score_vader

def test_vader_positive():
    df = score_vader(["I love this job and I got a promotion!"])
    assert df["vader_label"][0] == "positive"

def test_vader_negative():
    df = score_vader(["I hate being unemployed. This is terrible."])
    assert df["vader_label"][0] == "negative"

def test_roberta_output_shape():
    # Shape test only — does not require GPU
    import numpy as np
    from unittest.mock import MagicMock, patch
    mock_logits = MagicMock()
    mock_logits.logits = __import__("torch").tensor([[0.1, 0.2, 0.7]])
    # Full GPU test is in the slow suite
```

### `tests/test_topics.py`

```python
import pytest
import numpy as np
from src.modeling.topics import topic_name_map, fit_topic_model

UMAP_PARAMS = {"n_neighbors": 5, "n_components": 2, "min_dist": 0.0, "metric": "cosine"}
HDBSCAN_PARAMS = {"min_cluster_size": 5, "min_samples": 2, "metric": "euclidean",
                  "cluster_selection_method": "eom", "prediction_data": True}

@pytest.mark.slow
def test_fit_finds_topics():
    docs = (["got a new software engineering job"] * 50 +
            ["layoffs announced at big tech company"] * 50 +
            ["interview prep leetcode grind"] * 50)
    from sentence_transformers import SentenceTransformer
    embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(docs)
    model, topics = fit_topic_model(docs, embeddings, UMAP_PARAMS, HDBSCAN_PARAMS, seed=42)
    assert len(set(topics) - {-1}) >= 1   # at least 1 non-outlier topic

@pytest.mark.slow
def test_topic_name_map_has_int_keys():
    docs = (["software job offer salary negotiation"] * 50 +
            ["laid off severance package"] * 50)
    from sentence_transformers import SentenceTransformer
    embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(docs)
    model, _ = fit_topic_model(docs, embeddings, UMAP_PARAMS, HDBSCAN_PARAMS, seed=42)
    name_map = topic_name_map(model)
    assert all(isinstance(k, int) for k in name_map)
    assert all(isinstance(v, str) for v in name_map.values())
```

---

## Step 4 — Requirements, `pytest.ini`, and `Makefile`

### `requirements_cpu.txt` additions

```diff
 bertopic==0.16.4
 ...
+gensim==4.3.3
+transformers==4.48.0
+tqdm==4.67.1
+vaderSentiment==3.3.2
+sentence-transformers==3.4.1
```

### `pytest.ini`

```ini
[pytest]
markers =
    slow: marks tests as slow — deselect with -m "not slow"
```

### `Makefile`

```makefile
.PHONY: test test-fast smoke-test lint compile clean all help

PYTHON ?= python3

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

test: ## Run all tests including slow
	pytest tests/ -v

test-fast: ## Run tests excluding slow (< 60 s)
	pytest tests/ -v -m "not slow"

smoke-test: ## Verify key imports work and timestamps parse correctly
	@$(PYTHON) -c "from src.modeling import set_seed; set_seed(42); print('set_seed OK')"
	@$(PYTHON) -c "\
from src.modeling.load import parse_created_utc; import pandas as pd; \
ts = parse_created_utc(pd.Series([1700000000])); \
assert ts.dt.year[0] == 2023, f'Expected 2023, got {ts.dt.year[0]}'; \
print('parse_created_utc OK — got', ts[0])"

lint: ## py_compile all source files
	@find src/ -name '*.py' -exec $(PYTHON) -m py_compile {} \; && echo "lint OK"

compile: ## compileall on src/
	$(PYTHON) -m compileall -q src/ && echo "compile OK"

clean: ## Remove __pycache__ and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	@echo "clean OK"

all: compile lint test-fast smoke-test ## Full pre-commit check (excludes slow tests)
```

---

## Verification Plan

### Automated Tests

```bash
# Run from inside AI Implemented/
make smoke-test          # ~5 seconds — verifies set_seed + parse_created_utc
make test-fast           # ~30–60 seconds — all non-slow tests
make test                # full suite including BERTopic fit on synthetic docs
```

**Pass criteria:**

| Check | Pass condition |
|---|---|
| `smoke-test` | Prints `set_seed OK` and `parse_created_utc OK` with a 2023 date (not 1970) |
| `test-fast` | 0 failures, 0 errors |
| `test` | 0 failures; `test_topics.py` finds ≥ 1 non-outlier topic |

### Notebook Verification

| Notebook | Pass condition |
|---|---|
| `2.0_sampling` | Date range shows 2020–2025; all 8 subreddits listed (or explicit "missing" warning for ExperiencedDevs / csMajors if not yet scraped) |
| `3.0_topic_modeling` | `experiment_log.jsonl` has one line per sweep config; best config has outlier_pct < 20%; topic names are readable strings |
| `4.0_sentiment_analysis` | Agreement rate printed and ≥ 60%; `sentiment_df.parquet` saved |
| `5.0_results_and_viz` | 5 PNG files saved to `data/output/figures/` at 300 dpi; break date (30 Nov 2022) visible on temporal figures |

---

## Subreddit Reference

**In scope for this implementation pass (6 — raw data exists in `data/raw/`):**

| Subreddit | Analytical Role | Cohort / Split |
|---|---|---|
| `r/cscareerquestions` | Entry-level / general job-seeker | Specialisation + Junior (RQ3b future) |
| `r/cscareerquestionsEU` | European market | Geographic control |
| `r/cybersecurity` | Security domain | Specialisation (RQ3a) |
| `r/dataengineering` | Data engineering | Specialisation (RQ3a) |
| `r/datascience` | Data science | Specialisation (RQ3a) |
| `r/SoftwareEngineering` | General SWE | Specialisation (RQ3a) |

**Out of scope (no raw data yet — future scraping task):**

| Subreddit | Why needed |
|---|---|
| `r/ExperiencedDevs` | Senior cohort for RQ3b seniority split |
| `r/csMajors` | Student / pre-career cohort for RQ3b seniority split |

---

## Locked Decisions (inherited from `plans/research-plan.md`)

| Decision | Value |
|---|---|
| Break date | 30 November 2022 |
| Analysis window | 2020–2025 |
| Sentiment primary model | `cardiffnlp/twitter-roberta-base-sentiment` |
| Sentiment cross-check | VADER (`\|compound\| > 0.05` neutral band) |
| Topic modeling | BERTopic on submission titles |
| Sentiment scoring | On comments only |
| Reproducibility seed | 42 everywhere |
| No DeBERTa fine-tuning | Out of scope |
| No Granger causality | Out of scope |
| No Streamlit dashboard | Out of scope |
