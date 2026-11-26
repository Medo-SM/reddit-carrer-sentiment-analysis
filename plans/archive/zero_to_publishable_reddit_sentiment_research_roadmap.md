# From Zero to Publishable Paper: Longitudinal Reddit Sentiment Analysis (2018–2022)

This guide provides a complete end-to-end blueprint for building a reproducible Reddit sentiment analysis pipeline from scratch and elevating it into a peer-reviewed Computational Social Science (CSS) or Natural Language Processing (NLP) research paper.

---

## Phase 0: Environment Setup & Reproducible Architecture

Peer-reviewed computational research requires a clean, modular repository that separates raw data, configuration parameters, processing logic, statistical validation, and manuscript figures.

### Step 0.1: Create the Project Directory Structure
Open your terminal and create the following directory tree:

```bash
mkdir -p reddit-sentiment-paper/{config,data/{raw,processed,validation,results},src,figures,paper}
cd reddit-sentiment-paper
touch config/config.yaml
touch src/{__init__.py,ingestion.py,cleaning.py,scoring.py,aggregation.py,visualization.py,validation.py,statistics.py,topics.py}
touch main.py requirements.txt
```

Your workspace will look like this:

```text
reddit-sentiment-paper/
├── config/
│   └── config.yaml                # Central paths, dates, and model parameters
├── data/
│   ├── raw/                       # Raw .parquet Reddit dumps (read-only)
│   ├── processed/                 # Cleaned and scored .parquet checkpoints
│   ├── validation/                # Stratified samples for human annotation
│   └── results/                   # Quarterly aggregated CSV/Parquet tables
├── figures/                       # 300-DPI publication-ready plots (.png / .pdf)
├── paper/                         # LaTeX or Word manuscript files
├── src/
│   ├── __init__.py
│   ├── ingestion.py               # Parquet loading, date filtering, deduplication
│   ├── cleaning.py                # Sentiment-safe Reddit text cleaner
│   ├── scoring.py                 # VADER baseline + Hugging Face RoBERTa scorer
│   ├── aggregation.py             # Quarterly unweighted & upvote-weighted metrics
│   ├── visualization.py           # Multi-panel matplotlib time-series figures
│   ├── validation.py              # Stratified sampling & Cohen's Kappa / F1
│   ├── statistics.py              # Mann-Kendall, PELT structural breaks, Wilcoxon
│   └── topics.py                  # Log-odds keyword drivers per quarter
├── main.py                        # Master pipeline orchestrator
└── requirements.txt               # Pinned Python dependencies
```

### Step 0.2: Set Up Virtual Environment & Dependencies
Add the following packages to `requirements.txt`:

```text
pandas>=2.1.0
pyarrow>=14.0.0
numpy>=1.26.0
pyyaml>=6.0.1
vaderSentiment>=3.3.2
transformers>=4.36.0
torch>=2.1.0
tqdm>=4.66.0
matplotlib>=3.8.0
scipy>=1.11.0
scikit-learn>=1.3.0
pymannkendall>=1.4.3
ruptures>=1.1.9
```

Create and activate your virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 0.3: Create `config/config.yaml`
Never hardcode file paths or date cutoffs inside your Python classes. Populate `config/config.yaml` as follows:

```yaml
paths:
  raw_data_dir: "data/raw"
  cleaned_parquet: "data/processed/cleaned_reddit.parquet"
  scored_parquet: "data/processed/scored_reddit.parquet"
  quarterly_summary_csv: "data/results/quarterly_summary.csv"
  quarterly_by_sub_csv: "data/results/quarterly_by_subreddit.csv"
  validation_sample_csv: "data/validation/human_annotation_sample.csv"
  main_figure_path: "figures/quarterly_sentiment_trends.png"

parameters:
  # Use complete calendar quarters so edge quarters are not skewed by partial data
  start_date: "2018-01-01"
  end_date: "2021-12-31"
  columns:
    - "id"
    - "subreddit"
    - "created_utc"
    - "score"
    - "text"
  model_type: "vader"              # Options: "vader", "roberta", or "both"
  roberta_model: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  batch_size: 64
  validation_sample_size: 300
  random_seed: 42
```

---

## Phase 1: Data Ingestion & Temporal Standardization (`src/ingestion.py`)

Raw Reddit dumps often span multiple `.parquet` files, contain duplicate submission IDs, or include partial days outside your target window.

### Key Engineering Fixes
1. **Concatenate all files:** Collect filtered DataFrames in a list and combine them with `pd.concat()` *outside* the file loop.
2. **Complete quarters only:** Ensure `start_date` and `end_date` align with full 3-month quarterly boundaries (e.g., `2018-01-01` through `2021-12-31` for $16$ complete quarters, or through `2022-12-31` for $20$ complete quarters).
3. **Deduplication:** Drop duplicate `id` rows and sort chronologically.

### Code for `src/ingestion.py`

```python
import os
import pandas as pd


class ParquetHandling:
    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir

    def get_parquet_files(self) -> list[str]:
        """Recursively discovers all .parquet files in the raw data directory."""
        parquet_files = []
        for dir_path, _, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(dir_path, file))
        return sorted(parquet_files)

    def load_and_filter_by_date(
        self,
        parquet_paths: list[str],
        cols: list[str] | None = None,
        start_date: str = "2018-01-01",
        end_date: str = "2021-12-31",
    ) -> pd.DataFrame:
        """
        Loads multiple Parquet files, filters each by UTC date range,
        concatenates them, and deduplicates by Reddit submission ID.
        """
        if not parquet_paths:
            raise FileNotFoundError(f"No .parquet files found in {self.raw_dir}")

        dfs = []
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        for file_path in parquet_paths:
            df = pd.read_parquet(file_path, columns=cols)

            # Ensure created_utc is datetime64[ns]
            if not pd.api.types.is_datetime64_any_dtype(df["created_utc"]):
                df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")

            mask = (df["created_utc"] >= start_ts) & (df["created_utc"] <= end_ts)
            dfs.append(df.loc[mask])

        combined_df = pd.concat(dfs, ignore_index=True)

        # Deduplicate by submission/comment ID and sort chronologically
        combined_df = (
            combined_df.drop_duplicates(subset=["id"])
            .sort_values("created_utc")
            .reset_index(drop=True)
        )
        return combined_df
```

---

## Phase 2: Sentiment-Safe Text Preprocessing (`src/cleaning.py`)

### Theoretical Justification for Your Paper's Methodology Section
In standard topic modeling, researchers strip stopwords, punctuation, emojis, and capitalization. **In sentiment analysis, aggressive preprocessing introduces severe measurement error:**
* Removing stopwords deletes negations (`"not"`, `"no"`, `"never"`), inverting polarity from negative to positive.
* Removing punctuation and casing erases emotional intensifiers (`"AMAZING!!!"` vs. `"amazing"`).
* Removing non-word characters (`\W+`) deletes emojis and emoticons (`:)`, `:(`, `😡`).

Therefore, our sentiment-safe cleaner removes **structural platform artifacts** (HTML entities, URLs, Markdown syntax, bot boilerplate, and moderator deletion tags) while preserving human linguistic expression.

### Code for `src/cleaning.py`

```python
import html
import re
import pandas as pd


class SentimentTextCleaner:
    """
    Cleans raw Reddit text for sentiment analysis while preserving
    negations, capitalization, emojis, and expressive punctuation.
    """

    BOT_PATTERNS = [
        "I am a bot",
        "[removed by moderator]",
        "[deleted by user]",
        "This action was performed automatically",
    ]
    EMPTY_MARKERS = {"[deleted]", "[removed]", ""}

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        # 1. Decode HTML entities (&amp; -> &, &gt; -> >)
        text = html.unescape(text)

        # 2. Convert Markdown links [anchor text](https://...) -> anchor text
        text = re.sub(
            r"\[([^\]]+)\]\((?:https?://|www\.)\S+\)",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )

        # 3. Remove standalone URLs (escaped dot in www\. prevents matching "awww")
        text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)

        # 4. Remove Reddit Markdown formatting symbols without stripping ! or ?
        text = re.sub(r"[*_#`~>|]+", " ", text)

        # 5. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_dataframe(self, df: pd.DataFrame, col: str = "text") -> pd.DataFrame:
        df = df.copy()

        # Build boolean mask for noise/bot rows
        is_noise = df[col].isna() | df[col].str.strip().isin(self.EMPTY_MARKERS)
        for pattern in self.BOT_PATTERNS:
            is_noise = is_noise | df[col].str.contains(pattern, case=False, regex=False, na=False)

        df = df.loc[~is_noise].copy()
        df["clean_text"] = df[col].apply(self.clean_text)

        # Drop rows that became empty after stripping links/markdown
        df = df.loc[df["clean_text"] != ""].reset_index(drop=True)
        return df
```

---

## Phase 3: Multi-Model Sentiment Scoring (`src/scoring.py`)

To make your work publishable, avoid relying on a single lexicon. Reviewers strongly favor comparing a **rule-based lexicon baseline (VADER)** against a **contextual Transformer model (RoBERTa)**:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner):** Computes a normalized compound score $s_{\text{vader}} \in [-1, 1]$ using valence lexicons, capitalization heuristics, degree modifiers, and negation rules:
   $$s_{\text{vader}} = \frac{x}{\sqrt{x^2 + \alpha}}, \quad \alpha = 15$$
   where $x$ is the sum of valence scores of each token in the post. Standard classification thresholds are:
   * Positive: $s_{\text{vader}} \ge 0.05$
   * Negative: $s_{\text{vader}} \le -0.05$
   * Neutral: $-0.05 < s_{\text{vader}} < 0.05$

2. **RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`):** Produces softmax probabilities $P(\text{neg}), P(\text{neu}), P(\text{pos})$ where $\sum P = 1$. We define a continuous Transformer compound score $s_{\text{roberta}} \in [-1, 1]$ as:
   $$s_{\text{roberta}} = P(\text{pos}) - P(\text{neg})$$
   and assign the discrete label via $\arg\max_{c \in \{\text{neg}, \text{neu}, \text{pos}\}} P(c)$.

### Code for `src/scoring.py`

```python
import numpy as np
import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentScorer:
    def __init__(
        self,
        model_type: str = "vader",
        roberta_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        batch_size: int = 64,
    ):
        self.model_type = model_type.lower()
        self.roberta_model_name = roberta_model_name
        self.batch_size = batch_size
        self.vader = SentimentIntensityAnalyzer()

    @staticmethod
    def _vader_label(compound: float) -> str:
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        return "neutral"

    def score_vader(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
        df = df.copy()
        compounds = [
            self.vader.polarity_scores(text)["compound"]
            for text in tqdm(df[text_col], desc="Scoring with VADER")
        ]
        df["vader_compound"] = compounds
        df["vader_label"] = [self._vader_label(c) for c in compounds]

        # Set primary columns used downstream
        df["compound"] = df["vader_compound"]
        df["sentiment_label"] = df["vader_label"]
        return df

    def score_roberta(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        df = df.copy()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.roberta_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.roberta_model_name).to(device)
        model.eval()

        texts = df[text_col].tolist()
        roberta_compounds = []
        roberta_labels = []
        id2label = {0: "negative", 1: "neutral", 2: "positive"}

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Scoring RoBERTa ({device})"):
                batch_texts = texts[i : i + self.batch_size]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                # Continuous compound score = P(pos) - P(neg)
                batch_compound = probs[:, 2] - probs[:, 0]
                batch_label = [id2label[idx] for idx in np.argmax(probs, axis=1)]

                roberta_compounds.extend(batch_compound.tolist())
                roberta_labels.extend(batch_label)

        df["roberta_compound"] = roberta_compounds
        df["roberta_label"] = roberta_labels

        # Set primary columns used downstream
        df["compound"] = df["roberta_compound"]
        df["sentiment_label"] = df["roberta_label"]
        return df

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
        if self.model_type == "vader":
            return self.score_vader(df, text_col)
        elif self.model_type == "roberta":
            return self.score_roberta(df, text_col)
        elif self.model_type == "both":
            df = self.score_vader(df, text_col)
            df = self.score_roberta(df, text_col)
            return df
        raise ValueError(f"Unsupported model_type: {self.model_type}")
```

---

## Phase 4: Quarterly Aggregation & Upvote-Weighted Metrics (`src/aggregation.py`)

### Mathematical Definitions for Your Paper
For each quarter $q \in \{2018\text{Q}1, \dots, 2021\text{Q}4\}$ containing $N_q$ posts:

1. **Unweighted Quarterly Mean Sentiment ($\bar{S}_q$):**
   $$\bar{S}_q = \frac{1}{N_q} \sum_{i=1}^{N_q} s_i$$

2. **95% Confidence Interval ($\text{CI}_{95\%, q}$):**
   $$\text{CI}_{95\%, q} = \bar{S}_q \pm 1.96 \cdot \frac{\sigma_q}{\sqrt{N_q}}$$
   where $\sigma_q$ is the sample standard deviation of sentiment scores in quarter $q$.

3. **Upvote-Weighted Quarterly Sentiment ($\bar{S}_{q,\text{w}}$):**
   Because Reddit's voting mechanism surfaces community consensus, an unvoted post ($score = 1$) and a viral post ($score = 2500$) do not reflect equal community endorsement. We define non-negative post weight $w_i = \max(0, \text{score}_i) + 1$ and compute:
   $$\bar{S}_{q,\text{w}} = \frac{\sum_{i=1}^{N_q} w_i s_i}{\sum_{i=1}^{N_q} w_i}$$

4. **Polarization & Net Sentiment Ratio ($\text{NSR}_q$):**
   $$\text{NSR}_q = \frac{N_{q,\text{pos}} - N_{q,\text{neg}}}{N_q} \times 100$$

### Code for `src/aggregation.py`

```python
import numpy as np
import pandas as pd


class QuarterlyAggregator:
    def add_quarter_and_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["year_quarter"] = df["created_utc"].dt.to_period("Q").astype(str)
        df["weight"] = df["score"].clip(lower=0) + 1
        df["weighted_compound"] = df["compound"] * df["weight"]
        return df

    def _aggregate_group(self, df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        df = self.add_quarter_and_weights(df)

        summary = (
            df.groupby(group_cols, as_index=False)
            .agg(
                post_count=("id", "count"),
                mean_sentiment=("compound", "mean"),
                median_sentiment=("compound", "median"),
                std_sentiment=("compound", "std"),
                weighted_sum=("weighted_compound", "sum"),
                total_weight=("weight", "sum"),
                pct_positive=("sentiment_label", lambda s: (s == "positive").mean() * 100),
                pct_negative=("sentiment_label", lambda s: (s == "negative").mean() * 100),
                pct_neutral=("sentiment_label", lambda s: (s == "neutral").mean() * 100),
            )
            .sort_values(group_cols)
        )

        summary["weighted_sentiment"] = summary["weighted_sum"] / summary["total_weight"]
        summary["net_sentiment_ratio"] = summary["pct_positive"] - summary["pct_negative"]
        summary["ci_95"] = 1.96 * (summary["std_sentiment"] / np.sqrt(summary["post_count"]))

        return summary.drop(columns=["weighted_sum", "total_weight"]).reset_index(drop=True)

    def summarize_overall_by_quarter(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._aggregate_group(df, ["year_quarter"])

    def summarize_by_quarter_and_subreddit(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._aggregate_group(df, ["year_quarter", "subreddit"])
```

---

## Phase 5: Publication-Grade Visualization & Pipeline Orchestration

### Step 5.1: Code for `src/visualization.py`

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class QuarterlyVisualizer:
    def plot_quarterly_trends(
        self,
        summary: pd.DataFrame,
        output_path: str = "figures/quarterly_sentiment_trends.png",
    ) -> None:
        quarters = summary["year_quarter"].tolist()
        x = np.arange(len(quarters))

        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(13, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 2.0, 1.3]},
        )

        # Panel 1: Unweighted vs. Upvote-Weighted Mean Sentiment + 95% CI
        ax1.plot(
            x,
            summary["mean_sentiment"],
            marker="o",
            color="#1f77b4",
            linewidth=2.2,
            label="Unweighted Mean Sentiment",
        )
        ax1.fill_between(
            x,
            summary["mean_sentiment"] - summary["ci_95"],
            summary["mean_sentiment"] + summary["ci_95"],
            color="#1f77b4",
            alpha=0.2,
            label="95% Confidence Interval",
        )
        ax1.plot(
            x,
            summary["weighted_sentiment"],
            marker="s",
            linestyle="--",
            color="#ff7f0e",
            linewidth=1.8,
            label="Upvote-Weighted Sentiment",
        )
        ax1.axhline(0, color="black", linestyle=":", linewidth=1)
        ax1.set_ylabel("Compound Score")
        ax1.set_title(
            "Longitudinal Reddit Sentiment Dynamics by Quarter (2018–2022)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        # Panel 2: Polarization (% Positive vs. % Negative Share)
        ax2.plot(
            x,
            summary["pct_positive"],
            marker="^",
            color="#2ca02c",
            linewidth=2,
            label="% Positive Posts",
        )
        ax2.plot(
            x,
            summary["pct_negative"],
            marker="v",
            color="#d62728",
            linewidth=2,
            label="% Negative Posts",
        )
        ax2.set_ylabel("Share of Posts (%)")
        ax2.set_title(
            "Affective Polarization: Quarterly Positive vs. Negative Share",
            fontsize=11,
            fontweight="bold",
        )
        ax2.legend(loc="upper left", frameon=True)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

        # Panel 3: Quarterly Post Volume
        ax3.bar(x, summary["post_count"], color="#6c757d", alpha=0.8, width=0.6)
        ax3.set_ylabel("Post Volume")
        ax3.set_xlabel("Year-Quarter")
        ax3.set_title("Quarterly Submission Volume ($N_q$)", fontsize=11, fontweight="bold")
        ax3.grid(axis="y", linestyle="--", alpha=0.4)

        plt.xticks(x, quarters, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
```

### Step 5.2: Master Pipeline Orchestrator (`main.py`)

```python
import os
import yaml
from src.aggregation import QuarterlyAggregator
from src.cleaning import SentimentTextCleaner
from src.ingestion import ParquetHandling
from src.scoring import SentimentScorer
from src.visualization import QuarterlyVisualizer


def main() -> None:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    params = config["parameters"]

    # Ensure output directories exist
    for key in ["cleaned_parquet", "scored_parquet", "quarterly_summary_csv", "main_figure_path"]:
        os.makedirs(os.path.dirname(paths[key]), exist_ok=True)

    # Stage 1: Ingest & Filter by Date
    print("--- Stage 1: Loading & Filtering Raw Parquet Files ---")
    ingestor = ParquetHandling(paths["raw_data_dir"])
    parquet_files = ingestor.get_parquet_files()
    raw_df = ingestor.load_and_filter_by_date(
        parquet_paths=parquet_files,
        cols=params["columns"],
        start_date=params["start_date"],
        end_date=params["end_date"],
    )
    print(f"Loaded {len(raw_df):,} unique posts between {params['start_date']} and {params['end_date']}.")

    # Stage 2: Clean Text
    print("--- Stage 2: Running Sentiment-Safe Text Cleaner ---")
    cleaner = SentimentTextCleaner()
    clean_df = cleaner.clean_dataframe(raw_df, col="text")
    clean_df.to_parquet(paths["cleaned_parquet"], index=False)
    print(f"Saved {len(clean_df):,} cleaned rows to {paths['cleaned_parquet']}.")

    # Stage 3: Score Sentiment
    print(f"--- Stage 3: Scoring Sentiment ({params['model_type']}) ---")
    scorer = SentimentScorer(
        model_type=params["model_type"],
        roberta_model_name=params["roberta_model"],
        batch_size=params["batch_size"],
    )
    scored_df = scorer.score_dataframe(clean_df, text_col="clean_text")
    scored_df.to_parquet(paths["scored_parquet"], index=False)

    # Stage 4: Aggregate by Year-Quarter
    print("--- Stage 4: Aggregating by Year-Quarter ---")
    aggregator = QuarterlyAggregator()
    quarterly_summary = aggregator.summarize_overall_by_quarter(scored_df)
    quarterly_by_sub = aggregator.summarize_by_quarter_and_subreddit(scored_df)

    quarterly_summary.to_csv(paths["quarterly_summary_csv"], index=False)
    quarterly_by_sub.to_csv(paths["quarterly_by_sub_csv"], index=False)
    print(quarterly_summary[["year_quarter", "post_count", "mean_sentiment", "weighted_sentiment"]])

    # Stage 5: Visualize
    print("--- Stage 5: Generating Publication Figures ---")
    visualizer = QuarterlyVisualizer()
    visualizer.plot_quarterly_trends(quarterly_summary, output_path=paths["main_figure_path"])
    print(f"Figure saved to {paths['main_figure_path']}.")


if __name__ == "__main__":
    main()
```

---

## Phase 6: Research Upgrade 1 — Human Gold-Standard Validation (`src/validation.py`)

### Why Peer Reviewers Require This
Off-the-shelf sentiment models can misclassify domain-specific Reddit slang or sarcasm. To prove your sentiment scores are trustworthy, create a **stratified random sample of $n = 300$ posts**, have two humans independently label each post (`positive`, `neutral`, `negative`), and report:
1. **Inter-Annotator Agreement (Cohen's Kappa, $\kappa$):**
   $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
   where $p_o$ is observed agreement between annotators and $p_e$ is expected chance agreement ($\kappa \ge 0.70$ indicates substantial agreement).
2. **Macro-$F_1$ and Accuracy** of VADER and RoBERTa against the human consensus labels.

### Code for `src/validation.py`

```python
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score


class GoldStandardValidator:
    def export_stratified_sample(
        self,
        scored_df: pd.DataFrame,
        output_csv: str,
        sample_size: int = 300,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Exports a stratified sample across year_quarter so human annotators
        can label ground-truth sentiment without seeing model predictions.
        """
        df = scored_df.copy()
        df["year_quarter"] = df["created_utc"].dt.to_period("Q").astype(str)

        # Stratified sample proportional to quarter volume
        sample_df = (
            df.groupby("year_quarter", group_keys=False)
            .apply(lambda x: x.sample(frac=min(1.0, sample_size / len(df)), random_state=random_seed))
            .reset_index(drop=True)
        )

        # Export blind file for human raters
        blind_cols = ["id", "subreddit", "year_quarter", "clean_text"]
        blind_df = sample_df[blind_cols].copy()
        blind_df["annotator_1_label"] = ""
        blind_df["annotator_2_label"] = ""
        blind_df["consensus_label"] = ""
        blind_df.to_csv(output_csv, index=False)
        return sample_df

    def evaluate_annotations(
        self, annotated_csv: str, scored_df: pd.DataFrame
    ) -> dict[str, float]:
        """
        Computes Cohen's Kappa between human raters and evaluates model Macro-F1
        against human consensus labels.
        """
        human_df = pd.read_csv(annotated_csv)
        merged = pd.merge(
            human_df,
            scored_df[["id", "vader_label", "roberta_label"]],
            on="id",
            how="inner",
        )

        kappa = cohen_kappa_score(merged["annotator_1_label"], merged["annotator_2_label"])
        print(f"Inter-Annotator Agreement (Cohen's Kappa): {kappa:.3f}\n")

        if "vader_label" in merged.columns:
            print("=== VADER vs. Human Consensus ===")
            print(classification_report(merged["consensus_label"], merged["vader_label"], digits=3))

        if "roberta_label" in merged.columns:
            print("=== RoBERTa vs. Human Consensus ===")
            print(classification_report(merged["consensus_label"], merged["roberta_label"], digits=3))

        return {"cohen_kappa": kappa}
```

---

## Phase 7: Research Upgrade 2 — Statistical Hypothesis Testing (`src/statistics.py`)

Visual inspection of a chart is insufficient for a research paper. You need formal statistical tests to answer three core questions:

1. **Is there a statistically significant monotonic trend across quarters?**
   * **Test:** Mann-Kendall Non-Parametric Trend Test (reports Kendall's $\tau \in [-1, 1]$, Sen's slope, and $p$-value).
2. **At which exact quarter(s) did community sentiment undergo a structural regime shift?**
   * **Test:** PELT (Pruned Exact Linear Time) change-point detection on the quarterly sentiment series.
3. **Does upvote-weighted sentiment differ significantly from unweighted sentiment?**
   * **Test:** Paired Wilcoxon Signed-Rank Test across quarters ($H_0: \text{median}(\bar{S}_{q,\text{w}} - \bar{S}_q) = 0$).

### Code for `src/statistics.py`

```python
import numpy as np
import pandas as pd
import pymannkendall as mk
import ruptures as rpt
from scipy.stats import kruskal, wilcoxon


class QuarterlyStatisticalTests:
    def run_mann_kendall_trend(self, quarterly_summary: pd.DataFrame) -> dict:
        """Tests for a monotonic upward or downward trend across quarters."""
        result = mk.original_test(quarterly_summary["mean_sentiment"].values)
        return {
            "trend": result.trend,
            "p_value": float(result.p),
            "kendall_tau": float(result.Tau),
            "sens_slope": float(result.slope),
        }

    def detect_structural_breaks(
        self, quarterly_summary: pd.DataFrame, penalty: float = 1.5
    ) -> list[str]:
        """
        Uses PELT change-point detection to identify exact quarters where
        mean sentiment shifted regimes.
        """
        signal = quarterly_summary["mean_sentiment"].values
        signal_std = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        algo = rpt.Pelt(model="rbf", min_size=2).fit(signal_std)
        change_indices = algo.predict(pen=penalty)

        quarters = quarterly_summary["year_quarter"].tolist()
        break_quarters = [quarters[idx - 1] for idx in change_indices if idx < len(quarters)]
        return break_quarters

    def test_upvote_amplification(self, quarterly_summary: pd.DataFrame) -> dict:
        """
        Paired Wilcoxon signed-rank test comparing unweighted mean sentiment
        against upvote-weighted sentiment across quarters.
        """
        stat, p_val = wilcoxon(
            quarterly_summary["weighted_sentiment"],
            quarterly_summary["mean_sentiment"],
        )
        mean_diff = (
            quarterly_summary["weighted_sentiment"] - quarterly_summary["mean_sentiment"]
        ).mean()
        return {
            "wilcoxon_stat": float(stat),
            "p_value": float(p_val),
            "mean_amplification_diff": float(mean_diff),
        }

    def test_subreddit_heterogeneity(self, scored_df: pd.DataFrame) -> dict:
        """Kruskal-Wallis H-test checking if sentiment differs across subreddits."""
        groups = [group["compound"].values for _, group in scored_df.groupby("subreddit")]
        if len(groups) < 2:
            return {"kruskal_h": np.nan, "p_value": np.nan}
        stat, p_val = kruskal(*groups)
        return {"kruskal_h": float(stat), "p_value": float(p_val)}
```

---

## Phase 8: Research Upgrade 3 — Explaining Shifts via Topic Drivers (`src/topics.py`)

When your quarterly plot shows a sharp sentiment drop (e.g., in `2020Q1` or `2021Q3`), reviewers will ask: **"What topics drove that negative shift?"**

We answer this mathematically using **Monroe et al.'s (2008) Fightin' Words Log-Odds Ratio with an Uninformative Dirichlet Prior**, which identifies terms disproportionately overrepresented in negative posts during a target quarter $q$ compared to the rest of the corpus.

For word $w$ in group $i$ vs. background $j$:
$$\delta_w^{(i-j)} = \log \left( \frac{y_{iw} + \alpha_w}{n_i + \alpha_0 - (y_{iw} + \alpha_w)} \right) - \log \left( \frac{y_{jw} + \alpha_w}{n_j + \alpha_0 - (y_{jw} + \alpha_w)} \right)$$
$$z_w = \frac{\delta_w^{(i-j)}}{\sqrt{\frac{1}{y_{iw} + \alpha_w} + \frac{1}{y_{jw} + \alpha_w}}}$$

### Code for `src/topics.py`

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class QuarterlyTopicDrivers:
    def get_distinctive_terms_for_quarter(
        self,
        scored_df: pd.DataFrame,
        target_quarter: str,
        sentiment_filter: str = "negative",
        top_k: int = 15,
    ) -> pd.DataFrame:
        """
        Computes z-scored log-odds ratios to identify which words/bigrams
        drove negative (or positive) sentiment in a specific quarter.
        """
        df = scored_df.copy()
        df["year_quarter"] = df["created_utc"].dt.to_period("Q").astype(str)

        subset = df[df["sentiment_label"] == sentiment_filter]
        target_texts = subset.loc[subset["year_quarter"] == target_quarter, "clean_text"]
        bg_texts = subset.loc[subset["year_quarter"] != target_quarter, "clean_text"]

        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=10,
            max_features=10000,
        )
        vectorizer.fit(subset["clean_text"])

        y_target = np.asarray(vectorizer.transform(target_texts).sum(axis=0)).flatten()
        y_bg = np.asarray(vectorizer.transform(bg_texts).sum(axis=0)).flatten()
        vocab = np.array(vectorizer.get_feature_names_out())

        # Prior alpha = 1 (Laplace/Dirichlet smoothing)
        alpha = np.ones_like(y_target, dtype=float)
        alpha_0 = alpha.sum()

        n_target = y_target.sum()
        n_bg = y_bg.sum()

        log_odds_target = np.log((y_target + alpha) / (n_target + alpha_0 - y_target - alpha))
        log_odds_bg = np.log((y_bg + alpha) / (n_bg + alpha_0 - y_bg - alpha))

        delta = log_odds_target - log_odds_bg
        variance = (1.0 / (y_target + alpha)) + (1.0 / (y_bg + alpha))
        z_scores = delta / np.sqrt(variance)

        top_indices = np.argsort(z_scores)[::-1][:top_k]
        return pd.DataFrame(
            {
                "quarter": target_quarter,
                "sentiment": sentiment_filter,
                "term": vocab[top_indices],
                "z_score": z_scores[top_indices],
                "count_in_quarter": y_target[top_indices],
            }
        )
```

---

## Phase 9: Writing & Submitting the Research Manuscript (IMRaD Blueprint)

Once your pipeline generates the CSV tables, validation metrics, statistical test outputs, and 300-DPI plots, structure your paper using the standard **IMRaD** (Introduction, Methodology, Results, and Discussion) format.

### 1. Title & Abstract Formula
* **Title Template:** *Longitudinal Affective Dynamics on Reddit (2018–2022): A Quarterly Multi-Model Sentiment and Structural Break Analysis of [Your Domain/Subreddits]*
* **Abstract (150–250 words, 5 sentences):**
  1. *Context:* Online communities on Reddit serve as real-time barometers of public discourse around [your topic].
  2. *Gap:* However, prior studies frequently rely on short time horizons, destructive text preprocessing, or unweighted sentiment averages that ignore community voting dynamics.
  3. *Dataset & Methods:* Analyzing $N = 143,138$ submissions across $16$ calendar quarters ($2018\text{Q}1$–$2021\text{Q}4$), we compare a sentiment-safe VADER baseline against a domain-tuned RoBERTa Transformer validated on a human-annotated gold standard ($\kappa = \dots$).
  4. *Results:* Using Mann-Kendall trend tests, PELT structural break detection, and upvote-weighted sentiment aggregation, we identify statistically significant affective shifts in [Quarter X] ($\tau = \dots, p < 0.01$) driven by [top log-odds topics].
  5. *Impact:* Furthermore, we demonstrate that upvote weighting significantly amplifies [negative/positive] sentiment ($p < 0.001$), highlighting the role of platform curation in shaping perceived community mood.

### 2. Explicit Research Questions (Include at the End of Section 1: Introduction)
* **$\text{RQ}_1$ (Longitudinal Trajectory & Structural Breaks):** *How did community sentiment evolve across quarters between 2018 and 2022, and which specific quarters represent statistically significant regime shifts?*
* **$\text{RQ}_2$ (Collective Endorsement vs. Individual Expression):** *Does upvote-weighted sentiment ($\bar{S}_{q,\text{w}}$) diverge significantly from unweighted post sentiment ($\bar{S}_q$), indicating that community voting amplifies affective negativity or positivity?*
* **$\text{RQ}_3$ (Lexical Drivers of Affective Shifts):** *Which distinctive discourse themes ($n$-grams with highest log-odds $z$-scores) account for the sharpest quarterly sentiment contractions?*

### 3. Essential Tables & Figures to Include in the Manuscript
* **Table 1 (Dataset Summary):** Number of raw posts, posts removed by bot/moderator filters, final cleaned posts ($N$), date range, and breakdown by subreddit.
* **Table 2 (Gold-Standard Validation Benchmark):** Cohen's $\kappa$ between human annotators, plus Precision, Recall, and Macro-$F_1$ of VADER vs. RoBERTa on the $300$-post human validation set.
* **Figure 1 (The 3-Panel Matplotlib Chart):** Panel A (Unweighted vs. Upvote-Weighted Mean Sentiment with $95\%$ CI bands and vertical dashed lines marking PELT structural breaks), Panel B ($\%$ Positive vs. $\%$ Negative polarization share), and Panel C (Quarterly Post Volume $N_q$).
* **Table 3 (Statistical Hypothesis Tests):** Summary of Mann-Kendall $\tau$, Sen's slope, PELT break quarters, and Paired Wilcoxon test results.
* **Table 4 (Distinctive Topic Drivers):** Top $10$ overrepresented bigrams/unigrams ($z$-scores) during the most negative and most positive quarters.

### 4. Ethics, Reproducibility & Limitations Section (Mandatory for Peer Review)
* **Ethical Considerations:** State that all data analyzed consists of publicly accessible Reddit submissions, no private messages were accessed, and no individual usernames (`author`) are published orquoted verbatim in the manuscript to prevent reverse-search re-identification (aligned with AAAI ICWSM / AoIR ethics guidelines).
* **Threats to Validity (Limitations):**
  1. *Survivorship Bias:* Posts deleted by users or moderators prior to data collection (`[deleted]` / `[removed]`) cannot be recovered, potentially underrepresenting extreme or rule-violating toxicity.
  2. *Demographic Skew:* Reddit's user base is not a representative sample of the general population.
  3. *Sarcasm & Irony:* Even contextual Transformers remain imperfect at detecting multi-turn community inside jokes without parent-thread context.

### 5. Recommended Peer-Reviewed Target Venues
* **Conferences (archival proceedings):**
  * **AAAI ICWSM** (*International AAAI Conference on Web and Social Media*)
  * **ACM CSCW** (*Computer-Supported Cooperative Work and Social Computing*)
  * **IEEE/ACM ASONAM** (*Advances in Social Networks Analysis and Mining*)
  * **SocInfo** (*International Conference on Social Informatics*)
* **Journals:**
  * **EPJ Data Science** (Springer)
  * **PLOS ONE**
  * **Online Social Networks and Media** (Elsevier)
  * **Journal of Computational Social Science** (Springer)