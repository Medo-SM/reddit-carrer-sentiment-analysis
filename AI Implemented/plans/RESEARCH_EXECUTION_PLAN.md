# AI Research Execution Plan  
## *Tech Career Sentiment After ChatGPT — Complete Research Roadmap*

**Workspace:** `AI Implemented/` (strictly self-contained — zero writes outside this folder)  
**Target Venue:** ICWSM 2027 — Round 3 deadline **15 January 2027**  
**Current date:** 21 October 2026 | **Time remaining:** ~86 days  
**Seed:** 42 everywhere | **Break date:** 30 November 2022

> [!IMPORTANT]
> This plan is the AI's authoritative execution roadmap. All code, outputs, figures, tables, and draft text are written inside `AI Implemented/`. Raw data in `../data/raw/` is read-only. Nothing outside `AI Implemented/` is ever modified.

> [!NOTE]
> Infrastructure is already built (Steps 1–4 of `IMPLEMENTATION_PLAN.md` are complete). This plan covers the research workflow itself: running the pipeline, producing results, and delivering the complete paper-ready output.

---

## Locked Decisions (Non-Negotiable)

| Decision | Value |
|---|---|
| Break date | 30 November 2022 (ChatGPT launch) — pre-registered |
| Analysis window | 2020–2025 |
| Subreddits in scope | 6 with existing data (see Phase 2) |
| Unit of analysis | Topic modeling on **submission titles**; sentiment on **comments** |
| Sentiment primary | `cardiffnlp/twitter-roberta-base-sentiment` (3-class) |
| Sentiment cross-check | VADER — `|compound| > 0.05` neutral band |
| Reproducibility seed | 42 in every call |
| Out of scope | DeBERTa fine-tuning, Granger causality, Streamlit dashboard |

---

## Research Questions

| ID | Question | Primary Deliverable |
|---|---|---|
| **RQ1** | Does `cardiffnlp/twitter-roberta-base-sentiment` outperform VADER on tech career Reddit? | Figure 5 — agreement scatter + confusion matrix |
| **RQ2** | How do dominant career topics shift across the four structural regimes? | Figure 2 — topic evolution line chart |
| **RQ3a** | After ChatGPT, how does sentiment diverge across specializations? | Figure 3 — weekly sentiment by subreddit |
| **RQ3b** | After ChatGPT, how does sentiment diverge between seniority levels? | Figure 4 — ITS regression junior vs. control |

---

## Subreddits In Scope (6)

| Subreddit | Role | Used For |
|---|---|---|
| `r/cscareerquestions` | Entry-level / general | RQ2, RQ3a, RQ3b (junior proxy) |
| `r/cscareerquestionsEU` | Geographic control | RQ3b (control cohort) |
| `r/datascience` | Data science specialisation | RQ3a |
| `r/dataengineering` | Data engineering specialisation | RQ3a |
| `r/cybersecurity` | Security specialisation | RQ3a |
| `r/SoftwareEngineering` | General SWE | RQ3a |

> [!NOTE]
> `r/ExperiencedDevs` and `r/csMajors` require data that does not yet exist in `../data/raw/`. They are noted as future additions for the full seniority split (RQ3b) but are excluded from this implementation pass. RQ3b will use `r/cscareerquestions` as the junior proxy and `r/cscareerquestionsEU` as the control.

---

## Execution Phases

```
Phase 2 → Phase 3 → Phase 4 → Phase 5
Data       Measure   Analyse    Write
```

Phase 2 must complete before Phase 3. Phases 3 and 4 overlap (RQ1 is in Phase 3; RQ2/RQ3 are in Phase 4). Phase 5 starts from Phase 4's outputs.

---

## Phase 2 — Data Preparation

**Gate:** Stratified samples exist in `data/output/samples/2.0/` for all 6 subreddits; temporal coverage audit confirms dates span 2020–2025 with no unexplained gaps; privacy-preserving ID column present.

### Step 2.1 — Run Notebook `2.0_sampling.ipynb`

Execute the notebook end-to-end. Verify:

| Check | Expected |
|---|---|
| Parquet files found | 12 files (6 subreddits × 2 types: submissions + comments) |
| Samples written | `data/output/samples/2.0/` — 12 JSON files |
| Rows per file | Up to 30,000 (stratified by ISO week) |
| Date range | min ≥ 2020-01-01, max ≤ 2025-12-31 |
| 1970 dates | Zero — confirms Bug 1 is fixed |
| Coverage plot saved | `data/output/figures/weekly_coverage_audit.png` |

### Step 2.2 — Privacy Pass

After sampling, add a SHA-256 hashed author column to every sample file. This is required for ICWSM ethical release.

**Code to add to `src/ingest/sampling.py` (AI workspace copy):**

```python
import hashlib

SALT = "reddit_career_sentiment_2026"  # fixed salt; store in config or env var

def hash_author(author: str) -> str:
    """One-way SHA-256 hash of Reddit username with fixed salt."""
    if not isinstance(author, str) or author in ("[deleted]", ""):
        return "DELETED"
    raw = f"{SALT}:{author}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

Apply in the post-sampling step: `df["author_hash"] = df["author"].map(hash_author)` then drop the raw `author` column.

### Step 2.3 — Data Quality Filters

Apply after loading, before any modeling:

```python
# Remove deleted / removed posts
df = df[~df[text_col].isin(["[deleted]", "[removed]", ""])]

# Remove very short posts (noise)
df = df[df['cleaned_text'].str.split().str.len() >= 5]

# Remove probable bot accounts (exact duplicates, extremely high volume)
df = df.drop_duplicates(subset=["id"])

# Document counts before/after in experiment log
```

**Record the counts** (before / after filters) in `data/output/experiments/experiment_log.jsonl` via `log_experiment()`.

### Step 2.4 — Coverage Audit Report

Produce a written audit saved to `data/output/coverage_audit_report.md`:

```markdown
# Coverage Audit — 21 October 2026

| Subreddit | Type | N rows | Date min | Date max | Missing weeks | Notes |
|---|---|---|---|---|---|---|
| cscareerquestions | submissions | ... | ... | ... | ... | |
...

## Known Gaps
- June 2023 API blackout (Reddit API policy change): all subreddits show ~4-week gap.
  Handling: exclude this window from weekly share calculations; include as a robustness check.
```

---

## Phase 3 — Measurement (RQ1 Validation)

**Gate:** Agreement rate between VADER and RoBERTa is computed and reported; `sentiment_df.parquet` is written.

### Step 3.1 — Run Notebook `4.0_sentiment_analysis.ipynb`

Execute cells in order. Key outputs to verify:

| Output | Expected |
|---|---|
| VADER labels computed | All rows have `vader_label ∈ {negative, neutral, positive}` |
| RoBERTa scores computed | All rows have `negative`, `neutral`, `positive` probability columns summing to ≈ 1.0 |
| `roberta_label` column | `idxmax` of the three probability columns |
| Agreement rate printed | Expected > 60% (if lower, document and investigate — do **not** discard) |
| Top-20 disagreements | Printed to cell output — inspect manually for domain failure patterns |
| `sentiment_df.parquet` | Saved to `data/output/sentiment_df.parquet` |

### Step 3.2 — RQ1 Evidence Collection

From the notebook output, extract and record in `data/output/tables/rq1_model_comparison.csv`:

```
agreement_rate, vader_neutral_pct, roberta_neutral_pct, n_docs, n_disagreements
```

Write a short evidence note to `data/output/rq1_notes.md` answering:

1. What is the overall agreement rate?
2. Where do the models diverge most (what types of text)?
3. Is the disagreement systematic (e.g., VADER marks sarcastic career advice as positive)?
4. Does the evidence justify choosing RoBERTa as primary? State this explicitly.

---

## Phase 4 — Analysis (RQ2 + RQ3)

**Gate:** All five figures exist in `data/output/figures/` at 300 DPI; ITS coefficients are recorded with confidence intervals; BERTopic topics have been human-reviewed.

### Step 4.1 — Topic Modeling (RQ2) — Notebook `3.0_topic_modeling.ipynb`

#### 4.1.1 Embedding & HDBSCAN Sweep

Run `run_hdbscan_sweep()` over:
- `min_cluster_size ∈ [15, 25, 35, 50]`
- `min_samples ∈ [5, 10, 15]`

**Selection criteria for the best config:**

| Priority | Criterion |
|---|---|
| 1 | Outlier rate (`topic -1`) < 20% |
| 2 | Topic count between 10 and 30 (on 45k titles) |
| 3 | `coherence_cv` maximised among configs meeting 1 and 2 |

If no config meets criterion 1, select the one with the lowest outlier rate and document the deviation.

#### 4.1.2 Topic Sanity-Check (Human Review Gate)

After fitting the final model, print `topic_name_map(model)` and inspect the top-10 words per topic. The AI must verify each topic is interpretable:

```
Expected topic families for tech career data:
  - Job searching / applications / rejections
  - LeetCode / interview preparation
  - Compensation / salary / negotiation
  - Layoffs / PIPs / company news
  - AI tools / skills / fear of replacement
  - Education / degree / bootcamp
  - Remote work / location
  - Data science / ML roles (specialisation-specific)
  - Cybersecurity roles (specialisation-specific)
```

Any topic with fewer than 50 documents or incoherent top-words should be collapsed into topic -1 and documented.

#### 4.1.3 Topic Assignment on Comments

After fitting on submission titles, assign topics to comments by linking each comment to its parent submission's topic (by `parent_id` / `submission_id` join). This enables topic × sentiment cross-tabulation (Figure 5, RQ3).

```python
# Pseudocode for cross-reference join
df_comments = df_comments.merge(
    df_subs[['id', 'topic', 'topic_name']],
    left_on='parent_id',  # or link_id after stripping t3_ prefix
    right_on='id',
    how='left',
)
# Comments without a matched parent topic → topic -1
df_comments['topic'].fillna(-1, inplace=True)
```

#### 4.1.4 RQ2 Analysis

From the topic-assigned submissions DataFrame, compute:

```python
# Weekly topic share per subreddit
weekly_topic_share = (
    df_subs
    .assign(week=lambda d: d['created_utc_dt'].dt.to_period('W').dt.start_time)
    .groupby(['week', 'subreddit', 'topic_name'])
    .size()
    .unstack(fill_value=0)
    .apply(lambda x: x / x.sum(), axis=1)  # normalise to share
)
```

Identify the **top-5 topics by total volume** and track their weekly share across all four structural regimes:

| Regime | Date Range |
|---|---|
| COVID Freeze | 2020-03-01 → 2020-12-31 |
| Tech Hiring Boom | 2021-01-01 → 2022-06-30 |
| Layoff Contraction | 2022-07-01 → 2023-12-31 |
| AI Adoption | 2024-01-01 → 2025-12-31 |

Mark regime boundaries on Figure 2 with vertical shaded bands.

### Step 4.2 — Sentiment Divergence Analysis (RQ3)

#### 4.2.1 Build Weekly Outcome Series

From `sentiment_df.parquet`, build the primary outcome variable:

```python
BREAK = pd.Timestamp('2022-11-30')

# Weekly negative sentiment share per subreddit
weekly = (
    df_comments
    .assign(week=lambda d: d['created_utc_dt'].dt.to_period('W').dt.start_time)
    .groupby(['week', 'subreddit', 'roberta_label'])
    .size()
    .unstack(fill_value=0)
)
weekly['neg_share'] = weekly['negative'] / weekly.sum(axis=1)
```

Exclude the June 2023 API blackout window (approximately 2023-06-12 to 2023-07-10) from all weekly series. Mark it with a grey band on all temporal figures.

#### 4.2.2 RQ3a — Specialisation Split

Plot 4-panel figure (one panel per specialisation subreddit: `datascience`, `dataengineering`, `cybersecurity`, `SoftwareEngineering`). Each panel shows:
- Weekly `neg_share` (4-week rolling average)
- Vertical dashed line at 30 Nov 2022
- Grey band for June 2023 blackout

Export as `data/output/figures/fig3_specialisation_sentiment.png` (300 dpi).

For each subreddit, compute:
- Pre-break mean (2020-01-01 → 2022-11-29) negative share
- Post-break mean (2022-12-01 → 2025-12-31) negative share
- Delta (percentage points)
- 95% confidence interval on the delta via bootstrapping (1,000 iterations, seed 42)

Save to `data/output/tables/rq3a_specialisation_delta.csv`.

#### 4.2.3 RQ3b — Interrupted Time Series (ITS) Regression

**Model specification:**

```
neg_share_t = β₀ + β₁·t + β₂·D_t + β₃·(t − T_break)·D_t + ε_t
```

Where:
- `t` = week index (0 = first week in data)
- `D_t` = 1 if week ≥ break date, else 0
- `T_break` = week index of 30 Nov 2022

Fit separately for each subreddit using `statsmodels.formula.api.ols`. Report:

| Coefficient | Interpretation |
|---|---|
| `β₁` | Pre-break linear trend |
| `β₂` | Level shift at break (immediate impact) |
| `β₃` | Post-break slope change (sustained trend shift) |

```python
import statsmodels.formula.api as smf

def fit_its(df_weekly: pd.DataFrame) -> dict:
    """Fit interrupted time series model. df_weekly must have columns: t, neg_share, D."""
    df_weekly = df_weekly.copy()
    df_weekly['t_post'] = df_weekly['t'] * df_weekly['D']
    model = smf.ols('neg_share ~ t + D + t_post', data=df_weekly).fit()
    return {
        'beta1': model.params['t'],
        'beta2': model.params['D'],
        'beta3': model.params['t_post'],
        'beta2_pval': model.pvalues['D'],
        'beta3_pval': model.pvalues['t_post'],
        'beta2_ci': model.conf_int().loc['D'].tolist(),
        'beta3_ci': model.conf_int().loc['t_post'].tolist(),
        'r2': model.rsquared,
        'n_obs': int(model.nobs),
    }
```

Run for all 6 subreddits. Apply Holm–Bonferroni correction to the 12 p-values (2 coefficients × 6 subreddits). Save full results to `data/output/tables/rq3b_its_results.csv`.

**Figure 4** shows:
- Two panels: `r/cscareerquestions` (junior proxy) and `r/cscareerquestionsEU` (geographic control)
- Raw weekly neg_share (light) + rolling average (dark)
- Fitted ITS line pre and post break (from model)
- Break date marker
- β₂ and β₃ coefficients annotated on the plot

#### 4.2.4 Robustness Checks

After primary analysis, run and record:

| Check | Method | Expected |
|---|---|---|
| Placebo break dates | Repeat ITS at 2021-06-01 and 2020-11-30 | β₂ not significant — confirms the Nov 2022 effect is real |
| Exclude June 2023 blackout | Re-run ITS dropping those 4 weeks | Results should be similar |
| Submissions only vs. comments only | Repeat neg_share calculation on submissions (using selftext sentiment) | Note direction and magnitude |
| VADER primary | Swap VADER compound into the outcome series | Compare ITS coefficients with RoBERTa — cross-check RQ1 |

Document each check's outcome in `data/output/tables/robustness_checks.csv` and note any meaningful deviations.

---

## Phase 5 — Writing & Release

**Gate:** A stranger who reads the draft can follow the argument; every figure is generated by a notebook cell; every reference is verified against its source page.

### Step 5.1 — Run Notebook `5.0_results_and_viz.ipynb`

Execute all cells and verify all 5 figures are generated:

| Figure | File | Research Question |
|---|---|---|
| Figure 1 | `fig1_sentiment_by_subreddit.png` | Overview |
| Figure 2 | `fig2_regime_sentiment_evolution.png` | RQ2 |
| Figure 3 | `fig3_specialisation_sentiment.png` | RQ3a |
| Figure 4 | `fig4_its_regression.png` | RQ3b |
| Figure 5 | `fig5_vader_roberta_agreement.png` | RQ1 |

All must be 300 DPI PNG. All temporal figures must show the 30 Nov 2022 break line and the June 2023 blackout band.

### Step 5.2 — Paper Draft

Write the draft to `data/output/paper_draft.md`. Follow the **results-first, introduction-last** order:

#### Recommended Writing Sequence

```
1. Data & Measures section  ← write from the coverage audit and filter counts
2. Results section          ← one subsection per RQ, one paragraph per figure
3. Related Work section     ← position against Cao et al., Iordache et al., Brynjolfsson et al.
4. Discussion section       ← interpret, acknowledge limitations
5. Introduction             ← write last, once the findings are clear
6. Abstract                 ← write very last, from the Introduction
```

#### Paper Skeleton

```markdown
# Hiring Fatigue or Layoff Fear?
## Career Discourse Divergence Across Specializations After ChatGPT, 2020–2025

**Abstract** (150–250 words — write last)

---

## 1. Introduction
- Open with Brynjolfsson et al. (2026): 19% employment gap in AI-exposed junior roles.
- Pivot: we know from payroll data *that* the gap exists; we don't know *what junior communities are saying* differently from senior/control communities.
- State the contribution: RQ1 (model validation) + RQ2 (topic shifts) + RQ3 (sentiment divergence).
- Brief preview of findings (1 paragraph) — write after results are complete.

---

## 2. Related Work (750–1000 words)
- Social media as labor market signal: Aizenman et al. (2023), Antenucci et al. (2014).
- Reddit as data source: Baumgartner et al. (2020), REALM / Cheng et al. (2025).
- Closest competitor: Cao et al. (Nov 2025) — they measure volume; we measure content.
- Qualitative baseline: Iordache et al. (Sep 2026) — emotion themes; not longitudinal.
- Sentiment methodology: Zhang et al. (ACM TOSEM 2025), Barbieri et al. TweetEval (2020).
- Key differentiator paragraph: no prior study has simultaneously stratified by specialization AND compared seniority cohorts with ITS.

---

## 3. Data & Measures (600–800 words)
- Subreddits: table with 6 communities, roles, date ranges, N_submissions, N_comments.
- Data source: Academic Torrents (pre-2023) + Arctic Shift (post-2022). One source for the full window.
- Sampling: stratified by ISO week, 30k per subreddit/type file.
- Preprocessing: filters applied (deleted/removed, very short, duplicates). Report before/after counts.
- Privacy: SHA-256 hashed author IDs; no raw text released.
- Coverage audit: note June 2023 blackout; handled by exclusion in primary analysis.
- Sentiment: `cardiffnlp/twitter-roberta-base-sentiment` (3-class); justified by RQ1.
- Topic modeling: BERTopic on submission titles; `all-MiniLM-L6-v2` embeddings; HDBSCAN parameters (from sweep).

---

## 4. Results (1500–2000 words)

### 4.1 RQ1 — Model Validation
- Agreement rate: XX%.
- VADER systematic failures on career text (top-disagreement examples).
- Conclusion: RoBERTa adopted as primary; VADER retained as robustness check.
- Reference Figure 5.

### 4.2 RQ2 — Topic Evolution
- N_topics topics found (outlier rate: XX%).
- Top-5 topics by volume: list with example posts.
- Regime narrative: [COVID Freeze] → [Boom] → [Layoff Wave] → [AI Adoption].
- Key finding: which topics rose/fell at the 30 Nov 2022 break?
- Reference Figure 2.

### 4.3 RQ3a — Specialization Divergence
- Pre/post delta table for all 4 specialisation subreddits.
- Which communities showed significant increases in negative sentiment?
- Are data science / data engineering communities distinct from cybersecurity / SWE?
- Reference Figure 3.

### 4.4 RQ3b — Seniority Divergence (ITS)
- β₂ (level shift) and β₃ (slope change) for cscareerquestions vs. cscareerquestionsEU.
- Report with 95% CIs and Holm-corrected p-values.
- Effect size interpretation (e.g., "X percentage-point increase in negative share at the break").
- Reference Figure 4.

---

## 5. Discussion (600–800 words)
- Interpret findings in light of Brynjolfsson et al. labor market data.
- If null: a clean null in career discourse despite documented labor market shifts is itself informative.
- Limitations (mandatory — state honestly):
  - Community-level evidence, not individual-level.
  - Self-selection into subreddits (users who post ≠ all tech workers).
  - Label error from RoBERTa (trained on Twitter, not Reddit).
  - No causal claim about AI causing sentiment changes.
  - r/ExperiencedDevs and r/csMajors missing — RQ3b seniority split is incomplete.
  - June 2023 blackout gap.

---

## 6. Ethics Statement
- Data: Reddit public posts, collected per Reddit API Terms of Use and Academic Torrents terms.
- Privacy: author usernames replaced with one-way SHA-256 hashes before storage; no raw text released.
- Release: post IDs, weekly aggregates, codebook, prompts, and model weights will be released.
- No attempts to re-identify individuals were made.

---

## 7. Conclusion (200–300 words)
- One-sentence answer to each RQ.
- The novel finding.
- Future work: adding r/ExperiencedDevs and r/csMajors, fine-tuned domain model, causal analysis.

---

## References
(see Step 5.3 for verification requirements)
```

### Step 5.3 — Reference Verification

Every reference must be checked against its source page before submission. Verified references go into `data/output/references_verified.md`.

**Known citation errors to fix (from `plans/archive/literature_sweep_and_tightened_proposal.md`):**

| Wrong | Correct |
|---|---|
| arXiv:2311.15800 cited as "ChatGPT in education" | Actually: ChatGPT in **mental health** — Cai et al. — do NOT cite as education paper |
| Zhang et al. 2026 ResearchGate preprint | Correct citation: ACM TOSEM 2025, arXiv:2310.11113 |
| BERTopic citation | Cite to Grootendorst (2022) arXiv:2203.05794 directly |

**References to verify:**

| Reference | Verification URL | Status |
|---|---|---|
| Cao et al. (Nov 2025) | https://arxiv.org/abs/2511.16028 | ☐ |
| Brynjolfsson et al. (Aug 2026) | https://digitaleconomy.stanford.edu/app/uploads/2026/08/Canaries_August2026.pdf | ☐ |
| Iordache et al. (Sep 2026) | https://doi.org/10.3390/socsci15100663 | ☐ |
| Baumgartner et al. (2020) | ICWSM 2020 | ☐ |
| Barbieri et al. (2020) | https://arxiv.org/abs/2010.12421 | ☐ |
| Grootendorst (2022) | https://arxiv.org/abs/2203.05794 | ☐ |
| Zhang et al. (2025) | https://arxiv.org/abs/2310.11113 | ☐ |

### Step 5.4 — Release Package

Prepare `data/output/release/` with:

```
release/
├── README.md               ← data card: provenance, filters, label definitions
├── post_ids.csv            ← submission/comment IDs only (no text)
├── weekly_aggregates.csv   ← weekly neg/neu/pos share per subreddit
├── its_coefficients.csv    ← ITS β values, CIs, corrected p-values
├── topic_summary.csv       ← topic ID, top-10 words, weekly share by regime
├── codebook.md             ← 4-class label definitions + examples
└── environment.txt         ← pip freeze output from the run environment
```

---

## Figure Specifications

All figures must meet these standards for ICWSM submission:

| Requirement | Standard |
|---|---|
| Resolution | 300 DPI minimum |
| Format | PNG (primary), PDF available on request |
| Font | Sans-serif, ≥ 9pt labels |
| Colourblind-safe | Use seaborn `colorblind` palette or explicit hex codes |
| Break date marker | Dashed black vertical line labeled "ChatGPT launch (Nov 2022)" |
| Blackout marker | Grey shaded band labeled "Reddit API blackout (Jun–Jul 2023)" |
| Confidence bands | 95% CI shading on all rolling averages |
| Legend | Outside plot area to avoid overlap |

---

## Experiment Tracking Requirements

Every modeling run must produce a line in `data/output/experiments/experiment_log.jsonl` via `log_experiment()` containing at minimum:

```json
{
  "timestamp": "...",
  "params": {
    "run_type": "hdbscan_sweep|its_regression|sentiment_scoring",
    "subreddit": "...",
    "dataset_type": "submissions|comments",
    "n_docs": 12345,
    "seed": 42
  },
  "metrics": {
    "n_topics": 18,
    "outlier_pct": 0.14,
    "coherence_cv": 0.512,
    "agreement_rate": 0.71,
    "beta2": 0.034,
    "beta2_pval_corrected": 0.021
  },
  "versions": {
    "python": "...",
    "bertopic": "...",
    "torch": "..."
  }
}
```

---

## Timeline

| Date | Phase | Deliverable | Gate |
|---|---|---|---|
| **24 Oct 2026** | 2 | Stratified samples written; coverage audit saved | 12 JSON files in `data/output/samples/2.0/`; no 1970 dates |
| **7 Nov 2026** | 2+3 | Privacy pass done; sentiment scored; `sentiment_df.parquet` saved | Agreement rate computed and recorded |
| **14 Nov 2026** | 4 | BERTopic fitted; topics human-reviewed; ITS coefficients computed | `rq3b_its_results.csv` exists; figures 2–4 rendered |
| **21 Nov 2026** | 4 | Robustness checks complete; all 5 figures final | All figures in `data/output/figures/` at 300 DPI |
| **5 Dec 2026** | 5 | Data, Methods, and Results sections drafted | `paper_draft.md` contains these three sections |
| **19 Dec 2026** | 5 | Full draft including Related Work, Discussion, Ethics, Conclusion | `paper_draft.md` complete |
| **26 Dec 2026** | 5 | All references verified; release package prepared | `references_verified.md` complete; `release/` folder populated |
| **9 Jan 2027** | 5 | Final draft; figures proof-read | Zero unverified citations; zero unexplained deviations from pre-registration |
| **15 Jan 2027** | — | **ICWSM 2027 submission** | — |

---

## Reporting Standards

### What Must Be Reported Regardless of Results

- **All pre-registered tests** — report whether significant or not. Do not selectively omit null results.
- **Effect sizes with 95% confidence intervals** — not just p-values.
- **Holm–Bonferroni corrected p-values** for the 12 ITS hypothesis tests.
- **Outlier topic rate** — report the fraction of submissions in topic -1.
- **Data filters** — exact N before and after each filter.
- **Deviations from this plan** — any deviation must be noted in the paper with a justification.

### What Must NOT Be Done

- ❌ Tune HDBSCAN parameters after seeing the ITS results.
- ❌ Choose between VADER and RoBERTa based on which gives a stronger effect size.
- ❌ Drop subreddits after seeing their results.
- ❌ Move the break date from 30 November 2022.
- ❌ Add or remove hypotheses post-hoc without flagging them as exploratory.

> [!CAUTION]
> Mixed or null results are fully publishable at ICWSM if the design is clean and the limitations are honest. Do not manipulate any parameter to produce a positive result.

---

## Output Directory Map

```
AI Implemented/
└── data/
    └── output/
        ├── samples/2.0/              ← 12 stratified JSON sample files
        ├── cache/                    ← embeddings (.npy) + fitted topic model
        ├── experiments/
        │   └── experiment_log.jsonl  ← one line per modeling run
        ├── figures/
        │   ├── weekly_coverage_audit.png
        │   ├── fig1_sentiment_by_subreddit.png
        │   ├── fig2_regime_sentiment_evolution.png
        │   ├── fig3_specialisation_sentiment.png
        │   ├── fig4_its_regression.png
        │   └── fig5_vader_roberta_agreement.png
        ├── tables/
        │   ├── rq1_model_comparison.csv
        │   ├── rq3a_specialisation_delta.csv
        │   ├── rq3b_its_results.csv
        │   └── robustness_checks.csv
        ├── release/
        │   ├── README.md
        │   ├── post_ids.csv
        │   ├── weekly_aggregates.csv
        │   ├── its_coefficients.csv
        │   ├── topic_summary.csv
        │   ├── codebook.md
        │   └── environment.txt
        ├── sentiment_df.parquet      ← scored comments dataset
        ├── coverage_audit.csv        ← weekly volume per subreddit
        ├── coverage_audit_report.md  ← written gap documentation
        ├── rq1_notes.md              ← RQ1 evidence interpretation
        ├── references_verified.md    ← reference checklist
        └── paper_draft.md            ← full paper manuscript
```
