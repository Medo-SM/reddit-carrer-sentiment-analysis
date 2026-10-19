# Research Plan: Tech Career Sentiment After ChatGPT

Oct 19, 2026 · @Mohammed  
**Status: Active — supersedes `archive/research_proposal_original.md`, `archive/literature_sweep_and_tightened_proposal.md`, `archive/research_roadmap_after_chatgpt.md`**

---

## Locked Decisions

These are finalized and must not be changed without a note explaining why.

| Decision | Resolved value | Source |
|---|---|---|
| Break date | 30 November 2022 (ChatGPT launch) | Pre-registration |
| Data source | One source for the full window: Academic Torrents dumps + Arctic Shift for post-2022 | Avoids coverage inconsistency |
| Subreddits | 6 existing + 2 to add (see below) | Resolved Oct 19 |
| Analysis window | 2020–2025 | Consistent with data availability |
| Unit of analysis | Topic modeling on submissions (titles), sentiment on comments | `modeling-rework.md` §1 locked decisions |
| Sentiment primary model | `cardiffnlp/twitter-roberta-base-sentiment` | `modeling-rework.md` §2 Bug 6 |
| Sentiment cross-check | VADER (kept for validation, not primary) | `modeling-rework.md` §2 Bug 6 |
| No DeBERTa fine-tuning | Out of scope for this pass | Resolved Oct 19 |
| No Granger causality | Out of scope for this pass | Resolved Oct 19 |
| No Streamlit dashboard | Out of scope for this pass | Resolved Oct 19 |
| Reproducibility seed | 42 everywhere | `modeling-rework.md` §5 |

---

## Subreddits

### Currently have data for (6):
| Subreddit | Role in analysis |
|---|---|
| `r/cscareerquestions` | Entry-level / general job-seeker |
| `r/cscareerquestionsEU` | Entry-level, European market (geographic control) |
| `r/datascience` | Data specialization cohort |
| `r/dataengineering` | Data engineering specialization cohort |
| `r/cybersecurity` | Security specialization cohort |
| `r/SoftwareEngineering` | General SWE specialization cohort |

### To add data for (2):
| Subreddit | Role in analysis |
|---|---|
| `r/ExperiencedDevs` | Senior / experienced cohort |
| `r/csMajors` | Pre-career / student cohort (earlier-stage than cscareerquestions) |

> [!IMPORTANT]
> Add `r/ExperiencedDevs` and `r/csMajors` before starting Phase 3 (Measurement). Both are needed for the seniority split in RQ3.

---

## Research Questions

Narrowed from the original 4 to 3. Granger causality (original RQ4) and DeBERTa fine-tuning (original RQ1 extension) are dropped for this paper.

| ID | Question | Method | Deliverable |
|---|---|---|---|
| **RQ1** | Does `cardiffnlp/twitter-roberta-base-sentiment` outperform VADER on tech career Reddit discourse? | Side-by-side comparison on 600-post annotated sample; agreement rate + top-disagreement analysis | Validation figure + model justification in methods section |
| **RQ2** | How do dominant career topics shift across the four structural regimes (COVID freeze → 2021 boom → 2022–23 layoffs → 2024–25 AI adoption)? | BERTopic on submission titles; temporal evolution plots per subreddit | Topic evolution figure (Figure 2 in paper) |
| **RQ3** | After ChatGPT's release, how does career sentiment diverge (a) across specializations and (b) between seniority levels? | Sentiment trend by subreddit (specialization), plus junior (cscareerquestions + csMajors) vs. senior (ExperiencedDevs) comparison; ITS on weekly shares | Sentiment divergence figures (Figures 3–4 in paper) |

> [!NOTE]
> **Mixed or null results are publishable** if the design is clean. Do not tune anything to get a positive result.

---

## Novelty Statement

The open gap from the literature sweep (Oct 17, 2026):

> *"Junior vs. senior tech-community discourse after ChatGPT, against control communities, separating hiring talk from layoff talk — no match found in 11 searches."*

This paper extends that gap by **also stratifying across technical specializations** (data science vs. SWE vs. cybersecurity), which no prior study has done for this post-ChatGPT period.

**Differentiator from Cao et al. (2025):** They measure *volume* of LLM discussion as a leading labor indicator. This paper measures *sentiment content and topic composition* within career communities, with seniority and specialization as independent axes.

---

## Label Taxonomy (4-class, for RQ3 annotation)

| Label | Definition |
|---|---|
| AI-displacement concern | Fear of automated replacement, skill obsolescence, role elimination due to AI |
| Hiring / interview fatigue | Ghosting, multi-round OA/LeetCode inflation, application volume distress |
| Layoff / separation distress | Severance, PIPs, hiring freezes, company downsizing |
| Other | Everything else — career advice, compensation, remote work, optimism, etc. |

> [!NOTE]
> The original proposal used a 6-class taxonomy. The 4-class version is simpler, more defensible in annotation, and sufficient for the one core question. "Skill pivoting" and "career optimism" collapse into "other" — they can be extracted descriptively from BERTopic topics if interesting.

---

## Five Research Phases

### Phase 1 — Foundations ✅ (mostly done)
**Gate:** One question frozen, pre-registration drafted, environment running.

- [x] Repository set up with Git
- [x] Data ingestion pipeline (ZST → Parquet, Arctic Shift scraper)
- [ ] Write pre-registration on OSF (question, hypotheses, analysis plan, control selection)
- [ ] Fix citation errors from original proposal before anyone reads it

---

### Phase 2 — Data
**Gate:** Weekly volume table exists for all 8 subreddits, coverage audit written, no unexplained gaps.

- [ ] Add `r/ExperiencedDevs` and `r/csMajors` to scraper targets
- [ ] Generate stratified samples → `data/samples/2.0/` (~30k rows per subreddit/type)
- [ ] Run coverage audit (weekly volume plot per subreddit; mark June 2023 API blackout)
- [ ] Document every gap and how it's handled
- [ ] Hash author usernames (SHA-256 + salt) for privacy
- [ ] Drop bots, `[deleted]`/`[removed]` stubs, duplicates, very short posts

---

### Phase 3 — Measurement
**Gate:** Kappa and per-class F1 reported on a test set touched once; LLM labeling prompt frozen.

- [ ] Write annotation codebook for 4 labels (definition + 2 examples + 1 borderline each)
- [ ] Draw stratified sample of ~600 posts (across communities + time periods)
- [ ] Self-annotate development set; get second annotator; compute kappa; revise codebook
- [ ] Freeze LLM labeling prompt on development set only
- [ ] Evaluate prompt once on held-out test set → report per-class precision/recall/F1
- [ ] Label full corpus with frozen prompt
- [ ] **RQ1 validation:** run VADER and RoBERTa on 600-post sample; compute agreement rate; generate comparison figures

---

### Phase 4 — Analysis
**Gate:** All pre-registered tests run and reported; deviations listed; each figure answers one question.

- [ ] Build outcome series: weekly share per community per label
- [ ] Plot raw series; mark break date (30 Nov 2022) and layoff waves
- [ ] **RQ2:** Fit BERTopic on submission titles; generate topic evolution plots
- [ ] **RQ3a:** Sentiment trend by subreddit (specialization split)
- [ ] **RQ3b:** ITS regression on weekly sentiment share, junior vs. senior vs. control
- [ ] Add Layoffs.fyi weekly counts as covariate
- [ ] Robustness checks: placebo break dates, exclude June 2023 blackout, posts-only vs. posts+comments
- [ ] Report effect sizes with confidence intervals

---

### Phase 5 — Writing & Release
**Gate:** Stranger can follow the argument; code reproduces figures from released data.

- [ ] Outline paper around one question per figure
- [ ] Write data + measures sections from lab notebook first
- [ ] Write results, then related work, then introduction last
- [ ] Position explicitly against Cao et al., Iordache et al., and Brynjolfsson et al.
- [ ] Write limitations honestly (community-level not individual, self-selection, label error, no causal claim about AI)
- [ ] Write ethics statement (public data, hashed authors, no raw text released, ToU compliance)
- [ ] Verify every reference against its source page one-by-one
- [ ] Release: post IDs, labels, weekly aggregates, codebook, prompts, code, fixed environment file
- [ ] Target venue: **ICWSM 2027**, round 3 deadline **15 January 2027**

---

## Timeline

| Checkpoint | Goal |
|---|---|
| 24 October 2026 | Pre-registration posted; control subreddits fixed |
| 7 November 2026 | All 8 subreddits loaded; weekly volume + coverage audit done |
| 21 November 2026 | 600-post annotation done; kappa computed; prompt frozen |
| 12 December 2026 | Full corpus labeled; RQ1 + RQ2 + RQ3 analysis complete |
| 26 December 2026 | Figures done; methods and results drafted |
| 9 January 2027 | Full draft; outside reader; ethics statement; references verified |
| 15 January 2027 | **ICWSM 2027 submission** |

---

## Key References

- Cao et al. (Nov 2025) — *Can Online GenAI Discussion Serve as Bellwether for Labor Market Shifts?* arXiv:2511.16028 — **closest competitor**
- Brynjolfsson, Chandar, Chen (Aug 2026) — *Canaries in the Coal Mine?* Stanford Digital Economy Lab — **documents junior employment gap**
- Iordache et al. (Sep 2026) — *Fear, Blame, and Deservingness* — Social Sciences 15(10) — **qualitative overlap**
- Grootendorst (2022) — *BERTopic* arXiv:2203.05794 — **topic model citation**
- Barbieri et al. (2020) — *TweetEval* arXiv:2010.12421 — **RoBERTa model citation**
- Zhang et al. (2025) — *Revisiting Sentiment Analysis for SE in the Era of LLMs* ACM TOSEM — **baseline comparison**
- Baumgartner et al. (2020) — *The Pushshift Reddit Dataset* ICWSM — **data source citation**

> [!CAUTION]
> Fix before submission: the original proposal cites arXiv:2311.15800 as a ChatGPT-in-education paper — it actually studies ChatGPT in mental health. Check every reference against its source page before anyone external reads the paper.
