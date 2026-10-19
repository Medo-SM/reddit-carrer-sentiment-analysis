# Research Roadmap: Tech Career Discourse After ChatGPT

Oct 18, 2026 · @Mohammed

## How to use this roadmap

The research runs in five phases, and each phase ends with a gate you must pass before starting the next. No dates are attached; move on when the gate is met, not when you feel ready.

1. Foundations: skills, the one research question, the pre-registration, the setup.
2. Data: a clean weekly corpus with a coverage audit.
3. Measurement: trustworthy labels, proven against human annotation.
4. Analysis: estimates, placebo tests and robustness checks.
5. Writing and release: the paper, the ethics statement, the artifacts.

**Working rules**

- Build everything on one subreddit and one year first, then scale up.
- Keep a lab notebook (a dated markdown file in your repository) of every decision and every dead end. You will need it for the methods section.
- Never tune anything on the data you will report results from. Split off a development set early and a held-out test set you touch once.
- Write down surprising results and explain them before moving on. A result that looks too good is usually a bug or a confound.
- Keep the code in Git from day one, with a README that lets a stranger rerun it.

## Phase 1: Foundations

Goal: a frozen question, a pre-registration, and a working environment.

**Learn**

- Python and pandas: reading files, filtering, grouping by week, plotting. *Python for Data Analysis* by Wes McKinney is the standard text.
- Git and a virtual environment, so your work is reproducible.
- Basic statistics: distributions, confidence intervals, p-values and what they do not mean.
- Time-series ideas: trend, seasonality, autocorrelation. *Forecasting: Principles and Practice* by Hyndman and Athanasopoulos is free online.
- Causal-design ideas: interrupted time series, difference-in-differences, parallel trends. *Causal Inference: The Mixtape* by Scott Cunningham is free online.

**Do**

1. Read three or four recent ICWSM papers on Reddit to see what a finished study looks like, then read the Cao et al. and Canaries papers closely.
2. Set up the repository, a Python environment (pandas, DuckDB, zstandard, statsmodels, scikit-learn, matplotlib) and a notebook.
3. Write the one research question and the two hypotheses from the tightened proposal in your own words.
4. Choose the communities: junior, senior and controls. Pick the controls from published AI-exposure rankings before you look at any outcome data.
5. Fix the break date (30 November 2022), the start of the pre-period, the outcome definitions, the model, and the tests you will run. Write all of it into a pre-registration on OSF.
6. Do a second literature check with specific queries on Google Scholar and Semantic Scholar, such as Reddit career discourse junior senior AI, and add anything new to your notes.

**Gate:** you can explain the design to a stranger in five minutes, and the pre-registration is posted.

## Phase 2: Data

Goal: one clean table of posts per community and week, with a documented coverage audit.

**Learn**

- How the Reddit dumps are stored: compressed newline-delimited JSON, one file per month for submissions and comments. Read them as a stream with the zstandard library; you will need to raise its maximum window size or the files will not open.
- DuckDB and Parquet for fast queries over large files.
- Reddit's terms and research-ethics norms for scraped public posts.

**Do**

1. Choose one source for the whole window: the monthly Academic Torrents dumps or Arctic Shift. Check its coverage dates and the terms of use before downloading anything.
2. Write a script that streams one month for one subreddit, keeps only the fields you need (id, created time, subreddit, author, title, text, score), and writes Parquet. Test it on a single month first.
3. Scale to all communities and months. Store a log of every file processed.
4. Clean: drop removed and deleted stubs, known bots and moderator posts, exact duplicates, and very short items. Hash author names with a salted hash and keep the salt private.
5. Decide the unit of analysis (posts only, or posts plus comments) and write it into the pre-registration before comparing outcomes.
6. Build the weekly table: community, week, number of items, number of unique authors.
7. Run the coverage audit. Plot raw weekly volume for every community and look for gaps, sudden drops, the June 2023 API blackout, and growth that is not real. Write down every anomaly and how you handle it.

**Gate:** the weekly volume plot has no unexplained holes, and the audit note is written.

## Phase 3: Measurement

Goal: labels you can defend, with agreement and accuracy numbers to prove it.

**Learn**

- Classification metrics: precision, recall, F1, macro-F1, confusion matrices. The scikit-learn documentation explains each.
- Inter-annotator agreement: Cohen's kappa and what counts as acceptable.
- Prompting an LLM for classification: clear label definitions, examples, structured JSON output, fixed settings, and saving every raw response.

**Do**

1. Write the codebook: a definition, two or three clear examples, and a borderline example for each of the four labels (AI-displacement concern, hiring or interview fatigue, layoff or separation distress, other). Include rules for sarcasm and for posts about other people.
2. Draw a stratified sample of about 600 items across communities and periods. Split it into a development set and a held-out test set.
3. Annotate the development set yourself and with a second person, then compare. Discuss every disagreement, revise the codebook, and repeat on the test set only once the codebook is stable. Report kappa on the test set.
4. Write the LLM labeling prompt and tune it on the development set only. Version every prompt and keep the raw outputs.
5. Evaluate the final prompt once on the held-out test set against the adjudicated human labels. Report per-class precision, recall and F1, and look at where it fails, especially on sarcasm and jargon.
6. Check whether errors differ by community or period, because a labeler that is worse on one community would bias the comparison.
7. Label the full corpus, with a record of cost and run settings.

**Gate:** kappa and per-class F1 are reported on a test set you touched once, and you can state which errors remain.

## Phase 4: Analysis

Goal: estimates of how junior and senior communities changed after the break, with honest uncertainty and checks that could have falsified the result.

**Learn**

- Segmented (interrupted time series) regression and difference-in-differences, including event-study plots with leads and lags.
- Why serial correlation makes ordinary standard errors too small, and how Newey-West (HAC) standard errors in statsmodels address it.
- Seasonality: the academic calendar moves student communities every year, so you need month or week-of-year controls.
- Multiple testing: why you correct the two primary tests (Holm) and label everything else exploratory.

**Do**

1. Build the outcome series: for each community and week, the share of items in each label. Choose the denominator (all items, or items labeled non-other) in the pre-registration.
2. Plot every series raw before modeling. Mark 30 November 2022 and the major layoff waves.
3. Fit the primary models exactly as pre-registered: a segmented regression per community, then difference-in-differences comparing junior and senior communities against controls.
4. Draw the event-study plot. If the leads before the break are not flat, the parallel-trends assumption fails; report that rather than hiding it.
5. Add Layoffs.fyi weekly counts and the Indeed Hiring Lab postings index as covariates to test whether the junior hiring-fatigue pattern survives them.
6. Run robustness checks: placebo break dates, a different pre-period start, excluding the June 2023 blackout weeks, posts only versus posts plus comments, alternative control sets, and the self-reported-experience split inside r/cscareerquestions.
7. Check labeler sensitivity: rerun with a different prompt or model and see whether the conclusion holds.
8. Report effect sizes with confidence intervals, not only p-values. State plainly what the design cannot show: it measures what people write, not what happens to their jobs, and it cannot prove AI caused the change.

**Gate:** every pre-registered test is run and reported, deviations are listed, and each figure answers one question.

## Phase 5: Writing and release

Goal: a paper a reviewer can verify, plus artifacts others can reuse.

**Learn**

- The structure of a computational social science paper: question, related work, data, measures, methods, results, limitations, ethics.
- Research ethics for public social media data: minimizing quoted text, protecting authors, and respecting platform terms.
- How to write a methods section so a stranger could repeat it.

**Do**

1. Outline the paper around the one question. Each figure should answer one sentence of the argument: coverage audit, raw weekly series, segmented fits, event-study plot, robustness summary.
2. Write the data and measures sections first, from your lab notebook, then results, then related work, and the introduction last.
3. In related work, position against Cao et al. (volume as a leading indicator), the qualitative Reddit studies, and the Canaries employment findings, and say plainly what you add.
4. Write the limitations honestly: community-level not individual-level evidence, self-selection into subreddits, label error, and no causal claim about AI.
5. Write the ethics statement: public data, hashed authors, no raw text released, minimal quoting, and the terms of use you followed.
6. Verify every reference against its source page, one by one. A fabricated or garbled citation can get a paper rejected outright.
7. Release post IDs, labels, weekly aggregates, the codebook, prompts and code in a clean repository with a README and a fixed environment file.
8. Ask someone outside the project to read the draft and try to rerun the code. Fix what confuses them.
9. Choose the venue and check its current call for papers, page limit, anonymity rules and preprint policy before submitting. Do not submit the same paper to two venues at once.
10. Post a preprint when the venue's rules allow it.

**Gate:** a reader who was not involved can follow the argument, and the code reproduces your figures from the released data.
