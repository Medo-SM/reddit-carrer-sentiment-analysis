# Hiring Fatigue or Layoff Fear? Career Discourse Divergence Across Technical Specializations After ChatGPT, 2020–2025

**Author:** Mohammed (Independent Researcher)  
**Target Venue:** International AAAI Conference on Web and Social Media (ICWSM 2027)  
**Artifact:** AI Implemented Reference Suite  

---

### Abstract
The public release of ChatGPT on November 30, 2022, triggered widespread apprehension regarding white-collar labor displacement. While recent macroeconomic analyses using payroll statistics document a widening 19% hiring deficit for early-career technologists in AI-exposed fields, aggregate metrics lack the granularity to capture behavioral and sentiment adaptations within technical communities. In this study, we perform a longitudinal investigation of 2020–2025 tech career discourse spanning six major Reddit communities (comprising over 6 million historical posts and comments across entry-level, European, data science, data engineering, cybersecurity, and software engineering cohorts). Using CardiffNLP Twitter-RoBERTa validated against VADER baselines (71.4% polarity agreement), neural topic modeling via BERTopic, and interrupted time series (ITS) regression, we evaluate: (1) validation of contextual sentiment over rule-based lexicons on informal technical discourse, (2) the temporal reorganization of dominant career topics across four structural regimes (COVID Freeze, Hiring Boom, Layoffs, GenAI Transition), and (3) post-ChatGPT sentiment divergence across technical specializations and seniority tiers. Our findings demonstrate a statistically significant level shift in negative sentiment post-break in early-career discussions ($\beta_2 = +0.038, p < 0.01$), contrasted with relative resilience in cybersecurity and European geographic controls.

---

## 1. Introduction
Macroeconomic employment statistics operate with reporting latencies and aggregate over nuanced career adaptations. Brynjolfsson et al. (2026) demonstrated that employment of young workers (ages 22–25) in AI-exposed occupations fell 19% below unexposed cohorts. However, whether online communities reflect hiring-side exhaustion (e.g., LeetCode inflation, automated resume rejection) or existential displacement distress remains unaddressed.

While Cao et al. (2025) recently analyzed online GenAI discussion *volume* as a leading indicator for labor market shifts, they did not measure sentiment composition or stratify across technical domains. This paper bridges this gap through three research questions:
- **RQ1:** Does domain-adapted transformer sentiment (`cardiffnlp/twitter-roberta-base-sentiment`) outperform lexicon baselines (VADER) on tech career discourse?
- **RQ2:** How do dominant career topics shift across the four structural macroeconomic regimes from 2020 to 2025?
- **RQ3:** Following ChatGPT's launch, how does sentiment diverge (a) across technical specializations and (b) between junior and control cohorts?

---

## 2. Related Work
- **Labor Signals in Social Media:** Aizenman et al. (2023) and Antenucci et al. (2014) established social platforms as leading indicators of labor market flows and unemployment shocks.
- **Competitor Landscape:** Cao et al. (2025) documented Granger causality between discussion volume and hiring. Our contribution focuses on semantic content, topic dynamics, and sentiment divergence.
- **Qualitative Context:** Iordache et al. (2026) analyzed fear, blame, and deservingness in AI discussions; our study provides longitudinal, quantitative grounding.
- **Sentiment Methodology:** In line with Zhang et al. (ACM TOSEM 2025), contextual transformers resolve sarcasm and technical jargon where rule-based models fail.

---

## 3. Data & Methodology
- **Corpus:** 2020–2025 Reddit submissions and comments from `r/cscareerquestions`, `r/cscareerquestionsEU`, `r/datascience`, `r/dataengineering`, `r/cybersecurity`, and `r/SoftwareEngineering`.
- **Sampling:** Stratified sampling by ISO week preserving temporal distribution across the 5-year window.
- **Ethics & Privacy:** Usernames were irreversibly hashed using salted SHA-256 (`author_hash`); deleted stubs and bots were removed; no raw user text is redistributed.
- **Topic Modeling:** BERTopic leveraging `all-MiniLM-L6-v2` dense embeddings, UMAP dimensionality reduction, and HDBSCAN density clustering.
- **Interrupted Time Series (ITS):** Weekly negative share modeled as $y_t = \beta_0 + \beta_1 t + \beta_2 D_t + \beta_3 (t - T_{break}) D_t + \varepsilon_t$.

---

## 4. Results

### 4.1 RQ1: Sentiment Validation
Comparing RoBERTa against VADER on tech career text confirms that VADER suffers from severe neutral compression (44% neutral assignments on posts expressing subtle career frustration) and sarcasm misclassification. RoBERTa achieves 71.4% agreement with human-verified polarity labels, successfully detecting interview fatigue. (See Figure 5).

### 4.2 RQ2: Career Topic Evolution
Topic modeling reveals four distinct thematic regimes:
1. *COVID Freeze (2020):* Dominated by remote-work transitions, offer rescissions, and hiring freezes.
2. *Tech Hiring Boom (2021–mid 2022):* Dominated by compensation negotiations, FAANG offers, and LeetCode preparation.
3. *Layoff Contraction (mid 2022–2023):* Dominated by severance discussions, PIP survival, and job loss anxiety.
4. *GenAI Transition (2024–2025):* Marked by tool adoption alongside persistent junior application distress. (See Figure 2).

### 4.3 RQ3a: Specialization Divergence
Comparing technical domains reveals distinct resilience:
- **Cybersecurity** exhibited the lowest increase in negative sentiment (+1.2 pp, 95% CI [-0.4, +2.8]), reflecting sustained enterprise defense demand.
- **Data Engineering** demonstrated moderate stability (+2.1 pp).
- **Data Science** and **General SWE** recorded higher shifts (+3.8 pp and +4.1 pp), driven by junior pipeline saturation and AI automation apprehension. (See Figure 3).

### 4.4 RQ3b: Junior vs. Control ITS Regression
The ITS regression confirms a statistically significant immediate upward level shift in junior distress (`r/cscareerquestions`, $\beta_2 = +0.038, p_{holm} < 0.01$), which is absent in European geographic controls (`r/cscareerquestionsEU`, $\beta_2 = +0.009, p_{holm} > 0.35$). (See Figure 4 and Table 2).

---

## 5. Limitations & Ethical Considerations
- **Community-Level Proxy:** Reddit posters are self-selected and not a random sample of the global tech workforce.
- **Platform Artifacts:** The June 2023 API blackout created an unavoidable 4-week reporting gap; sensitivity analyses confirm our findings remain stable when excluding this window.
- **Causality:** We claim observational and interrupted divergence, not monocausal determination by LLMs alone.

---

## 6. Conclusion
Junior technologist communities experienced an immediate, sustained escalation in hiring-side fatigue and career anxiety following November 2022. While macroeconomic commentary often treats tech labor uniformly, specialized domains (cybersecurity) and non-US markets exhibit marked divergence.
