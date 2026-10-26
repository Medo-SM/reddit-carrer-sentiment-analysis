# RQ1 Validation Evidence: RoBERTa vs. VADER on Reddit Career Discourse

- **Benchmark Sample:** 1000 stratified comments across 6 communities.
- **Observed Agreement Rate:** 84.4%

### Error Pattern Analysis
1. **VADER Neutral Skew:** VADER assigns compound score = 0.00 to posts containing technical jargon or mixed sentiment.
2. **Sarcasm and Career Distress:** Phrases like *'Another great round of 500 applications!'* are marked positive by VADER lexicon rules, but correctly identified as negative/frustrated by RoBERTa contextual embeddings.
3. **Primary Model Justification:** RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`) provides superior sensitivity to informal tech career angst, validating its selection as primary measure.
