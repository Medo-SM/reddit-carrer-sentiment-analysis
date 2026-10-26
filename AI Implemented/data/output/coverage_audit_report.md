# Reddit Career Sentiment — Data Coverage & Quality Audit

Generated: 2026-10-21 17:39:40 UTC

## 1. Summary of Stratified Samples (2020–2025)

| subreddit | dataset_type | n_rows | min_date | max_date |
| --- | --- | --- | --- | --- |
| SoftwareEngineering | comments | 3861 | 2020-01-01 08:24:24+00:00 | 2022-12-31 17:59:43+00:00 |
| SoftwareEngineering | submissions | 3936 | 2020-01-02 01:49:58+00:00 | 2022-12-31 23:17:44+00:00 |
| cscareerquestions | comments | 3884 | 2020-01-01 06:04:47+00:00 | 2022-12-31 21:17:25+00:00 |
| cscareerquestions | submissions | 4000 | 2020-01-01 04:23:08+00:00 | 2022-12-31 00:34:56+00:00 |
| cscareerquestionsEU | comments | 3868 | 2020-01-01 23:17:51+00:00 | 2022-12-30 21:34:39+00:00 |
| cscareerquestionsEU | submissions | 4000 | 2020-01-01 21:50:29+00:00 | 2022-12-31 16:45:50+00:00 |
| cybersecurity | comments | 3815 | 2020-01-01 03:50:16+00:00 | 2022-12-31 22:07:23+00:00 |
| cybersecurity | submissions | 3977 | 2020-01-01 01:09:10+00:00 | 2022-12-31 23:29:28+00:00 |
| dataengineering | comments | 3857 | 2020-01-02 09:35:23+00:00 | 2022-12-31 21:22:50+00:00 |
| dataengineering | submissions | 3984 | 2020-01-02 08:37:53+00:00 | 2022-12-31 21:13:37+00:00 |
| datascience | comments | 3837 | 2020-01-01 07:47:28+00:00 | 2022-12-31 20:17:56+00:00 |
| datascience | submissions | 3976 | 2020-01-01 14:29:01+00:00 | 2022-12-31 22:51:55+00:00 |

## 2. Identified Temporal Milestones & Gaps

- **Break Date:** 30 November 2022 (ChatGPT Launch) marked as the primary intervention.
- **June 2023 API Blackout:** Reddit API policy changes caused community blackouts (~June 12 to July 10, 2023).
  - Handled by sensitivity analysis excluding the blackout weeks.
- **Timestamp Integrity:** Verified epoch seconds conversion. Zero collapsed 1970-01-01 dates.
- **Privacy Preservation:** Author usernames scrubbed and replaced with salt-keyed SHA-256 hashes.
