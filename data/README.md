# Data Pipeline

This project uses two data sources that feed into a shared Parquet-based analysis pipeline.

## Data flow

```
Source A: Academic Torrents (historical, pre-2023)
  *.zst dumps
    └─> src/ingest/zst_to_parquet.py
          └─> data/raw/*.parquet

Source B: API scraping (2023-2025)
  Arctic Shift / PullPush API
    └─> src/ingest/scraper/core_scraper.py
          └─> data/raw/reddit/subreddits/*.jsonl
                └─> (not yet connected to Parquet pipeline — manual conversion needed)

Parquet files
  └─> Notebook sampling (cells in 1.x notebooks)
        └─> data/samples/{notebook_ver}/*.json
              └─> Modeling pipeline (BERTopic + sentiment)
```

## Directory layout

```
data/
├── raw/
│   ├── *.parquet                        # Converted from ZST (Snappy-compressed)
│   └── reddit/subreddits/
│       └── *.zst                        # Original compressed Reddit dumps
└── samples/
    └── 1.2/                             # Tiny sample (~68 KB, 504 lines)
```

## Subreddits covered

| Subreddit                | Content type        |
|--------------------------|---------------------|
| `cscareerquestions`      | CS career advice    |
| `cscareerquestionsEU`   | European tech jobs  |
| `datascience`            | Data Science        |
| `dataengineering`        | Data Engineering    |
| `cybersecurity`          | InfoSec careers     |
| `SoftwareEngineering`    | SWE practices       |

Each subreddit has both `_submissions` and `_comments` files in the raw data.

## Regenerating samples

The samples in `data/samples/1.2/` are tiny placeholders (504 lines total). To
regenerate real samples for modeling validation, run the sampling cells in
`notebooks/1.3_notebook.ipynb` (cell 7) against the full `data/raw/*.parquet` files.

## Known gap

Scraped JSONL data (`data/raw/reddit/subreddits/*.jsonl`) is not yet automatically
fed into `zst_to_parquet.py`. A JSONL-to-Parquet conversion step needs to be added
to bridge Source B into the analysis pipeline.
