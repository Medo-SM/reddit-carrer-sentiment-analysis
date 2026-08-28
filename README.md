# Reddit Tech Career Sentiment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)
[![BERTopic](https://img.shields.io/badge/BERTopic-0.16.4-00A98F.svg)](https://maartengr.github.io/BERTopic/)
[![Release](https://img.shields.io/badge/version-0.1.0-blue.svg)](CHANGELOG.md)

An end-to-end data engineering, topic modeling, and sentiment analysis pipeline for exploring career trends, job market sentiments, and discussions across tech subreddits.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Target Subreddits](#-target-subreddits)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data Gathering & Ingestion](#-data-gathering--ingestion)
  - [Method 1: Academic Torrents Dump](#method-1-academic-torrents-dump)
  - [Method 2: PullPush / Pushshift API Scraper](#method-2-pullpush--pushshift-api-scraper)
  - [Data Repair & Verification Utilities](#data-repair--verification-utilities)
  - [ZST to Parquet Conversion](#zst-to-parquet-conversion)
- [Configuration Management](#-configuration-management)
- [Exploration & Modeling](#-exploration--modeling)
- [CI/CD & Changelog Automation](#-cicd--changelog-automation)
- [License](#-license)

---

## 🔍 Overview

The tech job market undergoes rapid shifts driven by economic cycles, AI developments, and hiring trends. **Reddit Tech Career Sentiment Analysis** provides a scalable framework to ingest, clean, process, and analyze millions of Reddit posts and comments from major software engineering, data science, and career-focused subreddits.

The pipeline combines historical Reddit dumps (up to December 2022) with custom API scrapers (2023–2025) to generate compressed, optimized Parquet datasets suitable for NLP tasks like BERTopic modeling, sentiment classification, and term frequency trend analysis.

---

## ✨ Key Features

- **Multi-Source Data Ingestion:**
  - Ingestion of large-scale `.zst` historical dumps via `pyzstd` streaming.
  - Async web scraper for continuous API data collection with HTTP rate-limit (429) backoff and state resumption.
- **Resilient Data Maintenance:**
  - Automated detection and repair of malformed/corrupted JSONL lines.
  - Verification script to identify missing scrape dates across multi-year spans.
- **Efficient Storage & ETL Pipeline:**
  - Memory-efficient batch processing converting raw JSON/JSONL/ZST formats into snappy-compressed `.parquet` files (`pyarrow` / `pandas`).
  - Text normalization, URL removal, and markdown scrubbing.
- **Advanced NLP & Modeling:**
  - Exploratory analysis notebooks for term frequency and n-gram extraction.
  - Topic extraction using **BERTopic** and pre-trained Transformer embeddings (PyTorch).
- **Flexible Configuration & Workflow Automation:**
  - Centralized YAML config (`config/config.yaml`) for directory paths, data limits, and hyperparameters.
  - Automated Conventional Commit changelog and GitHub Release generation (`.github/workflows/changelog.yml`).

---

## 🎯 Target Subreddits

The project focuses on career-centric tech subreddits:

| Subreddit | Domain |
| :--- | :--- |
| `r/cscareerquestions` | General Computer Science career advice and hiring |
| `r/cscareerquestionsEU` | European tech market discussions |
| `r/datascience` | Data Science career, tools, and industry trends |
| `r/dataengineering` | Data Engineering architecture and career discussions |
| `r/cybersecurity` | Information security careers and industry updates |
| `r/SoftwareEngineering` | Software engineering practices and career growth |

---

## 📁 Repository Structure

```text
reddit-carrer-sentiment-analysis/
├── .github/
│   └── workflows/
│       └── changelog.yml         # GitHub Actions semantic release & changelog workflow
├── config/
│   ├── config.yaml               # Centralized configuration (paths, preprocessing, training)
│   └── README.md                 # Usage documentation for config loading
├── data/                         # Data directory (gitignore raw datasets)
│   ├── raw/                      # Raw .zst or .jsonl dumps
│   ├── processed/                # Cleaned .parquet datasets
│   ├── samples/                  # Sample data subsets for quick prototyping
│   └── final/                    # Exported outputs for visualization
├── models/                       # Model checkpoints & serialized models
├── notebooks/                    # Jupyter notebooks for EDA and modeling
│   ├── 1.0_data_exploration.ipynb
│   ├── 1.1_notebook.ipynb
│   ├── 1.2_notebook.ipynb
│   ├── 1.3_notebook.ipynb
│   └── requirements.txt          # Notebook-specific environment requirements
├── src/
│   ├── __init__.py
│   └── ingest/
│       ├── __init__.py
│       ├── zst_to_parquet.py     # ZST stream decompressor & Parquet converter
│       └── scraper/
│           ├── __init__.py
│           ├── core_scraper.py   # Async PullPush API scraper with retry logic
│           ├── corrupted_line_locator.py # Line-by-line JSON validity inspector
│           ├── corruption_fix.py # Automated end-of-file truncation repair tool
│           └── missing_dates_check.py    # Date coverage audit tool
├── tests/                        # Unit and integration tests
│   └── __init__.py
├── CHANGELOG.md                  # Project changelog generated via conventional commits
├── LICENSE                       # MIT License
├── package.json                  # Semantic versioning manifest (v0.1.0)
├── requirements_cpu.txt          # Dependencies for CPU execution
├── requirements_gpu.txt          # Dependencies for CUDA/GPU execution
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python:** 3.10 or higher
- **Package Manager:** `pip` or `conda`
- **GPU (Optional):** NVIDIA GPU with CUDA support (for accelerated BERTopic/Transformer inference)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Medo-SM/reddit-carrer-sentiment-analysis.git
   cd reddit-carrer-sentiment-analysis
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   - **For CPU-only environments:**
     ```bash
     pip install -r requirements_cpu.txt
     ```

   - **For CUDA/GPU-enabled environments:**
     ```bash
     pip install -r requirements_gpu.txt
     ```

   - **For Jupyter Notebook exploration:**
     ```bash
     pip install -r notebooks/requirements.txt
     ```

---

## 📊 Data Gathering & Ingestion

Data acquisition is split into two complementary methods:

### Method 1: Academic Torrents Dump
Historical Reddit data up to **December 31, 2022** is available via the Academic Torrents dump:
- **Torrent Link:** [Top 40k Subreddits Dump (Academic Torrents)](https://academictorrents.com/details/3e3f64dee22dc304cdd2546254ca1f8e8ae542b4)
- Place downloaded `.zst` files under `data/raw/` or `data/reddit/subreddits/`.

### Method 2: PullPush / Pushshift API Scraper
To gather recent data (**2023 to 2025**), use the included async API scraper:

```bash
python src/ingest/scraper/core_scraper.py
```
- **Features:** Supports custom date ranges, handles HTTP 429 rate limits automatically, and resumes from the last scraped timestamp to prevent duplicates.

### Data Repair & Verification Utilities

If scraping is interrupted or network failures produce incomplete JSON files, use the provided diagnostic scripts in `src/ingest/scraper/`:

1. **Check missing dates:**
   ```bash
   python src/ingest/scraper/missing_dates_check.py
   ```
2. **Locate corrupted JSON lines:**
   ```bash
   python src/ingest/scraper/corrupted_line_locator.py
   ```
3. **Automatically fix & truncate corrupted lines:**
   ```bash
   python src/ingest/scraper/corruption_fix.py
   ```

### ZST to Parquet Conversion

Convert raw Zstandard compressed dumps (`.zst`) to optimized `.parquet` tables with text cleaning, Markdown stripping, and batch writing:

```bash
python src/ingest/zst_to_parquet.py
```

---

## ⚙️ Configuration Management

The project uses `config/config.yaml` to define dataset paths, text preprocessing parameters, and training settings.

Example usage in Python:

```python
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

raw_dir = config["paths"]["raw_data_dir"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
```

---

## 🧪 Exploration & Modeling

Explore data and experiment with models using the provided notebooks in `notebooks/`:

- `1.0_data_exploration.ipynb`: Data loading, word frequencies, and initial text cleaning diagnostics.
- `1.1_notebook.ipynb` & `1.2_notebook.ipynb`: Advanced EDA, sub-community analysis, and sentiment trends.
- `1.3_notebook.ipynb`: Topic modeling with BERTopic, sentiment scoring, and topic evolution visualization.

---

## 🔄 CI/CD & Changelog Automation

This repository uses GitHub Actions (`.github/workflows/changelog.yml`) powered by `conventional-changelog-action`.

To ensure automatic changelog updates and version bumps:
- Use [Conventional Commits](https://www.conventionalcommits.org/) format (e.g., `feat: add scraper retry logic`, `fix: resolve json decoding error`).
- On pushing to `main`, the workflow updates `CHANGELOG.md` and generates a GitHub Release automatically.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

