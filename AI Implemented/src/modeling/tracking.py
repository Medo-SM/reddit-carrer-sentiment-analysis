"""Lightweight JSONL experiment log — one line per modeling run."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def log_experiment(
    params: dict,
    metrics: dict,
    output_dir: str | Path = "data/output/experiments",
) -> Path:
    """Append one experiment record to experiment_log.jsonl.

    Args:
        params: All tunable values (n_docs, dataset_type, hdbscan_*, umap_*, seed, etc.)
        metrics: Outcome values (n_topics, outlier_pct, coherence_cv, agreement_rate, etc.)
        output_dir: Directory for the log file.

    Returns:
        Path to the log file.
    """
    versions = {
        "python": sys.version,
    }
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = "not_installed"

    try:
        import bertopic
        versions["bertopic"] = bertopic.__version__
    except ImportError:
        versions["bertopic"] = "not_installed"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "params": params,
        "metrics": metrics,
        "versions": versions,
    }
    path = Path(output_dir) / "experiment_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return path
