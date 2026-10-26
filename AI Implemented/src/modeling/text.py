"""Text preprocessing — thin vectorised wrapper over src.ingest.sampling.process_pipeline."""
from __future__ import annotations

import pandas as pd

try:
    from src.ingest.sampling import process_pipeline
except ImportError:
    try:
        from ..ingest.sampling import process_pipeline
    except Exception:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.ingest.sampling import process_pipeline


def apply_pipeline(series: pd.Series) -> pd.Series:
    """Apply the full cleaning pipeline to every string in a Series.

    Handles None / NaN values by returning an empty string.
    """
    return series.fillna("").astype(str).map(process_pipeline)
