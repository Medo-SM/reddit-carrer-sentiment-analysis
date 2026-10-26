"""Modeling package — call set_seed(42) at the top of every notebook."""
from __future__ import annotations

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and torch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
