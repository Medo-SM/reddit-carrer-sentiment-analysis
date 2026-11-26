import json
import tempfile
from pathlib import Path

from src.modeling.tracking import log_experiment


def test_log_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = log_experiment({"seed": 42}, {"n_topics": 15}, output_dir=tmp)
        assert Path(path).exists()


def test_log_appends():
    with tempfile.TemporaryDirectory() as tmp:
        path = log_experiment({"run": 1}, {"outlier_pct": 0.1}, output_dir=tmp)
        log_experiment({"run": 2}, {"outlier_pct": 0.2}, output_dir=tmp)
        lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2


def test_log_valid_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = log_experiment({"a": 1}, {"b": 2}, output_dir=tmp)
        line = Path(path).read_text(encoding="utf-8").strip()
        entry = json.loads(line)
        assert "timestamp" in entry and "params" in entry and "metrics" in entry
