import numpy as np
import pandas as pd

from src.modeling.stats import (
    aggregate_quarterly,
    assign_regime,
    holm_bonferroni,
    mann_kendall_test,
    regime_comparison,
)


def test_mann_kendall_increasing_trend():
    res = mann_kendall_test([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert res["trend"] == "increasing"
    assert res["p_value"] < 0.001
    assert np.isclose(res["sen_slope"], 1.0)


def test_mann_kendall_flat_no_trend():
    res = mann_kendall_test([5, 5, 5, 5, 5, 5])
    assert res["trend"] == "no_trend"
    assert res["p_value"] == 1.0


def test_holm_bonferroni_monotonic():
    adj = holm_bonferroni([0.001, 0.02, 0.5])
    assert adj[0] <= adj[1] <= adj[2]
    assert adj[0] == 0.003


def test_aggregate_quarterly_and_regimes():
    dates = pd.to_datetime(
        ["2019-05-01", "2020-06-01", "2021-06-01", "2022-09-01", "2022-10-01"], utc=True
    )
    df = pd.DataFrame({
        "subreddit": ["datascience"] * 5,
        "created_utc_dt": dates,
        "roberta_label": ["positive", "negative", "positive", "negative", "negative"],
        "positive": [0.8, 0.1, 0.9, 0.1, 0.1],
        "negative": [0.1, 0.8, 0.05, 0.8, 0.85],
        "score": [10, 2, 20, 5, 15],
    })
    regimes = assign_regime(df["created_utc_dt"])
    assert regimes.iloc[0].startswith("1_Pre-COVID")
    assert regimes.iloc[1].startswith("2_COVID")
    assert regimes.iloc[2].startswith("3_Hiring")
    assert regimes.iloc[3].startswith("4_Layoffs")

    q = aggregate_quarterly(df)
    assert "ALL" in set(q["subreddit"])
    assert "datascience" in set(q["subreddit"])

    rc = regime_comparison(df, n_boot=50, seed=42)
    assert set(rc["subreddit"]) == {"ALL", "datascience"}
