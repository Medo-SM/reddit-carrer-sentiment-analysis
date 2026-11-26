"""Longitudinal aggregation, Mann-Kendall trend tests, and 2018–2022 regime comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


REGIME_ORDER = [
    "1_Pre-COVID (2018-2020Q1)",
    "2_COVID Freeze (2020)",
    "3_Hiring Boom (2021-2022H1)",
    "4_Layoffs & Tightening (2022H2)",
]


def assign_regime(dt_series: pd.Series) -> pd.Series:
    """Map UTC timestamps onto the four 2018–2022 structural market regimes."""
    dt = pd.to_datetime(dt_series, utc=True)
    naive = dt.dt.tz_convert(None)
    conditions = [
        naive < pd.Timestamp("2020-03-01"),
        (naive >= pd.Timestamp("2020-03-01")) & (naive <= pd.Timestamp("2020-12-31")),
        (naive >= pd.Timestamp("2021-01-01")) & (naive <= pd.Timestamp("2022-05-31")),
        naive >= pd.Timestamp("2022-06-01"),
    ]
    return pd.Series(np.select(conditions, REGIME_ORDER, default="Other"), index=dt_series.index)


def aggregate_quarterly(df: pd.DataFrame, label_col: str = "roberta_label") -> pd.DataFrame:
    """Compute quarterly unweighted & upvote-weighted sentiment per subreddit and overall."""
    work = df.dropna(subset=[label_col, "created_utc_dt"]).copy()
    dt = pd.to_datetime(work["created_utc_dt"], utc=True).dt.tz_convert(None)
    work["quarter"] = dt.dt.to_period("Q").astype(str)

    if "positive" in work.columns and "negative" in work.columns:
        work["net_sentiment"] = work["positive"].astype(float) - work["negative"].astype(float)
    else:
        work["net_sentiment"] = np.where(
            work[label_col] == "positive", 1.0, np.where(work[label_col] == "negative", -1.0, 0.0)
        )

    if "score" in work.columns:
        scores = pd.to_numeric(work["score"], errors="coerce").fillna(0.0).clip(lower=0.0)
        work["weight"] = np.log1p(scores) + 1.0
    else:
        work["weight"] = 1.0

    def _agg_frame(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        rows = []
        for keys, grp in frame.groupby(group_cols, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            n = len(grp)
            w = grp["weight"].to_numpy(dtype=float)
            net = grp["net_sentiment"].to_numpy(dtype=float)
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "n_docs": int(n),
                    "neg_pct": round(float((grp[label_col] == "negative").mean() * 100), 2),
                    "neu_pct": round(float((grp[label_col] == "neutral").mean() * 100), 2),
                    "pos_pct": round(float((grp[label_col] == "positive").mean() * 100), 2),
                    "net_mean": round(float(net.mean()), 4),
                    "net_weighted": round(float(np.average(net, weights=w)), 4),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    by_sub = _agg_frame(work, ["subreddit", "quarter"])
    overall = _agg_frame(work, ["quarter"])
    overall.insert(0, "subreddit", "ALL")
    return pd.concat([overall, by_sub], ignore_index=True)


def mann_kendall_test(values: pd.Series | np.ndarray | list[float]) -> dict[str, float | str]:
    """Non-parametric Mann-Kendall monotonic trend test + Theil-Sen slope."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return {
            "n": int(n),
            "S": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "sen_slope": 0.0,
            "trend": "insufficient_data",
        }

    # Pairwise sign sum S
    s_stat = 0.0
    slopes = []
    for k in range(n - 1):
        diffs = x[k + 1 :] - x[k]
        s_stat += float(np.sum(np.sign(diffs)))
        steps = np.arange(1, n - k, dtype=float)
        slopes.extend((diffs / steps).tolist())

    # Variance with tie adjustment
    _, counts = np.unique(x, return_counts=True)
    tie_counts = counts[counts > 1]
    tie_term = float(np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_s <= 0:
        z_stat = 0.0
    elif s_stat > 0:
        z_stat = (s_stat - 1.0) / np.sqrt(var_s)
    elif s_stat < 0:
        z_stat = (s_stat + 1.0) / np.sqrt(var_s)
    else:
        z_stat = 0.0

    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    sen_slope = float(np.median(slopes)) if slopes else 0.0
    if p_value < 0.05:
        trend = "increasing" if z_stat > 0 else "decreasing"
    else:
        trend = "no_trend"

    return {
        "n": int(n),
        "S": float(s_stat),
        "z": round(float(z_stat), 4),
        "p_value": round(p_value, 6),
        "sen_slope": round(sen_slope, 4),
        "trend": trend,
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Step-down Holm-Bonferroni multiple-testing correction."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    corrected = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj = min(1.0, p_val * (m - rank))
        running_max = max(running_max, adj)
        corrected[orig_idx] = round(running_max, 6)
    return corrected


def regime_comparison(
    df: pd.DataFrame,
    label_col: str = "roberta_label",
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare negative sentiment share across the 4 regimes per subreddit and overall."""
    work = df.dropna(subset=[label_col, "created_utc_dt"]).copy()
    work["regime"] = assign_regime(work["created_utc_dt"])
    work["is_neg"] = (work[label_col] == "negative").astype(float)
    rng = np.random.RandomState(seed)

    targets = ["ALL"] + sorted(work["subreddit"].unique().tolist())
    records = []
    raw_p_layoff_vs_boom = []
    raw_p_kw = []

    for sub in targets:
        sub_df = work if sub == "ALL" else work[work["subreddit"] == sub]
        regime_rates = {}
        regime_groups = []
        for reg in REGIME_ORDER:
            vals = sub_df.loc[sub_df["regime"] == reg, "is_neg"].to_numpy()
            regime_rates[reg] = round(float(vals.mean() * 100), 2) if len(vals) > 0 else np.nan
            if len(vals) > 0:
                regime_groups.append(vals)

        if len(regime_groups) >= 2:
            try:
                kw_h, kw_p = stats.kruskal(*regime_groups)
            except ValueError:
                kw_h, kw_p = 0.0, 1.0
        else:
            kw_h, kw_p = 0.0, 1.0

        boom_vals = sub_df.loc[sub_df["regime"] == REGIME_ORDER[2], "is_neg"].to_numpy()
        layoff_vals = sub_df.loc[sub_df["regime"] == REGIME_ORDER[3], "is_neg"].to_numpy()
        if len(boom_vals) > 1 and len(layoff_vals) > 1:
            delta_pp = (layoff_vals.mean() - boom_vals.mean()) * 100.0
            boot = [
                (
                    rng.choice(layoff_vals, size=len(layoff_vals), replace=True).mean()
                    - rng.choice(boom_vals, size=len(boom_vals), replace=True).mean()
                )
                * 100.0
                for _ in range(n_boot)
            ]
            ci_low, ci_high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            _, mw_p = stats.mannwhitneyu(layoff_vals, boom_vals, alternative="two-sided")
        else:
            delta_pp, ci_low, ci_high, mw_p = np.nan, np.nan, np.nan, 1.0

        raw_p_kw.append(float(kw_p))
        raw_p_layoff_vs_boom.append(float(mw_p))
        records.append(
            {
                "subreddit": sub,
                "n_total": int(len(sub_df)),
                "pre_covid_neg_pct": regime_rates[REGIME_ORDER[0]],
                "covid_freeze_neg_pct": regime_rates[REGIME_ORDER[1]],
                "hiring_boom_neg_pct": regime_rates[REGIME_ORDER[2]],
                "layoffs_2022_neg_pct": regime_rates[REGIME_ORDER[3]],
                "delta_layoffs_vs_boom_pp": round(float(delta_pp), 2) if not np.isnan(delta_pp) else np.nan,
                "ci_95_low": round(ci_low, 2) if not np.isnan(ci_low) else np.nan,
                "ci_95_high": round(ci_high, 2) if not np.isnan(ci_high) else np.nan,
                "kruskal_H": round(float(kw_h), 3),
                "kruskal_p_raw": round(float(kw_p), 6),
                "mw_p_raw": round(float(mw_p), 6),
            }
        )

    out = pd.DataFrame(records)
    out["kruskal_p_holm"] = holm_bonferroni(raw_p_kw)
    out["mw_p_holm"] = holm_bonferroni(raw_p_layoff_vs_boom)
    return out
