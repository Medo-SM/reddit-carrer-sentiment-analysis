import pandas as pd

from src.modeling.load import draw_stratified_subset, parse_created_utc


def test_int_epoch_gives_2023():
    col = pd.Series([1700000000])
    result = parse_created_utc(col)
    assert result.dt.year[0] == 2023


def test_float_epoch():
    col = pd.Series([1700000000.0])
    result = parse_created_utc(col)
    assert result.dt.year[0] == 2023


def test_iso_string():
    col = pd.Series(["2023-11-14T10:00:00"])
    result = parse_created_utc(col)
    assert result.dt.year[0] == 2023


def test_datetime_noop():
    col = pd.to_datetime(pd.Series(["2023-11-14"]))
    result = parse_created_utc(col)
    assert result.dt.year[0] == 2023


def test_draw_stratified_subset_balances_subreddits():
    rows = []
    for sub in ["subA", "subB"]:
        for q_month in ["2019-01-15", "2020-04-15", "2021-07-15", "2022-10-15"]:
            for i in range(10):
                rows.append({"subreddit": sub, "created_utc_dt": pd.Timestamp(q_month, tz="UTC"), "val": i})
    df = pd.DataFrame(rows)
    sub_df = draw_stratified_subset(df, n_per_subreddit=8, seed=42)
    assert len(sub_df) == 16
    assert (sub_df["subreddit"].value_counts() == 8).all()
