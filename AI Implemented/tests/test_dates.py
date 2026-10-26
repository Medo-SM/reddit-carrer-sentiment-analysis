import pandas as pd
from src.modeling.load import parse_created_utc


def test_int_epoch_gives_2023():
    col = pd.Series([1700000000])  # 2023-11-14 in epoch seconds
    result = parse_created_utc(col)
    assert result.dt.year[0] == 2023, f"Expected 2023, got {result.dt.year[0]} (1970 = nanosecond bug)"


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
