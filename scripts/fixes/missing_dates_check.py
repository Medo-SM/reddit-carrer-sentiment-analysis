#!/usr/bin/env python3
"""Count unique dates in a JSONL file's created_utc column and compare to expected.

Usage:
    python scripts/missing_dates_check.py FILE
    python scripts/missing_dates_check.py FILE --expected 2223
    python scripts/missing_dates_check.py data/raw/reddit/subreddits/cscareerquestions_comments.jsonl
"""
import argparse
import sys

import pandas as pd


def check_missing_dates(file_path: str, expected_count: int) -> int:
    """Check for missing dates in the created_utc column.

    Returns the number of missing dates (0 = all present, negative = extra dates).
    """
    print(f"Checking date coverage in {file_path}...")
    chunks = pd.read_json(file_path, lines=True, chunksize=10000)

    all_date_list = []
    for chunk in chunks:
        chunk["created_utc"] = pd.to_datetime(chunk["created_utc"], unit="s", utc=True)
        chunk["created_utc"] = chunk["created_utc"].dt.date
        unique_dates = chunk["created_utc"].drop_duplicates().dropna()
        all_date_list.append(unique_dates)

    filtered_df = pd.concat(all_date_list, ignore_index=True)
    filtered_df = filtered_df.drop_duplicates()

    data_count = len(filtered_df)
    difference_count = expected_count - data_count

    print(f"Unique dates found: {data_count}")
    print(f"Expected dates:     {expected_count}")

    if data_count == expected_count:
        print("No missing dates.")
    elif data_count > expected_count:
        print(f"Extra dates found: {-difference_count}")
    else:
        print(f"Missing dates: {difference_count}")

    return difference_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count unique dates in a JSONL file and compare to expected."
    )
    parser.add_argument("file", help="Path to the JSONL file to check.")
    parser.add_argument(
        "--expected",
        type=int,
        default=2223,
        help="Expected number of unique dates (default: 2223).",
    )
    args = parser.parse_args()

    diff = check_missing_dates(args.file, args.expected)
    return 0 if diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
