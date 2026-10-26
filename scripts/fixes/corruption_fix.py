#!/usr/bin/env python3
"""Iteratively truncate corrupted trailing lines from a JSONL file.

Usage:
    python scripts/corruption_fix.py FILE
    python scripts/corruption_fix.py data/raw/reddit/subreddits/cscareerquestions_comments.jsonl
    python scripts/corruption_fix.py FILE --max-attempts 20
"""
import argparse
import json
import os
import sys

import pandas as pd


def fix_corruption(file_path: str, max_attempts: int = 10) -> bool:
    """Iteratively remove corrupted trailing lines until pandas can load the file.

    Returns True if the file is clean after fixing, False otherwise.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}.")
        return False

    print(f"Starting iterative fix for {file_path}...")
    attempt = 0
    file_is_clean = False

    while not file_is_clean and attempt < max_attempts:
        attempt += 1
        print(f"\n--- Attempt {attempt}/{max_attempts} ---")
        corrupted_line_index = -1

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                print("File is empty. No corruption to fix.")
                return True

            # Step 1: Inspect from the end for corruption
            for i in range(len(lines) - 1, -1, -1):
                try:
                    json.loads(lines[i])
                except json.JSONDecodeError:
                    corrupted_line_index = i + 1  # 1-based index
                    print(f"Found corruption at line {corrupted_line_index}.")
                    print(f"Corrupted snippet: {lines[i].strip()[:100]}...")
                    break

            if corrupted_line_index != -1:
                # Step 2: Truncate the file if corruption was found
                valid_lines_count = corrupted_line_index - 1
                if valid_lines_count <= 0:
                    print("Corruption is at or near the beginning of the file. "
                          "Cannot auto-fix by truncation.")
                    break

                if valid_lines_count < len(lines):
                    valid_lines = lines[:valid_lines_count]
                    num_removed = len(lines) - valid_lines_count

                    with open(file_path, "w", encoding="utf-8") as f_write:
                        f_write.writelines(valid_lines)
                    print(f"Successfully truncated the file. Removed {num_removed} lines.")
                else:
                    print("Corruption found, but no lines to remove. This is unexpected.")
                    break

            # Step 3: Verify with pandas
            try:
                df_check = pd.read_json(file_path, lines=True)
                print(f"File now successfully loaded by pandas after {attempt} "
                      f"attempts. Total records: {len(df_check):,}")

                if "created_utc" in df_check.columns:
                    df_check["created_utc_dt"] = pd.to_datetime(
                        df_check["created_utc"], unit="s", utc=True
                    )
                    min_date = df_check["created_utc_dt"].min()
                    max_date = df_check["created_utc_dt"].max()
                    print(f"Date Range: {min_date.strftime('%Y-%m-%d %H:%M:%S%z')} "
                          f"to {max_date.strftime('%Y-%m-%d %H:%M:%S%z')}")

                file_is_clean = True
            except ValueError as e:
                print(f"Still unable to load file with pandas after this attempt: {e}")
                if corrupted_line_index == -1:
                    print("No new corruption found at the end by line-by-line JSON parsing, "
                          "but pandas still fails.")
                    print("The corruption might be structural. Manual inspection or "
                          "full re-scrape may be needed.")
                    break
        except Exception as e:
            print(f"An unexpected error occurred during an iteration: {e}")
            break

    if not file_is_clean:
        print("\n--- FIXING FAILED ---")
        print("The file could not be cleaned after multiple attempts. "
              "Manual inspection or a full re-scrape is recommended.")

    return file_is_clean


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Iteratively fix corrupted trailing lines in a JSONL file."
    )
    parser.add_argument("file", help="Path to the JSONL file to fix.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="Maximum truncation attempts (default: 10).",
    )
    args = parser.parse_args()

    success = fix_corruption(args.file, args.max_attempts)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
