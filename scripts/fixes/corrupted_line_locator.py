#!/usr/bin/env python3
"""Inspect a JSONL file from the end to locate corrupted (non-JSON) lines.

Usage:
    python scripts/corrupted_line_locator.py [FILE]
    python scripts/corrupted_line_locator.py              # reads path from config
    python scripts/corrupted_line_locator.py data/raw/reddit/subreddits/cscareerquestions_comments.jsonl
"""
import argparse
import json
import os
import sys

import yaml


def locate_corruption(file_path: str) -> bool:
    """Scan file from end to start, reporting corrupted lines.

    Returns True if corruption was found, False otherwise.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}.")
        return False

    print(f"Inspecting {file_path} for corrupted lines...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            print("The file is empty. No lines to inspect.")
            return False

        corrupted_lines_found = False
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            try:
                json.loads(line)
                if corrupted_lines_found:
                    print(f"\nCorruption appears to end just before line {i + 1}.")
                    break
            except json.JSONDecodeError as e:
                if not corrupted_lines_found:
                    print("--- CORRUPTION DETECTED ---")
                    corrupted_lines_found = True
                print(f"Line {i + 1} is corrupted: {line.strip()}\nError: {e}")

        if not corrupted_lines_found:
            print("No corrupted lines detected from the end of the file.")
            print("The problem might be due to an overall file structure issue "
                  "or corruption not detectable by line-by-line JSON parsing.")

        return corrupted_lines_found
    except Exception as e:
        print(f"An unexpected error occurred during file inspection: {e}")
        return False


def _resolve_default_path() -> str | None:
    """Try to load a default path from config/config.yaml."""
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        raw_dir = cfg.get("paths", {}).get("raw_data_dir", "data/raw")
        # Return None — caller must provide a specific file
        print(f"Hint: raw data directory from config is '{raw_dir}'. "
              f"Pass a specific .jsonl file path as argument.")
        return None
    except (FileNotFoundError, yaml.YAMLError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Locate corrupted lines at the end of a JSONL file."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to the JSONL file to inspect. If omitted, reads config for hints.",
    )
    args = parser.parse_args()

    file_path = args.file
    if not file_path:
        file_path = _resolve_default_path()
        if not file_path:
            parser.print_help()
            return 1

    found = locate_corruption(file_path)
    return 0 if not found else 2


if __name__ == "__main__":
    sys.exit(main())
