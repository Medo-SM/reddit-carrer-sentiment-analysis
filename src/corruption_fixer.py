import json
import os
import pandas as pd
from datetime import datetime, timezone
DATA_DIR = "/content/drive/MyDrive/reddit_project"
file_path = f"{DATA_DIR}/cscareerquestions_comments.jsonl"

# This script attempts to iteratively fix the chosen jsonl file.
# It works by repeatedly finding the last corrupted line from the end, truncating the file to remove it,
# and then attempting to load the file with pandas.
# This process continues until pandas can successfully read the file or if a deeper corruption is suspected.

max_attempts = 10  # Set a limit to prevent infinite loops
attempt = 0
file_is_clean = False

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}. Please ensure the scraper has run at least once.")
else:
    print(f"Starting iterative fix for {file_path}...")
    while not file_is_clean and attempt < max_attempts:
        attempt += 1
        print(f"\n--- Attempt {attempt}/{max_attempts} ---")
        corrupted_line_index = -1

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                print("File is empty. No corruption to fix.")
                file_is_clean = True
                break

            # Step 1: Inspect from the end for corruption
            for i in range(len(lines) - 1, -1, -1):
                try:
                    json.loads(lines[i])
                except json.JSONDecodeError:
                    corrupted_line_index = i + 1 # 1-based index
                    print(f"Found corruption at line {corrupted_line_index}.\nCorrupted snippet: {lines[i].strip()[:100]}...")
                    break

            if corrupted_line_index != -1:
                # Step 2: Truncate the file if corruption was found
                valid_lines_count = corrupted_line_index - 1
                if valid_lines_count <= 0:
                    print("Corruption is at or near the beginning of the file. Cannot auto-fix by truncation.")
                    break # Give up if first line is corrupted

                if valid_lines_count < len(lines):
                    valid_lines = lines[:valid_lines_count]
                    num_removed = len(lines) - valid_lines_count

                    with open(file_path, 'w', encoding='utf-8') as f_write:
                        f_write.writelines(valid_lines)
                    print(f"Successfully truncated the file. Removed {num_removed} lines.")
                else:
                    print("Corruption found, but no lines to remove (corrupted line was beyond file length?). This is unexpected.")
                    break # Unexpected state

            # Step 3: Verify with pandas
            try:
                df_check = pd.read_json(file_path, lines=True)
                print(f"File now successfully loaded by pandas after {attempt} attempts. Total comments: {len(df_check):,}")

                # Convert 'created_utc' to datetime objects and find min/max dates
                df_check['created_utc_dt'] = pd.to_datetime(df_check['created_utc'], unit='s', utc=True)
                min_date = df_check['created_utc_dt'].min()
                max_date = df_check['created_utc_dt'].max()
                print(f"Date Range: {min_date.strftime('%Y-%m-%d %H:%M:%S%z')} to {max_date.strftime('%Y-%m-%d %H:%M:%S%z')}")

                print("You can now rerun the scraper (cell 'xIBFF7rdvbDH') to continue scraping from this point.")
                file_is_clean = True
            except ValueError as e:
                print(f"Still unable to load file with pandas after this attempt: {e}")
                if corrupted_line_index == -1: # No new corruption found, but pandas still fails
                    print("No new corruption found at the end by line-by-line JSON parsing, but pandas still fails.")
                    print("The corruption might be structural or not just at the end. Manual inspection or full re-scrape may be needed.")
                    break
        except Exception as e:
            print(f"An unexpected error occurred during an iteration: {e}")
            break

    if not file_is_clean:
        print("\n--- FIXING FAILED ---")
        print("The file could not be cleaned after multiple attempts. Manual inspection or a full re-scrape is recommended.")

