import json
import os

# check the last few lines of the chosen jsonl file to ensure they are properly formatted JSON objects.
# This can help diagnose issues with pd.read_json if the file is corrupted at the end.

DATA_DIR = "/content/drive/MyDrive/reddit_project"
file_path = f"{DATA_DIR}/cscareerquestions_comments.jsonl"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}.")
else:
    print(f"Inspecting {file_path} for corrupted lines...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print("The file is empty. No lines to inspect.")
        else:
            corrupted_lines_found = False
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                try:
                    json.loads(line)
                    # If we reach here, this line is valid. If we found corrupted lines previously,
                    # this marks the end of the corrupted block.
                    if corrupted_lines_found:
                        print(f"\nCorruption appears to end just before line {i + 1}.")
                        break
                except json.JSONDecodeError as e:
                    if not corrupted_lines_found:
                        print(f"--- CORRUPTION DETECTED ---")
                        corrupted_lines_found = True
                    print(f"Line {i + 1} is corrupted: {line.strip()}\nError: {e}")

            if not corrupted_lines_found:
                print("No corrupted lines detected from the end of the file.")
                print("The problem might be due to an overall file structure issue or corruption not detectable by line-by-line JSON parsing.")

    except Exception as e:
        print(f"An unexpected error occurred during file inspection: {e}")
