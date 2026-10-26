import io
import json
import os
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import pyzstd
except ImportError:
    pyzstd = None

# =============================================================================
# Configuration
# =============================================================================

CWD = Path.cwd()
INPUT_DIR = CWD / "reddit" / "subreddits"
OUTPUT_DIR = CWD

# --- Schema ---
TARGET_FIELDS_BASE = ["id", "subreddit", "created_utc", "score"]
# Handle both submission ('selftext') and comment ('body') schemas
TEXT_FIELDS = ["selftext", "body", "title"]

# --- Performance and Memory ---
BATCH_SIZE = 50_000


# =============================================================================
# Helper Functions
# =============================================================================

def clean_text(text: str) -> str:
    """
    Cleans raw Reddit text by removing URLs, Markdown, and normalizing whitespace.

    Args:
        text: The raw text content from a Reddit post or comment.

    Returns:
        The cleaned text, or an empty string if the input is invalid.
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text, flags=re.IGNORECASE)
    # 2. Remove Markdown markers (*, _, #)
    text = re.sub(r"[*_#`~>|]", "", text)
    # 3. Normalize whitespace (strips leading/trailing, replaces multiple spaces/newlines)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def read_zst_json_stream(file_path: Path):
    """
    Generator that yields JSON objects from a Zstandard-compressed file,
    optimized for large window sizes typical of Pushshift dumps.

    Args:
        file_path: Path to the .zst file.
    """
    if pyzstd is None:
        raise ImportError("pyzstd package is required to decompress .zst files.")
    dctx = pyzstd.ZstdDecompressor(option={pyzstd.DParameter.windowLogMax: 31})

    with file_path.open("rb") as f:
        buffer = b""
        while True:
            chunk = f.read(16384)
            if not chunk:
                break
            decompressed = dctx.decompress(chunk)
            buffer += decompressed
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# =============================================================================
# Main Processing Logic
# =============================================================================

def main():
    """
    Main function to orchestrate the conversion from ZST to Parquet.
    """
    print("Starting Reddit data conversion from ZST to Parquet...")
    print(f"Input directory: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    input_files = list(INPUT_DIR.glob("*.zst"))
    if not input_files:
        print(f"Error: No .zst files found in {INPUT_DIR}. Exiting.")
        return

    print(f"Found {len(input_files)} files to process.")
    total_processed_count = 0

    for file_path in input_files:
        output_file = OUTPUT_DIR / f"{file_path.stem}.parquet"
        if output_file.exists():
            output_file.unlink()
            print(f"Removed existing output file: {output_file}")

        print(f"\nProcessing file: {file_path.name} -> {output_file.name}")
        processed_count = 0
        batch_records = []
        writer = None

        try:
            for record in read_zst_json_stream(file_path):
                body_text = record.get("body", record.get("selftext", ""))
                if body_text in ("[deleted]", "[removed]"):
                    continue

                title = record.get("title", "")
                full_text = f"{title} {body_text}" if title else body_text
                cleaned_text = clean_text(full_text)

                if not cleaned_text:
                    continue

                extracted_data = {
                    field: record.get(field) for field in TARGET_FIELDS_BASE
                }
                extracted_data["text"] = cleaned_text
                batch_records.append(extracted_data)
                processed_count += 1
                total_processed_count += 1

                if len(batch_records) >= BATCH_SIZE:
                    df = pd.DataFrame(batch_records)
                    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")
                    df = df[["id", "subreddit", "created_utc", "score", "text"]]
                    table = pa.Table.from_pandas(df)
                    if writer is None:
                        writer = pq.ParquetWriter(output_file, table.schema, compression="snappy")
                    writer.write_table(table)
                    print(f"  ... saved batch of {len(batch_records)} records. Total for this file: {processed_count}")
                    batch_records.clear()

            if batch_records:
                df = pd.DataFrame(batch_records)
                df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")
                df = df[["id", "subreddit", "created_utc", "score", "text"]]
                table = pa.Table.from_pandas(df)
                if writer is None:
                    writer = pq.ParquetWriter(output_file, table.schema, compression="snappy")
                writer.write_table(table)
                print(f"\nSaved final batch of {len(batch_records)} records for {output_file.name}.")

        finally:
            if writer:
                writer.close()

        print(f"Finished processing {file_path.name}. Total records saved to {output_file.name}: {processed_count}")

    print(f"\nConversion complete. Total records saved across all files: {total_processed_count}")


if __name__ == "__main__":
    main()
