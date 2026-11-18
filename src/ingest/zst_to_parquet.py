import html
import io
import json
import os
import re
import pyzstd
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =============================================================================
# Configuration
# =============================================================================

# --- Input and Output Paths ---
# Assumes the script is in `data/` and the ZST files are in `data/reddit/subreddits/`
# Modify these paths if your directory structure is different.
CWD = Path.cwd()
INPUT_DIR = CWD / "reddit" / "subreddits"
OUTPUT_DIR = CWD

# --- Schema ---
TARGET_FIELDS_BASE = ["id", "subreddit", "created_utc", "score"]
# Handle both submission ('selftext') and comment ('body') schemas
TEXT_FIELDS = ["selftext", "body", "title"]

# --- Performance and Memory ---
# Batch size for processing records before writing to Parquet
BATCH_SIZE = 50_000


# =============================================================================
# Helper Functions
# =============================================================================

def clean_text(text: str) -> str:
    """
    Cleans raw Reddit text for sentiment analysis by decoding HTML entities,
    removing URLs and Markdown markers, and normalizing whitespace.
    """
    if not isinstance(text, str):
        return ""

    # 1. Decode Reddit HTML entities (&amp; -> &, &gt; -> >)
    text = html.unescape(text)
    # 2. Replace Markdown links [anchor text](url) with just the anchor text
    text = re.sub(r"\[([^\]]+)\]\((?:http|www)\S+\)", r"\1", text, flags=re.IGNORECASE)
    # 3. Remove remaining raw URLs (note the escaped dot in www\.)
    text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)
    # 4. Remove Reddit Markdown markers (*, _, #, `, ~, >, |)
    text = re.sub(r"[*_#`~>|]", "", text)
    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def read_zst_json_stream(file_path: Path):
    """
    Generator that yields JSON objects from a Zstandard-compressed file,
    optimized for large window sizes typical of Pushshift dumps.

    Args:
        file_path: Path to the .zst file.
    """
    # Use a decompressor with a large window to handle historical Reddit dumps
    dctx = pyzstd.ZstdDecompressor(option={pyzstd.DParameter.windowLogMax: 31})

    with file_path.open("rb") as f:
        # Read the file in chunks, decompress, and handle multi-line JSON objects
        buffer = b""
        while True:
            chunk = f.read(16384)
            if not chunk:
                break
            
            # Decompress the chunk
            decompressed = dctx.decompress(chunk)
            buffer += decompressed
            
            # Process lines
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    # Yield each valid JSON object
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Skip corrupted or malformed lines
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

    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Discover all .zst files in the input directory
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
                # 1. Skip deleted/removed content
                body_text = record.get("body", record.get("selftext", ""))
                if body_text in ("[deleted]", "[removed]"):
                    continue

                # 2. Extract and combine text fields
                title = record.get("title", "")
                # Combine title and body if both exist
                full_text = f"{title} {body_text}" if title else body_text
                cleaned_text = clean_text(full_text)

                # Skip if text is empty after cleaning
                if not cleaned_text:
                    continue

                # 3. Safely extract base fields
                extracted_data = {
                    field: record.get(field) for field in TARGET_FIELDS_BASE
                }
                extracted_data["text"] = cleaned_text

                batch_records.append(extracted_data)
                processed_count += 1
                total_processed_count += 1

                # 4. Write to Parquet in batches
                if len(batch_records) >= BATCH_SIZE:
                    df = pd.DataFrame(batch_records)
                    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")
                    df = df[["id", "subreddit", "created_utc", "score", "text"]]
                    table = pa.Table.from_pandas(df)
                    if writer is None:
                        writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
                    writer.write_table(table)
                    
                    print(f"  ... saved batch of {len(batch_records)} records. Total for this file: {processed_count}")
                    batch_records.clear() # Reset batch

            # Save any remaining records in the last batch
            if batch_records:
                df = pd.DataFrame(batch_records)
                df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")
                df = df[["id", "subreddit", "created_utc", "score", "text"]]
                table = pa.Table.from_pandas(df)
                if writer is None:
                    writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
                writer.write_table(table)
                print(f"\nSaved final batch of {len(batch_records)} records for {output_file.name}.")

        finally:
            if writer:
                writer.close()

        print(f"Finished processing {file_path.name}. Total records saved to {output_file.name}: {processed_count}")

    print(f"\nConversion complete. Total records saved across all files: {total_processed_count}")


if __name__ == "__main__":
    main()