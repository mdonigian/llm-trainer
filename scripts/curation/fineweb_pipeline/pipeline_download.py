#!/usr/bin/env python3
"""
Download FineWeb-Edu parquet files to local disk for fast local I/O.

Uses huggingface_hub.snapshot_download which supports automatic resume —
already-downloaded files are skipped on re-run.

Usage:
  python pipeline_download.py
  python pipeline_download.py --output-dir /workspace/fineweb-curation/raw_data
  python pipeline_download.py --config sample-100BT
"""

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from pipeline_config import DEFAULT_OUTPUT_BASE


def main():
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu parquet files to local disk",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{DEFAULT_OUTPUT_BASE}/raw_data",
        help="Local directory to download into (default: <output_base>/raw_data)",
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--config",
        default="sample-100BT",
        help="Dataset config/subset to download (default: sample-100BT)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"{args.config.replace('-', '/')}/*.parquet"
    if args.config == "sample-100BT":
        pattern = "sample/100BT/*.parquet"

    print(f"Dataset:    {args.dataset}")
    print(f"Config:     {args.config}")
    print(f"Pattern:    {pattern}")
    print(f"Output dir: {output_dir}")
    print()
    print("Starting download (resumes automatically if interrupted)...")

    t0 = time.time()
    local_path = snapshot_download(
        args.dataset,
        repo_type="dataset",
        allow_patterns=pattern,
        local_dir=str(output_dir),
    )
    elapsed = time.time() - t0

    parquet_files = sorted(Path(local_path).rglob("*.parquet"))
    total_bytes = sum(f.stat().st_size for f in parquet_files)

    print(f"\nDownload complete in {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    print(f"  Files: {len(parquet_files)}")
    print(f"  Total size: {total_bytes / 1e9:.1f} GB")
    print(f"  Location: {local_path}")
    print()
    print("Next step: run pipeline_classify.py with --local-dir pointing here:")
    print(f"  python pipeline_classify.py --local-dir {local_path} --compile")


if __name__ == "__main__":
    main()
