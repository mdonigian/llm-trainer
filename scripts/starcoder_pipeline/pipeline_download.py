#!/usr/bin/env python3
"""
Download StarCoderData parquet files to local disk for fast local I/O.

Uses huggingface_hub.snapshot_download which supports automatic resume —
already-downloaded files are skipped on re-run.

StarCoderData is organized by programming language (e.g., python/, javascript/)
so you can specify which languages to download.

Usage:
  python pipeline_download.py
  python pipeline_download.py --languages python javascript typescript go rust
  python pipeline_download.py --output-dir /workspace/starcoder-curation/raw_data
"""

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from pipeline_config import DEFAULT_DATASET, DEFAULT_OUTPUT_BASE, RECOMMENDED_LANGUAGES


def main():
    parser = argparse.ArgumentParser(
        description="Download StarCoderData parquet files to local disk",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{DEFAULT_OUTPUT_BASE}/raw_data",
        help="Local directory to download into (default: <output_base>/raw_data)",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset repo ID (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=RECOMMENDED_LANGUAGES,
        help="Programming languages to download (default: all recommended)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = args.languages or RECOMMENDED_LANGUAGES
    patterns = [f"{lang}/*.parquet" for lang in languages]

    print(f"Dataset:    {args.dataset}")
    print(f"Languages:  {', '.join(languages)}")
    print(f"Patterns:   {patterns}")
    print(f"Output dir: {output_dir}")
    print()
    print("Starting download (resumes automatically if interrupted)...")

    t0 = time.time()
    local_path = snapshot_download(
        args.dataset,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(output_dir),
    )
    elapsed = time.time() - t0

    parquet_files = sorted(Path(local_path).rglob("*.parquet"))
    total_bytes = sum(f.stat().st_size for f in parquet_files)

    lang_files = {}
    for f in parquet_files:
        lang = f.parent.name
        lang_files[lang] = lang_files.get(lang, 0) + 1

    print(f"\nDownload complete in {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    print(f"  Files: {len(parquet_files)}")
    print(f"  Total size: {total_bytes / 1e9:.1f} GB")
    print(f"  Location: {local_path}")
    print(f"\n  Per language:")
    for lang, count in sorted(lang_files.items()):
        print(f"    {lang}: {count} files")
    print()
    print("Next step: run pipeline_classify.py with --local-dir pointing here:")
    print(f"  python pipeline_classify.py --local-dir {local_path} --compile")


if __name__ == "__main__":
    main()
