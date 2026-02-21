#!/usr/bin/env python3
"""
Download a balanced sample from bigcode/starcoderdata.

Streams from Hugging Face language-by-language, saving N rows per language
as separate parquet files.  Requires accepting the dataset's terms of use
at https://huggingface.co/datasets/bigcode/starcoderdata and authenticating
via ``huggingface-cli login`` or the HF_TOKEN environment variable.

Usage:
  python download_starcoderdata.py
  python download_starcoderdata.py -n 10000 --languages python javascript go
  python download_starcoderdata.py -o training_data/starcoderdata/ --max-chars 100000
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

LANGUAGES = [
    "python", "javascript", "typescript", "java",
    "go", "rust", "sql", "shell",
]

DEFAULT_PER_LANGUAGE = 25_000
DEFAULT_MAX_CHARS = 50_000
DEFAULT_OUTPUT_DIR = "training_data/starcoderdata"
DEFAULT_SHUFFLE_BUFFER = 50_000
DEFAULT_SEED = 42


def download_language(
    language: str,
    n: int,
    max_chars: int,
    output_dir: str,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
    seed: int = DEFAULT_SEED,
) -> int:
    """Stream, shuffle, and save *n* rows for a single language."""
    from datasets import load_dataset

    output_path = Path(output_dir) / f"{language}.parquet"

    if output_path.exists():
        existing = len(pd.read_parquet(output_path, columns=["id"]))
        print(f"  {language}: already exists ({existing:,} rows), skipping")
        return existing

    print(f"  {language}: streaming {n:,} rows (shuffle buffer={shuffle_buffer:,}, seed={seed})...")
    ds = load_dataset(
        "bigcode/starcoderdata",
        data_dir=language,
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    rows = []
    pbar = tqdm(total=n, desc=f"    {language}", unit="row")

    for i, example in enumerate(ds):
        if i >= n:
            break
        content = example.get("content", "")
        original_length = len(content)
        if original_length > max_chars:
            content = content[:max_chars]
        rows.append({
            "id": f"{language}_{i:06d}",
            "content": content,
            "language": language,
            "content_length": original_length,
        })
        pbar.update(1)

    pbar.close()

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"    saved {len(df):,} rows to {output_path}")
    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Download a balanced sample from bigcode/starcoderdata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 25K rows per language (200K total, default)
  python download_starcoderdata.py

  # Download 10K rows for specific languages
  python download_starcoderdata.py -n 10000 --languages python javascript go
""",
    )
    parser.add_argument(
        "-n", "--per-language", type=int, default=DEFAULT_PER_LANGUAGE,
        help=f"Rows per language (default: {DEFAULT_PER_LANGUAGE:,})",
    )
    parser.add_argument(
        "-o", "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--languages", nargs="+", default=LANGUAGES,
        help=f"Languages to download (default: {' '.join(LANGUAGES)})",
    )
    parser.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS,
        help=f"Max chars to store per file (default: {DEFAULT_MAX_CHARS:,})",
    )
    parser.add_argument(
        "--shuffle-buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER,
        help=f"Shuffle buffer size for randomizing stream order (default: {DEFAULT_SHUFFLE_BUFFER:,})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for shuffle (default: {DEFAULT_SEED})",
    )

    args = parser.parse_args()

    print(f"Dataset:   bigcode/starcoderdata")
    print(f"Per lang:  {args.per_language:,} rows")
    print(f"Languages: {', '.join(args.languages)}")
    print(f"Output:    {args.output_dir}")
    print(f"Max chars: {args.max_chars:,}")
    print(f"Shuffle:   buffer={args.shuffle_buffer:,}, seed={args.seed}")
    print()

    total = 0
    for lang in args.languages:
        count = download_language(
            lang, args.per_language, args.max_chars, args.output_dir,
            shuffle_buffer=args.shuffle_buffer, seed=args.seed,
        )
        total += count

    print(f"\nDone! {total:,} total rows across {len(args.languages)} languages")


if __name__ == "__main__":
    main()
