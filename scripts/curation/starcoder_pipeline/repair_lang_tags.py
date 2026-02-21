#!/usr/bin/env python3
"""
Repair 'unknown' lang tags in scored shards by matching content hashes
against raw parquet files where the language is known from the directory name.

This avoids re-running the classifier — scores are correct, only lang is wrong.

Usage:
  python repair_lang_tags.py \
      --scored-dir /workspace/starcoder-curation/scored_shards \
      --raw-dir /workspace/starcoder-curation/raw_data \
      --languages typescript python rust go java jupyter-scripts-dedup-filtered
"""

import argparse
import hashlib
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def content_hash(text: str) -> str:
    """Fast hash of first 512 + last 512 chars to avoid hashing huge files."""
    key = text[:512] + "|" + text[-512:] + "|" + str(len(text))
    return hashlib.md5(key.encode("utf-8", errors="replace")).hexdigest()


def build_hash_to_lang(raw_dir: Path, languages: list[str]) -> dict[str, str]:
    """Scan raw parquets and build content_hash -> language mapping."""
    mapping: dict[str, str] = {}

    for lang in languages:
        lang_dir = raw_dir / lang
        if not lang_dir.is_dir():
            print(f"  {lang}: directory not found, skipping")
            continue

        files = sorted(lang_dir.glob("*.parquet"))
        print(f"  {lang}: {len(files)} files")

        for fpath in tqdm(files, desc=f"Hashing {lang}", leave=False):
            try:
                table = pq.read_table(fpath, columns=["content"])
            except Exception:
                continue

            for text in table.column("content").to_pylist():
                if text:
                    h = content_hash(text)
                    mapping[h] = lang

            del table

    print(f"  Built mapping: {len(mapping):,} unique content hashes")
    return mapping


def repair_shards(scored_dir: Path, mapping: dict[str, str]):
    """Scan scored shards and replace 'unknown' lang values using the mapping."""
    shard_files = sorted(scored_dir.glob("shard_*.parquet"))
    print(f"\nRepairing {len(shard_files)} scored shards...")

    total_fixed = 0
    total_unknown = 0
    total_still_unknown = 0
    lang_counts: dict[str, int] = {}

    for shard_path in tqdm(shard_files, desc="Repairing shards"):
        table = pq.read_table(shard_path)
        langs = table.column("lang").to_pylist()
        contents = table.column("content").to_pylist()

        unknown_count = sum(1 for l in langs if l == "unknown")
        if unknown_count == 0:
            del table
            continue

        total_unknown += unknown_count
        new_langs = list(langs)
        fixed_in_shard = 0

        for i, (lang, text) in enumerate(zip(langs, contents)):
            if lang == "unknown" and text:
                h = content_hash(text)
                resolved = mapping.get(h)
                if resolved:
                    new_langs[i] = resolved
                    fixed_in_shard += 1
                    lang_counts[resolved] = lang_counts.get(resolved, 0) + 1
                else:
                    total_still_unknown += 1

        if fixed_in_shard > 0:
            col_idx = table.schema.get_field_index("lang")
            table = table.set_column(col_idx, "lang", pa.array(new_langs, type=pa.string()))
            pq.write_table(table, shard_path)
            total_fixed += fixed_in_shard

        del table

    print(f"\nRepair complete:")
    print(f"  Total 'unknown' entries: {total_unknown:,}")
    print(f"  Fixed: {total_fixed:,}")
    print(f"  Still unknown: {total_still_unknown:,}")
    print(f"\n  Per language:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="Repair lang tags in scored shards")
    parser.add_argument("--scored-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--languages", nargs="*",
                        default=["typescript", "python", "rust", "go", "java",
                                 "jupyter-scripts-dedup-filtered"])
    args = parser.parse_args()

    scored_dir = Path(args.scored_dir)
    raw_dir = Path(args.raw_dir)

    print("Building content hash -> language mapping from raw parquets...")
    t0 = time.time()
    mapping = build_hash_to_lang(raw_dir, args.languages)
    print(f"  Hashing took {time.time() - t0:.0f}s")

    repair_shards(scored_dir, mapping)


if __name__ == "__main__":
    main()
