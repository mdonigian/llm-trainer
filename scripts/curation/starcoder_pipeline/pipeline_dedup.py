#!/usr/bin/env python3
"""
Stage 3: Deduplicate filtered code files using MinHash LSH.

Uses Rensa (Rust-backed MinHash + LSH) for high-throughput hashing and
indexing (~50x faster than datasketch).

Line-level 5-gram shingles capture structural similarity in code better
than word-level shingles. 128 MinHash permutations, 0.7 Jaccard threshold.
Keeps the code file with the highest relevance_score from each cluster.

Usage:
  python pipeline_dedup.py --input-dir filtered_shards/ --output-dir deduped_shards/
  python pipeline_dedup.py --input-dir filtered_shards/ --output-dir deduped_shards/ \
      --threshold 0.8 --num-perm 256 --workers 16
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rensa import RMinHash, RMinHashLSH
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_NUM_PERM = 128
DEFAULT_THRESHOLD = 0.7
DEFAULT_SHINGLE_SIZE = 5
DEFAULT_NUM_BANDS = 16
DEFAULT_WORKERS = min(os.cpu_count() or 4, 32)


# ---------------------------------------------------------------------------
# MinHash computation — line-based shingling for code
# ---------------------------------------------------------------------------

def _compute_minhash(text: str, num_perm: int, shingle_size: int) -> RMinHash:
    """Compute RMinHash using line-level n-gram shingles.

    Lines are stripped and empty lines are removed before shingling.
    Line-level shingles capture structural similarity in code better
    than word-level shingles.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    m = RMinHash(num_perm=num_perm, seed=42)
    if len(lines) < shingle_size:
        m.update(["\n".join(lines)])
    else:
        shingles = [
            "\n".join(lines[i : i + shingle_size])
            for i in range(len(lines) - shingle_size + 1)
        ]
        m.update(shingles)
    return m


def _compute_batch(texts_with_indices, num_perm, shingle_size):
    """Process a batch of (index, text) pairs. Used by ProcessPoolExecutor.

    RMinHash implements pickle (__getstate__/__setstate__), so objects
    can be returned across process boundaries.
    """
    results = []
    for idx, text in texts_with_indices:
        try:
            mh = _compute_minhash(str(text), num_perm, shingle_size)
            results.append((idx, mh))
        except Exception:
            results.append((idx, None))
    return results


# ---------------------------------------------------------------------------
# Pass 1: Compute signatures
# ---------------------------------------------------------------------------

def compute_all_signatures(input_dir: Path, num_perm: int, shingle_size: int,
                           num_workers: int):
    """Compute MinHash signatures for all documents using multiprocessing.

    Returns:
        minhashes: list of RMinHash objects (length N)
        relevance_scores: (N,) float32 array
        doc_locations: list of (shard_path, row_idx) tuples
        total_docs: int
    """
    shard_files = sorted(input_dir.glob("filtered_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No filtered shard files in {input_dir}")

    print(f"Found {len(shard_files)} filtered shards")

    total_docs = 0
    for sf in shard_files:
        total_docs += pq.read_metadata(sf).num_rows

    print(f"Total code files: {total_docs:,}")
    print(f"Using {num_workers} workers for MinHash computation")

    minhashes: list[RMinHash | None] = [None] * total_docs
    relevance_scores = np.empty(total_docs, dtype=np.float32)
    doc_locations = []

    offset = 0
    compute_fn = partial(_compute_batch, num_perm=num_perm, shingle_size=shingle_size)
    mp_chunk_size = 1000

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for shard_path in tqdm(shard_files, desc="Computing MinHash signatures"):
            table = pq.read_table(shard_path, columns=["content", "relevance_score"])
            shard_row = 0
            for rb in table.to_batches(max_chunksize=20_000):
                texts = rb.column("content").to_pylist()
                scores = rb.column("relevance_score").fill_null(0).to_numpy(
                    zero_copy_only=False).astype(np.float32, copy=False)
                n = len(texts)
                if n == 0:
                    continue

                doc_locations.extend((shard_path, shard_row + i) for i in range(n))
                relevance_scores[offset : offset + n] = scores

                futures = []
                for chunk_start in range(0, n, mp_chunk_size):
                    chunk = [
                        (offset + i, texts[i])
                        for i in range(chunk_start, min(chunk_start + mp_chunk_size, n))
                    ]
                    futures.append(pool.submit(compute_fn, chunk))

                for future in futures:
                    for idx, mh in future.result():
                        minhashes[idx] = mh

                offset += n
                shard_row += n
            del table

    for i in range(total_docs):
        if minhashes[i] is None:
            minhashes[i] = RMinHash(num_perm=num_perm, seed=42)

    print(f"Computed {total_docs:,} MinHash signatures")
    return minhashes, relevance_scores, doc_locations, total_docs


# ---------------------------------------------------------------------------
# Build LSH index and resolve clusters
# ---------------------------------------------------------------------------

def build_lsh_and_resolve(minhashes, relevance_scores, total_docs,
                          num_perm, threshold, num_bands):
    """Build Rensa LSH index and resolve duplicate clusters.

    Inserts all MinHash objects, queries each to find candidates, then
    resolves clusters keeping the highest-relevance document from each.
    """
    print(f"\nBuilding LSH index (threshold={threshold}, num_perm={num_perm}, "
          f"num_bands={num_bands})...")
    lsh = RMinHashLSH(threshold=threshold, num_perm=num_perm, num_bands=num_bands)

    for i in tqdm(range(total_docs), desc="LSH insertion", unit="doc"):
        lsh.insert(i, minhashes[i])

    print("Resolving duplicate clusters...")
    visited = set()
    keep_set = set(range(total_docs))
    cluster_map = {}
    cluster_id = 0
    total_removed = 0

    for doc_id in tqdm(range(total_docs), desc="Querying LSH", unit="doc"):
        if doc_id in visited:
            continue

        candidates = lsh.query(minhashes[doc_id])
        candidate_ids = [c for c in candidates if c != doc_id and c not in visited]
        if not candidate_ids:
            continue

        cluster_members = [doc_id] + candidate_ids
        best_id = max(cluster_members, key=lambda d: relevance_scores[d])

        for mid in cluster_members:
            cluster_map[mid] = cluster_id
            visited.add(mid)
            if mid != best_id:
                keep_set.discard(mid)
                total_removed += 1

        cluster_id += 1

    del lsh

    stats = {
        "total_docs": total_docs,
        "duplicates_removed": total_removed,
        "clusters_found": cluster_id,
        "docs_remaining": len(keep_set),
        "dedup_rate_pct": round(total_removed / total_docs * 100, 2) if total_docs > 0 else 0,
    }

    print(f"\nDeduplication results:")
    print(f"  Clusters found: {cluster_id:,}")
    print(f"  Duplicates removed: {total_removed:,}")
    print(f"  Code files remaining: {len(keep_set):,} ({100 - stats['dedup_rate_pct']:.1f}%)")

    return keep_set, cluster_map, stats


# ---------------------------------------------------------------------------
# Pass 2: Emit deduped shards
# ---------------------------------------------------------------------------

def emit_deduped(input_dir: Path, output_dir: Path, keep_set: set,
                 cluster_map: dict, doc_locations: list):
    """Re-read filtered shards and write only kept documents."""
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(input_dir.glob("filtered_*.parquet"))

    shard_keep: dict[Path, list[int]] = {}
    shard_clusters: dict[Path, dict[int, int]] = {}

    for doc_id in keep_set:
        shard_path, row_idx = doc_locations[doc_id]
        if shard_path not in shard_keep:
            shard_keep[shard_path] = []
            shard_clusters[shard_path] = {}
        shard_keep[shard_path].append(row_idx)
        if doc_id in cluster_map:
            shard_clusters[shard_path][row_idx] = cluster_map[doc_id]

    total_written = 0
    out_shard_idx = 0
    rows_per_shard = 1_000_000
    writer = None
    current_rows = 0
    output_schema = None

    for shard_path in tqdm(shard_files, desc="Writing deduped shards"):
        if shard_path not in shard_keep:
            continue

        keep_rows = sorted(shard_keep[shard_path])
        if not keep_rows:
            continue

        table = pq.read_table(shard_path)
        for ci, field in enumerate(table.schema):
            if pa.types.is_string(field.type) or pa.types.is_binary(field.type):
                table = table.set_column(
                    ci, field.name, table.column(field.name).cast(pa.large_string())
                )
        emit_table = table.take(keep_rows)
        del table

        cluster_ids = [shard_clusters.get(shard_path, {}).get(r, -1) for r in keep_rows]
        emit_table = emit_table.append_column(
            "dedup_cluster_id", pa.array(cluster_ids, type=pa.int64())
        )

        if output_schema is None:
            output_schema = emit_table.schema

        n_emit = emit_table.num_rows
        row_start = 0

        while row_start < n_emit:
            if writer is None:
                out_path = output_dir / f"deduped_{out_shard_idx:04d}.parquet"
                writer = pq.ParquetWriter(out_path, output_schema)
                current_rows = 0

            space = rows_per_shard - current_rows
            chunk_end = min(row_start + space, n_emit)
            writer.write_table(emit_table.slice(row_start, chunk_end - row_start))
            current_rows += chunk_end - row_start
            total_written += chunk_end - row_start
            row_start = chunk_end

            if current_rows >= rows_per_shard:
                writer.close()
                writer = None
                out_shard_idx += 1

        del emit_table

    if writer is not None:
        writer.close()

    print(f"  Written {total_written:,} code files across {out_shard_idx + 1} shards")
    return total_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.num_perm % args.num_bands != 0:
        raise ValueError(
            f"num_perm ({args.num_perm}) must be evenly divisible by "
            f"num_bands ({args.num_bands})"
        )

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Parameters: num_perm={args.num_perm}, threshold={args.threshold}, "
          f"num_bands={args.num_bands}, "
          f"shingle_size={args.shingle_size} (line-level), workers={args.workers}")

    t0 = time.time()

    print("\n--- Pass 1: Computing MinHash signatures (line-level shingles) ---")
    minhashes, relevance_scores, doc_locations, total_docs = compute_all_signatures(
        input_dir, args.num_perm, args.shingle_size, args.workers,
    )

    print("\n--- Building LSH index and resolving clusters ---")
    keep_set, cluster_map, stats = build_lsh_and_resolve(
        minhashes, relevance_scores, total_docs,
        args.num_perm, args.threshold, args.num_bands,
    )

    del minhashes, relevance_scores

    print("\n--- Pass 2: Writing deduped shards ---")
    total_written = emit_deduped(input_dir, output_dir, keep_set, cluster_map, doc_locations)

    stats["output_docs"] = total_written
    stats["parameters"] = {
        "num_perm": args.num_perm,
        "threshold": args.threshold,
        "num_bands": args.num_bands,
        "shingle_size": args.shingle_size,
        "shingle_type": "line-level",
        "workers": args.workers,
    }
    with open(output_dir / "dedup_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDeduplication complete in {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: MinHash LSH deduplication of filtered code files (StarCoder)",
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory of filtered parquet shards from Stage 2")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for deduped shards")
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM,
                        help=f"MinHash permutations (default: {DEFAULT_NUM_PERM})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Jaccard similarity threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--num-bands", type=int, default=DEFAULT_NUM_BANDS,
                        help=f"LSH bands; must divide num_perm evenly (default: {DEFAULT_NUM_BANDS})")
    parser.add_argument("--shingle-size", type=int, default=DEFAULT_SHINGLE_SIZE,
                        help=f"Line-level n-gram shingle size (default: {DEFAULT_SHINGLE_SIZE})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers for MinHash (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
