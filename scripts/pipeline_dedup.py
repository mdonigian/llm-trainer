#!/usr/bin/env python3
"""
Stage 3: Deduplicate filtered documents using MinHash LSH.

Uses 13-gram word shingles, 128 MinHash permutations, and 0.7 Jaccard
threshold to identify near-duplicate clusters. Keeps the document with
the highest relevance_score from each cluster.

Memory-efficient design for 117GB RAM with 50M+ documents:
  - Stores hash signatures as compact numpy arrays (50M × 128 × 4B = 25GB)
    instead of Python MinHash objects (~75GB)
  - Uses multiprocessing for MinHash computation across all CPU cores
  - Processes LSH insertion in chunks to control peak memory

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
from datasketch import MinHash, MinHashLSH
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_NUM_PERM = 128
DEFAULT_THRESHOLD = 0.7
DEFAULT_SHINGLE_SIZE = 13
DEFAULT_WORKERS = min(os.cpu_count() or 4, 32)
LSH_INSERT_CHUNK = 500_000


# ---------------------------------------------------------------------------
# MinHash computation — designed for multiprocessing
# ---------------------------------------------------------------------------

def _compute_minhash_signature(text: str, num_perm: int, shingle_size: int) -> np.ndarray:
    """Compute MinHash and return the raw hashvalues as a uint32 array.

    Returns a (num_perm,) uint32 array instead of a MinHash object to avoid
    the ~1.5KB Python object overhead per document. At 50M docs this saves
    ~75GB of RAM.
    """
    words = text.lower().split()
    m = MinHash(num_perm=num_perm)
    if len(words) < shingle_size:
        shingle = " ".join(words)
        m.update(shingle.encode("utf-8"))
    else:
        for i in range(len(words) - shingle_size + 1):
            shingle = " ".join(words[i : i + shingle_size])
            m.update(shingle.encode("utf-8"))
    return m.hashvalues.astype(np.uint32)


def _compute_batch(texts_with_indices, num_perm, shingle_size):
    """Process a batch of (index, text) pairs. Used by ProcessPoolExecutor."""
    results = []
    for idx, text in texts_with_indices:
        try:
            sig = _compute_minhash_signature(str(text), num_perm, shingle_size)
            results.append((idx, sig))
        except Exception:
            results.append((idx, None))
    return results


# ---------------------------------------------------------------------------
# Pass 1: Compute signatures + build LSH index
# ---------------------------------------------------------------------------

def compute_all_signatures(input_dir: Path, num_perm: int, shingle_size: int,
                           num_workers: int):
    """Compute MinHash signatures for all documents using multiprocessing.

    Returns:
        signatures: (N, num_perm) uint32 numpy array
        relevance_scores: (N,) float32 array
        doc_locations: list of (shard_path, row_idx) tuples
        total_docs: int
    """
    shard_files = sorted(input_dir.glob("filtered_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No filtered shard files in {input_dir}")

    print(f"Found {len(shard_files)} filtered shards")

    # First pass: count total docs and collect texts
    total_docs = 0
    for sf in shard_files:
        total_docs += pq.read_metadata(sf).num_rows

    print(f"Total documents: {total_docs:,}")
    print(f"Signature storage: {total_docs * num_perm * 4 / 1e9:.1f} GB")
    print(f"Using {num_workers} workers for MinHash computation")

    signatures = np.empty((total_docs, num_perm), dtype=np.uint32)
    relevance_scores = np.empty(total_docs, dtype=np.float32)
    doc_locations = []  # (shard_path, row_idx)

    offset = 0
    compute_fn = partial(_compute_batch, num_perm=num_perm, shingle_size=shingle_size)
    mp_chunk_size = 1000  # docs per task submitted to pool

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for shard_path in tqdm(shard_files, desc="Computing MinHash signatures"):
            table = pq.read_table(shard_path, columns=["text", "relevance_score"])
            texts = table.column("text").to_pylist()
            scores = table.column("relevance_score").to_pylist()
            n = len(texts)
            del table

            for i in range(n):
                doc_locations.append((shard_path, i))
                relevance_scores[offset + i] = float(scores[i]) if scores[i] is not None else 0.0

            # Submit work in chunks to the process pool
            indexed_texts = [(offset + i, texts[i]) for i in range(n)]
            futures = []
            for chunk_start in range(0, n, mp_chunk_size):
                chunk = indexed_texts[chunk_start : chunk_start + mp_chunk_size]
                futures.append(pool.submit(compute_fn, chunk))

            for future in futures:
                for idx, sig in future.result():
                    if sig is not None:
                        signatures[idx] = sig
                    else:
                        signatures[idx] = 0

            offset += n

    print(f"Computed {total_docs:,} signatures "
          f"({signatures.nbytes / 1e9:.1f} GB in memory)")
    return signatures, relevance_scores, doc_locations, total_docs


def build_lsh_and_resolve(signatures, relevance_scores, total_docs,
                          num_perm, threshold):
    """Build LSH index from compact signatures and resolve clusters.

    Inserts signatures in chunks to control peak memory, then queries
    to find duplicate clusters.
    """
    print(f"\nBuilding LSH index (threshold={threshold}, num_perm={num_perm})...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Insert in chunks — creating temporary MinHash objects only during insertion
    for start in tqdm(range(0, total_docs, LSH_INSERT_CHUNK), desc="LSH insertion"):
        end = min(start + LSH_INSERT_CHUNK, total_docs)
        for i in range(start, end):
            m = MinHash(num_perm=num_perm, hashvalues=signatures[i].astype(np.uint64))
            try:
                lsh.insert(str(i), m)
            except ValueError:
                pass  # identical hash already inserted

    # Resolve clusters
    print("Resolving duplicate clusters...")
    visited = set()
    keep_set = set(range(total_docs))
    cluster_map = {}
    cluster_id = 0
    total_removed = 0

    for doc_id in tqdm(range(total_docs), desc="Querying LSH"):
        if doc_id in visited:
            continue

        m = MinHash(num_perm=num_perm, hashvalues=signatures[doc_id].astype(np.uint64))
        try:
            candidates = lsh.query(m)
        except Exception:
            continue

        candidate_ids = [int(c) for c in candidates if int(c) != doc_id and int(c) not in visited]
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

    # Free the LSH index
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
    print(f"  Documents remaining: {len(keep_set):,} ({100 - stats['dedup_rate_pct']:.1f}%)")

    return keep_set, cluster_map, stats


# ---------------------------------------------------------------------------
# Pass 2: Emit deduped shards
# ---------------------------------------------------------------------------

def emit_deduped(input_dir: Path, output_dir: Path, keep_set: set,
                 cluster_map: dict, doc_locations: list):
    """Re-read filtered shards and write only kept documents."""
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(input_dir.glob("filtered_*.parquet"))

    # Build per-shard keep lists
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

    print(f"  Written {total_written:,} documents across {out_shard_idx + 1} shards")
    return total_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Parameters: num_perm={args.num_perm}, threshold={args.threshold}, "
          f"shingle_size={args.shingle_size}, workers={args.workers}")

    t0 = time.time()

    # Pass 1: Compute signatures with multiprocessing
    print("\n--- Pass 1: Computing MinHash signatures ---")
    signatures, relevance_scores, doc_locations, total_docs = compute_all_signatures(
        input_dir, args.num_perm, args.threshold, args.workers,
    )

    # Build LSH and resolve clusters
    print("\n--- Building LSH index and resolving clusters ---")
    keep_set, cluster_map, stats = build_lsh_and_resolve(
        signatures, relevance_scores, total_docs, args.num_perm, args.threshold,
    )

    # Free signature memory before pass 2
    del signatures, relevance_scores

    # Pass 2: Emit
    print("\n--- Pass 2: Writing deduped shards ---")
    total_written = emit_deduped(input_dir, output_dir, keep_set, cluster_map, doc_locations)

    stats["output_docs"] = total_written
    stats["parameters"] = {
        "num_perm": args.num_perm,
        "threshold": args.threshold,
        "shingle_size": args.shingle_size,
        "workers": args.workers,
    }
    with open(output_dir / "dedup_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDeduplication complete in {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: MinHash LSH deduplication of filtered documents",
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory of filtered parquet shards from Stage 2")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for deduped shards")
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM,
                        help=f"MinHash permutations (default: {DEFAULT_NUM_PERM})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Jaccard similarity threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--shingle-size", type=int, default=DEFAULT_SHINGLE_SIZE,
                        help=f"Word n-gram shingle size (default: {DEFAULT_SHINGLE_SIZE})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers for MinHash (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
