#!/usr/bin/env python3
"""
FineWeb-Edu Reasoning Complexity Classification

Classifies articles' reasoning complexity on a 1-5 scale using OpenAI's
Batch API.  Designed for very large datasets (tens of GB of parquet files)
that cannot fit in RAM.

Complexity Levels:
  1 — Factual/Declarative: States facts with no reasoning.
  2 — Single-step reasoning: One inference or comparison.
  3 — Multi-step reasoning: 2-4 chained logical steps.
  4 — Complex reasoning: 5+ steps, conditionals, multiple competing factors.
  5 — Formal/Abstract reasoning: Proofs, formal logic, axiomatic reasoning.

The submit step uses reservoir sampling to stream through arbitrarily large
parquet datasets one row-group at a time, never loading more than one
row-group (~1 000 rows) plus the reservoir (N sampled rows' id + text) into
memory.

Usage:
  python complexity_fineweb_edu.py submit input/fineweb-edu/ -n 10000
  python complexity_fineweb_edu.py status <batch_id_or_manifest>
  python complexity_fineweb_edu.py wait <batch_id_or_manifest>
  python complexity_fineweb_edu.py download <batch_id_or_manifest> -o results.parquet
  python complexity_fineweb_edu.py merge results.parquet input/fineweb-edu/ -o merged.parquet
"""

import argparse
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("Error: OPENAI_API_KEY not found. Set it in .env or as an environment variable.")

sync_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_MAX_CHARS = 8000
DEFAULT_POLL_INTERVAL = 30
DEFAULT_MAX_FILE_SIZE_MB = 150

DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ReasoningComplexity(BaseModel):
    """Structured output schema for reasoning complexity classification."""
    reasoning_complexity: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "The reasoning complexity level of the text. "
            "1 = Factual/Declarative (states facts, no reasoning). "
            "2 = Single-step reasoning (one inference or comparison). "
            "3 = Multi-step reasoning (2-4 chained logical steps). "
            "4 = Complex reasoning (5+ steps, conditionals, competing factors). "
            "5 = Formal/Abstract reasoning (proofs, formal logic, axiomatic reasoning)."
        ),
    )


SYSTEM_PROMPT = """You are an expert reasoning complexity classifier. Analyze the given text and determine its reasoning complexity level on a 1-5 scale.

Level 1 - Factual/Declarative: States facts with no reasoning.
Example: "The Pacific Ocean is the largest ocean on Earth, covering approximately 165.25 million square kilometers."

Level 2 - Single-step reasoning: One inference or comparison.
Example: "Because the boiling point decreases at higher altitudes, water boils faster in Denver than in Miami."

Level 3 - Multi-step reasoning: 2-4 chained logical steps.
Example: "Since demand increased while supply remained fixed, prices rose. Higher prices reduced consumer spending, which in turn slowed GDP growth in the following quarter."

Level 4 - Complex reasoning: 5+ steps, conditionals, multiple competing factors.
Example: "If the patient presents with symptom A but not B, and lab values show C above threshold, then condition X is likely unless the patient has history of D, in which case condition Y must be ruled out first by testing E."

Level 5 - Formal/Abstract reasoning: Proofs, formal logic, axiomatic reasoning.
Example: "Let f be a continuous function on [a,b]. By the intermediate value theorem, if f(a) < 0 and f(b) > 0, there exists c in (a,b) such that f(c) = 0. We proceed by contradiction..."

Classify the OVERALL reasoning complexity of the text — use the highest level of reasoning that is substantively present (not just mentioned in passing)."""

_SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_PROMPT}

# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def discover_parquet_files(input_path: str) -> List[Path]:
    """Return a sorted list of parquet files from a path (file or directory).

    Recursively searches subdirectories so nested layouts like
    ``sample/10BT/*.parquet`` are found automatically.
    """
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() != ".parquet":
            sys.exit(f"Expected a parquet file, got: {p.suffix}")
        return [p]
    elif p.is_dir():
        files = sorted(p.rglob("*.parquet"))
        if not files:
            sys.exit(f"No parquet files found under: {p}")
        return files
    else:
        sys.exit(f"Input path does not exist: {p}")


# ---------------------------------------------------------------------------
# Streaming reservoir sampling
# ---------------------------------------------------------------------------


def _count_total_rows(parquet_files: List[Path]) -> int:
    """Fast row count by reading only parquet metadata (no data loaded)."""
    total = 0
    for fp in parquet_files:
        meta = pq.read_metadata(fp)
        total += meta.num_rows
    return total


def reservoir_sample(
    parquet_files: List[Path],
    n: int,
    id_column: str,
    text_column: str,
    max_chars: int,
    seed: int,
) -> List[Tuple[str, str]]:
    """Reservoir-sample n rows from arbitrarily large parquet files.

    Streams one row-group at a time, reading only ``id_column`` and
    ``text_column``.  Peak memory ≈ one row-group + the reservoir of n
    (id, text) tuples.

    Returns a list of (doc_id, text) pairs.
    """
    rng = random.Random(seed)
    reservoir: List[Tuple[str, str]] = []  # (id, text)
    seen = 0

    total_rows = _count_total_rows(parquet_files)
    pbar = tqdm(total=total_rows, desc="Sampling", unit="row")

    for fp in parquet_files:
        pf = pq.ParquetFile(fp)
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=[id_column, text_column])
            ids = table.column(id_column)
            texts = table.column(text_column)
            rg_len = len(table)

            for i in range(rg_len):
                doc_id = str(ids[i].as_py())
                text = str(texts[i].as_py())
                if len(text) > max_chars:
                    text = text[:max_chars] + "..."

                if seen < n:
                    reservoir.append((doc_id, text))
                else:
                    j = rng.randint(0, seen)
                    if j < n:
                        reservoir[j] = (doc_id, text)
                seen += 1

            pbar.update(rg_len)
            del table, ids, texts

    pbar.close()
    return reservoir


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


def _make_request_line(doc_id: str, text: str, model: str, schema: dict) -> str:
    """Build a single JSONL request line for the Batch API."""
    request = {
        "custom_id": doc_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                _SYSTEM_MESSAGE,
                {"role": "user", "content": f"Analyze this text and classify its reasoning complexity:\n\n{text}"},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ReasoningComplexity",
                    "strict": True,
                    "schema": schema,
                },
            },
        },
    }
    return json.dumps(request) + "\n"


def _save_manifest(manifest_path: str, batch_ids: List[str], metadata: dict | None = None) -> None:
    """Save batch IDs and optional metadata to a JSON manifest file."""
    data: Dict[str, Any] = {"batch_ids": batch_ids}
    if metadata:
        data["metadata"] = metadata
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


def _load_manifest(manifest_path: str) -> List[str]:
    """Load batch IDs from a manifest file, or treat input as a single batch ID."""
    p = Path(manifest_path)
    if p.suffix == ".json" and p.exists():
        with open(p) as f:
            data = json.load(f)
        return data["batch_ids"]
    else:
        return [manifest_path]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_submit(args: argparse.Namespace) -> None:
    """Stream-sample rows via reservoir sampling and submit batch job(s).

    Peak memory ≈ one parquet row-group + N sampled (id, text) pairs.
    """
    input_path = args.input
    n_rows = args.num_rows
    text_column = args.text_column
    id_column = args.id_column
    max_chars = args.max_chars
    model = args.model
    seed = args.seed
    max_file_bytes = args.max_file_size_mb * 1024 * 1024
    jsonl_base = Path(args.jsonl or "complexity_batch_input.jsonl")
    manifest_path = args.manifest or str(jsonl_base.with_suffix(".manifest.json"))

    # Discover parquet files (recursive)
    parquet_files = discover_parquet_files(input_path)
    total_rows = _count_total_rows(parquet_files)
    print(f"Found {len(parquet_files)} parquet file(s), {total_rows:,} total rows")
    print(f"Reservoir sampling {n_rows:,} rows (seed={seed})...\n")

    # Reservoir sample — streams row-group by row-group
    sampled = reservoir_sample(parquet_files, n_rows, id_column, text_column, max_chars, seed)

    actual_n = len(sampled)

    # Confirmation prompt
    print(f"\n{'='*60}")
    print(f"Sampled {actual_n:,} rows from {total_rows:,} total")
    print(f"Model:  {model}")
    print(f"{'='*60}")
    print()
    confirm = input("Proceed with batch submission? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Aborted.")
        sys.exit(0)

    # Build the JSON schema for structured output
    schema = ReasoningComplexity.model_json_schema()
    schema["additionalProperties"] = False

    # Write JSONL, splitting into chunks when file size limit is reached
    print(f"\nPreparing {actual_n} requests...")

    chunk_paths: List[str] = []
    chunk_idx = 0
    current_size = 0
    current_file = None

    def _open_new_chunk():
        nonlocal chunk_idx, current_size, current_file
        if current_file:
            current_file.close()
        if len(chunk_paths) == 0 and chunk_idx == 0:
            path = str(jsonl_base)
        else:
            stem = jsonl_base.stem
            path = str(jsonl_base.with_name(f"{stem}_{chunk_idx}{jsonl_base.suffix}"))
        chunk_paths.append(path)
        current_file = open(path, "w")
        current_size = 0
        chunk_idx += 1

    _open_new_chunk()

    for doc_id, text in tqdm(sampled, desc="Writing JSONL"):
        line = _make_request_line(doc_id, text, model, schema)
        line_bytes = len(line.encode("utf-8"))

        if current_size + line_bytes > max_file_bytes and current_size > 0:
            _open_new_chunk()

        current_file.write(line)
        current_size += line_bytes

    if current_file:
        current_file.close()

    # Free the reservoir now that JSONL is written
    del sampled

    print(f"Wrote {actual_n} requests across {len(chunk_paths)} file(s)")

    # Upload and submit each chunk
    batch_ids: List[str] = []
    for i, chunk_path in enumerate(chunk_paths):
        label = f"[{i+1}/{len(chunk_paths)}] " if len(chunk_paths) > 1 else ""
        print(f"\n{label}Uploading {chunk_path}...")
        with open(chunk_path, "rb") as f:
            file_obj = sync_client.files.create(file=f, purpose="batch")
        print(f"{label}Uploaded file: {file_obj.id}")

        batch = sync_client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"fineweb-edu complexity classification (part {i+1}/{len(chunk_paths)})"},
        )
        batch_ids.append(batch.id)
        print(f"{label}Batch submitted: {batch.id} (status: {batch.status})")

    # Save manifest with metadata
    manifest_metadata = {
        "total_rows_submitted": actual_n,
        "total_source_rows": total_rows,
        "model": model,
        "seed": seed,
        "input_path": str(input_path),
    }
    _save_manifest(manifest_path, batch_ids, metadata=manifest_metadata)

    print(f"\n{'='*60}")
    print(f"Submitted {len(batch_ids)} batch job(s)")
    print(f"Manifest: {manifest_path}")
    print(f"\nCheck status:     python complexity_fineweb_edu.py status {manifest_path}")
    print(f"Wait for all:     python complexity_fineweb_edu.py wait {manifest_path}")
    print(f"Download results: python complexity_fineweb_edu.py download {manifest_path} -o results.parquet")


def cmd_status(args: argparse.Namespace) -> None:
    """Check the status of one or more Batch API jobs."""
    batch_ids = _load_manifest(args.batch_id)
    total_completed = 0
    total_requests = 0
    total_failed = 0

    for bid in batch_ids:
        batch = sync_client.batches.retrieve(bid)
        counts = batch.request_counts
        total_completed += counts.completed
        total_requests += counts.total
        total_failed += counts.failed
        print(f"Batch {bid}")
        print(f"  Status:    {batch.status}")
        print(f"  Completed: {counts.completed}/{counts.total}")
        print(f"  Failed:    {counts.failed}")
        if batch.output_file_id:
            print(f"  Output:    {batch.output_file_id}")

    if len(batch_ids) > 1:
        print(f"\nTotal: {total_completed}/{total_requests} completed, {total_failed} failed")


def cmd_wait(args: argparse.Namespace) -> None:
    """Poll until all batches complete."""
    batch_ids = _load_manifest(args.batch_id)
    poll_interval = args.poll_interval
    pending = set(batch_ids)

    print(f"Waiting for {len(pending)} batch job(s)...")
    while pending:
        total_completed = 0
        total_requests = 0
        total_failed = 0
        done_this_round = set()

        for bid in pending:
            batch = sync_client.batches.retrieve(bid)
            counts = batch.request_counts
            total_completed += counts.completed
            total_requests += counts.total
            total_failed += counts.failed

            if batch.status in ("completed", "failed", "expired", "cancelled"):
                done_this_round.add(bid)
                if batch.status != "completed":
                    print(f"\nBatch {bid} ended with status: {batch.status}")
                    if batch.errors:
                        for err in batch.errors.data[:5]:
                            print(f"  Error: {err.message}")

        status_msg = (
            f"Progress: {total_completed}/{total_requests} done, "
            f"{total_failed} failed | "
            f"{len(pending) - len(done_this_round)} batch(es) remaining"
        )
        print(f"\r{status_msg}", end="", flush=True)

        pending -= done_this_round
        if pending:
            time.sleep(poll_interval)

    print(f"\nAll {len(batch_ids)} batch job(s) finished.")


def _save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to the appropriate format based on file extension."""
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix == ".json":
        df.to_json(output_path, orient="records", indent=2)
    elif suffix == ".jsonl":
        df.to_json(output_path, orient="records", lines=True)
    else:
        df.to_parquet(output_path, index=False)


def _download_results(batch_ids: List[str]) -> Tuple[Dict[str, int], int]:
    """Download and parse results from batch jobs, streaming line-by-line.

    Returns (results_dict, error_count).  The results dict is {doc_id: level}
    and is intentionally lean (one int per row) so it stays small even for
    millions of rows.
    """
    results: Dict[str, int] = {}
    errors = 0

    for i, bid in enumerate(batch_ids):
        label = f"[{i+1}/{len(batch_ids)}] " if len(batch_ids) > 1 else ""
        batch = sync_client.batches.retrieve(bid)
        if not batch.output_file_id:
            print(f"{label}Batch {bid} has no output file (status: {batch.status}), skipping")
            continue

        print(f"{label}Downloading results for {bid}...")

        # Write the raw bytes to a temp file and iterate line-by-line to
        # avoid holding the full response text + a split list in memory
        response = sync_client.files.content(batch.output_file_id)
        raw_bytes = response.read()

        for line in io.BufferedReader(io.BytesIO(raw_bytes)):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj["custom_id"]
            if obj["response"]["status_code"] == 200:
                body = obj["response"]["body"]
                message_content = body["choices"][0]["message"]["content"]
                parsed = json.loads(message_content)
                results[doc_id] = parsed.get("reasoning_complexity", 0)
            else:
                errors += 1
                results[doc_id] = 0

        del raw_bytes

    return results, errors


def cmd_download(args: argparse.Namespace) -> None:
    """Download batch results and write a standalone results parquet.

    Produces a lightweight file with columns ``id`` and
    ``reasoning_complexity`` — no need to reload the massive source data.
    """
    batch_ids = _load_manifest(args.batch_id)
    output_path = Path(args.output)

    results, errors = _download_results(batch_ids)
    print(f"Parsed {len(results)} total results ({errors} errors)")

    # Build a small DataFrame from the results dict
    df = pd.DataFrame([
        {"id": doc_id, "reasoning_complexity": level}
        for doc_id, level in results.items()
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_output(df, output_path)
    print(f"\nSaved {len(df)} results to {output_path}")

    # Print distribution
    level_counts = {level: 0 for level in range(1, 6)}
    for level in range(1, 6):
        level_counts[level] = int((df["reasoning_complexity"] == level).sum())
    total = len(df)

    if total > 0:
        _print_distribution(level_counts, total)


def _print_distribution(level_counts: Dict[int, int], total: int) -> None:
    """Pretty-print the complexity level distribution."""
    print(f"\nComplexity Distribution ({total:,} classified rows):")
    print(f"{'Level':<45s} {'Count':>7s} {'Pct':>7s}")
    print(f"{'-'*45} {'-'*7} {'-'*7}")
    level_names = {
        1: "Level 1 - Factual/Declarative",
        2: "Level 2 - Single-step reasoning",
        3: "Level 3 - Multi-step reasoning",
        4: "Level 4 - Complex reasoning",
        5: "Level 5 - Formal/Abstract reasoning",
    }
    for level in range(1, 6):
        count = level_counts[level]
        pct = count / total * 100 if total > 0 else 0
        name = level_names.get(level, f"Level {level}")
        print(f"  {name:<43s} {count:>7d} {pct:>6.1f}%")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Print complexity distribution from results file(s).

    The results files are the lightweight outputs from ``download``
    (just ``id`` + ``reasoning_complexity``), so they load quickly.
    """
    input_files = discover_parquet_files(args.input)

    total = 0
    level_counts = {level: 0 for level in range(1, 6)}

    for f in input_files:
        df = pd.read_parquet(f, columns=["reasoning_complexity"])
        col = df["reasoning_complexity"].dropna()
        total += len(col)
        for level in range(1, 6):
            level_counts[level] += int((col == level).sum())
        del df, col

    if total == 0:
        sys.exit("No 'reasoning_complexity' values found in the data.")

    _print_distribution(level_counts, total)


def cmd_merge(args: argparse.Namespace) -> None:
    """Merge results back into the large source files, streaming row-group
    by row-group to keep memory low.

    Reads each source parquet one row-group at a time (~1000 rows), checks
    which IDs appear in the results, keeps only matching rows, appends the
    ``reasoning_complexity`` column, and writes them out.  Peak memory is
    roughly one row-group + the accumulated output rows (which is bounded
    by the number of results — typically thousands, not millions).
    """
    results_path = Path(args.results)
    source_path = args.source
    output_path = Path(args.output)
    id_column = args.id_column

    # Load the lightweight results file (just id + int)
    results_df = pd.read_parquet(results_path)
    if "id" not in results_df.columns or "reasoning_complexity" not in results_df.columns:
        sys.exit(f"Results file must have 'id' and 'reasoning_complexity' columns. "
                 f"Found: {list(results_df.columns)}")

    results_map: Dict[str, int] = dict(
        zip(results_df["id"].astype(str), results_df["reasoning_complexity"])
    )
    print(f"Loaded {len(results_map):,} results from {results_path}")
    del results_df

    # Discover source parquet files
    source_files = discover_parquet_files(source_path)
    total_source_rows = _count_total_rows(source_files)
    print(f"Found {len(source_files)} source file(s), {total_source_rows:,} total rows")
    print(f"Scanning for {len(results_map):,} matching IDs...\n")

    remaining_ids = set(results_map.keys())
    matched_chunks: List[pd.DataFrame] = []
    matched_count = 0

    pbar = tqdm(total=total_source_rows, desc="Scanning", unit="row")

    for fp in source_files:
        if not remaining_ids:
            pbar.update(pq.read_metadata(fp).num_rows)
            continue

        pf = pq.ParquetFile(fp)
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx)
            chunk_df = table.to_pandas()
            rg_len = len(chunk_df)
            del table

            if id_column not in chunk_df.columns:
                pbar.update(rg_len)
                del chunk_df
                continue

            ids = chunk_df[id_column].astype(str)
            mask = ids.isin(remaining_ids)
            n_matches = int(mask.sum())

            if n_matches > 0:
                hits = chunk_df.loc[mask].copy()
                hits["reasoning_complexity"] = hits[id_column].astype(str).map(results_map)
                matched_chunks.append(hits)
                matched_count += n_matches
                remaining_ids -= set(ids[mask])
                pbar.set_postfix(matched=f"{matched_count:,}/{len(results_map):,}")
                del hits

            pbar.update(rg_len)
            del chunk_df, ids, mask

    pbar.close()

    if not matched_chunks:
        sys.exit("No matching rows found in the source files.")

    merged_df = pd.concat(matched_chunks, ignore_index=True)
    del matched_chunks

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path, index=False)

    print(f"\nMatched {matched_count:,} / {len(results_map):,} results")
    if remaining_ids:
        print(f"  ({len(remaining_ids):,} IDs not found in source files)")
    print(f"Saved {len(merged_df):,} rows to {output_path}")
    print(f"Columns: {list(merged_df.columns)}")

    # Quick distribution
    level_counts = {level: 0 for level in range(1, 6)}
    for level in range(1, 6):
        level_counts[level] = int((merged_df["reasoning_complexity"] == level).sum())
    _print_distribution(level_counts, len(merged_df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu Reasoning Complexity Classification via OpenAI Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit 10,000 randomly sampled rows
  python complexity_fineweb_edu.py submit input/fineweb-edu/ -n 10000

  # Check batch status
  python complexity_fineweb_edu.py status complexity_batch_input.manifest.json

  # Wait for completion
  python complexity_fineweb_edu.py wait complexity_batch_input.manifest.json

  # Download results (lightweight parquet with id + reasoning_complexity)
  python complexity_fineweb_edu.py download complexity_batch_input.manifest.json -o results.parquet

  # Merge results with source data (streams row-group by row-group, low RAM)
  python complexity_fineweb_edu.py merge results.parquet input/fineweb-edu/ -o merged.parquet

  # Analyze complexity distribution
  python complexity_fineweb_edu.py analyze results.parquet
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- submit command ----
    p_submit = subparsers.add_parser("submit", help="Reservoir-sample rows and submit batch job")
    p_submit.add_argument("input", help="Path to parquet file or directory (searched recursively)")
    p_submit.add_argument("-n", "--num-rows", type=int, required=True,
                          help="Total number of rows to submit for complexity classification")
    p_submit.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    p_submit.add_argument("--text-column", default="text", help="Column containing text (default: text)")
    p_submit.add_argument("--id-column", default="id", help="Column with unique document IDs (default: id)")
    p_submit.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                          help=f"Max chars per text (default: {DEFAULT_MAX_CHARS})")
    p_submit.add_argument("--seed", type=int, default=DEFAULT_SEED,
                          help=f"Random seed for reservoir sampling (default: {DEFAULT_SEED})")
    p_submit.add_argument("--jsonl", default=None,
                          help="Path for intermediate JSONL (default: complexity_batch_input.jsonl)")
    p_submit.add_argument("--manifest", default=None,
                          help="Path for manifest file (default: complexity_batch_input.manifest.json)")
    p_submit.add_argument("--max-file-size-mb", type=int, default=DEFAULT_MAX_FILE_SIZE_MB,
                          help=f"Max JSONL file size in MB before splitting (default: {DEFAULT_MAX_FILE_SIZE_MB})")

    # ---- status command ----
    p_status = subparsers.add_parser("status", help="Check batch job status")
    p_status.add_argument("batch_id", help="Batch job ID or manifest file path")

    # ---- wait command ----
    p_wait = subparsers.add_parser("wait", help="Poll until batch completes")
    p_wait.add_argument("batch_id", help="Batch job ID or manifest file path")
    p_wait.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL,
                        help=f"Seconds between polls (default: {DEFAULT_POLL_INTERVAL})")

    # ---- download command ----
    p_download = subparsers.add_parser("download", help="Download results to a standalone parquet")
    p_download.add_argument("batch_id", help="Batch job ID or manifest file path")
    p_download.add_argument("-o", "--output", default="complexity_results.parquet",
                            help="Output parquet file (default: complexity_results.parquet)")

    # ---- merge command ----
    p_merge = subparsers.add_parser("merge",
                                    help="Merge results with large source files (streams row-group by row-group)")
    p_merge.add_argument("results", help="Path to results parquet from 'download' step")
    p_merge.add_argument("source", help="Path to source parquet file or directory (searched recursively)")
    p_merge.add_argument("-o", "--output", default="complexity_merged.parquet",
                         help="Output parquet file with full rows + complexity (default: complexity_merged.parquet)")
    p_merge.add_argument("--id-column", default="id", help="Column with unique document IDs (default: id)")

    # ---- analyze command ----
    p_analyze = subparsers.add_parser("analyze", help="Show complexity distribution stats")
    p_analyze.add_argument("input", help="Path to results parquet file or directory")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "wait":
        cmd_wait(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
