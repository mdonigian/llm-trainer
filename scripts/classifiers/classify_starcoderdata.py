#!/usr/bin/env python3
"""
StarCoder Code Classification

Classifies code samples on three dimensions using OpenAI's Batch API:
  1. Code Quality (1-5)
  2. Structured Data Relevance (0-3)
  3. Content Type (one of 9 categories)

Designed to work with the sampled parquet files produced by
download_starcoderdata.py.

Usage:
  python classify_starcoderdata.py submit training_data/starcoderdata/
  python classify_starcoderdata.py status <batch_id_or_manifest>
  python classify_starcoderdata.py wait <batch_id_or_manifest>
  python classify_starcoderdata.py download <batch_id_or_manifest> -o results.parquet
  python classify_starcoderdata.py merge results.parquet training_data/starcoderdata/ -o classified/
  python classify_starcoderdata.py analyze results.parquet
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

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

CONTENT_TYPES = [
    "library", "application", "test", "config", "tutorial",
    "data", "generated", "script", "other",
]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class CodeClassification(BaseModel):
    """Structured output for three-dimensional code classification."""
    quality: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Code quality score. "
            "1=Broken/gibberish, 2=Functional but poor, 3=Decent, "
            "4=Good, 5=Excellent/professional-grade."
        ),
    )
    structured_data: int = Field(
        ...,
        ge=0,
        le=3,
        description=(
            "Structured data relevance. "
            "0=None, 1=Minor, 2=Significant, 3=Primarily about structured data."
        ),
    )
    content_type: Literal[
        "library", "application", "test", "config", "tutorial",
        "data", "generated", "script", "other",
    ] = Field(
        ...,
        description="The primary content type of the code.",
    )


SYSTEM_PROMPT = """\
You are an expert code quality classifier. Evaluate code on three dimensions.

Dimension 1: Code Quality (1-5)
1 - Broken, incomplete, or gibberish. Syntax errors, random fragments, encoded/binary data.
2 - Functional but poor. No structure, unclear naming, no comments, copy-paste style.
3 - Decent. Readable, functional, basic organization. Typical StackOverflow answer quality.
4 - Good. Clean style, consistent naming, some documentation, proper error handling.
5 - Excellent. Idiomatic, well-documented, professional-grade. Demonstrates best practices.

Dimension 2: Structured Data Relevance (0-3)
0 - No structured data patterns whatsoever.
1 - Minor structured data usage. E.g. reads a config value, prints some JSON incidentally.
2 - Significant structured data handling. E.g. defines an API endpoint, parses JSON, works with database models, validates input schemas.
3 - Primarily about structured data. E.g. schema definitions, serialization libraries, API client/server implementations, data pipeline code, type system definitions.

Dimension 3: Content Type (pick one)
- library: Reusable module or package code
- application: Application or business logic
- test: Unit tests, integration tests, test fixtures
- config: Configuration, setup, build scripts, CI/CD
- tutorial: Educational example, demo, or walkthrough
- data: Data file, fixture, or dump disguised as code
- generated: Auto-generated code (swagger codegen, protobuf output, migration files, etc.)
- script: One-off script, automation, or CLI tool
- other: Does not fit any category above"""

_SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_PROMPT}

# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def discover_parquet_files(input_path: str) -> List[Path]:
    """Return a sorted list of parquet files from a path (file or directory)."""
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


def _count_total_rows(parquet_files: List[Path]) -> int:
    """Fast row count from parquet metadata."""
    return sum(pq.read_metadata(fp).num_rows for fp in parquet_files)


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


def _make_request_line(
    doc_id: str, code: str, language: str, model: str, schema: dict,
) -> str:
    """Build a single JSONL request line for the Batch API."""
    request = {
        "custom_id": doc_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                _SYSTEM_MESSAGE,
                {
                    "role": "user",
                    "content": f"Evaluate this {language} code:\n\n{code}",
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "CodeClassification",
                    "strict": True,
                    "schema": schema,
                },
            },
        },
    }
    return json.dumps(request) + "\n"


def _save_manifest(
    manifest_path: str, batch_ids: List[str], metadata: dict | None = None,
) -> None:
    """Save batch IDs and optional metadata to a JSON manifest."""
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
    return [manifest_path]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_submit(args: argparse.Namespace) -> None:
    """Read sampled parquets and submit batch job(s)."""
    input_path = args.input
    max_chars = args.max_chars
    model = args.model
    max_file_bytes = args.max_file_size_mb * 1024 * 1024
    content_column = args.content_column
    id_column = args.id_column
    language_column = args.language_column
    jsonl_base = Path(args.jsonl or "starcode_batch_input.jsonl")
    manifest_path = args.manifest or str(jsonl_base.with_suffix(".manifest.json"))

    parquet_files = discover_parquet_files(input_path)
    total_rows = _count_total_rows(parquet_files)
    print(f"Found {len(parquet_files)} parquet file(s), {total_rows:,} total rows")

    dfs = []
    for fp in parquet_files:
        df = pd.read_parquet(fp, columns=[id_column, content_column, language_column])
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    del dfs
    actual_n = len(all_df)

    print(f"\nLanguage breakdown:")
    for lang, count in all_df[language_column].value_counts().sort_index().items():
        print(f"  {lang}: {count:,}")

    print(f"\n{'='*60}")
    print(f"Total rows: {actual_n:,}")
    print(f"Model:      {model}")
    print(f"{'='*60}")
    print()
    if not args.yes:
        confirm = input("Proceed with batch submission? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    schema = CodeClassification.model_json_schema()
    schema["additionalProperties"] = False

    print(f"\nPreparing {actual_n:,} requests...")

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

    for _, row in tqdm(all_df.iterrows(), total=actual_n, desc="Writing JSONL"):
        doc_id = str(row[id_column])
        code = str(row[content_column])
        language = str(row[language_column])
        if len(code) > max_chars:
            code = code[:max_chars] + "..."
        line = _make_request_line(doc_id, code, language, model, schema)
        line_bytes = len(line.encode("utf-8"))

        if current_size + line_bytes > max_file_bytes and current_size > 0:
            _open_new_chunk()

        current_file.write(line)
        current_size += line_bytes

    if current_file:
        current_file.close()

    del all_df

    print(f"Wrote {actual_n:,} requests across {len(chunk_paths)} file(s)")

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
            metadata={
                "description": f"starcoderdata classification (part {i+1}/{len(chunk_paths)})",
            },
        )
        batch_ids.append(batch.id)
        print(f"{label}Batch submitted: {batch.id} (status: {batch.status})")

    manifest_metadata = {
        "total_rows_submitted": actual_n,
        "model": model,
        "input_path": str(input_path),
    }
    _save_manifest(manifest_path, batch_ids, metadata=manifest_metadata)

    print(f"\n{'='*60}")
    print(f"Submitted {len(batch_ids)} batch job(s)")
    print(f"Manifest: {manifest_path}")
    print(f"\nCheck status:     python classify_starcoderdata.py status {manifest_path}")
    print(f"Wait for all:     python classify_starcoderdata.py wait {manifest_path}")
    print(f"Download results: python classify_starcoderdata.py download {manifest_path} -o starcode_results.parquet")


def cmd_status(args: argparse.Namespace) -> None:
    """Check the status of batch job(s)."""
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


def _download_results(
    batch_ids: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Download and parse results from batch jobs.

    Returns ({doc_id: {quality, structured_data, content_type}}, error_count).
    """
    results: Dict[str, Dict[str, Any]] = {}
    errors = 0

    for i, bid in enumerate(batch_ids):
        label = f"[{i+1}/{len(batch_ids)}] " if len(batch_ids) > 1 else ""
        batch = sync_client.batches.retrieve(bid)
        if not batch.output_file_id:
            print(f"{label}Batch {bid} has no output file (status: {batch.status}), skipping")
            continue

        print(f"{label}Downloading results for {bid}...")
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
                results[doc_id] = {
                    "quality": parsed.get("quality", 0),
                    "structured_data": parsed.get("structured_data", 0),
                    "content_type": parsed.get("content_type", "other"),
                }
            else:
                errors += 1

        del raw_bytes

    return results, errors


def cmd_download(args: argparse.Namespace) -> None:
    """Download batch results to a standalone parquet."""
    batch_ids = _load_manifest(args.batch_id)
    output_path = Path(args.output)

    results, errors = _download_results(batch_ids)
    print(f"Parsed {len(results):,} total results ({errors} errors)")

    df = pd.DataFrame([
        {"id": doc_id, **vals}
        for doc_id, vals in results.items()
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df):,} results to {output_path}")

    if len(df) > 0:
        _print_distributions(df)


def cmd_merge(args: argparse.Namespace) -> None:
    """Merge classification results back into source parquet files."""
    results_path = Path(args.results)
    source_path = args.source
    output_path = Path(args.output)
    id_column = args.id_column

    results_df = pd.read_parquet(results_path)
    required = {"id", "quality", "structured_data", "content_type"}
    if not required.issubset(results_df.columns):
        sys.exit(f"Results must have columns {required}. Found: {list(results_df.columns)}")

    print(f"Loaded {len(results_df):,} results from {results_path}")

    source_files = discover_parquet_files(source_path)
    print(f"Found {len(source_files)} source file(s)")

    output_path.mkdir(parents=True, exist_ok=True)
    total_matched = 0

    classification_cols = results_df[["id", "quality", "structured_data", "content_type"]]

    for fp in source_files:
        source_df = pd.read_parquet(fp)
        merged = source_df.merge(
            classification_cols,
            left_on=id_column,
            right_on="id",
            how="left",
            suffixes=("", "_cls"),
        )
        if "id_cls" in merged.columns:
            merged.drop(columns=["id_cls"], inplace=True)

        merged["quality"] = merged["quality"].fillna(0).astype(int)
        merged["structured_data"] = merged["structured_data"].fillna(0).astype(int)
        merged["content_type"] = merged["content_type"].fillna("other")

        matched = int(merged["quality"].gt(0).sum())
        total_matched += matched

        out_fp = output_path / fp.name
        merged.to_parquet(out_fp, index=False)
        print(f"  {fp.name}: {matched:,}/{len(merged):,} classified -> {out_fp}")
        del source_df, merged

    print(f"\nMatched {total_matched:,}/{len(results_df):,} results total")


def _print_distributions(df: pd.DataFrame) -> None:
    """Print classification distributions."""
    total = len(df)
    error_count = int((df["quality"] == 0).sum())

    print(f"\nCode Quality Distribution ({total:,} samples):")
    quality_names = {
        1: "1 - Broken/gibberish",
        2: "2 - Functional but poor",
        3: "3 - Decent",
        4: "4 - Good",
        5: "5 - Excellent",
    }
    print(f"  {'Level':<30s} {'Count':>7s} {'Pct':>7s}")
    print(f"  {'-'*30} {'-'*7} {'-'*7}")
    for level in range(1, 6):
        count = int((df["quality"] == level).sum())
        pct = count / total * 100
        print(f"  {quality_names[level]:<30s} {count:>7,d} {pct:>6.1f}%")
    if error_count > 0:
        print(f"  ({error_count:,} rows had API errors, quality=0)")

    print(f"\nStructured Data Relevance ({total:,} samples):")
    sd_names = {
        0: "0 - None",
        1: "1 - Minor",
        2: "2 - Significant",
        3: "3 - Primary focus",
    }
    print(f"  {'Level':<30s} {'Count':>7s} {'Pct':>7s}")
    print(f"  {'-'*30} {'-'*7} {'-'*7}")
    valid = df[df["quality"] > 0]
    valid_total = len(valid) if len(valid) > 0 else 1
    for level in range(0, 4):
        count = int((valid["structured_data"] == level).sum())
        pct = count / valid_total * 100
        print(f"  {sd_names[level]:<30s} {count:>7,d} {pct:>6.1f}%")

    print(f"\nContent Type ({total:,} samples):")
    ct_counts = valid["content_type"].value_counts().sort_values(ascending=False)
    print(f"  {'Type':<20s} {'Count':>7s} {'Pct':>7s}")
    print(f"  {'-'*20} {'-'*7} {'-'*7}")
    for ct, count in ct_counts.items():
        pct = count / valid_total * 100
        print(f"  {ct:<20s} {count:>7,d} {pct:>6.1f}%")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Print classification distributions from results file(s)."""
    input_files = discover_parquet_files(args.input)
    dfs = [pd.read_parquet(f) for f in input_files]
    df = pd.concat(dfs, ignore_index=True)
    del dfs

    required = {"quality", "structured_data", "content_type"}
    if not required.issubset(df.columns):
        sys.exit(f"Data must have columns {required}. Found: {list(df.columns)}")

    _print_distributions(df)

    if "language" in df.columns:
        valid = df[df["quality"] > 0]
        if len(valid) > 0:
            print(f"\n{'='*60}")
            print("Per-language quality breakdown:")
            lang_stats = valid.groupby("language")["quality"].agg(["mean", "median", "count"])
            lang_stats = lang_stats.sort_values("mean", ascending=False)
            print(f"  {'Language':<15s} {'Mean':>6s} {'Median':>7s} {'Count':>7s}")
            print(f"  {'-'*15} {'-'*6} {'-'*7} {'-'*7}")
            for lang, row in lang_stats.iterrows():
                print(f"  {lang:<15s} {row['mean']:>6.2f} {row['median']:>7.1f} {int(row['count']):>7,d}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="StarCoder Code Classification via OpenAI Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit all downloaded samples for classification
  python classify_starcoderdata.py submit training_data/starcoderdata/

  # Check batch status
  python classify_starcoderdata.py status starcode_batch_input.manifest.json

  # Wait for completion
  python classify_starcoderdata.py wait starcode_batch_input.manifest.json

  # Download results (lightweight parquet with id + classification columns)
  python classify_starcoderdata.py download starcode_batch_input.manifest.json -o starcode_results.parquet

  # Merge results with source data
  python classify_starcoderdata.py merge starcode_results.parquet training_data/starcoderdata/ -o training_data/starcoderdata-classified/

  # Analyze distributions
  python classify_starcoderdata.py analyze starcode_results.parquet
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- submit ----
    p_submit = subparsers.add_parser("submit", help="Read samples and submit batch job")
    p_submit.add_argument("input", help="Path to parquet file or directory")
    p_submit.add_argument("--model", default=DEFAULT_MODEL,
                          help=f"OpenAI model (default: {DEFAULT_MODEL})")
    p_submit.add_argument("--content-column", default="content",
                          help="Column containing code (default: content)")
    p_submit.add_argument("--id-column", default="id",
                          help="Column with unique IDs (default: id)")
    p_submit.add_argument("--language-column", default="language",
                          help="Column with language name (default: language)")
    p_submit.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                          help=f"Max chars per code sample (default: {DEFAULT_MAX_CHARS})")
    p_submit.add_argument("--jsonl", default=None,
                          help="Path for JSONL file (default: starcode_batch_input.jsonl)")
    p_submit.add_argument("--manifest", default=None,
                          help="Path for manifest (default: starcode_batch_input.manifest.json)")
    p_submit.add_argument("--max-file-size-mb", type=int, default=DEFAULT_MAX_FILE_SIZE_MB,
                          help=f"Max JSONL size in MB before splitting (default: {DEFAULT_MAX_FILE_SIZE_MB})")
    p_submit.add_argument("-y", "--yes", action="store_true",
                          help="Skip confirmation prompt")

    # ---- status ----
    p_status = subparsers.add_parser("status", help="Check batch job status")
    p_status.add_argument("batch_id", help="Batch ID or manifest file")

    # ---- wait ----
    p_wait = subparsers.add_parser("wait", help="Poll until batch completes")
    p_wait.add_argument("batch_id", help="Batch ID or manifest file")
    p_wait.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL,
                        help=f"Seconds between polls (default: {DEFAULT_POLL_INTERVAL})")

    # ---- download ----
    p_download = subparsers.add_parser("download",
                                       help="Download results to a standalone parquet")
    p_download.add_argument("batch_id", help="Batch ID or manifest file")
    p_download.add_argument("-o", "--output", default="starcode_results.parquet",
                            help="Output parquet (default: starcode_results.parquet)")

    # ---- merge ----
    p_merge = subparsers.add_parser("merge",
                                    help="Merge results with source parquet files")
    p_merge.add_argument("results", help="Path to results parquet from 'download'")
    p_merge.add_argument("source", help="Path to source parquet file or directory")
    p_merge.add_argument("-o", "--output", default="training_data/starcoderdata-classified/",
                         help="Output directory (default: training_data/starcoderdata-classified/)")
    p_merge.add_argument("--id-column", default="id",
                         help="Column with unique IDs (default: id)")

    # ---- analyze ----
    p_analyze = subparsers.add_parser("analyze", help="Show classification distributions")
    p_analyze.add_argument("input", help="Path to results/classified parquet file or directory")

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
