#!/usr/bin/env python3
"""
FineWeb-Edu Content Categorization

Processes FineWeb-Edu dataset files and uses OpenAI's structured output API
to classify each row across 17 academic/subject categories.

Two processing modes:
  async  — Real-time concurrent API calls (fast, good for moderate datasets)
  batch  — OpenAI Batch API (50% cheaper, higher throughput, up to 24h)

Usage:
  python categorize.py async input.parquet -o output.parquet
  python categorize.py batch submit input.parquet
  python categorize.py batch status <batch_id>
  python categorize.py batch download <batch_id> input.parquet -o output.parquet
  python categorize.py analyze output.parquet
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("Error: OPENAI_API_KEY not found. Set it in .env or as an environment variable.")

async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
sync_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI flags)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_CONCURRENCY = 500
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_RETRY_DELAY = 1
DEFAULT_SAVE_EVERY = 1000
DEFAULT_MAX_CHARS = 8000
DEFAULT_POLL_INTERVAL = 30
DEFAULT_MAX_FILE_SIZE_MB = 150  # Stay under OpenAI's 200MB upload limit

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ContentCategories(BaseModel):
    """Structured output schema for content categorization."""
    mathematics_statistics: bool
    computer_science_software_engineering: bool
    machine_learning_ai: bool
    physical_sciences: bool
    life_sciences_biology: bool
    medicine_health: bool
    engineering_technology: bool
    business_economics: bool
    law_government: bool
    social_sciences: bool
    history_geography: bool
    philosophy_ethics: bool
    education_pedagogy: bool
    language_writing: bool
    arts_humanities: bool
    environmental_science_energy: bool
    personal_finance_practical_life: bool


CATEGORY_FIELDS = list(ContentCategories.model_fields.keys())

SYSTEM_PROMPT = """You are an expert content classifier. Analyze the given text and determine which academic/subject categories apply to it.

A text can belong to MULTIPLE categories. Mark a category as true if the content substantively covers topics in that category.

Categories:
- Mathematics & Statistics: Math, statistics, probability, algebra, calculus, etc.
- Computer Science & Software Engineering: Programming, algorithms, software development, systems
- Machine Learning & AI: ML, AI, neural networks, data science
- Physical Sciences: Physics, chemistry, astronomy, geology
- Life Sciences & Biology: Biology, genetics, ecology, zoology, botany
- Medicine & Health: Healthcare, medical topics, diseases, treatments
- Engineering & Technology: Engineering disciplines, technical systems
- Business & Economics: Business, economics, finance, management, marketing
- Law & Government: Legal topics, government, policy, regulations
- Social Sciences: Psychology, sociology, anthropology, political science
- History & Geography: Historical events, geography, civilizations
- Philosophy & Ethics: Philosophy, ethics, logic, moral reasoning
- Education & Pedagogy: Teaching, learning, educational methods
- Language & Writing: Linguistics, literature, writing, languages
- Arts & Humanities: Art, music, culture, humanities
- Environmental Science & Energy: Environment, climate, energy, sustainability
- Personal Finance & Practical Life: Personal finance, practical life skills"""

_SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_PROMPT}

# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_fineweb_edu_file(filepath: str) -> pd.DataFrame:
    """Load a FineWeb-Edu file (parquet, csv, json, jsonl)."""
    p = Path(filepath)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix == ".csv":
        return pd.read_csv(p)
    elif p.suffix == ".json":
        return pd.read_json(p)
    elif p.suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    else:
        sys.exit(f"Unsupported file format: {p.suffix}")


def get_text_content(row: pd.Series, text_column: str = "text") -> str:
    """Extract text from a DataFrame row."""
    if text_column in row:
        return str(row[text_column])
    for col in ("content", "document", "body"):
        if col in row:
            return str(row[col])
    sys.exit(f"Could not find text column. Available: {list(row.index)}")

# ---------------------------------------------------------------------------
# Async real-time processing
# ---------------------------------------------------------------------------

async def categorize_content(
    text: str,
    semaphore: asyncio.Semaphore,
    model: str,
    max_chars: int,
    max_retries: int,
    base_delay: float,
) -> Dict[str, bool]:
    """Categorize a single text with structured output (async + retries)."""
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    user_message = {"role": "user", "content": f"Analyze this text and categorize it:\n\n{text}"}

    async with semaphore:
        for attempt in range(max_retries):
            try:
                completion = await async_client.beta.chat.completions.parse(
                    model=model,
                    messages=[_SYSTEM_MESSAGE, user_message],
                    response_format=ContentCategories,
                )
                return completion.choices[0].message.parsed.model_dump()
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "rate_limit" in err_str.lower() or "429" in err_str
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.random()
                    if is_rate_limit:
                        delay *= 2
                    await asyncio.sleep(delay)
                else:
                    print(f"\nFailed after {max_retries} attempts: {e}")
                    return {f: False for f in CATEGORY_FIELDS}


async def run_async(args: argparse.Namespace) -> None:
    """Real-time async processing mode."""
    filepath = args.input
    output_path = args.output
    text_column = args.text_column
    max_rows = args.max_rows
    concurrency = args.concurrency
    model = args.model
    save_every = args.save_every
    max_chars = args.max_chars
    start_index = args.start_index

    print(f"Loading file: {filepath}")
    df = load_fineweb_edu_file(filepath)
    print(f"Loaded {len(df)} rows")

    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)
        print(f"Limited to {len(df)} rows")

    for field in CATEGORY_FIELDS:
        if field not in df.columns:
            df[field] = None

    remaining = [
        idx for idx in range(start_index, len(df))
        if df.loc[idx, CATEGORY_FIELDS[0]] is None
    ]
    print(f"\nProcessing {len(remaining)} rows (concurrency={concurrency}, model={model})...")

    texts_by_idx = {}
    for idx in remaining:
        try:
            texts_by_idx[idx] = get_text_content(df.iloc[idx], text_column)
        except Exception as e:
            print(f"Error reading row {idx}: {e}")
            texts_by_idx[idx] = ""

    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(remaining), desc="Categorizing")
    completed_count = 0
    lock = asyncio.Lock()

    async def _process_one(idx: int, text: str):
        nonlocal completed_count
        result = await categorize_content(
            text, semaphore, model, max_chars, args.max_retries, args.base_retry_delay
        )
        for field, value in result.items():
            df.at[idx, field] = value

        async with lock:
            completed_count += 1
            pbar.update(1)
            if output_path and completed_count % save_every == 0:
                df.to_parquet(output_path, index=False)
                pbar.set_postfix(saved=completed_count)

    tasks = [_process_one(idx, text) for idx, text in texts_by_idx.items()]
    await asyncio.gather(*tasks)
    pbar.close()

    if output_path:
        print(f"\nSaving final results to: {output_path}")
        df.to_parquet(output_path, index=False)
    print("Done!")

# ---------------------------------------------------------------------------
# Batch API processing
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
                {"role": "user", "content": f"Analyze this text and categorize it:\n\n{text}"},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ContentCategories",
                    "strict": True,
                    "schema": schema,
                },
            },
        },
    }
    return json.dumps(request) + "\n"


def _save_manifest(manifest_path: str, batch_ids: List[str]) -> None:
    """Save batch IDs to a JSON manifest file."""
    with open(manifest_path, "w") as f:
        json.dump({"batch_ids": batch_ids}, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


def _load_manifest(manifest_path: str) -> List[str]:
    """Load batch IDs from a manifest file, or treat input as a single batch ID."""
    p = Path(manifest_path)
    if p.suffix == ".json" and p.exists():
        with open(p) as f:
            data = json.load(f)
        return data["batch_ids"]
    else:
        # Assume it's a single batch ID string
        return [manifest_path]


def batch_submit(args: argparse.Namespace) -> None:
    """Prepare JSONL file(s), upload, and submit Batch API job(s).

    Automatically splits into multiple batches if the JSONL exceeds the
    file size limit (default 150MB, well under OpenAI's 200MB cap).
    All batch IDs are saved to a manifest JSON file.
    """
    filepath = args.input
    text_column = args.text_column
    id_column = args.id_column
    max_rows = args.max_rows
    max_chars = args.max_chars
    model = args.model
    max_file_bytes = args.max_file_size_mb * 1024 * 1024
    jsonl_base = Path(args.jsonl or "batch_input.jsonl")
    manifest_path = args.manifest or str(jsonl_base.with_suffix(".manifest.json"))

    df = load_fineweb_edu_file(filepath)
    if max_rows:
        df = df.iloc[:max_rows]

    if id_column not in df.columns:
        sys.exit(f"ID column '{id_column}' not found. Available columns: {list(df.columns)}")

    schema = ContentCategories.model_json_schema()
    schema["additionalProperties"] = False

    # --- Write JSONL, splitting into chunks when file size limit is reached ---
    print(f"Preparing {len(df)} requests (keyed on '{id_column}' column)...")

    chunk_paths: List[str] = []
    chunk_idx = 0
    current_size = 0
    current_file = None

    def _open_new_chunk():
        nonlocal chunk_idx, current_size, current_file
        if current_file:
            current_file.close()
        if len(chunk_paths) == 0 and chunk_idx == 0:
            # First chunk — use the base name directly
            path = str(jsonl_base)
        else:
            stem = jsonl_base.stem
            path = str(jsonl_base.with_name(f"{stem}_{chunk_idx}{jsonl_base.suffix}"))
        chunk_paths.append(path)
        current_file = open(path, "w")
        current_size = 0
        chunk_idx += 1

    _open_new_chunk()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing JSONL"):
        doc_id = str(row[id_column])
        text = get_text_content(row, text_column)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        line = _make_request_line(doc_id, text, model, schema)
        line_bytes = len(line.encode("utf-8"))

        # Start a new chunk if this line would push us over the limit
        if current_size + line_bytes > max_file_bytes and current_size > 0:
            _open_new_chunk()

        current_file.write(line)
        current_size += line_bytes

    if current_file:
        current_file.close()

    print(f"Wrote {len(df)} requests across {len(chunk_paths)} file(s)")

    # --- Upload and submit each chunk ---
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
            metadata={"description": f"fineweb-edu categorization (part {i+1}/{len(chunk_paths)})"},
        )
        batch_ids.append(batch.id)
        print(f"{label}Batch submitted: {batch.id} (status: {batch.status})")

    _save_manifest(manifest_path, batch_ids)

    print(f"\n{'='*60}")
    print(f"Submitted {len(batch_ids)} batch job(s)")
    print(f"Manifest: {manifest_path}")
    print(f"\nCheck status:     python categorize.py batch status {manifest_path}")
    print(f"Wait for all:     python categorize.py batch wait {manifest_path}")
    print(f"Download results: python categorize.py batch download {manifest_path} {filepath} -o output.parquet")


def batch_status(args: argparse.Namespace) -> None:
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


def batch_download(args: argparse.Namespace) -> None:
    """Download results from one or more batch jobs and merge with the original file."""
    batch_ids = _load_manifest(args.batch_id)
    filepath = args.input
    output_path = args.output
    max_rows = args.max_rows
    id_column = args.id_column

    # Collect results from all batch jobs
    results = {}
    errors = 0

    for i, bid in enumerate(batch_ids):
        label = f"[{i+1}/{len(batch_ids)}] " if len(batch_ids) > 1 else ""
        batch = sync_client.batches.retrieve(bid)
        if not batch.output_file_id:
            print(f"{label}Batch {bid} has no output file (status: {batch.status}), skipping")
            continue

        print(f"{label}Downloading results for {bid}...")
        content = sync_client.files.content(batch.output_file_id)

        for line in content.text.strip().split("\n"):
            obj = json.loads(line)
            doc_id = obj["custom_id"]
            if obj["response"]["status_code"] == 200:
                body = obj["response"]["body"]
                message_content = body["choices"][0]["message"]["content"]
                results[doc_id] = json.loads(message_content)
            else:
                errors += 1
                results[doc_id] = {f: False for f in CATEGORY_FIELDS}

    print(f"Parsed {len(results)} total results ({errors} errors)")

    df = load_fineweb_edu_file(filepath)
    if max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)

    if id_column not in df.columns:
        sys.exit(f"ID column '{id_column}' not found. Available columns: {list(df.columns)}")

    # Build a lookup from id -> row position for fast joining
    id_to_idx = {str(v): i for i, v in enumerate(df[id_column])}

    for field in CATEGORY_FIELDS:
        df[field] = None

    matched = 0
    for doc_id, categories in results.items():
        if doc_id in id_to_idx:
            idx = id_to_idx[doc_id]
            for field, value in categories.items():
                if field in CATEGORY_FIELDS:
                    df.at[idx, field] = value
            matched += 1

    print(f"Matched {matched}/{len(results)} results to rows")
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")


def batch_wait(args: argparse.Namespace) -> None:
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

# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def run_analyze(args: argparse.Namespace) -> None:
    """Print category distribution statistics."""
    filepath = args.input
    df = load_fineweb_edu_file(filepath)

    stats = []
    for field in CATEGORY_FIELDS:
        if field in df.columns:
            count = int(df[field].sum())
            pct = (count / len(df)) * 100
            stats.append({
                "Category": field.replace("_", " ").title(),
                "Count": count,
                "Percentage": f"{pct:.2f}%",
            })

    stats_df = pd.DataFrame(stats).sort_values("Count", ascending=False)
    print(f"\nCategory Distribution ({len(df)} rows):\n")
    print(stats_df.to_string(index=False))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FineWeb-Edu Content Categorization with OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- shared arguments ---
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    common.add_argument("--text-column", default="text", help="Column containing text (default: text)")
    common.add_argument("--max-rows", type=int, default=None, help="Max rows to process (default: all)")
    common.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help=f"Max chars per text (default: {DEFAULT_MAX_CHARS})")

    # ---- async command ----
    p_async = subparsers.add_parser("async", parents=[common], help="Real-time async processing")
    p_async.add_argument("input", help="Path to input file (parquet, csv, json, jsonl)")
    p_async.add_argument("-o", "--output", default="fineweb-edu-categorized.parquet", help="Output file path")
    p_async.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"Max concurrent requests (default: {DEFAULT_CONCURRENCY})")
    p_async.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help=f"Save checkpoint every N rows (default: {DEFAULT_SAVE_EVERY})")
    p_async.add_argument("--start-index", type=int, default=0, help="Start from row index (for resuming)")
    p_async.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"Max retries per request (default: {DEFAULT_MAX_RETRIES})")
    p_async.add_argument("--base-retry-delay", type=float, default=DEFAULT_BASE_RETRY_DELAY, help=f"Base retry delay seconds (default: {DEFAULT_BASE_RETRY_DELAY})")

    # ---- batch command ----
    p_batch = subparsers.add_parser("batch", help="OpenAI Batch API (50%% cheaper)")
    batch_sub = p_batch.add_subparsers(dest="batch_command", required=True)

    # batch submit
    p_submit = batch_sub.add_parser("submit", parents=[common], help="Prepare and submit a batch job")
    p_submit.add_argument("input", help="Path to input file")
    p_submit.add_argument("--id-column", default="id", help="Column with unique document IDs (default: id)")
    p_submit.add_argument("--jsonl", default=None, help="Path for intermediate JSONL (default: batch_input.jsonl)")
    p_submit.add_argument("--manifest", default=None, help="Path for manifest file (default: batch_input.manifest.json)")
    p_submit.add_argument("--max-file-size-mb", type=int, default=DEFAULT_MAX_FILE_SIZE_MB,
                          help=f"Max JSONL file size in MB before splitting (default: {DEFAULT_MAX_FILE_SIZE_MB})")

    # batch status
    p_status = batch_sub.add_parser("status", help="Check batch job status")
    p_status.add_argument("batch_id", help="Batch job ID")

    # batch wait
    p_wait = batch_sub.add_parser("wait", help="Poll until batch completes")
    p_wait.add_argument("batch_id", help="Batch job ID")
    p_wait.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL, help=f"Seconds between polls (default: {DEFAULT_POLL_INTERVAL})")

    # batch download
    p_download = batch_sub.add_parser("download", parents=[common], help="Download results and merge")
    p_download.add_argument("batch_id", help="Batch job ID")
    p_download.add_argument("input", help="Path to original input file")
    p_download.add_argument("--id-column", default="id", help="Column with unique document IDs (default: id)")
    p_download.add_argument("-o", "--output", default="fineweb-edu-categorized.parquet", help="Output file path")

    # ---- analyze command ----
    p_analyze = subparsers.add_parser("analyze", help="Show category distribution stats")
    p_analyze.add_argument("input", help="Path to categorized output file")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "async":
        asyncio.run(run_async(args))
    elif args.command == "batch":
        if args.batch_command == "submit":
            batch_submit(args)
        elif args.batch_command == "status":
            batch_status(args)
        elif args.batch_command == "wait":
            batch_wait(args)
        elif args.batch_command == "download":
            batch_download(args)
    elif args.command == "analyze":
        run_analyze(args)


if __name__ == "__main__":
    main()
