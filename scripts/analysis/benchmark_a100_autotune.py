#!/usr/bin/env python3
"""
A100 autotuning helper for Stage 1 classification throughput.

Runs short classification jobs across a configurable sweep grid, then writes:
  - benchmark_summary.json (all runs)
  - recommended_profile.json (best docs/sec profile)
"""

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _comma_ints(value: str):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _read_last_jsonl(path: Path):
    if not path.exists():
        return None
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last = json.loads(line)
    return last


def run_once(args, run_dir: Path, cfg: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "scored_shards"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("pipeline_classify.py")),
        "--local-dir",
        str(args.local_dir),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(cfg["batch_size"]),
        "--tokenize-chunk-size",
        str(cfg["chunk_size"]),
        "--prefetch-chunks",
        str(cfg["prefetch"]),
        "--collect-queue-size",
        str(cfg["collect_q"]),
        "--write-queue-size",
        str(cfg["write_q"]),
        "--read-batch-size",
        str(cfg["read_batch"]),
        "--max-shards",
        str(args.max_shards),
    ]
    if cfg["compile"]:
        cmd.append("--compile")
    if cfg["perf_mode"]:
        cmd.append("--perf-mode")
    if cfg["static_padding"]:
        cmd.append("--static-padding")
    if cfg["cuda_graphs"]:
        cmd.append("--cuda-graphs")
    if cfg["approximate_complexity"]:
        cmd.append("--approximate-complexity")
    if args.cc_dumps:
        cmd.extend(["--cc-dumps", *args.cc_dumps])

    print(f"\n=== Running config {cfg['run_id']} ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")

    progress_last = _read_last_jsonl(output_dir / "progress.jsonl")
    metrics_last = _read_last_jsonl(output_dir / "perf_metrics.jsonl")
    docs_per_sec = 0.0
    if progress_last and "overall_rate" in progress_last:
        docs_per_sec = float(progress_last["overall_rate"])

    result = {
        "config": cfg,
        "returncode": proc.returncode,
        "docs_per_sec": docs_per_sec,
        "progress_last": progress_last,
        "metrics_last": metrics_last,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Autotune A100 docs/sec for pipeline_classify")
    parser.add_argument("--local-dir", required=True, help="Local parquet root for classification input")
    parser.add_argument("--output-dir", default="autotune_results", help="Directory to write benchmark outputs")
    parser.add_argument("--max-shards", type=int, default=3, help="Shards per benchmark run")
    parser.add_argument("--batch-sizes", type=_comma_ints, default=[1536, 2048, 2560])
    parser.add_argument("--chunk-sizes", type=_comma_ints, default=[250000, 500000])
    parser.add_argument("--prefetch-chunks", type=_comma_ints, default=[2, 4])
    parser.add_argument("--collect-queues", type=_comma_ints, default=[2, 3])
    parser.add_argument("--write-queues", type=_comma_ints, default=[4, 8])
    parser.add_argument("--read-batches", type=_comma_ints, default=[8192, 32768])
    parser.add_argument("--compile-options", default="1", help="Comma list of 0/1")
    parser.add_argument("--perf-mode-options", default="1", help="Comma list of 0/1")
    parser.add_argument("--approximate-complexity-options", default="0,1", help="Comma list of 0/1")
    parser.add_argument("--static-padding-options", default="0,1", help="Comma list of 0/1")
    parser.add_argument("--cuda-graphs-options", default="0,1", help="Comma list of 0/1")
    parser.add_argument("--cc-dumps", nargs="*", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bool_grid = {
        "compile": [bool(int(x.strip())) for x in args.compile_options.split(",") if x.strip()],
        "perf_mode": [bool(int(x.strip())) for x in args.perf_mode_options.split(",") if x.strip()],
        "approximate_complexity": [
            bool(int(x.strip())) for x in args.approximate_complexity_options.split(",") if x.strip()
        ],
        "static_padding": [bool(int(x.strip())) for x in args.static_padding_options.split(",") if x.strip()],
        "cuda_graphs": [bool(int(x.strip())) for x in args.cuda_graphs_options.split(",") if x.strip()],
    }

    configs = []
    run_id = 0
    for (
        batch_size,
        chunk_size,
        prefetch,
        collect_q,
        write_q,
        read_batch,
        compile_flag,
        perf_mode_flag,
        approx_flag,
        static_flag,
        graph_flag,
    ) in itertools.product(
        args.batch_sizes,
        args.chunk_sizes,
        args.prefetch_chunks,
        args.collect_queues,
        args.write_queues,
        args.read_batches,
        bool_grid["compile"],
        bool_grid["perf_mode"],
        bool_grid["approximate_complexity"],
        bool_grid["static_padding"],
        bool_grid["cuda_graphs"],
    ):
        if graph_flag and not static_flag:
            continue
        cfg = {
            "run_id": run_id,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "prefetch": prefetch,
            "collect_q": collect_q,
            "write_q": write_q,
            "read_batch": read_batch,
            "compile": compile_flag,
            "perf_mode": perf_mode_flag,
            "approximate_complexity": approx_flag,
            "static_padding": static_flag,
            "cuda_graphs": graph_flag,
        }
        configs.append(cfg)
        run_id += 1

    results = []
    for cfg in configs:
        run_dir = out_dir / f"run_{cfg['run_id']:04d}"
        result = run_once(args, run_dir, cfg)
        results.append(result)
        print(
            f"Run {cfg['run_id']:04d}: rc={result['returncode']} docs/sec={result['docs_per_sec']:.1f} "
            f"approx={cfg['approximate_complexity']} graphs={cfg['cuda_graphs']}"
        )

    successful = [r for r in results if r["returncode"] == 0]
    best = max(successful, key=lambda r: r["docs_per_sec"]) if successful else None

    summary = {
        "total_runs": len(results),
        "successful_runs": len(successful),
        "best_docs_per_sec": best["docs_per_sec"] if best else 0.0,
        "best_config": best["config"] if best else None,
        "results": results,
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "recommended_profile.json").write_text(
        json.dumps(
            {
                "best_config": best["config"] if best else None,
                "best_docs_per_sec": best["docs_per_sec"] if best else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_dir / 'benchmark_summary.json'}")
    print(f"Wrote {out_dir / 'recommended_profile.json'}")


if __name__ == "__main__":
    main()

