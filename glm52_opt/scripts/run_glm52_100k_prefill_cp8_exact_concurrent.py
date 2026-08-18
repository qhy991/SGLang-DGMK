#!/usr/bin/env python3
"""Fail-closed concurrent exact-token runner for CP8/EP8 100K prefill.

The generic OpenAI-compatible benchmark turns token IDs into text and asks the
server to tokenize that text again.  For GLM-5.2 the round trip can shorten a
100K request, which silently changes both the radix-cache hit and the CP-local
matrix shape.  This runner sends ``input_ids`` directly and gives every suffix
a unique first token.  Consequently every measured request has the same
auditable contract: 89,984 cached tokens and 10,048 scheduled tokens on each
attention-CP rank (10,016 real uncached tokens plus runtime padding).  ``--x``
is both the request-concurrency limit and the multiplier for ten measured
requests, matching the frozen x1/x11 campaign without text-tokenization drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


FATAL_SERVER_PATTERNS = (
    "Traceback (most recent call last)",
    "illegal memory access",
    "CUBLAS_STATUS_EXECUTION_FAILED",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "KV cache pool is full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument(
        "--arm", choices=("baseline", "phase1", "phase2", "all"), required=True
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--model", default="/mnt/b300-shared/models/GLM-5.2-FP8"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/mnt/b300-shared/home/qinhaiyan/wwxq/"
            "ShareGPT_V3_unfiltered_cleaned_split.json"
        ),
    )
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--requests-per-x", type=int, default=10)
    parser.add_argument("--prefix-len", type=int, default=90_000)
    parser.add_argument("--suffix-len", type=int, default=10_000)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--warmup-timeout-s", type=float, default=3600.0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/"
            "glm52_cp8_ep8_exact_runs"
        ),
    )
    # Accepted for launcher compatibility.  Exact-token baseline profiling is
    # wired separately after a stable no-profiler baseline exists.
    parser.add_argument("--allow-existing-hit-evidence", action="store_true")
    parser.add_argument("--capture-trigger", type=Path, default=None)
    parser.add_argument("--capture-start-timeout-s", type=float, default=15.0)
    parser.add_argument("--capture-stop-timeout-s", type=float, default=60.0)
    args = parser.parse_args()
    if args.x <= 0:
        parser.error("--x must be positive")
    if args.requests_per_x <= 0:
        parser.error("--requests-per-x must be positive")
    if args.output_len != 1:
        parser.error("this TTFT contract requires --output-len 1")
    if args.capture_start_timeout_s <= 0:
        parser.error("--capture-start-timeout-s must be positive")
    if args.capture_stop_timeout_s <= 0:
        parser.error("--capture-stop-timeout-s must be positive")
    for path in (Path(args.model), args.dataset, args.server_log):
        if not path.exists():
            parser.error(f"missing required path: {path}")
    return args


def repeat_to_length(values: list[int], length: int) -> list[int]:
    if not values:
        raise ValueError("cannot repeat an empty token sequence")
    return (values * ((length + len(values) - 1) // len(values)))[:length]


def build_exact_inputs(
    tokenizer: Any,
    dataset_path: Path,
    *,
    count: int,
    prefix_len: int,
    suffix_len: int,
) -> tuple[list[int], list[list[int]], list[int]]:
    raw = json.loads(dataset_path.read_text())
    prompts: list[str] = []
    for item in raw:
        turns = item.get("conversations", item.get("conversation", []))
        if len(turns) >= 2:
            prompts.append(turns[0]["value"])
        if len(prompts) == count:
            break
    if len(prompts) != count:
        raise RuntimeError(f"dataset has {len(prompts)} usable prompts; need {count}")

    prefix = repeat_to_length(tokenizer.encode(prompts[0]), prefix_len)
    special_ids = set(tokenizer.all_special_ids)
    sentinel_ids: list[int] = []
    candidate = 1_000
    while len(sentinel_ids) < count:
        if candidate < tokenizer.vocab_size and candidate not in special_ids:
            sentinel_ids.append(candidate)
        candidate += 1
    if len(set(sentinel_ids)) != count:
        raise AssertionError("suffix sentinels are not unique")

    rows: list[list[int]] = []
    for prompt, sentinel in zip(prompts, sentinel_ids, strict=True):
        suffix = repeat_to_length(tokenizer.encode(prompt), suffix_len)
        suffix[0] = sentinel
        row = prefix + suffix
        if len(row) != prefix_len + suffix_len:
            raise AssertionError("exact-token row length drift")
        rows.append(row)
    return prefix, rows, sentinel_ids


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def one_request(
    session: requests.Session,
    url: str,
    input_ids: list[int],
    *,
    timeout: float,
    output_len: int,
) -> tuple[float, int]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": False,
    }
    start = time.perf_counter()
    response = session.post(url, json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - start) * 1_000.0
    response.raise_for_status()
    body = response.json()
    records = body if isinstance(body, list) else [body]
    if len(records) != 1:
        raise RuntimeError(f"expected one response record, got {len(records)}")
    output_ids = records[0].get("output_ids")
    if output_ids is None:
        token_logprobs = records[0].get("meta_info", {}).get(
            "output_token_logprobs"
        )
        if token_logprobs:
            output_ids = [token_logprobs[0][1]]
    if output_ids is None or len(output_ids) != output_len:
        raise RuntimeError(f"expected {output_len} output token, got {output_ids}")
    return elapsed_ms, int(output_ids[0])


def one_isolated_request(
    url: str,
    input_ids: list[int],
    *,
    timeout: float,
    output_len: int,
) -> tuple[float, int]:
    """Issue one measured request from a task-local HTTP session."""

    with requests.Session() as session:
        return one_request(
            session,
            url,
            input_ids,
            timeout=timeout,
            output_len=output_len,
        )


def validate_measured_server_delta(
    delta: str,
    *,
    expected_ranks: int,
    requests_count: int,
    expected_cached_tokens: int,
    expected_scheduled_tokens: int,
    arm: str,
) -> dict[str, Any]:
    fatal = [pattern for pattern in FATAL_SERVER_PATTERNS if pattern in delta]
    if fatal:
        raise RuntimeError(f"measured server log contains fatal patterns: {fatal}")

    pattern = re.compile(
        r"ATTN_CP(\d+) TP\d+ EP\d+\] Prefill batch, .*?"
        r"#new-token: (\d+), #cached-token: (\d+)"
    )
    rows = [tuple(map(int, match.groups())) for match in pattern.finditer(delta)]
    expected_total = expected_ranks * requests_count
    matching = [
        row
        for row in rows
        if row[1] == expected_scheduled_tokens
        and row[2] == expected_cached_tokens
    ]
    per_rank = {
        str(rank): sum(row[0] == rank for row in matching)
        for rank in range(expected_ranks)
    }
    positive_cache_values = sorted({row[2] for row in rows if row[2] > 0})
    scheduled_values = sorted({row[1] for row in rows if row[2] > 0})
    if (
        len(matching) != expected_total
        or any(value != requests_count for value in per_rank.values())
        or positive_cache_values != [expected_cached_tokens]
        or scheduled_values != [expected_scheduled_tokens]
    ):
        raise RuntimeError(
            "exact-token CP8 cache/shape contract failed: "
            f"matching={len(matching)}/{expected_total}, per_rank={per_rank}, "
            f"cached={positive_cache_values}, scheduled={scheduled_values}"
        )

    e2e_hits = [
        line for line in delta.splitlines() if "glm52_opt HIT e2e_prefill/" in line
    ]
    if arm == "baseline" and e2e_hits:
        raise RuntimeError("baseline emitted an optimized E2E-prefill HIT marker")
    return {
        "expected_cached_tokens": expected_cached_tokens,
        "expected_scheduled_tokens_per_rank": expected_scheduled_tokens,
        "matching_prefill_lines": len(matching),
        "per_rank_matching_lines": per_rank,
        "positive_cache_values": positive_cache_values,
        "scheduled_values": scheduled_values,
        "e2e_hit_lines": len(e2e_hits),
    }


def arm_causal_capture(args: argparse.Namespace) -> None:
    """Open the profiler gate after warmup and before measured traffic."""

    trigger = args.capture_trigger
    if trigger is None:
        return
    if trigger.exists():
        raise RuntimeError(f"capture trigger already exists: {trigger}")
    before = args.server_log.read_text(errors="replace").count("nsys capture START")
    trigger.parent.mkdir(parents=True, exist_ok=True)
    trigger.touch()
    deadline = time.monotonic() + args.capture_start_timeout_s
    while time.monotonic() < deadline:
        current = args.server_log.read_text(errors="replace").count(
            "nsys capture START"
        )
        if current - before >= args.expected_ranks:
            return
        time.sleep(0.1)
    raise RuntimeError(
        "capture gate did not start on all workers: "
        f"expected={args.expected_ranks}, trigger={trigger}"
    )


def wait_causal_capture_stop(args: argparse.Namespace) -> None:
    """Wait for every worker to close the cudaProfilerApi capture window."""

    if args.capture_trigger is None:
        return
    deadline = time.monotonic() + args.capture_stop_timeout_s
    while time.monotonic() < deadline:
        count = args.server_log.read_text(errors="replace").count("nsys capture STOP")
        if count >= args.expected_ranks:
            return
        time.sleep(0.1)
    raise RuntimeError(
        "capture gate did not stop on all workers: "
        f"expected={args.expected_ranks}, trigger={args.capture_trigger}"
    )


def main() -> None:
    args = parse_args()
    prompts = args.x * args.requests_per_x
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prefix, inputs, sentinels = build_exact_inputs(
        tokenizer,
        args.dataset,
        count=prompts,
        prefix_len=args.prefix_len,
        suffix_len=args.suffix_len,
    )
    canonical_inputs = json.dumps(inputs, separators=(",", ":")).encode()
    input_sha256 = hashlib.sha256(canonical_inputs).hexdigest()
    sentinel_sha256 = hashlib.sha256(
        json.dumps(sentinels, separators=(",", ":")).encode()
    ).hexdigest()

    base_url = args.base_url.rstrip("/")
    generate_url = f"{base_url}/generate"
    session = requests.Session()
    health = session.get(f"{base_url}/v1/models", timeout=30.0)
    health.raise_for_status()
    flush = session.post(f"{base_url}/flush_cache", timeout=300.0)
    flush.raise_for_status()
    if "Cache flushed" not in flush.text:
        raise RuntimeError(f"unexpected flush response: {flush.text.strip()}")

    run_stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    run_dir = args.result_root / f"{run_stamp}_{args.arm}_exact_x{args.x}"
    run_dir.mkdir(parents=True, exist_ok=False)
    initial_offset = args.server_log.stat().st_size
    warmup_ms, warmup_output_id = one_request(
        session,
        generate_url,
        prefix,
        timeout=args.warmup_timeout_s,
        output_len=args.output_len,
    )
    measured_offset = args.server_log.stat().st_size
    arm_causal_capture(args)

    ttft_ms: list[float] = [0.0] * prompts
    output_ids: list[int] = [-1] * prompts
    benchmark_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.x) as executor:
        futures = {
            executor.submit(
                one_isolated_request,
                generate_url,
                input_ids,
                timeout=args.request_timeout_s,
                output_len=args.output_len,
            ): index
            for index, input_ids in enumerate(inputs)
        }
        for future in as_completed(futures):
            index = futures[future]
            elapsed_ms, output_id = future.result()
            ttft_ms[index] = elapsed_ms
            output_ids[index] = output_id
    benchmark_duration_s = time.perf_counter() - benchmark_start
    if any(value <= 0.0 for value in ttft_ms) or any(value < 0 for value in output_ids):
        raise RuntimeError("concurrent exact request result was not populated")

    # The HTTP response is emitted after the server's per-request logging.  A
    # short grace period also lets asynchronous watchdog errors reach the log
    # before the run is accepted.
    time.sleep(1.0)
    wait_causal_capture_stop(args)
    with args.server_log.open("rb") as source:
        source.seek(initial_offset)
        full_delta = source.read().decode("utf-8", "replace")
        source.seek(measured_offset)
        measured_delta = source.read().decode("utf-8", "replace")
    (run_dir / "server_delta.log").write_text(full_delta)
    (run_dir / "measured_server_delta.log").write_text(measured_delta)

    expected_cached = args.prefix_len // args.page_size * args.page_size
    server_evidence = validate_measured_server_delta(
        measured_delta,
        expected_ranks=args.expected_ranks,
        requests_count=prompts,
        expected_cached_tokens=expected_cached,
        expected_scheduled_tokens=10_048,
        arm=args.arm,
    )
    total_input_tokens = prompts * (args.prefix_len + args.suffix_len)
    total_generated_tokens = prompts * args.output_len
    measured_metrics = {
        "successful_requests": prompts,
        "benchmark_duration_s": benchmark_duration_s,
        "total_input_tokens": total_input_tokens,
        "total_generated_tokens": total_generated_tokens,
        "request_throughput": prompts / benchmark_duration_s,
        "total_token_throughput": (
            total_input_tokens + total_generated_tokens
        )
        / benchmark_duration_s,
        "mean_ttft_ms": statistics.fmean(ttft_ms),
        "median_ttft_ms": statistics.median(ttft_ms),
        "p90_ttft_ms": percentile(ttft_ms, 0.90),
        "ttft_samples_ms": ttft_ms,
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "arm": args.arm,
        "run_dir": str(run_dir),
        "server_log": str(args.server_log),
        "request_mode": "concurrent_exact_input_ids",
        "parallelism_contract": {
            "tp": 8,
            "dp": 1,
            "attention_cp": 8,
            "attention_tp": 1,
            "ep": 8,
            "request_concurrency": args.x,
        },
        "workload_contract": {
            "requests": prompts,
            "logical_prefix_tokens": args.prefix_len,
            "page_aligned_cached_tokens": expected_cached,
            "suffix_tokens": args.suffix_len,
            "output_tokens": args.output_len,
            "scheduled_tokens_per_cp_rank": 10_048,
            "unique_suffix_first_token": True,
            "input_ids_sha256": input_sha256,
            "sentinel_ids_sha256": sentinel_sha256,
        },
        "warmup": {
            "input_tokens": len(prefix),
            "elapsed_ms": warmup_ms,
            "output_token_id": warmup_output_id,
            "excluded_from_measured_metrics": True,
        },
        "measured_metrics": measured_metrics,
        "output_token_ids": output_ids,
        "server_evidence": server_evidence,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
