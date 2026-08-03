#!/usr/bin/env python3
"""Measure repeated decode batches while reusing one audited KV-cache build.

The upstream one-batch benchmark intentionally flushes and rebuilds the radix
cache for every case.  That is useful for independent samples, but prohibitively
expensive for a 32K-prompt, global-BS=128 decode campaign and it mixes cache-build
drift into a sub-percent decode comparison.

This runner instead performs exactly one flush and one prefix warmup per server.
Every following request reuses the same ``prefix_len`` tokens and appends a
different deterministic suffix.  A unique first suffix token prevents a prior
measured request from turning the next request into a 100%-cache-hit replay.
Prometheus counters are checked around every request, so the claimed KV state is
part of the result rather than an assumption.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import time
from array import array
from pathlib import Path
from typing import Any

import requests


DEFAULT_TIMEOUT_SECONDS = 7200
PROTOCOL_VERSION = "fixed-kv-decode-series-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--result-filename", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=32768)
    parser.add_argument("--prefix-len", type=int, default=32704)
    parser.add_argument("--output-len", type=int, default=48)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Decode-shaped samples after prefix warmup that are recorded only in a sidecar.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=0,
        help="Read model config.json when zero.",
    )
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument(
        "--token-plan",
        type=Path,
        help=(
            "optional fixed-kv-natural-token-plan-v1 uint32 file; when set, "
            "prefixes and every suffix come from its recorded dataset tokens"
        ),
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--cache-hit-tolerance",
        type=float,
        default=0.001,
        help="Maximum absolute deviation from prefix_len/input_len.",
    )
    return parser.parse_args()


def read_vocab_size(model_path: Path, explicit: int) -> int:
    if explicit > 0:
        return explicit
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text())
        vocab_size = int(config["vocab_size"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"cannot determine vocab size from {config_path}; pass --vocab-size: {exc}"
        ) from exc
    if vocab_size <= 0:
        raise SystemExit(f"invalid vocab_size={vocab_size} in {config_path}")
    return vocab_size


def validate_args(args: argparse.Namespace, vocab_size: int) -> None:
    if args.batch_size <= 0 or args.runs <= 0 or args.output_len <= 0:
        raise SystemExit("batch-size, runs, and output-len must be positive")
    if args.warmup_runs < 0:
        raise SystemExit("warmup-runs must be non-negative")
    if not 0 < args.prefix_len < args.input_len:
        raise SystemExit("require 0 < prefix-len < input-len")
    if args.runs + args.warmup_runs >= vocab_size:
        raise SystemExit(
            "runs + warmup-runs must be smaller than vocab-size so each suffix "
            "can start with a distinct token"
        )
    if not 0 < args.cache_hit_tolerance < 0.1:
        raise SystemExit("cache-hit-tolerance must be in (0, 0.1)")


def metrics_cache_tokens(
    session: requests.Session, base_url: str, timeout: int
) -> tuple[float, float] | None:
    """Return summed (cached, prompt) counters from all exposed ranks."""

    try:
        response = session.get(f"{base_url}/metrics", timeout=min(timeout, 30))
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] metrics snapshot failed: {exc}", flush=True)
        return None

    cached = 0.0
    prompt = 0.0
    for line in response.text.splitlines():
        if line.startswith("sglang:cached_tokens_total{"):
            match = re.search(r"\}\s+([\d.eE+-]+)$", line)
            if match:
                cached += float(match.group(1))
        elif line.startswith("sglang:prompt_tokens_total{"):
            match = re.search(r"\}\s+([\d.eE+-]+)$", line)
            if match:
                prompt += float(match.group(1))
    return cached, prompt


def cache_hit_delta(
    before: tuple[float, float] | None,
    after: tuple[float, float] | None,
) -> float | None:
    if before is None or after is None:
        return None
    cached_delta = after[0] - before[0]
    prompt_delta = after[1] - before[1]
    if prompt_delta <= 0:
        return None
    return cached_delta / prompt_delta


def flush_cache(session: requests.Session, base_url: str, timeout: int) -> None:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = session.post(f"{base_url}/flush_cache", timeout=timeout)
            response.raise_for_status()
            print(f"[VALID] prefix cache flushed attempt={attempt}", flush=True)
            return
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2)
    raise RuntimeError(f"failed to flush prefix cache: {last_error}")


def build_base_prefixes(
    *, batch_size: int, prefix_len: int, vocab_size: int, seed: int
) -> tuple[list[list[int]], list[int]]:
    """Build deterministic random-id-style prefixes with distinct offsets."""

    rng = random.Random(seed)
    offsets = rng.sample(range(vocab_size), batch_size)
    prefixes = [
        [int((offset + request_index + position) % vocab_size) for position in range(prefix_len)]
        for request_index, offset in enumerate(offsets)
    ]
    return prefixes, offsets


def build_suffix(
    *,
    request_index: int,
    sequence_index: int,
    suffix_len: int,
    vocab_size: int,
    seed: int,
) -> list[int]:
    # For any one cached base prefix, sequence_index maps to a distinct first
    # token.  Therefore a previous sample cannot cache the current suffix.
    branch_origin = (seed * 8191 + 17) % vocab_size
    first = (branch_origin + sequence_index) % vocab_size
    suffix = [first]
    for position in range(1, suffix_len):
        suffix.append(
            int((first + position * 104729 + request_index * 8191) % vocab_size)
        )
    return suffix


def protocol_id(
    args: argparse.Namespace,
    vocab_size: int,
    sequence_index: int,
    input_source_id: str,
) -> str:
    spec = {
        "protocol": PROTOCOL_VERSION,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "prefix_len": args.prefix_len,
        "output_len": args.output_len,
        "vocab_size": vocab_size,
        "sequence_index": sequence_index,
        "input_source_id": input_source_id,
    }
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_token_plan(
    path: Path,
    args: argparse.Namespace,
    vocab_size: int,
) -> tuple[list[list[int]], array, dict[str, Any]]:
    metadata_path = path.with_name(path.name + ".json")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "format": "fixed-kv-natural-token-plan-v1",
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "prefix_len": args.prefix_len,
        "suffix_len": args.input_len - args.prefix_len,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "sequence_count_including_correctness": args.warmup_runs + args.runs + 1,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"token plan {metadata_path} has {key}={metadata.get(key)!r}; "
                f"expected {value!r}"
            )
    actual_sha256 = sha256_file(path)
    if metadata.get("plan_sha256") != actual_sha256:
        raise RuntimeError(
            f"token plan hash mismatch: {actual_sha256} != "
            f"{metadata.get('plan_sha256')}"
        )
    expected_tokens = args.batch_size * int(metadata["stream_tokens"])
    if path.stat().st_size != expected_tokens * 4:
        raise RuntimeError(
            f"token plan byte size {path.stat().st_size} != {expected_tokens * 4}"
        )
    tokens = array("I")
    with path.open("rb") as handle:
        tokens.fromfile(handle, expected_tokens)
    if len(tokens) != expected_tokens:
        raise RuntimeError(
            f"token plan has {len(tokens)} uint32 values, expected {expected_tokens}"
        )
    if tokens and max(tokens) >= vocab_size:
        raise RuntimeError(
            f"token plan contains token {max(tokens)} outside vocab_size={vocab_size}"
        )
    stream_tokens = int(metadata["stream_tokens"])
    prefixes = [
        list(
            tokens[
                request_index * stream_tokens : request_index * stream_tokens
                + args.prefix_len
            ]
        )
        for request_index in range(args.batch_size)
    ]
    return prefixes, tokens, metadata


def planned_suffix(
    tokens: array,
    metadata: dict[str, Any],
    *,
    request_index: int,
    sequence_index: int,
) -> list[int]:
    stream_tokens = int(metadata["stream_tokens"])
    prefix_len = int(metadata["prefix_len"])
    suffix_len = int(metadata["suffix_len"])
    start = (
        request_index * stream_tokens
        + prefix_len
        + sequence_index * suffix_len
    )
    return list(tokens[start : start + suffix_len])


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def warm_prefixes(
    session: requests.Session,
    base_url: str,
    prefixes: list[list[int]],
    timeout: int,
) -> float:
    payload = {
        "input_ids": prefixes,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 1,
            "ignore_eos": True,
        },
        "stream": False,
    }
    start = time.perf_counter()
    response = session.post(f"{base_url}/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    elapsed = time.perf_counter() - start
    print(
        f"[VALID] warmed {len(prefixes)} prefixes x {len(prefixes[0])} tokens "
        f"in {elapsed:.3f}s",
        flush=True,
    )
    return elapsed


def run_decode_batch(
    *,
    session: requests.Session,
    base_url: str,
    input_ids: list[list[int]],
    output_len: int,
    stream_interval: int,
    timeout: int,
) -> tuple[float, float, float | None, dict[str, Any]]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
            "stream_interval": stream_interval,
        },
        "return_logprob": False,
        "stream": True,
    }
    metrics_before = metrics_cache_tokens(session, base_url, timeout)
    start = time.perf_counter()
    last_ttft = 0.0
    with session.post(
        f"{base_url}/generate",
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for raw_chunk in response.iter_lines(decode_unicode=False):
            chunk = raw_chunk.decode("utf-8")
            if not chunk or not chunk.startswith("data:"):
                continue
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip())
            if "error" in data:
                raise RuntimeError(f"generation failed: {data}")
            meta = data.get("meta_info", {})
            finish_reason = meta.get("finish_reason")
            if finish_reason is not None and finish_reason.get("type") != "length":
                raise RuntimeError(f"unexpected finish_reason: {finish_reason}")
            if meta.get("completion_tokens") == 1:
                # The batch response emits one first-token event per request;
                # overwriting this timestamp gives TTFT of the final request.
                last_ttft = time.perf_counter() - start
    latency = time.perf_counter() - start
    if last_ttft <= 0 or latency <= last_ttft:
        raise RuntimeError(
            f"invalid streamed timing: latency={latency}, last_ttft={last_ttft}"
        )
    metrics_after = metrics_cache_tokens(session, base_url, timeout)
    hit_rate = cache_hit_delta(metrics_before, metrics_after)

    server_info: dict[str, Any] = {}
    try:
        info_response = session.get(f"{base_url}/server_info", timeout=min(timeout, 30))
        info_response.raise_for_status()
        server_info = info_response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[WARN] server_info failed: {exc}", flush=True)
    return latency, last_ttft, hit_rate, server_info


def run_correctness_probe(
    *,
    session: requests.Session,
    base_url: str,
    input_ids: list[list[int]],
    output_len: int,
    timeout: int,
) -> dict[str, Any]:
    """Generate one untimed batch and hash every greedy output token ID."""

    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "top_logprobs_num": 0,
        "stream": False,
    }
    response = session.post(f"{base_url}/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    records = body if isinstance(body, list) else [body]
    if len(records) != len(input_ids):
        raise RuntimeError(
            f"correctness probe expected {len(input_ids)} responses, got "
            f"{len(records)}"
        )

    output_token_ids: list[list[int]] = []
    for index, record in enumerate(records):
        ids = record.get("output_ids")
        if ids is None:
            logprobs = record.get("meta_info", {}).get("output_token_logprobs")
            if logprobs is not None:
                ids = [entry[1] for entry in logprobs]
        if ids is None:
            raise RuntimeError(
                f"correctness probe response {index} has no output token IDs"
            )
        normalized = [int(token_id) for token_id in ids]
        if len(normalized) != output_len:
            raise RuntimeError(
                f"correctness probe response {index} has {len(normalized)} "
                f"tokens, expected {output_len}"
            )
        output_token_ids.append(normalized)

    canonical = json.dumps(output_token_ids, separators=(",", ":")).encode()
    return {
        "batch_size": len(output_token_ids),
        "output_len": output_len,
        "output_token_count": sum(len(row) for row in output_token_ids),
        "output_token_ids_sha256": hashlib.sha256(canonical).hexdigest(),
        "per_request_output_token_ids_sha256": [
            hashlib.sha256(
                json.dumps(row, separators=(",", ":")).encode()
            ).hexdigest()
            for row in output_token_ids
        ],
        # 128*48 integer IDs are small enough to retain, and are necessary to
        # distinguish one-token numerical drift from a wholesale wrong-route
        # failure when hashes differ across independently launched services.
        "output_token_ids": output_token_ids,
    }


def last_server_metrics(server_info: dict[str, Any]) -> tuple[float, float]:
    states = server_info.get("internal_states") or []
    state = states[0] if states else {}
    throughput = state.get("last_gen_throughput") or -1.0
    accept_len = state.get("avg_spec_accept_length") or -1.0
    return float(throughput), float(accept_len)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    output_path = Path(args.result_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warmup_path = output_path.with_name(output_path.name + ".warmup.jsonl")
    protocol_path = output_path.with_name(output_path.name + ".protocol.json")
    correctness_path = output_path.with_name(
        output_path.name + ".correctness.json"
    )
    output_path.write_text("")
    warmup_path.write_text("")

    base_url = args.base_url.rstrip("/")
    vocab_size = read_vocab_size(model_path, args.vocab_size)
    validate_args(args, vocab_size)
    target_hit_rate = args.prefix_len / float(args.input_len)
    suffix_len = args.input_len - args.prefix_len
    token_plan: array | None = None
    token_plan_metadata: dict[str, Any] | None = None
    if args.token_plan is not None:
        prefixes, token_plan, token_plan_metadata = load_token_plan(
            args.token_plan.resolve(), args, vocab_size
        )
        offsets: list[int] = []
        input_source = "sharegpt-token-plan"
        input_source_id = (
            f"{token_plan_metadata['format']}:"
            f"{token_plan_metadata['plan_sha256']}"
        )
    else:
        prefixes, offsets = build_base_prefixes(
            batch_size=args.batch_size,
            prefix_len=args.prefix_len,
            vocab_size=vocab_size,
            seed=args.seed,
        )
        input_source = "deterministic-random-ids"
        input_source_id = "deterministic-random-ids-v1"
    protocol = {
        "protocol": PROTOCOL_VERSION,
        "label": args.label,
        "base_url": base_url,
        "model_path": str(model_path),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "prefix_len": args.prefix_len,
        "uncached_tokens_per_request": suffix_len,
        "output_len": args.output_len,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "vocab_size": vocab_size,
        "target_cache_hit_rate": target_hit_rate,
        "cache_hit_tolerance": args.cache_hit_tolerance,
        "input_source": input_source,
        "input_source_id": input_source_id,
        "base_offsets_sha256": (
            hashlib.sha256(
                json.dumps(offsets, separators=(",", ":")).encode()
            ).hexdigest()
            if offsets
            else None
        ),
        "token_plan": (
            {
                "path": str(args.token_plan.resolve()),
                "metadata_path": str(
                    args.token_plan.resolve().with_name(
                        args.token_plan.name + ".json"
                    )
                ),
                "format": token_plan_metadata["format"],
                "plan_sha256": token_plan_metadata["plan_sha256"],
                "sampled_record_ids_sha256": token_plan_metadata[
                    "sampled_record_ids_sha256"
                ],
                "records_tokenized": token_plan_metadata["records_tokenized"],
            }
            if token_plan_metadata is not None
            else None
        ),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True), flush=True)

    with requests.Session() as session:
        flush_cache(session, base_url, args.timeout)
        warm_elapsed = warm_prefixes(session, base_url, prefixes, args.timeout)

        measured_index = 0
        total_sequences = args.warmup_runs + args.runs
        for sequence_index in range(total_sequences):
            is_warmup = sequence_index < args.warmup_runs
            if not is_warmup:
                measured_index += 1
            inputs = []
            for request_index, prefix in enumerate(prefixes):
                suffix = (
                    planned_suffix(
                        token_plan,
                        token_plan_metadata,
                        request_index=request_index,
                        sequence_index=sequence_index,
                    )
                    if token_plan is not None
                    and token_plan_metadata is not None
                    else build_suffix(
                        request_index=request_index,
                        sequence_index=sequence_index,
                        suffix_len=suffix_len,
                        vocab_size=vocab_size,
                        seed=args.seed,
                    )
                )
                inputs.append(prefix + suffix)
            latency, last_ttft, hit_rate, server_info = run_decode_batch(
                session=session,
                base_url=base_url,
                input_ids=inputs,
                output_len=args.output_len,
                stream_interval=args.stream_interval,
                timeout=args.timeout,
            )
            if hit_rate is None:
                raise RuntimeError("cache-hit metrics unavailable for an audited fixed-KV run")
            hit_error = abs(hit_rate - target_hit_rate)
            if hit_error > args.cache_hit_tolerance:
                raise RuntimeError(
                    f"cache-hit rate {hit_rate:.6f} is not the requested fixed-KV "
                    f"state {target_hit_rate:.6f} +/- {args.cache_hit_tolerance:.6f}"
                )

            itl_ms = (latency - last_ttft) / args.output_len * 1000.0
            last_gen_throughput, accept_len = last_server_metrics(server_info)
            run_number = sequence_index + 1 if is_warmup else measured_index
            run_name = (
                f"{args.label}_decode_bs{args.batch_size}_warmup{run_number}"
                if is_warmup
                else f"{args.label}_decode_bs{args.batch_size}_i{run_number}"
            )
            row = {
                "protocol": PROTOCOL_VERSION,
                "input_source": input_source,
                "input_source_id": input_source_id,
                "run_name": run_name,
                "label": args.label,
                "is_warmup": is_warmup,
                "sequence_index": sequence_index,
                "measured_index": None if is_warmup else measured_index,
                "prompt_set_id": protocol_id(
                    args, vocab_size, sequence_index, input_source_id
                ),
                "batch_size": args.batch_size,
                "input_len": args.input_len,
                "prefix_len": args.prefix_len,
                "uncached_tokens_per_request": suffix_len,
                "output_len": args.output_len,
                "latency": latency,
                "last_ttft": last_ttft,
                "itl_ms": itl_ms,
                "input_throughput": args.batch_size * args.input_len / last_ttft,
                "output_throughput": args.batch_size
                * args.output_len
                / (latency - last_ttft),
                "overall_throughput": args.batch_size
                * (args.input_len + args.output_len)
                / latency,
                "last_gen_throughput": last_gen_throughput,
                "acc_length": accept_len,
                "cache_hit_rate": hit_rate,
                "cache_hit_rate_target": target_hit_rate,
                "cache_hit_rate_error": hit_error,
                "prefix_warmup_elapsed_s": warm_elapsed,
            }
            write_jsonl(warmup_path if is_warmup else output_path, row)
            phase = "WARMUP" if is_warmup else "RUN"
            print(
                f"[{phase}] {run_name} sequence={sequence_index}/{total_sequences - 1} "
                f"latency={latency:.6f}s last_ttft={last_ttft:.6f}s "
                f"itl={itl_ms:.6f}ms cache_hit={hit_rate:.6f}",
                flush=True,
            )
            del inputs
            gc.collect()

        # Keep output validation outside the timed series.  It uses the next
        # deterministic suffix, so it neither replays a measured suffix nor
        # perturbs any recorded latency.  All A/B/A arms must reproduce all
        # 128*output_len greedy token IDs exactly.
        correctness_sequence_index = total_sequences
        correctness_inputs = []
        for request_index, prefix in enumerate(prefixes):
            suffix = (
                planned_suffix(
                    token_plan,
                    token_plan_metadata,
                    request_index=request_index,
                    sequence_index=correctness_sequence_index,
                )
                if token_plan is not None and token_plan_metadata is not None
                else build_suffix(
                    request_index=request_index,
                    sequence_index=correctness_sequence_index,
                    suffix_len=suffix_len,
                    vocab_size=vocab_size,
                    seed=args.seed,
                )
            )
            correctness_inputs.append(prefix + suffix)
        correctness = run_correctness_probe(
            session=session,
            base_url=base_url,
            input_ids=correctness_inputs,
            output_len=args.output_len,
            timeout=args.timeout,
        )
        correctness.update(
            protocol=PROTOCOL_VERSION,
            input_source=input_source,
            input_source_id=input_source_id,
            label=args.label,
            sequence_index=correctness_sequence_index,
            prompt_set_id=protocol_id(
                args,
                vocab_size,
                correctness_sequence_index,
                input_source_id,
            ),
        )
        correctness_path.write_text(
            json.dumps(correctness, indent=2, sort_keys=True) + "\n"
        )
        print(
            "[VALID] correctness probe "
            f"tokens={correctness['output_token_count']} "
            f"sha256={correctness['output_token_ids_sha256']}",
            flush=True,
        )

    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line]
    if len(rows) != args.runs:
        raise RuntimeError(f"expected {args.runs} measured rows, found {len(rows)}")
    print(
        f"[DONE] label={args.label} measured={len(rows)} output={output_path} "
        f"warmups={args.warmup_runs} warmup_output={warmup_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
