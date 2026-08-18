#!/usr/bin/env python3
"""Build a deterministic natural-token plan for fixed-KV decode series.

The source ShareGPT file is a large JSON array.  This script streams it, takes a
deterministic reservoir sample, tokenizes real conversations, and writes compact
little-endian uint32 streams.  Each request stream contains one 32K-class prefix
followed by one natural suffix block for every warmup, measured, and correctness
invocation.  Suffix blocks for a request have distinct first tokens so an earlier
invocation cannot turn a later one into a full-suffix cache hit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from array import array
from pathlib import Path
from typing import Any, Iterator


PLAN_FORMAT = "fixed-kv-natural-token-plan-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=32768)
    parser.add_argument("--prefix-len", type=int, default=32704)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-records", type=int, default=8192)
    return parser.parse_args()


def iter_json_array(path: Path, chunk_chars: int = 1 << 20) -> Iterator[Any]:
    """Yield one value at a time from a top-level JSON array."""

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    eof = False
    with path.open(encoding="utf-8") as handle:
        while True:
            if not eof and (position >= len(buffer) or len(buffer) - position < 4096):
                buffer = buffer[position:] + handle.read(chunk_chars)
                position = 0
                eof = handle.tell() == path.stat().st_size

            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"empty JSON input: {path}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"expected a top-level JSON array: {path}")
                started = True
                position += 1
                continue

            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"unterminated JSON array: {path}")
                continue

            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                buffer = buffer[position:] + handle.read(chunk_chars)
                position = 0
                eof = handle.tell() == path.stat().st_size
                continue
            yield value
            position = end


def reservoir_records(path: Path, count: int, seed: int) -> tuple[list[dict], int]:
    rng = random.Random(seed)
    selected: list[dict] = []
    observed = 0
    for observed, value in enumerate(iter_json_array(path), 1):
        if not isinstance(value, dict) or not isinstance(
            value.get("conversations"), list
        ):
            continue
        if len(selected) < count:
            selected.append(value)
            continue
        replacement = rng.randrange(observed)
        if replacement < count:
            selected[replacement] = value
    rng.shuffle(selected)
    return selected, observed


def normalized_messages(record: dict) -> list[dict[str, str]]:
    roles = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
    }
    messages = []
    for turn in record.get("conversations", []):
        if not isinstance(turn, dict):
            continue
        role = roles.get(str(turn.get("from", "")).lower())
        value = turn.get("value")
        if role is not None and isinstance(value, str) and value:
            messages.append({"role": role, "content": value})
    return messages


def tokenize_records(
    records: list[dict], model_path: Path, minimum_tokens: int
) -> tuple[array, list[str]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True
    )
    corpus = array("I")
    record_ids = []
    for record in records:
        messages = normalized_messages(record)
        if not messages:
            continue
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            token_ids = (
                encoded.input_ids if hasattr(encoded, "input_ids") else encoded
            )
        except (TypeError, ValueError):
            text = "\n".join(
                f"<{message['role']}>\n{message['content']}" for message in messages
            )
            token_ids = tokenizer(text, add_special_tokens=True).input_ids
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if not token_ids:
            continue
        corpus.extend(int(token_id) for token_id in token_ids)
        if tokenizer.eos_token_id is not None:
            corpus.append(int(tokenizer.eos_token_id))
        record_ids.append(str(record.get("id", "")))
        if len(corpus) >= minimum_tokens:
            break
    return corpus, record_ids


def build_plan(
    corpus: array,
    *,
    batch_size: int,
    prefix_len: int,
    suffix_len: int,
    sequence_count: int,
) -> tuple[array, int]:
    plan = array("I")
    cursor = 0
    skipped_suffix_blocks = 0
    for request_index in range(batch_size):
        prefix_end = cursor + prefix_len
        if prefix_end > len(corpus):
            raise ValueError(
                f"token corpus ended while building prefix {request_index}; "
                f"have {len(corpus)} tokens"
            )
        plan.extend(corpus[cursor:prefix_end])
        cursor = prefix_end

        used_first_tokens: set[int] = set()
        for sequence_index in range(sequence_count):
            while True:
                suffix_end = cursor + suffix_len
                if suffix_end > len(corpus):
                    raise ValueError(
                        "token corpus ended while selecting distinct natural suffixes; "
                        f"request={request_index} sequence={sequence_index} "
                        f"have={len(corpus)} need>{suffix_end}"
                    )
                first_token = int(corpus[cursor])
                suffix = corpus[cursor:suffix_end]
                cursor = suffix_end
                if first_token in used_first_tokens:
                    skipped_suffix_blocks += 1
                    continue
                used_first_tokens.add(first_token)
                plan.extend(suffix)
                break
    return plan, skipped_suffix_blocks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.runs <= 0 or args.warmup_runs < 0:
        raise SystemExit("batch-size/runs must be positive and warmup-runs non-negative")
    if not 0 < args.prefix_len < args.input_len:
        raise SystemExit("require 0 < prefix-len < input-len")
    if args.sample_records <= 0:
        raise SystemExit("sample-records must be positive")
    if sys.byteorder != "little":
        raise SystemExit("the v1 token-plan format requires a little-endian host")

    suffix_len = args.input_len - args.prefix_len
    sequence_count = args.warmup_runs + args.runs + 1
    stream_tokens = args.prefix_len + sequence_count * suffix_len
    expected_tokens = args.batch_size * stream_tokens
    records, observed_records = reservoir_records(
        args.dataset, args.sample_records, args.seed
    )
    # Leave generous slack for natural suffix chunks rejected because their
    # first token duplicates an earlier suffix for the same request.
    corpus, record_ids = tokenize_records(
        records, args.model_path, minimum_tokens=int(expected_tokens * 1.25)
    )
    plan, skipped_suffix_blocks = build_plan(
        corpus,
        batch_size=args.batch_size,
        prefix_len=args.prefix_len,
        suffix_len=suffix_len,
        sequence_count=sequence_count,
    )
    if len(plan) != expected_tokens:
        raise RuntimeError(f"plan has {len(plan)} tokens, expected {expected_tokens}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        plan.tofile(handle)
    plan_sha256 = sha256_file(args.output)
    metadata = {
        "format": PLAN_FORMAT,
        "dataset_path": str(args.dataset.resolve()),
        "dataset_bytes": args.dataset.stat().st_size,
        "model_path": str(args.model_path.resolve()),
        "seed": args.seed,
        "records_observed": observed_records,
        "records_sampled": len(records),
        "records_tokenized": len(record_ids),
        "sampled_record_ids_sha256": hashlib.sha256(
            json.dumps(record_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "token_corpus_size": len(corpus),
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "prefix_len": args.prefix_len,
        "suffix_len": suffix_len,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "sequence_count_including_correctness": sequence_count,
        "stream_tokens": stream_tokens,
        "plan_tokens": len(plan),
        "plan_bytes": args.output.stat().st_size,
        "plan_sha256": plan_sha256,
        "skipped_suffix_blocks_for_unique_first_token": skipped_suffix_blocks,
    }
    metadata_path = args.output.with_name(args.output.name + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
