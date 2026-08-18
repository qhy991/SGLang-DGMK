#!/usr/bin/env python3
"""Run a frozen, per-example GSM8K quality check against an SGLang server.

This intentionally reuses SGLang's built-in GSM8K evaluator and chat sampler,
but retains the per-question responses and scores needed for an A/B/A audit.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import site
import time
from pathlib import Path
from typing import Any

import sglang.test

# Some campaign worktrees intentionally carry a reduced sglang.test package
# that includes run_eval.py and simple_eval_common.py but omits the GSM8K leaf
# module.  Extend only that package's search path with the active interpreter's
# installed test directory.  The resolved files and hashes are persisted below.
for site_root in site.getsitepackages():
    installed_test = Path(site_root) / "sglang" / "test"
    if installed_test.is_dir() and str(installed_test) not in sglang.test.__path__:
        sglang.test.__path__.append(str(installed_test))

from sglang.test.simple_eval_common import ChatCompletionSampler
from sglang.test.simple_eval_gsm8k import (
    GSM8KEval,
    get_answer_value,
    get_one_example,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_record(obj: Any) -> dict[str, Any]:
    source = inspect.getsourcefile(obj)
    if source is None:
        raise RuntimeError(f"cannot resolve source for {obj!r}")
    path = Path(source).resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Offset into scored rows after the frozen few-shot prefix",
    )
    parser.add_argument("--num-threads", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    if (
        args.num_examples <= 0
        or args.num_shots < 0
        or args.start_offset < 0
        or args.num_threads <= 0
    ):
        raise SystemExit("example/thread counts must be positive and shots non-negative")
    if not args.data_path.is_file():
        raise SystemExit(f"missing GSM8K dataset: {args.data_path}")

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    eval_obj = GSM8KEval(
        num_examples=args.start_offset + args.num_examples,
        num_threads=args.num_threads,
        num_shots=args.num_shots,
        data_path=str(args.data_path),
    )
    eval_obj._lines = eval_obj._lines[args.start_offset :]
    if len(eval_obj._lines) != args.num_examples:
        raise SystemExit(
            f"requested {args.num_examples} examples, loaded {len(eval_obj._lines)}"
        )

    # Pin both states explicitly.  Omitting the template argument can inherit a
    # model-specific default-thinking policy and exhaust max_tokens before the
    # OpenAI-compatible response has a final ``content`` field.
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking}
    }
    sampler = ChatCompletionSampler(
        base_url=args.base_url.rstrip("/") + "/v1",
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        extra_body=extra_body,
    )

    start = time.perf_counter()
    result = eval_obj(sampler)
    elapsed = time.perf_counter() - start

    if len(result.convos) != args.num_examples:
        raise RuntimeError(
            f"expected {args.num_examples} conversations, got {len(result.convos)}"
        )

    samples = []
    for offset, (line, convo) in enumerate(zip(eval_obj._lines, result.convos)):
        dataset_index = args.num_shots + args.start_offset + offset
        response = ""
        if convo and isinstance(convo[-1], dict):
            response = str(convo[-1].get("content") or "")
        correct = get_answer_value(line["answer"])
        extracted = get_answer_value(response)
        prompt = eval_obj._build_prefix(offset) + get_one_example(
            eval_obj._lines, offset, include_answer=False
        )
        samples.append(
            {
                "ordinal": offset,
                "dataset_index": dataset_index,
                "question_sha256": hashlib.sha256(
                    line["question"].encode("utf-8")
                ).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "correct_answer": correct,
                "extracted_answer": extracted,
                "correct": bool(extracted == correct),
                "response_sha256": hashlib.sha256(
                    response.encode("utf-8")
                ).hexdigest(),
                "response": response,
            }
        )

    correct_count = sum(sample["correct"] for sample in samples)
    computed_score = correct_count / len(samples)
    if abs(float(result.score) - computed_score) > 1e-12:
        raise RuntimeError(
            f"built-in score {result.score} disagrees with retained score {computed_score}"
        )

    empty_responses = sum(not sample["response"] for sample in samples)
    payload = {
        "protocol": "sglang-builtin-gsm8k-retained-v1",
        "label": args.label,
        "dataset": {
            "path": str(args.data_path.resolve()),
            "size_bytes": args.data_path.stat().st_size,
            "sha256": _sha256(args.data_path),
            "num_shots": args.num_shots,
            "num_examples": args.num_examples,
            "start_offset": args.start_offset,
            "selected_dataset_indices": [
                sample["dataset_index"] for sample in samples
            ],
        },
        "request": {
            "base_url": args.base_url,
            "model": sampler.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "num_threads": args.num_threads,
            "enable_thinking": args.enable_thinking,
        },
        "provenance": {
            "GSM8KEval": _module_record(GSM8KEval),
            "ChatCompletionSampler": _module_record(ChatCompletionSampler),
        },
        "metrics": {
            "score": computed_score,
            "correct": correct_count,
            "total": len(samples),
            "latency_s": elapsed,
            "completion_tokens": sum(sampler._completion_tokens),
            "output_throughput_token_s": (
                sum(sampler._completion_tokens) / elapsed if elapsed > 0 else None
            ),
            "empty_responses": empty_responses,
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "label": args.label,
                "score": computed_score,
                "correct": correct_count,
                "total": len(samples),
                "latency_s": elapsed,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    if empty_responses:
        raise SystemExit(
            f"quality gate invalid: {empty_responses}/{len(samples)} empty responses"
        )


if __name__ == "__main__":
    main()
