#!/usr/bin/env python3
"""Send a deterministic 8 x 4096-token greedy batch to an SGLang server."""

from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000/generate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prompts = [
        [1000 + ((request_id * 97 + position * 13) % 10000) for position in range(4096)]
        for request_id in range(8)
    ]
    payload = {
        "input_ids": prompts,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 1,
        },
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        result = json.load(response)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    items = result if isinstance(result, list) else [result]
    output_ids = [item.get("output_ids") for item in items]
    print(json.dumps({"count": len(items), "output_ids": output_ids}))


if __name__ == "__main__":
    main()
