"""Pure host contracts for GLM-5.2 FlashMLA scheduler row interpretation.

Both buckets consume SGLang's canonical request-major metadata tensor.  M32
changes only how the candidate main kernel maps its physical ``blockIdx.y`` to
one of those canonical rows; no framework-side tensor permutation is allowed.
"""

from __future__ import annotations

import hashlib
import json

NUM_SCHEDULER_ROWS = 148
NUM_PRODUCTIVE_ROWS = 128
SCHEDULER_CONTRACT_VERSION = "glm52-flashmla-inline-row-map-v1"

SCHEDULER_CONTRACT_BY_M: dict[int, dict[str, str]] = {
    16: {
        "scheduler_order": "request_major",
        "scheduler_mapping_location": "canonical_metadata",
        "scheduler_metadata_order": "request_major",
    },
    32: {
        "scheduler_order": "split_major_round_robin_inline",
        "scheduler_mapping_location": "main_kernel_physical_to_canonical_row",
        "scheduler_metadata_order": "request_major",
    },
}


def physical_to_canonical_rows(m: int) -> tuple[int, ...]:
    """Return the exact physical CTA row -> canonical metadata row mapping."""
    if m == 16:
        return tuple(range(NUM_SCHEDULER_ROWS))
    if m != 32:
        raise ValueError(f"unsupported GLM-5.2 FlashMLA M={m}")
    mapping = tuple(
        (
            (physical_row % 32) * 4 + physical_row // 32
            if physical_row < NUM_PRODUCTIVE_ROWS
            else physical_row
        )
        for physical_row in range(NUM_SCHEDULER_ROWS)
    )
    if set(mapping) != set(range(NUM_SCHEDULER_ROWS)):
        raise AssertionError("M32 inline scheduler row map is not a bijection")
    return mapping


def scheduler_permutation_sha256(m: int) -> str:
    """Hash compact JSON for the exact physical-to-canonical row mapping."""
    encoded = json.dumps(physical_to_canonical_rows(m), separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def scheduler_contract(m: int) -> dict[str, str]:
    """Return manifest/provider provenance for one host-known bucket."""
    return {
        **SCHEDULER_CONTRACT_BY_M[m],
        "scheduler_contract_version": SCHEDULER_CONTRACT_VERSION,
        "scheduler_permutation_sha256": scheduler_permutation_sha256(m),
    }
