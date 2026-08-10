"""Opt-in loader for the SGLang MoK prefill experiment."""

import os


if os.environ.get("MOK_SGLANG_PREFILL", "0") == "1":
    import mok_sglang_prefill_patch  # noqa: F401
