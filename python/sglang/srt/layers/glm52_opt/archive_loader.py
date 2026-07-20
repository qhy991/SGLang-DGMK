"""Load Kernel-Harness archive candidates by reference path."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from sglang.srt.layers.glm52_opt.config import archive_root

RunFn = Callable[[dict], object]


def _resolve_candidate_path(archive_ref: str) -> Path:
    root = archive_root()
    if archive_ref.endswith(".py"):
        path = root / archive_ref
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    cand = root / archive_ref / "candidate" / "candidate.py"
    if not cand.is_file():
        raise FileNotFoundError(cand)
    return cand


@lru_cache(maxsize=64)
def load_run_fn(archive_ref: str) -> RunFn:
    path = _resolve_candidate_path(archive_ref)
    name = f"glm52_archive_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import candidate {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    run = getattr(mod, "run", None)
    if run is None:
        raise AttributeError(f"{path} missing run()")
    return run
