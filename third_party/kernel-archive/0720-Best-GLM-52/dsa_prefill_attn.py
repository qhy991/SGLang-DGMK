"""GLM-5.2 DSA Sparse Attention (prefill) — the one file to edit for this task.

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=1024:

    q                (1024, 64, 576)          torch.bfloat16
    kv               (65536, 1, 576)          torch.bfloat16
    indices          (1024, 1, 2048)          torch.int32

Return the output. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 5e-06. `./run.sh --describe` prints all of it.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

from sgl_kernel.flash_mla import flash_mla_sparse_fwd


def run(inputs: dict):
    # Starting point: the reference call itself — correct, speedup ~1.0. Replace it.
    return flash_mla_sparse_fwd(inputs["q"], inputs["kv"], inputs["indices"],
                                inputs["sm_scale"], inputs["d_v"])
