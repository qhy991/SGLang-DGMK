#!/usr/bin/env bash
# Exact 8xB300 GLM-5.2 decode A/B for the post-MoE inter-layer megakernel.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LABELS=${LABELS:-"shared_nq_stock shared_nq_fused"}
export N_RUNS=${N_RUNS:-5}
export S=${S:-32768}
export GLOBAL_BS_LIST=${GLOBAL_BS_LIST:-"128"}
export OUT_LEN=${OUT_LEN:-240}
export RUN_ID=${RUN_ID:-shared_add_norm_quant_b300_n${N_RUNS}_s${S}_$(date -u +%Y%m%dT%H%M%SZ)}

exec bash "$SCRIPT_DIR/run_shared_expert_swiglu_quant_b300_ab.sh"
