#!/usr/bin/env bash
# Exact 8xB300 GLM-5.2 decode A/B for router padded-ID mask fusion.
# Reuses the serving-safe paired runner so stock and candidate keep identical
# real weights, 32K KV prefixes, global BS=128, and the existing winner stack.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LABELS=${LABELS:-"router_stock router_fused"}
export N_RUNS=${N_RUNS:-5}
export S=${S:-32768}
export GLOBAL_BS_LIST=${GLOBAL_BS_LIST:-"128"}
export OUT_LEN=${OUT_LEN:-240}
export RUN_ID=${RUN_ID:-router_pad_mask_b300_n${N_RUNS}_s${S}_$(date -u +%Y%m%dT%H%M%SZ)}

exec bash "$SCRIPT_DIR/run_shared_expert_swiglu_quant_b300_ab.sh"
