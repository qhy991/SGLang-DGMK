# GLM-5.2 accelerating hotspots — enable in SGLang

Default **off**. Local graph wins registered on branch
`goal/glm52-hotspot-accel-bundle`. TP8/DP8/EP8 still required for production-on.

> **E2E decode TPOT（OPT0 vs winners）**：见
> [`DECODE_WINNERS_E2E_TPOT.md`](DECODE_WINNERS_E2E_TPOT.md)
> （分支 `docs/glm52-decode-winners-e2e-tpot`：只用有 e2e 收益的算子；
> global BS=128/256，N=100，median ITL）。

## Fixed-N/K decode GEMMs (`e2e_candidates`)

| Op | Graph vs stock (approx) | Notes |
|---|---|---|
| `o_proj` | M16 ~1.39–1.44×, M32 ~1.06–1.08× | `compiled_dims=nk`, graph-only |
| `fused_qkv_a_proj` | M16/M32 ~1.28–1.35× leaf | graph-only |
| `index_q_upproj` | M16 ~1.22×, M32 ~1.19× leaf | graph-only |

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=e2e_candidates
export SGLANG_GLM52_OPT_OPS=o_proj,fused_qkv_a_proj,index_q_upproj
export SGLANG_GLM52_OPT_M_BUCKETS='o_proj:16|32,fused_qkv_a_proj:16|32,index_q_upproj:16|32'
```

## FlashMLA decode + prefill (`hotspot_candidates`)

| Op | Identity | Graph |
|---|---|---|
| `dsa_decode_attn` | **preferred:** `r2a` + `combine_c2` | vs P1+c2: M16 ~1.09×, M32 ~flat |
| `dsa_decode_attn` | baseline: P1 + `combine_c2` | vs stock: M16 ~1.27×, M32 ~1.14× |
| `dsa_prefill_attn` | `b3_b5_native_exact` | vs stock: M1024–4096 ~1.05–1.10× |

Single-bucket wins are intentionally registered (M16 decode r2a).

```bash
HOTSPOT="$(python - <<'PY'
from pathlib import Path
import sglang.srt.layers.glm52_opt.hotspot_candidates as m
print(Path(m.__file__).resolve().parent)
PY
)"

# Decode preferred stack (r2a main + c2 combine)
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode
export SGLANG_GLM52_OPT_M_BUCKETS='dsa_decode_attn:16|32'
export SGLANG_GLM52_HOTSPOT_MODULE="$HOTSPOT/flashmla_stack_r2a_c2_provider.py"
export GLM52_FLASHMLA_USE_PREBUILT=1
# serve: --dsa-decode-backend flashmla_kv

# Optional: prior P1+c2-only stack (vs stock)
# export SGLANG_GLM52_HOTSPOT_MODULE="$HOTSPOT/flashmla_combine_decode_provider.py"
# export GLM52_FLASHMLA_COMBINE_VARIANT=combine_c2_bucket_stages

# Prefill (separate process / OPT_OPS swap)
export SGLANG_GLM52_OPT_OPS=dsa_prefill_attn
export SGLANG_GLM52_OPT_M_BUCKETS='dsa_prefill_attn:1024|2048|4096'
export SGLANG_GLM52_HOTSPOT_MODULE="$HOTSPOT/flashmla_sparse_prefill_provider.py"
export GLM52_DSA_PREFILL_VARIANT=b3_b5_native_exact
export GLM52_DSA_PREFILL_USE_PREBUILT=1
# serve: --dsa-prefill-backend flashmla_kv
```

## MoE (existing hotspot hooks)

- `moe_gate_proj` / fused W13 BM16: still external-acceptance via
  `SGLANG_GLM52_HOTSPOT_MODULE` pointing at the W13 provider (see
  `goal/glm52-hotspot-moe-w13-decode`).
- `moe_down_proj` W2 graph-only BM16: leaf win but region historically failed
  the 1.03 gate; keep default off unless you accept ~1–3% region uncertainty.

## Not registered (no usable ≥1% SGLang path this round)

- `index_score` prefill/decode PTX (~1.027× leaf, region diluted)
- MoE SwiGLU region rescue (leaf large, region Amdahl-capped ~1.02×)
