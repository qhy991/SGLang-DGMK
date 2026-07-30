# GLM-5.2 FlashMLA hotspot candidates (vendored into SGLang)

Default-off external-acceptance survivors (CUDA 13.2 / sm_100 prebuilts):

| Binary | Role | Local graph |
| --- | --- | --- |
| `stack_r2a…__combine_c2…` `.so` | **Preferred decode:** r2a + c2 | vs P1+c2: M16 ~1.09×, M32 ~flat |
| `combine_c2_bucket_stages` `.so` | Decode P1 main + combine_c2 | vs stock: M16 ~1.27×, M32 ~1.14× |
| `p1_consumer_scale` `.so` | Decode P1 main + stock combine | vs stock: M16/M32 ~1.06× |
| `dsa_prefill … b3_b5_native_exact` `.so` | **Prefill** selected identity | vs stock: M1024–4096 ~1.05–1.10× |

See also `glm52_opt/hotspot_accel_enable.md` for fixed-N/K GEMMs (`o_proj`,
`fused_qkv_a_proj`, `index_q_upproj`).

Production default stays **off**. TP8/DP8/EP8 acceptance is still required for
`production-win`.

## Enable decode stack

```bash
HOTSPOT="$(python - <<'PY'
from pathlib import Path
import sglang.srt.layers.glm52_opt.hotspot_candidates as m
print(Path(m.__file__).resolve().parent)
PY
)"

export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode
export SGLANG_GLM52_OPT_M_BUCKETS='dsa_decode_attn:16|32'
# Preferred: r2a + c2 (M16 win; M32 neutral)
export SGLANG_GLM52_HOTSPOT_MODULE="$HOTSPOT/flashmla_stack_r2a_c2_provider.py"
export GLM52_FLASHMLA_USE_PREBUILT=1
# Or prior P1+c2 vs stock:
# export SGLANG_GLM52_HOTSPOT_MODULE="$HOTSPOT/flashmla_combine_decode_provider.py"
# export GLM52_FLASHMLA_COMBINE_VARIANT=combine_c2_bucket_stages
```

`--dsa-decode-backend flashmla_kv` is mandatory.

## Enable prefill

```bash
export SGLANG_GLM52_OPT_OPS=dsa_prefill_attn
export SGLANG_GLM52_OPT_M_BUCKETS='dsa_prefill_attn:1024|2048|4096'
export SGLANG_GLM52_HOTSPOT_MODULE="$HOTSPOT/flashmla_sparse_prefill_provider.py"
export GLM52_DSA_PREFILL_VARIANT=b3_b5_native_exact
export GLM52_DSA_PREFILL_USE_PREBUILT=1
```

`--dsa-prefill-backend flashmla_kv` is mandatory.
