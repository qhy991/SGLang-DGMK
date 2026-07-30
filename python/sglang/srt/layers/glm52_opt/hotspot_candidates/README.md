# GLM-5.2 FlashMLA hotspot candidates (vendored into SGLang)

Default-off external-acceptance survivors for `flashmla_kv` sparse FP8 decode:

| Binary | Role | Local graph vs stock |
| --- | --- | --- |
| `combine_c2_bucket_stages` `.so` | **P1 main + combine_c2** (preferred) | M16 ~1.27×, M32 ~1.14× |
| `p1_consumer_scale` `.so` | P1 main + stock combine | M16/M32 ~1.06× |

Both cubins are **sm_100 / sm_100f**, built with **CUDA 13.2**. On CUDA 13.1 hosts
that cannot JIT `cvt.rn.bf16x2.e4m3x2`, load the prebuilt `.so` (see below).

Production default stays **off**. TP8/DP8/EP8 acceptance is still required for
`production-win`.

## Enable (stack winner)

```bash
REPO_ROOT="$(python -c 'import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[3])')"
# editable install: prefer the source tree path of this package instead
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
export SGLANG_GLM52_HOTSPOT_MODULE="$HOTSPOT/flashmla_combine_decode_provider.py"
export GLM52_FLASHMLA_COMBINE_VARIANT=combine_c2_bucket_stages
export GLM52_FLASHMLA_USE_PREBUILT=1
export SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=1
```

`--dsa-decode-backend flashmla_kv` is mandatory on the serve launch.
