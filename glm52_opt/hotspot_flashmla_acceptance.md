# GLM-5.2 FlashMLA hotspot — enable for external acceptance

Local disposition: **external-acceptance-candidate** (default **off**).

| Candidate | Graph containing vs stock |
| --- | ---: |
| P1 main only (`p1_consumer_scale`) | M16 **1.069–1.073×**, M32 **1.064–1.067×** |
| **P1 + combine_c2** (preferred) | M16 **1.271–1.275×**, M32 **1.138–1.143×** |

Graph-only dispatch is already in this branch (`d7fe89a71` lineage): eager falls
back to stock. Prebuilt `.so` files live under
`python/sglang/srt/layers/glm52_opt/hotspot_candidates/prebuilt/` (CUDA 13.2 /
sm_100). Use them on CUDA 13.1 hosts that cannot JIT tip PTX.

## Preferred enable (stack)

```bash
HOTSPOT="$(python - <<'PY'
from pathlib import Path
import sglang.srt.layers.glm52_opt.hotspot_candidates as pkg
print(Path(pkg.__file__).resolve().parent if hasattr(pkg, "__file__") else "")
PY
)"
# fallback if package __init__ absent:
HOTSPOT="${HOTSPOT:-$PWD/python/sglang/srt/layers/glm52_opt/hotspot_candidates}"

export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode
export SGLANG_GLM52_OPT_M_BUCKETS='dsa_decode_attn:16|32'
export SGLANG_GLM52_HOTSPOT_MODULE="$HOTSPOT/flashmla_combine_decode_provider.py"
export GLM52_FLASHMLA_COMBINE_VARIANT=combine_c2_bucket_stages
export GLM52_FLASHMLA_USE_PREBUILT=1
export SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=1
```

Serve with `--dsa-decode-backend flashmla_kv` and the same TP/DP/EP topology as
the reference arm. See `hotspot_candidates/MANIFEST.json` for sha256 pins.
