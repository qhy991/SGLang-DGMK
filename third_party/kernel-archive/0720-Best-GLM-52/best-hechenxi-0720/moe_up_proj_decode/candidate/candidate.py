from __future__ import annotations
import os, warnings
import torch
import deep_gemm
_HERE = os.path.dirname(os.path.abspath(__file__))
_pack = None
try:
    from torch.utils.cpp_extension import load as _load_ext
    _ext = _load_ext(name="glm52_moe_up_prepack_wo", sources=[os.path.join(_HERE,"scale_pack.cu")], verbose=False)
    _pack = _ext
except Exception as _e:
    _pack = None
    warnings.warn("ext build failed: "+str(_e), RuntimeWarning)

def _pack_scales_torch(x_scale, w_scale):
    from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor as _packt
    n = w_scale.shape[1]*128
    row = torch.arange(n, device=w_scale.device)//128
    return _packt(x_scale), _packt(w_scale.index_select(-2, row))

def _get_pdl():
    try: return deep_gemm.get_pdl()
    except Exception: return None
def _set_pdl(v):
    if v is None: return
    try: deep_gemm.set_pdl(v)
    except Exception: pass

_CACHE = {}

@torch.no_grad()
def run(inputs: dict):
    x_fp8=inputs["x_fp8"]; w_fp8=inputs["w_fp8"]
    x_scale=inputs["x_scale"]; w_scale=inputs["w_scale"]; out=inputs["out"]
    key=(x_fp8.shape[0], out.shape[1], x_fp8.shape[2], x_fp8.device.index)
    wp=_CACHE.get(key)
    if wp is None:
        if _pack is not None: _xp0,wp=_pack.pack_scales(x_scale,w_scale)
        else: _xp0,wp=_pack_scales_torch(x_scale,w_scale)
        wp=wp.clone(); _CACHE[key]=wp
    if _pack is not None: xp=_pack.pack_x_scale(x_scale)
    else: xp,_wp0=_pack_scales_torch(x_scale,w_scale)
    saved=_get_pdl()
    try:
        _set_pdl(True)
        deep_gemm.fp8_m_grouped_gemm_nt_masked((x_fp8,xp),(w_fp8,wp),out,inputs["masked_m"],inputs["expected_m"],disable_ue8m0_cast=True)
    finally:
        _set_pdl(saved)
    return out
