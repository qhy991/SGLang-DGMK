# Prior Knowledge — GLM-5.2 o_proj prefill

## Live baseline (2026-07-18)

Host `dry-vm-embraces-fin-03`, idle NVIDIA B200 GPU 0, Kernel-Harness commit
`7d79e5e`, DeepGEMM 0.1.4, CUDA 13.0. Command:

```bash
cd /home/qinhaiyan/Kernel-Harness/testbench/tasks/glm52/o_proj_prefill
CUDA_VISIBLE_DEVICES=0 REMOTE_GPU_ID=0 ./run.sh
```

| M | reference us | reference FP8 utilization | bound | target us | target utilization |
|---:|---:|---:|---|---:|---:|
| 1024 | 90.54 | 50.60% | compute | 65.4 | 70% |
| 2048 | 141.94 | 64.55% | compute | 122.2 | 75% |
| 4096 | 271.35 | 67.53% | compute | 229.1 | 80% |

The default candidate is the reference call and was correct on all shapes, with
0 wins / 0 regressions / 3 neutral and exit 1.

## Interpretation

- All shapes are above the B200 FP8 roofline ridge point and are compute-bound.
- Required latency reductions versus this live baseline are approximately
  27.8%, 13.9%, and 15.6%.
- The prior `o_proj_decode` win (`compiled_dims="nk"`) was a small-M
  memory/under-utilization result. It is a useful low-risk lever to re-test, not
  evidence that the same mechanism will reach these compute-bound targets.
- KernelWiki page `kernel-deepgemm` identifies SM100 tcgen05/TMEM native block
  scaling and JIT shape specialization as the relevant implementation family.
