# Partial evidence only

This locked TP4 campaign is intentionally preserved but is not an official
performance or acceptance result.

- The campaign acquired physical GPUs 0,1,2,3 and completed initial environment
  and preflight, reachability, semantic, three M16 baseline, and M16
  reference-control steps.
- `paired/m16_c10d_inplace.json` was fully serialized and acknowledged over the
  TP CPU group according to the recorded source control flow, but its command
  never returned. The contemporaneous operator observation in
  `INCIDENT_RECEIPT.json` records all four workers still live and futex-waiting
  for more than six minutes after the payload and after the last logged
  device-context barrier warning. The recorded source flow is consistent with
  a stall in the normal shutdown path's NCCL device-group barrier, but no stack
  samples were captured. The process group was deliberately terminated to
  release the scheduler locks.
- Consequently `status.tsv` has no `paired/m16_c10d_inplace` row, exit code, or
  finish timestamp, and the after-state receipts were never reached. A payload
  on disk is not a completed run receipt.
- Independent raw-sample rederivation also found launch-envelope violations:
  M16 reference-control has 14 of 200 calls above 500,000 ns (maximum
  1,065,874 ns), and M16 c10d has 5 of 200 (maximum 910,102 ns). The three M16
  baselines contain two further violations. All timing in this archive is
  therefore rejected, irrespective of the shutdown hang.
- The c10d payload's exact values and ABI checks completed, but its observed
  paired-p50 speedup was only 1.018643x, below the predeclared 1.03 diagnostic
  gate. This number is included only to identify the failed attempt and must
  not be cited as valid performance evidence.

The follow-up source change replaces only the TP AllReduce runner's final NCCL
entry barrier with its existing TP CPU-group rendezvous. It also retries an
entire logical A/B pair solely when independently auditable host start brackets
exceed the fixed 500 us envelope. Logical A/B order stays fixed while the exact
input variant alternates per physical attempt. Every completed attempt is kept
in either the successful result or a bounded alignment-failure receipt, with
CUDA latency excluded from admission. A new clean locked campaign is required.
