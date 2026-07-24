# Default path one_batch smoke

Config: OPT=0, deepep=low_latency, flashmla_kv, DP8 batch=8

| run | last_ttft | latency | cache_hit | input_throughput |
|-----|----------:|--------:|----------:|-----------------:|
| default_decode_M16 | 0.5353 | 1.2805 | 0.9998 | 979590.34 |
| default_prefill_M1024 | 2.0494 | 2.0495 | 0.9846 | 259824.97 |
| default_prefill_M2048 | 3.8537 | 3.8538 | 0.9697 | 140300.92 |
