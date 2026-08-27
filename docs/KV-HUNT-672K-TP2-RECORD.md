# 672K KV on TP2 — the hunt past the 507K wall (2026-08-27)

**New TP2 record: 672,606 fp8 KV tokens serving stable** on 2x DGX Spark (Bluey+Asusi),
up from the previous 507,041 ceiling. Stress-verified: 24K-token prefill (33.5s), vision
probe pass, soak stable. Config delta from the 507K recipe: `--kv-cache-memory 5905580032`
(5.5 GiB/rank instead of 4445787956).

**What made it possible** (vs the original Reddie+Spark4 ladder that walled at 4.14G):

1. **LOCAL weights on both ranks** — no NFS server co-resident with a serving rank.
2. **Aggressive `drop_caches` immediately before launch** on both nodes.
3. Background cache-flusher loop during the load (`sync; echo 3 > drop_caches` every 30s).

## The 8-attempt ladder (full log: kv_hunt_log.md)

| # | KV/rank | Tokens | Extra levers | Result |
|---|---------|--------|--------------|--------|
| 1 | 5.5G | 672,606 | flush-before-launch | **SERVING, stress-stable — RECORD** |
| 2 | 6.5G | 796,779 | — | allocates, worker dies in warmup (NVRM OOM) |
| 3 | 6.0G | 734,693 | + flusher loop | allocates, worker dies in warmup (NVRM OOM, 110GiB "free") |
| 4 | 6.0G | 734,693 | + autotune/JIT/CuTeDSL warmup OFF | same death — warmup-off is NOT the lever |
| 5 | 7.5G | 920,953 | + fresh reboot + vm.compact_memory | **alloc succeeds on both ranks**, dies in `compile_or_warm_up_model` |
| 6 | 7.5G | 920,953 | + `--max-num-batched-tokens 1024` | silent segfault (no NVRM/oom-kill) — do NOT set mnbt below `index_topk` (2048) |
| 7 | 7.5G | 920,953 | MTP4→MTP3 | same silent segfault on first KV touch |
| 8 | 7.0G | ~860K | MTP3 + fresh reboot | dead — hunt closed |

## Findings for GB10 (SM121) operators

- **The wall is not a budget, it's physics + fragmentation.** vLLM/torch report 110 GiB
  "Initial free memory" while NVRM refuses the KV carveout
  (`NV_ERR_NO_MEMORY` from `_memdescAllocInternal @ mem_desc.c:1359`). MemAvailable lies
  on unified memory; only the NVRM allocator's opinion counts.
- **Phantom reserve / first-touch death:** above ~6 GiB/rank the reservation can SUCCEED
  (vLLM happily prints 920,953 tokens) but physical backing is absent; the first real touch
  during the warmup forward kills the worker with exit code None — no Python exception,
  no NVRM log line, no oom-kill. If your boot dies 4–6s after
  "Initial free memory ... reserved N GiB", suspect this.
- **Fresh reboot + `vm.compact_memory` moves the alloc ceiling** (6.0G refused on a dirty
  node → 7.5G reserved after reboot) but does not create physical capacity.
- **Warmup knobs that do NOT fix it:** `enable_flashinfer_autotune:false`,
  `enable_cutedsl_warmup:false`, `enable_jit_warmup:false` (all via `--kernel-config`),
  MTP 4→3.
- **Trap:** `--max-num-batched-tokens` below the DSA indexer's `index_topk` (2048)
  segfaults GLM-5.3-Flash in warmup. Keep it ≥ 2048.
- **Practical per-rank fp8 KV ceilings observed:** head-with-NFS-duty ≈ 4.14 GiB;
  clean rank with local weights ≈ 5.5 GiB. Need more? Go TP4 (9 GiB/rank held there
  thanks to smaller per-rank weight residency → 1.26M tokens).

## Bonus: production crash post-mortem (same day)

The Reddie+Spark4 TP2 production endpoint died mid-service on a 19K-token text prefill:
worker hung (`RPC call to sample_tokens timed out`) with NVRM OOM underneath — the boot
before it had NOT flushed caches, so the load left the node with no transient headroom.
**Hardening rule: every (re)launch gets `sync; echo 3 > /proc/sys/vm/drop_caches` on all
ranks first.** After relaunching with the flush, the identical 20.8K-token prefill passed.
