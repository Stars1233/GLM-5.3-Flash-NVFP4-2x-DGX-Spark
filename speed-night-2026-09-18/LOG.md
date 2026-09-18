# TP2 speed night — 2026-09-18 (02:30 → 10:00 EDT)

**Goal:** faster GLM-5.3-Flash on 2× DGX Spark (TP2): TTFT, prefill, decode, aggregate C1–C6, KV pool.
**Weights:** `nvidia/GLM-5.3-Flash-NVFP4` (ModelOpt, `ALLOW_MODELOPT`-class build) — the RedHat
compressed-tensors copy was not on the fleet tonight. **Every number in this file is on the nvidia
ModelOpt quant; do not compare 1:1 with the RedHat numbers in CURRENT.md.** Repo default stays RedHat.
**Two lanes, always both busy:** A = Reddie(head :8000)+Spark4, B = Bluey(head :8001)+Asusi.
**Harness:** `bench_tp2_night.py` (this folder), runs on the head node, temp 0, streaming, median/p90/peak.
Prompts: count100/count300 (ceiling only), code, json, sql, tooluse, math, prose, narrative, summary.
Sweep C1–C6 uses the 8 real prompts rotated (no counting). Prefill is cold (salted). longctx = issue #14 (32K × C1/C2).
**Launcher:** `tp2-exp.sh` (this folder) — every knob defaults to the shipped recipe.

## Results (append-only; JSON in results/)

| # | time | lane | label | diff vs shipped | C1 | C2 | C4 | C6 | count100 | code | prose | json | math | verdict |
|---|------|------|-------|-----------------|----|----|----|----|----------|------|-------|------|------|---------|
| 1 | 04:58–05:39 | B | B0-baseline | none (shipped recipe, nvidia quant) | 25.3 | 32.6 | 41.0 | 34.6 | 39.0 | 26.9 | 10.9 | 24.1 | 28.3 | baseline, **clamped fleet**; sweep = peak of 3 (medians hit by JIT stalls) |
| 2 | 05:17–06:07 | A | A0-baseline | none (shipped recipe, nvidia quant), full suite | 27.2 | 34.6 | 40.5 | 34.2 | 39.0 | 26.7 | 10.8 | 24.5 | 28.1 | baseline; cold prefill 685/721/709 tok/s @6K/30K/114K; longctx 114K C1 12.2 → C2 5.5 per-stream (#14 reproduces) |
| 3 | 05:57–06:41 | B | B1-roce | **ROCE=1** (b12x RoCEnante one-shot all-reduce, ported from the DS4 lane, bind-mount only) | 27.8 | 38.5 | 45.5 | 38.8 | 38.5 | 27.4 | 11.1 | 25.2 | 26.8 | boots, engages (`RoCEnante all-reduce is live: 393216 bytes`), stable through full bench. Single stream flat (compute-clamped fleet hides it); aggregate peaks +5–18% at every level vs B0. Keep. |
| 4 | 07:01–07:37 | B | B2-roce-k5 | ROCE=1 + **k=5** | 27.0 | 29.9 | 39.9 | 37.8 | 29.5 | 23.3 | 10.7 | 20.6 | 23.7 | k=5 loses every single-stream prompt (−15…−24%) and is flat at C4–C6 vs B1. **k=7 stays.** Acceptance ratio rises (0.73 vs 0.64 code) while mean accepted length falls (4.66 vs 5.47) — ratio is the wrong metric, length tracks throughput. |
| 5 | 07:11–07:47 | A | A1b-seqs32 | **seqs 6→32** (mnbt 8192) | 24.9 | 31.2 | 34.5 | 30.1 | – | – | – | – | – | sweep to C16: C8 31.5 / C12 **37.4** / C16 32.7 peak, but TTFT p90 60 / 99 / 179 s. On the clamped fleet batching only stretches the step; +10% aggregate at C12 is not worth the queueing. **Retest after power cycle** (TP4 saw +175% from the same knob with compute headroom). |
| 6 | 08:15–08:45 | A | A2-prefix-kv8 | **PREFIX_FIX=1 KV_MEM=8 GiB** (seqs 6, mnbt 8192) | 26.9 | 28.4 | 32.5 | 32.2 | 40.5* | 26.2 | 10.8 | 22.4* | 27.2 | prefix cache hits +9216 on a 13K repeat, TTFT −70%; KV pool 714,240 (+33%). *count100/json medians contaminated by the concurrent prefix probe (peak shown). Sweep slightly under A0 (noise-level, 2 rounds). |
| 7 | 08:19–08:47 | B | B3-roce-dynk | ROCE=1 + **dynamic k** `[[1,3,7],[4,512,5]]` | 26.8 | 31.0 | 39.4 | 35.5 | 37.7 | 27.6 | 11.6 | 28.0 | 28.0 | single stream = k7 (as designed); C4–C6 peaks a little under static k7+RoCE (B1: 45.5/48.8/38.8). No benefit shown; **not adopted**. |

## Findings / notes (append-only)

- **03:10 incident — first boot wedged all four nodes.** Both lanes loaded weights (90.46 GiB/rank
  on the nvidia quant, ~3 GiB more than RedHat) then hit the memory wall in the profiling pass:
  Bluey's host OOM killer shot the worker (`cicc` — the CUDA JIT compiler — invoked it,
  MemAvailable 4 GiB); Reddie/Spark4/Asusi went into a kernel page-allocator livelock and were
  watchdog-rebooted (~5 min lost each, Asusi lost its manual NFS mounts). Fix in `tp2-exp.sh`:
  `MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1` (no nvcc storm during profiling), persistent
  `TILELANG_CACHE_DIR`/`TRITON_CACHE_DIR` under /cache (only the first boot compiles),
  `--limit-mm-per-prompt '{"image":2,"video":0}'` (skips the max-size video encoder profile),
  and `--memory 112g --memory-swap 112g` so any future overrun is a clean container OOM, not a
  reboot. Lesson for the repo: on TP2 the nvidia ModelOpt quant leaves ~27 GiB per node for
  everything that is not weights; every knob that raises the profiling peak (seqs, mnbt, KV pin)
  spends from that budget.
- **03:59–04:15 — GPUs stuck at 611–721 MHz after the watchdog reboots.** First baselines on both
  lanes came in at count100 38.5 / code 27.3 / prose 10.8 tok/s with acceptance normal (0.93 /
  0.63 / 0.16), i.e. a flat ~195 ms/step against the ~78 ms/step the 09-02 recipe documents.
  `nvidia-smi` on the three rebooted nodes: P0, no throttle reason, 611–721 MHz at 9–12 W under
  load; Bluey (never rebooted, has `gb10-clock-cap.service`) at 2184 MHz. `-lgc 2200,2200` live
  did nothing. RoCE was clean (200 Gb/s, GIDs right, NCCL traffic flowing), so it was the clock,
  not the fabric. Fix: install the clock-cap unit on all four and **clean-reboot** the three
  nodes; they came back idling at 305 MHz (the cap floor being honored = the driver is in
  control again). Partial stuck-clock numbers kept as `B0-baseline-CLOCKSTUCK` for the record.
  Lesson: after any unclean reset, check `nvidia-smi --query-gpu=clocks.sm` under load before
  trusting a single number; 96% util at ~10 W is a spinning GPU, not a working one.
- KV pool on the nvidia quant at the 6 GiB pin: **536,832 tokens** (233 × 2304), not the
  678,661 CURRENT.md quotes for RedHat — the checkpoint's KV layout differs.
- **04:30–04:50 — the clock is a power clamp, not a governor bug, and reboot does not clear
  it.** Same 4096² bf16 matmul in a throwaway container: Bluey **64.8 TFLOPS / 115 GB/s at
  2164 MHz, 60 W**; Asusi/Spark4/Reddie **26–33 TFLOPS / 50–80 GB/s at 611–890 MHz, ~14 W**.
  After `nvidia-smi -r` the idle clock reads 2054 MHz and collapses to 611 the instant load
  starts; `-lgc 3003,3003`, `-ac`, `-pl`, GPU reset, clean reboot: no effect. No thermal or
  powercap sysfs, no throttle reason reported. A ~14 W ceiling on an otherwise healthy GPU
  after a watchdog reset looks like the platform power budget (USB-C PD contract / EC) being
  stuck at a fallback level; only an AC cycle is likely to clear it. **Everything measured
  tonight is on a fleet with 3 of 4 GPUs at ~40% speed; only relative deltas are meaningful and
  even those are compute-skewed. DS4, when restored at 10:00, will be slow too until the three
  nodes are power-cycled.** Preflight added to the recommendation: run the matmul probe
  (`speed-night-2026-09-18/gputest.sh`) on every node before trusting any number; < 50 TFLOPS
  means a clamped GPU.
- 04:46 — my clock experiment left a `docker run` hung on Spark4's GPU after `nvidia-smi -r`;
  killing it faulted the GPU (SMMU CMD_SYNC timeouts, NVRM asserts) and cost a forced reboot.
  Do not `nvidia-smi -r` a GB10 and then launch CUDA work on it without a reboot in between.
- **06:36 — A1 (seqs 32 + mnbt 16384) died: `NVRM: NV_ERR_NO_MEMORY` on both nodes in the same
  second, at the first real traffic after boot.** Root cause in the log: with `--kv-cache-memory`
  pinned, this vLLM **skips memory profiling** ("reserved 6.0 GiB ... skipped memory profiling.
  This does not respect gpu_memory_utilization"), so nothing checks that the activation peak of
  a bigger `max-num-batched-tokens` fits in the ~16 GiB left after 90.5 GiB weights + 6 GiB KV.
  **mnbt 16384 does not fit TP2 on this quant.** Retrying seqs 32 at mnbt 8192. Also switched
  the per-launch threshold flusher (expired after 25 min, 1 min before the death) for an
  unconditional 20 s flusher that lives as long as the container (`flusher_lane.sh`).
- **08:15 — A2: #18 prefix-cache fix + KV 8 GiB, both good.** `kv_cache_coordinator.py` from
  `patch_prefix_cache_draft_group.py` applied inside the v11 image (self-check OK, md5
  `317934d8…`) and bind-mounted: a 13,263-token prompt sent three times → hits **+0 / +9216 /
  +9216** (two full 4608-token blocks; the engine pads the attention block to 4608 for mamba
  page alignment, so prompts under 4608 tokens never hit — my first 4263-token probe showed +0
  for that reason, not a bug), TTFT **21.1 s → 6.3 s (−70%)**. KV pin 6 → 8 GiB: pool
  **536,832 → 714,240 tokens (+33%)**, booted clean with mnbt 8192, no NVRM pressure.
