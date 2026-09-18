# TP2 speed night — 2026-09-18 (02:30–10:00 EDT)

Two TP2 lanes on the four Sparks all night (A = Reddie+Spark4, B = Bluey+Asusi), nvidia
`GLM-5.3-Flash-NVFP4` (ModelOpt) + DFlash2, `--enforce-eager`, temperature 0, one harness
(`speed-night-2026-09-18/bench_tp2_night.py`). Raw JSON per run in
`speed-night-2026-09-18/results/`, running log in `speed-night-2026-09-18/LOG.md`.

## Read this first: the fleet was crippled all night

At 03:10 the first boot wedged all four nodes (memory, see §3). Three of them were
watchdog-rebooted, and **came back with the GPU power-clamped at ~14 W**: 611–890 MHz under
load, 26–33 TFLOPS on a bf16 matmul against **65 TFLOPS at 2164 MHz on Bluey**, which never
rebooted. `nvidia-smi -r`, `-lgc`, `-ac`, `-pl`, and two clean reboots did not clear it; it
looks like the platform power budget stuck at a fallback level after the unclean reset, and
only an AC power cycle is likely to fix it.

Consequences:

- Every number measured before 09:20 (everything in `speed-night-2026-09-18/LOG.md` and the
  `*-CLAMPED.json` files) is ~2.5× slower than this hardware does; §4 has the healthy-fleet
  before/after taken after Tony's power cycle.
- **Before trusting any number from this fleet, run
  `speed-night-2026-09-18/gputest.sh` on every node.** Under 50 TFLOPS is a clamped GPU.

The one lane-to-lane comparison that mattered came out clean: A0 and B0 baselines agree
within 1% on every prompt, so within-night deltas are real.

## 1. What to ship (the new default)

| change | measured | transfers to a healthy fleet? |
|---|---|---|
| **b12x RoCEnante one-shot RoCE all-reduce** (`ROCE=1`, bind-mount only, no rebuild) | boots, engages (`RoCEnante all-reduce is live: 393216 bytes`), stable through every bench. Aggregate peaks +5–18% at every level C1–C6; single stream flat | yes, and it should be *larger*: with the GPU clamped, a 195 ms step hides the ~9 ms of NCCL it removes. The DS4 lane's speed run was built on exactly this |
| **#18 prefix-cache repair** (`PREFIX_FIX=1`, `kv_cache_coordinator.py` bind-mount) | 13K-token repeat: hits +9216, **TTFT 21.1 s → 6.3 s (−70%)**. First measurement of #18 on TP2 with the shipped v11 image | yes, clock-independent |
| **KV pin 6 → 8 GiB** | pool **536,832 → 714,240 tokens (+33%)**, boots clean at mnbt 8192 | yes (memory math) |
| **Boot hardening** (§3): `MAX_JOBS=2`, persistent TileLang/Triton/FlashInfer caches, video profile off, `--memory 112g`, lifetime flusher | first boot 41 min → every later boot 16–19 min; zero deaths after the fix | yes |

Unchanged on purpose: **k=7**, **seqs 6**, **mnbt 8192**, eager (see §2).

## 2. What was tested and did not make it

| experiment | result | verdict |
|---|---|---|
| k=5 (B2) | every single-stream prompt −15…−24%, flat at C4–C6 | keep k=7. Acceptance *ratio* went up (0.73 vs 0.64 on code) while mean accepted *length* went down (4.66 vs 5.47); length is the metric that tracks tok/s |
| dynamic k `[[1,3,7],[4,512,5]]` (B3) | see LOG.md row 7 | |
| seqs 6 → 32 (A1b) | +10% aggregate at C12 only, TTFT p90 60–179 s at C8–C16 | not on this fleet. TP4 got +175% from this knob with compute headroom; **retest after the power cycle** |
| seqs 32 + mnbt 16384 (A1) | `NVRM: NV_ERR_NO_MEMORY` on both nodes at the first real batch | **does not fit**. With `--kv-cache-memory` pinned this vLLM *skips memory profiling*, so nothing catches an activation peak that overruns the ~16 GiB left after 90.5 GiB weights + KV |

## 3. Boot reliability (the 03:10 incident)

Both lanes loaded 90.46 GiB/rank (the nvidia quant is ~3 GiB/rank bigger than RedHat) and hit
the wall in the post-load phase: Bluey's OOM killer shot the worker — invoked by `cicc`, the
CUDA JIT compiler, with MemAvailable at 4 GiB — while the other three nodes went into a
kernel page-allocator livelock and were watchdog-reset. Default `MAX_JOBS` lets FlashInfer
spawn a compiler per CPU while the model is already resident; on 128 GB unified memory that
is fatal. Fixes now in `speed-night-2026-09-18/tp2-exp.sh`:

- `MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1` — no compiler storm.
- `TILELANG_CACHE_DIR`/`TRITON_CACHE_DIR` under `/cache`, and `/root/.cache/flashinfer`
  bind-mounted to the host — the fp4 CUTLASS GEMM variants compile once (they took the first
  boot from 16 to 41 min), never again.
- `--limit-mm-per-prompt '{"image":2,"video":0}'` — images still work; the max-size *video*
  encoder profile at boot is skipped.
- `--memory 112g --memory-swap 112g` — an overrun is a clean container OOM, not a reboot.
  (Note: UMA GPU allocations are not cgroup-charged; the cap only bounds host-side memory.)
- `flusher_lane.sh` — unconditional 20 s page-cache drop for the life of the container. The
  threshold flusher expired 25 min after launch; A1 died one minute later with 9 GiB of
  reclaimable cache sitting there.
- `nvidia-smi -r` on a GB10 followed by CUDA work without a reboot faulted the GPU (SMMU
  timeouts). Don't.

## 4. Numbers — healthy fleet (after the 09:20 power cycle)

Same nvidia weights, same harness, both lanes benchmarked in the same hour: lane A = final
config (`ROCE=1 PREFIX_FIX=1 KV_MEM=8589934592`), lane B = shipped recipe. Temperature 0,
median of 3 reps; sweep = median / peak of 3 rounds, mixed real prompts (no counting).
GPUs verified first: 81.5 / 79.5 / 80.4 TFLOPS on the three power-cycled nodes.

**Aggregate tok/s, C1–C6**

| | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| baseline | 43.4 / 44.5 | 29.0 / 49.7 | 30.2 / 51.1 | 50.3 / 64.5 | 44.4 / 62.0 | 47.8 / 48.9 |
| final | 42.7 / 46.1 | 33.8 / 52.0 | 35.9 / 57.7 | 59.0 / 60.5 | 40.9 / 66.3 | 51.6 / 65.8 |
| Δ median | −2% | +17% | +19% | +17% | −8% | +8% |

**Cold prefill (salted prompts, TTFT → tok/s)**

| prompt | baseline | final | Δ |
|---|---|---|---|
| 5,942 tok | 4.98 s → 1,192 | 3.95 s → 1,506 | +26% |
| 29,868 tok | 31.4 s → 952 | 24.1 s → 1,242 | +30% |
| 113,910 tok | 115.8 s → 984 | 85.3 s → 1,336 | +36% |

**Single stream decode tok/s (median)**

| | count100 | count300 | code | json | sql | tooluse | math | prose | narrative | summary |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 67.0 | 51.4 | 41.8 | 39.5 | 39.5 | 47.2 | 43.7 | 16.7 | 17.0 | 19.1 |
| final | 61.3* | 57.7 | 44.7 | 42.8 | 39.6 | 47.7 | 46.2 | 18.1 | 17.9 | 20.9 |

\*lane A's count100 ran while lane B was still streaming weights over the fabric; its peak was 65.5.

**Other:** 114K-token long context C1 21.1 → 19.1 tok/s, C2 8.5 → 9.2 per stream (issue #14
unchanged). KV pool 536,832 → 714,240 tokens. 13K-token repeat TTFT 21.1 s → 6.3 s (#18).

**Headline: prefill +26–36%, aggregate +8–19% at C2–C4 and C6, single-stream decode
unchanged (flat to +8%).** The decode step itself was not touched tonight; that is the next
target (adaptive verification length and FP8 dense projections — the attention/KDA projections
in the nvidia checkpoint are bf16, 132 modules excluded from NVFP4).

## 5. What to do next (in order)

1. **Power-cycle Reddie, Spark4, Asusi** (full AC off/on). Run `gputest.sh` on all four;
   expect ~65 TFLOPS each.
2. Re-run `bench_tp2_night.py --suite all` on the shipped recipe and on `ROCE=1 PREFIX_FIX=1
   KV_MEM=8589934592` — that is the real before/after for this night's changes.
3. Then retest `SEQS=32` (mnbt 8192) with the sweep to C16; on a healthy fleet the TP4
   result says this is the aggregate lever.
4. Get the RedHat checkpoint back on the fleet; the nvidia quant costs ~3 GiB/rank of the
   memory that every knob above spends.
