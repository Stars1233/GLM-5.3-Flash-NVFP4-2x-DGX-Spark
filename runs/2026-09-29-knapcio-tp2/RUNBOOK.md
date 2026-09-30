# Runbook: knapcio's stack on two Sparks (TP2 port, 2026-09-29)

The speed stack is **[knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)**
(MIT; vLLM-derived files Apache-2.0), pinned at commit **`770d115`**. It ships for **four** Sparks only. This folder
is our port to two: a 4-line change to his launcher (`start_tp2.sh`), two lane configs, and the measurements. His
stack is itself built on this repo family's v11 image, RoCE all-reduce port and DFlash2 prefix-cache repair.

The TP4 sibling repo has the full four-node runbook
([`runs/2026-09-29-knapcio-stack/RUNBOOK.md`](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/blob/main/runs/2026-09-29-knapcio-stack/RUNBOOK.md)):
clone at the pinned commit, build his image on our v11 tag, convert the weights (CPU, about 50 s), copy the FP8
drafter. Everything there applies here; this file covers only what is different at TP2.

## What the port changes

`start_tp2.sh` is his `start.sh` with exactly this diff (plus one header comment):

```diff
-    --tensor-parallel-size 4 --nnodes 4 --node-rank $r ...
+    --tensor-parallel-size 2 --nnodes 2 --node-rank $r ...
-    for r in 0 1 2 3; do        (preflight, stop, status)
+    for r in 0 1; do
-    for r in 3 2 1 0; do run_rank $r; done
+    for r in 1 0; do run_rank $r; done
```

The env files (`env.tp2-low`, `env.tp2-high`) then turn off the three pieces of his stack that are built for four
ranks, and resize memory:

| setting | why |
|---|---|
| `GLM_PREFILL_SHARD=0` (and `_PAD=0`) | mHC prefill sharding hard-codes `TP = 4` and **refuses to boot** on any other layout |
| `PF3_ARM=off` | routed-MoE prefill kernels tuned on TP4 shapes |
| `GATHER_ROUTE=0` | the NCCL route for prefill-shard row gathers; nothing to route with sharding off |
| `KV_BYTES=6442450944` (6 GiB/rank) | see memory, below |
| `MM_IMAGES=2`, `MM_CACHE_GB=1` | smaller image-encoder budget, to make room for the KV pin |
| `CAPTURE_SIZES` to 128 rows, `CAPTURE_MAX=128` | graphs above 128 rows only serve 28+ concurrent requests |
| `MAX_MODEL_LEN=262144` | this repo's TP2 context |

`PF3_ARM` and `GATHER_ROUTE` are consumed while `profiles/current.env` is sourced, so the env files set them
**before** the `source` line. The prefill-shard switch lives inside the profile's `EXTRA_ENV` string, so the env
files rewrite it with a bash substitution after the `source` line.

Everything on the decode side stays on and armed at TP2: 8-bit dense layers and their decode kernels, certified LM
head, draft truncation, device-side draft-length selection, FP8 drafter, RoCE one-shot all-reduce (world=2),
FlashKDA (self-test PASS at 32 heads per rank). His dense-kernel and Marlin tune tables are keyed to TP4 shapes;
entries that do not match fall back to stock, which is why the TP2 prefill is not faster than knapcio's TP4 per GPU.

## Memory: 6 GiB of KV (default), not his 24

Each node holds half the model: **87.2 GiB of weights per rank** plus 1.3 GiB of drafter. His TP4 profile pins
24 GiB of KV per rank, which cannot fit. What we tried, all on 2026-09-29:

| KV pin per rank | KV pool | free memory, tightest node | result |
|---|---|---|---|
| 8 GiB, graphs to 256 rows | 745,547 tokens | 1 to 2 GiB at idle | booted; too thin to load-test |
| 7 GiB (+ image budget trimmed) | 654,157 tokens | 2 GiB idle, **0** under a 110K prompt | **ran out of GPU memory** (`NV_ERR_NO_MEMORY`), no crash |
| **6 GiB (+ image budget trimmed), default** | **560,362 tokens** | 4 to 5 GiB idle, 1 to 2 GiB under load | passes one 111K prompt (3/3 needles), two concurrent 114K prompts, the full suite |
| 4 GiB | 372,773 tokens | 3 to 7 GiB idle, 3 GiB under load | passes everything; 13 to 17% faster prefill than 6 GiB |

6 GiB fits two full 262K requests at once. The price is headroom: prefill is 13 to 17% slower than at 4 GiB, and on
lane A (whose head also serves the weights to its worker over NFS) concurrency at C3 to C6 and 114K-token decode are
slower too (README, "6 GiB (default) or 4 GiB"). Set `KV_BYTES=4294967296` for the 4 GiB config.

The trimmed image budget is `MM_IMAGES=2`, `MM_CACHE_GB=1` (his profile: 16 images, 4 GB). With
`--kv-cache-memory-bytes` pinned, vLLM skips memory profiling, so nothing checks any of this for you: watch
`MemAvailable` on both nodes on the first boot and on the first long prompt.

## Launch a lane (from its head)

```bash
cp start_tp2.sh env.tp2-low env.tp2-high /srv/glm53-knapcio/     # edit HOSTS/IPS/paths first
cd /srv/glm53-knapcio
ENV_FILE=env.tp2-low  bash start_tp2.sh serve    # lane A: Reddie (head) + Spark4, :8000, thinking low
ENV_FILE=env.tp2-high bash start_tp2.sh serve    # lane B: Bluey (head) + Asusi, :8001, thinking high
ENV_FILE=env.tp2-low  bash start_tp2.sh stop     # ALWAYS pass ENV_FILE: without it start.sh reads .env
```

Boot to `/health` 200: **140 to 190 s**, weights load in 44 s with his fast loader (the previous TP2 recipe took
about 11 minutes to load and 16 to serve).

## The two lanes

Both lanes run the identical stack; the only difference is the server's default reasoning effort
(`DEFAULT_EFFORT`, i.e. `--default-chat-template-kwargs '{"reasoning_effort": ...}'`). Clients can override it per
request with `"chat_template_kwargs": {"reasoning_effort": "low" | "high" | "max"}`; his chat template has no
off switch. Changing the default needs a restart.

## Files here

| path | what |
|---|---|
| `start_tp2.sh` | his `start.sh` @770d115, 2 ranks / 2 nodes |
| `env.tp2-low`, `env.tp2-high` | lane A and lane B |
| `bench/bench_tp2_night.py` | the speed-night harness, with `--effort` and time-to-answer / thinking-length metrics |
| `results/` | raw JSON of both lanes, at the 6 GiB default (`-kv6`) and the 4 GiB option (`-kv4`) |
| `charts/` | the README charts (no dependencies), light and dark |
