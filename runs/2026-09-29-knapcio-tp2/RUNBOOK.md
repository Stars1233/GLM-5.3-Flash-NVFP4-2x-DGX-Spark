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
| `KV_BYTES=4294967296` (4 GiB/rank) | see memory, below |
| `CAPTURE_SIZES` to 128 rows, `CAPTURE_MAX=128` | graphs above 128 rows only serve 28+ concurrent requests |
| `MAX_MODEL_LEN=262144` | this repo's TP2 context |

`PF3_ARM` and `GATHER_ROUTE` are consumed while `profiles/current.env` is sourced, so the env files set them
**before** the `source` line. The prefill-shard switch lives inside the profile's `EXTRA_ENV` string, so the env
files rewrite it with a bash substitution after the `source` line.

Everything on the decode side stays on and armed at TP2: 8-bit dense layers and their decode kernels, certified LM
head, draft truncation, device-side draft-length selection, FP8 drafter, RoCE one-shot all-reduce (world=2),
FlashKDA (self-test PASS at 32 heads per rank). His dense-kernel and Marlin tune tables are keyed to TP4 shapes;
entries that do not match fall back to stock, which is why the TP2 prefill is not faster than knapcio's TP4 per GPU.

## Memory: why 4 GiB of KV, not his 24

Each node holds half the model: **87.2 GiB of weights per rank** plus 1.3 GiB of drafter. His TP4 profile pins
24 GiB of KV per rank, which cannot fit. Our first boot used 8 GiB of KV and graphs up to 256 rows: it came up
healthy, but every node sat at **1 to 2 GiB free at idle**, too thin to push a 114K-token prefill through without
risking the host-OOM wedge these nodes are prone to. 4 GiB of KV and graphs to 128 rows leaves **3 to 7 GiB free**,
the same margin the previous TP2 recipe benchmarked with. We also tried **7 GiB** (with the image budget cut to 2 per
prompt): it booted with a 654,157-token pool and 2 GiB free on each head, then a 110K-token prompt drove both heads to
0 GiB free and one logged 15 `NV_ERR_NO_MEMORY` allocation failures. No crash, no reboot, but it does not fit. The KV pool is **372,773 tokens**: one full 262K request
plus room for short ones.

With `--kv-cache-memory-bytes` pinned, vLLM skips memory profiling, so nothing checks this for you. Watch
`MemAvailable` on both nodes on the first boot.

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
| `results/` | raw JSON of both lanes |
| `charts/` | the README charts (no dependencies), light and dark |
