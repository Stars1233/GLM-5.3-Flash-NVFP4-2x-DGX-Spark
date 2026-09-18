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

## Findings / notes (append-only)

