# KV Hunt — lane 2 (Bluey+Asusi) — goal: >507K tokens validated, target 900K+
# Started 2026-08-27 ~05:00 UTC by Knox. Baseline: v8 fp8 kv-cache-memory 4445787956 = 507,041 tok.
# Advantages vs original ladder: LOCAL weights both nodes (no NFS pressure), fresh-ish nodes.
# Ladder: 5.5G(672K) -> 6.5G(797K) -> 7.5G(921K); then vm-tuning, staggered loads, InstantTensor variants.
# Hygiene: logs before teardown, reboot nodes every ~5 cycles, lane 1 untouchable.
=== ATTEMPT1 5.5G LAUNCHED 05:04:13 ===
ATTEMPT1 RESULT: TIMEOUT 13min
ATTEMPT1 RESULT: SERVING (extended cycle 12) — (EngineCore pid=319) INFO 08-27 05:18:29 [kv_cache_utils.py:2141] GPU KV cache size: 672,606 tokens, Maximum concurrency for 262,144 tokens per request: 2.57x
ATTEMPT1 STRESS: 20K prefill + vision PASS, 2min soak OK — 672,606 tok STABLE. NEW TP2 RECORD.
ATTEMPT1 NOTE: one NVRM NV_ERR_NO_MEMORY at 05:17:44 during KV alloc — survived anyway. Edge is near.
=== ATTEMPT2 6.5G LAUNCHED 05:25:58 ===
ATTEMPT2 RESULT: HEAD DIED cycle 49
(EngineCore pid=319)     raise RuntimeError("cancelled")
(EngineCore pid=319) RuntimeError: cancelled
(APIServer pid=1)     raise RuntimeError(
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
ATTEMPT2 POSTMORTEM: head allocated 796,779 tok but Asusi rank died in warmup (NCCL heartbeat, NVRM OOM x2 on Bluey during alloc). 6.5G allocates, cannot survive warmup.
=== ATTEMPT3 6.0G + cache-flusher LAUNCHED 05:43:38 ===
ATTEMPT3 RESULT: HEAD DIED cycle 14
(EngineCore pid=319) RuntimeError: cancelled
(APIServer pid=1)     raise RuntimeError(
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
=== ATTEMPT4 6.0G warmup-OFF + flushers LAUNCHED 06:01:05 ===
ATTEMPT4 RESULT: HEAD DIED cycle 16 (2nd window)
(EngineCore pid=319) RuntimeError: cancelled
(APIServer pid=1)     raise RuntimeError(
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
ATTEMPT4 POSTMORTEM: Asusi had 110.73GiB free yet NVRM refused 6GB KV alloc = physical fragmentation, not budget. Warmup-off made no difference. New plan: reboot both + vm.compact_memory, go 7.5G.
=== ATTEMPT5 7.5G fresh-reboot + compaction LAUNCHED 06:20:09 ===
ATTEMPT5 RESULT: HEAD DIED cycle 16 (2nd window)
(EngineCore pid=319) RuntimeError: cancelled
(APIServer pid=1)     raise RuntimeError(
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
ATTEMPT5 POSTMORTEM: KV alloc SUCCEEDED (920,953 tok both ranks reserved 7.5G) — died in compile_or_warm_up_model dummy forward (activation transient). Lever: max-num-batched-tokens 2048->1024.
=== ATTEMPT6 7.5G + mnbt1024 + fresh reboot LAUNCHED 06:39:11 ===
ATTEMPT6 RESULT: HEAD DIED cycle 17 (2nd window)
(EngineCore pid=319) RuntimeError: cancelled
(APIServer pid=1)     raise RuntimeError(
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
ATTEMPT6 POSTMORTEM: segfault in warmup (no NVRM, no oom-kill) — mnbt 1024 < index_topk 2048 = likely OOB in DSA indexer. Reverting mnbt. ATTEMPT7 = Tony call: MTP3.
=== ATTEMPT7 7.5G MTP3 fresh-reboot LAUNCHED 06:58:45 ===
ATTEMPT7 RESULT: HEAD DIED cycle 20 (2nd window)
(EngineCore pid=319) RuntimeError: cancelled
(APIServer pid=1)     raise RuntimeError(
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
ATTEMPT7 POSTMORTEM: MTP3 no help — same silent segfault on first KV touch in warmup, no NVRM/oom-kill. Theory: 7.5G reserve succeeds but physical backing absent; first touch faults. 7.5G per rank = physically unavailable. Bisecting down with reboot ritual.
=== ATTEMPT8 7.0G MTP3 fresh-reboot LAUNCHED 07:17:42 ===
ATTEMPT8 RESULT: HEAD DIED cycle 19 (2nd window)
(EngineCore pid=319) RuntimeError: cancelled
(APIServer pid=1)     raise RuntimeError(
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
ATTEMPT8 RESULT confirmed dead. HUNT CLOSED per Tony order. Final: 5.5G/672,606 tok = TP2 ceiling on Bluey+Asusi. 6.0G+ = phantom reserve, dies on first touch. Pivot: TP4 MTP4 300K all four nodes.
