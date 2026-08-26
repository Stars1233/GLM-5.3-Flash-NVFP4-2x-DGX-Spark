# GLM-5.3-Flash on 2x NVIDIA DGX Spark

Serving [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total / 18B active MoE, released 2026-08-26) across two DGX Spark (GB10, SM121) nodes at tensor-parallel 2, using the [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) weight-only NVFP4 quant. **262,144-token context. Working, benchmarked, same-day as the model drop.**

As far as we can tell this was the first working GLM-5.3-Flash deployment on DGX Spark hardware. Getting there took fixing **seven distinct day-0 bugs** across vLLM, FlashInfer, and their dependency chain — every one is documented in [docs/DEPLOY-REPORT.md](docs/DEPLOY-REPORT.md) with root causes, receipts, and the probe scripts that found them.

## Results

| Metric | Day-1 (no MTP) |
|---|---|
| TTFT (median, 3 runs) | 0.239 s |
| Decode | ~14.3 tok/s |
| Context | 262,144 tokens |
| KV pool | 7.19 GiB bf16 = 603,144 tokens (2.3x concurrent full-context) |
| Boot time | ~14 min (~25 with MTP draft head) |

MTP speculative decode (`{"method":"mtp","num_speculative_tokens":4}`) is the phase-2 speed lift — the checkpoint ships a native MTP head.

## Why this needs a patched image

The vLLM PR authors' day-0 image (`vllm/vllm-openai:glm53-flash-arm64-cu130`) works on B200. On GB10/SM121 it fails five separate ways. Our derivative (`docker/Dockerfile.glm53-sm121*`, applied in order v1→v7) fixes:

1. **NoPE MLA vs the SM12x sparse backend** — the only stock capability-12 sparse-attention backend requires the packed `fp8_ds_mla` cache layout, which hardcodes DeepSeek's `pe_dim=64`. GLM-5.3 is NoPE (`qk_rope_head_dim=0`) → assert death in warmup. Fix: extend vLLM's SM90 NoPE sparse-MLA backend (FlashInfer `BatchMLAPagedAttentionWrapper`, plain bf16 cache) to SM121 with the FA2 path — probed directly on the GPU with the model's real shape before trusting it (`probes/probe_sm121_nope_mla.py`).
2. **FlashInfer 0.6.17 FA2 MLA NaN** — the FA2 scheduler produces NaN for 64–256-row batches on SM121 (bisect: `probes/probe_fa2_bisect.py`). Normal prompts land exactly there. Fix: FlashInfer **0.6.18 nightly**.
3. **The nightly's dependency sabotage** — installing it silently downgrades `nvidia-nccl-cu13` to 2.29.7 (NCCL "internal error" on the Spark IB fabric; re-pin **2.30.7**) and skews `nvidia-cutlass-dsl` to a mixed 4.7.0/4.6.2 install (CuTeDSL warmup ICE; re-pin **4.6.2**). Audit transitive pins after ANY pip install in these images.
4. **PDL on unvalidated silicon** — vLLM enables Programmatic Dependent Launch for capability ≥ 9, including SM121, in the Triton kernels carrying KDA recurrent state. Gated off on SM12x.
5. **Indexer uninitialized top-k** — the kpool top-k destination was `torch.empty` and the kernels only guarantee the first `min(k, valid)` entries; short rows carried garbage pool ids → bogus token indices → attention gathers uninitialized KV → NaN lottery. Fix: init to `-1` + clamp expanded pool ids (`docker/patch_v7.py`).

Two serve-flag landmines (no code needed):
- `--block-size 2304` — vLLM's hybrid block aligner picks a size whose kpool storage tiles by 32, but DeepGEMM's arch-12 fp8 paged-MQA accepts only 64-entry pool pages. 2304 is a multiple of kpool·64 and of the MLA 128 alignment.
- `--gpu-memory-utilization 0.85` — 0.78–0.80 starve the bf16 KV cache at 131K+. (Credit: barrydeen's independent recipe.)

## Quickstart

Nodes: head owns the weights and serves `:8000`; worker mounts them over NFS at the same path.

```bash
# 1. Build the patched image on the head, ship to the worker
docker build -f docker/Dockerfile.glm53-sm121 -t glm53:sm121-v1 .   # then v2..v7 in order, each FROM the previous
docker save glm53:sm121-v7 | ssh worker docker load

# 2. Pre-launch ritual (BOTH nodes, every launch — GB10 unified memory)
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
# worker: verify the NFS mount answers: ls /var/tmp/glm-5.3-flash-nvfp4/config.json

# 3. Launch worker FIRST, wait ~25 s, then head
./launch-glm53-vllm-tp2.sh 1   # on the worker
./launch-glm53-vllm-tp2.sh 0   # on the head
```

Edit the top of `launch-glm53-vllm-tp2.sh` for your IPs/paths. Full serve args, NCCL fabric env, and rationale for every flag: [docs/DEPLOY-REPORT.md](docs/DEPLOY-REPORT.md).

Smoke test:
```bash
curl http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Say hello and name yourself."}],"max_tokens":40,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Hard-won operational rules

- **Tear down BOTH ranks before relaunching either.** A new rank that rendezvouses with a dying one hangs or dies confusingly.
- **`grep '^IMAGE' launch-*.sh` on BOTH nodes before every launch.** Two of our "mystery" garbage boots were a silent image-version mismatch between ranks (in-place remote edits had failed silently). Copy whole files between nodes, never sed over ssh.
- **Capture `docker logs` before `docker rm -f`.**
- Two consecutive unexplained deaths = stop and diagnose, never crash-loop.
- `max_tokens` includes reasoning tokens when thinking is on; pass `chat_template_kwargs: {"enable_thinking": false}` per-request to disable.

## fp8 KV cache: not yet possible on GB10

We tried. FlashInfer's fa2 fp8-MLA dequant kernel is genuinely Hopper-only (relaxing the SM90 gate → CUDA "invalid argument" at launch: `probes/probe_fa2_fp8.py`), and the Blackwell-native trtllm path's packed layout has no NoPE variant. Until upstream adds either, bf16 KV it is — 603K tokens is still 2.3x concurrent 262K requests.

## Debugging kit (reusable for any day-0 model on new silicon)

- `probes/probe_sm121_nope_mla.py` — probe a FlashInfer kernel with your model's real geometry BEFORE patching arch gates.
- `probes/probe_fa2_bisect.py` — NaN bisect harness over batch shapes.
- `probes/probe_mhc.py` — A/B a Triton/TileLang kernel vs its torch reference.
- The deploy report describes the env-gated forward-hook NaN localizer (`GLM53_NAN_DEBUG=1` build) that names the first module emitting non-finite values — how we localized both NaN sources.
- `probes/bench_glm53.py` — 3-run TTFT/decode benchmark.

## Credits

- Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · Quant: [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) (their sm_121 notes were used directly)
- **barrydeen** — the gmu 0.85 reference config and quantization-coverage table from their independently published DGX Spark recipe
- vLLM [PR #53906](https://github.com/vllm-project/vllm/pull/53906) authors for the day-0 image; FlashInfer for the 0.6.18 SM90-NoPE MLA path
- Deployed and debugged by Knox (Claude) for [@tonyd2wild](https://github.com/tonyd2wild)
