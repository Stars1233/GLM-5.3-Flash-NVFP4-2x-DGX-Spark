#!/usr/bin/env bash
# tp2-exp.sh -- parameterized two-lane TP2 launcher for the 2026-09-18 speed night.
# Derived from launch-glm53-vllm-tp2-dflash2.sh (repo main, sha 5b37138...); every knob
# below defaults to the shipped recipe so an unset env == the baseline.
#
# usage: LANE=A|B EXP_NAME=<label> [knobs...] ./tp2-exp.sh <0|1>
#   lane A: head Reddie 192.168.192.2 (rank 0) + worker Spark4 192.168.192.4 (rank 1), :8000, mport 29521
#   lane B: head Bluey  192.168.192.1 (rank 0) + worker Asusi  192.168.192.3 (rank 1), :8001, mport 29522
# knobs (env): IMAGE MODEL_HOST_PATH GMU MAXLEN SEQS MNBT KV_MEM KV_DTYPE SPEC_JSON SPEC_K
#              EAGER(1) CUDAGRAPH_MODE PREFIX_FIX(0) NCCL_EXTRA VLLM_EXTRA DRAFTER
set -euo pipefail
NODE_RANK="${1:?usage: LANE=A|B EXP_NAME=x tp2-exp.sh <0|1>}"
LANE="${LANE:?LANE=A|B}"; EXP_NAME="${EXP_NAME:?EXP_NAME=label}"

IMAGE="${IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}"
NAME="vllm_glm53_${LANE}"
MODEL_PATH="/models/glm-5.3-flash-nvfp4"
CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
DRAFTER="${DRAFTER:-/var/tmp/models/GLM-5.3-Flash-DFlash2}"

case "$LANE" in
  A) HEAD_IP=192.168.192.2; WORKER_IP=192.168.192.4; MPORT=29521; PORT=8000
     DEF_MODEL_HEAD=/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia
     DEF_MODEL_WORKER=/mnt/reddie-models/GLM-5.3-Flash-NVFP4-nvidia ;;
  B) HEAD_IP=192.168.192.1; WORKER_IP=192.168.192.3; MPORT=29522; PORT=8001
     DEF_MODEL_HEAD=/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia
     DEF_MODEL_WORKER=/mnt/bluey-models/GLM-5.3-Flash-NVFP4-nvidia ;;
  *) echo "LANE must be A or B" >&2; exit 2 ;;
esac
case "$NODE_RANK" in
  0) HOST_IP=$HEAD_IP;   HEADLESS="";           DEF_MODEL=$DEF_MODEL_HEAD ;;
  1) HOST_IP=$WORKER_IP; HEADLESS="--headless"; DEF_MODEL=$DEF_MODEL_WORKER ;;
  *) echo "rank must be 0 or 1" >&2; exit 2 ;;
esac
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$DEF_MODEL}"

# shipped-recipe defaults
GMU="${GMU:-0.85}"; MAXLEN="${MAXLEN:-262144}"; SEQS="${SEQS:-6}"; MNBT="${MNBT:-8192}"
KV_MEM="${KV_MEM:-6442450944}"; KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"; SPEC_K="${SPEC_K:-7}"
# DYNK=1: the k7-below-C4 / k5-above schedule from docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md
if [ "${DYNK:-0}" = "1" ] && [ -z "${SPEC_JSON:-}" ]; then
  SPEC_JSON='{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,3,7],[4,512,5]]}'
fi
if [ -z "${SPEC_JSON:-}" ]; then
  SPEC_JSON="{\"method\":\"dflash\",\"model\":\"/models/dflash2-draft\",\"num_speculative_tokens\":${SPEC_K}}"
fi
EAGER="${EAGER:-1}"
if [ "$EAGER" = "1" ]; then
  GRAPH_ARGS="--enforce-eager"
else
  GRAPH_ARGS="--compilation-config {\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}\"}"
fi
KV_ARGS=""; [ "$KV_MEM" != "0" ] && KV_ARGS="--kv-cache-memory $KV_MEM"
MM_LIMIT="${MM_LIMIT:-{\"image\":2,\"video\":0}}"

# #18 prefix-cache repair for the drafter group, bind-mounted onto v11 (no rebuild)
PREFIX_MOUNT=""
if [ "${PREFIX_FIX:-0}" = "1" ]; then
  test -f "$HOME/patches/kv_cache_coordinator.py" || { echo "MISSING ~/patches/kv_cache_coordinator.py" >&2; exit 3; }
  PREFIX_MOUNT="-v $HOME/patches/kv_cache_coordinator.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py:ro"
fi

# b12x RoCEnante one-shot RoCE all-reduce (port of the DS4 lane's sr2roce patch set to this
# vLLM tree; files staged in ~/patches/glm-roce by the 2026-09-18 feasibility study). ROCE=1
# bind-mounts the b12x package + 5 patched vLLM files and turns the backend on. Rollback: ROCE=0.
ROCE_MOUNTS=""; ROCE_ENV=""
if [ "${ROCE:-0}" = "1" ]; then
  R="$HOME/patches/glm-roce"; D=/usr/local/lib/python3.12/dist-packages
  for f in b12x b12x-1.3.0.dist-info b12x_roce_all_reduce.py cuda_communicator.py parallel_state.py envs.py gpu_worker.py; do
    test -e "$R/$f" || { echo "MISSING $R/$f" >&2; exit 3; }
  done
  ROCE_MOUNTS="-v $R/b12x:$D/b12x:ro -v $R/b12x-1.3.0.dist-info:$D/b12x-1.3.0.dist-info:ro -v $R/b12x-roce:/opt/b12x-roce:ro
    -v $R/b12x_roce_all_reduce.py:$D/vllm/distributed/device_communicators/b12x_roce_all_reduce.py:ro
    -v $R/cuda_communicator.py:$D/vllm/distributed/device_communicators/cuda_communicator.py:ro
    -v $R/parallel_state.py:$D/vllm/distributed/parallel_state.py:ro
    -v $R/envs.py:$D/vllm/envs.py:ro
    -v $R/gpu_worker.py:$D/vllm/v1/worker/gpu_worker.py:ro"
  ROCE_ENV="-e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=${ROCE_AR_MAX:-2MB} -e VLLM_ROCE_ALLGATHER_MAX_SIZE=${ROCE_AG_MAX:-16MB} -e VLLM_ROCE_ALLGATHER_ENABLE=${ROCE_AG:-1} -e B12X_ROCE_HCA=rocep1s0f0 -e B12X_ROCE_GID_INDEX=${GID_INDEX:-3} -e B12X_ROCE_SPIN_LIMIT=300000000 -e B12X_ROCE_CACHE_DIR=/opt/b12x-roce/cache"
fi

test -f "$MODEL_HOST_PATH/config.json" || { echo "MISSING $MODEL_HOST_PATH/config.json" >&2; exit 3; }
test -f "$MODEL_HOST_PATH/chat_template_mm.jinja" || { echo "MISSING chat_template_mm.jinja in $MODEL_HOST_PATH" >&2; exit 3; }
test -f "$HOME/patches/sparse_attn_indexer_kpool.py" || { echo "MISSING ~/patches/sparse_attn_indexer_kpool.py" >&2; exit 3; }
test -f "$DRAFTER/config.json" || { echo "MISSING drafter at $DRAFTER" >&2; exit 3; }
mkdir -p "$CACHE_HOST_PATH/flashinfer" "$CACHE_HOST_PATH/tilelang" "$CACHE_HOST_PATH/triton"
docker rm -f "$NAME" 2>/dev/null || true

echo "[$EXP_NAME] lane=$LANE rank=$NODE_RANK host=$HOST_IP model=$MODEL_HOST_PATH gmu=$GMU len=$MAXLEN seqs=$SEQS mnbt=$MNBT kv=$KV_MEM/$KV_DTYPE spec=$SPEC_JSON eager=$EAGER prefix_fix=${PREFIX_FIX:-0} roce=${ROCE:-0} nccl_extra='${NCCL_EXTRA:-}' extra='${VLLM_EXTRA:-}'"

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  `# cgroup cap (same as the DS4 + TP4 launchers): an overrun becomes a clean container OOM` \
  `# instead of the host page-allocator livelock + watchdog reboot seen 2026-09-18 03:10.` \
  --memory "${MEM_CAP:-112g}" --memory-swap "${MEM_CAP:-112g}" \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  `# FlashInfer JIT output (fp4 CUTLASS GEMM variants, minutes each with MAX_JOBS=2) lives in` \
  `# /root/.cache/flashinfer and would die with the container. Persist it on the host.` \
  -v "$CACHE_HOST_PATH/flashinfer:/root/.cache/flashinfer" \
  -e VLLM_HOST_IP=$HOST_IP \
  -e VLLM_CACHE_ROOT=/cache/vllm-tp2-$EXP_NAME \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  `# JIT storm control: cicc invoked the OOM killer during profiling on 2026-09-18. Cap nvcc` \
  `# parallelism and keep TileLang/Triton caches across experiments (they do not depend on` \
  `# the serving knobs), so only the first boot pays the compile.` \
  -e MAX_JOBS=${MAX_JOBS:-2} -e FLASHINFER_NVCC_THREADS=1 \
  -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton \
  -e PYTORCH_CUDA_ALLOC_CONF=${ALLOC_CONF:-expandable_segments:True} \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=${GID_INDEX:-3} \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  ${NCCL_EXTRA:-} $ROCE_ENV \
  -v $HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro \
  $PREFIX_MOUNT $ROCE_MOUNTS \
  -v "$DRAFTER:/models/dflash2-draft:ro" \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAXLEN" \
    --max-num-seqs "$SEQS" --block-size 2304 --moe-backend marlin \
    --speculative-config "$SPEC_JSON" \
    --kv-cache-dtype "$KV_DTYPE" $KV_ARGS \
    $GRAPH_ARGS --max-num-batched-tokens "$MNBT" \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --default-chat-template-kwargs '{"enable_thinking":false}' \
    --chat-template /models/glm-5.3-flash-nvfp4/chat_template_mm.jinja \
    `# images stay on; video=0 skips the max-size video encoder profile (memory spike at boot)` \
    --limit-mm-per-prompt "$MM_LIMIT" \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    ${VLLM_EXTRA:-} \
    $HEADLESS

sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited" >&2; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
