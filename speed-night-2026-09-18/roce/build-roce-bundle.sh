#!/usr/bin/env bash
# build-roce-bundle.sh -- assemble ~/patches/glm-roce on a node so tp2-exp.sh ROCE=1 works.
# The five vLLM files next to this script are the b12x RoCEnante adapter ported to the
# sm121-v11-dflash2 tree (from local-inference-lab/vllm#597 via the DS4 lane's sr2roce set).
# The b12x runtime itself (Luke Alonso, Apache-2.0, comm/roce at b58f34ea) is pulled out of
# the DS4 image that already carries it; no build, no pip.
set -euo pipefail
SRC_IMAGE="${SRC_IMAGE:-vllm-dsv41:exl3b-roce}"
OUT="${OUT:-$HOME/patches/glm-roce}"; D=/usr/local/lib/python3.12/dist-packages
mkdir -p "$OUT"
cp "$(dirname "$0")"/{b12x_roce_all_reduce,cuda_communicator,parallel_state,envs,gpu_worker}.py "$OUT/"
cid=$(docker create "$SRC_IMAGE")
docker cp "$cid:$D/b12x" "$OUT/b12x"
docker cp "$cid:$D/b12x-1.3.0.dist-info" "$OUT/b12x-1.3.0.dist-info"
docker cp "$cid:/opt/b12x-roce" "$OUT/b12x-roce"
docker rm "$cid" >/dev/null
find "$OUT" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
chmod -R a+rX "$OUT"
echo "bundle at $OUT: $(ls "$OUT" | tr '\n' ' ')"
echo "b12x roce sha: $(cat "$OUT/b12x-roce/B12X_ROCE_SHA" 2>/dev/null)"
