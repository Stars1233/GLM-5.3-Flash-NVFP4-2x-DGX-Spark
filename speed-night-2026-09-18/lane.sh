#!/usr/bin/env bash
# lane.sh -- reload one TP2 lane with a config and wait for it to serve.
# usage: ./lane.sh <A|B> <exp_name> ["ENV=VAL ENV2=VAL ..."]
#   A = reddie(head :8000) + spark4     B = bluey(head :8001) + asusi
# Steps: save old logs -> rm both containers -> drop_caches+compact both -> start memfree guard
#        -> launch worker -> 25s -> launch head -> poll /health + 1-token canary (<=30 min) -> print KV pool.
set -uo pipefail
cd "$(dirname "$0")"
LANE="${1:?A|B}"; EXP="${2:?exp_name}"; ENVS="${3:-}"
case "$LANE" in
  A) HEAD=reddie; WORKER=spark4; URL=http://100.113.138.96:8000 ;;
  B) HEAD=bluey;  WORKER=asusi;  URL=http://100.92.77.51:8001 ;;
  *) echo "lane A|B" >&2; exit 2 ;;
esac
NAME="vllm_glm53_$LANE"
T0=$(date +%s)
log() { echo "[$(date +%H:%M:%S) lane$LANE $EXP] $*"; }

for n in $HEAD $WORKER; do
  ./spark.sh $n "mkdir -p ~/speednight/logs; docker logs $NAME > ~/speednight/logs/\$(date +%H%M)-prev-$NAME.log 2>&1 || true; mkdir -p /var/tmp/glm53-vllm-cache/flashinfer; docker cp $NAME:/root/.cache/flashinfer/. /var/tmp/glm53-vllm-cache/flashinfer/ 2>/dev/null || true; docker rm -f $NAME >/dev/null 2>&1 || true; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1 || true; pkill -xf \"bash \$HOME/memfree_flusher.sh\" 2>/dev/null; pkill -xf \"bash \$HOME/flusher_lane.sh\" 2>/dev/null; setsid nohup bash ~/flusher_lane.sh </dev/null >/dev/null 2>&1 & echo \"\$(hostname) free: \$(free -g | awk '/Mem/{print \$4\"G free \"\$6\"G cache\"}')\"" 2>&1 | tail -1
done

log "launch worker $WORKER"
./spark.sh $WORKER "LANE=$LANE EXP_NAME=$EXP $ENVS ~/tp2-exp.sh 1" 2>&1 | tail -2 || { log "WORKER LAUNCH FAILED"; exit 1; }
sleep 25
log "launch head $HEAD"
./spark.sh $HEAD "LANE=$LANE EXP_NAME=$EXP $ENVS ~/tp2-exp.sh 0" 2>&1 | tail -2 || { log "HEAD LAUNCH FAILED"; exit 1; }

log "waiting for /health at $URL"
for i in $(seq 1 120); do
  sleep 15
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health)
  if [ "$code" = "200" ]; then
    r=$(curl -s -m 120 $URL/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4,"temperature":0}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"][:20])' 2>/dev/null)
    if [ -n "$r" ]; then
      pool=$(curl -s -m 10 $URL/metrics | grep '^vllm:cache_config_info' | grep -o 'num_gpu_blocks="[0-9]*"\|block_size="[0-9]*"' | tr '\n' ' ')
      log "READY in $(( $(date +%s) - T0 ))s canary='$r' $pool"
      exit 0
    fi
  fi
  # dead container check
  alive=$(./spark.sh $HEAD "docker ps --format '{{.Names}}' | grep -c $NAME" 2>/dev/null | tail -1)
  if [ "$alive" = "0" ]; then
    log "HEAD CONTAINER DIED after $(( $(date +%s) - T0 ))s"
    ./spark.sh $HEAD "docker logs $NAME 2>&1 | grep -iE 'error|exception|killed|nccl|died' | tail -15"
    exit 1
  fi
  [ $((i % 8)) -eq 0 ] && log "still loading ($(( $(date +%s) - T0 ))s) http=$code"
done
log "TIMEOUT after 30 min"; exit 1
