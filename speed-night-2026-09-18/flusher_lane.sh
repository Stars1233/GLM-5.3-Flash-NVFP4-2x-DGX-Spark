#!/bin/bash
# flusher_lane.sh -- unconditional page-cache flusher for the life of a serving container.
# The UMA page cache starves the NVRM allocator (NV_ERR_NO_MEMORY at 06:36 on 2026-09-18 with
# 9 GiB of reclaimable cache sitting there); threshold flushers expire or never fire. Every 20 s
# for up to 10 h: drop clean cache, and compact when MemFree < 6 GiB.
LOG=~/flusher_lane.log; t0=$(date +%s); n=0
echo "$(date +%T) flusher_lane start" >> $LOG
while [ $(( $(date +%s) - t0 )) -lt 36000 ]; do
  sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  f=$(awk '/^MemFree:/{print int($2/1048576)}' /proc/meminfo)
  [ "$f" -lt 6 ] && echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1
  n=$((n+1)); [ $((n % 30)) -eq 0 ] && echo "$(date +%T) tick $n MemFree ${f}G" >> $LOG
  sleep 20
done
