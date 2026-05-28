#!/bin/bash
set -e
MODE="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/curobo/bin/python
SCRIPT=/data/workspace/curobo_ethanqjiang/workspace_rviz/sample_workspace.py
if [ "$MODE" = "smoke" ]; then
    OUT=/tmp/smoke_self
    rm -rf "$OUT"; mkdir -p "$OUT"
    exec "$PY" -u "$SCRIPT" --step 0.5 --collision self --out-dir "$OUT"
elif [ "$MODE" = "full" ]; then
    OUT=/data/workspace/curobo_ethanqjiang/workspace_rviz/runs/self_collision_step0p2
    mkdir -p "$OUT"
    exec "$PY" -u "$SCRIPT" --step 0.2 --collision self --out-dir "$OUT"
else
    echo "usage: $0 smoke|full"; exit 2
fi
