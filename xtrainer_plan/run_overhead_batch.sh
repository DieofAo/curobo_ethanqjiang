#!/usr/bin/env bash
set -eo pipefail
unset PYTHONPATH || true
source /home/ethanqjiang/miniconda3/etc/profile.d/conda.sh
conda activate curobo
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${TASK_ROOT}/scripts/run_overhead_batch.py" "$@"
