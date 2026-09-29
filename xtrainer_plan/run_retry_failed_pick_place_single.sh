#!/usr/bin/env bash
# Retry one result directory's failed pick-and-place items with the single-arm model.
set -eo pipefail

unset PYTHONPATH || true
set +u
# shellcheck disable=SC1091
source /home/ethanqjiang/miniconda3/etc/profile.d/conda.sh
conda activate curobo
set -u

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${TASK_ROOT}/scripts/retry_failed_pick_place_single.py" "$@"
