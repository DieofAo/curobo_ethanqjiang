#!/usr/bin/env bash
# Confirmed mount: new LINK_0 +Z = old task +X. Single arm, TCP 0.19 m.
# Default: recorded overhead preset, 20x20 grid, timestamped output directories.
# Smoke: bash run_pick_place_overhead.sh --rows 3 --cols 3
# Display: bash run_overhead_rviz.sh --traj <result directory>
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for arg in "$@"; do
  if [[ "$arg" == "--dual" ]]; then
    echo "[ERR] overhead preset is single-arm only" >&2
    exit 2
  fi
done
exec bash "$TASK_ROOT/run_pick_place.sh" \
  --config "$TASK_ROOT/config/pick_place_overhead_default.json" "$@"
