#!/usr/bin/env bash
# 独立 12-DOF 双臂交错 pick & place 的便捷入口。
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${TASK_ROOT}/run_pick_place.sh" --dual "$@"
