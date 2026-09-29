#!/usr/bin/env bash
# 无轨迹静态显示同侧双臂和 dual pick/place YAML 中的两个 place TF。
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
CONFIG="${TASK_ROOT}/config/dual_pick_place_default.yaml"
DISPLAY_TARGET="${DISPLAY:-:0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)  CONFIG="$2"; shift 2 ;;
    --display) DISPLAY_TARGET="$2"; shift 2 ;;
    *)
      echo "[ERR] 未知参数: $1" >&2
      echo "用法: ./run_dual_place_rviz.sh [--config FILE] [--display :0]" >&2
      exit 2
      ;;
  esac
done

[[ "$CONFIG" != /* ]] && CONFIG="${TASK_ROOT}/${CONFIG}"
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "[ERR] 找不到 ${ROS_SETUP}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ROS_SETUP"

SETTINGS="$(
  python3 "${TASK_ROOT}/scripts/publish_dual_place_tf_ros.py" \
    --config "$CONFIG" --print-settings
)"
MOUNTS_FILE="$(python3 -c \
  'import json,sys; print(json.loads(sys.argv[1])["mounts"])' "$SETTINGS")"
BASE_FRAME="$(python3 -c \
  'import json,sys; print(json.loads(sys.argv[1])["base_frame"])' "$SETTINGS")"

echo "[dual place rviz] settings=${SETTINGS}"
exec env DISPLAY="$DISPLAY_TARGET" roslaunch \
  "${TASK_ROOT}/launch/display_xtrainer_traj.launch" \
  repo_root:="$REPO_ROOT" \
  task_root:="$TASK_ROOT" \
  start_player:=false \
  active_both:=true \
  active_arm:=left \
  show_scene:=true \
  base_frame:="$BASE_FRAME" \
  mounts_file:="$MOUNTS_FILE" \
  dual_config:="$CONFIG" \
  scan_dir:=
