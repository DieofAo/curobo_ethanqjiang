#!/usr/bin/env bash
# Display a single XTrainer mounted above the unchanged pick/place task frame.
#
# Examples:
#   ./run_overhead_rviz.sh --traj results_pick_place/<run>
#   ./run_overhead_rviz.sh --config config/pick_place_overhead.yaml
#   ./run_overhead_rviz.sh --traj <run> --screenshot <run>/rviz_overhead.png \
#       --exit-after-screenshot
#   ./run_overhead_rviz.sh --config <yaml> --validate-only
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
BUILDER="${TASK_ROOT}/scripts/build_overhead_scene_urdf.py"
RVIZ_CONFIG="${TASK_ROOT}/config/xtrainer_overhead.rviz"
OVERHEAD_LAUNCH="${TASK_ROOT}/launch/display_overhead_traj.launch"

TRAJ=""
CONFIG=""
SPEED="1.0"
LOOP="false"
START_RVIZ="true"
VALIDATE_ONLY="false"
DISPLAY_TARGET="${DISPLAY:-:0}"
SCREENSHOT=""
SCREENSHOT_DELAY="7"
EXIT_AFTER_SCREENSHOT="false"
PLAYER_EXTRA=()
PREVIEW_JOINT_DEG=()
ASSEMBLY_ARGS=()

usage() {
  cat <<'EOF'
用法:
  ./run_overhead_rviz.sh --traj <结果目录|trajectory.npz>
  ./run_overhead_rviz.sh --config <overhead pick/place yaml>

选项:
  --speed N                  轨迹播放速度，默认 1.0
  --display-hz N             时间戳模式显示刷新上限，默认 50；高倍速跳过显示帧
  --start-percent N          从原轨迹时间进度N%开始，范围0–100
  --start-time SEC           从原轨迹起点后的SEC秒开始，与倍速无关
  --start-frame N            从轨迹采样帧N开始（0基）
  --start-item N             从第N个原物料case的循环起点开始（1基）
  --joint-deg J1 J2 J3 J4 J5 J6  config-only 静态预览关节角（度）
  --loop                     循环播放
  --display DISPLAY          X display，默认继承 DISPLAY，否则 :0
  --no-rviz                  只发布模型/轨迹
  --no-assembly              不显示原料台 CAD 装配体（默认显示；仅可视化）
  --assembly-mounts FILE     装配体显示配置，默认已去掉旧 LINK0 的额外 Z 偏转和 Y 平移
  --assembly-arm left|right  旧 LINK_0 对应的 CAD 挂载，默认 left；不添加第二臂
  --assembly-mesh FILE       自定义装配体 STL，按毫米缩放至米
  --validate-only            只校验 config、M/C、TCP 和生成的 URDF
  --screenshot FILE.png      RViz 启动后捕获其窗口（需要 xwininfo/PyGObject）
  --screenshot-delay SEC     找到窗口后再等待多久截图，默认 7 秒
  --exit-after-screenshot    截图后关闭本次 roslaunch；否则继续显示
  -h, --help                 显示本说明

其余参数透传给 play_trajectory_ros.py。config 预览使用零关节角；正式效果请
优先给 --traj，让安装模型和轨迹都读取该结果中保存的同一份 resolved config。
四种 --start-* 起点选择互斥；--loop 每次都从选择的起点开始。只影响可视化。
EOF
}

need_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "[ERR] $1 需要参数" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --traj) need_value "$@"; TRAJ="$2"; shift 2 ;;
    --config) need_value "$@"; CONFIG="$2"; shift 2 ;;
    --joint-deg)
      if [[ $# -lt 7 ]]; then echo "[ERR] --joint-deg 需要 6 个角度" >&2; exit 2; fi
      PREVIEW_JOINT_DEG=("${@:2:6}"); shift 7 ;;
    --speed) need_value "$@"; SPEED="$2"; shift 2 ;;
    --loop) LOOP="true"; shift ;;
    --display) need_value "$@"; DISPLAY_TARGET="$2"; shift 2 ;;
    --no-rviz) START_RVIZ="false"; shift ;;
    --no-assembly) ASSEMBLY_ARGS+=("$1"); shift ;;
    --assembly-mounts|--assembly-arm|--assembly-mesh)
      need_value "$@"; ASSEMBLY_ARGS+=("$1" "$2"); shift 2 ;;
    --validate-only) VALIDATE_ONLY="true"; shift ;;
    --screenshot) need_value "$@"; SCREENSHOT="$2"; shift 2 ;;
    --screenshot-delay) need_value "$@"; SCREENSHOT_DELAY="$2"; shift 2 ;;
    --exit-after-screenshot) EXIT_AFTER_SCREENSHOT="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) PLAYER_EXTRA+=("$1"); shift ;;
  esac
done

if [[ -n "$TRAJ" && -n "$CONFIG" ]]; then
  echo "[ERR] --traj 与 --config 只能给一个；结果 metadata 已保存 resolved config" >&2
  exit 2
fi
if [[ -n "$TRAJ" && ${#PREVIEW_JOINT_DEG[@]} -gt 0 ]]; then
  echo "[ERR] --joint-deg 仅用于 config 静态预览，不能覆盖真实轨迹" >&2
  exit 2
fi
if [[ -z "$TRAJ" && -z "$CONFIG" ]]; then
  echo "[ERR] 必须给 --traj 或 --config" >&2
  usage >&2
  exit 2
fi
if [[ -n "$SCREENSHOT" && "$START_RVIZ" != "true" ]]; then
  echo "[ERR] --screenshot 不能与 --no-rviz 同用" >&2
  exit 2
fi
if [[ ! -f "$BUILDER" || ! -f "$RVIZ_CONFIG" || ! -f "$OVERHEAD_LAUNCH" ]]; then
  echo "[ERR] overhead RViz 文件不完整: ${BUILDER} / ${RVIZ_CONFIG} / ${OVERHEAD_LAUNCH}" >&2
  exit 1
fi

# roslaunch 的子进程 cwd 不保证等于调用目录；轨迹必须先规范成绝对路径。
if [[ -n "$TRAJ" ]]; then
  if [[ "$TRAJ" != /* ]]; then
    RESOLVED_TRAJ=""
    for candidate in "$(pwd)/${TRAJ}" "${TASK_ROOT}/${TRAJ}" "${REPO_ROOT}/${TRAJ}"; do
      if [[ -e "$candidate" ]]; then
        RESOLVED_TRAJ="$candidate"
        break
      fi
    done
    TRAJ="$RESOLVED_TRAJ"
  fi
  if [[ -z "$TRAJ" || ! -e "$TRAJ" ]]; then
    echo "[ERR] 找不到轨迹: ${TRAJ:-<unresolved>}" >&2
    exit 1
  fi
  if [[ -d "$TRAJ" ]]; then
    TRAJ="$(cd "$TRAJ" && pwd)"
    if [[ ! -f "${TRAJ}/trajectory.npz" ]]; then
      echo "[ERR] 结果目录缺少 trajectory.npz: ${TRAJ}" >&2
      exit 1
    fi
    if [[ -f "${TRAJ}/plan_failed.json" ]]; then
      echo "[ERR] 拒绝播放失败/不安全轨迹: ${TRAJ}" >&2
      exit 1
    fi
  else
    TRAJ="$(cd "$(dirname "$TRAJ")" && pwd)/$(basename "$TRAJ")"
    if [[ "$(basename "$TRAJ")" != "trajectory.npz" ]]; then
      echo "[ERR] --traj 文件必须是 trajectory.npz: ${TRAJ}" >&2
      exit 1
    fi
    if [[ -f "$(dirname "$TRAJ")/plan_failed.json" ]]; then
      echo "[ERR] 拒绝播放失败/不安全轨迹: $(dirname "$TRAJ")" >&2
      exit 1
    fi
  fi
fi

SOURCE_ARGS=()
if [[ -n "$TRAJ" ]]; then
  SOURCE_ARGS=(--traj "$TRAJ")
else
  SOURCE_ARGS=(--config "$CONFIG")
fi

SCENE_URDF="$(mktemp /tmp/xtrainer_overhead_scene.XXXXXX.urdf)"
ROSCORE_PID=""
LAUNCH_PID=""
cleanup() {
  if [[ -n "$LAUNCH_PID" ]]; then
    kill "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
  fi
  if [[ -n "$ROSCORE_PID" ]]; then
    kill "$ROSCORE_PID" 2>/dev/null || true
    wait "$ROSCORE_PID" 2>/dev/null || true
  fi
  rm -f "$SCENE_URDF"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

python3 "$BUILDER" "${SOURCE_ARGS[@]}" "${ASSEMBLY_ARGS[@]}" --output "$SCENE_URDF"
SETTINGS="$(python3 "$BUILDER" "${SOURCE_ARGS[@]}" "${ASSEMBLY_ARGS[@]}" --print-settings)"
echo "[overhead rviz] resolved scene:"
printf '%s\n' "$SETTINGS"

TASK_FRAME="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["task_frame"])' "$SETTINGS")"
BASE_LINK="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["base_link"])' "$SETTINGS")"
if [[ "$TASK_FRAME" != "task_world" || "$BASE_LINK" != "LINK_0" ]]; then
  echo "[ERR] 当前 RViz preset 要求 task_frame=task_world、base_link=LINK_0，" \
       "实际 ${TASK_FRAME}/${BASE_LINK}" >&2
  exit 2
fi

if command -v check_urdf >/dev/null 2>&1; then
  check_urdf "$SCENE_URDF" >/dev/null
fi
echo "[overhead rviz] URDF 校验通过: task_world -> LINK_0；播放器 frame=LINK_0"

if [[ "$VALIDATE_ONLY" == "true" ]]; then
  exit 0
fi
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "[ERR] 找不到 ${ROS_SETUP}，可用 ROS_SETUP=... 指定" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ROS_SETUP"

if ! rostopic list >/dev/null 2>&1; then
  echo "[overhead rviz] 未检测到 roscore，正在本地启动 ..."
  roscore >/tmp/xtrainer_overhead_roscore.log 2>&1 &
  ROSCORE_PID=$!
  for _ in $(seq 1 30); do
    rostopic list >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

START_PLAYER="false"
TRAJ_ARG="${TASK_ROOT}/results/latest"
if [[ -n "$TRAJ" ]]; then
  START_PLAYER="true"
  TRAJ_ARG="$TRAJ"
fi
PLAYER_ARGS=""
PREVIEW_ZEROS="{}"
if [[ ${#PREVIEW_JOINT_DEG[@]} -gt 0 ]]; then
  PREVIEW_ZEROS="$(python3 -c 'import json,math,sys; q=[float(v) for v in sys.argv[1:]]; assert len(q)==6 and all(math.isfinite(v) for v in q); print(json.dumps({f"J_{i+1}":math.radians(v) for i,v in enumerate(q)}))' "${PREVIEW_JOINT_DEG[@]}")"
fi
if [[ ${#PLAYER_EXTRA[@]} -gt 0 ]]; then
  PLAYER_ARGS="${PLAYER_EXTRA[*]}"
fi

LAUNCH_ARGS=(
  "${OVERHEAD_LAUNCH}"
  repo_root:="${REPO_ROOT}"
  task_root:="${TASK_ROOT}"
  traj:="${TRAJ_ARG}"
  speed:="${SPEED}"
  loop:="${LOOP}"
  start_rviz:="${START_RVIZ}"
  start_player:="${START_PLAYER}"
  urdf_file:="${SCENE_URDF}"
  rviz_config:="${RVIZ_CONFIG}"
  base_frame:=LINK_0
  task_frame:=task_world
  player_extra_args:="${PLAYER_ARGS}"
  preview_joint_positions:="${PREVIEW_ZEROS}"
)

echo "[overhead rviz] display=${DISPLAY_TARGET} player=${START_PLAYER} traj=${TRAJ_ARG}"

if [[ -z "$SCREENSHOT" ]]; then
  env DISPLAY="$DISPLAY_TARGET" roslaunch "${LAUNCH_ARGS[@]}" &
  LAUNCH_PID=$!
  wait "$LAUNCH_PID"
  LAUNCH_PID=""
  exit 0
fi

for tool in xwininfo python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "[ERR] --screenshot 缺少命令: ${tool}" >&2
    exit 1
  fi
done

if [[ "$SCREENSHOT" != /* ]]; then
  SCREENSHOT="$(pwd)/${SCREENSHOT}"
fi
mkdir -p "$(dirname "$SCREENSHOT")"

existing_rviz_windows() {
  env DISPLAY="$DISPLAY_TARGET" xwininfo -root -tree 2>/dev/null \
    | awk 'tolower($0) ~ /rviz/ && $1 ~ /^0x/ {print $1}'
}

BEFORE_WINDOWS="$(existing_rviz_windows || true)"
env DISPLAY="$DISPLAY_TARGET" roslaunch "${LAUNCH_ARGS[@]}" &
LAUNCH_PID=$!

RVIZ_WINDOW=""
for _ in $(seq 1 60); do
  while IFS= read -r candidate; do
    [[ -z "$candidate" ]] && continue
    if ! grep -Fxq "$candidate" <<<"$BEFORE_WINDOWS"; then
      RVIZ_WINDOW="$candidate"
      break
    fi
  done < <(existing_rviz_windows || true)
  [[ -n "$RVIZ_WINDOW" ]] && break
  kill -0 "$LAUNCH_PID" 2>/dev/null || break
  sleep 0.5
done
if [[ -z "$RVIZ_WINDOW" ]]; then
  echo "[ERR] 30 秒内没有找到新 RViz 窗口；检查 DISPLAY/XAUTHORITY" >&2
  exit 1
fi

sleep "$SCREENSHOT_DELAY"
# Qt creates short-lived startup windows. Resolve again after rendering has
# settled instead of keeping an ID that may already have been destroyed.
RVIZ_WINDOW=""
while IFS= read -r candidate; do
  [[ -z "$candidate" ]] && continue
  if ! grep -Fxq "$candidate" <<<"$BEFORE_WINDOWS"; then
    if env DISPLAY="$DISPLAY_TARGET" xwininfo -id "$candidate" 2>/dev/null \
        | awk '/Width:/ {w=$2} /Height:/ {h=$2} END {exit !(w>=500 && h>=300)}'; then
      RVIZ_WINDOW="$candidate"
      break
    fi
  fi
done < <(existing_rviz_windows || true)
if [[ -z "$RVIZ_WINDOW" ]]; then
  echo "[ERR] 没有找到已渲染的 RViz 主窗口" >&2
  exit 1
fi
echo "[overhead rviz] 捕获 RViz window=${RVIZ_WINDOW} -> ${SCREENSHOT}"
env DISPLAY="$DISPLAY_TARGET" python3 - "$RVIZ_WINDOW" "$SCREENSHOT" <<'PY'
import sys
import gi
gi.require_version("Gdk", "3.0")
gi.require_version("GdkX11", "3.0")
from gi.repository import Gdk, GdkX11
display = Gdk.Display.get_default()
window = GdkX11.X11Window.foreign_new_for_display(display, int(sys.argv[1], 16))
if window is None:
    raise RuntimeError("RViz window no longer exists")
pixels = Gdk.pixbuf_get_from_window(window, 0, 0, window.get_width(), window.get_height())
if pixels is None:
    raise RuntimeError("Cannot capture RViz window")
pixels.savev(sys.argv[2], "png", [], [])
PY
if [[ ! -s "$SCREENSHOT" ]]; then
  echo "[ERR] 截图未生成: ${SCREENSHOT}" >&2
  exit 1
fi
echo "[overhead rviz] screenshot=${SCREENSHOT}"

if [[ "$EXIT_AFTER_SCREENSHOT" == "true" ]]; then
  exit 0
fi
wait "$LAUNCH_PID"
LAUNCH_PID=""
