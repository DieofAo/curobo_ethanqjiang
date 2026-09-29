#!/usr/bin/env bash
# =============================================================================
# 在 rviz 中播放已规划的 XTrainer 轨迹 (ROS1 noetic)
#
# 用法:
#   ./run_rviz.sh                                  # 自动用各 results* 下最新成功结果
#   ./run_rviz.sh --traj results/20260828_120000
#   ./run_rviz.sh --speed 0.5 --loop
#   ./run_rviz.sh --no-rviz                        # 只发布, 不起 rviz
#   ./run_rviz.sh --no-scene                       # 不加载料台装配体, 只看单臂
#   ./run_rviz.sh --arm right                      # 换成右侧挂载位为被规划的臂
#   ./run_rviz.sh --mounts config/cad_mounts_same_side.yaml
#                                                  # 指定双臂挂载位姿
#   ./run_rviz.sh --mounts config/cad_mounts_right_y_plus_0p4.yaml
#                                                  # 二号臂沿场景原 +Y 平移 0.4m
#   ./run_rviz.sh --no-static-arm                  # 只显示料台, 不显示另一条静态臂
#   ./run_rviz.sh --both-arms                      # 让场景中第二条臂可动;
#                                                  # 6-DOF 旧轨迹仍镜像同动,
#                                                  # 12-DOF 轨迹则按各臂独立关节播放
#   ./run_rviz.sh --both-arms --dual-check         # 播放前现场做双臂碰撞检测,
#                                                  撞到碰撞帧即暂停(需 conda curobo)
#                                                  # 轨迹目录已存在检测报告时自动使用
#
# place 扫描点云 (默认自动发布最新一次扫描: /place_scan/top1_points 为
# top1 成功点, /place_scan/stage12_points 为阶段1/2可行但阶段3未覆盖点):
#   ./run_rviz.sh --no-scan-cloud                  # 不发布扫描点云
#   ./run_rviz.sh --scan-dir results_place_scan/20260901_181723
#   ./run_rviz.sh --scan-rank 3                    # 看第3名 (透传给点云脚本)
#
# 其余参数会透传给 play_trajectory_ros.py
# =============================================================================
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"

TRAJ=""
SPEED="1.0"
LOOP="false"
START_RVIZ="true"
SHOW_SCENE="true"
SHOW_STATIC_ARM="true"
ACTIVE_ARM="left"
BOTH_ARMS="false"
MOUNTS_FILE=""
MOUNTS_SOURCE="default"
DUAL_CHECK="false"
DUAL_MARGIN=""
SCAN_CLOUD="true"
SCAN_DIR=""
SCAN_EXTRA=()
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --traj)           TRAJ="$2"; shift 2 ;;
    --speed)          SPEED="$2"; shift 2 ;;
    --loop)           LOOP="true"; shift ;;
    --no-rviz)        START_RVIZ="false"; shift ;;
    --no-scene)       SHOW_SCENE="false"; shift ;;
    --no-static-arm)  SHOW_STATIC_ARM="false"; shift ;;
    --arm)            ACTIVE_ARM="$2"; shift 2 ;;
    --mounts)         MOUNTS_FILE="$2"; MOUNTS_SOURCE="--mounts"; shift 2 ;;
    --both-arms)      BOTH_ARMS="true"; shift ;;
    --dual-check)     DUAL_CHECK="true"; shift ;;
    --dual-margin)    DUAL_MARGIN="$2"; shift 2 ;;
    --no-scan-cloud)  SCAN_CLOUD="false"; shift ;;
    --scan-dir)       SCAN_CLOUD="true"; SCAN_DIR="$2"; shift 2 ;;
    --scan-rank)      SCAN_EXTRA+=("--rank" "$2"); shift 2 ;;
    *)                EXTRA+=("$1"); shift ;;
  esac
done
# 未指定轨迹则取各类 results 下「含有 trajectory.npz 且未失败」
# 的最新目录。用 npz 修改时间跨目录比较，避免目录名字典序影响结果。
# 注意 1: 规划失败目录一定有 plan_failed.json，也可能保留诊断 trajectory.npz；
#         无论自动选择还是显式 --traj 都必须拒绝，否则可能播放已知不安全轨迹。
# 注意 2: 多目标变体模式下轨迹在 results/<时间戳>/<变体名>/ 里, 因此要向下多找一层。
if [[ -z "$TRAJ" ]]; then
  RESULT_ROOTS=()
  for result_root in \
      "${TASK_ROOT}/results" \
      "${TASK_ROOT}/results_pick_place" \
      "${TASK_ROOT}/results_dual_pick_place"; do
    [[ -d "$result_root" ]] && RESULT_ROOTS+=("$result_root")
  done
  LATEST_ANY=""
  LATEST_TRAJ_RECORD=""
  if [[ ${#RESULT_ROOTS[@]} -gt 0 ]]; then
    LATEST_ANY_RECORD="$(
      find "${RESULT_ROOTS[@]}" -maxdepth 1 -mindepth 1 -type d \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1
    )"
    [[ -n "$LATEST_ANY_RECORD" ]] && LATEST_ANY="${LATEST_ANY_RECORD#* }"
    LATEST_TRAJ_RECORD="$(
      find "${RESULT_ROOTS[@]}" -maxdepth 3 -mindepth 2 -type f \
        -name trajectory.npz -printf '%T@ %p\n' 2>/dev/null \
      | sort -n \
      | while IFS= read -r record; do
          candidate="${record#* }"
          [[ -f "$(dirname "$candidate")/plan_failed.json" ]] || printf '%s\n' "$record"
        done \
      | tail -1
    )"
  fi
  TRAJ="${LATEST_TRAJ_RECORD#* }"
  if [[ -z "$LATEST_TRAJ_RECORD" ]]; then
    echo "[ERR] results/results_pick_place/results_dual_pick_place 下没有成功的轨迹" >&2
    if [[ -n "$LATEST_ANY" && -f "${LATEST_ANY}/plan_failed.json" ]]; then
      echo "[ERR] 最近一次规划失败: ${LATEST_ANY}/plan_failed.json" >&2
    fi
    exit 1
  fi
  TRAJ="$(dirname "$TRAJ")"
  echo "[run_rviz] 自动选择最新轨迹: ${TRAJ}"
  # 若同一批里还有其他变体, 列出来提示可切换
  SUM="$(dirname "$TRAJ")/variants_summary.json"
  if [[ -f "$SUM" ]]; then
    echo "[run_rviz] 该批次为多目标变体, 同批可选:"
    find "$(dirname "$TRAJ")" -maxdepth 2 -name trajectory.npz 2>/dev/null | sort | while read -r f; do
      d="$(dirname "$f")"
      mark=" "; [[ "$d" == "$TRAJ" ]] && mark="*"
      echo "[run_rviz]   ${mark} $(basename "$d")"
    done
    echo "[run_rviz]   用 --traj <目录> 指定其他变体"
  fi
  # 若存在更新的失败目录, 说明用户最近改了配置但没规划成功, 必须明确告警
  if [[ -n "$LATEST_ANY" && "$LATEST_ANY" != "$TRAJ" && -f "${LATEST_ANY}/plan_failed.json" ]]; then
    echo "[run_rviz] ############################################################" >&2
    echo "[run_rviz] 警告: 存在更新的失败结果 $(basename "$LATEST_ANY")" >&2
    echo "[run_rviz]       即将播放的是较早的成功轨迹 $(basename "$TRAJ")," >&2
    echo "[run_rviz]       它不包含你最近对配置的修改!" >&2
    echo "[run_rviz]       失败详情: ${LATEST_ANY}/plan_failed.json" >&2
    echo "[run_rviz] ############################################################" >&2
    sleep 3
  fi
fi
[[ "$TRAJ" != /* ]] && TRAJ="$(cd "$(dirname "$TRAJ")" && pwd)/$(basename "$TRAJ")"
if [[ ! -f "${TRAJ}/trajectory.npz" && ! -f "$TRAJ" ]]; then
  echo "[ERR] 找不到 ${TRAJ}/trajectory.npz" >&2
  [[ -f "${TRAJ}/plan_failed.json" ]] && \
    echo "[ERR] 该目录是失败结果, 详见 ${TRAJ}/plan_failed.json" >&2
  exit 1
fi

# 新的双臂轨迹会把规划时的 mounts 写进 meta。未显式传 --mounts 时优先
# 沿用它，使同侧规划直接播放时不会退回旧的对角挂载。旧轨迹无该字段
# 时仍使用 cad_mounts.yaml，保持原行为。
TRAJ_DIR="$TRAJ"
[[ -f "$TRAJ_DIR" ]] && TRAJ_DIR="$(dirname "$TRAJ_DIR")"
if [[ -f "${TRAJ_DIR}/plan_failed.json" ]]; then
  echo "[ERR] 拒绝播放失败/不安全轨迹: ${TRAJ_DIR}" >&2
  echo "[ERR] 失败详情: ${TRAJ_DIR}/plan_failed.json" >&2
  exit 1
fi
if [[ -z "$MOUNTS_FILE" ]]; then
  TRAJ_META="${TRAJ_DIR}/trajectory_meta.json"
  MOUNTS_FROM_META=""
  if [[ -f "$TRAJ_META" ]]; then
    MOUNTS_FROM_META="$(
      python3 -c 'import json, sys; print((json.load(open(sys.argv[1])).get("robot") or {}).get("mounts") or "")' \
        "$TRAJ_META" 2>/dev/null
    )" || MOUNTS_FROM_META=""
  fi
  if [[ -n "$MOUNTS_FROM_META" ]]; then
    MOUNTS_FILE="$MOUNTS_FROM_META"
    MOUNTS_SOURCE="trajectory_meta.json"
  else
    MOUNTS_FILE="${TASK_ROOT}/config/cad_mounts_right_y_plus_0p4.yaml"
  fi
fi

# 显式 --mounts 的相对路径先按当前目录解析；meta 内的项目路径则确定性地
# 按「仓库根」、「task 根」解析，避免启动时的 cwd 意外改变场景。
if [[ "$MOUNTS_FILE" != /* ]]; then
  if [[ "$MOUNTS_SOURCE" == "--mounts" && -f "$MOUNTS_FILE" ]]; then
    MOUNTS_FILE="$(cd "$(dirname "$MOUNTS_FILE")" && pwd)/$(basename "$MOUNTS_FILE")"
  elif [[ -f "${REPO_ROOT}/${MOUNTS_FILE}" ]]; then
    MOUNTS_FILE="${REPO_ROOT}/${MOUNTS_FILE}"
  elif [[ -f "${TASK_ROOT}/${MOUNTS_FILE}" ]]; then
    MOUNTS_FILE="${TASK_ROOT}/${MOUNTS_FILE}"
  fi
fi
if [[ ! -f "$MOUNTS_FILE" && \
      ( "$SHOW_SCENE" == "true" || "$DUAL_CHECK" == "true" ) ]]; then
  echo "[ERR] 找不到 mounts 配置: ${MOUNTS_FILE}" >&2
  exit 1
fi

# 独立双臂轨迹已在 npz 中带 second_J_* 12 个 DOF。即使用户没有
# 显式写 --both-arms，也必须让 scene URDF 中第二条臂保留活动关节。
TRAJ_NPZ="$TRAJ"
[[ -d "$TRAJ_NPZ" ]] && TRAJ_NPZ="${TRAJ_NPZ}/trajectory.npz"
if [[ "$BOTH_ARMS" != "true" ]] && python3 - "$TRAJ_NPZ" <<'PY'
import sys

import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as archive:
    raw_names = archive["joint_names"]
names = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
         for x in raw_names]
sys.exit(0 if len(names) >= 12 and any(x.startswith("second_") for x in names) else 1)
PY
then
  BOTH_ARMS="true"
  echo "[run_rviz] 检测到独立 12-DOF 双臂轨迹, 已自动启用 --both-arms"
fi

# place 扫描点云: 未指定目录则自动取 results_place_scan 下最新一次。
# 只有 stage12 的目录也能发 (退化为 IK 可行点); 目录无效时自动关闭并提示。
if [[ "$SCAN_CLOUD" == "true" ]]; then
  if [[ -z "$SCAN_DIR" ]]; then
    LATEST_S3="$(find "${TASK_ROOT}/results_place_scan" -maxdepth 2 -mindepth 2 \
                -name 'stage*_result.json' 2>/dev/null | sort | tail -1)"
    [[ -n "$LATEST_S3" ]] && SCAN_DIR="$(dirname "$LATEST_S3")"
  fi
  if [[ -n "$SCAN_DIR" && -f "${SCAN_DIR}/stage12_result.json" ]]; then
    [[ "$SCAN_DIR" != /* ]] && SCAN_DIR="$(cd "$SCAN_DIR" && pwd)"
    echo "[run_rviz] 扫描点云: ${SCAN_DIR} -> /place_scan/top1_points + /place_scan/stage12_points"
  else
    [[ -n "$SCAN_DIR" ]] && \
      echo "[run_rviz] 警告: ${SCAN_DIR} 缺少 stage12_result.json, 不发点云" >&2
    SCAN_DIR=""
  fi
fi

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "[ERR] 找不到 ${ROS_SETUP}, 用 ROS_SETUP=... 指定" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ROS_SETUP"

# 若无 master 则本地起一个
if ! rostopic list >/dev/null 2>&1; then
  echo "[run_rviz] 未检测到 roscore, 正在本地启动 ..."
  roscore >/tmp/xtrainer_roscore.log 2>&1 &
  ROSCORE_PID=$!
  trap 'kill ${ROSCORE_PID} 2>/dev/null || true' EXIT
  for _ in $(seq 1 30); do
    rostopic list >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

# 双臂同动需要第二条臂存在, --no-static-arm 与之互斥
if [[ "$BOTH_ARMS" == "true" && "$SHOW_STATIC_ARM" != "true" ]]; then
  echo "[run_rviz] 提示: --both-arms 需要第二条臂, 已忽略 --no-static-arm" >&2
  SHOW_STATIC_ARM="true"
fi

# 双臂碰撞暂停: 支持旧的 6-DOF 镜像轨迹与新的 12-DOF 独立轨迹。
DUAL_JSON=""
if [[ "$BOTH_ARMS" == "true" ]]; then
  DUAL_JSON="${TRAJ_DIR}/dual_arm_collision.json"
  if [[ ! -f "$DUAL_JSON" || "$DUAL_CHECK" == "true" ]]; then
    if [[ "$DUAL_CHECK" == "true" ]]; then
      # 现场重算: conda curobo 环境 (activate 会设好 PYTHONPATH, 不能 unset)
      CONDA_ENV="${CONDA_ENV:-curobo}"
      CONDA_SH=""
      for p in "$HOME/miniconda3/etc/profile.d/conda.sh" \
               "$HOME/anaconda3/etc/profile.d/conda.sh" \
               "/opt/conda/etc/profile.d/conda.sh"; do
        [[ -f "$p" ]] && CONDA_SH="$p" && break
      done
      if [[ -z "$CONDA_SH" ]]; then
        echo "[run_rviz] 找不到 conda.sh, 无法现场检测, 跳过" >&2
      else
        CHK_ARGS=()
        [[ -n "$DUAL_MARGIN" ]] && CHK_ARGS+=(--margin-mm "$DUAL_MARGIN")
        set +u
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
        set -u
        ( cd "$TASK_ROOT" && python scripts/check_dual_arm_collision.py \
            "$TRAJ" --arm "$ACTIVE_ARM" --mounts "$MOUNTS_FILE" \
            ${CHK_ARGS[@]+"${CHK_ARGS[@]}"} ) \
          || echo "[run_rviz] 双臂检测有碰撞, 播放时将暂停在碰撞帧" >&2
        # 检测脚本 -p 输出覆盖, 确保路径还是这个
        DUAL_JSON="${TRAJ_DIR}/dual_arm_collision.json"
      fi
    fi
  fi
  if [[ ! -f "$DUAL_JSON" ]]; then
    echo "[run_rviz] 提示: 无双臂碰撞报告, 不启用碰撞暂停" >&2
    DUAL_JSON=""
  fi
fi

echo "[run_rviz] traj=${TRAJ}  speed=${SPEED}  loop=${LOOP}  rviz=${START_RVIZ}"
echo "[run_rviz] scene=${SHOW_SCENE}  active_arm=${ACTIVE_ARM}  static_arm=${SHOW_STATIC_ARM}  both_arms=${BOTH_ARMS}"
echo "[run_rviz] mounts=${MOUNTS_FILE}  source=${MOUNTS_SOURCE}"
PLAYER_ARGS=""
[[ ${#EXTRA[@]} -gt 0 ]] && PLAYER_ARGS="${EXTRA[*]}"
[[ -n "$DUAL_JSON" ]] && PLAYER_ARGS="${PLAYER_ARGS} --collision-json ${DUAL_JSON}"

SCAN_ARGS=""
[[ ${#SCAN_EXTRA[@]} -gt 0 ]] && SCAN_ARGS="${SCAN_EXTRA[*]}"

exec roslaunch "${TASK_ROOT}/launch/display_xtrainer_traj.launch" \
  repo_root:="${REPO_ROOT}" \
  task_root:="${TASK_ROOT}" \
  traj:="${TRAJ}" \
  speed:="${SPEED}" \
  loop:="${LOOP}" \
  start_rviz:="${START_RVIZ}" \
  show_scene:="${SHOW_SCENE}" \
  active_arm:="${ACTIVE_ARM}" \
  mounts_file:="${MOUNTS_FILE}" \
  show_static_arm:="${SHOW_STATIC_ARM}" \
  active_both:="${BOTH_ARMS}" \
  player_extra_args:="${PLAYER_ARGS}" \
  scan_dir:="${SCAN_DIR}" \
  scan_extra_args:="${SCAN_ARGS}"
