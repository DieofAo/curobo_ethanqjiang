#!/usr/bin/env bash
# 播放独立的 12-DOF 双臂 IK 候选（仅 RViz 可视化，不是可执行轨迹）。
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${TASK_ROOT}/results_dual_pick_place"
EXPORTER="${TASK_ROOT}/scripts/export_dual_ik_results.py"
IK_RESULTS_NAME="ik_results.npz"

IK_RESULTS=""
PLAN_FAILED=""
OUT_DIR=""
FRAME_SECONDS="2.0"
FRAME_SECONDS_OVERRIDDEN="false"
PLAY_ONCE="false"
RATE_OVERRIDDEN="false"
SCAN_OVERRIDDEN="false"
EXTRA=()

usage() {
  cat <<'EOF'
用法:
  ./run_dual_ik_rviz.sh
  ./run_dual_ik_rviz.sh --ik-results <ik_results.npz|目录>
  ./run_dual_ik_rviz.sh --plan-failed <plan_failed.json> [--out <目录>]

选项:
  --frame-seconds SEC  导出文件中的候选间隔，默认 2.0 秒
  --once               只播一轮；默认循环
  -h, --help           显示帮助

其余参数原样传给 run_rviz.sh。默认追加:
  --rate-hz 0.5 --loop --no-scan-cloud

注意：每帧是独立 IK 构型，帧间没有轨迹规划或碰撞检查，只能可视化。
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
    --ik-results)
      need_value "$@"
      IK_RESULTS="$2"
      shift 2
      ;;
    --plan-failed)
      need_value "$@"
      PLAN_FAILED="$2"
      shift 2
      ;;
    --out)
      need_value "$@"
      OUT_DIR="$2"
      shift 2
      ;;
    --frame-seconds)
      need_value "$@"
      FRAME_SECONDS="$2"
      FRAME_SECONDS_OVERRIDDEN="true"
      shift 2
      ;;
    --once)
      PLAY_ONCE="true"
      shift
      ;;
    --rate-hz)
      need_value "$@"
      RATE_OVERRIDDEN="true"
      EXTRA+=("$1" "$2")
      shift 2
      ;;
    --scan-dir)
      need_value "$@"
      SCAN_OVERRIDDEN="true"
      EXTRA+=("$1" "$2")
      shift 2
      ;;
    --no-scan-cloud)
      SCAN_OVERRIDDEN="true"
      EXTRA+=("$1")
      shift
      ;;
    --traj)
      echo "[ERR] 请用 --ik-results，不要向此入口传 --traj" >&2
      exit 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$IK_RESULTS" && -n "$PLAN_FAILED" ]]; then
  echo "[ERR] --ik-results 与 --plan-failed 不能同时使用" >&2
  exit 2
fi
if [[ -n "$IK_RESULTS" && -n "$OUT_DIR" ]]; then
  echo "[ERR] --out 只与 --plan-failed/自动导出一起使用" >&2
  exit 2
fi
if [[ ! -f "$EXPORTER" ]]; then
  echo "[ERR] 找不到导出脚本: ${EXPORTER}" >&2
  exit 1
fi

# 未显式指定来源时，优先播放最近已经导出的 IK artifact。文件名特意不是
# trajectory.npz，因此不会进入 run_rviz.sh 的成功轨迹自动选择逻辑。
if [[ -z "$IK_RESULTS" && -z "$PLAN_FAILED" && -d "$RESULT_ROOT" ]]; then
  LATEST_IK_RECORD="$({
    find "$RESULT_ROOT" -maxdepth 4 -mindepth 2 -type f \
      -name "$IK_RESULTS_NAME" -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -n | tail -1)"
  if [[ -n "$LATEST_IK_RECORD" ]]; then
    IK_RESULTS="${LATEST_IK_RECORD#* }"
    echo "[dual IK rviz] 自动选择最新 IK 结果: ${IK_RESULTS}"
  fi
fi

# 没有现成 artifact 时，跳过 planning_in_progress 等半成品，选择最新一个
# 真正含 pipeline_attempts[*].home_root 的完整失败记录。
if [[ -z "$IK_RESULTS" && -z "$PLAN_FAILED" ]]; then
  PLAN_FAILED="$(python3 - "$RESULT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if root.is_dir():
    candidates = sorted(
        root.glob("*/plan_failed.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as stream:
                record = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("stage") == "planning_in_progress":
            continue
        attempts = record.get("pipeline_attempts")
        if not isinstance(attempts, list):
            continue
        if any(
            isinstance(attempt, dict)
            and isinstance(attempt.get("home_root"), dict)
            and isinstance(attempt["home_root"].get("home_joint_deg"), list)
            for attempt in attempts
        ):
            print(path.resolve())
            break
PY
)"
  if [[ -z "$PLAN_FAILED" ]]; then
    echo "[ERR] ${RESULT_ROOT} 下既没有 ik_results.npz，也没有含 Home roots 的完整失败记录" >&2
    exit 1
  fi
  echo "[dual IK rviz] 将从最新完整失败记录导出: ${PLAN_FAILED}"
fi

if [[ -n "$PLAN_FAILED" ]]; then
  if [[ -d "$PLAN_FAILED" ]]; then
    PLAN_FAILED="${PLAN_FAILED%/}/plan_failed.json"
  fi
  EXPORT_ARGS=(--plan-failed "$PLAN_FAILED" --frame-seconds "$FRAME_SECONDS")
  if [[ -n "$OUT_DIR" ]]; then
    EXPORT_ARGS+=(--out "$OUT_DIR")
  else
    OUT_DIR="$(dirname "$PLAN_FAILED")/ik_playback"
  fi
  python3 "$EXPORTER" "${EXPORT_ARGS[@]}"
  IK_RESULTS="${OUT_DIR%/}/ik_results.npz"
fi

if [[ -d "$IK_RESULTS" ]]; then
  IK_RESULTS="${IK_RESULTS%/}/ik_results.npz"
fi
if [[ ! -f "$IK_RESULTS" ]]; then
  echo "[ERR] 找不到 IK 结果: ${IK_RESULTS}" >&2
  exit 1
fi
IK_RESULTS="$(cd "$(dirname "$IK_RESULTS")" && pwd)/$(basename "$IK_RESULTS")"

# 不只凭文件名信任 artifact。启动 ROS 前同时验证数值数组、
# companion metadata、同侧 mounts 信息和“仅可视化”安全标记。
python3 - "$IK_RESULTS" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

npz_path = Path(sys.argv[1])
meta_path = npz_path.parent / "trajectory_meta.json"
if not meta_path.is_file():
    raise SystemExit(f"[ERR] IK artifact 缺少 metadata: {meta_path}")
try:
    with meta_path.open("r", encoding="utf-8") as stream:
        meta = json.load(stream)
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"[ERR] IK metadata 无效: {exc}")
allowed_types = {
    "dual_arm_discrete_ik_candidates",
    "dual_arm_ik_prescreen_trace",
}
if meta.get("artifact_type") not in allowed_types:
    raise SystemExit(
        f"[ERR] 不是可识别的双臂 IK artifact: "
        f"{meta.get('artifact_type')!r}"
    )
if (
    meta.get("visualization_only") is not True
    or meta.get("safe_to_execute") is not False
    or meta.get("transition_checked") is not False
):
    raise SystemExit("[ERR] IK metadata 缺少完整的仅可视化/禁止执行标记")
robot = meta.get("robot") or {}
if not isinstance(robot.get("mounts"), str) or not robot["mounts"]:
    raise SystemExit("[ERR] IK metadata 缺少 robot.mounts，拒绝用默认装配猜测")
try:
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {
            "joint_names", "positions", "times",
            "ee_positions", "ee_quats_wxyz",
            "second_ee_positions", "second_ee_quats_wxyz",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"缺少数组 {missing}")
        positions = np.asarray(archive["positions"])
        times = np.asarray(archive["times"])
        names = [
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in archive["joint_names"]
        ]
        n_points = positions.shape[0] if positions.ndim == 2 else -1
        if positions.ndim != 2 or positions.shape[1] != 12 or n_points <= 0:
            raise ValueError(f"positions 必须是 (N,12)，实际 {positions.shape}")
        if len(names) != 12 or not any(name.startswith("second_") for name in names):
            raise ValueError(f"joint_names 不是 second_ 双臂 12-DOF: {names}")
        shapes = {
            "times": (n_points,),
            "ee_positions": (n_points, 3),
            "ee_quats_wxyz": (n_points, 4),
            "second_ee_positions": (n_points, 3),
            "second_ee_quats_wxyz": (n_points, 4),
        }
        for key, shape in shapes.items():
            value = np.asarray(archive[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{key} 形状/数值非法: {value.shape}")
        if not np.isfinite(positions).all():
            raise ValueError("positions 含 NaN/Inf")
        if int(meta.get("n_points", -1)) != n_points:
            raise ValueError(
                f"metadata n_points={meta.get('n_points')} 与 NPZ {n_points} 不一致"
            )
except (OSError, ValueError, KeyError) as exc:
    raise SystemExit(f"[ERR] IK NPZ 无效: {exc}")
PY

RUN_ARGS=(--traj "$IK_RESULTS")
if [[ "$RATE_OVERRIDDEN" != "true" ]]; then
  # fixed-rate 模式会忽略 NPZ times，因此必须让频率与 frame_seconds
  # 一致。未显式覆盖时优先沿用 artifact metadata；默认 2 秒即 0.5 Hz。
  if [[ "$FRAME_SECONDS_OVERRIDDEN" != "true" ]]; then
    IK_META="$(dirname "$IK_RESULTS")/trajectory_meta.json"
    if [[ -f "$IK_META" ]]; then
      META_FRAME_SECONDS="$(python3 - "$IK_META" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as stream:
        value = json.load(stream).get("frame_seconds")
    if value is not None:
        print(value)
except (OSError, ValueError, AttributeError):
    pass
PY
)"
      [[ -n "$META_FRAME_SECONDS" ]] && FRAME_SECONDS="$META_FRAME_SECONDS"
    fi
  fi
  DEFAULT_RATE_HZ="$(python3 - "$FRAME_SECONDS" <<'PY'
import math
import sys

try:
    seconds = float(sys.argv[1])
except ValueError:
    raise SystemExit("frame-seconds 必须是有限正数")
if not math.isfinite(seconds) or seconds <= 0.0:
    raise SystemExit("frame-seconds 必须是有限正数")
print(f"{1.0 / seconds:.12g}")
PY
)"
  RUN_ARGS+=(--rate-hz "$DEFAULT_RATE_HZ")
fi
if [[ "$PLAY_ONCE" != "true" ]]; then
  RUN_ARGS+=(--loop)
fi
if [[ "$SCAN_OVERRIDDEN" != "true" ]]; then
  RUN_ARGS+=(--no-scan-cloud)
fi

echo "[dual IK rviz] ############################################################"
echo "[dual IK rviz] 仅播放离散 IK 候选；候选间转换未经规划，禁止下发真机"
echo "[dual IK rviz] ############################################################"
exec "${TASK_ROOT}/run_rviz.sh" "${RUN_ARGS[@]}" "${EXTRA[@]}"
