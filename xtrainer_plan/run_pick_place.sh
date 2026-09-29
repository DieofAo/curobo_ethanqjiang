#!/usr/bin/env bash
# =============================================================================
# XTrainer 多物料上下料（pick & place）循环轨迹规划。自动切到 conda curobo 环境。
#
# 用法:
#   ./run_pick_place.sh
#   ./run_pick_place.sh --rows 3 --cols 4
#   ./run_pick_place.sh --grasp-range -60 60 --grasp-step 5
#   ./run_pick_place.sh --max-items 2            # 只跑前 2 个物料，快速验证
#   ./run_pick_place.sh --config config/my_pick_place.yaml
#   ./run_pick_place.sh --dual                  # 12-DOF 独立双臂交错联合规划
#   ./run_pick_place.sh --dual --max-items 2    # 双臂合计先跑 2 个物料验证
#
# 单臂和双臂入口都会应用 pick_place.link0_target_transform。
#
#   # 只扫框选区域「最外围一圈」，到不了的点位直接跳过（快速摸清工作区边界）
#   ./run_pick_place.sh --perimeter --order ring --on-fail skip
#
#   # 抓取角与放置角是否耦合（默认独立搜索，笛卡尔积）
#   ./run_pick_place.sh --couple-place           # 放置角强制等于抓取角
#   ./run_pick_place.sh --no-couple-place        # 两侧独立搜索
#
#   # 第二阶段：在第一阶段结果之上，再绕「第一阶段转完后的局部 Z 轴」搜索
#   #（局部 Z = 该姿态旋转矩阵的第三列方向，不是 LINK_0 的 Z 轴）
#   ./run_pick_place.sh --stage2                             # 用 yaml 里的 stage2 参数
#   ./run_pick_place.sh --stage2 --stage2-range -90 0 --stage2-step 15
#   ./run_pick_place.sh --stage2 --stage2-axis y             # 换成绕局部 Y
#   ./run_pick_place.sh --no-stage2                          # 只做第一阶段
#
#   # 抬升点 -> 目标点沿任务坐标 Z 轴直线插拔；经 C 映射到规划 root 笛卡尔轴
#   #（默认 grasp 与 place 都开）
#   ./run_pick_place.sh --no-linear-place        # 关掉直线约束
#   ./run_pick_place.sh --linear-kinds place     # 只约束放置段
#   ./run_pick_place.sh --linear-dev 1.5         # 直线段横向偏移上限收紧到 1.5mm
#   ./run_pick_place.sh --linear-rot 3           # 直线段姿态偏差上限收紧到 3deg
# =============================================================================
set -eo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-curobo}"

# --dual 只用于选择入口，不透传给 Python 的任务参数解析器。
DUAL_MODE="false"
FORWARD_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--dual" ]]; then
    DUAL_MODE="true"
  else
    FORWARD_ARGS+=("$arg")
  fi
done

# ROS 的 python3.8 site-packages 会污染 conda py3.11 的 import, 必须清掉
unset PYTHONPATH || true

CONDA_SH=""
for p in "$HOME/miniconda3/etc/profile.d/conda.sh" \
         "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "/opt/conda/etc/profile.d/conda.sh"; do
  [[ -f "$p" ]] && CONDA_SH="$p" && break
done
if [[ -z "$CONDA_SH" ]]; then
  echo "[ERR] 找不到 conda.sh, 请手动 conda activate ${CONDA_ENV} 后直接运行:" >&2
  if [[ "$DUAL_MODE" == "true" ]]; then
    echo "      python ${TASK_ROOT}/scripts/plan_dual_pick_place.py ${FORWARD_ARGS[*]}" >&2
  else
    echo "      python ${TASK_ROOT}/scripts/plan_pick_place.py ${FORWARD_ARGS[*]}" >&2
  fi
  exit 1
fi

# conda 的 cuda-nvcc 激活脚本在 set -u 下会因 NVCC_PREPEND_FLAGS 未定义而报错
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

echo "[run_pick_place] env=${CONDA_ENV}  python=$(which python)  dual=${DUAL_MODE}"
cd "$TASK_ROOT"
if [[ "$DUAL_MODE" == "true" ]]; then
  exec python scripts/plan_dual_pick_place.py "${FORWARD_ARGS[@]}"
fi
exec python scripts/plan_pick_place.py "${FORWARD_ARGS[@]}"
