#!/usr/bin/env bash
# =============================================================================
# place 位置区域扫描 —— 找「哪个 place 位置能让更远的 grasp 点可达」。
# 自动切到 conda curobo 环境。
#
# 三阶段流程:
#   阶段1  逐个 grasp 位置试 IK，某角度失败就试下一个旋转角，全失败则跳过
#   阶段2  同上，逐个 place 位置试 IK
#   阶段3  只对两侧都可行的 (grasp, place) 组合做轨迹规划，
#          规划失败或关节变化超限则跳过该 grasp 点
# 排名: 先要求三阶段都成功，再按「成功的 grasp 点位数」从多到少取 Top-N
#
# 用法:
#   ./run_place_scan.sh --stage 12               # 阶段1+2（IK 筛选，快），先看筛选结果
#   ./run_place_scan.sh --stage 3 --out-dir results_place_scan/20260901_120000
#                                                # 阶段3（轨迹规划，慢），读盘续跑
#   ./run_place_scan.sh                          # 三阶段依次跑（默认 all）
#   ./run_place_scan.sh --max-place 4 --max-grasp 12   # 调试用小规模
#
#   # 区域与密度
#   ./run_place_scan.sh --place-step 0.02                 # place 网格放粗到 2cm
#   ./run_place_scan.sh --grasp-step 0.02                 # 阶段1 grasp 网格 2cm
#   ./run_place_scan.sh --no-grasp-perimeter              # 阶段1 grasp 扫全网格（慢）
#   ./run_place_scan.sh --grasp-edge x_min --grasp-x-range -0.56 -0.2
#                                                # 只算离基座最远那一列
#   ./run_place_scan.sh --stage3-grasp-step 0.05          # 阶段3 放粗（快）
#   ./run_place_scan.sh --top-n 5                         # 只要前 5 名
#
#   # 角度搜索（覆盖 pick_place.angle_search，逻辑与正式规划共用）
#   ./run_place_scan.sh --couple-place                    # 放置角由抓取角决定
#   ./run_place_scan.sh --rot2 --rot2-step 30             # 启用第二阶段旋转
#
# 长时间任务请脱离终端（阶段3 可能数小时）:
#   setsid nohup ./run_place_scan.sh --stage 3 --out-dir results_place_scan/<时间戳> \
#     > /tmp/scan3.log 2>&1 < /dev/null & disown
#
# 只出图（不重新计算）:
#   python3 scripts/plot_place_scan.py results_place_scan/<时间戳>
# =============================================================================
set -eo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-curobo}"

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
  echo "      python ${TASK_ROOT}/scripts/scan_place_region.py $*" >&2
  exit 1
fi

# conda 的 cuda-nvcc 激活脚本在 set -u 下会因 NVCC_PREPEND_FLAGS 未定义而报错
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

echo "[run_place_scan] env=${CONDA_ENV}  python=$(which python)"
cd "$TASK_ROOT"
exec python scripts/scan_place_region.py "$@"
