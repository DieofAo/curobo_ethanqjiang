#!/usr/bin/env bash
# =============================================================================
# XTrainer 轨迹规划 (curobo). 自动切到 conda curobo 环境。
#
# 用法:
#   ./run_plan.sh
#   ./run_plan.sh --start-position -0.31 -0.05 0.2 --goal-position -0.34 -0.09 0.22
#   ./run_plan.sh --config config/my_task.yaml
#   ./run_plan.sh --no-lift --enable-graph
#   CONDA_ENV=curobo ./run_plan.sh
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
  echo "      python ${TASK_ROOT}/scripts/plan_trajectory.py $*" >&2
  exit 1
fi

# conda 的 cuda-nvcc 激活脚本在 set -u 下会因 NVCC_PREPEND_FLAGS 未定义而报错
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

echo "[run_plan] env=${CONDA_ENV}  python=$(which python)"
cd "$TASK_ROOT"
exec python scripts/plan_trajectory.py "$@"
