#!/usr/bin/env bash
# Resume the audited V67 negative-angle smoke -> Top 5 full-grid pipeline.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SMOKE_ROOT FULL_ROOT [--poll-seconds N] [--wait-timeout-seconds N]" >&2
  exit 2
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../.." && pwd)"
conda_curobo=/home/ethanqjiang/miniconda3/envs/curobo
cuda_targets="$conda_curobo/targets/x86_64-linux"
media_bin="$script_dir/system_media_bin"
for required in "$conda_curobo/bin/python" "$conda_curobo/bin/ninja" \
                "$conda_curobo/bin/nvcc" "$cuda_targets/include" \
                "$cuda_targets/lib" /opt/ros/noetic/lib "$repo/src"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required cuRobo/CUDA/ROS path: $required" >&2
    exit 1
  fi
done
for media_tool in ffmpeg ffprobe; do
  if [[ ! -x "$media_bin/$media_tool" || ! -x "/usr/bin/$media_tool" ]]; then
    echo "Missing executable system media wrapper or binary: $media_tool" >&2
    exit 1
  fi
done
export PATH="$media_bin:$conda_curobo/bin:$PATH"
export PYTHONPATH="$repo/src"
export LD_LIBRARY_PATH="$conda_curobo/lib:$cuda_targets/lib:/opt/ros/noetic/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$cuda_targets/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$cuda_targets/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CUDA_HOME="$conda_curobo"
export MAX_JOBS=1
exec "$conda_curobo/bin/python" -u "$script_dir/orchestrate_negative_top5_full.py" "$@"
