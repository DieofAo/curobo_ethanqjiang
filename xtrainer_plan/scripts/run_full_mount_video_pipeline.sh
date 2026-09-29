#!/usr/bin/env bash
# Render every successful saved trajectory (40x) and the exact max/min cases (1x).
# Run selected candidates only after their full-planning audits have finished.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 FULL_RESULT_ROOT [CANDIDATE ...]" >&2
  exit 2
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task="$(cd "$script_dir/.." && pwd)"
root="$(realpath "$1")"
shift
port="${XTRAINER_VIDEO_ROS_PORT:-11376}"
nested_display="${XTRAINER_VIDEO_DISPLAY:-:96}"
desktop_display="${XTRAINER_DESKTOP_DISPLAY:-:0}"
if [[ ! "$port" =~ ^[0-9]+$ || ! "$nested_display" =~ ^:[0-9]+$ ]]; then
  echo "Invalid ROS port or nested X display" >&2
  exit 2
fi
display_number="${nested_display#:}"
# The planner uses conda CUDA libraries; this recorder uses system Python,
# system ROS/RViz, Xephyr and Mesa. Keep their shared libraries isolated in
# this child shell so conda libffi/libcurl/Qt cannot be picked up at runtime.
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT CPATH LIBRARY_PATH CUDA_HOME
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QML2_IMPORT_PATH QT_QPA_FONTDIR
export PYTHONNOUSERSITE=1
video_path="$script_dir/system_media_bin:/opt/ros/noetic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH="$video_path"
source /opt/ros/noetic/setup.bash
export PATH="$video_path"
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages
export LD_LIBRARY_PATH=/opt/ros/noetic/lib
export QT_QPA_PLATFORM=xcb
for media_tool in ffmpeg ffprobe; do
  if [[ ! -x "$script_dir/system_media_bin/$media_tool" ]]; then
    echo "Missing system media wrapper: $media_tool" >&2
    exit 1
  fi
done
/usr/bin/python3 - "$script_dir" <<'PY_VIDEO_ENV'
import sys
sys.path.insert(0, sys.argv[1])
import numpy
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from rviz import bindings
import rospy
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
import xtrainer_common, play_trajectory_ros, build_overhead_scene_urdf
import audit_isolated_rviz_video, assemble_full_mount_video_chunks
import extract_pick_place_span_extremes, review_full_mount_video_boundaries
import write_v66_video_readme, combine_full_mount_extreme_videos
print(f"VIDEO ENV VERIFIED: {sys.executable}; numpy {numpy.__version__}; ROS/RViz/PyQt imports")
PY_VIDEO_ENV
export ROS_MASTER_URI="http://127.0.0.1:$port"
export ROS_IP=127.0.0.1
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export QT_X11_NO_MITSHM=1
export XAUTHORITY="${XAUTHORITY:-/home/ethanqjiang/.Xauthority}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# The manifest fixes candidate order. Optional names select a verified subset.
selection=$(/usr/bin/python3 - "$root" "$@" <<'PY_SELECT'
import json, pathlib, re, sys
root=pathlib.Path(sys.argv[1]).resolve()
manifest=json.loads((root/'manifest.json').read_text())
rows=manifest['candidates']
if len(rows)!=5 or manifest['n_configs']!=5:
    raise SystemExit('Expected five full-run candidates in manifest')
names=[row['name'] for row in rows]
if len(set(names))!=5 or any(not re.fullmatch(r'[A-Za-z0-9_]+',name) for name in names):
    raise SystemExit('Invalid or duplicate manifest names')
for row in rows:
    name=row['name']
    if pathlib.Path(row['result']).resolve()!=root/'runs'/name:
        raise SystemExit(f'Manifest result path differs: {name}')
requested=sys.argv[2:] or names
if len(requested)!=len(set(requested)) or set(requested)-set(names):
    raise SystemExit('Requested names must be distinct manifest candidates')
print('\n'.join(requested))
PY_SELECT
)
mapfile -t selected <<< "$selection"
# Check selected inputs before starting a recorder.
/usr/bin/python3 - "$root" "${selected[@]}" <<'PY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
status=json.loads((root/'sweep_status.json').read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
for name in sys.argv[2:]:
    run=root/'runs'/name
    if status['outcomes'][name]['state']!='completed_verified':
        raise SystemExit(f'Full-run audits incomplete: {name}')
    for file in ('trajectory.npz','trajectory_meta.json','independent_verification.json','joint_limit_clip_audit.json'):
        if not (run/file).is_file(): raise SystemExit(f'Missing {run/file}')
    meta=json.loads((run/'trajectory_meta.json').read_text())
    independent=json.loads((run/'independent_verification.json').read_text())
    clip=json.loads((run/'joint_limit_clip_audit.json').read_text())
    for file,audit in (('independent_verification.json',independent),('joint_limit_clip_audit.json',clip)):
        if audit.get('passed') is not True or audit.get('verification_completed') is not True:
            raise SystemExit(f'Incomplete or failed audit: {run/file}')
    npz_hash=digest(run/'trajectory.npz')
    meta_hash=digest(run/'trajectory_meta.json')
    if independent['source']['npz_sha256']!=npz_hash or independent['source']['metadata_sha256']!=meta_hash:
        raise SystemExit(f'Independent audit source hashes differ: {name}')
    if clip['source']['trajectory_sha256']!=npz_hash or clip['source']['config_source_sha256']!=meta_hash:
        raise SystemExit(f'Joint-limit audit source hashes differ: {name}')
    if meta.get('n_items_total') != 400 or meta.get('n_items_success',0) < 3:
        raise SystemExit(f'Unexpected 20x20 result: {run}')
    print(f'INPUT VERIFIED {name}: {meta["n_items_success"]}/400',flush=True)
PY

if [[ -e "/tmp/.X${display_number}-lock" ]]; then
  echo "Nested X display $nested_display already used; stop rather than attach to someone else." >&2
  exit 1
fi
if rostopic list >/dev/null 2>&1; then
  echo "ROS port $port already used; stop rather than attach to someone else." >&2
  exit 1
fi
mkdir -p "$root/videos"
session_dir="$(mktemp -d "$root/videos/.recording_session_XXXXXX")"
DISPLAY="$desktop_display" Xephyr "$nested_display" -screen 1500x1000 -ac -noreset >"$session_dir/xephyr.log" 2>&1 &
xephyr_pid=$!
roscore -p "$port" >"$session_dir/roscore.log" 2>&1 &
roscore_pid=$!
cleanup() {
  kill "$roscore_pid" "$xephyr_pid" 2>/dev/null || true
  wait "$roscore_pid" "$xephyr_pid" 2>/dev/null || true
}
trap cleanup EXIT
for attempt in $(seq 1 40); do
  if DISPLAY="$nested_display" xdpyinfo >/dev/null 2>&1 && rostopic list >/dev/null 2>&1; then break; fi
  sleep 0.5
done
DISPLAY="$nested_display" xdpyinfo >/dev/null
rostopic list >/dev/null
# Minimize the nested desktop window, leaving the user's existing RViz visible.
/usr/bin/python3 "$task/scripts/minimize_private_xephyr.py" "$nested_display"
export DISPLAY="$nested_display"

restart_xephyr() {
  kill "$xephyr_pid" 2>/dev/null || true
  wait "$xephyr_pid" 2>/dev/null || true
  for attempt in $(seq 1 40); do
    [[ ! -e "/tmp/.X${display_number}-lock" ]] && break
    sleep 0.2
  done
  if [[ -e "/tmp/.X${display_number}-lock" ]]; then
    echo "Xephyr display $nested_display did not release its lock" >&2
    exit 1
  fi
  DISPLAY="$desktop_display" Xephyr "$nested_display" -screen 1500x1000 -ac -noreset \
      >"$session_dir/xephyr_${display_number}_restart_$(date +%s).log" 2>&1 &
  xephyr_pid=$!
  for attempt in $(seq 1 40); do
    DISPLAY="$nested_display" xdpyinfo >/dev/null 2>&1 && break
    sleep 0.3
  done
  DISPLAY="$nested_display" xdpyinfo >/dev/null
  /usr/bin/python3 "$task/scripts/minimize_private_xephyr.py" "$nested_display"
  echo "Fresh Xephyr $nested_display PID $xephyr_pid for next video chunk"
}

for name in "${selected[@]}"; do
  run="$root/runs/$name"
  folder="$root/videos/$name"
  full="$folder/full_40x"
  extremes="$folder/joint_span_extremes"
  mkdir -p "$full/segments"
  echo "===== $name full saved motion video in fresh RViz chunks ====="
  n_frames=$(/usr/bin/python3 - "$run" <<'PYFRAMES'
import json,pathlib,sys
meta=json.loads((pathlib.Path(sys.argv[1])/'trajectory_meta.json').read_text())
n=int(meta['n_points'])
selected=list(range(0,n,40))
print(len(selected)+(selected[-1]!=n-1))
PYFRAMES
)
  first=0
  rendered_since_restart=0
  while (( first < n_frames )); do
    last=$((first+999))
    if (( last >= n_frames )); then last=$((n_frames-1)); fi
    segment="${name}_frames_${first}_${last}.mp4"
    audit="video_render_audit_${first}.json"
    if (( first == 0 )); then audit=video_render_audit.json; fi
    if [[ -f "$full/segments/$segment" && -f "$full/segments/$audit" ]]; then
      echo "REUSE completed segment frames $first..$last"
    else
      if [[ -e "$full/segments/$segment" ]]; then
        mkdir -p "$full/failed_capture"
        mv "$full/segments/$segment" "$full/failed_capture/${segment%.mp4}_interrupted.mp4"
      fi
      if (( rendered_since_restart )); then restart_xephyr; fi
      /usr/bin/python3 "$task/scripts/render_full_mount_video.py" \
          --run "$run" --out-dir "$full/segments" --experiment-label "$name" \
          --master-port "$port" --first-video-frame "$first" --last-video-frame "$last" \
          --output-name "$segment" 2>&1 | tee "$full/segments/render_${first}_${last}.log"
      rendered_since_restart=1
    fi
    first=$((last+1))
  done
  /usr/bin/python3 "$task/scripts/assemble_full_mount_video_chunks.py" \
      --run "$run" --out-dir "$full" --label "$name" 2>&1 | tee "$full/assemble.log"
  cp "$full/segments/preview.png" "$full/preview.png"
  /usr/bin/python3 "$task/scripts/audit_isolated_rviz_video.py" \
      --video "$full/${name}_complete_40x.mp4" --mode full \
      --render-audit "$full/video_render_audit.json" --source-run "$run" \
      --out "$full/video_decode_audit.json" 2>&1 | tee "$full/decode.log"
  /usr/bin/python3 "$task/scripts/review_full_mount_video_boundaries.py" \
      --video "$full/${name}_complete_40x.mp4" --render-audit "$full/video_render_audit.json" \
      --out-dir "$full" 2>&1 | tee "$full/boundary_review.log"
  first_success=$(/usr/bin/python3 - "$run" <<'PY'
import json, pathlib, sys
meta=json.loads((pathlib.Path(sys.argv[1])/'trajectory_meta.json').read_text())
print(next(int(item['index'])+1 for item in meta['items'] if item.get('success')))
PY
)
  echo "===== $name exact max/min excluding first saved case $first_success ====="
  restart_xephyr
  /usr/bin/python3 "$task/scripts/extract_pick_place_span_extremes.py" \
      --result "$run" --out "$extremes" --exclude-case-number "$first_success" \
      2>&1 | tee "$folder/extreme_extraction.log"
  /usr/bin/python3 "$task/scripts/render_overhead_case_videos_isolated.py" \
      --root "$extremes" --tiers max min --experiment-label "$name" \
      --master-port "$port" --title-mode joint-range 2>&1 | tee "$extremes/render.log"
  /usr/bin/python3 "$task/scripts/combine_full_mount_extreme_videos.py" \
      --root "$extremes" 2>&1 | tee "$extremes/combine.log"
  combined=$(/usr/bin/python3 - "$extremes" <<'PY'
import json,pathlib,sys
print(json.loads((pathlib.Path(sys.argv[1])/'side_by_side_manifest.json').read_text())['output'])
PY
)
  /usr/bin/python3 "$task/scripts/audit_isolated_rviz_video.py" \
      --video "$combined" --mode comparison \
      --out "$extremes/side_by_side_video_audit.json" 2>&1 | tee "$extremes/decode.log"
  for tier in max min; do
    case_video=$(/usr/bin/python3 - "$extremes" "$tier" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);tier=sys.argv[2]
row=next(row for row in json.loads((root/'manifest.json').read_text())['selected'] if row['label']==tier)
print(root/f'{tier}_case_{row["original_case_number_1based"]:03d}.mp4')
PY
)
    /usr/bin/python3 "$task/scripts/audit_isolated_rviz_video.py" \
        --video "$case_video" --mode clip \
        --out "$extremes/${tier}_video_decode_audit.json" >/dev/null
  done
  /usr/bin/python3 "$task/scripts/write_v66_video_readme.py" --result "$run" --videos "$folder"
  echo "COMPLETE $name videos and audits"
done
