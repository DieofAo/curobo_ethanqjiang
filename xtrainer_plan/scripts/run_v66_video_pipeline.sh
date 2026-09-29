#!/usr/bin/env bash
# Render every successful saved trajectory (40x) and the exact max/min cases (1x).
# Run only after all four full-planning audits have finished.
set -euo pipefail

repo=/home/ethanqjiang/workspace/curobo
task="$repo/xtrainer_plan"
root="$task/results_overhead/20260928/v66_v64_8of9_local_y_full"
port=11355
nested_display=:95
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI="http://127.0.0.1:$port"
export ROS_IP=127.0.0.1
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export QT_X11_NO_MITSHM=1
export XAUTHORITY=/home/ethanqjiang/.Xauthority
export XDG_RUNTIME_DIR=/run/user/1000

# Optional names select already audited runs; no argument processes all four.
if [[ $# -eq 0 ]]; then
  selected=(v64_39 v64_10 v64_47 v64_52)
else
  selected=("$@")
fi
for selected_name in "${selected[@]}"; do
  case "$selected_name" in
    v64_39|v64_10|v64_47|v64_52) ;;
    *) echo "Unknown run $selected_name" >&2; exit 2 ;;
  esac
done
# Check selected inputs before starting a recorder.
/usr/bin/python3 - "$root" "${selected[@]}" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
for name in sys.argv[2:]:
    run=root/'runs'/name
    for file in ('trajectory.npz','trajectory_meta.json','independent_verification.json','joint_limit_clip_audit.json'):
        if not (run/file).is_file(): raise SystemExit(f'Missing {run/file}')
    meta=json.loads((run/'trajectory_meta.json').read_text())
    for file in ('independent_verification.json','joint_limit_clip_audit.json'):
        audit=json.loads((run/file).read_text())
        if audit.get('passed') is not True or audit.get('verification_completed') is not True:
            raise SystemExit(f'Incomplete or failed audit: {run/file}')
    if meta.get('n_items_total') != 400 or meta.get('n_items_success',0) < 2:
        raise SystemExit(f'Unexpected 20x20 result: {run}')
    print(f'INPUT VERIFIED {name}: {meta["n_items_success"]}/400',flush=True)
PY

if [[ -e /tmp/.X95-lock ]]; then
  echo 'Nested X display :95 already used; stop rather than attach to someone else.' >&2
  exit 1
fi
if rostopic list >/dev/null 2>&1; then
  echo 'ROS port 11355 already used; stop rather than attach to someone else.' >&2
  exit 1
fi
mkdir -p "$root/videos"
DISPLAY=:0 Xephyr "$nested_display" -screen 1500x1000 -ac -noreset >"$root/videos/xephyr.log" 2>&1 &
xephyr_pid=$!
roscore -p "$port" >"$root/videos/roscore.log" 2>&1 &
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
  local display_number="${nested_display#:}"
  for attempt in $(seq 1 40); do
    [[ ! -e "/tmp/.X${display_number}-lock" ]] && break
    sleep 0.2
  done
  if [[ -e "/tmp/.X${display_number}-lock" ]]; then
    echo "Xephyr display $nested_display did not release its lock" >&2
    exit 1
  fi
  DISPLAY=:0 Xephyr "$nested_display" -screen 1500x1000 -ac -noreset \
      >"$root/videos/xephyr_${display_number}_restart_$(date +%s).log" 2>&1 &
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
