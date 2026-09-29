#!/usr/bin/env bash
# Open four real desktop RViz trajectory players, each on its own ROS master.
# Start only after the four audited full runs have completed.
set -euo pipefail
repo=/home/ethanqjiang/workspace/curobo
task="$repo/xtrainer_plan"
root="$task/results_overhead/20260928/v66_v64_8of9_local_y_full"
out="$root/rviz_sessions"
source /opt/ros/noetic/setup.bash
export DISPLAY=:0
export XAUTHORITY=/home/ethanqjiang/.Xauthority
export XDG_RUNTIME_DIR=/run/user/1000
export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export DISABLE_ROS1_EOL_WARNINGS=1
mkdir -p "$out"
/usr/bin/python3 - "$root" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
for name in ('v64_39','v64_10','v64_47','v64_52'):
    run=root/'runs'/name
    for file in ('trajectory.npz','trajectory_meta.json','independent_verification.json','joint_limit_clip_audit.json'):
        if not (run/file).exists():raise SystemExit(f'Missing audited run: {run/file}')
    for file in ('independent_verification.json','joint_limit_clip_audit.json'):
        check=json.loads((run/file).read_text())
        if check.get('passed') is not True or check.get('verification_completed') is not True:
            raise SystemExit(f'Audit not passed: {run/file}')
PY
names=(v64_39 v64_10 v64_47 v64_52)
ports=(11361 11362 11363 11364)
ids=()
for i in 0 1 2 3; do
  name=${names[$i]}
  port=${ports[$i]}
  run="$root/runs/$name"
  export ROS_MASTER_URI="http://127.0.0.1:$port"
  export ROS_IP=127.0.0.1
  window=""
  if rostopic list >/dev/null 2>&1; then
    if [[ -f "$out/${name}_window_id" && -f "$out/${name}_rviz.log" ]] \
       && rg -Fq "traj=${run}" "$out/${name}_rviz.log"; then
      window=$(cat "$out/${name}_window_id")
      DISPLAY=:0 xwininfo -id "$window" >/dev/null
      echo "REUSED $name on ROS port $port, window $window, run $run"
    else
      echo "ROS port $port is already used; refusing to attach to an unrelated master" >&2
      exit 1
    fi
  else
    before=$(DISPLAY=:0 xwininfo -root -tree | /usr/bin/python3 -c 'import re,sys; print(" ".join(re.findall(r"^\s*(0x[0-9a-f]+)\s+\"[^\"]* - RViz\":\s*\(\"rviz\"",sys.stdin.read(),re.M)))')
    nohup roscore -p "$port" >"$out/${name}_roscore.log" 2>&1 </dev/null &
    echo $! >"$out/${name}_roscore.pid"
    for attempt in $(seq 1 40); do
      rostopic list >/dev/null 2>&1 && break
      sleep 0.5
    done
    rostopic list >/dev/null
    nohup "$task/run_overhead_rviz.sh" --traj "$run" --speed 10 --display-hz 50 --loop --display :0 \
       >"$out/${name}_rviz.log" 2>&1 </dev/null &
    echo $! >"$out/${name}_launcher.pid"
    for attempt in $(seq 1 90); do
      window=$(DISPLAY=:0 xwininfo -root -tree | /usr/bin/python3 -c 'import re,sys; text=sys.stdin.read(); before=set(sys.argv[1].split()); ids=re.findall(r"^\s*(0x[0-9a-f]+)\s+\"[^\"]* - RViz\":\s*\(\"rviz\"",text,re.M);print(next((v for v in ids if v not in before),""))' "$before")
      if [[ -n "$window" ]]; then break; fi
      if ! kill -0 "$(cat "$out/${name}_launcher.pid")" 2>/dev/null; then
        echo "RViz launcher failed for $name; see $out/${name}_rviz.log" >&2
        exit 1
      fi
      sleep 1
    done
  fi
  if [[ -z "$window" ]]; then
    echo "No RViz window appeared for $name; see $out/${name}_rviz.log" >&2
    exit 1
  fi
  ids+=("$window:$name  10x saved trajectory")
  echo "$window" >"$out/${name}_window_id"
  echo "OPENED $name on ROS port $port, window $window, run $run"
  ROS_MASTER_URI="http://127.0.0.1:$port" rosnode list >"$out/${name}_rosnodes.txt"
  if ! rg -q 'overhead_rviz' "$out/${name}_rosnodes.txt"; then
    echo "RViz node not registered for $name" >&2
    exit 1
  fi
  if ! rg -Fq "traj=${run}" "$out/${name}_rviz.log"; then
    echo "Launcher log does not pin expected saved trajectory for $name" >&2
    exit 1
  fi
done
/usr/bin/python3 "$task/scripts/tile_xtrainer_rviz_windows.py" "${ids[@]}" | tee "$out/tile.log"
for spec in "${ids[@]}"; do
  id=${spec%%:*}
  label=${spec#*:}
  DISPLAY=:0 xprop -id "$id" -f _NET_WM_NAME 8u -set _NET_WM_NAME "$label" >/dev/null
  DISPLAY=:0 xprop -id "$id" WM_NAME _NET_WM_NAME
  DISPLAY=:0 xwininfo -id "$id" | rg 'Width:|Height:|Absolute upper-left X:|Absolute upper-left Y:'
done | tee "$out/window_verification.log"
echo "Four audited full-run RViz players are visible; original V65 RViz was not touched."
