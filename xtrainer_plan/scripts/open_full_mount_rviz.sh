#!/usr/bin/env bash
# Open one audited five-candidate full-run RViz on its own ROS port.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 FULL_RESULT_ROOT CANDIDATE" >&2
  exit 2
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task="$(cd "$script_dir/.." && pwd)"
root="$(realpath "$1")"
name="$2"
port_base="${XTRAINER_RVIZ_PORT_BASE:-11371}"
if [[ ! "$port_base" =~ ^[0-9]+$ ]] || (( port_base < 1024 || port_base > 65531 )); then
  echo "Invalid XTRAINER_RVIZ_PORT_BASE" >&2
  exit 2
fi
port=$(/usr/bin/python3 - "$root" "$name" "$port_base" <<'PY'
import hashlib,json,pathlib,re,sys
root=pathlib.Path(sys.argv[1]).resolve()
name=sys.argv[2]
base=int(sys.argv[3])
manifest=json.loads((root/'manifest.json').read_text())
rows=manifest['candidates']
if manifest['n_configs']!=5 or len(rows)!=5:
    raise SystemExit('Expected a five-candidate full-run manifest')
names=[row['name'] for row in rows]
if len(set(names))!=5 or any(not re.fullmatch(r'[A-Za-z0-9_]+',item) for item in names):
    raise SystemExit('Invalid or duplicate manifest names')
if name not in names:
    raise SystemExit(f'Candidate not in manifest: {name}')
row=rows[names.index(name)]
run=root/'runs'/name
if pathlib.Path(row['result']).resolve()!=run:
    raise SystemExit(f'Manifest result path differs: {name}')
for filename in ('trajectory.npz','trajectory_meta.json','independent_verification.json','joint_limit_clip_audit.json'):
    if not (run/filename).is_file():
        raise SystemExit(f'Missing audited input: {run/filename}')
status=json.loads((root/'sweep_status.json').read_text())
if status['outcomes'][name]['state']!='completed_verified':
    raise SystemExit(f'Full-run audits incomplete: {name}')
meta=json.loads((run/'trajectory_meta.json').read_text())
if meta['n_items_total']!=400 or meta['n_items_success']<3:
    raise SystemExit(f'Unexpected full-run grid or success count: {name}')
independent=json.loads((run/'independent_verification.json').read_text())
clip=json.loads((run/'joint_limit_clip_audit.json').read_text())
for filename,audit in (('independent_verification.json',independent),('joint_limit_clip_audit.json',clip)):
    if audit.get('passed') is not True or audit.get('verification_completed') is not True:
        raise SystemExit(f'Failed or incomplete audit: {run/filename}')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
npz_hash=digest(run/'trajectory.npz')
meta_hash=digest(run/'trajectory_meta.json')
if independent['source']['npz_sha256']!=npz_hash or independent['source']['metadata_sha256']!=meta_hash:
    raise SystemExit(f'Independent audit source hashes differ: {name}')
if clip['source']['trajectory_sha256']!=npz_hash or clip['source']['config_source_sha256']!=meta_hash:
    raise SystemExit(f'Joint-limit audit source hashes differ: {name}')
print(base+names.index(name))
PY
)
out="$root/rviz_sessions"
mkdir -p "$out"
source /opt/ros/noetic/setup.bash
export DISPLAY="${XTRAINER_DESKTOP_DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/ethanqjiang/.Xauthority}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export ROS_MASTER_URI="http://127.0.0.1:$port"
export ROS_IP=127.0.0.1
export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export DISABLE_ROS1_EOL_WARNINGS=1
if rostopic list >/dev/null 2>&1; then
  echo "ROS port $port is already in use; refusing to attach to another session." >&2
  exit 1
fi
before=$(xwininfo -root -tree | /usr/bin/python3 -c 'import re,sys; print(" ".join(re.findall(r"^\s*(0x[0-9a-f]+)\s+\"[^\"]* - RViz\":\s*\(\"rviz\"",sys.stdin.read(),re.M)))')
nohup roscore -p "$port" >"$out/${name}_roscore.log" 2>&1 </dev/null &
roscore_pid=$!
echo "$roscore_pid" >"$out/${name}_roscore.pid"
for attempt in $(seq 1 40); do
  rostopic list >/dev/null 2>&1 && break
  sleep .5
done
rostopic list >/dev/null
nohup "$task/run_overhead_rviz.sh" --traj "$root/runs/$name" --speed 10 \
  --display-hz 50 --loop --display "$DISPLAY" \
  >"$out/${name}_rviz.log" 2>&1 </dev/null &
launcher_pid=$!
echo "$launcher_pid" >"$out/${name}_launcher.pid"
window=""
for attempt in $(seq 1 90); do
  window=$(xwininfo -root -tree | /usr/bin/python3 -c 'import re,sys; text=sys.stdin.read(); before=set(sys.argv[1].split()); ids=re.findall(r"^\s*(0x[0-9a-f]+)\s+\"[^\"]* - RViz\":\s*\(\"rviz\"",text,re.M); print(next((item for item in ids if item not in before),""))' "$before")
  [[ -n "$window" ]] && break
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    echo "RViz launcher exited; see $out/${name}_rviz.log" >&2
    exit 1
  fi
  sleep 1
done
if [[ -z "$window" ]]; then
  echo "RViz window did not appear; see $out/${name}_rviz.log" >&2
  exit 1
fi
printf '%s\n' "$window" >"$out/${name}_window_id"
xprop -id "$window" -f _NET_WM_NAME 8u -set _NET_WM_NAME "$name  saved trajectory  ROS:$port" >/dev/null
rosnode list >"$out/${name}_rosnodes.txt"
if ! rg -q 'overhead_rviz' "$out/${name}_rosnodes.txt" || \
   ! rg -Fq "traj=$root/runs/$name" "$out/${name}_rviz.log"; then
  echo "RViz process or saved trajectory source did not verify; see $out/${name}_rviz.log" >&2
  exit 1
fi
echo "OPENED $name  ROS:$port  window:$window  run:$root/runs/$name"
