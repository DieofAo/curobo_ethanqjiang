#!/usr/bin/env bash
# Reconnect one RViz GUI to its already-running, audited saved-trajectory player.
set -eo pipefail
name=${1:?run name required}
port=${2:?ROS port required}
case "$name:$port" in
  v64_39:11361|v64_10:11362|v64_47:11363|v64_52:11364) ;;
  *) echo "Unexpected run/ROS port pair: $name:$port" >&2; exit 2 ;;
esac
source /opt/ros/noetic/setup.bash
set -u
export DISPLAY=:0
export XAUTHORITY=/home/ethanqjiang/.Xauthority
export XDG_RUNTIME_DIR=/run/user/1000
export ROS_MASTER_URI="http://127.0.0.1:$port"
export ROS_IP=127.0.0.1
export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export DISABLE_ROS1_EOL_WARNINGS=1
exec rviz -d /home/ethanqjiang/workspace/curobo/xtrainer_plan/config/xtrainer_overhead.rviz -f task_world __name:=overhead_rviz
