#!/usr/bin/env python3
"""Title and tile the four already-running V66 RViz GUI clients by ROS/PID."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

from verify_v66_four_rviz import OUT, RUNS, ROOT, node_pid

TASK = Path(__file__).resolve().parents[1]


def main() -> None:
    env = dict(os.environ, DISPLAY=":0", XAUTHORITY="/home/ethanqjiang/.Xauthority")
    tree = subprocess.check_output(["xwininfo", "-root", "-tree"], env=env, text=True)
    candidates = []
    for match in re.finditer(r'^\s*(0x[0-9a-f]+)\s+"([^"]+)":\s*\("rviz" "rviz"\)', tree, re.M):
        window_id, title = match.groups()
        if not (title.endswith("- RViz") or "saved trajectory" in title):
            continue
        prop = subprocess.check_output(["xprop", "-id", window_id, "_NET_WM_PID"], env=env, text=True)
        pid_match = re.search(r"=\s*(\d+)", prop)
        if pid_match:
            candidates.append((window_id, int(pid_match.group(1))))
    specs = []
    verification = []
    for name, port in RUNS.items():
        ros_env = dict(env, ROS_MASTER_URI=f"http://127.0.0.1:{port}", ROS_IP="127.0.0.1")
        rviz_pid = node_pid("/overhead_rviz", ros_env)
        player_pid = node_pid("/overhead_traj_player", ros_env)
        expected = str((ROOT / "runs" / name).resolve())
        command = (Path("/proc") / str(player_pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        if f"--traj {expected}" not in command:
            raise RuntimeError(f"{name} player has unexpected source: {command}")
        windows = [window_id for window_id, pid in candidates if pid == rviz_pid]
        if len(windows) != 1:
            raise RuntimeError(f"{name}: expected one main window for RViz PID {rviz_pid}; found {windows}")
        window_id = windows[0]
        specs.append(f"{window_id}:{name}  10x saved trajectory")
        (OUT / f"{name}_window_id").write_text(window_id + "\n")
        verification.append(f"{name}: ROS {port}, player PID {player_pid}, RViz PID {rviz_pid}, window {window_id}, source {expected}")
    out = subprocess.check_output([sys.executable, str(TASK / "scripts/tile_xtrainer_rviz_windows.py"), *specs],
                                  env=env, text=True)
    (OUT / "tile.log").write_text(out)
    for spec in specs:
        window_id, label = spec.split(":", 1)
        subprocess.run(["xprop", "-id", window_id, "-f", "_NET_WM_NAME", "8u", "-set",
                        "_NET_WM_NAME", label], env=env, check=True, stdout=subprocess.DEVNULL)
        verification.append(subprocess.check_output(
            ["xprop", "-id", window_id, "_NET_WM_PID", "_NET_WM_NAME"], env=env, text=True).strip())
    (OUT / "window_verification.log").write_text("\n".join(verification) + "\n")
    print(out, end="")
    for row in verification[:4]:
        print(row)


if __name__ == "__main__":
    main()
