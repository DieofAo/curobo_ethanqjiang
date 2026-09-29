#!/usr/bin/env python3
"""Verify four live RViz windows show distinct, audited saved trajectories."""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1] / "results_overhead/20260928/v66_v64_8of9_local_y_full"
OUT = ROOT / "rviz_sessions"
RUNS = {"v64_39": 11361, "v64_10": 11362, "v64_47": 11363, "v64_52": 11364}


def call(args: list[str], env: dict[str, str], timeout: int = 8) -> str:
    return subprocess.check_output(args, env=env, text=True, timeout=timeout)


def node_pid(name: str, env: dict[str, str]) -> int:
    info = call(["rosnode", "info", name], env)
    match = re.search(r"Pid:\s*(\d+)", info)
    if not match:
        raise RuntimeError(f"Missing process id for {name}: {info}")
    return int(match.group(1))


def main() -> None:
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "runs": {}}
    for name, port in RUNS.items():
        env = dict(os.environ, DISPLAY=":0", XAUTHORITY="/home/ethanqjiang/.Xauthority",
                   ROS_MASTER_URI=f"http://127.0.0.1:{port}", ROS_IP="127.0.0.1")
        expected = str((ROOT / "runs" / name).resolve())
        window_id = (OUT / f"{name}_window_id").read_text().strip()
        rviz_pid = node_pid("/overhead_rviz", env)
        player_pid = node_pid("/overhead_traj_player", env)
        publisher_pid = node_pid("/overhead_robot_state_publisher", env)
        window = call(["xprop", "-id", window_id, "_NET_WM_PID", "_NET_WM_NAME"], env)
        window_pid = re.search(r"_NET_WM_PID\(CARDINAL\) = (\d+)", window)
        assert window_pid and int(window_pid.group(1)) == rviz_pid, (name, window)
        assert name in window, (name, window)
        cmdline = (pathlib.Path("/proc") / str(player_pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert f"--traj {expected}" in cmdline, (name, cmdline)
        maps = (pathlib.Path("/proc") / str(rviz_pid) / "maps").read_text()
        assert "swrast_dri.so" in maps and "libGLX_mesa" in maps, name
        sample = call(["rostopic", "echo", "-n", "1", "/joint_states"], env, timeout=8)
        assert "position:" in sample and "name:" in sample, name
        report["runs"][name] = {
            "ros_port": port, "source_run": expected, "rviz_pid": rviz_pid,
            "player_pid": player_pid, "publisher_pid": publisher_pid,
            "window_id": window_id, "mesa_software_gl": True, "joint_state_received": True,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "active_rviz_verification.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)
    for name, row in report["runs"].items():
        print(f"{name}: ROS {row['ros_port']}, RViz PID {row['rviz_pid']}, window {row['window_id']}, saved trajectory verified")


if __name__ == "__main__":
    main()
