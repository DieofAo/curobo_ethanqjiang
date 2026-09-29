#!/usr/bin/env python3
"""Write per-candidate RViz commands for a five-mount audited full experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Five-candidate full-run result root")
    parser.add_argument("--out", type=Path, help="Default: ROOT/RVIZ_COMMANDS.md")
    parser.add_argument("--port-base", type=int, default=11371,
                        help="First ROS port; candidates use this and the next four ports")
    args = parser.parse_args()
    root = args.root.resolve()
    if not 1024 <= args.port_base <= 65531:
        parser.error("--port-base must leave room for five nonprivileged ROS ports")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["candidates"]
    if manifest["n_configs"] != len(rows) or len(rows) != 5:
        raise ValueError("Expected five full-run candidates in manifest")
    names = [row["name"] for row in rows]
    if len(set(names)) != 5 or any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in names):
        raise ValueError("Invalid or duplicate candidate names")
    out = (args.out or root / "RVIZ_COMMANDS.md").resolve()
    task = Path(__file__).resolve().parents[1]
    lines = [
        "# 五组全量候选的 RViz 播放命令", "",
        "每组使用独立 ROS master 和已保存的完整抓放轨迹；`--speed 10` 只改变显示速度。",
        "安装坐标和角度均按 `task_world` 与原始基座局部 `+Y` 轴定义。",
        f"端口从 {args.port_base} 开始，避开此前 V66 使用的 11361–11364；启动脚本还会检查实际占用。若端口已占用，请用 `--port-base` 重新生成命令。", "",
        "| 候选 | 基座原点 (m) | 局部 +Y 倾角 | ROS 端口 |",
        "| --- | --- | ---: | ---: |",
    ]
    for i, row in enumerate(rows):
        name = row["name"]
        if Path(row["result"]).resolve() != root / "runs" / name:
            raise ValueError(f"Manifest result path differs: {name}")
        position = row["base_xyz_m"]
        tilt = row["local_y_tilt_deg"]
        if len(position) != 3 or not float(tilt) < 0:
            raise ValueError(f"Expected a negative-tilt mount: {name}")
        port = args.port_base + i
        lines.append(f"| `{name}` | ({position[0]:+.2f}, {position[1]:+.2f}, {position[2]:+.2f}) | {tilt:g}° | {port} |")
    lines += ["", "以下每组需要两个终端。只有对应全量轨迹及审核完成后才启动。", ""]
    for i, row in enumerate(rows):
        name = row["name"]
        port = args.port_base + i
        run = root / "runs" / name
        lines += [
            f"## {name}", "",
            "终端 1：", "", "```bash", "source /opt/ros/noetic/setup.bash",
            f"ROS_MASTER_URI=http://127.0.0.1:{port} ROS_IP=127.0.0.1 roscore -p {port}",
            "```", "", "终端 2：", "", "```bash", "source /opt/ros/noetic/setup.bash",
            f"ROS_MASTER_URI=http://127.0.0.1:{port} ROS_IP=127.0.0.1 "
            "__GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe "
            "DISABLE_ROS1_EOL_WARNINGS=1 DISPLAY=:0 XAUTHORITY=/home/ethanqjiang/.Xauthority \\",
            f"  {shlex.quote(str(task / 'run_overhead_rviz.sh'))} --traj {shlex.quote(str(run))} "
            "--speed 10 --display-hz 50 --loop --display :0",
            "```", "",
        ]
    lines += [
        "也可由仓库脚本先核对保存轨迹和两项独立审核，再后台启动单组（以下以清单首组为例）：", "",
        "```bash",
        f"XTRAINER_RVIZ_PORT_BASE={args.port_base} bash {shlex.quote(str(task / 'scripts/open_full_mount_rviz.sh'))} "
        f"{shlex.quote(str(root))} {shlex.quote(names[0])}",
        "```", "",
        "`ROS_MASTER_URI` 指向各自的 ROS master；`task_world` 是抓取点、基座原点和放置点所用的任务世界坐标系。",
        "J1–J6 是六个运动关节；RViz 播放的轨迹来自已保存采样，不重新规划。", "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
