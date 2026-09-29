#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
探测在给定的目标位姿下, LINK_6(含夹爪) 的碰撞球会占据多大范围,
从而确定墙盒 (wall.bounds_override) 需要放宽到多少。

关键点: 夹爪碰撞球在 LINK_6 局部坐标系下是固定的, 而 LINK_6 的位姿由
用户给定的目标位姿唯一确定, 因此夹爪的世界位置与 IK 解无关, 可直接算。

用法(conda curobo 环境):
  python scripts/probe_ee_extent.py
  python scripts/probe_ee_extent.py --config config/my_task.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_trajectory import load_robot_cfg_dict  # noqa: E402
from xtrainer_common import (  # noqa: E402
    build_pose_sequence,
    load_task_config,
    quat_wxyz_to_matrix,
)

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel  # noqa: E402
from curobo.types.robot import RobotConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="探测 LINK_6(含夹爪) 碰撞球范围")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--margin", type=float, default=0.02, help="建议值的额外余量(m)")
    args = ap.parse_args()

    cfg = load_task_config(args.config)
    rd = load_robot_cfg_dict(cfg["robot"])
    coll_links: List[str] = list(rd["robot_cfg"]["kinematics"]["collision_link_names"])
    # 碰撞球挂在法兰 link(LINK_6) 上; ee_link 已改为 TCP_LINK(纯坐标系, 无球),
    # 因此这里必须用 flange_link 取球, 并以 flange_link 的位姿作为局部系原点。
    ee_link = cfg["robot"].get("flange_link") or "LINK_6"

    km = CudaRobotModel(RobotConfig.from_dict(rd["robot_cfg"]).kinematics)
    idx_map = km.kinematics_config.link_sphere_idx_map.cpu().numpy()
    ee_li = coll_links.index(ee_link)
    ee_mask = idx_map == ee_li

    # 取零位构型, 提取 LINK_6 的球在 LINK_6 局部系下的坐标
    q0 = torch.zeros((1, km.get_dof()), device="cuda:0")
    st = km.get_state(q0)
    sph0 = st.link_spheres_tensor[0].detach().cpu().numpy()[ee_mask]  # [k, 4]
    all_links = list(km.link_names)
    if ee_link in all_links:
        li = all_links.index(ee_link)
        ee_p0 = st.links_position[0, li].detach().cpu().numpy()
        ee_q0 = st.links_quaternion[0, li].detach().cpu().numpy()
    else:
        ee_p0 = st.ee_position[0].detach().cpu().numpy()
        ee_q0 = st.ee_quaternion[0].detach().cpu().numpy()
    R0 = quat_wxyz_to_matrix(ee_q0)
    # world = R0 @ local + ee_p0  =>  local = R0^T @ (world - ee_p0)
    local_c = (sph0[:, :3] - ee_p0[None, :]) @ R0
    local_r = sph0[:, 3]
    print(f"[INFO] {ee_link} 有 {len(local_r)} 个碰撞球, 在 {ee_link} 局部系下:")
    for i, (c, r) in enumerate(zip(local_c, local_r)):
        print(f"       sph[{i}] local=[{c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f}] r={r:.4f}")
    print(f"[INFO] 局部 z 范围: [{(local_c[:,2]-local_r).min():+.4f}, "
          f"{(local_c[:,2]+local_r).max():+.4f}] (夹爪沿 +z 伸出)")

    b = cfg["workspace"]["bounds"]
    lo = np.array([b["x"][0], b["y"][0], b["z"][0]], dtype=np.float64)
    hi = np.array([b["x"][1], b["y"][1], b["z"][1]], dtype=np.float64)
    print(f"\n[INFO] 声明的工作空间 lo={lo.tolist()} hi={hi.tolist()}")

    # 逐目标位姿计算 LINK_6 所有球的世界包围盒
    seq = build_pose_sequence(cfg["task"])
    glo = np.full(3, np.inf)
    ghi = np.full(3, -np.inf)
    print(f"\n{'位姿':<12s} {'x_min':>7s} {'x_max':>7s} {'y_min':>7s} {'y_max':>7s} "
          f"{'z_min':>7s} {'z_max':>7s}   越界(mm) x/y/z")
    for p in seq:
        R = quat_wxyz_to_matrix(p.quat_wxyz)
        wc = local_c @ R.T + p.position[None, :]
        mn = (wc - local_r[:, None]).min(axis=0)
        mx = (wc + local_r[:, None]).max(axis=0)
        glo = np.minimum(glo, mn)
        ghi = np.maximum(ghi, mx)
        under = np.maximum(lo - mn, 0.0)
        over = np.maximum(mx - hi, 0.0)
        v = np.maximum(under, over) * 1000.0
        print(f"{p.name:<12s} {mn[0]:+7.3f} {mx[0]:+7.3f} {mn[1]:+7.3f} {mx[1]:+7.3f} "
              f"{mn[2]:+7.3f} {mx[2]:+7.3f}   {v[0]:5.1f}/{v[1]:5.1f}/{v[2]:5.1f}")

    print(f"\n[INFO] 全部位姿合并的 {ee_link} 包围盒:")
    print(f"       x=[{glo[0]:+.4f},{ghi[0]:+.4f}] y=[{glo[1]:+.4f},{ghi[1]:+.4f}] "
          f"z=[{glo[2]:+.4f},{ghi[2]:+.4f}]")

    m = float(args.margin)
    need_lo = np.minimum(lo, glo - m)
    need_hi = np.maximum(hi, ghi + m)
    changed = ~(np.isclose(need_lo, lo) & np.isclose(need_hi, hi))
    print(f"\n[建议] config/task_default.yaml -> workspace.wall.bounds_override "
          f"(含 {m * 1000:.0f}mm 余量):")
    for i, axis in enumerate("xyz"):
        tag = "  <== 需放宽" if changed[i] else ""
        print(f"         {axis}: [{need_lo[i]:.3f}, {need_hi[i]:.3f}]{tag}")
    print("\n       说明: bounds 保持你声明的实际工作空间(用于软校验 ee 原点),")
    print("             bounds_override 仅供墙使用, 让夹爪也能装进盒子。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
