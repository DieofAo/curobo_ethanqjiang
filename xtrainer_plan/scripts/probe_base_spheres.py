#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
探测 XTrainer 各 link 碰撞球在 base 系下的分布, 用于确定工作空间墙的
base 让位孔尺寸。

LINK_0 的球完全静止; LINK_1 的球全部位于 J_1 旋转轴上, 因此也与关节角无关。
这两个 link 构成"底座立柱", 必然穿透 x=0 与 z=0.1 两个墙面, 需要开孔让位。

用法(conda curobo 环境):
  python scripts/probe_base_spheres.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_trajectory import load_robot_cfg_dict  # noqa: E402
from xtrainer_common import load_task_config  # noqa: E402

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel  # noqa: E402
from curobo.types.robot import RobotConfig  # noqa: E402


def main() -> int:
    cfg = load_task_config()
    robot_dict = load_robot_cfg_dict(cfg["robot"])
    coll_links = list(robot_dict["robot_cfg"]["kinematics"]["collision_link_names"])

    km = CudaRobotModel(RobotConfig.from_dict(robot_dict["robot_cfg"]).kinematics)
    dof = km.get_dof()
    idx_map = km.kinematics_config.link_sphere_idx_map.cpu().numpy()
    print(f"[INFO] dof={dof}, collision_link_names={coll_links}")
    print(f"[INFO] total spheres={idx_map.shape[0]}, idx_map unique={sorted(set(idx_map.tolist()))}")

    # 随机 + 极值构型, 统计每个 link 的球包围盒
    n = 20000
    lo, hi = km.get_joint_limits().position
    g = torch.Generator(device="cuda:0").manual_seed(0)
    q = torch.rand((n, dof), device="cuda:0", generator=g) * (hi - lo) + lo
    q[0] = 0.0  # 零位也放进去
    sph = km.get_state(q).link_spheres_tensor.detach().cpu().numpy()  # [n, n_sph, 4]

    print(f"\n{'link':<10s} {'n_sph':>5s}  "
          f"{'x_min':>7s} {'x_max':>7s} {'y_min':>7s} {'y_max':>7s} {'z_min':>7s} {'z_max':>7s}  静止?")
    static_links = []
    for li, name in enumerate(coll_links):
        mask = idx_map == li
        if not mask.any():
            continue
        s = sph[:, mask, :]                     # [n, k, 4]
        c, r = s[..., :3], s[..., 3:4]
        mn = (c - r).reshape(-1, 3).min(axis=0)
        mx = (c + r).reshape(-1, 3).max(axis=0)
        # 判断球心是否随关节变化
        spread = float(np.abs(c - c[0:1]).max())
        is_static = spread < 1e-4
        if is_static:
            static_links.append(name)
        print(f"{name:<10s} {int(mask.sum()):>5d}  "
              f"{mn[0]:+7.3f} {mx[0]:+7.3f} {mn[1]:+7.3f} {mx[1]:+7.3f} {mn[2]:+7.3f} {mx[2]:+7.3f}  "
              f"{'YES' if is_static else f'no (spread={spread:.3f})'}")

    print(f"\n[INFO] 与关节角无关的 link(可视为底座立柱): {static_links}")
    if static_links:
        m = np.zeros_like(idx_map, dtype=bool)
        for name in static_links:
            m |= idx_map == coll_links.index(name)
        s = sph[0][m]
        c, r = s[:, :3], s[:, 3:4]
        mn = (c - r).min(axis=0)
        mx = (c + r).max(axis=0)
        print(f"[INFO] 立柱包围盒: x=[{mn[0]:+.3f},{mx[0]:+.3f}] "
              f"y=[{mn[1]:+.3f},{mx[1]:+.3f}] z=[{mn[2]:+.3f},{mx[2]:+.3f}]")
        print("\n[建议] config/task_default.yaml -> workspace.wall.base_clearance:")
        print(f"         y_half: {max(abs(mn[1]), abs(mx[1])) + 0.01:.3f}")
        print(f"         z_max:  {mx[2] + 0.01:.3f}")
        print(f"         x_min:  {mn[0] - 0.01:.3f}")
        print("       (各方向已留 10mm 余量)")
        print("\n[逐球明细]")
        for i, (cc, rr) in enumerate(zip(c, r[:, 0])):
            print(f"         sph[{i}] center=[{cc[0]:+.4f},{cc[1]:+.4f},{cc[2]:+.4f}] r={rr:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
