#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本: 定位起始/目标位姿 IK 失败的原因。

分别在三种世界下求 IK, 并逐个 link 报告碰撞距离, 用于判断是
「自碰撞」「与工作空间墙碰撞」还是「运动学不可达」。

用法(conda curobo 环境):
  python scripts/diagnose_ik.py
  python scripts/diagnose_ik.py --config config/my_task.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_trajectory import load_robot_cfg_dict, make_world_config  # noqa: E402
from xtrainer_common import (  # noqa: E402
    PoseSpec,
    build_pose_sequence,
    load_task_config,
    quat_angle_deg,
)

from curobo.geom.sdf.world import CollisionCheckerType  # noqa: E402
from curobo.geom.types import WorldConfig  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.math import Pose  # noqa: E402
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig  # noqa: E402


def make_dummy_world() -> WorldConfig:
    """curobo 的 primitive collision 不允许零障碍物, 因此放一个远处的小盒子作占位。"""
    return WorldConfig.from_dict(
        {"cuboid": {"dummy_far": {"dims": [0.01, 0.01, 0.01], "pose": [50.0, 50.0, 50.0, 1, 0, 0, 0]}}}
    )


def make_ik(robot_dict: Dict[str, Any], world, pl: Dict[str, Any], self_coll: bool) -> IKSolver:
    cfg = IKSolverConfig.load_from_robot_config(
        robot_dict,
        world,
        rotation_threshold=float(pl["rotation_threshold"]),
        position_threshold=float(pl["position_threshold"]),
        num_seeds=128,
        self_collision_check=self_coll,
        self_collision_opt=self_coll,
        tensor_args=TensorDeviceType(),
        use_cuda_graph=False,
        collision_activation_distance=float(pl["collision_activation_distance"]),
        collision_checker_type=CollisionCheckerType.MESH,
    )
    return IKSolver(cfg)


def report_link_collisions(
    ik: IKSolver, q: torch.Tensor, world_has_obstacles: bool, coll_links: List[str]
) -> None:
    """逐 link 报告自碰撞与世界碰撞情况。coll_links 顺序必须与 link_sphere_idx_map 一致。"""
    kin = ik.kinematics
    state = kin.get_state(q)
    sph = state.link_spheres_tensor  # [B, n_sph, 4]
    idx_map = kin.kinematics_config.link_sphere_idx_map.cpu().numpy()

    # 世界碰撞: 逐球查询 sdf
    if world_has_obstacles:
        rollout = ik.solver.safety_rollout
        wcoll = rollout.primitive_collision_constraint
        d = wcoll.forward(sph.unsqueeze(1)).reshape(-1).detach().cpu().numpy()
        print(f"    世界碰撞代价(>0 即碰撞): total={d.sum():.6f}")
        # 逐 link 汇总: 定位是哪个 link 撞墙
        per_link: Dict[str, float] = {}
        for li, name in enumerate(coll_links):
            mask = idx_map == li
            if not mask.any():
                continue
            s = sph[:, mask, :].unsqueeze(1)
            dv = wcoll.forward(s).reshape(-1).detach().cpu().numpy()
            per_link[name] = float(dv.sum())
        for name, v in per_link.items():
            flag = "  <== 撞墙" if v > 1e-9 else ""
            print(f"      {name:<10s} world_cost={v:+.6f}{flag}")

    # 自碰撞
    try:
        sc = ik.solver.safety_rollout.robot_self_collision_constraint
        d_self = sc.forward(sph.unsqueeze(1)).reshape(-1).detach().cpu().numpy()
        print(f"    自碰撞代价(>0 即碰撞): {d_self.sum():+.6f}")
    except Exception as e:  # noqa: BLE001
        print(f"    自碰撞查询失败: {e}")


def diagnose_pose(
    p: PoseSpec, robot_dict: Dict[str, Any], world_wall, pl: Dict[str, Any]
) -> None:
    ta = TensorDeviceType()
    coll_links = list(robot_dict["robot_cfg"]["kinematics"]["collision_link_names"])
    goal = Pose(
        position=ta.to_device(p.position.reshape(1, 3)),
        quaternion=ta.to_device(p.quat_wxyz.reshape(1, 4)),
    )
    empty_world = make_dummy_world()

    print(f"\n{'=' * 72}\n位姿 {p.name}: xyz={p.position.round(4).tolist()} "
          f"rpy={p.rpy_deg.round(2).tolist()}\n{'=' * 72}")

    cases = [
        ("A. 纯运动学(无自碰撞, 无墙)", empty_world, False),
        ("B. 自碰撞 only(无墙)", empty_world, True),
        ("C. 自碰撞 + 工作空间墙", world_wall, True),
    ]
    q_ok = None
    for title, world, self_coll in cases:
        ik = make_ik(robot_dict, world, pl, self_coll)
        res = ik.solve_single(goal)
        n_ok = int(res.success.sum().item()) if res.success is not None else 0
        pe = float(res.position_error.min().item()) if res.position_error is not None else -1
        re = float(res.rotation_error.min().item()) if res.rotation_error is not None else -1
        tag = "OK " if n_ok > 0 else "FAIL"
        print(f"\n  [{tag}] {title}")
        print(f"         success={n_ok}  min_pos_err={pe * 1000:.4f}mm  min_rot_err={re:.5f}")
        if n_ok > 0:
            sol = res.solution[res.success].view(-1, ik.dof)
            q_ok = sol[0:1]
            print(f"         q = {sol[0].detach().cpu().numpy().round(4).tolist()}")
        elif q_ok is not None:
            # 用上一档(更宽松)的解, 检查它在当前档为何不合法
            print("         用上一档的解检查当前档的碰撞情况:")
            report_link_collisions(ik, q_ok, world is world_wall, coll_links)
        del ik
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description="XTrainer IK 失败诊断")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    cfg = load_task_config(args.config)
    rb, task, ws, pl = cfg["robot"], cfg["task"], cfg["workspace"], cfg["planner"]

    robot_dict = load_robot_cfg_dict(rb)
    world_wall = make_world_config(ws)

    for p in build_pose_sequence(task):
        diagnose_pose(p, robot_dict, world_wall, pl)

    print(f"\n{'=' * 72}")
    print("解读:")
    print("  A FAIL            -> 位姿运动学不可达, 需要改位置/姿态")
    print("  A OK, B FAIL      -> 该位姿必然自碰撞, 需要改姿态")
    print("  B OK, C FAIL      -> 撞工作空间墙; 看上面逐 link 报告, ")
    print("                       若是 LINK_0/LINK_1 则要放大 base_clearance 让位孔")
    print("  C OK              -> 该位姿可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
