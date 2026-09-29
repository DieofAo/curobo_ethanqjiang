#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为「起终点处于不同 IK 分支」的情况自动搜索过渡路点。

背景: 当 start 与 goal 的 IK 解相距上百度时(肘/腕翻转), 在很小的笛卡尔位移内
      直接规划会 FINETUNE_TRAJOPT_FAIL。此时需要在中间插入过渡位姿, 把一次
      大翻转拆成若干可完成的小段。

做法: 在 start 与 goal 的关节空间之间做线性插值, 对每个插值构型做 FK 得到
      笛卡尔位姿, 再筛掉越界/碰撞的, 最后输出可直接粘进
      task.waypoints.extra 的 yaml 片段。

关节空间插值天然保证「同分支、路径最短」, 因此由它导出的中间位姿必然可达且连续。

用法(conda curobo 环境):
  python scripts/find_transition_waypoints.py
  python scripts/find_transition_waypoints.py --n-mid 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_trajectory import (  # noqa: E402
    load_robot_cfg_dict,
    make_world_config,
    restrict_world_collision_to_links,
    solve_ik_for_pose,
)
from xtrainer_common import (  # noqa: E402
    build_pose_sequence,
    check_in_bounds,
    load_task_config,
    quat_wxyz_to_rpy_deg,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="搜索过渡路点以分解 IK 分支翻转")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--n-mid", type=int, default=1,
                    help="插入的过渡路点数量, 1~3 通常够用")
    ap.add_argument("--n-probe", type=int, default=41,
                    help="关节空间插值的采样密度")
    args = ap.parse_args()

    cfg = load_task_config(args.config)
    rb, task, ws, pl = cfg["robot"], cfg["task"], cfg["workspace"], cfg["planner"]
    robot_dict = load_robot_cfg_dict(rb)
    world = make_world_config(ws)
    wall_links = list((ws.get("wall") or {}).get("collision_link_names") or [])

    seq = build_pose_sequence(task)
    # 只关心 start 侧最后一个点 与 goal 侧第一个点之间的翻转
    # 默认序列: start, start_lift, [extra...], goal_lift, goal
    i_from = 1 if len(seq) > 2 else 0
    i_to = len(seq) - 2 if len(seq) > 2 else len(seq) - 1
    p_from, p_to = seq[i_from], seq[i_to]
    print(f"[INFO] 分解区间: {p_from.name} -> {p_to.name}")

    q_from, _ = solve_ik_for_pose(robot_dict, world, pl, p_from, wall_link_names=wall_links)
    q_to, _ = solve_ik_for_pose(robot_dict, world, pl, p_to, wall_link_names=wall_links)
    if q_from is None or q_to is None:
        print("[FAIL] 端点 IK 无解, 无法搜索过渡路点")
        return 2
    print(f"[INFO] q_from = {np.degrees(q_from).round(1).tolist()}")
    print(f"[INFO] q_to   = {np.degrees(q_to).round(1).tolist()}")
    print(f"[INFO] 关节最大变化 = {np.degrees(np.abs(q_to - q_from)).max():.0f}deg")

    # ---- 关节空间线性插值 + FK + 校验 ----
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.robot import RobotConfig
    from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

    km = CudaRobotModel(RobotConfig.from_dict(robot_dict["robot_cfg"]).kinematics)
    rw = RobotWorld(RobotWorldConfig.load_from_config(robot_config=robot_dict, world_model=None))

    ts = np.linspace(0.0, 1.0, int(args.n_probe))
    qs = q_from[None, :] + ts[:, None] * (q_to - q_from)[None, :]
    import torch

    q_t = km.tensor_args.to_device(qs) if hasattr(km, "tensor_args") else torch.as_tensor(
        qs, dtype=torch.float32, device="cuda:0"
    )
    st = km.get_state(q_t)
    ee_pos = st.ee_position.detach().cpu().numpy()
    ee_quat = st.ee_quaternion.detach().cpu().numpy()

    # 自碰撞
    d_self = rw.get_self_collision_distance(st.link_spheres_tensor.unsqueeze(1))
    if d_self.dim() > 1:
        d_self = d_self.squeeze(-1)
    self_ok = (d_self.reshape(-1).detach().cpu().numpy() <= 0)

    # 世界碰撞(墙): 必须按真实的碰撞球来查, 只查 ee 原点会漏掉伸出的夹爪。
    # 复用规划时同一套 rollout + per-link 屏蔽, 保证判定口径与规划完全一致。
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    ik_probe = IKSolver(
        IKSolverConfig.load_from_robot_config(
            robot_dict,
            world,
            rotation_threshold=float(pl["rotation_threshold"]),
            position_threshold=float(pl["position_threshold"]),
            num_seeds=1,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=TensorDeviceType(),
            use_cuda_graph=False,
            collision_activation_distance=float(pl["collision_activation_distance"]),
            collision_checker_type=CollisionCheckerType.MESH,
        )
    )
    if wall_links:
        restrict_world_collision_to_links(ik_probe, wall_links, quiet=True)
    wcost = None
    for rollout in ik_probe.get_all_rollout_instances():
        c = getattr(rollout, "primitive_collision_constraint", None)
        if c is not None and getattr(c, "enabled", True):
            wcost = c
            break
    if wcost is None:
        print("[WARN] 找不到世界碰撞 cost, 跳过撞墙校验")
        world_ok = np.ones(len(ts), dtype=bool)
    else:
        wc = wcost.forward(st.link_spheres_tensor.unsqueeze(1)).reshape(len(ts), -1)
        world_ok = (wc.sum(dim=-1) <= 0).detach().cpu().numpy()

    # ee 原点是否在声明的工作空间内(仅信息用, 不作为筛选条件)
    in_box, _ = check_in_bounds(ee_pos, ws)

    ok = self_ok & world_ok
    print(f"\n[INFO] 关节插值 {len(ts)} 个采样点:")
    print(f"       自碰撞通过   {int(self_ok.sum())}/{len(ts)}")
    print(f"       墙碰撞通过   {int(world_ok.sum())}/{len(ts)}   <- 按真实碰撞球(含夹爪)判定")
    print(f"       ee 原点在 bounds 内 {int(in_box.sum())}/{len(ts)} (仅参考)")
    print(f"       全部通过     {int(ok.sum())}/{len(ts)}")
    if not ok.any():
        print("\n[FAIL] 关节空间直线路径整段都撞墙, 无法从中取过渡路点。")
        print("       这条翻转路径会让末端/夹爪扫出工作空间盒。可选:")
        print("         1) 放宽 wall.bounds_override (尤其 z 上界), 给翻转留出空间")
        print("         2) 改用与起点同 IK 分支的目标姿态, 从根本上避免翻转")
        print("         3) wall.collision_link_names 改为只约束靠法兰的球")
        return 3
    if not ok.all():
        bad = np.where(~ok)[0]
        print(f"[WARN] {len(bad)} 个采样点不可用, 过渡路点会自动避开")

    # ---- 均匀挑选 n_mid 个可用的过渡点 ----
    cand = np.where(ok)[0]
    cand = cand[(cand > 0) & (cand < len(ts) - 1)]
    if len(cand) == 0:
        print("[FAIL] 插值路径上没有可用的中间构型")
        return 3

    n = max(1, int(args.n_mid))
    picks: List[int] = []
    for k in range(1, n + 1):
        target_t = k / (n + 1)
        j = cand[int(np.argmin(np.abs(ts[cand] - target_t)))]
        if j not in picks:
            picks.append(int(j))

    print(f"\n[RESULT] 建议插入 {len(picks)} 个过渡路点:")
    print("         把下面这段粘到 config 的 task.waypoints.extra 下\n")
    print("    extra:")
    for j in picks:
        pos = ee_pos[j]
        rpy = quat_wxyz_to_rpy_deg(ee_quat[j])
        qd = np.degrees(qs[j])
        print(f"      # t={ts[j]:.2f}  ee=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] "
              f"rpy=[{rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}]")
        print(f"      - joint_deg: [{', '.join(f'{v:.2f}' for v in qd)}]")

    print("\n[提示] 用 joint_deg(关节角)而非 position/rpy_deg, 是为了锁定构型:")
    print("       若写成位姿, IK 会为该位姿重新选解, 可能又跳到别的分支, 前功尽弃。")
    print("       关节目标走 plan_single_js, 保证整条路径构型连续。")
    print("       粘贴后直接 ./run_plan.sh 即可。若仍失败, 加大 --n-mid 再试。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
