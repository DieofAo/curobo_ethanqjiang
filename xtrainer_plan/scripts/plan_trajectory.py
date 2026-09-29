#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XTrainer 轨迹规划 (curobo MotionGen)

功能:
  1. 从 yaml / 命令行读取起始与目标位姿(位置 + rpy 度, 相对 base=LINK_0)
  2. 可选自动插入"抬升"路点(进入抓取 / 进入放置), 也支持手工 extra 路点
  3. 全程开启自碰撞检测
  4. 用 6 面 cuboid 墙把末端硬约束在 x[-0.5,0] y[-0.2,0.2] z[0.1,0.7] 内
     (底座位置开让位孔, 见 xtrainer_common.build_workspace_wall_cuboids)
  5. 分段串联规划(start -> wp1 -> ... -> goal), 段间速度归零, 拼接成一条轨迹
  6. 导出 trajectory.npz + trajectory_meta.json, 供 ROS1 播放脚本读取

必须在 conda curobo 环境下运行:
  conda activate curobo && python plan_trajectory.py

用法:
  python plan_trajectory.py
  python plan_trajectory.py --config config/my_task.yaml
  python plan_trajectory.py --start-position -0.31 -0.05 0.2 --start-rpy -135.56 -0.93 -8.9 \
                            --goal-position -0.34 -0.09 0.22 --goal-rpy 177 53 -84
  python plan_trajectory.py --no-lift              # 不要抬升路点, 直接 start -> goal
  python plan_trajectory.py --lift-z 0.10 0.12     # 分别设置起点/目标抬升高度
  python plan_trajectory.py --no-wall              # 关掉硬墙, 只做规划后软校验
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtrainer_common import (  # noqa: E402
    PoseSpec,
    apply_variant_combo,
    build_pose_variants,
    build_variant_combos,
    build_pose_sequence,
    dump_json,
    build_workspace_wall_cuboids,
    check_in_bounds,
    load_task_config,
    parse_rigid_transform_matrix,
    quat_angle_deg,
    quat_wxyz_to_rpy_deg,
    quat_wxyz_to_xyzw,
    resolve_repo_path,
    save_trajectory,
)


# ============================== 参数 ==============================


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="XTrainer 轨迹规划 (curobo)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=str, default=None, help="任务 yaml, 默认 config/task_default.yaml")

    g = ap.add_argument_group("位姿覆盖 (相对 base=LINK_0, rpy 单位度, ROS 固定轴 XYZ)")
    g.add_argument("--start-position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    g.add_argument("--start-rpy", type=float, nargs=3, default=None, metavar=("R", "P", "Y"))
    g.add_argument("--goal-position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    g.add_argument("--goal-rpy", type=float, nargs=3, default=None, metavar=("R", "P", "Y"))
    g.add_argument("--start-joint-state", type=float, nargs="+", default=None,
                   help="起始关节角(rad, 6 个). 不给则从 start 位姿 IK 求解")

    g = ap.add_argument_group("路点")
    g.add_argument("--no-lift", action="store_true", help="禁用自动抬升路点")
    g.add_argument("--lift-z", type=float, nargs="+", default=None,
                   help="抬升高度(m). 1 个值 = 起点/目标相同; 2 个值 = 分别指定")
    g.add_argument("--lift-axis", type=str, default=None, choices=["base_z", "tool_z_neg"])

    g = ap.add_argument_group("目标姿态变体(批量对比)")
    g.add_argument("--variant", type=str, nargs="+", default=None, metavar="IDX_OR_NAME",
                   help="只跑指定变体(索引或名字), 便于单独复现某一个")
    g.add_argument("--no-variants", action="store_true",
                   help="忽略 goal_variants, 只按 goal 规划一次")
    g.add_argument("--goal-rotations", type=str, default=None,
                   help="覆盖变体列表, 格式 'r,p,y; r,p,y; ...' (度)")
    g.add_argument("--variant-mode", type=str, default=None, choices=["right", "left"],
                   help="right=绕工具自身轴(默认); left=绕 base 固定轴")

    g = ap.add_argument_group("工作空间")
    g.add_argument("--bounds-x", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    g.add_argument("--bounds-y", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    g.add_argument("--bounds-z", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    g.add_argument("--no-wall", action="store_true", help="不加 cuboid 硬墙, 仅规划后软校验")

    g = ap.add_argument_group("规划器")
    g.add_argument("--ee-link", type=str, default=None, help="规划的目标 link, 默认 LINK_6")
    g.add_argument("--no-self-collision", action="store_true", help="关闭自碰撞检测(调试用)")
    g.add_argument("--enable-graph", action="store_true", help="启用图搜索(PRM)寻找绕行种子")
    g.add_argument("--velocity-scale", type=float, default=None)
    g.add_argument("--interpolation-dt", type=float, default=None)
    g.add_argument("--max-attempts", type=int, default=None)
    g.add_argument("--num-trajopt-seeds", type=int, default=None,
                   help="trajopt 种子数, 越多越可能找到关节位移更小的解(也更慢)")
    g.add_argument("--cspace-weight", type=float, nargs="+", default=None,
                   metavar="W",
                   help="关节空间距离权重(6 个值), 调高大关节权重可减少整臂绕行")

    g = ap.add_argument_group("输出")
    g.add_argument("--out-dir", type=str, default=None, help="输出目录, 默认 results/<时间戳>")
    g.add_argument("--no-timestamp", action="store_true", help="输出目录不追加时间戳")
    return ap


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """命令行覆盖 yaml 配置。"""
    t, ws, pl, rb = cfg["task"], cfg["workspace"], cfg["planner"], cfg["robot"]

    if args.start_position is not None:
        t["start"]["position"] = list(args.start_position)
    if args.start_rpy is not None:
        t["start"]["rpy_deg"] = list(args.start_rpy)
    if args.goal_position is not None:
        t["goal"]["position"] = list(args.goal_position)
    if args.goal_rpy is not None:
        t["goal"]["rpy_deg"] = list(args.goal_rpy)
    if args.start_joint_state is not None:
        t["start_joint_state"] = list(args.start_joint_state)

    lift = t.setdefault("waypoints", {}).setdefault("auto_lift", {})
    if args.no_lift:
        lift["enable"] = False
    if args.lift_z is not None:
        lift["enable"] = True
        if len(args.lift_z) == 1:
            lift["start_lift_z"] = lift["goal_lift_z"] = float(args.lift_z[0])
        elif len(args.lift_z) == 2:
            lift["start_lift_z"] = float(args.lift_z[0])
            lift["goal_lift_z"] = float(args.lift_z[1])
        else:
            raise ValueError("--lift-z 只接受 1 或 2 个值")
    if args.lift_axis is not None:
        lift["lift_axis"] = args.lift_axis

    gv = t.setdefault("goal_variants", {})
    if args.no_variants:
        gv["enable"] = False
    if args.goal_rotations is not None:
        rots = []
        for part in args.goal_rotations.split(";"):
            part = part.strip()
            if not part:
                continue
            vals = [float(x) for x in part.replace(",", " ").split()]
            if len(vals) != 3:
                raise ValueError(f"--goal-rotations 每组需 3 个值, 得到 {part!r}")
            rots.append(vals)
        if rots:
            gv["enable"] = True
            gv["rotations"] = rots
    if args.variant_mode is not None:
        gv["mode"] = args.variant_mode

    for axis, val in (("x", args.bounds_x), ("y", args.bounds_y), ("z", args.bounds_z)):
        if val is not None:
            ws["bounds"][axis] = list(val)
    if args.no_wall:
        ws.setdefault("wall", {})["enable"] = False

    if args.ee_link is not None:
        rb["ee_link"] = args.ee_link
    if args.no_self_collision:
        pl["self_collision_check"] = False
        pl["self_collision_opt"] = False
    if args.enable_graph:
        pl["enable_graph"] = True
    if args.velocity_scale is not None:
        pl["velocity_scale"] = float(args.velocity_scale)
    if args.interpolation_dt is not None:
        pl["interpolation_dt"] = float(args.interpolation_dt)
    if args.max_attempts is not None:
        pl["max_attempts"] = int(args.max_attempts)
    if args.num_trajopt_seeds is not None:
        pl["num_trajopt_seeds"] = int(args.num_trajopt_seeds)
    if args.cspace_weight is not None:
        rb["cspace_distance_weight"] = [float(v) for v in args.cspace_weight]

    if args.out_dir is not None:
        cfg["output"]["dir"] = args.out_dir
    if args.no_timestamp:
        cfg["output"]["add_timestamp"] = False
    return cfg


# ============================== 世界 / 机器人 ==============================


def make_world_config(ws_cfg: Dict[str, Any]):
    """构造 WorldConfig: 工作空间 6 面墙。"""
    from curobo.geom.types import WorldConfig

    wall_cfg = ws_cfg.get("wall") or {}
    if not wall_cfg.get("enable", False):
        print("[WORLD] 硬墙已关闭, 世界为空(仅自碰撞 + 规划后软校验)")
        # curobo 的 primitive collision 不允许零障碍物, 放一个远处小盒子占位
        return WorldConfig.from_dict(
            {
                "cuboid": {
                    "dummy_far": {"dims": [0.01, 0.01, 0.01], "pose": [50.0, 50.0, 50.0, 1, 0, 0, 0]}
                }
            }
        )

    cuboids = build_workspace_wall_cuboids(ws_cfg)
    cuboid_dict = {c["name"]: {"dims": c["dims"], "pose": c["pose"]} for c in cuboids}
    print(f"[WORLD] 工作空间墙: {len(cuboid_dict)} 个 cuboid")
    for c in cuboids:
        d, p = c["dims"], c["pose"]
        print(
            f"         {c['name']:<20s} dims=[{d[0]:.3f},{d[1]:.3f},{d[2]:.3f}] "
            f"center=[{p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}]"
        )
    return WorldConfig.from_dict({"cuboid": cuboid_dict})


def load_robot_cfg_dict(robot_cfg_yaml: Dict[str, Any]) -> Dict[str, Any]:
    """加载并按需修改 curobo robot yml。"""
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml

    d = load_yaml(join_path(get_robot_configs_path(), robot_cfg_yaml["robot_yml"]))
    kin = d["robot_cfg"]["kinematics"]

    kin["base_link"] = robot_cfg_yaml["base_link"]
    kin["ee_link"] = robot_cfg_yaml["ee_link"]
    # link_names 里额外带上 gripper_link 与 flange_link(碰撞球所在 link),
    # 便于同时输出多个 link 的 FK 供 tf 发布与越界报告使用。
    extra = [robot_cfg_yaml["ee_link"]]
    for ln in ("gripper_link", robot_cfg_yaml.get("flange_link") or "LINK_6"):
        if ln and ln not in extra:
            extra.append(ln)
    kin["link_names"] = extra

    # 双臂模型 (mimic 6-DOF 或 independent 12-DOF) 需要说明第二臂
    # 的前缀, 以便:
    #   * 工作空间墙把第二臂的同名 link 一并约束住 (见 main 里的 wall_links);
    #   * link_names 带上第二臂 ee link, 便于报告里直接看第二臂位姿。
    # 前缀必须与 build_scene_urdf.py --planning-prefix /
    # build_dual_robot_yml.py --prefix 一致 (默认 second_)。
    dual_prefix = robot_cfg_yaml.get("dual_arm_prefix") or ""
    if dual_prefix:
        if "xtrainer_dual_" not in str(kin.get("urdf_path", "")):
            print("[WARN] dual_arm_prefix 已设置, 但 urdf_path 不像 XTrainer "
                  "双臂模型, 第二臂 link 可能不存在")
        second_extra = [
            robot_cfg_yaml.get("second_ee_link") or dual_prefix + robot_cfg_yaml["ee_link"],
            dual_prefix + "gripper_link",
            robot_cfg_yaml.get("second_flange_link")
            or dual_prefix + (robot_cfg_yaml.get("flange_link") or "LINK_6"),
        ]
        for link_name in second_extra:
            if link_name and link_name not in kin["link_names"]:
                kin["link_names"].append(link_name)

    buf = float(robot_cfg_yaml.get("collision_sphere_buffer") or 0.0)
    if buf != 0.0:
        kin["collision_sphere_buffer"] = buf

    rc = robot_cfg_yaml.get("retract_config")
    if rc is not None:
        rc = list(rc)
        n_joint = len(kin["cspace"]["joint_names"])
        if len(rc) * 2 == n_joint and dual_prefix:
            rc = rc + rc
        if len(rc) != n_joint:
            raise ValueError(f"retract_config 需要 {n_joint} 个值, 实际 {len(rc)} 个")
        kin["cspace"]["retract_config"] = rc

    clip = robot_cfg_yaml.get("joint_limit_clip")
    if clip is not None:
        kin["cspace"]["position_limit_clip"] = float(clip)

    # 关节空间距离权重: 选择最优种子时按此权重计算关节路径长度
    # (evaluator.compute_path_length_cost: sum(|vel| * cspace_distance_weight))。
    # 调高大关节(J1~J3)的权重, 会让种子筛选更偏向"大关节少动"的解。
    # 在本任务配置里覆盖, 不改共享的 xtrainer.yml, 以免影响其他脚本。
    cdw = robot_cfg_yaml.get("cspace_distance_weight")
    if cdw is not None:
        n_joint = len(kin["cspace"]["joint_names"])
        cdw = list(cdw)
        if len(cdw) * 2 == n_joint and dual_prefix:
            cdw = cdw + cdw
        if len(cdw) != n_joint:
            raise ValueError(
                f"cspace_distance_weight 需要 {n_joint} 个值, 实际 {len(cdw)} 个"
            )
        old = list(kin["cspace"]["cspace_distance_weight"])
        kin["cspace"]["cspace_distance_weight"] = [float(v) for v in cdw]
        print(f"[ROBOT] cspace_distance_weight: {old} -> {kin['cspace']['cspace_distance_weight']}")
    else:
        print(f"[ROBOT] cspace_distance_weight = {kin['cspace']['cspace_distance_weight']} (yml 默认)")

    print(f"[ROBOT] {robot_cfg_yaml['robot_yml']}  {kin['base_link']} -> {kin['ee_link']}")
    print(f"[ROBOT] link_names(FK 输出) = {kin['link_names']}")
    print(f"[ROBOT] collision_link_names = {kin['collision_link_names']}")
    print(f"[ROBOT] collision_sphere_buffer = {kin.get('collision_sphere_buffer')}")
    print(f"[ROBOT] position_limit_clip = {kin['cspace'].get('position_limit_clip')}")
    return d


def make_motion_gen(
    robot_dict: Dict[str, Any], world, pl: Dict[str, Any],
    pre_warmup: Optional[Callable[[Any], None]] = None,
):
    """构造 MotionGen；可在 CUDA graph warmup 前安装代价函数补丁。"""
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    tensor_args = TensorDeviceType()
    checker = getattr(
        CollisionCheckerType, str(pl.get("collision_checker_type", "MESH")).upper()
    )
    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_dict,
        world,
        tensor_args,
        interpolation_dt=float(pl["interpolation_dt"]),
        collision_checker_type=checker,
        collision_activation_distance=float(pl["collision_activation_distance"]),
        self_collision_check=bool(pl["self_collision_check"]),
        self_collision_opt=bool(pl["self_collision_opt"]),
        num_ik_seeds=int(pl["num_ik_seeds"]),
        num_trajopt_seeds=int(pl["num_trajopt_seeds"]),
        num_graph_seeds=int(pl["num_graph_seeds"]),
        trajopt_tsteps=int(pl["trajopt_tsteps"]),
        position_threshold=float(pl["position_threshold"]),
        rotation_threshold=float(pl["rotation_threshold"]),
        velocity_scale=float(pl["velocity_scale"]),
        acceleration_scale=float(pl["acceleration_scale"]),
        jerk_scale=float(pl["jerk_scale"]),
        use_cuda_graph=bool(pl.get("use_cuda_graph", True)),
        store_debug_in_result=False,
    )
    mg = MotionGen(mg_cfg)
    if pre_warmup is not None:
        pre_warmup(mg)
    print("[PLAN] warming up MotionGen ...")
    t0 = time.time()
    mg.warmup(
        enable_graph=bool(pl.get("enable_graph", False)),
        warmup_js_trajopt=bool(pl.get("warmup_js_trajopt", False)),
    )
    print(f"[PLAN] warmup done in {time.time() - t0:.1f}s")
    print(f"[PLAN] self_collision_check = {pl['self_collision_check']}")
    return mg


# ============================== 只让末端受墙约束 ==============================


def restrict_world_collision_to_links(
    solver, wall_link_names: List[str], quiet: bool = False
) -> Dict[str, Any]:
    """让「墙(世界障碍物)」只约束 wall_link_names 里的 link, 自碰撞检测保持覆盖全部 link。

    solver 可以是 MotionGen 或 IKSolver(任何提供 get_all_rollout_instances() 的对象)。

    原理:
      curobo 的世界碰撞与自碰撞是两个彼此独立的 cost, 各自接收同一份碰撞球张量
      (arm_base.cost_fn 中分别调用 primitive_collision_cost.forward(state.robot_spheres)
       与 robot_self_collision_cost.forward(state.robot_spheres))。
      而 sphere_obb kernel 对负半径球会直接 early-return 写 0 代价:
        curobo/curobolib/cpp/sphere_obb_kernel.cu: if (sphere_cache.w < 0.0) { out=0; return; }
      因此只要在「世界碰撞」的 forward 入口处把非目标 link 的球半径改成负数,
      世界碰撞就会跳过它们; 自碰撞 cost 收到的仍是未修改的原始张量, 完全不受影响。

      注: 不能用官方的 CudaRobotModelConfig.disable_link_spheres(), 因为它直接改
      kinematics 里的球半径, 会让自碰撞也一起失效
      (self_collision_kernel.cu 中 `if (sph1.w <= 0) continue` 那段被注释掉了,
       负半径会使 r_diff = r1+r2 变成大负数, distance 恒为负 -> 自碰撞恒不触发)。

    Returns:
        报告 dict, 记录被屏蔽/保留的 link 与球数量。
    """
    import torch

    kin = solver.kinematics
    idx_map = kin.kinematics_config.link_sphere_idx_map
    coll_links: List[str] = list(kin.generator_config.collision_link_names)

    # link_sphere_idx_map 的取值是运动树全局 link 索引 (_name_to_idx_map),
    # 不是 collision_link_names 的序号。单臂时两者恰好同序, 但方案 A 的
    # mimic 双臂模型里第二臂 link 的全局索引 != 碰撞列表序号, 必须用
    # link_name_to_idx_map 做转换, 否则墙约束会罩到错误的球上。
    name_to_idx = dict(kin.kinematics_config.link_name_to_idx_map)

    unknown = [n for n in wall_link_names if n not in coll_links]
    if unknown:
        raise ValueError(
            f"wall.collision_link_names 中的 {unknown} 不在 collision_link_names={coll_links} 里"
        )

    keep_mask = torch.zeros_like(idx_map, dtype=torch.bool)
    for n in wall_link_names:
        keep_mask |= idx_map == name_to_idx[n]
    ignore_mask = ~keep_mask  # 需要被世界碰撞忽略的球
    n_keep = int(keep_mask.sum().item())
    n_ignore = int(ignore_mask.sum().item())

    # 对所有 rollout 实例(IK / trajopt / finetune / graph 的 safety rollout 等)打补丁
    patched = 0
    for rollout in solver.get_all_rollout_instances():
        for attr in ("primitive_collision_cost", "primitive_collision_constraint"):
            cost = getattr(rollout, attr, None)
            if cost is None or getattr(cost, "_xtrainer_link_masked", False):
                continue

            def make_wrapper(orig_fn, mask):
                n_sph = int(mask.shape[0])

                def wrapper(robot_spheres_in, *a, **kw):
                    # clone 后置负半径, 绝不改动上游张量(自碰撞用的是同一个对象)
                    s = robot_spheres_in.clone()
                    # 仅当输入包含全部碰撞球时才施加屏蔽。诊断代码可能只传某个 link
                    # 的球子集, 此时球数不匹配, 屏蔽掩码无意义, 直接放行。
                    if s.shape[-2] == n_sph:
                        s[..., mask, 3] = -100.0
                    return orig_fn(s, *a, **kw)

                return wrapper

            cost.forward = make_wrapper(cost.forward, ignore_mask)
            cost._xtrainer_link_masked = True
            patched += 1

    report = {
        "wall_link_names": list(wall_link_names),
        "all_collision_links": coll_links,
        "n_spheres_under_wall": n_keep,
        "n_spheres_ignored_by_wall": n_ignore,
        "patched_cost_terms": patched,
    }
    if not quiet:
        print(f"[WALL] 墙只约束 {wall_link_names}  ({n_keep} 个球)")
        print(f"[WALL] 墙忽略其余 {n_ignore} 个球; 自碰撞仍覆盖全部 {n_keep + n_ignore} 个球")
        print(f"[WALL] 已打补丁的 cost 项: {patched}")
    return report


# ============================== IK / 规划 ==============================


def solve_ik_for_pose(
    robot_dict: Dict[str, Any], world, pl: Dict[str, Any], pose_spec: PoseSpec,
    wall_link_names: Optional[List[str]] = None, seed_q=None
):
    """给定位姿求 IK(带自碰撞与世界碰撞检查)。返回 (q[dof] 或 None, result)。

    用独立的 IKSolver 实例, 不能复用 mg.ik_solver: MotionGen 内部的 IK solver 带
    cuda graph, 一旦先用 SINGLE goal 类型调用过, 后续 plan_single 传入不同 goal
    类型时会报 "changing goal type, cuda graph reset not available"。
    """
    import torch
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    ta = TensorDeviceType()
    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_dict,
        world,
        rotation_threshold=float(pl["rotation_threshold"]),
        position_threshold=float(pl["position_threshold"]),
        num_seeds=max(int(pl["num_ik_seeds"]), 64),
        self_collision_check=bool(pl["self_collision_check"]),
        self_collision_opt=bool(pl["self_collision_opt"]),
        tensor_args=ta,
        use_cuda_graph=False,
        collision_activation_distance=float(pl["collision_activation_distance"]),
        collision_checker_type=getattr(
            CollisionCheckerType, str(pl.get("collision_checker_type", "MESH")).upper()
        ),
    )
    ik = IKSolver(ik_cfg)
    if wall_link_names:
        restrict_world_collision_to_links(ik, wall_link_names, quiet=True)

    goal = Pose(
        position=ta.to_device(pose_spec.position.reshape(1, 3)),
        quaternion=ta.to_device(pose_spec.quat_wxyz.reshape(1, 4)),
    )
    seed = None
    if seed_q is not None:
        q = np.asarray(seed_q, dtype=np.float64).reshape(1, 1, -1)
        seed = ta.to_device(np.repeat(q, ik.num_seeds, axis=1))
    res = ik.solve_single(goal, seed_config=seed)
    if res.success is None or not bool(res.success.any()):
        return None, res
    sol = res.solution[res.success].view(-1, ik.dof)
    if seed_q is not None:
        d = torch.linalg.norm(sol - ta.to_device(np.asarray(seed_q).reshape(1, -1)), dim=-1)
        best = sol[int(torch.argmin(d))]
    else:
        best = sol[0]
    return best.detach().cpu().numpy().astype(np.float64), res


def plan_joint_interpolated(
    mg, start_q: np.ndarray, goal_q: np.ndarray, pl: Dict[str, Any]
) -> Optional[Dict[str, np.ndarray]]:
    """关节空间直线轨迹(带梯形速度规划) + 逐点碰撞校验。

    用于 plan_single_js 报 GRAPH_FAIL 的兜底。关节空间直线在构型上天然连续、
    不会跳分支, 且路径最短; 只要逐点校验通过(自碰撞 + 世界碰撞 + 限位), 就是安全的。

    Returns:
        {'position','velocity','acceleration','dt'} 或 None(校验不通过)
    """
    import torch

    q0 = np.asarray(start_q, dtype=np.float64)
    q1 = np.asarray(goal_q, dtype=np.float64)
    dq = q1 - q0
    if np.abs(dq).max() < 1e-9:
        return None

    dt = float(pl["interpolation_dt"])
    # 依据关节速度/加速度上限估算所需时长, 取最紧的那个关节
    lim = mg.kinematics.get_joint_limits()
    v_max = lim.velocity[1].cpu().numpy() * float(pl["velocity_scale"])
    a_max = lim.acceleration[1].cpu().numpy() * float(pl["acceleration_scale"])
    v_max = np.maximum(np.abs(v_max), 1e-3)
    a_max = np.maximum(np.abs(a_max), 1e-3)
    d = np.abs(dq)
    # 梯形(必要时三角形)速度曲线所需时间
    t_tri = 2.0 * np.sqrt(np.maximum(d, 0.0) / a_max)          # 三角形(未达 v_max)
    t_tra = d / v_max + v_max / a_max                           # 梯形
    peak_v = 0.5 * a_max * (t_tri / 2.0) * 2.0
    T = float(np.max(np.where(peak_v <= v_max, t_tri, t_tra)))
    T = max(T, 10.0 * dt)
    n = int(np.ceil(T / dt)) + 1

    # 用 5 次多项式做时间标定(首末速度与加速度均为 0), 保证平滑起停
    s = np.linspace(0.0, 1.0, n)
    scale = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5             # 位置
    d_scale = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / T     # 速度
    dd_scale = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / (T * T)  # 加速度

    pos = q0[None, :] + scale[:, None] * dq[None, :]
    vel = d_scale[:, None] * dq[None, :]
    acc = dd_scale[:, None] * dq[None, :]

    # ---- 逐点校验 ----
    q_t = mg.tensor_args.to_device(pos)
    st = mg.kinematics.get_state(q_t)

    # 关节限位
    lo = lim.position[0].cpu().numpy()
    hi = lim.position[1].cpu().numpy()
    if (pos < lo[None, :] - 1e-6).any() or (pos > hi[None, :] + 1e-6).any():
        print("    [FAIL] 关节空间插值超出关节限位")
        return None
    # 速度/加速度上限
    if (np.abs(vel) > np.abs(lim.velocity[1].cpu().numpy())[None, :] + 1e-6).any():
        print("    [WARN] 关节空间插值速度超过上限, 已按上限重算时长")
    # 自碰撞 + 世界碰撞
    sph = st.link_spheres_tensor.unsqueeze(1)
    bad_self = bad_world = 0
    for rollout in mg.get_all_rollout_instances():
        c = getattr(rollout, "robot_self_collision_constraint", None)
        if c is not None and getattr(c, "enabled", True):
            bad_self = int((c.forward(sph).reshape(-1) > 0).sum().item())
            break
    for rollout in mg.get_all_rollout_instances():
        c = getattr(rollout, "primitive_collision_constraint", None)
        if c is not None and getattr(c, "enabled", True):
            bad_world = int((c.forward(sph).reshape(-1) > 0).sum().item())
            break
    if bad_self or bad_world:
        print(f"    [FAIL] 关节空间插值有碰撞: 自碰撞 {bad_self} 点, 世界碰撞 {bad_world} 点")
        return None

    print(f"    [OK] 关节空间插值: {n} 点  {T:.2f}s  自碰撞/世界碰撞校验通过")
    return {"position": pos, "velocity": vel, "acceleration": acc, "dt": dt, "duration": T}


def make_hold_axis_metric(mg, free_axis: str = "z", hold_rotation: bool = True):
    """构造「只允许沿 base 某一坐标轴平移」的 PoseCostMetric(直线运动约束)。

    curobo 的 hold_partial_pose 会把 hold_vec_weight 里为 1 的位姿分量在整段
    轨迹上都拉回目标值, 只留下为 0 的那一维自由 —— 于是末端只能沿该维直线移动。

    hold_vec_weight 排布为 [rx, ry, rz, x, y, z], 1=保持, 0=自由。

    Args:
        free_axis: 允许自由变化的平移轴 x/y/z。
        hold_rotation: 是否同时锁死姿态(通常是要的, 直线运动不该带旋转)。

    Note:
        project_to_goal_frame=False 是关键: curobo 默认在「目标坐标系」下算距离,
        那样约束的是工具自身的轴; 置 False 才是在机器人 base(LINK_0) 系下约束,
        与本任务「沿 LINK_0 的 Z 轴直线」的语义一致。
    """
    from curobo.rollout.cost.pose_cost import PoseCostMetric

    ax = {"x": 0, "y": 1, "z": 2}.get(str(free_axis).lower())
    if ax is None:
        raise ValueError(f"free_axis 只支持 x/y/z, 得到 {free_axis}")
    w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] if hold_rotation else [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    w[3 + ax] = 0.0
    return PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=mg.tensor_args.to_device(w),
        project_to_goal_frame=False,
    )


def plan_segment(
    mg, start_q: np.ndarray, target: PoseSpec, pl: Dict[str, Any], pose_metric=None
):
    """规划一段: 从 start_q 出发到 target。返回 MotionGenResult。

    若 target 带有 joint_config(关节空间路点), 走 plan_single_js 直接在关节空间规划,
    这样能保证构型(IK 分支)与前一路点一致 —— 用位姿目标时 IK 可能又选到别的分支。

    Args:
        pose_metric: 可选的 PoseCostMetric, 用于给「这一段」加位姿约束(如直线运动)。
            MotionGen 内部会在本次规划前 apply、结束后自动 reset, 因此逐段传入即可,
            不会污染其他段。只对位姿目标生效(关节空间目标没有位姿 cost)。
    """
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    ta = mg.tensor_args
    start_state = JointState.from_position(
        ta.to_device(np.asarray(start_q, dtype=np.float64).reshape(1, -1)),
        joint_names=mg.joint_names,
    )
    plan_cfg = MotionGenPlanConfig(
        enable_graph=bool(pl.get("enable_graph", False)),
        enable_graph_attempt=int(pl.get("enable_graph_attempt", 3)),
        max_attempts=int(pl["max_attempts"]),
        timeout=float(pl["timeout"]),
        enable_finetune_trajopt=True,
        parallel_finetune=True,
        pose_cost_metric=pose_metric,
    )

    q_goal = getattr(target, "joint_config", None)
    if q_goal is not None:
        goal_state = JointState.from_position(
            ta.to_device(np.asarray(q_goal, dtype=np.float64).reshape(1, -1)),
            joint_names=mg.joint_names,
        )
        res = mg.plan_single_js(start_state, goal_state, plan_cfg.clone())
        if res.success is not None and bool(res.success.item()):
            return res
        # plan_single_js 在本机上对某些关节目标会持续报 GRAPH_FAIL, 即使
        # 关节空间直线路径经校验完全无碰撞。此时退化为自己生成关节空间轨迹,
        # 并逐点做碰撞/限位校验 —— 见 plan_joint_interpolated。
        print(f"    [INFO] plan_single_js 失败({res.status}), 改用关节空间插值 + 逐点校验")
        return None

    goal_pose = Pose(
        position=ta.to_device(target.position.reshape(1, 3)),
        quaternion=ta.to_device(target.quat_wxyz.reshape(1, 4)),
    )
    return mg.plan_single(start_state, goal_pose, plan_cfg.clone())


def compute_fk(mg, q: np.ndarray, link_names: List[str]) -> Dict[str, np.ndarray]:
    """批量 FK。返回 {link: {'pos':[N,3], 'quat_wxyz':[N,4]}} 展平成扁平 dict。"""
    ta = mg.tensor_args
    q_t = ta.to_device(np.asarray(q, dtype=np.float64).reshape(-1, len(mg.joint_names)))
    state = mg.kinematics.get_state(q_t)
    out: Dict[str, np.ndarray] = {}
    all_links = list(mg.kinematics.link_names)
    for ln in link_names:
        if ln not in all_links:
            continue
        i = all_links.index(ln)
        out[f"{ln}/pos"] = state.links_position[:, i, :].detach().cpu().numpy().astype(np.float64)
        out[f"{ln}/quat"] = (
            state.links_quaternion[:, i, :].detach().cpu().numpy().astype(np.float64)
        )
    out["ee/pos"] = state.ee_position.detach().cpu().numpy().astype(np.float64)
    out["ee/quat"] = state.ee_quaternion.detach().cpu().numpy().astype(np.float64)
    return out


def compute_gripper_extent_report(
    mg, positions: np.ndarray, ws: Dict[str, Any], flange_link: str, times: np.ndarray
) -> Dict[str, Any]:
    """报告 flange_link(碰撞球所在 link, 通常是 LINK_6) 全部碰撞球相对 bounds 的越界量。

    纯信息输出, 不影响成功判定。统计使用规划基座坐标系中的真实碰撞球包络；
    越界是否可接受取决于任务、安装方向及硬墙配置，不能预先认定为正常。

    注意: ee_link 现在是 TCP_LINK(纯坐标系, 无碰撞球), 因此这里必须显式传入
    带碰撞球的法兰 link, 不能直接用 ee_link。
    """
    coll_links = list(mg.kinematics.generator_config.collision_link_names)
    if flange_link not in coll_links:
        return {"checked": False, "reason": f"{flange_link} 不在 collision_link_names"}

    idx_map = mg.kinematics.kinematics_config.link_sphere_idx_map.cpu().numpy()
    mask = idx_map == coll_links.index(flange_link)

    sph = (
        mg.kinematics.get_state(mg.tensor_args.to_device(positions))
        .link_spheres_tensor.detach()
        .cpu()
        .numpy()[:, mask, :]
    )  # [N, k, 4]
    c, r = sph[..., :3], sph[..., 3:4]
    lo_pt = (c - r).reshape(-1, 3)
    hi_pt = (c + r).reshape(-1, 3)

    oriented = ws.get("oriented_bounds")
    if oriented:
        frame = parse_rigid_transform_matrix(
            oriented["frame_transform"], "workspace.oriented_bounds.frame_transform")
        centers_in_bounds = (c - frame[:3, 3]) @ frame[:3, :3]
        check_lo = (centers_in_bounds - r).reshape(-1, 3)
        check_hi = (centers_in_bounds + r).reshape(-1, 3)
        b = oriented["reference_bounds"]
    else:
        check_lo, check_hi = lo_pt, hi_pt
        b = ws["bounds"]
    lo = np.array([b["x"][0], b["y"][0], b["z"][0]], dtype=np.float64)
    hi = np.array([b["x"][1], b["y"][1], b["z"][1]], dtype=np.float64)
    under = np.maximum(lo[None, :] - check_lo, 0.0)
    over = np.maximum(check_hi - hi[None, :], 0.0)
    viol_xyz = np.maximum(under, over).reshape(positions.shape[0], -1, 3).max(axis=1)
    viol = viol_xyz.max(axis=1)

    bbox_lo = lo_pt.min(axis=0)
    bbox_hi = hi_pt.max(axis=0)
    n_bad = int((viol > 1e-9).sum())
    i_max = int(np.argmax(viol))
    rep = {
        "checked": True,
        "link": flange_link,
        "n_spheres": int(mask.sum()),
        "bbox_min": bbox_lo.tolist(),
        "bbox_max": bbox_hi.tolist(),
        "n_points_outside_bounds": n_bad,
        "max_violation_mm": float(viol.max()) * 1000.0,
        "max_violation_xyz_mm": (viol_xyz.max(axis=0) * 1000.0).tolist(),
        "violation_frame": "original_task_workspace" if oriented else "robot_base",
        "max_violation_at_t": float(times[i_max]),
        "note": (
            "碰撞球包络相对真实 workspace.bounds 的信息报告；"
            "不参与成功判定，也不等同于 wall.bounds_override 的碰撞检查。"
            "须结合实际安装方向和任务判断越界，不自动认定为正常。"
        ),
    }
    print(f"[CHECK] 夹爪范围({flange_link} 全部 {int(mask.sum())} 个球):")
    print(f"         bbox x=[{bbox_lo[0]:+.3f},{bbox_hi[0]:+.3f}] "
          f"y=[{bbox_lo[1]:+.3f},{bbox_hi[1]:+.3f}] z=[{bbox_lo[2]:+.3f},{bbox_hi[2]:+.3f}]")
    print(f"         相对 bounds 越界: {n_bad}/{positions.shape[0]} 点, "
          f"最大 {viol.max() * 1000:.1f}mm "
          f"(x/y/z = {viol_xyz.max(axis=0)[0] * 1000:.1f}/"
          f"{viol_xyz.max(axis=0)[1] * 1000:.1f}/{viol_xyz.max(axis=0)[2] * 1000:.1f}mm) "
          f"[仅供参考, 不判失败]")
    return rep


def check_ik_branch_consistency(
    robot_dict: Dict[str, Any],
    world,
    pl: Dict[str, Any],
    seq: List[PoseSpec],
    start_q: np.ndarray,
    wall_links: Optional[List[str]],
    warn_deg: float,
) -> Dict[str, Any]:
    """预检各路点的 IK 解, 找出关节空间跳变过大的段。

    同一个笛卡尔位姿通常有多组 IK 解(肘部上/下翻转、腕部翻转等)。若相邻两个路点
    的解落在不同分支, 即使笛卡尔位移只有几厘米, 关节也要转上百度, trajopt 几乎
    必然收敛失败(FINETUNE_TRAJOPT_FAIL)。

    这里用"上一路点的关节角作为 IK 种子"来求解, 使各路点尽量留在同一分支;
    若仍存在大跳变, 说明目标姿态本身要求换分支, 提前给出可操作的提示。
    """
    qs: List[Optional[np.ndarray]] = [np.asarray(start_q, dtype=np.float64)]
    for p in seq[1:]:
        # 关节空间路点直接用给定关节角, 无需 IK(它本就锁定了构型)
        if getattr(p, "joint_config", None) is not None:
            qs.append(np.asarray(p.joint_config, dtype=np.float64))
            continue
        # 先以上一路点为种子(倾向同分支); 若求不出解, 说明该位姿在上一分支里不可达,
        # 退化为无种子求解, 以便继续量化跳变幅度
        q, _ = solve_ik_for_pose(
            robot_dict, world, pl, p, wall_link_names=wall_links, seed_q=qs[-1]
        )
        if q is None:
            q, _ = solve_ik_for_pose(robot_dict, world, pl, p, wall_link_names=wall_links)
            if q is not None:
                print(f"[IK-CHK] 注意: {p.name} 以前一路点为种子无解, "
                      f"但无种子有解 -> 该位姿必须换 IK 分支")
        qs.append(q)

    segs: List[Dict[str, Any]] = []
    print("\n[IK-CHK] 各路点 IK 解(以前一路点为种子, 尽量留在同一 IK 分支):")
    for p, q in zip(seq, qs):
        s = "无解" if q is None else np.round(np.degrees(q), 1).tolist()
        print(f"         {p.name:<12s} q_deg={s}")

    big: List[int] = []
    for i in range(len(seq) - 1):
        if qs[i] is None or qs[i + 1] is None:
            continue
        d_deg = np.degrees(np.abs(qs[i + 1] - qs[i]))
        mx = float(d_deg.max())
        segs.append(
            {
                "index": i,
                "from": seq[i].name,
                "to": seq[i + 1].name,
                "joint_delta_deg": d_deg.round(2).tolist(),
                "max_delta_deg": mx,
                "cartesian_dist_m": float(
                    np.linalg.norm(seq[i + 1].position - seq[i].position)
                ),
            }
        )
        flag = ""
        if mx > warn_deg:
            big.append(i)
            flag = "   <== 关节跳变过大, 可能换了 IK 分支"
        print(
            f"         段{i + 1} {seq[i].name} -> {seq[i + 1].name}: "
            f"笛卡尔 {segs[-1]['cartesian_dist_m'] * 1000:.0f}mm, "
            f"关节最大 {mx:.0f}deg{flag}"
        )

    if big:
        print(f"\n[IK-CHK] 警告: 段 {[i + 1 for i in big]} 的关节跳变超过 {warn_deg:.0f}deg。")
        # 区分两种成因: (a) 该关节转过 180 度以上但受限位无法走 ±360 等价的短边
        #               (b) 真正的 IK 分支切换(多关节同时大幅变化)
        jl = None
        try:
            from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
            from curobo.types.robot import RobotConfig

            km = CudaRobotModel(RobotConfig.from_dict(robot_dict["robot_cfg"]).kinematics)
            lo, hi = [t.cpu().numpy() for t in km.get_joint_limits().position]
            jl = (lo, hi)
        except Exception:  # noqa: BLE001
            pass

        forced: List[str] = []
        if jl is not None:
            lo, hi = jl
            for i in big:
                if qs[i] is None or qs[i + 1] is None:
                    continue
                for j in range(len(lo)):
                    d = qs[i + 1][j] - qs[i][j]
                    if abs(np.degrees(d)) < 180.0:
                        continue
                    # 走反向短边意味着终点 ± 360 度, 检查是否仍在限位内
                    shorter = qs[i + 1][j] - np.sign(d) * 2 * np.pi
                    if not (lo[j] <= shorter <= hi[j]):
                        forced.append(
                            f"{km.joint_names[j]}"
                            f"(转 {abs(np.degrees(d)):.0f}deg, 限位 "
                            f"[{np.degrees(lo[j]):.0f},{np.degrees(hi[j]):.0f}]deg "
                            f"内无 ±360 等价短边 -> 运动学必需)"
                        )
        if forced:
            print("         其中以下关节的大幅转动是运动学必需的, 无法通过调参优化:")
            for s in sorted(set(forced)):
                print(f"           - {s}")
        print("         这通常意味着两端位姿的 IK 解处于不同分支(肘/腕翻转),")
        print("         在很小的笛卡尔位移内要求整臂大翻转, trajopt 很可能失败。")
        print("         可尝试:")
        print("           1) 调整目标姿态 rpy, 使其与起始姿态处于同一 IK 分支")
        print("           2) 在 task.waypoints.extra 里手工插入过渡路点, 分解这次翻转")
        print("           3) task.start_joint_state 指定与目标同分支的起始关节角")
        print("           4) --enable-graph 让图搜索找绕行路径(对大翻转不总是有效)")
    return {"per_segment": segs, "large_jump_segments": big, "warn_deg": warn_deg}


def compute_joint_limit_margin_report(
    mg, positions: np.ndarray, joint_names: List[str], tol_deg: float = 1.0
) -> Dict[str, Any]:
    """统计轨迹上各关节离限位的最小余量, 并数出"贴限位"的点数。

    贴限位意味着该关节已转不动, 优化器只能牺牲末端精度 —— 这是"规划成功但
    位姿误差偏大"的常见原因, 必须显式暴露出来。
    """
    lim = mg.kinematics.get_joint_limits().position
    lo = lim[0].cpu().numpy()
    hi = lim[1].cpu().numpy()
    p = np.asarray(positions, dtype=np.float64)
    margin = np.minimum(p - lo[None, :], hi[None, :] - p)  # [N, dof]
    min_margin = np.degrees(margin.min(axis=0))
    tol = np.radians(tol_deg)
    n_near = (margin < tol).sum(axis=0)

    hit = [(joint_names[i], int(n_near[i]), float(min_margin[i]))
           for i in range(len(joint_names)) if n_near[i] > 0]
    if hit:
        print(f"[LIMIT] 以下关节贴到限位(余量 < {tol_deg}deg), "
              f"会导致末端精度下降:")
        for n, c, mm in hit:
            i = joint_names.index(n)
            print(f"         {n}: {c}/{p.shape[0]} 点, 最小余量 {mm:.2f}deg, "
                  f"限位 [{np.degrees(lo[i]):.1f}, {np.degrees(hi[i]):.1f}]deg")
    else:
        print(f"[LIMIT] 全部关节离限位余量 >= {tol_deg}deg "
              f"(最小 {min_margin.min():.1f}deg @ {joint_names[int(np.argmin(min_margin))]})")

    return {
        "tol_deg": tol_deg,
        "joint_names": list(joint_names),
        "min_margin_deg": min_margin.round(3).tolist(),
        "n_near_limit": [int(v) for v in n_near],
        "total_near_limit_points": int(n_near.sum()),
        "joints_hitting_limit": [n for n, _, _ in hit],
    }


def compute_joint_motion_report(
    positions: np.ndarray, joint_names: List[str], cspace_weight: Optional[List[float]] = None
) -> Dict[str, Any]:
    """统计整条轨迹的关节运动量, 用于评估"关节变化是否够小"。

    - net_deg  : 起点到终点的净变化(可能被往返抵消)
    - total_deg: 沿轨迹累计的总行程(往返也计入), 这是真正衡量"绕不绕"的指标
    """
    p = np.asarray(positions, dtype=np.float64)
    net = np.degrees(np.abs(p[-1] - p[0]))
    total = np.degrees(np.abs(np.diff(p, axis=0)).sum(axis=0))
    weighted = None
    if cspace_weight is not None:
        w = np.asarray(cspace_weight, dtype=np.float64)
        weighted = float((total * w).sum())

    print("[MOTION] 关节运动量统计:")
    print(f"         {'joint':<8s} {'净变化(deg)':>12s} {'总行程(deg)':>12s}")
    for i, n in enumerate(joint_names):
        extra = "   <== 行程远大于净变化(有往返绕行)" if total[i] > net[i] * 2 + 15 else ""
        print(f"         {n:<8s} {net[i]:12.1f} {total[i]:12.1f}{extra}")
    print(f"         {'合计':<8s} {net.sum():12.1f} {total.sum():12.1f}")
    if weighted is not None:
        print(f"         加权路径长度(按 cspace_distance_weight) = {weighted:.1f}")

    return {
        "joint_names": list(joint_names),
        "net_deg": net.round(3).tolist(),
        "total_deg": total.round(3).tolist(),
        "net_sum_deg": float(net.sum()),
        "total_sum_deg": float(total.sum()),
        "max_total_deg": float(total.max()),
        "weighted_path_length_deg": weighted,
        "cspace_distance_weight": (
            [float(v) for v in cspace_weight] if cspace_weight is not None else None
        ),
    }


# ============================== 主流程 ==============================


def plan_one_variant(
    cfg: Dict[str, Any],
    task: Dict[str, Any],
    out_dir: Path,
    ctx: Dict[str, Any],
    variant: Optional[Dict[str, Any]] = None,
) -> int:
    """对单个目标(变体)完成一次完整规划并落盘。

    Args:
        cfg:     完整配置(用于写入 meta 快照)
        task:    已套用变体的 task 配置
        out_dir: 本次输出目录
        ctx:     共享资源 {mg, robot_dict, world, wall_links, wall_report}
                 —— MotionGen 构建耗时, 多变体之间复用同一实例
        variant: 变体描述(None 表示单目标模式)

    Returns:
        0 成功, 非 0 失败
    """
    rb, ws, pl = cfg["robot"], cfg["workspace"], cfg["planner"]
    mg = ctx["mg"]
    robot_dict = ctx["robot_dict"]
    world = ctx["world"]
    wall_links = ctx["wall_links"]
    wall_report = ctx["wall_report"]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] {out_dir}")

    # ---------- 位姿序列 ----------
    seq = build_pose_sequence(task)
    print(f"\n[SEQ] {len(seq)} 个位姿, {len(seq) - 1} 段:")
    for i, p in enumerate(seq):
        print(f"  [{i}] {p}")

    # 位姿本身先做工作空间检查
    pos_all = np.stack([p.position for p in seq], axis=0)
    inside, viol = check_in_bounds(pos_all, ws)
    for i, (p, ok, v) in enumerate(zip(seq, inside, viol)):
        if not ok:
            print(f"  [WARN] 位姿 [{i}] {p.name} 超出工作空间 {v * 1000:.1f}mm, 该段很可能规划失败")

    joint_names = list(mg.joint_names)
    dof = len(joint_names)

    # ---------- 起始关节角 ----------
    print()
    if task.get("start_joint_state"):
        start_q = np.asarray(task["start_joint_state"], dtype=np.float64).reshape(-1)
        if start_q.size != dof:
            print(f"[FAIL] start_joint_state 长度 {start_q.size} != dof {dof}")
            return 2
        print(f"[IK] 使用给定起始关节角: {start_q.round(4).tolist()}")
    else:
        print("[IK] 从 start 位姿求解起始关节角 ...")
        start_q, ik_res = solve_ik_for_pose(
            robot_dict, world, pl, seq[0], wall_link_names=wall_links
        )
        if start_q is None:
            print("[FAIL] 起始位姿 IK 无解(可能超出工作空间/自碰撞/与墙碰撞)")
            print(f"       position_error={ik_res.position_error}")
            return 3
        print(f"[IK] start_q = {start_q.round(4).tolist()}")

    # 校验起始关节角对应的实际位姿
    fk0 = compute_fk(mg, start_q.reshape(1, -1), [rb["ee_link"]])
    p_err = float(np.linalg.norm(fk0["ee/pos"][0] - seq[0].position))
    r_err = quat_angle_deg(fk0["ee/quat"][0], seq[0].quat_wxyz)
    print(f"[IK] start FK 校验: 位置误差={p_err * 1000:.2f}mm 姿态误差={r_err:.2f}deg")

    # ---------- IK 分支一致性预检 ----------
    ik_check: Dict[str, Any] = {"checked": False}
    if pl.get("check_ik_branch", True):
        ik_check = check_ik_branch_consistency(
            robot_dict, world, pl, seq, start_q, wall_links,
            float(pl.get("ik_branch_warn_deg", 90.0)),
        )
        ik_check["checked"] = True

    # ---------- 分段规划 ----------
    print(f"\n{'=' * 70}\n[PLAN] 开始分段规划\n{'=' * 70}")
    seg_reports: List[Dict[str, Any]] = []
    all_pos: List[np.ndarray] = []
    all_vel: List[np.ndarray] = []
    all_acc: List[np.ndarray] = []
    all_t: List[np.ndarray] = []
    seg_boundaries: List[int] = []
    t_offset = 0.0
    cur_q = start_q.copy()
    interp_dt = float(pl["interpolation_dt"])

    for si in range(len(seq) - 1):
        src, dst = seq[si], seq[si + 1]
        is_js = getattr(dst, "joint_config", None) is not None
        print(f"\n--- 段 {si + 1}/{len(seq) - 1}: {src.name} -> {dst.name} ---")
        if is_js:
            print(f"    关节目标(plan_single_js) q_deg="
                  f"{np.degrees(dst.joint_config).round(2).tolist()}")
        else:
            print(f"    目标 xyz={dst.position.round(4).tolist()} "
                  f"rpy={dst.rpy_deg.round(2).tolist()}")
        t0 = time.time()
        res = plan_segment(mg, cur_q, dst, pl)
        dt_solve = time.time() - t0

        # res is None: 关节目标且 plan_single_js 失败 -> 用关节空间插值兜底
        js_fallback = None
        if res is None:
            js_fallback = plan_joint_interpolated(mg, cur_q, dst.joint_config, pl)
            if js_fallback is None:
                print(f"    [FAIL] 关节空间插值兜底也不可行  solve={dt_solve:.2f}s")
                dump_json(
                    {
                        "failed_segment": si,
                        "from": src.to_dict(),
                        "to": dst.to_dict(),
                        "status": "JS_FALLBACK_FAIL",
                        "ik_branch_check": ik_check,
                        "config": cfg,
                    },
                    out_dir / "plan_failed.json",
                )
                return 4
            dt_solve = time.time() - t0
            ok = True
        else:
            ok = res.success is not None and bool(res.success.item())

        if not ok:
            print(f"    [FAIL] status={res.status}  solve={dt_solve:.2f}s")
            print(f"           position_error={res.position_error} rotation_error={res.rotation_error}")
            print(f"           valid_query={res.valid_query}")
            jump = [s for s in ik_check.get("per_segment", []) if s["index"] == si]
            if jump and jump[0]["max_delta_deg"] > ik_check.get("warn_deg", 90.0):
                print(f"    原因很可能是 IK 分支跳变: 该段笛卡尔仅 "
                      f"{jump[0]['cartesian_dist_m'] * 1000:.0f}mm 但关节需转 "
                      f"{jump[0]['max_delta_deg']:.0f}deg。")
                print("    建议: 调整目标 rpy 使其与起始同分支, 或在 task.waypoints.extra "
                      "插入过渡路点分解翻转。")
            else:
                print("    提示: 试 --enable-graph 启用绕行搜索, 或放宽 --bounds-*, "
                      "或减小 lift 高度")
            dump_json(
                {
                    "failed_segment": si,
                    "from": src.to_dict(),
                    "to": dst.to_dict(),
                    "status": str(res.status),
                    "valid_query": bool(res.valid_query),
                    "ik_branch_check": ik_check,
                    "config": cfg,
                },
                out_dir / "plan_failed.json",
            )
            return 4

        if js_fallback is not None:
            q = js_fallback["position"]
            v = js_fallback["velocity"]
            a = js_fallback["acceleration"]
            seg_dt = float(js_fallback["dt"])
            used_graph = False
            attempts = 1
        else:
            traj = res.get_interpolated_plan()
            q = traj.position.detach().cpu().numpy().astype(np.float64)
            v = (
                traj.velocity.detach().cpu().numpy().astype(np.float64)
                if traj.velocity is not None
                else np.zeros_like(q)
            )
            a = (
                traj.acceleration.detach().cpu().numpy().astype(np.float64)
                if traj.acceleration is not None
                else np.zeros_like(q)
            )
            seg_dt = float(res.interpolation_dt)
            used_graph = bool(res.used_graph)
            attempts = int(res.attempts)
        n = q.shape[0]
        # 段间去掉重复的首点(等于上一段末点), 保证时间与关节都连续
        s = 1 if si > 0 else 0
        t_local = t_offset + np.arange(s, n, dtype=np.float64) * seg_dt

        all_pos.append(q[s:])
        all_vel.append(v[s:])
        all_acc.append(a[s:])
        all_t.append(t_local)
        t_offset = float(t_local[-1]) + seg_dt
        seg_boundaries.append(int(sum(x.shape[0] for x in all_pos)))
        cur_q = q[-1].copy()
        interp_dt = seg_dt

        fk_end = compute_fk(mg, cur_q.reshape(1, -1), [rb["ee_link"]])
        print(
            f"    [OK] {n} 点  {n * seg_dt:.2f}s  solve={dt_solve:.2f}s  "
            f"attempts={attempts} used_graph={used_graph}"
            f"{' (关节空间插值)' if js_fallback is not None else ''}"
        )
        if is_js:
            # 关节目标: 用关节误差衡量, 位姿误差无意义(dst.position 只是显示用)
            pe = 0.0
            re = 0.0
            qe = float(np.degrees(np.abs(cur_q - dst.joint_config)).max())
            print(f"    [OK] 终点关节误差: 最大={qe:.3f}deg")
        else:
            pe = float(np.linalg.norm(fk_end["ee/pos"][0] - dst.position))
            re = quat_angle_deg(fk_end["ee/quat"][0], dst.quat_wxyz)
            warn = ""
            # position_threshold 是收敛判据; 实际误差接近它时说明该目标较难达到,
            # 明确提示出来, 避免" 成功但精度不佳"被忽略
            if pe > 0.5 * float(pl["position_threshold"]) or re > 0.5:
                warn = "   <== 误差偏大, 该目标位姿较难精确到达"
            print(f"    [OK] 终点误差: 位置={pe * 1000:.2f}mm 姿态={re:.2f}deg{warn}")
        seg_reports.append(
            {
                "index": si,
                "from": src.to_dict(),
                "to": dst.to_dict(),
                "mode": "joint_interpolated" if js_fallback is not None
                else ("joint_space" if is_js else "pose"),
                "n_points": int(n),
                "duration_s": float(n * seg_dt),
                "solve_time_s": float(dt_solve),
                "attempts": attempts,
                "used_graph": used_graph,
                "position_error_mm": pe * 1000.0,
                "rotation_error_deg": re,
            }
        )

    positions = np.concatenate(all_pos, axis=0)
    velocities = np.concatenate(all_vel, axis=0)
    accelerations = np.concatenate(all_acc, axis=0)
    times = np.concatenate(all_t, axis=0)
    n_total = positions.shape[0]
    print(f"\n[TRAJ] 总计 {n_total} 点, 时长 {times[-1]:.2f}s, dt={interp_dt}")

    # ---------- 全轨迹 FK ----------
    # ee_link = TCP_LINK(纯坐标系); 另外把 gripper_link 与法兰 link 也算出来供 tf/报告用
    flange_link = rb.get("flange_link") or "LINK_6"
    link_list = [rb["ee_link"]]
    for ln in ("gripper_link", flange_link):
        if ln and ln not in link_list:
            link_list.append(ln)
    fk = compute_fk(mg, positions, link_list)
    ee_pos = fk["ee/pos"]
    ee_quat = fk["ee/quat"]

    # ---------- 工作空间软校验 ----------
    viol_report: Dict[str, Any] = {"checked": False}
    if ws.get("check_after_plan", True):
        margin = float(ws.get("check_margin", 0.0))
        inside, viol = check_in_bounds(ee_pos, ws, margin=margin)
        n_bad = int((~inside).sum())
        max_v = float(viol.max()) if viol.size else 0.0
        viol_report = {
            "checked": True,
            "margin_m": margin,
            "n_points": n_total,
            "n_violation": n_bad,
            "max_violation_mm": max_v * 1000.0,
        }
        if n_bad == 0:
            print(f"[CHECK] 工作空间: 全部 {n_total} 点合法 (margin={margin * 1000:.1f}mm)")
        else:
            print(
                f"[CHECK] 工作空间: {n_bad}/{n_total} 点越界, 最大 {max_v * 1000:.1f}mm "
                f"(margin={margin * 1000:.1f}mm)"
            )
            idx = np.where(~inside)[0]
            for i in idx[:5]:
                print(f"         t={times[i]:.2f}s xyz={ee_pos[i].round(4).tolist()} "
                      f"越界={viol[i] * 1000:.1f}mm")
            if ws.get("fail_on_violation", False):
                print("[FAIL] fail_on_violation=true, 终止")
                return 5

    # ---------- 关节运动量统计 ----------
    motion_report = compute_joint_motion_report(
        positions,
        joint_names,
        list(mg.kinematics.generator_config.cspace.cspace_distance_weight.cpu().numpy())
        if hasattr(mg.kinematics.generator_config.cspace, "cspace_distance_weight")
        else rb.get("cspace_distance_weight"),
    )

    # ---------- 关节限位余量 ----------
    limit_report = compute_joint_limit_margin_report(mg, positions, joint_names)

    # ---------- 夹爪范围报告 ----------
    gripper_report: Dict[str, Any] = {"checked": False}
    if ws.get("report_gripper_extent", False):
        gripper_report = compute_gripper_extent_report(mg, positions, ws, flange_link, times)

    # ---------- 自碰撞复核 ----------
    # 用 SelfCollisionCost 对最终轨迹逐点独立复核, 与优化器内部的判定解耦
    self_coll_report: Dict[str, Any] = {"checked": False}
    if pl.get("self_collision_check", True):
        try:
            sph = mg.kinematics.get_state(
                mg.tensor_args.to_device(positions)
            ).link_spheres_tensor.unsqueeze(1)  # [N, 1, n_sph, 4]
            cost_fn = None
            for rollout in mg.get_all_rollout_instances():
                for attr in ("robot_self_collision_constraint", "robot_self_collision_cost"):
                    c = getattr(rollout, attr, None)
                    if c is not None and getattr(c, "enabled", True):
                        cost_fn = c
                        break
                if cost_fn is not None:
                    break
            if cost_fn is None:
                raise AttributeError("找不到可用的 self collision cost 实例")
            d = cost_fn.forward(sph).reshape(-1).detach().cpu().numpy()
            n_bad = int((d > 0).sum())
            self_coll_report = {
                "checked": True,
                "n_points": n_total,
                "n_collision": n_bad,
                "max_cost": float(d.max()),
            }
            if n_bad == 0:
                print(f"[CHECK] 自碰撞: 全部 {n_total} 点无碰撞")
            else:
                print(f"[CHECK] 自碰撞: {n_bad}/{n_total} 点碰撞, 最大代价 {d.max():.4f}")
        except Exception as e:  # noqa: BLE001  复核失败不应中断主流程
            print(f"[CHECK] 自碰撞复核跳过: {e}")
            self_coll_report = {"checked": False, "reason": str(e)}

    # ---------- 保存 ----------
    meta: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "robot": {
            "robot_yml": rb["robot_yml"],
            "base_link": rb["base_link"],
            "ee_link": rb["ee_link"],
            "flange_link": flange_link,
            "urdf": rb["urdf"],
            "urdf_abs": str(resolve_repo_path(rb["urdf"])),
            "joint_names": joint_names,
        },
        "rotation_convention": "rpy_deg, fixed-axis XYZ extrinsic (ROS/URDF), R = Rz*Ry*Rx",
        "quaternion_order": {"npz_ee_quats": "wxyz", "ros": "xyzw"},
        "variant": variant,
        "pose_sequence": [p.to_dict() for p in seq],
        "workspace": ws,
        "planner": pl,
        "segments": seg_reports,
        "n_points": int(n_total),
        "interpolation_dt": interp_dt,
        "total_duration_s": float(times[-1]),
        "segment_boundaries": seg_boundaries,
        "workspace_check": viol_report,
        "joint_motion": motion_report,
        "joint_limit_margin": limit_report,
        "gripper_extent_check": gripper_report,
        "ik_branch_check": ik_check,
        "self_collision_check": self_coll_report,
        "wall_link_restriction": wall_report,
        "wall_cuboids": build_workspace_wall_cuboids(ws)
        if (ws.get("wall") or {}).get("enable", False)
        else [],
        "config": cfg,
    }
    npz_path, meta_path = save_trajectory(
        out_dir,
        joint_names,
        positions,
        velocities,
        accelerations,
        times,
        ee_pos,
        ee_quat,
        meta,
    )
    # 附带一份人类可读的 csv, 便于快速核对
    csv_path = out_dir / "trajectory.csv"
    hdr = ["t"] + [f"q_{n}" for n in joint_names] + ["ee_x", "ee_y", "ee_z", "ee_r", "ee_p", "ee_yw"]
    rpy_all = np.stack([quat_wxyz_to_rpy_deg(q) for q in ee_quat], axis=0)
    with open(csv_path, "w") as f:
        f.write(",".join(hdr) + "\n")
        for i in range(n_total):
            row = [f"{times[i]:.4f}"]
            row += [f"{v:.6f}" for v in positions[i]]
            row += [f"{v:.6f}" for v in ee_pos[i]]
            row += [f"{v:.3f}" for v in rpy_all[i]]
            f.write(",".join(row) + "\n")

    print(f"\n[SAVE] {npz_path}")
    print(f"[SAVE] {meta_path}")
    print(f"[SAVE] {csv_path}")
    return 0


# ============================== 批量调度 ==============================


def main() -> int:
    args = build_argparser().parse_args()
    cfg = apply_cli_overrides(load_task_config(args.config), args)
    rb, task, ws, pl, outc = (
        cfg["robot"], cfg["task"], cfg["workspace"], cfg["planner"], cfg["output"]
    )

    # ---------- 展开 start x goal 组合 ----------
    variants = build_variant_combos(task)
    if args.variant is not None:
        sel = {str(x) for x in args.variant}
        variants = [
            v for v in variants
            if str(v["index"]) in sel
            or v["name"] in sel
            or v["start"]["name"] in sel
            or v["goal"]["name"] in sel
        ]
        if not variants:
            print(f"[FAIL] --variant {args.variant} 没匹配到任何组合")
            return 2
    multi = len(variants) > 1 or (variants and variants[0]["name"] != "base")

    # ---------- 根输出目录 ----------
    root_dir = Path(outc["dir"])
    if not root_dir.is_absolute():
        root_dir = Path(__file__).resolve().parents[1] / root_dir
    if outc.get("add_timestamp", True):
        root_dir = root_dir / time.strftime("%Y%m%d_%H%M%S")

    if multi:
        n_s = variants[0]["n_start"]
        n_g = variants[0]["n_goal"]
        print(f"\n{'#' * 100}")
        print(f"# 姿态变体组合: start {n_s} 个 x goal {n_g} 个 = {len(variants)} 次独立规划")
        print(f"# base start rpy={task['start']['rpy_deg']}  "
              f"base goal rpy={task['goal']['rpy_deg']}")
        print(f"# mode=right 绕工具自身轴; left 绕 base 固定轴")
        print(f"{'#' * 100}")
        print(f"  {'#':<3} {'组合':<38} {'start delta':<16} {'start rpy':<24} "
              f"{'goal delta':<16} {'goal rpy'}")
        for v in variants:
            sv, gv = v["start"], v["goal"]
            print(f"  {v['index']:<3} {v['name']:<38} "
                  f"{str(sv['delta_rpy_deg']):<16} "
                  f"{str(np.round(sv['rpy_deg'], 1).tolist()):<24} "
                  f"{str(gv['delta_rpy_deg']):<16} "
                  f"{np.round(gv['rpy_deg'], 1).tolist()}")

    # ---------- 共享资源: MotionGen 只构建一次 ----------
    print()
    world = make_world_config(ws)
    robot_dict = load_robot_cfg_dict(rb)
    mg = make_motion_gen(robot_dict, world, pl)

    wall_cfg = ws.get("wall") or {}
    wall_report: Dict[str, Any] = {"applied": False}
    wall_links: List[str] = []
    if wall_cfg.get("enable", False) and wall_cfg.get("collision_link_names"):
        wall_links = list(wall_cfg["collision_link_names"])
        # 方案 A (mimic 耦合双臂): 第二臂的同名 link 也要受墙约束,
        # 否则第二臂不受料台约束
        dual_prefix = rb.get("dual_arm_prefix") or ""
        if dual_prefix:
            wall_links = wall_links + [dual_prefix + n for n in wall_links]
        wall_report = restrict_world_collision_to_links(mg, wall_links)
        wall_report["applied"] = True
    elif wall_cfg.get("enable", False):
        print("[WALL] collision_link_names 为空, 墙约束全部 link")

    jl = mg.kinematics.get_joint_limits().position
    print(f"[ROBOT] joints={list(mg.joint_names)}")
    print(f"[ROBOT] limit lower={jl[0].cpu().numpy().round(4).tolist()}")
    print(f"[ROBOT] limit upper={jl[1].cpu().numpy().round(4).tolist()}")

    ctx = {
        "mg": mg,
        "robot_dict": robot_dict,
        "world": world,
        "wall_links": wall_links,
        "wall_report": wall_report,
    }

    # ---------- 逐变体规划 ----------
    summary: List[Dict[str, Any]] = []
    for n_done, v in enumerate(variants):
        if multi:
            sv, gv = v["start"], v["goal"]
            print(f"\n\n{'#' * 100}")
            print(f"# 组合 {n_done + 1}/{len(variants)}: {v['name']}")
            print(f"#   start delta={sv['delta_rpy_deg']} -> rpy="
                  f"{np.round(sv['rpy_deg'], 2).tolist()}")
            print(f"#   goal  delta={gv['delta_rpy_deg']} -> rpy="
                  f"{np.round(gv['rpy_deg'], 2).tolist()}")
            print(f"{'#' * 100}")
            sub_dir = root_dir / v["name"]
            task_v = apply_variant_combo(task, v)
        else:
            sub_dir = root_dir
            task_v = apply_variant_combo(task, v)

        rc = plan_one_variant(cfg, task_v, sub_dir, ctx, v if multi else None)
        entry: Dict[str, Any] = {
            "index": v["index"],
            "name": v["name"],
            "start_delta_rpy_deg": v["start"]["delta_rpy_deg"],
            "start_position": v["start"]["position"],
            "start_rpy_deg": v["start"]["rpy_deg"],
            "goal_delta_rpy_deg": v["goal"]["delta_rpy_deg"],
            "goal_position": v["goal"]["position"],
            "goal_rpy_deg": v["goal"]["rpy_deg"],
            "mode": v["goal"]["mode"] if v["goal"]["mode"] != "none" else v["start"]["mode"],
            "success": rc == 0,
            "dir": str(sub_dir),
        }
        # 成功则把关键指标读回来做汇总
        meta_f = sub_dir / "trajectory_meta.json"
        if rc == 0 and meta_f.exists():
            with open(meta_f, "r") as f:
                m = json.load(f)
            entry.update(
                {
                    "n_points": m.get("n_points"),
                    "duration_s": m.get("total_duration_s"),
                    "joint_total_deg": (m.get("joint_motion") or {}).get("total_sum_deg"),
                    "joint_net_deg": (m.get("joint_motion") or {}).get("net_sum_deg"),
                    "ws_violation": (m.get("workspace_check") or {}).get("n_violation"),
                    "ws_max_violation_mm": (m.get("workspace_check") or {}).get(
                        "max_violation_mm"
                    ),
                    "self_collision": (m.get("self_collision_check") or {}).get("n_collision"),
                    "j6_near_limit": (m.get("joint_limit_margin") or {}).get(
                        "total_near_limit_points"
                    ),
                    "limit_joints": (m.get("joint_limit_margin") or {}).get(
                        "joints_hitting_limit"
                    ),
                    "max_position_error_mm": max(
                        [s.get("position_error_mm", 0.0) for s in m.get("segments") or [0]]
                        or [0.0]
                    ),
                }
            )
        elif rc != 0:
            fail_f = sub_dir / "plan_failed.json"
            if fail_f.exists():
                with open(fail_f, "r") as f:
                    fj = json.load(f)
                entry["fail_segment"] = f"{fj['from']['name']} -> {fj['to']['name']}"
                entry["fail_status"] = fj.get("status")
        summary.append(entry)

    if not multi:
        d = summary[0]["dir"]
        print(f"\n下一步在 ROS 环境播放:\n  ./run_rviz.sh --traj {d}")
        return 0 if summary[0]["success"] else 4

    # ---------- 汇总 ----------
    def _d(v):
        return f"[{v[0]:g},{v[1]:g},{v[2]:g}]"

    W = 118
    print(f"\n\n{'=' * W}")
    print("[SUMMARY] start x goal 姿态组合对比")
    print(f"{'=' * W}")
    print(f"{'#':<3} {'start':<14} {'goal':<14} {'结果':<6} {'点数':>6} "
          f"{'时长s':>7} {'关节行程':>9} {'往返':>6} {'误差mm':>8} {'越界':>5} {'自碰':>5}  备注")
    print("-" * W)
    for e in summary:
        sd = _d(e["start_delta_rpy_deg"])
        gd = _d(e["goal_delta_rpy_deg"])
        if not e["success"]:
            print(f"{e['index']:<3} {sd:<14} {gd:<14} {'FAIL':<6} "
                  f"{'':>6} {'':>7} {'':>9} {'':>6} {'':>8} {'':>5} {'':>5}  "
                  f"{e.get('fail_status', '')} @ {e.get('fail_segment', '')}")
            continue
        tot = e.get("joint_total_deg") or 0.0
        net = e.get("joint_net_deg") or 1.0
        ratio = tot / max(net, 1e-9)
        note = []
        if e.get("max_position_error_mm", 0) > 0.5:
            note.append("误差偏大")
        if ratio > 1.15:
            note.append("有往返绕行")
        if e.get("j6_near_limit"):
            lj = ",".join(e.get("limit_joints") or []) or "关节"
            note.append(f"{lj}贴限位x{e['j6_near_limit']}")
        print(f"{e['index']:<3} {sd:<14} {gd:<14} {'OK':<6} "
              f"{e.get('n_points', 0):>6} {e.get('duration_s', 0):>7.2f} "
              f"{tot:>9.1f} {ratio:>6.2f} "
              f"{e.get('max_position_error_mm', 0):>8.3f} "
              f"{e.get('ws_violation', 0):>5} {e.get('self_collision', 0):>5}  "
              f"{', '.join(note)}")
    print("-" * W)
    n_ok = sum(1 for e in summary if e["success"])
    print(f"成功 {n_ok}/{len(summary)}   "
          f"start/goal 列为各自的 rpy 增量(度); 关节行程=沿轨迹累计度数; "
          f"往返=行程/净变化(越接近1越无绕行)")

    ok_list = [e for e in summary if e["success"]]
    if ok_list:
        best_j = min(ok_list, key=lambda e: e.get("joint_total_deg") or 9e9)
        best_e = min(ok_list, key=lambda e: e.get("max_position_error_mm") or 9e9)
        print(f"\n  关节行程最小: #{best_j['index']} {best_j['name']} "
              f"({best_j.get('joint_total_deg', 0):.1f}deg)")
        print(f"  精度最高    : #{best_e['index']} {best_e['name']} "
              f"({best_e.get('max_position_error_mm', 0):.3f}mm)")

    dump_json(
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_start": {
                "position": task["start"]["position"],
                "rpy_deg": task["start"]["rpy_deg"],
            },
            "base_goal": {
                "position": task["goal"]["position"],
                "rpy_deg": task["goal"]["rpy_deg"],
            },
            "mode": summary[0]["mode"] if summary else None,
            "n_start_variants": variants[0]["n_start"] if variants else 0,
            "n_goal_variants": variants[0]["n_goal"] if variants else 0,
            "n_combos": len(summary),
            "n_success": n_ok,
            "combos": summary,
        },
        root_dir / "variants_summary.json",
    )
    print(f"\n[SAVE] {root_dir / 'variants_summary.json'}")
    print(f"\n下一步在 ROS 环境逐个播放(每个变体一个子目录):")
    for e in summary:
        if e["success"]:
            print(f"  ./run_rviz.sh --traj {e['dir']}")
    return 0 if n_ok > 0 else 4


if __name__ == "__main__":
    sys.exit(main())
