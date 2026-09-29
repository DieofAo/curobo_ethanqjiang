#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XTrainer 多物料上下料（pick & place）循环轨迹规划

场景:
  第一阶段: 在给定 x/y 范围内布一个物料阵列（横向 x 竖向），逐个抓取
  第二阶段: 全部搬到同一个固定放置点
  每次放置后返回下一个上料位, 最终拼成「一条」完整长轨迹

轨迹结构(每个物料一轮):
  home -> g_lift -> g -> g_lift -> p_lift -> p -> p_lift -> (下一轮 g_lift) -> ...
  最后停在最后一次放置点(不回 home)

姿态搜索:
  抓取姿态 = base_rpy.grasp 右乘 [+n, 0, 0]
  放置姿态 = base_rpy.place 右乘 [-n, 0, 0]   (与抓取相反)
  n 从 min_deg 起按 step_deg 递增, 取第一个满足判据的角度。
  优先复用上一个物料成功的角度。

  两侧默认「独立」搜索(笛卡尔积)。置 angle_search.couple_place_to_grasp=true
  可改为「耦合」: 只搜抓取角, 放置角由抓取角直接决定, 候选数从 |g|x|p| 降到 |g|。
  耦合关系由 grasp.min_deg 的符号整轮判定:
      grasp.min_deg <  0  ->  p =  g  (放置角等于抓取角)
      grasp.min_deg >= 0  ->  p = -g  (放置角与抓取角相反)

第二阶段搜索(angle_search.stage2.enable=true, 默认关):
  在第一阶段结果之上再转一次。stage2.frame 决定绕哪根轴:

    frame='base'(默认): 绕 base(LINK_0) 的 stage2.axis 轴转 -> R2 = R(z,a2) @ R1
      等价于「取 R1^-1 的第三列作为工具系转轴再右乘」(R1 @ R1^T z = z),
      两种写法数值一致(实测 ~1e-16)。TCP 指向会被摆动(绕竖直轴摆方位角)。

    frame='tool': 绕「第一阶段转完之后」的工具局部轴转 -> R2 = R1 @ R(z,a2),
      即绕 R1 旋转矩阵的第三列。TCP 指向不变, 只绕自身自转。

  注意 base_rpy.grasp=[-180,0,0] 时工具 Z 指向 LINK_0 的 -Z, 因此两种 frame 下
  同一个 a2 的转动方向相反, 切 frame 时角度范围的符号要重新确认。

  流程(逐物料):
      1) 先跑完整的第一阶段搜索(含 trajopt), 得到该物料自己的 a1*
      2) 固定 a1*, 再搜 a2 (grasp/place 两侧不耦合, 笛卡尔积)
      3) a2 候选首项恒为 (0,0); 若全部 a2 都不可行, 回退到第一阶段轨迹
  因此开启第二阶段不会让原本可行的物料变失败, 只是耗时增加。

判据: 每一段内 J1~J4 中最大的关节变化 <= max_joint_delta_deg (默认 120°)

直线段: linear_move.enable=true 时, 对 kinds 里的路点类型(grasp/place)把
  「抬升点 -> 目标点」约束为沿修正前任务 LINK_0 的 Z 轴平移。该方向会随
  link0_target_transform 映射到规划 root 的笛卡尔轴(姿态同时锁死), 规划后
  用 FK 实测横向偏移与姿态偏差复核。默认 grasp 与 place 两端都垂直插拔。

失败处理: 立即停止, 保存已成功部分的轨迹 + 失败物料序号与原因。

必须在 conda curobo 环境下运行:
  ./run_pick_place.sh
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_trajectory import (  # noqa: E402
    compute_fk,
    compute_gripper_extent_report,
    compute_joint_limit_margin_report,
    compute_joint_motion_report,
    load_robot_cfg_dict,
    make_hold_axis_metric,
    make_motion_gen,
    make_world_config,
    plan_segment,
    restrict_world_collision_to_links,
    solve_ik_for_pose,
)
from xtrainer_common import (  # noqa: E402
    PoseSpec,
    check_in_bounds,
    compose_left,
    compose_right,
    deep_update,
    dump_json,
    matrix_to_quat_wxyz,
    normalize_link0_target_transform_config_layer,
    parse_rigid_transform_matrix,
    quat_angle_deg,
    quat_wxyz_to_matrix,
    quat_wxyz_to_rpy_deg,
    resolve_repo_path,
    rpy_deg_to_quat_wxyz,
    save_trajectory,
    tool_z_axis,
)

TASK_ROOT = Path(__file__).resolve().parents[1]


# ============================== 配置 ==============================


def load_pick_place_config(path: Optional[str] = None) -> Dict[str, Any]:
    """加载 pick&place 配置, 自定义文件会与默认配置深层合并(可只写差异项)。"""
    import yaml

    default_path = TASK_ROOT / "config" / "pick_place_default.yaml"
    with open(default_path, "r") as f:
        cfg = normalize_link0_target_transform_config_layer(
            yaml.safe_load(f) or {}, str(default_path)
        )
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            for cand in (Path.cwd() / p, TASK_ROOT / p, TASK_ROOT / "config" / p):
                if cand.exists():
                    p = cand
                    break
        if not p.exists():
            raise FileNotFoundError(f"config not found: {path}")
        if p.resolve() != default_path.resolve():
            with open(p, "r") as f:
                user_layer = normalize_link0_target_transform_config_layer(
                    yaml.safe_load(f) or {}, str(p)
                )
                cfg = deep_update(cfg, user_layer)
    return cfg


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="XTrainer 多物料上下料循环轨迹规划",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=str, default=None,
                    help="任务 yaml, 默认 config/pick_place_default.yaml")

    g = ap.add_argument_group("抓取阵列")
    g.add_argument("--rows", type=int, default=None, help="沿 x 的点数(竖向)")
    g.add_argument("--cols", type=int, default=None, help="沿 y 的点数(横向)")
    g.add_argument("--x-range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    g.add_argument("--y-range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    g.add_argument("--grasp-z", type=float, default=None, help="抓取点高度 (m)")
    g.add_argument("--order", type=str, default=None,
                   choices=["row_major", "snake", "ring"],
                   help="遍历顺序. ring=沿外围闭环走一圈, 仅配合 --perimeter 使用")
    g.add_argument("--perimeter", dest="perimeter", action="store_true", default=None,
                   help="只搜索框选区域「最外围一圈」, 丢弃内部点。"
                        "建议同时用 --on-fail skip, 到不了的点位直接跳过")
    g.add_argument("--no-perimeter", dest="perimeter", action="store_false",
                   help="扫描完整网格(默认)")
    g.add_argument("--max-items", type=int, default=None,
                   help="只处理前 N 个物料(调试用)")

    g = ap.add_argument_group("放置点")
    g.add_argument("--place-position", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"))
    g.add_argument("--place-tool-z-rotation-deg", type=float, default=None,
                   help="放置候选及第二阶段完成后，额外绕其工具自身 Z 轴固定旋转(度)，默认0")

    g = ap.add_argument_group("姿态角搜索(抓取/放置)")
    g.add_argument("--grasp-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="抓取侧角度范围(度)")
    g.add_argument("--grasp-step", type=float, default=None, help="抓取侧步长(度)")
    g.add_argument("--grasp-axis", type=str, default=None, choices=["x", "y", "z"])
    g.add_argument("--place-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="放置侧角度范围(度). 耦合模式下无效")
    g.add_argument("--place-step", type=float, default=None,
                   help="放置侧步长(度). 耦合模式下无效")
    g.add_argument("--place-axis", type=str, default=None, choices=["x", "y", "z"])
    g.add_argument("--couple-place", dest="couple_place", action="store_true", default=None,
                   help="只搜索抓取角, 放置角强制等于抓取角(候选从 |g|x|p| 降到 |g|)")
    g.add_argument("--no-couple-place", dest="couple_place", action="store_false",
                   help="抓取角与放置角独立搜索(笛卡尔积)")
    g.add_argument("--search-order", type=str, default=None, choices=["abs", "asc", "desc"],
                   help="单侧候选角顺序: abs=绝对值递增, asc=数值递增, desc=数值递减")
    g.add_argument("--search-strategy", type=str, default=None,
                   choices=["abs_sum", "coarse_to_fine"],
                   help="遍历策略: abs_sum=按|g|+|p|递增; "
                        "coarse_to_fine=分轮粗扫, 保证 grasp 角早期全覆盖. "
                        "耦合模式下无效")
    g.add_argument("--coarse-place", type=int, default=None,
                   help="coarse_to_fine 每轮给每个 grasp 角配多少个 place 角. 耦合模式下无效")
    g.add_argument("--stage2", dest="stage2", action="store_true", default=None,
                   help="开启第二阶段: 在第一阶段结果之上, 再绕「第一阶段转完后的"
                        "局部轴」搜索(默认绕局部 Z, 即该姿态旋转矩阵的第三列方向)")
    g.add_argument("--no-stage2", dest="stage2", action="store_false",
                   help="关闭第二阶段, 只做第一阶段搜索")
    g.add_argument("--stage2-axis", type=str, default=None, choices=["x", "y", "z"],
                   help="第二阶段的转轴")
    g.add_argument("--stage2-frame", type=str, default=None, choices=["base", "tool"],
                   help="第二阶段转轴所在坐标系: base=绕 LINK_0 的轴(TCP 指向被摆动, "
                        "默认); tool=绕第一阶段转完后的工具局部轴(TCP 指向不变, 只自转)")
    g.add_argument("--stage2-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="第二阶段角度范围(度), 如 -90 0")
    g.add_argument("--stage2-step", type=float, default=None,
                   help="第二阶段步长(度). 可比第一阶段配得更粗以控制组合规模")
    g.add_argument("--stage2-max-trials", type=int, default=None,
                   help="第二阶段最多尝试的组合数, 0=不限")
    g.add_argument("--max-trials", type=int, default=None,
                   help="单物料最多尝试的组合数, 0=不限")
    g.add_argument("--max-joint-delta", type=float, default=None,
                   help="J1~J4 单段最大变化上限(度)")
    g.add_argument("--no-prescreen", action="store_true",
                   help="不用 IK 预筛, 每个角度都做完整规划(慢)")
    g.add_argument("--criterion-joints", type=int, nargs="+", default=None, metavar="J",
                   help="参与判据的关节(1-based). 如 --criterion-joints 1 2 3 4 "
                        "可回退到只管 J1~J4 的 v1 行为")
    g.add_argument("--min-limit-margin", type=float, default=None,
                   help="各关节离限位的最小余量(度). 0=关闭该判据")

    g = ap.add_argument_group("其他")
    g.add_argument("--lift-z", type=float, nargs="+", default=None,
                   help="抬升高度(m). 1 个值=抓取/放置相同; 2 个值=分别指定")
    g.add_argument("--linear-place", dest="linear_place", action="store_true", default=None,
                   help="开启直线约束(抬升点 -> 目标点沿 base Z 轴直线)")
    g.add_argument("--no-linear-place", dest="linear_place", action="store_false",
                   help="关闭直线约束")
    g.add_argument("--linear-kinds", type=str, nargs="+", default=None,
                   choices=["grasp", "place"],
                   help="哪些段走直线, 默认取 yaml 的 linear_move.kinds")
    g.add_argument("--linear-dev", type=float, default=None,
                   help="直线段横向偏移上限(mm)")
    g.add_argument("--linear-rot", type=float, default=None,
                   help="直线段姿态偏差上限(deg)")
    g.add_argument("--on-fail", type=str, default=None, choices=["stop", "skip"])
    g.add_argument("--out-dir", type=str, default=None)
    g.add_argument("--no-timestamp", action="store_true")
    g.add_argument("--no-incremental-save", action="store_true",
                   help="关闭增量落盘(默认每个物料跑完就写一次 trajectory.npz/csv/meta, "
                        "中途中断也能保留已完成部分)")
    return ap


def apply_cli(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    pp = cfg["pick_place"]
    gg, asr, cr = pp["grasp_grid"], pp["angle_search"], pp["criterion"]

    if args.rows is not None:
        gg["rows"] = int(args.rows)
    if args.cols is not None:
        gg["cols"] = int(args.cols)
    if args.x_range is not None:
        gg["x_range"] = list(args.x_range)
    if args.y_range is not None:
        gg["y_range"] = list(args.y_range)
    if args.grasp_z is not None:
        gg["z"] = float(args.grasp_z)
    if args.order is not None:
        gg["order"] = args.order
    if args.perimeter is not None:
        gg["perimeter_only"] = bool(args.perimeter)
    if args.place_position is not None:
        pp["place"]["position"] = list(args.place_position)
    if getattr(args, "place_tool_z_rotation_deg", None) is not None:
        pp["place"]["tool_z_rotation_deg"] = args.place_tool_z_rotation_deg

    if args.grasp_range is not None:
        asr["grasp"]["min_deg"], asr["grasp"]["max_deg"] = (float(v) for v in args.grasp_range)
    if args.grasp_step is not None:
        asr["grasp"]["step_deg"] = float(args.grasp_step)
    if args.grasp_axis is not None:
        asr["grasp"]["axis"] = args.grasp_axis
    if args.place_range is not None:
        asr["place"]["min_deg"], asr["place"]["max_deg"] = (float(v) for v in args.place_range)
    if args.place_step is not None:
        asr["place"]["step_deg"] = float(args.place_step)
    if args.place_axis is not None:
        asr["place"]["axis"] = args.place_axis
    if args.couple_place is not None:
        asr["couple_place_to_grasp"] = bool(args.couple_place)
    if args.stage2 is not None:
        asr.setdefault("stage2", {})["enable"] = bool(args.stage2)
    if args.stage2_axis is not None:
        asr.setdefault("stage2", {})["axis"] = args.stage2_axis
    if args.stage2_frame is not None:
        asr.setdefault("stage2", {})["frame"] = args.stage2_frame
    if args.stage2_range is not None:
        st2 = asr.setdefault("stage2", {})
        st2["min_deg"], st2["max_deg"] = (float(v) for v in args.stage2_range)
    if args.stage2_step is not None:
        asr.setdefault("stage2", {})["step_deg"] = float(args.stage2_step)
    if args.stage2_max_trials is not None:
        asr.setdefault("stage2", {})["max_trials"] = int(args.stage2_max_trials)
    if args.search_order is not None:
        asr["order"] = args.search_order
    if args.search_strategy is not None:
        asr["strategy"] = args.search_strategy
    if args.coarse_place is not None:
        asr["coarse_place"] = int(args.coarse_place)
    if args.max_trials is not None:
        asr["max_trials"] = int(args.max_trials)
    if args.max_joint_delta is not None:
        cr["max_joint_delta_deg"] = float(args.max_joint_delta)
    if args.no_prescreen:
        cr["prescreen_by_ik"] = False
    if args.criterion_joints is not None:
        cr["joints"] = [int(v) for v in args.criterion_joints]
    if args.min_limit_margin is not None:
        cr["min_limit_margin_deg"] = float(args.min_limit_margin)

    if args.lift_z is not None:
        if len(args.lift_z) == 1:
            pp["lift"]["grasp_z"] = pp["lift"]["place_z"] = float(args.lift_z[0])
        elif len(args.lift_z) == 2:
            pp["lift"]["grasp_z"] = float(args.lift_z[0])
            pp["lift"]["place_z"] = float(args.lift_z[1])
        else:
            raise ValueError("--lift-z 只接受 1 或 2 个值")
    if args.linear_place is not None:
        pp.setdefault("linear_move", {})["enable"] = bool(args.linear_place)
    if args.linear_kinds is not None:
        pp.setdefault("linear_move", {})["kinds"] = list(args.linear_kinds)
    if args.linear_dev is not None:
        pp.setdefault("linear_move", {})["max_deviation_mm"] = float(args.linear_dev)
    if args.linear_rot is not None:
        pp.setdefault("linear_move", {})["max_rotation_deg"] = float(args.linear_rot)
    if args.on_fail is not None:
        pp["on_fail"]["mode"] = args.on_fail
    if args.out_dir is not None:
        cfg["output"]["dir"] = args.out_dir
    if args.no_timestamp:
        cfg["output"]["add_timestamp"] = False
    return cfg


# ============================== 任务坐标变换 ==============================


def transform_pose(spec: PoseSpec, transform: np.ndarray) -> PoseSpec:
    """将任务位姿左乘齐次变换，即 ``T_effective = C @ T_current``。"""
    correction = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    position = correction[:3, :3] @ spec.position + correction[:3, 3]
    rotation = correction[:3, :3] @ quat_wxyz_to_matrix(spec.quat_wxyz)
    return PoseSpec(
        spec.name,
        position,
        matrix_to_quat_wxyz(rotation),
        spec.kind,
        spec.joint_config,
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """将一组 LINK_0 下的位置左乘任务坐标变换。"""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    correction = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return values @ correction[:3, :3].T + correction[:3, 3]


def transformed_cardinal_axis(
    transform: np.ndarray,
    axis: str,
    *,
    atol: float = 1e-7,
) -> str:
    """返回轴经变换后对应的 root x/y/z；非笛卡尔轴时拒绝。"""
    axis_name = str(axis).lower()
    if axis_name not in ("x", "y", "z"):
        raise ValueError(f"free_axis 只支持 x/y/z，收到 {axis!r}")
    correction = parse_rigid_transform_matrix(transform, "link0 target transform")
    source = np.zeros(3, dtype=np.float64)
    source[{"x": 0, "y": 1, "z": 2}[axis_name]] = 1.0
    direction = correction[:3, :3] @ source
    target_index = int(np.argmax(np.abs(direction)))
    expected = np.zeros(3, dtype=np.float64)
    expected[target_index] = 1.0 if direction[target_index] >= 0.0 else -1.0
    if not np.allclose(direction, expected, rtol=0.0, atol=atol):
        raise ValueError(
            f"变换后的 {axis_name.upper()} 轴不是规划 root 的 x/y/z 轴："
            f"direction={direction.round(9).tolist()}"
        )
    return ("x", "y", "z")[target_index]


def transformed_linear_direction(transform: np.ndarray, axis: str) -> np.ndarray:
    """Return a task-frame Cartesian axis as a unit vector in the planning root."""
    axis_name = str(axis).lower()
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis_name)
    if axis_index is None:
        raise ValueError(f"free_axis 只支持 x/y/z，收到 {axis!r}")
    correction = parse_rigid_transform_matrix(transform, "link0 target transform")
    return correction[:3, axis_index].copy()


# ============================== 阵列布点 ==============================


def build_grasp_points(gg: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    """在 x/y 范围内均匀布点(含端点)。rows 沿 x(竖向), cols 沿 y(横向)。

    rows 或 cols 为 1 时取该范围中点, 避免退化到端点。

    perimeter_only=true 时只保留「最外围一圈」, 即 row 或 col 落在首/末的点,
    内部点全部丢弃。用于快速摸清工作区边界: 外围最难到达, 中间能到的概率更高,
    因此扫一圈就能大致判定整个框选区域是否可用。
    rows<=2 或 cols<=2 时所有点本来就都在外围, 该选项无实际效果。

    order='ring' 仅在 perimeter_only 下有意义: 按闭环顺时针走一圈
    (上行 -> 右列 -> 下行 -> 左列), 相邻点始终相邻, 空行程最短。
    """
    x0, x1 = (float(v) for v in gg["x_range"])
    y0, y1 = (float(v) for v in gg["y_range"])
    z = float(gg["z"])
    rows = max(1, int(gg["rows"]))
    cols = max(1, int(gg["cols"]))

    xs = np.array([(x0 + x1) * 0.5]) if rows == 1 else np.linspace(x0, x1, rows)
    ys = np.array([(y0 + y1) * 0.5]) if cols == 1 else np.linspace(y0, y1, cols)

    order = str(gg.get("order", "row_major"))
    perim = bool(gg.get("perimeter_only", False))

    def is_perim(i: int, j: int) -> bool:
        return i == 0 or i == rows - 1 or j == 0 or j == cols - 1

    rc: List[Tuple[int, int]] = []
    if perim and order == "ring":
        # 闭环走一圈: 上行(j 递增) -> 右列(i 递增) -> 下行(j 递减) -> 左列(i 递减)
        if rows == 1:
            rc = [(0, j) for j in range(cols)]
        elif cols == 1:
            rc = [(i, 0) for i in range(rows)]
        else:
            rc = [(0, j) for j in range(cols)]
            rc += [(i, cols - 1) for i in range(1, rows)]
            rc += [(rows - 1, j) for j in range(cols - 2, -1, -1)]
            rc += [(i, 0) for i in range(rows - 2, 0, -1)]
    else:
        for i in range(rows):
            js = range(cols - 1, -1, -1) if (order == "snake" and i % 2 == 1) else range(cols)
            for j in js:
                if perim and not is_perim(i, j):
                    continue
                rc.append((i, j))

    pts: List[Dict[str, Any]] = []
    for i, j in rc:
        pts.append(
            {
                "index": len(pts),
                "row": int(i),
                "col": int(j),
                "position": [float(xs[i]), float(ys[j]), z],
            }
        )
    if max_items is not None:
        pts = pts[: int(max_items)]
    return pts


def axis_delta(axis: str, deg: float) -> List[float]:
    """把「绕某轴转 deg 度」转成 rpy 增量三元组。"""
    a = str(axis).lower()
    if a == "x":
        return [float(deg), 0.0, 0.0]
    if a == "y":
        return [0.0, float(deg), 0.0]
    if a == "z":
        return [0.0, 0.0, float(deg)]
    raise ValueError(f"angle_search.axis 只支持 x/y/z, 得到 {axis}")


def side_candidates(
    side_cfg: Dict[str, Any], order: str = "abs", label: str = ""
) -> List[float]:
    """生成单侧(抓取或放置)的候选角度序列(度)。

    order='abs' 时按 |角度| 从小到大排序, 同绝对值先正后负 ——
    这样优先尝试贴近基准姿态的角度, 通常更接近工艺意图。
    order='asc'/'desc' 分别按角度数值递增/递减排序。

    min_deg > max_deg 时自动互换并告警: 负向范围很容易被顺手写成
    "0, -90"(口语顺序), 那样 (hi-lo) 为负会只生成 1 个候选并静默失效。
    """
    lo = float(side_cfg["min_deg"])
    hi = float(side_cfg["max_deg"])
    if lo > hi:
        print(f"[WARN] {label or 'angle_search'} 的 min_deg({lo:g}) > max_deg({hi:g}), "
              f"已自动互换为 [{hi:g}, {lo:g}]。负向范围请写 min_deg={hi:g} max_deg={lo:g}")
        lo, hi = hi, lo
    st = abs(float(side_cfg.get("step_deg", 5.0))) or 5.0
    n = int(np.floor((hi - lo) / st + 1e-9)) + 1
    cands = [lo + k * st for k in range(max(n, 1))]
    if cands and cands[-1] < hi - 1e-9:
        cands.append(hi)
    # 去重(浮点)
    uniq: List[float] = []
    for c in cands:
        if not any(abs(c - u) < 1e-9 for u in uniq):
            uniq.append(c)
    if str(order) == "abs":
        uniq.sort(key=lambda v: (abs(v), -v))
    elif str(order) == "desc":
        uniq.sort(reverse=True)
    else:
        uniq.sort()
    return uniq


def couple_sign(grasp_cfg: Dict[str, Any]) -> float:
    """耦合模式下放置角相对抓取角的符号: p = sign * g。

    规则(按 grasp.min_deg 的符号整轮判定, 不是逐候选判定):
        grasp.min_deg <  0  ->  +1, 放置角「等于」抓取角
        grasp.min_deg >= 0  ->  -1, 放置角「相反」于抓取角

    由来: 抓取侧向负向搜时(min_deg<0), 取放两端需要同向的工具转角;
    抓取侧从 0 起向正向搜时, 放置端需要反向转回来(见 yaml 里
    "放置姿态 = base_rpy 右乘 [-n,0,0], 与抓取相反" 的原始约定)。
    """
    return 1.0 if float(grasp_cfg["min_deg"]) < 0.0 else -1.0


def angle_combos(
    asr: Dict[str, Any], first: Optional[Tuple[float, float]] = None
) -> List[Tuple[float, float]]:
    """抓取角 x 放置角 的候选组合。

    first 不为 None 时把它排到最前(复用上一个物料成功的组合)。
    受 max_trials 限制以防组合爆炸(0/null = 不限, 完整遍历)。

    ---- 耦合模式 (couple_place_to_grasp=true) ----
    只搜索抓取角, 放置角由抓取角直接决定, 候选退化为 [(g, sign*g) for g in gs],
    数量从 |gs|x|ps| 降到 |gs|。sign 见 couple_sign():
        grasp.min_deg <  0  ->  p =  g  (相等)
        grasp.min_deg >= 0  ->  p = -g  (相反)
    注意: 两侧仍各自绕 angle_search.<side>.axis 旋转、各自叠加在自己的 base_rpy 上,
          耦合的只是「角度数值」, 不是最终姿态。
    place.min_deg/max_deg/step_deg 在此模式下被忽略(角度全部来自 grasp 侧),
    strategy / coarse_place 也不再起作用(没有 place 维度可分轮扫)。

    ---- 独立模式 (默认) ----
    strategy 决定笛卡尔积的遍历顺序:
      - 'abs_sum': 按 |g| + |p| 递增。优先"两边都贴近基准姿态"的组合。
        缺点: 被 max_trials 截断时, 大角度的 grasp 可能一次都没试到。
      - 'coarse_to_fine': 分轮次。
          第 1 轮: 每个 grasp 角只配 coarse_place 个 place 角(按 |p| 最小的几个),
                   于是所有 grasp 角(含大角度)都能在早期被覆盖一遍;
          第 2 轮起: 逐步补上剩余的 place 角。
        这样即使被 max_trials 截断, grasp 维度也已经"粗扫"过全程,
        不会出现整段 grasp 角从未尝试的盲区。
    """
    order = str(asr.get("order", "abs"))
    gs = side_candidates(asr["grasp"], order, "angle_search.grasp")
    strategy = str(asr.get("strategy", "abs_sum")).lower()

    combos: List[Tuple[float, float]] = []
    if bool(asr.get("couple_place_to_grasp", False)):
        sign = couple_sign(asr["grasp"])
        # + 0.0 消掉 -1*0.0 产生的 -0.0, 否则日志会打出 "p=-0deg"
        combos = [(g, sign * g + 0.0) for g in gs]
    else:
        ps = side_candidates(asr["place"], order, "angle_search.place")
        if strategy in ("coarse_to_fine", "coarse", "c2f"):
            # side_candidates 在 order='abs' 下已按 |角度| 升序, 直接切片即取"最粗"的几个
            n_coarse = max(int(asr.get("coarse_place", 3) or 3), 1)
            n_p = len(ps)
            # 轮次 k 取 ps 的第 [k*n_coarse, (k+1)*n_coarse) 段
            for k in range((n_p + n_coarse - 1) // n_coarse):
                chunk = ps[k * n_coarse : (k + 1) * n_coarse]
                round_combos = [(g, p) for g in gs for p in chunk]
                if order == "abs":
                    # 轮内仍让小角度优先
                    round_combos.sort(
                        key=lambda t: (abs(t[0]) + abs(t[1]), abs(t[0]), abs(t[1]))
                    )
                combos.extend(round_combos)
        else:
            for g in gs:
                for p in ps:
                    combos.append((g, p))
            if order == "abs":
                # 按两侧绝对值之和递增, 让"两边都小"的组合优先
                combos.sort(key=lambda t: (abs(t[0]) + abs(t[1]), abs(t[0]), abs(t[1])))

    if first is not None:
        combos = [first] + [
            c for c in combos
            if not (abs(c[0] - first[0]) < 1e-9 and abs(c[1] - first[1]) < 1e-9)
        ]
    mt = asr.get("max_trials")
    if mt:
        combos = combos[: int(mt)]
    return combos


def stage2_combos(asr: Dict[str, Any]) -> List[Tuple[float, float]]:
    """第二阶段候选: (a2_grasp, a2_place) 的笛卡尔积。

    第二阶段绕「第一阶段转完之后」的局部轴(stage2.axis, 默认 z)再转一次。
    两侧「不耦合」—— grasp 与 place 各自独立搜自己的 a2, 因此是笛卡尔积。
    (couple_place_to_grasp 只作用于第一阶段, 不影响这里。)

    首项恒为 (0, 0): 即"不做第二阶段旋转", 于是第一阶段的结果一定被优先采用,
    这既保证了「保留第一步探索功能」, 也让 (b1) 回退天然成立 ——
    第二阶段全灭时最差也会落回 (0, 0)。

    stage2.step_deg 可以配得比第一阶段粗, 以控制 |a2_g| x |a2_p| 的规模。
    stage2.max_trials 可另行限制总组合数(0/null = 不限)。
    """
    st2 = dict(asr.get("stage2") or {})
    order = str(st2.get("order", asr.get("order", "abs")))
    cands = side_candidates(st2, order, "angle_search.stage2")
    combos = [(g, p) for g in cands for p in cands]
    if order == "abs":
        combos.sort(key=lambda t: (abs(t[0]) + abs(t[1]), abs(t[0]), abs(t[1])))
    # 把 (0,0) 提到最前(若在候选内), 保证优先复用第一阶段结果
    zero = [c for c in combos if abs(c[0]) < 1e-9 and abs(c[1]) < 1e-9]
    if zero:
        combos = zero + [c for c in combos if c not in zero]
    else:
        combos = [(0.0, 0.0)] + combos
    mt = st2.get("max_trials")
    if mt:
        combos = combos[: int(mt)]
    return combos


def get_place_tool_z_rotation_deg(pp: Dict[str, Any]) -> float:
    """Validate an optional fixed rotation about the final place TCP local Z."""
    value = (pp.get("place") or {}).get("tool_z_rotation_deg", 0.0)
    try:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("boolean is not an angle")
        angle = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("pick_place.place.tool_z_rotation_deg must be a finite number") from exc
    if not np.isfinite(angle):
        raise ValueError("pick_place.place.tool_z_rotation_deg must be a finite number")
    return angle


def make_round_poses(
    grasp_pos: List[float],
    place_pos: List[float],
    angle_grasp_deg: float,
    angle_place_deg: float,
    pp: Dict[str, Any],
    item_idx: int,
    angle2_grasp_deg: float = 0.0,
    angle2_place_deg: float = 0.0,
) -> List[PoseSpec]:
    """构造一个物料的完整路点序列(不含起点):
        g_lift -> g -> g_lift -> p_lift -> p -> p_lift

    姿态分两阶段合成。第一阶段恒为「右乘」= 绕工具自身的 <side>.axis(默认 X)转:
        R1 = R_base @ R(<side>.axis, a1)

    第二阶段由 stage2.frame 决定绕哪根轴转:

      frame='base' (默认): 绕 base(LINK_0) 的 stage2.axis 轴转, TCP 指向被摆动。
        R2 = R(base_axis, a2) @ R1          <- 左乘
        等价写法: 取 R1^-1 的第三列(=R1 的第三行)作为「工具系下的转轴」再右乘 ——
        因为 R1 @ (R1^T @ z) = z, 两种写法数值完全一致(实测误差 ~1e-16)。
        这里用左乘实现, 比构造 Rodrigues 更简洁。

      frame='tool': 绕「第一阶段转完之后」的工具局部 stage2.axis 轴转,
        即绕 R1 旋转矩阵的第三列(当 axis=z)。夹爪指向不变, 只绕自身自转。
        R2 = R1 @ R(stage2.axis, a2)        <- 右乘

    注意 base_rpy.grasp=[-180,0,0] 时工具 Z 指向 LINK_0 的 **-Z**, 因此两种 frame
    下同一个 a2 的转动方向是相反的, 切换 frame 时角度范围的符号要重新确认。

    a2=0 时两种 frame 都退化为纯第一阶段。

    可选 place.tool_z_rotation_deg 在两阶段之后仅对放置姿态右乘:
        R_place_final = R2_place @ Rz(tool_z_rotation_deg)
    不改变任何候选角数值/顺序、抓取姿态或 TCP 的位置/Z 轴方向。
    """
    place_tool_z_deg = get_place_tool_z_rotation_deg(pp)
    asr = pp["angle_search"]
    lift = pp["lift"]
    st2 = asr.get("stage2") or {}
    st2_axis = str(st2.get("axis") or "z")
    st2_frame = str(st2.get("frame", "base")).lower()
    compose2 = compose_left if st2_frame == "base" else compose_right

    q_grasp = compose2(
        compose_right(
            rpy_deg_to_quat_wxyz(pp["base_rpy"]["grasp"]),
            axis_delta(asr["grasp"]["axis"], angle_grasp_deg),
        ),
        axis_delta(st2_axis, angle2_grasp_deg),
    )
    q_place = compose2(
        compose_right(
            rpy_deg_to_quat_wxyz(pp["base_rpy"]["place"]),
            axis_delta(asr["place"]["axis"], angle_place_deg),
        ),
        axis_delta(st2_axis, angle2_place_deg),
    )
    if place_tool_z_deg != 0.0:
        q_place = compose_right(q_place, [0.0, 0.0, place_tool_z_deg])

    def lifted(pos: List[float], q: np.ndarray, dz: float) -> np.ndarray:
        p = np.asarray(pos, dtype=np.float64)
        if str(lift.get("axis", "base_z")) == "tool_z_neg":
            return p - float(dz) * tool_z_axis(q)
        return p + np.array([0.0, 0.0, float(dz)])

    gz = float(lift["grasp_z"])
    pz = float(lift["place_z"])
    gp = np.asarray(grasp_pos, dtype=np.float64)
    ppos = np.asarray(place_pos, dtype=np.float64)
    tag = f"i{item_idx}"
    seq: List[PoseSpec] = []
    if gz > 1e-9:
        seq.append(PoseSpec(f"{tag}_g_lift_in", lifted(gp, q_grasp, gz), q_grasp, "lift"))
    seq.append(PoseSpec(f"{tag}_grasp", gp, q_grasp, "grasp"))
    if gz > 1e-9:
        seq.append(PoseSpec(f"{tag}_g_lift_out", lifted(gp, q_grasp, gz), q_grasp, "lift"))
    if pz > 1e-9:
        seq.append(PoseSpec(f"{tag}_p_lift_in", lifted(ppos, q_place, pz), q_place, "lift"))
    seq.append(PoseSpec(f"{tag}_place", ppos, q_place, "place"))
    if pz > 1e-9:
        seq.append(PoseSpec(f"{tag}_p_lift_out", lifted(ppos, q_place, pz), q_place, "lift"))
    return seq


# ============================== 判据 ==============================


def check_joint_delta(
    q_from: np.ndarray, q_to: np.ndarray, cr: Dict[str, Any]
) -> Tuple[bool, float, int]:
    """判据一: 段内 joints 里最大的关节变化 <= max_joint_delta_deg。

    Returns:
        (是否通过, 最大变化度数, 触发上限的关节序号 1-based)
    """
    idx = [int(j) - 1 for j in cr.get("joints", [1, 2, 3, 4])]
    d = np.degrees(np.abs(np.asarray(q_to) - np.asarray(q_from)))
    sub = d[idx]
    k = int(np.argmax(sub))
    return float(sub[k]) <= float(cr["max_joint_delta_deg"]), float(sub[k]), idx[k] + 1


def check_limit_margin(
    q: np.ndarray, lo: np.ndarray, hi: np.ndarray, cr: Dict[str, Any]
) -> Tuple[bool, float, int]:
    """判据二: 各关节离限位的余量 >= min_limit_margin_deg。

    贴限位意味着该关节已转不动, 优化器只能牺牲末端精度去凑目标, 是
    「规划成功但误差偏大」的首要原因, 因此单独设一条门槛。

    Args:
        q: [dof] 或 [N, dof] 关节角(rad)

    Returns:
        (是否通过, 最小余量度数, 余量最小的关节序号 1-based)
    """
    thr = cr.get("min_limit_margin_deg")
    p = np.asarray(q, dtype=np.float64).reshape(-1, lo.shape[0])
    margin = np.degrees(np.minimum(p - lo[None, :], hi[None, :] - p)).min(axis=0)
    k = int(np.argmin(margin))
    if not thr:
        return True, float(margin[k]), k + 1
    return float(margin[k]) >= float(thr), float(margin[k]), k + 1


def prescreen_angle(
    robot_dict: Dict[str, Any],
    world,
    pl: Dict[str, Any],
    wall_links: List[str],
    q_start: np.ndarray,
    seq: List[PoseSpec],
    cr: Dict[str, Any],
    lo: np.ndarray,
    hi: np.ndarray,
) -> Tuple[bool, Dict[str, Any]]:
    """用 IK 快速预筛某个角度: 逐路点求 IK(以上一点为种子), 检查两条判据。

    比直接做 trajopt 快一个量级, 用于在角度扫描中剔除明显不可行的候选。
    """
    qs: List[Optional[np.ndarray]] = [np.asarray(q_start, dtype=np.float64)]
    for p in seq:
        q, _ = solve_ik_for_pose(
            robot_dict, world, pl, p, wall_link_names=wall_links, seed_q=qs[-1]
        )
        if q is None:
            # 换分支再试一次, 避免因种子过近而漏解
            q, _ = solve_ik_for_pose(robot_dict, world, pl, p, wall_link_names=wall_links)
        if q is None:
            return False, {"reason": "IK 无解", "at": p.name}
        qs.append(q)

    # 判据二: 各路点离限位余量(路点本身贴限位, 中间轨迹只会更糟)
    ok_m, margin, jm = check_limit_margin(np.stack(qs, axis=0), lo, hi, cr)
    if not ok_m:
        return False, {
            "reason": "贴近关节限位",
            "joint": f"J_{jm}",
            "min_margin_deg": margin,
            "at": "路点",
        }

    # 判据一: 逐段关节变化
    worst = 0.0
    worst_at = ""
    worst_j = 0
    for i in range(len(qs) - 1):
        ok, dmax, j = check_joint_delta(qs[i], qs[i + 1], cr)
        if dmax > worst:
            worst, worst_j = dmax, j
            worst_at = seq[i].name if i == 0 else f"{seq[i - 1].name}->{seq[i].name}"
        if not ok:
            return False, {
                "reason": "关节变化超限",
                "at": worst_at,
                "max_delta_deg": dmax,
                "joint": f"J_{j}",
            }
    return True, {
        "max_delta_deg": worst,
        "at": worst_at,
        "joint": f"J_{worst_j}",
        "min_margin_deg": margin,
    }


# ============================== 规划一轮 ==============================


def check_straightness(
    mg, q: np.ndarray, target: PoseSpec, ee_link: str, free_axis: str = "z",
    free_direction: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Measure sampled TCP deviation from the requested Cartesian approach line.

    Historical axis metrics use the maximum error in either held root coordinate.
    An oblique line uses Euclidean perpendicular distance to its unit direction.
    Both paths independently check orientation drift in degrees.
    """
    fk = compute_fk(mg, q, [ee_link])
    key = ee_link if f"{ee_link}/pos" in fk else "ee"
    delta = np.asarray(fk[f"{key}/pos"], dtype=np.float64) - target.position[None, :]
    if free_direction is None:
        ax = {"x": 0, "y": 1, "z": 2}[str(free_axis).lower()]
        lat = [i for i in range(3) if i != ax]
        d_lat = float(np.abs(delta[:, lat]).max())
    else:
        direction = np.asarray(free_direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction).all() or abs(norm - 1.0) > 1e-6:
            raise ValueError("linear free_direction must be a finite unit vector")
        lateral = delta - (delta @ direction)[:, None] * direction[None, :]
        d_lat = float(np.linalg.norm(lateral, axis=1).max())
    d_rot = float(max(quat_angle_deg(target.quat_wxyz, x)
                      for x in fk[f"{key}/quat"]))
    return d_lat, d_rot


def plan_sequence(
    mg,
    q_start: np.ndarray,
    seq: List[PoseSpec],
    pl: Dict[str, Any],
    cr: Dict[str, Any],
    lo: np.ndarray,
    hi: np.ndarray,
    lin: Optional[Dict[str, Any]] = None,
    ee_link: str = "TCP_LINK",
) -> Tuple[bool, Dict[str, Any]]:
    """连续规划一串路点。返回 (成功, 结果字典)。

    结果字典含 position/velocity/acceleration/dt 与逐段报告。
    任一段违反判据即返回 False 与失败信息 —— 校验用的是插值后的完整轨迹,
    因此比预筛(只看路点)更严格, 能挡住"路点合规但中途甩出去"的情况。

    Args:
        lin: 直线段配置(pick_place.linear_move)。对 kind 在 lin['kinds'] 里的路点,
            给该段加「只允许沿 base free_axis 轴平移」的位姿约束, 并在规划后复核直线度。
    """
    cur = np.asarray(q_start, dtype=np.float64).copy()
    pos_l: List[np.ndarray] = []
    vel_l: List[np.ndarray] = []
    acc_l: List[np.ndarray] = []
    segs: List[Dict[str, Any]] = []
    dt = float(pl["interpolation_dt"])

    lin = lin or {}
    lin_on = bool(lin.get("enable", False))
    lin_kinds = set(lin.get("kinds") or ["place"])
    lin_axis = str(lin.get("free_axis", "z"))
    lin_method = str(lin.get("method", "axis_metric"))
    lin_tol_m = float(lin.get("max_deviation_mm", 3.0)) / 1000.0
    lin_tol_deg = float(lin.get("max_rotation_deg", 2.0))
    lin_direction = None
    lin_step_m = None
    if lin_on and lin_method == "waypoints_fk":
        lin_direction = np.asarray(lin["free_direction_root"], dtype=np.float64).reshape(3)
        if (not np.isfinite(lin_direction).all()
                or abs(float(np.linalg.norm(lin_direction)) - 1.0) > 1e-6):
            raise ValueError("linear free_direction_root must be a finite unit vector")
        lin_step_m = float(lin.get("waypoint_step_m", 0.0075))
        if not np.isfinite(lin_step_m) or lin_step_m <= 0.0:
            raise ValueError("linear waypoint_step_m must be finite and positive")
    metric = None

    for si, dst in enumerate(seq):
        is_lin = lin_on and dst.kind in lin_kinds
        if is_lin and lin_method == "axis_metric" and metric is None:
            metric = make_hold_axis_metric(
                mg, lin_axis, bool(lin.get("hold_rotation", True))
            )
        targets = [dst]
        if is_lin and lin_method == "waypoints_fk":
            if si == 0:
                return False, {"failed_at": dst.name, "failed_index": si,
                               "status": "LINEAR_START_MISSING", "linear": True,
                               "segments": segs}
            source = seq[si - 1]
            displacement = dst.position - source.position
            lateral = displacement - np.dot(displacement, lin_direction) * lin_direction
            if (np.linalg.norm(lateral) > 1e-8
                    or quat_angle_deg(source.quat_wxyz, dst.quat_wxyz) > 1e-4):
                return False, {"failed_at": dst.name, "failed_index": si,
                               "status": "LINEAR_ENDPOINT_MISMATCH", "linear": True,
                               "segments": segs}
            n_steps = max(1, int(np.ceil((np.linalg.norm(displacement) - 1e-12) / lin_step_m)))
            targets = [PoseSpec(
                f"{dst.name}_linear_step{k}",
                source.position + displacement * (k / n_steps),
                dst.quat_wxyz, dst.kind,
            ) for k in range(1, n_steps)] + [dst]

        t0 = time.time()
        q_parts, v_parts, a_parts = [], [], []
        attempts = 0
        stage_cur = cur
        stage_dt = None
        for step_index, target in enumerate(targets, start=1):
            res = plan_segment(
                mg, stage_cur, target, pl,
                pose_metric=metric if is_lin and lin_method == "axis_metric" else None,
            )
            ok = res is not None and res.success is not None and bool(res.success.item())
            if not ok:
                return False, {
                    "failed_at": dst.name,
                    "failed_index": si,
                    "linear_step": step_index if len(targets) > 1 else None,
                    "status": str(res.status) if res is not None else "JS_FAIL",
                    "linear": is_lin,
                    "segments": segs,
                }
            traj = res.get_interpolated_plan()
            q_step = traj.position.detach().cpu().numpy().astype(np.float64)
            v_step = (traj.velocity.detach().cpu().numpy().astype(np.float64)
                      if traj.velocity is not None else np.zeros_like(q_step))
            a_step = (traj.acceleration.detach().cpu().numpy().astype(np.float64)
                      if traj.acceleration is not None else np.zeros_like(q_step))
            this_dt = float(res.interpolation_dt)
            if stage_dt is not None and abs(this_dt - stage_dt) > 1e-9:
                return False, {"failed_at": dst.name, "failed_index": si,
                               "status": "LINEAR_TIMESTEP_MISMATCH", "linear": is_lin,
                               "segments": segs}
            stage_dt = this_dt
            q_parts.append(q_step if step_index == 1 else q_step[1:])
            v_parts.append(v_step if step_index == 1 else v_step[1:])
            a_parts.append(a_step if step_index == 1 else a_step[1:])
            stage_cur = q_step[-1].copy()
            attempts += int(res.attempts)
        dt_solve = time.time() - t0
        q = np.concatenate(q_parts, axis=0)
        v = np.concatenate(v_parts, axis=0)
        a = np.concatenate(a_parts, axis=0)
        dt = stage_dt

        # 判据一: 段内关节变化(用整段轨迹的极值, 而非仅端点差)
        d_seg = np.degrees(q.max(axis=0) - q.min(axis=0))
        idx = [int(j) - 1 for j in cr.get("joints", [1, 2, 3, 4])]
        k = int(np.argmax(d_seg[idx]))
        dmax = float(d_seg[idx][k])
        if dmax > float(cr["max_joint_delta_deg"]):
            return False, {
                "failed_at": dst.name,
                "failed_index": si,
                "status": f"JOINT_DELTA_EXCEED J_{idx[k] + 1}={dmax:.1f}deg",
                "segments": segs,
            }
        # 判据二: 整段轨迹的限位余量
        ok_m, margin, jm = check_limit_margin(q, lo, hi, cr)
        if not ok_m:
            return False, {
                "failed_at": dst.name,
                "failed_index": si,
                "status": f"LIMIT_MARGIN J_{jm}={margin:.2f}deg",
                "segments": segs,
            }
        # 判据三(仅直线段): 直线度复核。hold_partial_pose 是软约束, 必须实测。
        d_lat = d_rot = None
        if is_lin:
            d_lat, d_rot = check_straightness(
                mg, q, dst, ee_link, lin_axis,
                free_direction=lin_direction if lin_method == "waypoints_fk" else None,
            )
            if d_lat > lin_tol_m:
                return False, {
                    "failed_at": dst.name,
                    "failed_index": si,
                    "status": f"NOT_STRAIGHT dev={d_lat * 1000:.1f}mm",
                    "linear": True,
                    "segments": segs,
                }
            if lin.get("hold_rotation", True) and d_rot > lin_tol_deg:
                return False, {
                    "failed_at": dst.name,
                    "failed_index": si,
                    "status": f"ROT_DRIFT {d_rot:.2f}deg",
                    "linear": True,
                    "segments": segs,
                }

        pos_l.append(q[1:] if pos_l else q)
        vel_l.append(v[1:] if len(vel_l) else v)
        acc_l.append(a[1:] if len(acc_l) else a)
        seg: Dict[str, Any] = {
            "index": si,
            "to": dst.name,
            "kind": dst.kind,
            "n_points": int(q.shape[0]),
            "solve_time_s": float(dt_solve),
            "max_joint_delta_deg": dmax,
            "min_limit_margin_deg": margin,
            "attempts": attempts,
        }
        if is_lin:
            seg["linear"] = {
                "method": lin_method,
                "free_axis": lin_axis,
                "free_direction_root": (lin_direction.tolist()
                                        if lin_direction is not None else None),
                "waypoint_step_m": lin_step_m if lin_method == "waypoints_fk" else None,
                "subsegments": len(targets),
                "lateral_dev_mm": d_lat * 1000.0,
                "rotation_dev_deg": d_rot,
            }
        segs.append(seg)
        cur = q[-1].copy()

    return True, {
        "position": np.concatenate(pos_l, axis=0),
        "velocity": np.concatenate(vel_l, axis=0),
        "acceleration": np.concatenate(acc_l, axis=0),
        "dt": dt,
        "q_end": cur,
        "segments": segs,
    }


def write_trajectory_csv(
    out_dir: Path, joint_names: List[str], times: np.ndarray,
    positions: np.ndarray, ee_pos: np.ndarray, ee_quat: np.ndarray,
) -> Path:
    """写 trajectory.csv(时间 + 关节角 + 末端位姿)。

    增量保存与最终保存共用同一份实现, 避免两处格式漂移。
    """
    csv_path = Path(out_dir) / "trajectory.csv"
    rpy_all = np.stack([quat_wxyz_to_rpy_deg(q) for q in ee_quat], axis=0)
    with open(csv_path, "w") as f:
        f.write(",".join(["t"] + [f"q_{n}" for n in joint_names]
                         + ["ee_x", "ee_y", "ee_z", "ee_r", "ee_p", "ee_yw"]) + "\n")
        for i in range(positions.shape[0]):
            f.write(",".join([f"{times[i]:.4f}"]
                             + [f"{v:.6f}" for v in positions[i]]
                             + [f"{v:.6f}" for v in ee_pos[i]]
                             + [f"{v:.3f}" for v in rpy_all[i]]) + "\n")
    return csv_path


# ============================== 主流程 ==============================


def single_ee_motion_gen_config(robot_dict: Dict[str, Any], ee_link: str) -> Dict[str, Any]:
    """Keep only the task EE in MG's pose outputs; preserve all collision geometry.

    MotionGen.warmup passes every FK output as a pose goal. This CuRobo version
    retains auxiliary link goals when later calls use link_poses=None. Single-arm
    pick/place constrains only the TCP, so auxiliary FK outputs must not become
    hidden fixed targets. Collision link names/spheres are deliberately unchanged.
    """
    result = copy.deepcopy(robot_dict)
    result["robot_cfg"]["kinematics"]["link_names"] = [ee_link]
    return result


def main() -> int:
    args = build_argparser().parse_args()
    cfg = apply_cli(load_pick_place_config(args.config), args)
    rb, pp, ws, pl, outc = (
        cfg["robot"], cfg["pick_place"], cfg["workspace"], cfg["planner"], cfg["output"]
    )
    gg, asr, cr = pp["grasp_grid"], pp["angle_search"], pp["criterion"]
    try:
        place_tool_z_deg = get_place_tool_z_rotation_deg(pp)
    except ValueError as exc:
        print(f"[FAIL] 放置 TCP 固定自转角无效: {exc}")
        return 6
    if place_tool_z_deg != 0.0:
        print(f"[PLACE TOOL] 每个放置候选最终绕自身 Z 轴右乘固定 {place_tool_z_deg:g}deg；"
              "抓取/位置/候选配对不变")
    try:
        link0_target_transform = parse_rigid_transform_matrix(
            pp.get("link0_target_transform", np.eye(4)),
            "pick_place.link0_target_transform",
        )
    except (TypeError, ValueError) as exc:
        print(f"[FAIL] LINK_0 任务坐标变换无效: {exc}")
        return 6
    print("[FRAME] LINK_0 任务目标左乘变换 C =")
    print(np.array2string(link0_target_transform, precision=6, suppress_small=True))

    # ---------- 输出目录 ----------
    out_dir = Path(outc["dir"])
    if not out_dir.is_absolute():
        out_dir = TASK_ROOT / out_dir
    if outc.get("add_timestamp", True):
        out_dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] {out_dir}")

    # ---------- 物料阵列 ----------
    items = build_grasp_points(gg, args.max_items)
    place_pos = list(pp["place"]["position"])
    place_pos_effective = transform_points(
        np.asarray(place_pos, dtype=np.float64)[None, :],
        link0_target_transform,
    )[0]
    for item in items:
        item["effective_position"] = transform_points(
            np.asarray(item["position"], dtype=np.float64)[None, :],
            link0_target_transform,
        )[0].tolist()
    perim_on = bool(gg.get("perimeter_only", False))
    n_rows, n_cols = max(1, int(gg["rows"])), max(1, int(gg["cols"]))
    print(f"\n{'=' * 88}")
    print(f"[GRID] 物料阵列 {gg['rows']} 行(x) x {gg['cols']} 列(y) = {len(items)} 个, "
          f"z={gg['z']}, 顺序={gg.get('order')}")
    if perim_on:
        n_full_grid = n_rows * n_cols
        if n_rows <= 2 or n_cols <= 2:
            print(f"[GRID] perimeter_only=true, 但 rows/cols <= 2, "
                  f"所有点本来就在外围, 该选项无实际效果")
        else:
            print(f"[GRID] perimeter_only=true -> 只扫最外围一圈, "
                  f"{n_full_grid} 个点中保留 {len(items)} 个, "
                  f"丢弃内部 {n_full_grid - len(items)} 个")
        if str(pp["on_fail"]["mode"]) != "skip":
            print(f"[GRID] 提示: on_fail={pp['on_fail']['mode']}, "
                  f"外围点最难到达, 建议配 on_fail=skip 才能「到不了就跳过」"
                  f"并把整圈都扫完")
    print(f"{'=' * 88}")
    for it in items:
        print(f"  #{it['index']:<3} row={it['row']} col={it['col']}  "
              f"raw={np.round(it['position'], 4).tolist()}  "
              f"effective={np.round(it['effective_position'], 4).tolist()}")
    print(f"[GRID] 放置点(原 LINK_0): {place_pos}")
    print(f"[GRID] 放置点(变换后): {place_pos_effective.round(6).tolist()}")

    coupled = bool(asr.get("couple_place_to_grasp", False))
    gs_prev = side_candidates(asr["grasp"], asr.get("order", "abs"), "angle_search.grasp")
    ps_prev = side_candidates(asr["place"], asr.get("order", "abs"), "angle_search.place")
    combos_prev = angle_combos(asr)
    print(f"\n[ANGLE] 抓取侧: 绕工具 {asr['grasp']['axis'].upper()} 轴 "
          f"[{asr['grasp']['min_deg']:g}, {asr['grasp']['max_deg']:g}] deg, "
          f"步长 {asr['grasp']['step_deg']:g} -> {len(gs_prev)} 个候选")
    if coupled:
        # 耦合模式: 放置角由抓取角直接决定, place 的角度范围配置整段失效。
        # 关系(相等/相反)由 grasp.min_deg 的符号整轮判定。
        sign = couple_sign(asr["grasp"])
        rel = "等于" if sign > 0 else "相反于"
        expr = "p = g" if sign > 0 else "p = -g"
        print(f"[ANGLE] 放置侧: 绕工具 {asr['place']['axis'].upper()} 轴, "
              f"角度强制{rel}抓取角 ({expr}); "
              f"因 grasp.min_deg={asr['grasp']['min_deg']:g} "
              f"{'< 0' if sign > 0 else '>= 0'}")
        print(f"[ANGLE] 两侧耦合, 候选 = {len(gs_prev)} 组 (g, {expr.split('=')[1].strip()}), "
              f"实际尝试上限 {len(combos_prev)} 组 (order={asr.get('order', 'abs')})")
        print(f"[ANGLE] 注意: place 的 min_deg/max_deg/step_deg 与 strategy/"
              f"coarse_place 在耦合模式下均被忽略")
        n_full = len(gs_prev)
    else:
        print(f"[ANGLE] 放置侧: 绕工具 {asr['place']['axis'].upper()} 轴 "
              f"[{asr['place']['min_deg']:g}, {asr['place']['max_deg']:g}] deg, "
              f"步长 {asr['place']['step_deg']:g} -> {len(ps_prev)} 个候选")
        print(f"[ANGLE] 两侧独立, 笛卡尔积 = {len(gs_prev) * len(ps_prev)} 组, "
              f"实际尝试上限 {len(combos_prev)} 组 (order={asr.get('order', 'abs')})")
        n_full = len(gs_prev) * len(ps_prev)
        strat = str(asr.get("strategy", "abs_sum"))
        print(f"[ANGLE] 遍历策略 = {strat}", end="")
        if strat.lower() in ("coarse_to_fine", "coarse", "c2f"):
            print(f" (每轮每个 grasp 角配 {asr.get('coarse_place', 3)} 个 place 角)", end="")
        print()
    if len(combos_prev) >= n_full:
        print(f"[ANGLE] 覆盖率 100% (完整遍历, 无截断)")
    else:
        n_g_seen = len({g for g, _ in combos_prev})
        print(f"[ANGLE] 覆盖率 {len(combos_prev) / n_full * 100:.1f}% "
              f"(被 max_trials={asr.get('max_trials')} 截断; "
              f"grasp 角覆盖 {n_g_seen}/{len(gs_prev)} 个)")

    # ---------- 第二阶段(在第一阶段结果之上再转一次) ----------
    st2_cfg = dict(asr.get("stage2") or {})
    st2_on = bool(st2_cfg.get("enable", False))
    st2_axis = str(st2_cfg.get("axis", "z"))
    st2_frame = str(st2_cfg.get("frame", "base")).lower()
    st2_cands: List[Tuple[float, float]] = []
    if st2_on:
        if st2_frame not in ("base", "tool"):
            print(f"[FAIL] angle_search.stage2.frame 只支持 base/tool, 收到 {st2_frame!r}")
            return 6
        st2_cands = stage2_combos(asr)
        st2_side = side_candidates(st2_cfg, st2_cfg.get("order", asr.get("order", "abs")),
                                   "angle_search.stage2")
        col = {"x": 1, "y": 2, "z": 3}.get(st2_axis.lower(), 3)
        if st2_frame == "base":
            print(f"[STAGE2] 已开启: 在第一阶段结果之上, 再绕 "
                  f"base({rb['base_link']}) 的 {st2_axis.upper()} 轴转动")
            print(f"[STAGE2]   frame=base -> 实现为左乘 R({st2_axis.upper()}, a2) @ R1; "
                  f"等价于取 R1^-1 的第{col}列作为工具系转轴再右乘")
            print(f"[STAGE2]   该模式下 TCP 的指向会被摆动(绕竖直轴摆一个方位角)")
        else:
            print(f"[STAGE2] 已开启: 在第一阶段结果之上, 再绕「第一阶段转完后的"
                  f"工具局部 {st2_axis.upper()} 轴」转动")
            print(f"[STAGE2]   frame=tool -> 实现为右乘 R1 @ R({st2_axis.upper()}, a2); "
                  f"即绕 R1 的第{col}列方向转动")
            print(f"[STAGE2]   该模式下 TCP 的指向不变, 只绕自身轴自转")
        print(f"[STAGE2]   范围 [{st2_cfg.get('min_deg')}, {st2_cfg.get('max_deg')}] deg, "
              f"步长 {st2_cfg.get('step_deg')} -> {len(st2_side)} 个候选/侧")
        print(f"[STAGE2]   两侧不耦合(各自独立搜), 笛卡尔积 = "
              f"{len(st2_side) ** 2} 组, 实际尝试上限 {len(st2_cands)} 组"
              + (f" (被 stage2.max_trials={st2_cfg.get('max_trials')} 截断)"
                 if st2_cfg.get("max_trials") else ""))
        print(f"[STAGE2]   第一阶段角逐物料各自沿用其搜索结果; "
              f"第二阶段全灭则回退到 a2=0(即纯第一阶段结果)")
    else:
        print("[STAGE2] 未开启 (仅做第一阶段搜索)")
    jn = ",".join(f"J{j}" for j in cr.get("joints", [1, 2, 3, 4]))
    print(f"[CRIT ] 判据1: 每段内 [{jn}] 最大变化 <= {cr['max_joint_delta_deg']:g} deg")
    if cr.get("min_limit_margin_deg"):
        print(f"[CRIT ] 判据2: 全部关节离限位余量 >= {cr['min_limit_margin_deg']:g} deg")
    print(f"[CRIT ] 预筛={'开' if cr.get('prescreen_by_ik', True) else '关'}")

    # ---------- 直线段约束 ----------
    lin = dict(pp.get("linear_move") or {})
    if lin.get("enable", False):
        lin_kinds = list(lin.get("kinds") or ["place"])
        configured_axis = str(lin.get("free_axis", "z")).lower()
        bad_kinds = [k for k in lin_kinds if k not in ("grasp", "place")]
        if bad_kinds:
            print(f"[FAIL] linear_move.kinds 只支持 grasp/place, 收到 {bad_kinds}")
            return 6
        # hold_partial_pose 要求「被锁住的位姿分量」在起点与终点必须一致(curobo 内部
        # 会做 <5mm 的校验, 不满足直接返回 INVALID_PARTIAL_POSE_COST_METRIC)。
        # 本任务里 x_lift_in = x + [0,0,lift_z] 只有在 lift.axis=base_z 时
        # 才满足「x/y 相同、姿态相同」, 用 tool_z_neg 会直接失败, 提前拦下。
        # grasp 与 place 两侧都走同一个 lifted(), 因此约束任一侧都有此要求。
        if str(pp["lift"].get("axis", "base_z")) != "base_z":
            print(f"[FAIL] linear_move.enable=true 时 lift.axis 必须为 base_z "
                  f"(当前 {pp['lift'].get('axis')}), 否则抬升点与目标点的 x/y 不同, "
                  f"无法构成沿 {configured_axis.upper()} 轴的直线")
            return 6
        if configured_axis != "z":
            print("[FAIL] lift.axis=base_z 时 linear_move.free_axis 必须为 z, "
                  f"实际 {configured_axis!r}")
            return 6
        method = str(lin.get("method", "axis_metric"))
        if method not in ("axis_metric", "waypoints_fk"):
            print(f"[FAIL] linear_move.method 不支持 {method!r}")
            return 6
        lin["method"] = method
        lin["configured_free_axis"] = configured_axis
        if method == "axis_metric":
            try:
                effective_axis = transformed_cardinal_axis(
                    link0_target_transform, configured_axis
                )
            except (TypeError, ValueError) as exc:
                print("[FAIL] LINK_0 目标变换与 axis_metric 直线约束不兼容: "
                      f"{exc}。倾斜安装请显式选用 linear_move.method=waypoints_fk。")
                return 6
            lin["free_axis"] = effective_axis
            if effective_axis != configured_axis:
                print(f"[TARGET TF] linear free_axis: {configured_axis} -> "
                      f"root {effective_axis}")
        else:
            try:
                direction = transformed_linear_direction(
                    link0_target_transform, configured_axis
                )
                waypoint_step_m = float(lin.get("waypoint_step_m", 0.0075))
                if not np.isfinite(waypoint_step_m) or waypoint_step_m <= 0.0:
                    raise ValueError("waypoint_step_m 必须为有限正数")
            except (TypeError, ValueError) as exc:
                print(f"[FAIL] waypoints_fk 配置无效: {exc}")
                return 6
            lin["free_axis"] = "vector"
            lin["free_direction_root"] = direction.tolist()
            lin["waypoint_step_m"] = waypoint_step_m
            effective_axis = "vector"
            print(f"[TARGET TF] 原世界 {configured_axis.upper()} -> root 方向 "
                  f"{np.round(direction, 6).tolist()}；每 {waypoint_step_m * 1000:g}mm "
                  "设置一个中间位姿，逐采样 FK 复核整段直线")
        # 抬升高度为 0 时对应的 *_lift_in 路点根本不会生成, 约束等于空转
        for kind, zkey in (("grasp", "grasp_z"), ("place", "place_z")):
            if kind in lin_kinds and float(pp["lift"][zkey]) <= 1e-9:
                print(f"[WARN] lift.{zkey} ~= 0, {kind}_lift_in 路点不存在, "
                      f"{kind} 段的直线约束无实际作用")
        print(f"[LINE ] 直线约束: method={method}, 对 {lin_kinds} 段沿原任务 "
              f"{configured_axis.upper()} 方向插拔"
              f"{', 姿态锁死' if lin.get('hold_rotation', True) else ''}; "
              f"横向偏移 <= {lin.get('max_deviation_mm', 3.0):g}mm, "
              f"姿态偏差 <= {lin.get('max_rotation_deg', 2.0):g}deg")
    else:
        print("[LINE ] 直线约束: 关闭")

    # ---------- 世界 + 机器人 ----------
    print()
    world = make_world_config(ws)
    robot_dict = load_robot_cfg_dict(rb)
    wall_cfg = ws.get("wall") or {}
    wall_links: List[str] = []
    wall_report: Dict[str, Any] = {"applied": False}
    if wall_cfg.get("enable", False) and wall_cfg.get("collision_link_names"):
        wall_links = list(wall_cfg["collision_link_names"])
        dual_prefix = rb.get("dual_arm_prefix") or ""
        if dual_prefix:
            wall_links += [dual_prefix + name for name in wall_links]
    def apply_wall_filter_before_warmup(solver) -> None:
        nonlocal wall_report
        wall_report = restrict_world_collision_to_links(solver, wall_links)
        wall_report["applied"] = True

    # CUDA graph replay does not execute newly installed Python wrappers.
    # Apply the configured world-link mask before capture, as in the dual entry.
    mg_robot_dict = (single_ee_motion_gen_config(robot_dict, rb["ee_link"])
                     if not rb.get("dual_arm_prefix") else robot_dict)
    print(f"[PLAN] MotionGen pose outputs = {mg_robot_dict['robot_cfg']['kinematics']['link_names']}; "
          "collision geometry unchanged")
    mg = make_motion_gen(
        mg_robot_dict, world, pl,
        pre_warmup=apply_wall_filter_before_warmup if wall_links else None,
    )

    joint_names = list(mg.joint_names)
    dof = len(joint_names)
    _jl = mg.kinematics.get_joint_limits().position
    q_lo = _jl[0].cpu().numpy().astype(np.float64)
    q_hi = _jl[1].cpu().numpy().astype(np.float64)
    print(f"[ROBOT] 关节限位(deg): "
          f"{[f'{a:.0f}~{b:.0f}' for a, b in zip(np.degrees(q_lo), np.degrees(q_hi))]}")

    # ---------- home 起始关节角 ----------
    print()
    home = pp["home"]
    if home.get("joint_deg"):
        q_home = np.radians(np.asarray(home["joint_deg"], dtype=np.float64)).reshape(-1)
        if q_home.size != dof:
            print(f"[FAIL] home.joint_deg 长度 {q_home.size} != dof {dof}")
            return 2
        print(f"[HOME] 使用给定关节角: {np.degrees(q_home).round(2).tolist()}")
    else:
        hp_raw = PoseSpec.from_rpy_deg(
            "home", home["position"], home["rpy_deg"], "start"
        )
        hp = transform_pose(hp_raw, link0_target_transform)
        q_home_seed = None
        if home.get("ik_seed_joint_deg") is not None:
            q_home_seed = np.radians(
                np.asarray(home["ik_seed_joint_deg"], dtype=np.float64)
            ).reshape(-1)
            if q_home_seed.size != dof:
                print(
                    f"[FAIL] home.ik_seed_joint_deg 长度 {q_home_seed.size} "
                    f"!= dof {dof}"
                )
                return 2
            if not np.isfinite(q_home_seed).all():
                print("[FAIL] home.ik_seed_joint_deg 包含非有限值")
                return 2
            print(
                "[HOME] IK 分支种子: "
                f"{np.degrees(q_home_seed).round(2).tolist()}"
            )
        q_home, ik_res = solve_ik_for_pose(
            robot_dict, world, pl, hp,
            wall_link_names=wall_links,
            seed_q=q_home_seed,
        )
        if q_home is None and q_home_seed is not None:
            print("[HOME] 指定分支 IK 无解，退回无 seed 全局搜索")
            q_home, ik_res = solve_ik_for_pose(
                robot_dict, world, pl, hp, wall_link_names=wall_links
            )
        if q_home is None:
            print("[FAIL] home 位姿 IK 无解")
            print(f"       position_error={ik_res.position_error}")
            dump_json({"stage": "home_ik", "config": cfg}, out_dir / "plan_failed.json")
            return 3
        print(f"[HOME] {hp}")
        print(f"[HOME] q = {np.degrees(q_home).round(2).tolist()}")

    # ---------- 逐物料规划 ----------
    print(f"\n{'=' * 88}\n[PLAN] 开始逐物料规划\n{'=' * 88}")
    all_pos: List[np.ndarray] = []
    all_vel: List[np.ndarray] = []
    all_acc: List[np.ndarray] = []
    item_reports: List[Dict[str, Any]] = []
    pose_records: List[Dict[str, Any]] = []
    q_cur = q_home.copy()
    interp_dt = float(pl["interpolation_dt"])
    last_ok_angle: Optional[Tuple[float, float]] = None
    failed_info: Optional[Dict[str, Any]] = None
    skipped: List[Dict[str, Any]] = []
    t_all = time.time()

    for it in items:
        idx = it["index"]
        print(f"\n--- 物料 {idx + 1}/{len(items)}  抓取点 "
              f"{np.round(it['position'], 4).tolist()} ---")
        reuse = last_ok_angle if asr.get("reuse_last_success", True) else None
        cands = angle_combos(asr, first=reuse)
        tried: List[Dict[str, Any]] = []
        chosen: Optional[Dict[str, Any]] = None

        def attempt(ang_g: float, ang_p: float,
                    a2_g: float = 0.0, a2_p: float = 0.0) -> Optional[Dict[str, Any]]:
            """试一组角度: 预筛 -> 完整规划。成功返回 chosen 字典, 失败返回 None。

            失败原因会追加进外层的 tried 列表(用于失败报告)。
            """
            raw_seq = make_round_poses(
                it["position"], place_pos, ang_g, ang_p, pp, idx, a2_g, a2_p
            )
            seq = [
                transform_pose(pose, link0_target_transform)
                for pose in raw_seq
            ]
            rec = {"angle_grasp_deg": ang_g, "angle_place_deg": ang_p,
                   "angle2_grasp_deg": a2_g, "angle2_place_deg": a2_p}

            # 预筛: 用 IK 快速判断判据
            if cr.get("prescreen_by_ik", True):
                ok, info = prescreen_angle(
                    robot_dict, world, pl, wall_links, q_cur, seq, cr, q_lo, q_hi
                )
                if not ok:
                    detail = info.get("joint", "")
                    if info.get("max_delta_deg") is not None:
                        detail += f" {info['max_delta_deg']:.0f}deg"
                    if info.get("min_margin_deg") is not None:
                        detail += f" 余量{info['min_margin_deg']:.1f}deg"
                    print(f"  -> 预筛不通过: {info.get('reason')} {detail} "
                          f"@ {info.get('at', '')}")
                    tried.append({**rec, "stage": "prescreen", **info})
                    return None
                print(f"  -> 预筛通过(最大变化 {info['max_delta_deg']:.0f}deg, "
                      f"限位余量 {info.get('min_margin_deg', 0):.1f}deg), 规划中 ...",
                      end="", flush=True)

            ok, res = plan_sequence(mg, q_cur, seq, pl, cr, q_lo, q_hi, lin, rb["ee_link"])
            if not ok:
                print(f"  -> 规划失败: {res.get('status')} @ {res.get('failed_at')}")
                tried.append({**rec, "stage": "plan",
                              **{k: v for k, v in res.items() if k != "segments"}})
                return None
            n = res["position"].shape[0]
            worst = max(x["max_joint_delta_deg"] for x in res["segments"])
            marg = min(x["min_limit_margin_deg"] for x in res["segments"])
            lin_devs = [x["linear"]["lateral_dev_mm"] for x in res["segments"] if "linear" in x]
            lin_txt = f"  直线偏移 {max(lin_devs):.2f}mm" if lin_devs else ""
            print(f"  -> OK  {n} 点  {n * res['dt']:.2f}s  "
                  f"最大变化 {worst:.0f}deg  限位余量 {marg:.1f}deg{lin_txt}")
            return {"angle_grasp_deg": ang_g, "angle_place_deg": ang_p,
                    "angle2_grasp_deg": a2_g, "angle2_place_deg": a2_p,
                    "result": res, "seq": seq,
                    "worst_delta": worst, "min_margin": marg,
                    "max_linear_dev_mm": max(lin_devs) if lin_devs else None}

        # ---------- 第一阶段: 绕工具 <side>.axis 搜索 ----------
        for ang_g, ang_p in cands:
            mark = ""
            if reuse is not None and abs(ang_g - reuse[0]) < 1e-9 and abs(ang_p - reuse[1]) < 1e-9:
                mark = " (复用上次成功组合)"
            print(f"    g={ang_g:+.0f} p={ang_p:+.0f}deg{mark}", end="", flush=True)
            chosen = attempt(ang_g, ang_p)
            if chosen is not None:
                break

        # ---------- 第二阶段: 固定本物料的第一阶段角, 再绕局部轴搜索 ----------
        # 采用「本物料自己」第一阶段搜出来的角(逐物料各不相同), 而非全局写死一个值。
        # 候选首项恒为 (0,0) 且第一阶段结果已在手, 因此第二阶段全灭时自动回退
        # 到纯第一阶段的轨迹 —— 不会因为开了第二阶段反而让物料失败。
        if st2_on and chosen is not None:
            base_chosen = chosen
            a1_g = chosen["angle_grasp_deg"]
            a1_p = chosen["angle_place_deg"]
            print(f"    [S2] 固定第一阶段 g={a1_g:+.0f} p={a1_p:+.0f}deg, "
                  f"绕 {'base ' + rb['base_link'] if st2_frame == 'base' else '工具局部'} "
                  f"{st2_axis.upper()} 轴搜索 {len(st2_cands)} 组")
            chosen2: Optional[Dict[str, Any]] = None
            for a2_g, a2_p in st2_cands:
                if abs(a2_g) < 1e-9 and abs(a2_p) < 1e-9:
                    # (0,0) 等价于第一阶段结果, 直接复用, 不必重算
                    chosen2 = base_chosen
                    print(f"    [S2] a2g=+0 a2p=+0deg  -> 复用第一阶段结果")
                    continue
                print(f"    [S2] a2g={a2_g:+.0f} a2p={a2_p:+.0f}deg", end="", flush=True)
                got = attempt(a1_g, a1_p, a2_g, a2_p)
                if got is not None:
                    chosen2 = got
                    break
            if chosen2 is not None and chosen2 is not base_chosen:
                chosen = chosen2
            else:
                # (b1) 回退: 第二阶段无可行解, 沿用第一阶段轨迹
                print(f"    [S2] 第二阶段无可行解, 回退到第一阶段结果 "
                      f"(a2=0)")
                chosen = base_chosen

        if chosen is None:
            print(f"    [FAIL] 物料 {idx + 1} 在 {len(cands)} 组候选(抓取角 x 放置角)中均无可行解")
            failed_info = {
                "failed_item_index": idx,
                # 历史字段始终表示实际送入规划器的 root 坐标；原始任务
                # 网格值另存，避免非单位 C 时下游把两个坐标系混在一起。
                "failed_item_position": it["effective_position"],
                "failed_item_position_raw": it["position"],
                "failed_item_effective_position": it["effective_position"],
                "n_combos_tried": len(cands),
                "candidates": tried,
            }
            if str(pp["on_fail"]["mode"]) == "stop":
                break
            # skip 模式下会有多个物料失败, 逐个记录 —— 只留最后一个会丢信息
            print("    [SKIP] on_fail=skip, 跳过该物料")
            skipped.append({
                "index": idx,
                "row": it["row"],
                "col": it["col"],
                "position": it["effective_position"],
                "position_raw": it["position"],
                "effective_position": it["effective_position"],
                "n_combos_tried": len(cands),
            })
            item_reports.append({
                "index": idx, "success": False,
                "row": it["row"], "col": it["col"],
                "position": it["effective_position"],
                "position_raw": it["position"],
                "effective_position": it["effective_position"],
                "n_angles_tried": len(tried),
            })
            # 跳过的物料不产生轨迹点, 但要把「已跳过」这件事落盘,
            # 否则中途中断后看不出哪些点位试过且失败了。
            if not args.no_incremental_save:
                dump_json({"skipped": skipped, "last_failed_detail": failed_info,
                           "n_items_done": len(item_reports),
                           "n_items_total": len(items)},
                          out_dir / "plan_skipped.json")
            continue

        res = chosen["result"]
        all_pos.append(res["position"])
        all_vel.append(res["velocity"])
        all_acc.append(res["acceleration"])
        interp_dt = res["dt"]
        q_cur = res["q_end"].copy()
        last_ok_angle = (chosen["angle_grasp_deg"], chosen["angle_place_deg"])
        for p in chosen["seq"]:
            pose_records.append({**p.to_dict(), "item_index": idx})
        item_reports.append(
            {
                "index": idx,
                "success": True,
                "row": it["row"],
                "col": it["col"],
                "position": it["effective_position"],
                "position_raw": it["position"],
                "effective_position": it["effective_position"],
                "angle_grasp_deg": chosen["angle_grasp_deg"],
                "angle_place_deg": chosen["angle_place_deg"],
                # 第二阶段增量角(绕第一阶段转完后的局部 stage2.axis 轴)。
                # 未开启第二阶段或已回退时为 0。
                "angle2_grasp_deg": chosen.get("angle2_grasp_deg", 0.0),
                "angle2_place_deg": chosen.get("angle2_place_deg", 0.0),
                "grasp_rpy_deg": quat_wxyz_to_rpy_deg(chosen["seq"][1].quat_wxyz).tolist(),
                "place_rpy_deg": quat_wxyz_to_rpy_deg(
                    [p for p in chosen["seq"] if p.kind == "place"][0].quat_wxyz
                ).tolist(),
                "n_points": int(res["position"].shape[0]),
                "duration_s": float(res["position"].shape[0] * res["dt"]),
                "max_joint_delta_deg": chosen["worst_delta"],
                "min_limit_margin_deg": chosen["min_margin"],
                "max_linear_dev_mm": chosen["max_linear_dev_mm"],
                "n_angles_tried": len(tried) + 1,
                "segments": res["segments"],
            }
        )

        # ---------- 增量落盘 ----------
        # 每个物料跑完就把「到目前为止」的完整轨迹写一次盘。
        # 目的: 长任务(几十个物料/数小时)中途被 Ctrl-C、断电、OOM 时不至于
        #       颗粒无收 —— 已完成的部分随时可用 run_rviz.sh 播放。
        # 代价: 每次重写整个 npz/csv, 单次约几十毫秒, 相对单物料 6~15s 的
        #       规划耗时可忽略。
        # 注意: 这里只写轨迹与轻量进度, 不做 FK/越界/自碰撞等全量校验报告
        #       (那些放在最后统一算, 避免每轮重复计算拖慢速度)。
        if not args.no_incremental_save:
            try:
                _p = np.concatenate(all_pos, axis=0)
                _v = np.concatenate(all_vel, axis=0)
                _a = np.concatenate(all_acc, axis=0)
                _t = np.arange(_p.shape[0], dtype=np.float64) * interp_dt
                _fk = compute_fk(mg, _p, [rb["ee_link"]])
                save_trajectory(
                    out_dir, joint_names, _p, _v, _a, _t,
                    _fk["ee/pos"], _fk["ee/quat"],
                    {
                        "partial": True,
                        "note": "增量保存: 规划进行中, 这是已完成物料的轨迹; "
                                "全部跑完后会被最终结果覆盖",
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "n_items_done": len(item_reports),
                        "n_items_total": len(items),
                        "n_items_success": sum(1 for r in item_reports
                                               if r.get("success")),
                        "n_items_skipped": len(skipped),
                        "items": item_reports,
                        "skipped": skipped,
                        "pose_sequence": pose_records,
                        "interpolation_dt": interp_dt,
                        "n_points": int(_p.shape[0]),
                        "total_duration_s": float(_t[-1]) if _t.size else 0.0,
                        "robot": {"joint_names": joint_names,
                                  "base_link": rb["base_link"],
                                  "ee_link": rb["ee_link"]},
                        "config": cfg,
                    },
                )
                write_trajectory_csv(out_dir, joint_names, _t, _p,
                                     _fk["ee/pos"], _fk["ee/quat"])
                print(f"    [SAVE] 已存进度 {len(item_reports)}/{len(items)} 个物料, "
                      f"{_p.shape[0]} 点 -> {out_dir.name}/trajectory.{{npz,csv}}")
            except Exception as e:  # 存盘失败不能影响规划主流程
                print(f"    [WARN] 增量保存失败(不影响继续规划): {e}")

    n_ok_items = sum(1 for r in item_reports if r.get("success"))
    print(f"\n[PLAN] 完成 {n_ok_items}/{len(items)} 个物料, 累计 {time.time() - t_all:.1f}s")

    if not all_pos:
        print("[FAIL] 没有任何物料成功, 无轨迹可保存")
        dump_json(
            {"stage": "no_item_success", "failed": failed_info, "config": cfg},
            out_dir / "plan_failed.json",
        )
        return 4

    positions = np.concatenate(all_pos, axis=0)
    velocities = np.concatenate(all_vel, axis=0)
    accelerations = np.concatenate(all_acc, axis=0)
    n_total = positions.shape[0]
    times = np.arange(n_total, dtype=np.float64) * interp_dt
    print(f"[TRAJ] 总计 {n_total} 点, 时长 {times[-1]:.2f}s, dt={interp_dt}")

    # ---------- FK 与校验 ----------
    # Single-arm reports consume only TCP FK; gripper extent uses collision spheres.
    flange_link = rb.get("flange_link") or "LINK_6"
    link_list = [rb["ee_link"]]
    for ln in ("gripper_link", flange_link):
        if ln and ln not in link_list:
            link_list.append(ln)
    fk = compute_fk(mg, positions, link_list)
    ee_pos = fk["ee/pos"]
    ee_quat = fk["ee/quat"]

    viol_report: Dict[str, Any] = {"checked": False}
    if ws.get("check_after_plan", True):
        margin = float(ws.get("check_margin", 0.0))
        inside, viol = check_in_bounds(ee_pos, ws, margin=margin)
        n_bad = int((~inside).sum())
        viol_report = {
            "checked": True,
            "margin_m": margin,
            "n_points": n_total,
            "n_violation": n_bad,
            "max_violation_mm": float(viol.max()) * 1000.0 if viol.size else 0.0,
        }
        print(f"[CHECK] 工作空间: {n_bad}/{n_total} 越界, "
              f"最大 {viol_report['max_violation_mm']:.1f}mm")

    motion_report = compute_joint_motion_report(
        positions, joint_names, rb.get("cspace_distance_weight")
    )
    limit_report = compute_joint_limit_margin_report(
        mg, positions, joint_names, float(pl.get("limit_margin_warn_deg", 1.0))
    )
    gripper_report: Dict[str, Any] = {"checked": False}
    if ws.get("report_gripper_extent", False):
        gripper_report = compute_gripper_extent_report(mg, positions, ws, flange_link, times)

    self_coll_report: Dict[str, Any] = {"checked": False}
    if pl.get("self_collision_check", True):
        try:
            sph = mg.kinematics.get_state(
                mg.tensor_args.to_device(positions)
            ).link_spheres_tensor.unsqueeze(1)
            cost_fn = None
            for rollout in mg.get_all_rollout_instances():
                for attr in ("robot_self_collision_constraint", "robot_self_collision_cost"):
                    c = getattr(rollout, attr, None)
                    if c is not None and getattr(c, "enabled", True):
                        cost_fn = c
                        break
                if cost_fn is not None:
                    break
            d = cost_fn.forward(sph).reshape(-1).detach().cpu().numpy()
            n_bad = int((d > 0).sum())
            self_coll_report = {"checked": True, "n_points": n_total, "n_collision": n_bad}
            print(f"[CHECK] 自碰撞: {n_bad}/{n_total} 点碰撞")
        except Exception as e:  # noqa: BLE001
            self_coll_report = {"checked": False, "reason": str(e)}
            print(f"[CHECK] 自碰撞复核跳过: {e}")

    # ---------- 保存 ----------
    meta: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_type": "pick_place_cycle",
        "robot": {
            "robot_yml": rb["robot_yml"],
            "base_link": rb["base_link"],
            "ee_link": rb["ee_link"],
            "flange_link": flange_link,
            "urdf": rb["urdf"],
            "urdf_abs": str(resolve_repo_path(rb["urdf"])),
            "joint_names": joint_names,
            "link0_target_transform": link0_target_transform.tolist(),
        },
        "rotation_convention": "rpy_deg, fixed-axis XYZ extrinsic (ROS/URDF), R = Rz*Ry*Rx",
        "quaternion_order": {"npz_ee_quats": "wxyz", "ros": "xyzw"},
        "home_joint_deg": np.degrees(q_home).round(4).tolist(),
        "grid": {
            "rows": gg["rows"], "cols": gg["cols"], "z": gg["z"],
            "x_range": gg["x_range"], "y_range": gg["y_range"],
            "order": gg.get("order"), "n_items": len(items),
            "perimeter_only": perim_on,
        },
        # place_position 保持“实际规划 root 坐标”语义，供轨迹播放器直接发布；
        # 原始任务配置另存，避免下游再次重复应用 LINK_0 修正。
        "place_position": place_pos_effective.tolist(),
        "place_position_raw": place_pos,
        "angle_search": asr,
        "linear_move": lin,
        "criterion": cr,
        "n_items_total": len(items),
        "n_items_success": n_ok_items,
        "items": item_reports,
        "failed": failed_info,
        # on_fail=skip 时被跳过(所有角度都到不了)的物料清单
        "skipped": skipped,
        "n_items_skipped": len(skipped),
        "pose_sequence": pose_records,
        "workspace": ws,
        "planner": pl,
        "n_points": int(n_total),
        "interpolation_dt": interp_dt,
        "total_duration_s": float(times[-1]),
        "workspace_check": viol_report,
        "joint_motion": motion_report,
        "joint_limit_margin": limit_report,
        "gripper_extent_check": gripper_report,
        "self_collision_check": self_coll_report,
        "wall_link_restriction": wall_report,
        "motiongen_pose_links": list(mg_robot_dict["robot_cfg"]["kinematics"]["link_names"]),
        "config": cfg,
    }
    npz_path, meta_path = save_trajectory(
        out_dir, joint_names, positions, velocities, accelerations,
        times, ee_pos, ee_quat, meta,
    )
    csv_path = write_trajectory_csv(
        out_dir, joint_names, times, positions, ee_pos, ee_quat
    )

    # ---------- 汇总 ----------
    print(f"\n{'=' * 88}")
    print("[SUMMARY] 各物料结果")
    print(f"{'=' * 88}")
    print(f"{'#':<4}{'row':>4}{'col':>4}  {'抓取点':<24}{'抓取角':>7}{'放置角':>7}"
          + (f"{'S2抓':>7}{'S2放':>7}" if st2_on else "")
          + f"{'点数':>7}{'时长s':>8}{'最大变化':>9}{'限位余量':>9}{'直线mm':>8}{'试组':>6}")
    print("-" * 114)
    for r in item_reports:
        if not r.get("success"):
            print(f"{r['index']:<4}{r.get('row', ''):>4}{r.get('col', ''):>4}  "
                  f"{str(np.round(r['position'], 3).tolist()):<24}{'SKIP':>7}")
            continue
        dev = r.get("max_linear_dev_mm")
        print(f"{r['index']:<4}{r['row']:>4}{r['col']:>4}  "
              f"{str(np.round(r['position'], 3).tolist()):<24}"
              f"{r['angle_grasp_deg']:>7.0f}{r['angle_place_deg']:>7.0f}"
              + (f"{r.get('angle2_grasp_deg', 0.0):>7.0f}"
                 f"{r.get('angle2_place_deg', 0.0):>7.0f}" if st2_on else "")
              + f"{r['n_points']:>7}{r['duration_s']:>8.2f}"
              f"{r['max_joint_delta_deg']:>9.1f}{r['min_limit_margin_deg']:>9.1f}"
              f"{('-' if dev is None else f'{dev:.2f}'):>8}"
              f"{r['n_angles_tried']:>6}")
    print("-" * 114)
    used = sorted({(r["angle_grasp_deg"], r["angle_place_deg"])
                   for r in item_reports if r.get("success")})
    print(f"成功 {n_ok_items}/{len(items)} 个物料, 使用的(抓取角, 放置角)组合: "
          + ", ".join(f"({g:+.0f}, {p:+.0f})" for g, p in used))
    if st2_on:
        used2 = sorted({(r.get("angle2_grasp_deg", 0.0), r.get("angle2_place_deg", 0.0))
                        for r in item_reports if r.get("success")})
        n_fb = sum(1 for r in item_reports if r.get("success")
                   and abs(r.get("angle2_grasp_deg", 0.0)) < 1e-9
                   and abs(r.get("angle2_place_deg", 0.0)) < 1e-9)
        print(f"[STAGE2] 使用的第二阶段(抓取角, 放置角)组合: "
              + ", ".join(f"({g:+.0f}, {p:+.0f})" for g, p in used2))
        print(f"[STAGE2] 其中 {n_fb}/{n_ok_items} 个物料回退到 a2=0(纯第一阶段结果)")

    # 被跳过(到不了)的点位单独列一遍 —— perimeter_only 扫边界时这就是主要结论
    if skipped:
        print(f"\n[SKIPPED] {len(skipped)}/{len(items)} 个点位所有角度都到不了, 已跳过:")
        for s in skipped:
            print(f"    #{s['index']:<3} row={s['row']} col={s['col']}  "
                  f"pos={np.round(s['position'], 4).tolist()}  "
                  f"(试了 {s['n_combos_tried']} 组角度)")
        if perim_on:
            print(f"[SKIPPED] 外围可达率 {n_ok_items}/{len(items)} = "
                  f"{n_ok_items / max(len(items), 1) * 100:.0f}%")

    # 注意: skip 模式下 failed_info 只是「最后一个被跳过的物料」的诊断详情,
    # 整条轨迹其实是完整保存的(跳过的点不入轨迹), 因此不该按「中断失败」来报。
    skip_mode = str(pp["on_fail"]["mode"]) == "skip"
    if failed_info is not None and not skip_mode:
        print(f"\n{'!' * 88}")
        print(f"[FAILED] 物料 #{failed_info['failed_item_index']} "
              f"pos={np.round(failed_info['failed_item_position'], 4).tolist()} "
              f"在 {failed_info['n_combos_tried']} 组候选中均失败")
        for c in failed_info["candidates"][:12]:
            print(f"    g={c['angle_grasp_deg']:+.0f} p={c['angle_place_deg']:+.0f}deg "
                  f"[{c.get('stage')}] {c.get('reason') or c.get('status')} "
                  f"{c.get('joint', '')} "
                  f"{('%.0fdeg' % c['max_delta_deg']) if c.get('max_delta_deg') is not None else ''} "
                  f"{('余量%.1f' % c['min_margin_deg']) if c.get('min_margin_deg') is not None else ''} "
                  f"@ {c.get('at') or c.get('failed_at', '')}")
        print(f"[FAILED] 已保存前 {n_ok_items} 个物料的轨迹, 轨迹停在失败处之前")
        print(f"{'!' * 88}")
        dump_json(failed_info, out_dir / "plan_failed.json")
    elif skipped:
        # 跳过的点位诊断详情另存一份, 方便排查「为什么这个点到不了」
        dump_json({"skipped": skipped, "last_failed_detail": failed_info},
                  out_dir / "plan_skipped.json")

    print(f"\n[SAVE] {npz_path}")
    print(f"[SAVE] {meta_path}")
    print(f"[SAVE] {csv_path}")
    if failed_info is not None and not skip_mode:
        print(f"[SAVE] {out_dir / 'plan_failed.json'}")
    elif skipped:
        print(f"[SAVE] {out_dir / 'plan_skipped.json'}")
    print(f"\n下一步在 ROS 环境播放:\n  ./run_rviz.sh --traj {out_dir}")
    # stop 模式有失败时返回非 0(轨迹已保存); skip 模式属正常完成
    return 0 if (failed_info is None or skip_mode) else 5


if __name__ == "__main__":
    sys.exit(main())
