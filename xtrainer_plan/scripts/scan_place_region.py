#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
place 位置区域扫描 —— 找「哪个 place 位置能让更远的 grasp 点可达」。

问题: place 点放在哪里最好? 放得靠近基座, 远处的料件够不到; 放得太远, 自己就
      到不了。需要在一片候选区域里逐个评价, 看哪个位置能覆盖最多的 grasp 点。

三阶段流程:
  阶段1 (stage1) grasp 位置 IK 筛选
      逐个 grasp 位置试 IK。某个角度失败就试 yaml 里配置的下一个旋转角
      (angle_search 的完整候选, 含耦合与第二阶段), 全部角度都失败则跳过该位置。
      「保留所有可行角度」而非只留第一个 —— 因为某个角度虽然 IK 可行, 组合到
      阶段3 时可能关节跳变过大, 留多个角度可让阶段3 有备选。
      产出: 每个 grasp 位置的可行角度列表 + 对应的关节解。

  阶段2 (stage2) place 位置 IK 筛选
      同上, 逐个 place 位置试 IK, 保留所有可行角度。
      阶段1/2 可以分开缓存的前提: grasp 侧路点的位姿只依赖 grasp 位置与 grasp 角,
      place 侧只依赖 place 位置与 place 角 (已实测验证, 同一 grasp 配不同 place
      的 IK 解差 0.0003 度)。因此 N_place x N_grasp 次整轮 IK 降为
      (N_place + N_grasp) 次单点 IK。

  阶段3 (stage3) 对两侧都成功的组合做轨迹规划
      只对「阶段1 和阶段2 都有可行角度」的 (grasp, place) 组合调用 plan_sequence
      (与正式规划同一函数, 含直线约束/自碰撞/墙约束)。规划失败或关节变化超过
      criterion.max_joint_delta_deg 则跳过该 grasp 点。
      阶段3 的 grasp 网格可单独放粗(stage3.grasp_region.step)以控制耗时。

排名: 先要求「三个阶段都成功」(该 place 自身 IK 可行, 且至少有一个 grasp 能
      规划成功), 再按「规划成功的 grasp 点位数」从多到少排, 取 Top-N。

一致性: robot/workspace/planner/criterion/lift/linear_move/base_rpy/angle_search
        全部继承 config/pick_place_default.yaml, 与 run_pick_place.sh 同口径。

必须在 conda curobo 环境下运行:
  ./run_place_scan.sh --stage 12    # 阶段1+2 (IK 筛选, 快), 结果存盘
  ./run_place_scan.sh --stage 3     # 阶段3 (轨迹规划, 慢), 读盘续跑
  ./run_place_scan.sh --stage all   # 依次跑完三个阶段
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
    load_robot_cfg_dict,
    make_motion_gen,
    make_world_config,
    restrict_world_collision_to_links,
    solve_ik_for_pose,
)
from plan_pick_place import (  # noqa: E402
    angle_combos,
    axis_delta,
    check_joint_delta,
    check_limit_margin,
    load_pick_place_config,
    plan_sequence,
    stage2_combos,
)
from xtrainer_common import (  # noqa: E402
    PoseSpec,
    compose_left,
    compose_right,
    dump_json,
    rpy_deg_to_quat_wxyz,
    tool_z_axis,
)

TASK_ROOT = Path(__file__).resolve().parent.parent


# ============================== 配置 ==============================


def load_scan_config(path: Optional[str] = None) -> Dict[str, Any]:
    """读扫描配置, 并把 base_config 指向的正式规划配置合并进来。"""
    import yaml

    p = Path(path) if path else TASK_ROOT / "config" / "place_scan_default.yaml"
    if not p.is_absolute():
        p = (TASK_ROOT / p).resolve()
    if not p.is_file():
        raise SystemExit(f"[ERR] 找不到配置 {p}")
    with open(p, "r", encoding="utf-8") as f:
        scan = yaml.safe_load(f) or {}

    base_rel = scan.get("base_config") or "config/pick_place_default.yaml"
    base_path = Path(base_rel)
    if not base_path.is_absolute():
        base_path = (TASK_ROOT / base_path).resolve()
    base = load_pick_place_config(str(base_path))
    scan["_base"] = base
    scan["_config_path"] = str(p)
    scan["_base_path"] = str(base_path)
    return scan


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=str, default=None,
                    help="扫描 yaml, 默认 config/place_scan_default.yaml")
    ap.add_argument("--stage", type=str, default="all",
                    choices=["12", "3", "all"],
                    help="12=阶段1+2 (grasp/place 的 IK 筛选, 快); "
                         "3=阶段3 (对两侧都可行的组合做轨迹规划, 慢); "
                         "all=依次跑完三个阶段")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="输出目录. --stage 3 必须指向已有的 stage12 输出目录")
    ap.add_argument("--no-timestamp", action="store_true")

    g = ap.add_argument_group("place 区域")
    g.add_argument("--place-x-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    g.add_argument("--place-y-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    g.add_argument("--place-z", type=float, default=None)
    g.add_argument("--place-step", type=float, default=None, help="place 网格间距 (m)")

    g = ap.add_argument_group("grasp 区域")
    g.add_argument("--grasp-x-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    g.add_argument("--grasp-y-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    g.add_argument("--grasp-z", type=float, default=None)
    g.add_argument("--grasp-step", type=float, default=None, help="第1级 grasp 网格间距 (m)")
    g.add_argument("--grasp-perimeter", dest="grasp_perimeter",
                   action="store_true", default=None,
                   help="第1级 grasp 只扫最外围一圈(默认)")
    g.add_argument("--no-grasp-perimeter", dest="grasp_perimeter",
                   action="store_false", help="第1级 grasp 扫全网格(慢很多)")
    g.add_argument("--grasp-edge", type=str, default=None,
                   choices=["all", "x_min", "x_max", "y_min", "y_max"],
                   help="只取 grasp 区域的某一条边. x_min=x 最小那条边(离基座最远的一列). "
                        "优先于 --grasp-perimeter")
    g.add_argument("--place-edge", type=str, default=None,
                   choices=["all", "x_min", "x_max", "y_min", "y_max"],
                   help="只取 place 区域的某一条边")

    g = ap.add_argument_group(
        "姿态角搜索(覆盖 pick_place.angle_search, 逻辑与正式规划完全共用)")
    g.add_argument("--grasp-angle-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="抓取角范围(度)")
    g.add_argument("--grasp-angle-step", type=float, default=None,
                   help="抓取角步长(度)")
    g.add_argument("--place-angle-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="放置角范围(度). 耦合时无效")
    g.add_argument("--place-angle-step", type=float, default=None,
                   help="放置角步长(度). 耦合时无效")
    g.add_argument("--couple-place", dest="couple_place", action="store_true",
                   default=None,
                   help="放置角由抓取角决定(相等/相反由 grasp.min_deg 符号定)")
    g.add_argument("--no-couple-place", dest="couple_place", action="store_false",
                   help="抓取角与放置角独立搜索")
    # 这里的「第二阶段旋转」= pick_place 的 angle_search.stage2, 与本脚本的
    # 「阶段1/2/3」不是一回事, 故 CLI 用 rot2 前缀避免混淆。
    g.add_argument("--rot2", dest="rot2", action="store_true", default=None,
                   help="启用第二阶段旋转(angle_search.stage2), 候选数会成倍增长")
    g.add_argument("--no-rot2", dest="rot2", action="store_false",
                   help="只做第一阶段角度搜索")
    g.add_argument("--rot2-frame", type=str, default=None,
                   choices=["base", "tool"],
                   help="第二阶段旋转的转轴坐标系: base=绕 LINK_0; tool=绕工具局部轴")
    g.add_argument("--rot2-axis", type=str, default=None, choices=["x", "y", "z"])
    g.add_argument("--rot2-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="第二阶段旋转角范围(度)")
    g.add_argument("--rot2-step", type=float, default=None,
                   help="第二阶段旋转角步长(度)")

    g = ap.add_argument_group("阶段3 与其他")
    g.add_argument("--top-n", type=int, default=None,
                   help="最终 place 排名取前 N 个(默认 10)")
    g.add_argument("--stage3-grasp-step", type=float, default=None,
                   help="阶段3 的 grasp 网格间距 (m). 建议比阶段1 粗以控制耗时")
    g.add_argument("--stage3-grasp-x-max", type=float, default=None,
                   help="阶段3 只规划 x <= 该值的 grasp 点(离基座更远的那部分). "
                        "阶段1/2 的筛选仍覆盖全区域")
    g.add_argument("--stage3-place-x-max", type=float, default=None,
                   help="阶段3 只规划 x <= 该值的 place 点")
    g.add_argument("--max-angle-trials", type=int, default=None,
                   help="阶段3 每个 grasp 点最多试几组角度(默认 3)")
    g.add_argument("--max-place", type=int, default=None,
                   help="只评估前 N 个 place(调试用)")
    g.add_argument("--max-grasp", type=int, default=None,
                   help="只评估前 N 个 grasp(调试用)")
    g.add_argument("--no-plot", action="store_true", help="不出图")
    return ap


def apply_cli(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    pr, gr = cfg["place_region"], cfg["grasp_region"]
    s3 = cfg.setdefault("stage3", {})
    if args.place_x_range is not None:
        pr["x_range"] = [float(v) for v in args.place_x_range]
    if args.place_y_range is not None:
        pr["y_range"] = [float(v) for v in args.place_y_range]
    if args.place_z is not None:
        pr["z"] = float(args.place_z)
    if args.place_step is not None:
        pr["step"] = float(args.place_step)
    if args.grasp_x_range is not None:
        gr["x_range"] = [float(v) for v in args.grasp_x_range]
    if args.grasp_y_range is not None:
        gr["y_range"] = [float(v) for v in args.grasp_y_range]
    if args.grasp_z is not None:
        gr["z"] = float(args.grasp_z)
    if args.grasp_step is not None:
        gr["step"] = float(args.grasp_step)
    if args.grasp_perimeter is not None:
        gr["perimeter_only"] = bool(args.grasp_perimeter)
    if args.grasp_edge is not None:
        gr["edge"] = args.grasp_edge
    if args.place_edge is not None:
        pr["edge"] = args.place_edge
    if args.top_n is not None:
        s3["top_n"] = int(args.top_n)
    # 角度相关的 CLI 一律写进顶层 angle 覆盖层, 由 resolve_angle_search 合并
    ang = cfg.setdefault("angle", {})
    if args.grasp_angle_range is not None:
        g = ang.setdefault("grasp", {})
        g["min_deg"], g["max_deg"] = (float(v) for v in args.grasp_angle_range)
    if args.grasp_angle_step is not None:
        ang.setdefault("grasp", {})["step_deg"] = float(args.grasp_angle_step)
    if args.place_angle_range is not None:
        p = ang.setdefault("place", {})
        p["min_deg"], p["max_deg"] = (float(v) for v in args.place_angle_range)
    if args.place_angle_step is not None:
        ang.setdefault("place", {})["step_deg"] = float(args.place_angle_step)
    if args.couple_place is not None:
        ang["couple_place_to_grasp"] = bool(args.couple_place)
    if args.rot2 is not None:
        ang.setdefault("stage2", {})["enable"] = bool(args.rot2)
    if args.rot2_frame is not None:
        ang.setdefault("stage2", {})["frame"] = args.rot2_frame
    if args.rot2_axis is not None:
        ang.setdefault("stage2", {})["axis"] = args.rot2_axis
    if args.rot2_range is not None:
        st = ang.setdefault("stage2", {})
        st["min_deg"], st["max_deg"] = (float(v) for v in args.rot2_range)
    if args.rot2_step is not None:
        ang.setdefault("stage2", {})["step_deg"] = float(args.rot2_step)
    if args.stage3_grasp_step is not None:
        s3.setdefault("grasp_region", {})["step"] = float(args.stage3_grasp_step)
    if args.stage3_grasp_x_max is not None:
        s3["grasp_x_max"] = float(args.stage3_grasp_x_max)
    if args.stage3_place_x_max is not None:
        s3["place_x_max"] = float(args.stage3_place_x_max)
    if args.max_angle_trials is not None:
        s3["max_angle_trials"] = int(args.max_angle_trials)
    return cfg


# ============================== 网格 ==============================


def grid_points(x_range, y_range, z: float, step: float,
                perimeter_only: bool = False,
                edge: str = "all") -> List[Dict[str, Any]]:
    """按 step 在 x/y 范围内布点(含端点)。返回带 (row, col) 的点列表。

    row 沿 x, col 沿 y, 与 plan_pick_place.build_grasp_points 的约定一致。

    筛选优先级: edge != 'all' 时按 edge 取单条边(此时忽略 perimeter_only);
    否则 perimeter_only 决定是否只留最外围一圈。

    edge 可选:
        all   = 不按边筛选(默认)
        x_min = 只取 x 最小的那条边(row 0), 即离基座最远的一列
        x_max = 只取 x 最大的那条边(row nx-1)
        y_min = 只取 y 最小的那条边(col 0)
        y_max = 只取 y 最大的那条边(col ny-1)
    """
    x0, x1 = (float(v) for v in x_range)
    y0, y1 = (float(v) for v in y_range)
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    st = abs(float(step)) or 0.01
    nx = max(1, int(np.floor((x1 - x0) / st + 1e-9)) + 1)
    ny = max(1, int(np.floor((y1 - y0) / st + 1e-9)) + 1)
    xs = np.array([x0]) if nx == 1 else x0 + np.arange(nx) * st
    ys = np.array([y0]) if ny == 1 else y0 + np.arange(ny) * st

    ed = str(edge or "all").lower()
    valid = ("all", "x_min", "x_max", "y_min", "y_max")
    if ed not in valid:
        raise SystemExit(f"[ERR] edge 只支持 {valid}, 收到 {edge!r}")

    def keep(i: int, j: int) -> bool:
        if ed == "x_min":
            return i == 0
        if ed == "x_max":
            return i == nx - 1
        if ed == "y_min":
            return j == 0
        if ed == "y_max":
            return j == ny - 1
        # edge='all': 回落到 perimeter_only 的规则
        if perimeter_only and nx > 2 and ny > 2:
            return i == 0 or i == nx - 1 or j == 0 or j == ny - 1
        return True

    pts: List[Dict[str, Any]] = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if not keep(i, j):
                continue
            pts.append({
                "index": len(pts), "row": i, "col": j,
                "position": [float(x), float(y), float(z)],
            })
    return pts, int(nx), int(ny)


# ============================== 路点构造 ==============================


def side_poses(
    pos: List[float], q_tool: np.ndarray, lift_z: float, lift_axis: str,
    tag: str, kind: str,
) -> List[PoseSpec]:
    """构造某一侧(抓取或放置)的 3 个路点: lift_in -> 本体 -> lift_out。

    与 plan_pick_place.make_round_poses 的单侧结构保持一致 —— 这样两边的
    IK 可行性判定口径相同。lift_z<=0 时只有本体一个点。
    """
    p = np.asarray(pos, dtype=np.float64)
    if lift_z <= 1e-9:
        return [PoseSpec(f"{tag}_{kind}", p, q_tool, kind)]
    if str(lift_axis) == "tool_z_neg":
        lifted = p - float(lift_z) * tool_z_axis(q_tool)
    else:
        lifted = p + np.array([0.0, 0.0, float(lift_z)])
    return [
        PoseSpec(f"{tag}_{kind[0]}_lift_in", lifted, q_tool, "lift"),
        PoseSpec(f"{tag}_{kind}", p, q_tool, kind),
        PoseSpec(f"{tag}_{kind[0]}_lift_out", lifted, q_tool, "lift"),
    ]


def deep_merge(base: Dict[str, Any], ov: Dict[str, Any]) -> Dict[str, Any]:
    """把 ov 递归合并进 base 的深拷贝(ov 优先)。"""
    out = copy.deepcopy(base)
    for k, v in (ov or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_angle_search(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """扫描用的角度配置。

    = 正式规划的 pick_place.angle_search  (完整继承)
      叠加 place_scan 里 stage1.angle 的覆盖项

    这样耦合符号规则(couple_sign)、遍历策略(strategy/coarse_to_fine)、
    max_trials、第二阶段(stage2 + frame) 全部与 run_pick_place.sh 同一套逻辑,
    扫描不再自己实现一份简化版, 避免两边行为漂移。
    """
    base_asr = cfg["_base"]["pick_place"]["angle_search"]
    return deep_merge(base_asr, (cfg.get("angle") or {}))


def tool_quat(base_rpy: List[float], axis: str, a1: float,
              st2_axis: str = "z", a2: float = 0.0,
              st2_frame: str = "base") -> np.ndarray:
    """两阶段合成工具姿态, 与 plan_pick_place.make_round_poses 完全一致。

        第一阶段恒为右乘(绕工具自身 axis 轴):  R1 = R_base @ R(axis, a1)
        第二阶段按 frame:
            base -> R2 = R(st2_axis, a2) @ R1   (左乘, 绕 LINK_0 的轴)
            tool -> R2 = R1 @ R(st2_axis, a2)   (右乘, 绕工具局部轴)
    """
    q1 = compose_right(rpy_deg_to_quat_wxyz(base_rpy), axis_delta(axis, a1))
    if abs(a2) < 1e-12:
        return q1
    compose2 = compose_left if str(st2_frame).lower() == "base" else compose_right
    return compose2(q1, axis_delta(st2_axis, a2))


def scan_angle_combos(asr: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:
    """扫描用的完整角度候选: (a1_grasp, a1_place, a2_grasp, a2_place)。

    直接调用 plan_pick_place 的 angle_combos / stage2_combos, 因此:
      * 第一阶段的耦合关系(相等/相反, 由 grasp.min_deg 符号决定)、
        遍历策略、max_trials 截断 —— 全部与正式规划一致
      * 第二阶段开启时, 对每组第一阶段角再套一层 stage2 候选;
        stage2_combos 的首项恒为 (0,0), 即「不做第二阶段」, 保证纯第一阶段的
        组合一定排在前面 —— 与正式规划「第二阶段全灭则回退」的语义一致。

    扫描只关心「是否存在可行角度」, 所以这里不需要正式规划里的
    reuse_last_success(那是为了加速连续物料的搜索)。
    """
    a1_list = angle_combos(asr)
    st2 = asr.get("stage2") or {}
    if not bool(st2.get("enable", False)):
        return [(g, p, 0.0, 0.0) for g, p in a1_list]
    a2_list = stage2_combos(asr)
    out: List[Tuple[float, float, float, float]] = []
    for a2g, a2p in a2_list:
        for g, p in a1_list:
            out.append((g, p, a2g, a2p))
    return out


# ============================== 阶段 1 / 2: 单侧 IK 筛选 ==============================


def screen_side(
    robot_dict, world, pl, wall_links, q_home: np.ndarray, q_lo, q_hi,
    cr: Dict[str, Any],
    pts: List[Dict[str, Any]], angles: List[Tuple[float, float]],
    base_rpy: List[float], axis: str, lift_z: float, lift_axis: str,
    kind: str, label: str, st2_axis: str = "z", st2_frame: str = "base",
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:
    """对一侧(grasp 或 place)的所有位置做 IK 筛选。

    对每个位置, 按 angles 的顺序逐个试; 某个角度 IK 失败就试下一个角度,
    全部角度都失败则该位置被「跳过」(不出现在返回的 dict 里)。

    「保留所有可行角度」: 不在找到第一个可行角就停 —— 因为该角度组合到阶段3
    时可能关节跳变过大, 多留几个角度可让阶段3 有备选。

    同时用 limit_margin 判据过滤: 贴死限位的解在阶段3 基本规划不出来, 提前剔掉。

    返回 (feasible, stats):
        feasible = {point_index: [ {angle:(a1,a2), q:(n_wp x dof)}, ... ]}
                   只含「至少有一个可行角度」的位置
        stats    = 统计信息(用于日志与结果自描述)
    """
    feasible: Dict[int, List[Dict[str, Any]]] = {}
    n_ik_call = n_ik_fail = n_margin_fail = 0
    total = len(pts) * len(angles)
    t0 = time.time()
    done = 0
    for pt in pts:
        oks: List[Dict[str, Any]] = []
        for a1, a2 in angles:
            done += 1
            q_tool = tool_quat(base_rpy, axis, a1, st2_axis, a2, st2_frame)
            seq = side_poses(pt["position"], q_tool, lift_z, lift_axis,
                             f"{label}{pt['index']}", kind)
            qs: List[np.ndarray] = []
            seed = q_home
            ok = True
            for sp in seq:
                n_ik_call += 1
                q, _ = solve_ik_for_pose(robot_dict, world, pl, sp,
                                         wall_link_names=wall_links, seed_q=seed)
                if q is None:
                    # seed 可能把解带进坏的局部域, 无 seed 再试一次
                    q, _ = solve_ik_for_pose(robot_dict, world, pl, sp,
                                             wall_link_names=wall_links)
                if q is None:
                    ok = False
                    n_ik_fail += 1
                    break
                qs.append(q)
                seed = q
            if not ok:
                continue
            arr = np.stack(qs, axis=0)
            ok_m, _, _ = check_limit_margin(arr, q_lo, q_hi, cr)
            if not ok_m:
                n_margin_fail += 1
                continue
            oks.append({"angle": (float(a1), float(a2)), "q": arr})
        if oks:
            feasible[pt["index"]] = oks
        if done % max(1, total // 20) < len(angles) or done >= total:
            el = time.time() - t0
            eta = el / max(done, 1) * (total - done)
            print(f"    [{label}] {done}/{total} 组  可行位置 {len(feasible)}  "
                  f"已用 {el:.0f}s  剩 {eta:.0f}s", flush=True)

    stats = {
        "n_points": len(pts), "n_angles": len(angles),
        "n_feasible_points": len(feasible),
        "n_skipped_points": len(pts) - len(feasible),
        "n_ik_calls": n_ik_call, "n_ik_fail": n_ik_fail,
        "n_margin_fail": n_margin_fail,
        "elapsed_s": time.time() - t0,
    }
    return feasible, stats


# ============================== 阶段 1+2: IK 筛选 ==============================


def stage12_screen(
    cfg, mg, robot_dict, world, wall_links,
    q_home, q_lo, q_hi, out_dir, args,
):
    """阶段1(grasp 侧) + 阶段2(place 侧) 的 IK 筛选。

    两侧各自独立筛选 —— 前提是该侧路点的位姿只依赖自己那一侧的位置与角度
    (已实测验证)。因此不必做 N_place x N_grasp 次整轮 IK。
    """
    base = cfg["_base"]
    pp, pl = base["pick_place"], base["planner"]
    cr, lift = pp["criterion"], pp["lift"]
    asr = resolve_angle_search(cfg)
    st2 = asr.get("stage2") or {}
    st2_on = bool(st2.get("enable", False))
    st2_axis, st2_frame = str(st2.get("axis", "z")), str(st2.get("frame", "base"))

    pr, gr = cfg["place_region"], cfg["grasp_region"]
    places, pnx, pny = grid_points(pr["x_range"], pr["y_range"], pr["z"], pr["step"],
                                   edge=str(pr.get("edge", "all")))
    grasps, gnx, gny = grid_points(gr["x_range"], gr["y_range"], gr["z"], gr["step"],
                                   bool(gr.get("perimeter_only", True)),
                                   edge=str(gr.get("edge", "all")))
    if args.max_place:
        places = places[: args.max_place]
    if args.max_grasp:
        grasps = grasps[: args.max_grasp]

    combos = scan_angle_combos(asr)
    g_angles = sorted({(c[0], c[2]) for c in combos})
    p_angles = sorted({(c[1], c[3]) for c in combos})

    p_edge, g_edge = str(pr.get("edge", "all")), str(gr.get("edge", "all"))
    print(f"\n{'=' * 92}")
    print("[STAGE 1+2] IK 筛选")
    print(f"{'=' * 92}")
    print(f"[S12] grasp 区域 x{gr['x_range']} y{gr['y_range']} z={gr['z']} "
          f"step={gr['step']} -> {gnx}x{gny}"
          + (f", 只取 {g_edge} 边 -> {len(grasps)} 个点" if g_edge != "all"
             else (f", 只扫外围一圈 -> {len(grasps)} 个点"
                   if gr.get("perimeter_only", True) else f" = {len(grasps)} 个点")))
    if g_edge == "x_min":
        print(f"[S12]   注意: x_min 是离基座最远的一列, 最容易超出臂展")
    print(f"[S12] place 区域 x{pr['x_range']} y{pr['y_range']} z={pr['z']} "
          f"step={pr['step']} -> {pnx}x{pny}"
          + (f", 只取 {p_edge} 边 -> {len(places)} 个候选" if p_edge != "all"
             else f" = {len(places)} 个候选"))

    cpl = bool(asr.get("couple_place_to_grasp", False))
    print(f"[S12] 角度候选 {len(combos)} 组 (继承 pick_place.angle_search):")
    print(f"[S12]   第一阶段 grasp 绕工具 {asr['grasp']['axis'].upper()} "
          f"[{asr['grasp']['min_deg']:g}, {asr['grasp']['max_deg']:g}] "
          f"step {asr['grasp']['step_deg']:g}")
    if cpl:
        from plan_pick_place import couple_sign
        rel = "p = g (相等)" if couple_sign(asr["grasp"]) > 0 else "p = -g (相反)"
        print(f"[S12]   耦合=开 {rel}; place 的角度范围被忽略")
    else:
        print(f"[S12]   耦合=关, place 绕工具 {asr['place']['axis'].upper()} "
              f"[{asr['place']['min_deg']:g}, {asr['place']['max_deg']:g}] "
              f"step {asr['place']['step_deg']:g}; strategy={asr.get('strategy')}")
    print(f"[S12]   第二阶段=" + (
        f"开: 绕 {'base ' + base['robot']['base_link'] if st2_frame == 'base' else '工具局部'} "
        f"{st2_axis.upper()} 轴 [{st2.get('min_deg')}, {st2.get('max_deg')}] "
        f"step {st2.get('step_deg')}" if st2_on else "关"))
    print(f"[S12]   去重后每侧: grasp {len(g_angles)} 个角组合, "
          f"place {len(p_angles)} 个角组合")
    print(f"[S12] 判据: limit_margin >= {cr.get('min_limit_margin_deg')} deg "
          f"(阶段3 再用 max_joint_delta_deg={cr.get('max_joint_delta_deg')} 过滤)")
    print(f"[S12] 单侧 IK 次数 = (grasp {len(grasps)}x{len(g_angles)} + "
          f"place {len(places)}x{len(p_angles)}) x 3 路点 = "
          f"{(len(grasps) * len(g_angles) + len(places) * len(p_angles)) * 3}")

    gz, pz = float(lift["grasp_z"]), float(lift["place_z"])
    lax = str(lift.get("axis", "base_z"))

    print(f"\n[S12] 阶段1/2: 筛选 grasp 位置 ...")
    g_ok, g_stats = screen_side(
        robot_dict, world, pl, wall_links, q_home, q_lo, q_hi, cr,
        grasps, g_angles, pp["base_rpy"]["grasp"], asr["grasp"]["axis"],
        gz, lax, "grasp", "G", st2_axis, st2_frame)
    print(f"[S12] 阶段1 完成: {g_stats['n_feasible_points']}/{len(grasps)} 个 grasp "
          f"位置可行, 跳过 {g_stats['n_skipped_points']} 个 "
          f"(用时 {g_stats['elapsed_s']:.0f}s)")

    print(f"\n[S12] 阶段2/2: 筛选 place 位置 ...")
    p_ok, p_stats = screen_side(
        robot_dict, world, pl, wall_links, q_home, q_lo, q_hi, cr,
        places, p_angles, pp["base_rpy"]["place"], asr["place"]["axis"],
        pz, lax, "place", "P", st2_axis, st2_frame)
    print(f"[S12] 阶段2 完成: {p_stats['n_feasible_points']}/{len(places)} 个 place "
          f"位置可行, 跳过 {p_stats['n_skipped_points']} 个 "
          f"(用时 {p_stats['elapsed_s']:.0f}s)")

    for lab, st, key in (("grasp", g_stats, "angle.grasp"),
                         ("place", p_stats, "angle.place")):
        if st["n_feasible_points"] == 0:
            print(f"[WARN] {lab} 侧「全部」位置 IK 无解 -> 阶段3 无可算组合。"
                  f"\n       常见原因: {key} 的角度范围不含可行角, 或该区域超出臂展。"
                  f"\n       建议放宽角度范围 / 收窄区域后重跑。")

    payload = {
        "stage": "12",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "place_grid": {"nx": pnx, "ny": pny, **pr},
        "grasp_grid": {"nx": gnx, "ny": gny, **gr, "n_points": len(grasps)},
        "angle_search": asr,
        "angle_combos": [list(t) for t in combos],
        "places": places,
        "grasps": grasps,
        # 只存可行角度列表(不存关节矩阵, 那个太大; 阶段3 会按角度重算 IK 种子)
        "grasp_feasible": {str(k): [list(v["angle"]) for v in vs]
                           for k, vs in g_ok.items()},
        "place_feasible": {str(k): [list(v["angle"]) for v in vs]
                           for k, vs in p_ok.items()},
        "stage1_stats": g_stats,
        "stage2_stats": p_stats,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "base_config_path": cfg["_base_path"],
    }
    dump_json(payload, out_dir / "stage12_result.json")
    print(f"\n[S12] 已保存 {out_dir / 'stage12_result.json'}")
    # 说明: 这里不保存 IK 关节解。阶段3 调用的 plan_sequence 会自行做完整规划
    # (含它自己的 IK/trajopt), 外部塞进去的关节解用不上, 存了只是浪费磁盘。
    # 阶段1/2 传给阶段3 的有效信息就是「哪些位置 + 哪些角度可行」。
    return payload


# ============================== 阶段 3: 轨迹规划 ==============================


def stage3_plan(
    cfg, s12, mg, robot_dict, world, wall_links,
    q_home, q_lo, q_hi, out_dir, args,
):
    """阶段3: 对「阶段1 和阶段2 都可行」的 (grasp, place) 组合做轨迹规划。

    对每个 place, 遍历所有可行 grasp 点; 每个 grasp 点按「两侧可行角度的交集
    候选」逐组试规划, 规划成功且关节变化未超限即计入该 place 的成功数。
    规划失败或关节变化过大则跳过这个 grasp 点。

    grasp 网格可通过 stage3.grasp_region.step 单独放粗以控制耗时
    (阶段1 用细网格找边界, 阶段3 用粗网格验证)。
    """
    base = cfg["_base"]
    pp, pl = base["pick_place"], base["planner"]
    cr, lift, lin = pp["criterion"], pp["lift"], pp.get("linear_move") or {}
    s3 = cfg.get("stage3") or {}
    asr = resolve_angle_search(cfg)
    st2 = asr.get("stage2") or {}
    st2_axis, st2_frame = str(st2.get("axis", "z")), str(st2.get("frame", "base"))
    ee_link = base["robot"]["ee_link"]
    gz, pz = float(lift["grasp_z"]), float(lift["place_z"])
    lax = str(lift.get("axis", "base_z"))

    # 阶段3 的 grasp 网格(默认放粗)。落在阶段1 可行集之外的点直接不算。
    gr3 = dict(cfg["grasp_region"])
    gr3.update(s3.get("grasp_region") or {})
    grasps3, gnx, gny = grid_points(gr3["x_range"], gr3["y_range"], gr3["z"],
                                    gr3["step"],
                                    bool(gr3.get("perimeter_only", False)),
                                    edge=str(gr3.get("edge", "all")))
    if args.max_grasp:
        grasps3 = grasps3[: args.max_grasp]

    g_feas = {int(k): [tuple(a) for a in v]
              for k, v in s12["grasp_feasible"].items()}
    p_feas = {int(k): [tuple(a) for a in v]
              for k, v in s12["place_feasible"].items()}
    places = s12["places"]
    grasps1 = s12["grasps"]
    if args.max_place:
        places = places[: args.max_place]

    # ---------- 阶段3 专属的 x 上限过滤 ----------
    # 只对「x 足够小(足够远离基座)」的点做规划。用途: 阶段1/2 已经把整片区域都
    # 筛过一遍, 但真正关心的往往只是远端那部分; 在这里再收一刀可大幅缩短
    # 阶段3 的耗时, 且不影响阶段1/2 已有的筛选结果(它们仍是全区域的)。
    # null / 不配置 = 不过滤。
    gx_max = s3.get("grasp_x_max")
    px_max = s3.get("place_x_max")
    n_g_before, n_p_before = len(grasps3), len(places)
    if gx_max is not None:
        grasps3 = [g for g in grasps3 if g["position"][0] <= float(gx_max) + 1e-9]
    if px_max is not None:
        places = [p for p in places if p["position"][0] <= float(px_max) + 1e-9]

    # 阶段3 用了不同的 grasp 网格, 需要把阶段1 的可行角度按「最近点」映射过来。
    # 容差取阶段1 网格间距的 0.75 倍; 超出则认为阶段1 没覆盖该点。
    g1_pos = np.array([g["position"][:2] for g in grasps1]) if grasps1 else np.zeros((0, 2))
    g1_idx = [g["index"] for g in grasps1]
    tol = float(s12["grasp_grid"].get("step", 0.01)) * 0.75
    all_g_angles = sorted({(c[0], c[2]) for c in
                           (tuple(x) for x in s12["angle_combos"])})

    def grasp_angles_for(pos) -> Tuple[List[Tuple[float, float]], bool]:
        """返回 (可行角度列表, 是否命中阶段1 的筛选结果)。

        未命中(阶段1 没算过这个点)时退回全部候选角度 —— 宁可多试也不漏解,
        但这会让阶段1 的加速对该点失效, 故用第二个返回值供上层统计告警。
        """
        if len(g1_pos) == 0:
            return all_g_angles, False
        d = np.linalg.norm(g1_pos - np.asarray(pos[:2]), axis=1)
        k = int(np.argmin(d))
        if d[k] > tol:
            return all_g_angles, False
        # 命中: 空列表 = 阶段1 明确判定该点不可行, 阶段3 直接跳过
        return g_feas.get(g1_idx[k], []), True

    max_try = max(1, int(s3.get("max_angle_trials", 3)))
    place_order = [p for p in places if p["index"] in p_feas]

    print(f"\n{'=' * 92}")
    print("[STAGE 3] 轨迹规划")
    print(f"{'=' * 92}")
    g_edge3 = str(gr3.get("edge", "all"))
    print(f"[S3] grasp 网格 {gnx}x{gny} step={gr3['step']}"
          + (f", 只取 {g_edge3} 边" if g_edge3 != "all"
             else (", 只扫外围" if gr3.get("perimeter_only") else " (全网格, 实心图)"))
          + f" -> {n_g_before} 个点")
    # x 上限过滤的效果单独报一行, 免得误以为网格算错了
    if gx_max is not None or px_max is not None:
        print(f"[S3] x 上限过滤(仅阶段3 生效, 阶段1/2 仍是全区域):")
        if gx_max is not None:
            print(f"[S3]   grasp x <= {float(gx_max):g} -> "
                  f"{n_g_before} 保留 {len(grasps3)} 个 "
                  f"(滤掉 {n_g_before - len(grasps3)} 个离基座较近的)")
        if px_max is not None:
            print(f"[S3]   place x <= {float(px_max):g} -> "
                  f"{n_p_before} 保留 {len(places)} 个 "
                  f"(滤掉 {n_p_before - len(places)} 个)")
        if not grasps3 or not places:
            print(f"[FAIL] x 上限过滤后 grasp({len(grasps3)}) 或 place({len(places)}) "
                  f"为空, 无可规划组合。")
            print(f"       grasp x 范围 {gr3['x_range']}, "
                  f"place x 范围 {cfg['place_region']['x_range']}; "
                  f"请放宽 stage3.grasp_x_max / place_x_max")
            return {"stage": 3, "results": [], "ranking": [],
                    "top_place_indices": [], "n_all_stages_ok": 0,
                    "grasp_grid": {"nx": gnx, "ny": gny, **gr3,
                                   "n_points": 0}, "grasps": []}
    print(f"[S3] place: 阶段2 可行的 {len(place_order)}/{len(places)} 个")
    print(f"[S3] 每个 grasp 点最多试 {max_try} 组角度 "
          f"(取两侧可行角度的组合, 优先小角度)")
    print(f"[S3] 关节变化判据 max_joint_delta_deg="
          f"{cr.get('max_joint_delta_deg')} (joints={cr.get('joints')}), "
          f"超限即跳过该 grasp 点")
    n_pairs = len(place_order) * len(grasps3)
    # 覆盖率体检: 阶段1 若只扫了外围, 阶段3 的内部点在阶段1 没有数据, 只能
    # 「退回全部候选角度」—— 等于阶段1 的筛选对这些点没起作用, 阶段3 会多跑
    # 很多注定失败的 trajopt。这里显式告警并给出可执行建议。
    n_mapped = sum(1 for g in grasps3 if grasp_angles_for(g["position"])[1])
    if n_mapped < len(grasps3):
        print(f"[WARN] 阶段3 的 {len(grasps3)} 个 grasp 点中, 只有 {n_mapped} 个能"
              f"映射到阶段1 的筛选结果")
        print(f"       其余 {len(grasps3) - n_mapped} 个点阶段1 没算过"
              f"(阶段1 perimeter_only={s12['grasp_grid'].get('perimeter_only')}, "
              f"step={s12['grasp_grid'].get('step')}; "
              f"阶段3 perimeter_only={gr3.get('perimeter_only')}, "
              f"step={gr3.get('step')}), 将退回全部 {len(all_g_angles)} 个候选角度,"
              f" 阶段1 的加速效果对它们无效")
        print(f"       建议: 让阶段1 覆盖阶段3 的点 —— 要么阶段1 也用全网格"
              f"(--no-grasp-perimeter), 要么把两者 step 设成一致")
    est_h = n_pairs * 8.0 / 3600
    print(f"[S3] 最多 {n_pairs} 个组合待规划, 最坏耗时约 {est_h:.1f} 小时"
          f"(按单次规划 8s 估; 两侧无可行角的组合会被直接跳过, 实际更快)")

    out: List[Dict[str, Any]] = []
    t_all = time.time()
    for k, pl_pt in enumerate(place_order):
        pidx = pl_pt["index"]
        p_angs = p_feas.get(pidx, [])
        succ: List[int] = []
        detail: List[Dict[str, Any]] = []
        t0 = time.time()
        n_plan = n_skip_no_angle = n_fail_plan = n_fail_delta = 0

        for gi, g_pt in enumerate(grasps3):
            g_angs, _hit = grasp_angles_for(g_pt["position"])
            if not g_angs or not p_angs:
                # 阶段1 或阶段2 判定不可行 -> 跳过, 不消耗 trajopt
                n_skip_no_angle += 1
                detail.append({"grasp_index": g_pt["index"], "ok": False,
                               "reason": "no_feasible_ik"})
                continue
            # 两侧可行角度的组合, 按 |a1|+|a2| 小的优先(贴近基准姿态)
            trials = [(ga, gb, pa, pb) for (ga, gb) in g_angs for (pa, pb) in p_angs]
            trials.sort(key=lambda t: (abs(t[0]) + abs(t[2]),
                                       abs(t[1]) + abs(t[3])))
            trials = trials[:max_try]

            ok_any = False
            info: Dict[str, Any] = {}
            for a1g, a2g, a1p, a2p in trials:
                q_g = tool_quat(pp["base_rpy"]["grasp"], asr["grasp"]["axis"],
                                a1g, st2_axis, a2g, st2_frame)
                q_p = tool_quat(pp["base_rpy"]["place"], asr["place"]["axis"],
                                a1p, st2_axis, a2p, st2_frame)
                seq = (side_poses(g_pt["position"], q_g, gz, lax, f"s3_{gi}", "grasp")
                       + side_poses(pl_pt["position"], q_p, pz, lax, f"s3_{gi}", "place"))
                n_plan += 1
                ok, res = plan_sequence(mg, q_home, seq, pl, cr, q_lo, q_hi,
                                        lin, ee_link)
                if ok:
                    # plan_sequence 内部已按 criterion 校验过关节变化,
                    # 这里再取出实测值用于报告
                    worst = max(x["max_joint_delta_deg"] for x in res["segments"])
                    ok_any = True
                    info = {"angle": [a1g, a1p, a2g, a2p],
                            "n_points": int(res["position"].shape[0]),
                            "max_joint_delta_deg": worst}
                    break
                st = str(res.get("status") or "")
                if "joint" in st.lower() or "delta" in st.lower():
                    n_fail_delta += 1
                else:
                    n_fail_plan += 1
                info = {"angle": [a1g, a1p, a2g, a2p], "status": st,
                        "failed_at": res.get("failed_at")}
            if ok_any:
                succ.append(g_pt["index"])
            detail.append({"grasp_index": g_pt["index"], "ok": ok_any, **info})
            if (gi + 1) % max(1, len(grasps3) // 8) == 0:
                el = time.time() - t0
                print(f"[S3]   place#{pidx} {gi + 1}/{len(grasps3)}  "
                      f"成功 {len(succ)}  已用 {el:.0f}s", flush=True)

        sset = set(succ)
        gp = np.array([g["position"] for g in grasps3 if g["index"] in sset]) \
            if succ else np.zeros((0, 3))
        rec = {
            "place_index": pidx, "position": pl_pt["position"],
            "row": pl_pt.get("row"), "col": pl_pt.get("col"),
            # 三阶段都成功: place 自身 IK 可行(能进到这里) 且 至少一个 grasp 规划成功
            "all_stages_ok": len(succ) > 0,
            "n_success": len(succ), "n_grasp_total": len(grasps3),
            "success_ratio": len(succ) / max(len(grasps3), 1),
            "max_dist": float(np.linalg.norm(gp[:, :2], axis=1).max()) if len(gp) else 0.0,
            "max_x_span": float(gp[:, 0].max() - gp[:, 0].min()) if len(gp) else 0.0,
            "success_indices": succ,
            "n_place_feasible_angles": len(p_angs),
            "n_plan_calls": n_plan,
            "n_skip_no_ik": n_skip_no_angle,
            "n_fail_plan": n_fail_plan,
            "n_fail_joint_delta": n_fail_delta,
            "detail": detail,
            "elapsed_s": time.time() - t0,
        }
        out.append(rec)
        print(f"[S3] place {k + 1}/{len(place_order)} #{pidx} "
              f"pos={np.round(pl_pt['position'], 4).tolist()}: "
              f"成功 {len(succ)}/{len(grasps3)} "
              f"({rec['success_ratio'] * 100:.0f}%), "
              f"跳过(无IK) {n_skip_no_angle}, 规划失败 {n_fail_plan}, "
              f"用时 {rec['elapsed_s']:.0f}s")
        # 每个 place 跑完存一次, 中途打断不丢结果
        dump_json({"stage": 3,
                   "grasp_grid": {"nx": gnx, "ny": gny, **gr3,
                                  "n_points": len(grasps3)},
                   "grasps": grasps3, "results": out},
                  out_dir / "stage3_result.json")

    # 排名: 先要求三阶段都成功, 再按成功 grasp 点数从多到少
    top_n = int((cfg.get("stage3") or {}).get("top_n")
                or cfg.get("top_n") or 10)
    ok_list = [r for r in out if r["all_stages_ok"]]
    ranked = sorted(ok_list, key=lambda r: (-r["n_success"], -r["max_dist"],
                                            r["place_index"]))
    print(f"\n[S3] 全部完成, 累计 {time.time() - t_all:.0f}s")
    print(f"[S3] 三阶段全部成功的 place: {len(ok_list)}/{len(out)}")
    print(f"\n[S3] Top-{min(top_n, len(ranked))} (先要求三阶段都成功, "
          f"再按成功 grasp 点数排序):")
    print(f"    {'名次':<5}{'#':<5}{'place 位置':<26}{'成功grasp':>10}"
          f"{'占比':>8}{'最远m':>9}{'x跨度m':>9}{'用时s':>8}")
    print("    " + "-" * 82)
    for i, r in enumerate(ranked[:top_n]):
        print(f"    {i + 1:<5}{r['place_index']:<5}"
              f"{str(np.round(r['position'], 3).tolist()):<26}"
              f"{r['n_success']:>10}{r['success_ratio'] * 100:>7.1f}%"
              f"{r['max_dist']:>9.3f}{r['max_x_span']:>9.3f}"
              f"{r['elapsed_s']:>8.0f}")

    payload = {
        "stage": 3,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "grasp_grid": {"nx": gnx, "ny": gny, **gr3, "n_points": len(grasps3)},
        "grasps": grasps3,
        "results": out,
        "top_n": top_n,
        "n_all_stages_ok": len(ok_list),
        # 阶段3 专属的 x 上限过滤(阶段1/2 不受影响), 记录下来便于解读图与复现
        "x_filter": {
            "grasp_x_max": gx_max, "place_x_max": px_max,
            "n_grasp_before": n_g_before, "n_grasp_after": len(grasps3),
            "n_place_before": n_p_before, "n_place_after": len(places),
        },
        "ranking": [r["place_index"] for r in ranked],
        "top_place_indices": [r["place_index"] for r in ranked[:top_n]],
    }
    dump_json(payload, out_dir / "stage3_result.json")
    print(f"\n[S3] 已保存 {out_dir / 'stage3_result.json'}")
    return payload


# ============================== 主流程 ==============================


def main() -> int:
    args = build_argparser().parse_args()
    cfg = apply_cli(load_scan_config(args.config), args)
    base = cfg["_base"]
    rb, ws, pl, pp = base["robot"], base["workspace"], base["planner"], base["pick_place"]

    print(f"[CFG] 扫描配置 {cfg['_config_path']}")
    print(f"[CFG] 继承基础配置 {cfg['_base_path']}")

    out_dir = Path(args.out_dir) if args.out_dir else \
        TASK_ROOT / "results_place_scan"
    if not out_dir.is_absolute():
        out_dir = (TASK_ROOT / out_dir).resolve()
    if args.out_dir is None and not args.no_timestamp:
        out_dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] {out_dir}")

    world = make_world_config(ws)
    robot_dict = load_robot_cfg_dict(rb)
    mg = make_motion_gen(robot_dict, world, pl)
    wall_cfg = ws.get("wall") or {}
    wall_links: List[str] = []
    if wall_cfg.get("enable", False) and wall_cfg.get("collision_link_names"):
        wall_links = list(wall_cfg["collision_link_names"])
        restrict_world_collision_to_links(mg, wall_links)
    jl = mg.kinematics.get_joint_limits().position
    q_lo = jl[0].cpu().numpy().astype(np.float64)
    q_hi = jl[1].cpu().numpy().astype(np.float64)

    # home 关节角: 与正式规划一致
    home = pp["home"]
    if home.get("joint_deg"):
        q_home = np.radians(np.asarray(home["joint_deg"], dtype=np.float64)).reshape(-1)
    else:
        hp = PoseSpec("home", np.asarray(home["position"], dtype=np.float64),
                      rpy_deg_to_quat_wxyz(home["rpy_deg"]), "home")
        q_home, _ = solve_ik_for_pose(robot_dict, world, pl, hp,
                                      wall_link_names=wall_links)
        if q_home is None:
            print("[FAIL] home 位姿 IK 无解")
            return 2
    print(f"[HOME] q = {np.degrees(q_home).round(2).tolist()}")

    s12res: Optional[Dict[str, Any]] = None
    if args.stage in ("12", "all"):
        s12res = stage12_screen(cfg, mg, robot_dict, world, wall_links,
                                q_home, q_lo, q_hi, out_dir, args)
    if args.stage in ("3", "all"):
        if s12res is None:
            p = out_dir / "stage12_result.json"
            if not p.is_file():
                print(f"[FAIL] 找不到 {p}"
                      f"\n       阶段3 需要阶段1+2 的筛选结果。请先跑 --stage 12,"
                      f"\n       或用 --out-dir 指向已有的 stage12 输出目录。")
                return 3
            with open(p, "r", encoding="utf-8") as f:
                s12res = json.load(f)
            print(f"[S3] 读取已有阶段1+2 结果 {p}")
        stage3_plan(cfg, s12res, mg, robot_dict, world, wall_links,
                    q_home, q_lo, q_hi, out_dir, args)

    if not args.no_plot:
        print("\n[PLOT] 出图 ...")
        import subprocess
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "plot_place_scan.py"),
             str(out_dir)], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip()[-800:])

    print(f"\n[DONE] 结果目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
