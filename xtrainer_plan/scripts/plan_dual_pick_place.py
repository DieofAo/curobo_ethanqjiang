#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XTrainer 独立双臂交错 pick & place 联合规划。

这不是把一条 6-DOF 轨迹复制给另一条臂。每个离散端点先由两条 6-DOF
链分别枚举 IK 分支，再做 Cartesian 组合，并由 12-DOF 模型过滤关节限位、
自碰/臂间碰撞与世界碰撞。普通重叠段把选中的 12-DOF 关节目标交给联合
MotionGen；带 hold/直线 Cartesian metric 的段仍做双 TCP 位姿规划。因此
连续轨迹始终在同一个 horizon 内检查/优化两臂的碰撞球。

应用层用一个明确的连续流水节拍做 CuRobo 本身没有提供的任务级排程：

  tick:  0  1  2  3  4  5  6  7  8  9 ...
  arm1: A0 A0 A0 A0 A0 A0 A1 A1 A1 A1 ...
  arm2: -- -- -- B0 B0 B0 B0 B0 B0 B1 ...

默认 arm2 滞后 3 个工艺路点启动。arm1 完成一个物料的第 6 个路点后
立即进入下一个抓取上方，arm2 完成后也立即循环；物料轮次之间没有
同步屏障。每个联合段都从上段的 12-DOF 末状态继续，因此输出是一条
同时间轴的联合轨迹。

默认只生成一份 arm1/LINK_0 世界网格：两臂从各自靠近的一端向中间处理，
同一 source 不重复分配。姿态搜索、工作空间与关节判据沿用同一配置。
arm2 的 place 直接表达在 arm1/LINK_0 系；默认实时复制 arm1 place 的
y/z，仅把 x 改成配置的 ``second_place_x``。共享布局中两臂的实际 Home
是各自最终首件的
g_lift_in 预抓取位姿；arm2 完成后在同一联合时间线回到该 Home 并保持。
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_scene_urdf import pose_to_mat  # noqa: E402
from check_dual_arm_collision import check_pair_collisions  # noqa: E402
from export_dual_ik_results import (  # noqa: E402
    build_ik_playback_data,
    write_ik_playback_artifact,
)
from plan_pick_place import (  # noqa: E402
    angle_combos,
    apply_cli,
    build_argparser,
    build_grasp_points,
    load_pick_place_config,
    make_round_poses,
    stage2_combos,
)
from plan_trajectory import (  # noqa: E402
    compute_fk,
    compute_joint_limit_margin_report,
    compute_joint_motion_report,
    load_robot_cfg_dict,
    make_hold_axis_metric,
    make_motion_gen,
    make_world_config,
    restrict_world_collision_to_links,
)
from xtrainer_common import (  # noqa: E402
    PoseSpec,
    check_in_bounds,
    deep_update,
    dump_json,
    matrix_to_quat_wxyz,
    normalize_link0_target_transform_config_layer,
    parse_rigid_transform_matrix,
    quat_angle_deg,
    quat_wxyz_to_matrix,
    resolve_repo_path,
    save_trajectory,
)

TASK_ROOT = Path(__file__).resolve().parents[1]
DUAL_DEFAULT = TASK_ROOT / "config" / "dual_pick_place_default.yaml"


@dataclass(frozen=True)
class AngleChoice:
    """一个物料的两阶段姿态搜索结果。"""

    grasp: float
    place: float
    stage2_grasp: float = 0.0
    stage2_place: float = 0.0

    @property
    def primary(self) -> Tuple[float, float]:
        return self.grasp, self.place

    def to_dict(self) -> Dict[str, float]:
        return {
            "angle_grasp_deg": self.grasp,
            "angle_place_deg": self.place,
            "angle2_grasp_deg": self.stage2_grasp,
            "angle2_place_deg": self.stage2_place,
        }


@dataclass
class FirstPregraspHomeRoot:
    """一个首件姿态组合下的一条独立 IK 安全组合 Home 分支。"""

    q_home: np.ndarray
    arm1_home: PoseSpec
    arm2_home: PoseSpec
    arm1_choice: AngleChoice
    arm2_choice: Optional[AngleChoice]
    angle_pair_trial: int
    ik_branch_index: int
    distance_to_seed: float

    def to_dict(self, include_joint_state: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "angle_pair_trial": int(self.angle_pair_trial),
            "ik_branch_index": int(self.ik_branch_index),
            "distance_to_seed": float(self.distance_to_seed),
            "arm1_home_grasp_deg": float(self.arm1_choice.grasp),
            "arm2_home_grasp_deg": (
                None if self.arm2_choice is None
                else float(self.arm2_choice.grasp)
            ),
            "choice_scope": (
                "Home 位姿只锁定 grasp 角；choice.place 仅为候选排序"
                "上下文，最终 place 角以 pipeline choice 为准"
            ),
            "arm1_choice": self.arm1_choice.to_dict(),
            "arm2_choice": (
                None if self.arm2_choice is None else self.arm2_choice.to_dict()
            ),
            "arm1_home": self.arm1_home.to_dict(),
            "arm2_home": self.arm2_home.to_dict(),
        }
        if include_joint_state:
            out["home_joint_deg"] = np.degrees(
                np.asarray(self.q_home, dtype=np.float64)
            ).tolist()
        return out


@dataclass
class SeparateArmIkContext:
    """两条 6-DOF IK 链及其 12-DOF 组合安全检查上下文。"""

    arm_solvers: Tuple[Any, Any]
    root_to_arm: Tuple[np.ndarray, np.ndarray]
    q_lo: np.ndarray
    q_hi: np.ndarray
    collision_filter: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, Any]]]
    max_candidates_per_arm: int = 8
    max_pair_trials: int = 64
    max_goal_solutions: int = 4
    collision_margin_mm: float = 0.0
    # 强引用独立的 12-DOF 检查器，避免闭包之外被释放；它绝不能是
    # MotionGen 自带的 IK/rollout，否则变批次验碰会破坏 CUDA Graph 缓存。
    collision_probe: Any = None


@dataclass
class SeparateArmIkResult:
    """兼容原调用方 ``result.status`` 访问的独立 IK 组合结果。"""

    status: str
    arm_results: Tuple[Any, Any]
    report: Dict[str, Any]


@dataclass
class PlannedRound:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    dt: float
    q_end: np.ndarray
    segments: List[Dict[str, Any]]
    arm1_choice: Optional[AngleChoice]
    arm2_choice: Optional[AngleChoice]
    delay_stages: int
    arm1_sequence: List[PoseSpec]
    arm2_sequence: List[PoseSpec]
    # 某条臂完成 terminal Home 后，后续所有 chunk 都相对此固定关节参考
    # 做累计漂移检查，不能在每段起点悄悄刷新参考。
    parked_joint_refs: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
    # 该 chunk 在 MotionGen 之前由两臂独立 IK 组合出的离散构型。只用于
    # RViz 诊断；相邻 IK 点之间不是已规划、已验碰的运动。
    ik_trace: Optional[Dict[str, Any]] = None


@dataclass
class PlannedPipeline:
    """一条已经完整通过搜索的连续双臂流水线。"""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    dt: float
    q_end: np.ndarray
    segments: List[Dict[str, Any]]
    chunks: List[PlannedRound]
    arm1_choices: List[AngleChoice]
    arm2_choices: List[AngleChoice]
    arm1_sequences: List[List[PoseSpec]]
    arm2_sequences: List[List[PoseSpec]]
    angle_stage: str
    search_nodes: int
    block_attempts: Dict[str, int]
    arm2_terminal: Optional[PoseSpec] = None
    parked_joint_refs: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)


def load_dual_config(path: Optional[str]) -> Dict[str, Any]:
    """加载单臂默认 -> 双臂默认 -> 用户差异配置。"""
    cfg = normalize_link0_target_transform_config_layer(
        load_pick_place_config(None),
        str(TASK_ROOT / "config" / "pick_place_default.yaml"),
    )
    with open(DUAL_DEFAULT, "r", encoding="utf-8") as f:
        dual_layer = normalize_link0_target_transform_config_layer(
            yaml.safe_load(f) or {}, str(DUAL_DEFAULT)
        )
        cfg = deep_update(cfg, dual_layer)
    if path is None:
        return cfg
    p = Path(path)
    if not p.is_absolute():
        for candidate in (Path.cwd() / p, TASK_ROOT / p, TASK_ROOT / "config" / p):
            if candidate.exists():
                p = candidate
                break
    if not p.exists():
        raise FileNotFoundError(f"config not found: {path}")
    if p.resolve() != DUAL_DEFAULT.resolve():
        with open(p, "r", encoding="utf-8") as f:
            user_layer = normalize_link0_target_transform_config_layer(
                yaml.safe_load(f) or {}, str(p)
            )
            cfg = deep_update(cfg, user_layer)
    return cfg


def dual_argparser() -> argparse.ArgumentParser:
    ap = build_argparser()
    ap.description = "XTrainer 独立 12-DOF 双臂交错 pick & place 联合规划"
    g = ap.add_argument_group("双臂联合排程")
    g.add_argument("--arm2-place-position", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="二号臂 place，一号臂 LINK_0 系")
    g.add_argument("--mounts", type=str, default=None,
                   help="双臂挂载配置，默认同侧 cad_mounts_same_side.yaml")
    g.add_argument("--start-delay-stages", type=int, default=None,
                   help="连续流水中二号臂比一号臂固定滞后的工艺路点数")
    g.add_argument("--max-start-delay-stages", type=int, default=None,
                   help="兼容旧配置；连续流水不会动态改变固定相位")
    g.add_argument("--max-pair-angle-trials", type=int, default=None,
                   help="给定上游状态时每次流水 block 的角度对尝试上限，0=不限")
    g.add_argument(
        "--no-ik-jump-check",
        dest="prescreen_joint_delta_check",
        action="store_false",
        default=None,
        help=(
            "仅关闭 block 独立 IK 组合预筛中的关节跳变拒绝；"
            "完整 MotionGen 轨迹仍执行 max-joint-delta 验收"
        ),
    )
    fallback = g.add_mutually_exclusive_group()
    fallback.add_argument(
        "--sequential-fallback", dest="sequential_fallback", action="store_true",
        default=None, help="旧版选项；连续跨物料流水会拒绝该降级模式"
    )
    fallback.add_argument(
        "--no-sequential-fallback", dest="sequential_fallback", action="store_false",
        help=argparse.SUPPRESS,
    )
    return ap


def apply_dual_cli(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = apply_cli(cfg, args)
    dc = cfg.setdefault("dual_arm", {})
    rb = cfg["robot"]
    if args.arm2_place_position is not None:
        dc["second_place_position"] = list(args.arm2_place_position)
    if args.mounts is not None:
        rb["mounts"] = args.mounts
    if args.start_delay_stages is not None:
        dc["start_delay_stages"] = int(args.start_delay_stages)
    if args.max_start_delay_stages is not None:
        dc["max_start_delay_stages"] = int(args.max_start_delay_stages)
    if args.max_pair_angle_trials is not None:
        dc["max_pair_angle_trials"] = int(args.max_pair_angle_trials)
    if args.prescreen_joint_delta_check is not None:
        cfg["pick_place"]["criterion"]["prescreen_joint_delta_check"] = bool(
            args.prescreen_joint_delta_check
        )
    if args.sequential_fallback is not None:
        dc["sequential_fallback"] = bool(args.sequential_fallback)
    return cfg


# ============================== 坐标变换 ==============================


def transform_pose(spec: PoseSpec, transform: np.ndarray, name_prefix: str = "") -> PoseSpec:
    """PoseSpec 左乘齐次变换；输出与联合 robot base_link 同系。"""
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    pos = t[:3, :3] @ spec.position + t[:3, 3]
    rot = t[:3, :3] @ quat_wxyz_to_matrix(spec.quat_wxyz)
    return PoseSpec(name_prefix + spec.name, pos, matrix_to_quat_wxyz(rot), spec.kind)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return p @ t[:3, :3].T + t[:3, 3]


def arm_target_transforms(
    link0_target_transform: np.ndarray,
    arm2_in_arm1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回两臂“本地目标 -> 联合 root”的最终变换。

    ``link0_target_transform`` 的语义固定为对当前已经构造出的联合
    ``LINK_0`` 目标做公共左乘：

    ``T_goal_effective = T_link0 @ T_goal_current``。

    因而一号臂原来的 ``I`` 变为 ``T_link0``，二号臂原来的 ``T12``
    变为 ``T_link0 @ T12``。这里绝不能写成 ``T12 @ T_link0``；后者是
    在二号臂自己的局部基座中先做补偿，语义不同。
    """
    correction = parse_rigid_transform_matrix(
        link0_target_transform, "pick_place.link0_target_transform"
    )
    t12 = parse_rigid_transform_matrix(
        arm2_in_arm1, "robot.arm2_in_arm1_transform"
    )
    return correction.copy(), correction @ t12


def transformed_cardinal_axis(
    transform: np.ndarray,
    axis: str,
    *,
    atol: float = 1e-7,
) -> str:
    """返回刚体变换后仍与 root 笛卡尔轴重合的轴名（忽略正负号）。

    CuRobo 当前的 ``hold_partial_pose`` 只能释放 root 的 x/y/z 分量，
    不能表达任意斜轴。该 helper 用于在启用直线插拔时安全地映射轴；若
    旋转后的方向不是笛卡尔轴，调用方必须 fail closed。
    """
    axis_name = str(axis).lower()
    source_index = {"x": 0, "y": 1, "z": 2}.get(axis_name)
    if source_index is None:
        raise ValueError(f"free_axis 只支持 x/y/z，实际 {axis!r}")
    rigid = parse_rigid_transform_matrix(transform, "target transform")
    direction = rigid[:3, source_index]
    target_index = int(np.argmax(np.abs(direction)))
    expected = np.zeros(3, dtype=np.float64)
    expected[target_index] = (
        1.0 if direction[target_index] >= 0.0 else -1.0
    )
    if not np.allclose(direction, expected, atol=atol, rtol=0.0):
        raise ValueError(
            f"{axis_name} 轴变换后为 {np.round(direction, 9).tolist()}，"
            "不是联合 root 的 x/y/z 轴"
        )
    return ("x", "y", "z")[target_index]


def allocate_shared_grid_outside_in(
    items: Sequence[Dict[str, Any]],
    arm2_in_arm1: np.ndarray,
    max_items: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把一份世界坐标网格去重分给两臂，并从两端向中间排序。

    输入 ``position`` 统一解释为 arm1/LINK_0（联合 root）坐标。几何 x
    中线归一号臂；一号臂按 root x 降序，二号臂按 root x 升序。显式
    ``max_items`` 是两臂合计额度，奇数时一号臂多一件。
    """
    source = list(items)
    if not source:
        return [], []
    if max_items is not None and int(max_items) < 0:
        raise ValueError("max_items 不能为负数")
    t12 = np.asarray(arm2_in_arm1, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(t12).all():
        raise ValueError("arm2_in_arm1 含 NaN/Inf")
    t21 = np.linalg.inv(t12)

    roots: List[np.ndarray] = []
    for index, item in enumerate(source):
        root = np.asarray(item.get("root_position", item.get("position")), dtype=np.float64)
        if root.shape != (3,) or not np.isfinite(root).all():
            raise ValueError(f"共享网格 item {index} 的 position 非法")
        roots.append(root)
    # 分区线属于机器人安装几何，不应随用户框选的网格范围漂移。联合 root
    # 就是一号臂基座 x=0；二号臂基座为 t12 的平移，因此二者 x 中线如下。
    split_x = 0.5 * float(t12[0, 3])

    arm1: List[Dict[str, Any]] = []
    arm2: List[Dict[str, Any]] = []
    for item, root in zip(source, roots):
        record = dict(item)
        record["root_position"] = root.tolist()
        if root[0] >= split_x:
            record["position"] = root.tolist()
            arm1.append(record)
        else:
            record["position"] = transform_points(root[None, :], t21)[0].tolist()
            arm2.append(record)

    # sorted 是稳定排序，同一 x 内保留原 grid 顺序。
    arm1.sort(key=lambda item: -float(item["root_position"][0]))
    arm2.sort(key=lambda item: float(item["root_position"][0]))
    if max_items is not None:
        total = int(max_items)
        arm1_quota = (total + 1) // 2
        arm2_quota = total // 2
        arm1 = arm1[:arm1_quota]
        arm2 = arm2[:arm2_quota]
    return arm1, arm2


def derive_second_place_position(
    first_place_position: Sequence[float], dual_cfg: Dict[str, Any]
) -> np.ndarray:
    """返回一号臂基坐标系中的二号臂 place。

    默认严格复制一号臂 place 的 y/z，只按需求覆盖 x；完整的
    ``second_place_position`` 仅作为显式调试覆盖项。
    """
    explicit = dual_cfg.get("second_place_position")
    if explicit is not None:
        out = np.asarray(explicit, dtype=np.float64)
    else:
        out = np.asarray(first_place_position, dtype=np.float64).copy()
        if out.shape != (3,):
            raise ValueError(f"一号臂 place 需要 3 个值，实际 shape={out.shape}")
        out[0] = float(dual_cfg.get("second_place_x", -0.47))
    if out.shape != (3,):
        raise ValueError(f"二号臂 place 需要 3 个值，实际 shape={out.shape}")
    return out


def mounts_relative_transform(path: str, active: str = "left") -> Tuple[np.ndarray, Dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        for candidate in (Path.cwd() / p, TASK_ROOT.parent / p, TASK_ROOT / p):
            if candidate.exists():
                p = candidate
                break
    if not p.exists():
        raise FileNotFoundError(f"mounts config not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        mounts = yaml.safe_load(f)
    other = "right" if active == "left" else "left"
    m1 = pose_to_mat(mounts[f"{active}_arm"]["xyz"], mounts[f"{active}_arm"]["rpy"])
    m2 = pose_to_mat(mounts[f"{other}_arm"]["xyz"], mounts[f"{other}_arm"]["rpy"])
    return np.linalg.inv(m1) @ m2, mounts


def urdf_link_transform(urdf_path: str, base_link: str, target_link: str) -> np.ndarray:
    """从 URDF fixed/revolute 链的 q=0 origin 计算 base -> target 变换。"""
    root = ET.parse(resolve_repo_path(urdf_path)).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain: List[np.ndarray] = []
    current = target_link
    visited = set()
    while current != base_link:
        if current in visited or current not in by_child:
            raise ValueError(f"URDF 中找不到 {base_link} -> {target_link} 链")
        visited.add(current)
        joint = by_child[current]
        origin = joint.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
        chain.append(pose_to_mat(xyz, rpy))
        current = joint.find("parent").get("link")
    result = np.eye(4)
    for transform in reversed(chain):
        result = result @ transform
    return result


def make_arm_sequence(
    item_local: Sequence[float],
    place_local: Sequence[float],
    choice: AngleChoice,
    pp: Dict[str, Any],
    item_idx: int,
    local_to_root: np.ndarray,
    prefix: str,
) -> List[PoseSpec]:
    local = make_round_poses(
        list(item_local), list(place_local), choice.grasp, choice.place, pp, item_idx,
        choice.stage2_grasp, choice.stage2_place,
    )
    return [transform_pose(p, local_to_root, prefix) for p in local]


def grasp_prefix(sequence: Sequence[PoseSpec]) -> List[PoseSpec]:
    """返回一件物料从抓取上方到抓取后抬升的路点。"""
    prefix: List[PoseSpec] = []
    for pose in sequence:
        # make_round_poses 的 place 侧以 p_lift_in 或 place 开始。不能只看
        # kind，因为抓取/放置两侧的抬升点都叫 lift。
        if "_p_lift_in" in pose.name or pose.name.endswith("_place"):
            break
        prefix.append(pose)
    if not prefix:
        raise ValueError("物料序列没有抓取侧路点")
    return prefix


def filter_dual_items_by_primary_ik(
    arm1_items: Sequence[Dict[str, Any]],
    arm2_items: Sequence[Dict[str, Any]],
    max_items: Optional[int],
    on_fail_mode: str,
    probe: Callable[[int, Dict[str, Any], Dict[str, Any]], Tuple[bool, Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """在运动前筛掉明确无 primary grasp IK 的物料对。

    ``skip`` 会继续向后扫描，直到保留 ``max_items`` 个可达点；只有 probe
    明确返回 ``NO_PRIMARY_IK`` 才可跳过。探测异常及其他失败一律 fail closed。
    ``stop`` 保持旧语义，只截取前 N 项且完全不调用 probe。
    """
    if len(arm1_items) != len(arm2_items):
        raise ValueError("双臂 IK 预筛输入数量不一致")
    mode = str(on_fail_mode)
    if mode not in ("stop", "skip"):
        raise ValueError(f"on_fail.mode 只支持 stop/skip，实际 {mode!r}")
    limit = None if max_items is None else max(int(max_items), 0)
    if mode == "stop":
        end = len(arm1_items) if limit is None else limit
        return list(arm1_items[:end]), list(arm2_items[:end]), []

    selected1: List[Dict[str, Any]] = []
    selected2: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for source_index, (item1, item2) in enumerate(zip(arm1_items, arm2_items)):
        if limit is not None and len(selected1) >= limit:
            break
        ok, detail = probe(source_index, item1, item2)
        status = str(detail.get("status", ""))
        if ok:
            selected1.append(item1)
            selected2.append(item2)
            continue
        if status != "NO_PRIMARY_IK":
            raise RuntimeError(
                f"物料 {source_index} IK 预筛未给出可跳过结论: "
                f"status={status or 'UNKNOWN'}"
            )
        skipped.append({
            "source_item_index": int(item1.get("index", source_index)),
            "arm1_local_position": list(item1.get("position", [])),
            "arm1_raw_root_position": list(item1.get("root_position", [])),
            "arm2_local_position": list(item2.get("position", [])),
            "arm2_raw_root_position": list(item2.get("root_position", [])),
            **detail,
        })
    return selected1, selected2, skipped


def filter_arm_items_by_primary_ik(
    items: Sequence[Dict[str, Any]],
    max_items: Optional[int],
    on_fail_mode: str,
    arm: int,
    probe: Callable[[int, Dict[str, Any]], Tuple[bool, Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按单臂独立筛选共享网格任务，IK 无解只跳过该臂的 source。"""
    if int(arm) not in (1, 2):
        raise ValueError(f"arm 只支持 1/2，实际 {arm}")
    mode = str(on_fail_mode)
    if mode not in ("stop", "skip"):
        raise ValueError(f"on_fail.mode 只支持 stop/skip，实际 {mode!r}")
    limit = None if max_items is None else max(int(max_items), 0)
    if mode == "stop":
        end = len(items) if limit is None else limit
        return list(items[:end]), []

    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for source_offset, item in enumerate(items):
        if limit is not None and len(selected) >= limit:
            break
        ok, detail = probe(source_offset, item)
        status = str(detail.get("status", ""))
        if ok:
            selected_item = dict(item)
            selected_choice = (detail.get("probe") or {}).get("selected_choice")
            if isinstance(selected_choice, dict):
                # 仅供随后动态首件 Home 优先选择已证明整段 grasp prefix
                # 可达的角度；公开 metadata 会显式挑字段，不会泄漏私有键。
                selected_item["_primary_grasp_choice"] = dict(selected_choice)
            selected.append(selected_item)
            continue
        if status != "NO_PRIMARY_IK":
            raise RuntimeError(
                f"arm{arm} 物料 {item.get('index', source_offset)} IK 预筛"
                f"未给出可跳过结论: status={status or 'UNKNOWN'}"
            )
        skipped.append({
            "arm": int(arm),
            "source_item_index": int(item.get("index", source_offset)),
            "local_position": list(item.get("position", [])),
            "raw_root_position": list(item.get("root_position", [])),
            **detail,
        })
    return selected, skipped


def classify_pipeline_ik_retry(
    failures: Sequence[Dict[str, Any]],
    search: Dict[str, Any],
    selected_item_map: Sequence[Dict[str, Any]],
    *,
    phase: int,
    sequence_length: int,
) -> Optional[Dict[str, Any]]:
    """判断整线失败能否唯一归因到一个可在运动前剔除的 source。

    这里只接受独立 IK/组合预筛明确返回无解的 terminal failure。轨迹优化、
    判据、异常、未知状态或全局搜索节点预算耗尽时都返回 ``None``。多个
    block 可能在不同事件失败，因此会把全部 terminal failure 映射为活跃
    dense item 集合；所有集合的交集必须恰好是一个物料，且至少有一条
    singleton 证据。最深 global event 只用于诊断，绝不靠 pose 名称猜测。

    返回的是“当前配置搜索内的唯一 IK 冲突项”，不是数学上的全空间无解
    证明。调用方只能在任何轨迹发布/执行前剔除它，并从 q_home 重规划整线。
    """
    if bool(search.get("node_budget_exhausted", False)):
        return None
    if not selected_item_map:
        return None
    try:
        delay = int(phase)
        length = int(sequence_length)
    except (TypeError, ValueError):
        return None
    if delay < 1 or length <= delay:
        return None

    n_items = len(selected_item_map)
    terminal: List[Tuple[int, Dict[str, Any]]] = []
    for failure in failures:
        if not isinstance(failure, dict):
            return None
        if str(failure.get("status", "")) == "DOWNSTREAM_BACKTRACK":
            continue
        if (
            str(failure.get("angle_stage", "")) != "primary"
            or str(failure.get("stage", "")) != "joint_ik_prescreen"
            or str(failure.get("status", "")) != "IK_PRESCREEN_FAIL"
        ):
            return None
        start = failure.get("global_event_start")
        event = failure.get("event")
        # bool 是 int 的子类，但这里不接受 True/False 充当事件编号。
        if (
            isinstance(start, bool)
            or isinstance(event, bool)
            or not isinstance(start, (int, np.integer))
            or not isinstance(event, (int, np.integer))
        ):
            return None
        start_i, event_i = int(start), int(event)
        if start_i < 0 or event_i < 0:
            return None
        global_event = start_i + event_i
        if global_event >= delay + n_items * length:
            return None
        terminal.append((global_event, failure))

    if not terminal:
        return None
    deepest_event = max(item[0] for item in terminal)
    conflict_sets: List[set[int]] = []
    for global_event, _failure in terminal:
        active: set[int] = set()
        if global_event < n_items * length:
            active.add(global_event // length)
        shifted = global_event - delay
        if 0 <= shifted < n_items * length:
            active.add(shifted // length)
        if not active or any(index < 0 or index >= n_items for index in active):
            return None
        conflict_sets.append(active)

    common = set.intersection(*conflict_sets)
    if len(common) != 1 or not any(len(conflict) == 1 for conflict in conflict_sets):
        return None
    dense_index = next(iter(common))
    source = selected_item_map[dense_index].get("source_item_index")
    if isinstance(source, bool) or not isinstance(source, (int, np.integer)):
        return None

    evidence: List[Dict[str, Any]] = []
    for (global_event, failure), conflict in zip(terminal, conflict_sets):
        if dense_index not in conflict:
            continue
        if len(conflict) != 1 and global_event != deepest_event:
            continue
        evidence.append({
            "global_event": int(global_event),
            "pipeline_block": failure.get("pipeline_block"),
            "event_in_block": int(failure["event"]),
            "trial": failure.get("trial"),
            "targets": list(failure.get("targets") or []),
            "reason": failure.get("reason"),
            "conflict_item_indices": sorted(int(index) for index in conflict),
        })
        if len(evidence) >= 3:
            break

    return {
        "status": "PIPELINE_JOINT_IK_SEARCH_FAILED",
        "scope": "configured_search",
        "pipeline_item_index": int(dense_index),
        "source_item_index": int(source),
        "deepest_global_event": int(deepest_event),
        "terminal_failure_count": len(terminal),
        "terminal_conflict_item_indices": [
            sorted(int(index) for index in conflict) for conflict in conflict_sets
        ],
        "terminal_failure_examples": evidence,
        "search": dict(search),
    }


# ============================== 角度与节拍 ==============================


def arm_angle_choices(
    asr: Dict[str, Any], reuse: Optional[AngleChoice] = None
) -> List[AngleChoice]:
    reuse_primary = reuse.primary if reuse is not None else None
    primary = angle_combos(asr, first=reuse_primary)
    st2_on = bool((asr.get("stage2") or {}).get("enable", False))
    second = stage2_combos(asr) if st2_on else [(0.0, 0.0)]
    # 第一轮先让每个 primary 都保留 stage2=(0,0) 的原始候选，避免启用
    # stage2 后被 trial cap 困在第一个 primary 的大量二阶段组合里。
    out = [AngleChoice(g, p, 0.0, 0.0) for g, p in primary]
    out += [
        AngleChoice(g, p, g2, p2)
        for g2, p2 in second
        if abs(g2) > 1e-9 or abs(p2) > 1e-9
        for g, p in primary
    ]
    if reuse is not None:
        out = [reuse] + [x for x in out if x != reuse]
    return out


def arm_primary_choices(
    asr: Dict[str, Any], reuse: Optional[AngleChoice] = None
) -> List[AngleChoice]:
    """只返回一阶段候选；供严格的两阶段规划流程使用。"""
    first = reuse.primary if reuse is not None else None
    return [AngleChoice(g, p) for g, p in angle_combos(asr, first=first)]


def paired_angle_choices(
    arm1: Sequence[AngleChoice],
    arm2: Sequence[AngleChoice],
    limit: int = 0,
) -> Iterable[Tuple[AngleChoice, AngleChoice]]:
    """先覆盖两侧完整 grasp 范围，再按索引对角线尝试非对称组合。"""
    emitted = 0
    emitted_indices = set()

    def first_per_grasp(seq: Sequence[AngleChoice]) -> List[int]:
        seen = set()
        indices = []
        for index, choice in enumerate(seq):
            if choice.grasp not in seen:
                seen.add(choice.grasp)
                indices.append(index)
        return indices

    # coarse_to_fine 的每个 grasp 首选 place 候选优先同角配对。默认 24 次
    # 上限下可先覆盖全部 12 个 grasp，而不是只触及索引和 0..6。
    for i, j in zip(first_per_grasp(arm1), first_per_grasp(arm2)):
        yield arm1[i], arm2[j]
        emitted_indices.add((i, j))
        emitted += 1
        if limit and emitted >= limit:
            return

    for total in range(len(arm1) + len(arm2) - 1):
        lo = max(0, total - len(arm2) + 1)
        hi = min(len(arm1) - 1, total)
        for i in range(lo, hi + 1):
            j = total - i
            if (i, j) in emitted_indices:
                continue
            yield arm1[i], arm2[j]
            emitted += 1
            if limit and emitted >= limit:
                return


def paired_stage2_choices(
    arm1: Sequence[AngleChoice],
    arm2: Sequence[AngleChoice],
) -> Iterable[Tuple[AngleChoice, AngleChoice]]:
    """先同序覆盖每个二阶段候选，再补非对称组合。

    二阶段已经固定了一阶段 ``grasp/place``，不能再用
    :func:`paired_angle_choices` 按 ``choice.grasp`` 去重（所有候选的该字段
    都相同）。这里直接按候选索引覆盖，且调用方负责跳过已缓存的双零组合。
    """
    emitted_indices = set()
    for index in range(min(len(arm1), len(arm2))):
        emitted_indices.add((index, index))
        yield arm1[index], arm2[index]

    for total in range(len(arm1) + len(arm2) - 1):
        lo = max(0, total - len(arm2) + 1)
        hi = min(len(arm1) - 1, total)
        for i in range(lo, hi + 1):
            j = total - i
            if (i, j) in emitted_indices:
                continue
            yield arm1[i], arm2[j]


def delay_candidates(base: int, maximum: int, sequence_len: int) -> List[int]:
    """一号臂至少先行一步，再试更大/更小延迟；永不同时起步或整轮串行。"""
    if sequence_len <= 1:
        return []
    upper = min(max(int(maximum), 0), max(sequence_len - 1, 0))
    lower = 1
    if upper < lower:
        return []
    first = min(max(int(base), lower), upper)
    return (
        [first]
        + [v for v in range(first + 1, upper + 1)]
        + [v for v in range(first - 1, lower - 1, -1)]
    )


def scheduled_targets(
    arm1: Sequence[PoseSpec],
    arm2: Sequence[PoseSpec],
    hold1: PoseSpec,
    hold2: PoseSpec,
    delay: int,
) -> List[Tuple[PoseSpec, PoseSpec, bool, bool]]:
    """生成联合目标时间线。bool 表示该臂在此事件是否前进一个工艺点。"""
    if not arm1 and not arm2:
        return []
    d = max(int(delay), 0) if arm1 and arm2 else 0
    n = max(len(arm1), d + len(arm2))
    cur1, cur2 = hold1, hold2
    out: List[Tuple[PoseSpec, PoseSpec, bool, bool]] = []
    for tick in range(n):
        adv1 = tick < len(arm1)
        adv2 = d <= tick < d + len(arm2)
        if adv1:
            cur1 = arm1[tick]
        if adv2:
            cur2 = arm2[tick - d]
        if adv1 or adv2:
            out.append((cur1, cur2, adv1, adv2))
    return out


def scheduled_pipeline_targets(
    arm1_items: Sequence[Sequence[PoseSpec]],
    arm2_items: Sequence[Sequence[PoseSpec]],
    hold1: PoseSpec,
    hold2: PoseSpec,
    phase: int,
) -> List[Tuple[PoseSpec, PoseSpec, bool, bool]]:
    """把两臂的所有物料展开到一条无逐轮屏障的连续时间线。

    arm1 从 tick 0 连续执行，arm2 从 ``phase`` 开始；一条臂完成当前
    item 的最后一个工艺路点后，下一个 tick 立即进入下一 item。末尾较早
    完成的一臂保持其最终目标，直到另一臂 flush 完成。
    """
    flat1 = [pose for item in arm1_items for pose in item]
    flat2 = [pose for item in arm2_items for pose in item]
    d = int(phase)
    if d < 1:
        raise ValueError("连续流水 phase 必须至少为 1，保证一号臂先行")
    if not flat1 and not flat2:
        return []
    n_ticks = max(len(flat1), d + len(flat2))
    cur1, cur2 = hold1, hold2
    out: List[Tuple[PoseSpec, PoseSpec, bool, bool]] = []
    for tick in range(n_ticks):
        active1 = tick < len(flat1)
        index2 = tick - d
        active2 = 0 <= index2 < len(flat2)
        if active1:
            cur1 = flat1[tick]
        if active2:
            cur2 = flat2[index2]
        if active1 or active2:
            out.append((cur1, cur2, active1, active2))
    return out


# ======================== 独立 IK + CuRobo 联合规划 =========================


def pose_from_fk(mg, q: np.ndarray, link: str, name: str) -> PoseSpec:
    fk = compute_fk(mg, np.asarray(q).reshape(1, -1), [link])
    return PoseSpec(name, fk[f"{link}/pos"][0], fk[f"{link}/quat"][0], "hold")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def solve_arm_ik_solutions(
    ik: Any,
    target: PoseSpec,
    seed_q: Optional[np.ndarray] = None,
    max_solutions: int = 0,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """枚举一个 6-DOF 机械臂的去重 IK 分支。"""
    from curobo.types.math import Pose

    ta = ik.tensor_args
    dof = int(ik.dof)
    if dof <= 0:
        raise ValueError(f"单臂 IK solver dof 非法: {dof}")

    retract = seed = None
    seed_array: Optional[np.ndarray] = None
    if seed_q is not None:
        seed_array = np.asarray(seed_q, dtype=np.float64).reshape(-1)
        if seed_array.shape != (dof,) or not np.isfinite(seed_array).all():
            raise ValueError(
                "单臂 IK 多分支 seed 必须是有限的 "
                f"{dof} 维向量，实际 {seed_array.shape}"
            )
        q = seed_array.reshape(1, -1)
        retract = ta.to_device(q)
        seed = ta.to_device(q.reshape(1, 1, -1))
    return_seeds = int(
        getattr(ik, "num_seeds", getattr(ik, "_num_seeds", 1))
    )
    return_seeds = max(return_seeds, 1)
    goal = Pose(
        position=ta.to_device(target.position.reshape(1, 3)),
        quaternion=ta.to_device(target.quat_wxyz.reshape(1, 4)),
    )
    res = ik.solve_single(
        goal,
        retract_config=retract,
        seed_config=seed,
        return_seeds=return_seeds,
    )

    limit = int(max_solutions or 0)
    if limit < 0:
        raise ValueError("单臂 IK 多分支 max_solutions 不能为负数")
    if res.success is None or res.solution is None:
        return np.empty((0, dof), dtype=np.float64), res, {
            "n_success": 0,
            "n_unique": 0,
            "n_returned": 0,
            "max_solutions": limit,
            "truncated": False,
        }
    success = _as_numpy(res.success).reshape(-1).astype(bool)
    raw = _as_numpy(res.solution).reshape(-1, dof).astype(np.float64)
    if success.size != raw.shape[0]:
        raise RuntimeError(
            "单臂 IK 多分支 success/solution 数量不一致: "
            f"{success.size} != {raw.shape[0]}"
        )
    solutions = raw[success]
    if solutions.size == 0:
        return np.empty((0, dof), dtype=np.float64), res, {
            "n_success": 0,
            "n_unique": 0,
            "n_returned": 0,
            "max_solutions": limit,
            "truncated": False,
        }
    if not np.isfinite(solutions).all():
        raise RuntimeError("单臂 IK 多分支结果含 NaN/Inf")

    if seed_array is not None:
        order = np.argsort(
            np.linalg.norm(solutions - seed_array[None, :], axis=1),
            kind="stable",
        )
        solutions = solutions[order]

    # 同一 IK basin 在多 seed 下常只有数值微扰。0.25deg 的 L∞ 聚类
    # 会保留真正的肘/腕分支，又不让微扰占满每姿态的分支预算。
    unique: List[np.ndarray] = []
    branch_tolerance = np.radians(0.25)
    for solution in solutions:
        if any(
            float(np.max(np.abs(solution - existing))) <= branch_tolerance
            for existing in unique
        ):
            continue
        unique.append(solution.copy())
    array = np.stack(unique, axis=0)
    n_unique = int(array.shape[0])
    if limit > 0:
        array = array[:limit]
    return array, res, {
        "n_success": int(solutions.shape[0]),
        "n_unique": n_unique,
        "n_returned": int(array.shape[0]),
        "max_solutions": limit,
        "truncated": bool(limit and n_unique > limit),
        "dedup_linf_tolerance_deg": 0.25,
    }


def combine_independent_ik_candidates(
    arm1_solutions: np.ndarray,
    arm2_solutions: np.ndarray,
    seed_q: Optional[np.ndarray],
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    collision_filter: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, Any]]],
    max_pair_trials: int = 0,
    max_solutions: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cartesian 组合两臂 IK，并在截断输出前做完整 12-DOF 安全过滤。"""
    arm1 = np.asarray(arm1_solutions, dtype=np.float64)
    arm2 = np.asarray(arm2_solutions, dtype=np.float64)
    if arm1.ndim != 2 or arm2.ndim != 2 or arm1.shape[1:] != arm2.shape[1:]:
        raise ValueError(
            f"单臂 IK 候选 shape 不一致: arm1={arm1.shape}, arm2={arm2.shape}"
        )
    if arm1.shape[0] == 0 or arm2.shape[0] == 0:
        return np.empty((0, arm1.shape[1] * 2), dtype=np.float64), {
            "status": "EMPTY_ARM_IK_CANDIDATES",
            "n_arm1_candidates": int(arm1.shape[0]),
            "n_arm2_candidates": int(arm2.shape[0]),
            "n_pair_candidates": 0,
            "n_pair_trials": 0,
            "n_safe_pairs": 0,
        }
    if not np.isfinite(arm1).all() or not np.isfinite(arm2).all():
        raise ValueError("单臂 IK 候选含 NaN/Inf")
    dof = int(arm1.shape[1] * 2)
    lower = np.asarray(q_lo, dtype=np.float64).reshape(-1)
    upper = np.asarray(q_hi, dtype=np.float64).reshape(-1)
    if lower.shape != (dof,) or upper.shape != (dof,):
        raise ValueError(
            f"组合 IK 关节限位应为 {dof} 维，实际 {lower.shape}/{upper.shape}"
        )
    seed = None
    if seed_q is not None:
        seed = np.asarray(seed_q, dtype=np.float64).reshape(-1)
        if seed.shape != (dof,) or not np.isfinite(seed).all():
            raise ValueError(f"组合 IK seed 应为有限 {dof} 维，实际 {seed.shape}")
    pair_limit = int(max_pair_trials or 0)
    output_limit = int(max_solutions or 0)
    if pair_limit < 0 or output_limit < 0:
        raise ValueError("组合 IK 的候选/输出上限不能为负数")

    pairs = np.stack([
        np.concatenate((q1, q2)) for q1 in arm1 for q2 in arm2
    ])
    if seed is not None:
        order = np.argsort(
            np.linalg.norm(pairs - seed[None, :], axis=1), kind="stable"
        )
        pairs = pairs[order]
    n_pair_candidates = int(pairs.shape[0])
    pair_budget_exhausted = bool(pair_limit and n_pair_candidates > pair_limit)
    if pair_limit:
        pairs = pairs[:pair_limit]
    n_pair_trials = int(pairs.shape[0])

    within_limits = np.all(
        (pairs >= lower[None, :] - 1e-7)
        & (pairs <= upper[None, :] + 1e-7),
        axis=1,
    )
    limit_rejected = int((~within_limits).sum())
    candidates = pairs[within_limits]
    collision_report: Dict[str, Any] = {}
    if candidates.shape[0]:
        try:
            collision_free, collision_report = collision_filter(candidates)
            collision_free = np.asarray(collision_free, dtype=bool).reshape(-1)
            if collision_free.shape != (candidates.shape[0],):
                raise RuntimeError(
                    "组合 IK 碰撞过滤 mask 数量不一致: "
                    f"{collision_free.shape} != {(candidates.shape[0],)}"
                )
        except Exception as exc:  # noqa: BLE001 - 安全检查异常必须 fail closed
            # CUDA kernel fault 会毒化当前 context，不能伪装成普通“该 IK
            # 组合不可用”后继续下一候选；立即上抛才能保留准确同步栈。
            if "CUDA error" in str(exc) or "cudaError" in str(exc):
                raise
            return np.empty((0, dof), dtype=np.float64), {
                "status": "IK_PAIR_COLLISION_CHECK_ERROR",
                "reason": str(exc),
                "n_arm1_candidates": int(arm1.shape[0]),
                "n_arm2_candidates": int(arm2.shape[0]),
                "n_pair_candidates": n_pair_candidates,
                "n_pair_trials": n_pair_trials,
                "n_limit_rejected": limit_rejected,
                "n_safe_pairs": 0,
                "pair_budget_exhausted": pair_budget_exhausted,
            }
        safe = candidates[collision_free]
        collision_rejected = int((~collision_free).sum())
    else:
        safe = np.empty((0, dof), dtype=np.float64)
        collision_rejected = 0
    n_safe = int(safe.shape[0])
    returned = safe[:output_limit] if output_limit else safe
    status = "SEPARATE_IK_OK" if returned.shape[0] else (
        "NO_COLLISION_FREE_IK_PAIR" if candidates.shape[0]
        else "NO_IK_PAIR_WITHIN_LIMITS"
    )
    return returned, {
        "status": status,
        "n_arm1_candidates": int(arm1.shape[0]),
        "n_arm2_candidates": int(arm2.shape[0]),
        "n_pair_candidates": n_pair_candidates,
        "n_pair_trials": n_pair_trials,
        "n_limit_rejected": limit_rejected,
        "n_collision_rejected": collision_rejected,
        "n_safe_pairs": n_safe,
        "n_returned": int(returned.shape[0]),
        "max_pair_trials": pair_limit,
        "max_solutions": output_limit,
        "pair_budget_exhausted": pair_budget_exhausted,
        "output_truncated": bool(output_limit and n_safe > output_limit),
        "collision": collision_report,
    }


def combined_ik_collision_filter(
    collision_probe: Any,
    joint_positions: np.ndarray,
    dual_arm_prefix: str,
    collision_margin_mm: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """用隔离的 12-DOF probe 对离散拼接状态逐行验碰。

    ``collision_probe`` 是关闭 CUDA graph 的独立 IKSolver。不能复用
    MotionGen 内部 rollout 的碰撞 cost：SelfCollisionCost 会按这里的可变
    batch 重建缓存，随后主规划器重放已捕获图时可能访问失效地址。
    """
    from curobo.types.state import JointState

    q = np.asarray(joint_positions, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] == 0 or not np.isfinite(q).all():
        raise ValueError(f"组合碰撞检查 q 非法: shape={q.shape}")
    if q.shape[1] != int(collision_probe.dof):
        raise ValueError(
            f"组合碰撞检查应为 {collision_probe.dof} DOF，实际 {q.shape}"
        )
    margin_mm = float(collision_margin_mm)
    if not np.isfinite(margin_mm) or margin_mm < 0.0:
        raise ValueError(f"collision_margin_mm 非法: {collision_margin_mm}")

    q_tensor = collision_probe.tensor_args.to_device(q.astype(np.float32))
    joint_state = JointState.from_position(
        q_tensor, joint_names=collision_probe.joint_names,
    )
    # 公开 API 同时检查 bound、world 与 self/inter-arm constraints。
    metrics = collision_probe.check_constraints(joint_state)
    safe_constraint = _as_numpy(metrics.feasible).reshape(-1).astype(bool)
    if safe_constraint.shape != (q.shape[0],):
        raise RuntimeError(
            "12-DOF constraint mask 数量不一致: "
            f"{safe_constraint.shape} != {(q.shape[0],)}"
        )

    # 分项值仅取隔离 probe 的 aux rollout constraint，供失败报告计数。
    state = collision_probe.kinematics.get_state(q_tensor)
    sphere_rows = state.link_spheres_tensor
    spheres = sphere_rows.unsqueeze(1)
    if not bool(spheres.isfinite().all()):
        raise RuntimeError("组合 IK 的 FK 碰撞球含 NaN/Inf")
    rollout = collision_probe.rollout_fn
    self_cost = getattr(rollout, "robot_self_collision_constraint", None)
    world_cost = getattr(rollout, "primitive_collision_constraint", None)
    if self_cost is None:
        raise RuntimeError("12-DOF self/inter-arm collision cost 未启用")
    if world_cost is None:
        raise RuntimeError("12-DOF world collision cost 未启用")

    def bad_mask(cost: Any) -> np.ndarray:
        value = cost.forward(spheres).reshape(q.shape[0], -1)
        if not bool(value.isfinite().all()):
            raise RuntimeError("组合 IK 碰撞代价含 NaN/Inf")
        return _as_numpy((value > 0).any(dim=1)).astype(bool)

    self_bad = bad_mask(self_cost)
    world_bad = bad_mask(world_cost)
    unexplained_bad = ~safe_constraint & ~(self_bad | world_bad)

    # CuRobo self-collision constraint 判定实体相交；用户配置的额外臂间
    # clearance 由同一批 FK 碰撞球做 Cartesian 球对检查。
    kin_cfg = collision_probe.kinematics.kinematics_config
    idx_map = _as_numpy(kin_cfg.link_sphere_idx_map).reshape(-1)
    name_to_idx = dict(kin_cfg.link_name_to_idx_map)
    idx_to_name = {int(value): name for name, value in name_to_idx.items()}
    names = [idx_to_name[int(value)] for value in idx_map]
    arm2_mask = np.asarray(
        [name.startswith(str(dual_arm_prefix)) for name in names], dtype=bool
    )
    if not arm2_mask.any() or arm2_mask.all():
        raise RuntimeError(
            f"前缀 {dual_arm_prefix!r} 无法拆分 12-DOF 碰撞球"
        )
    sphere_np = _as_numpy(sphere_rows).astype(np.float32, copy=False)
    pair_report = check_pair_collisions(
        sphere_np[:, ~arm2_mask], sphere_np[:, arm2_mask],
        [name for name, mask in zip(names, ~arm2_mask) if mask],
        [name for name, mask in zip(names, arm2_mask) if mask],
        margin_mm / 1000.0,
    )
    margin_bad = np.zeros(q.shape[0], dtype=bool)
    collision_indices = np.asarray(
        pair_report.get("collision_indices", []), dtype=np.int64
    ).reshape(-1)
    if collision_indices.size:
        if (
            int(collision_indices.min()) < 0
            or int(collision_indices.max()) >= q.shape[0]
        ):
            raise RuntimeError("臂间碰撞报告包含越界状态索引")
        margin_bad[collision_indices] = True

    safe = safe_constraint & ~margin_bad
    return safe, {
        "checked": True,
        "probe": "dedicated_12dof_iksolver_no_cuda_graph",
        "n_states": int(q.shape[0]),
        "n_probe_constraint_rejected": int((~safe_constraint).sum()),
        "n_self_or_inter_arm_collision": int(self_bad.sum()),
        "n_world_collision": int(world_bad.sum()),
        "n_other_constraint_rejected": int(unexplained_bad.sum()),
        "n_inter_arm_margin_collision": int(margin_bad.sum()),
        "n_margin_only_rejected": int((safe_constraint & margin_bad).sum()),
        "collision_margin_mm": margin_mm,
        "n_collision_free": int(safe.sum()),
        "inter_arm": pair_report,
    }


def make_separate_arm_ik_context(
    combined_robot_dict: Dict[str, Any],
    world: Any,
    wall_link_names: Sequence[str],
    robot_cfg: Dict[str, Any],
    planner_cfg: Dict[str, Any],
    dual_cfg: Dict[str, Any],
    arm2_from_root: np.ndarray,
) -> SeparateArmIkContext:
    """构造单臂 IK，以及与 MotionGen 完全隔离的 12-DOF 验碰器。"""
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    weights = list(robot_cfg.get("cspace_distance_weight") or [])
    retract = robot_cfg.get("retract_config")
    single_robot_cfg: Dict[str, Any] = {
        "robot_yml": str(robot_cfg.get("single_arm_robot_yml", "xtrainer.yml")),
        "base_link": str(robot_cfg["base_link"]),
        "ee_link": str(robot_cfg["ee_link"]),
        "flange_link": str(robot_cfg.get("flange_link") or "LINK_6"),
        "collision_sphere_buffer": float(
            robot_cfg.get("collision_sphere_buffer") or 0.0
        ),
        "joint_limit_clip": robot_cfg.get("joint_limit_clip"),
        "cspace_distance_weight": weights[:6] if weights else None,
        "retract_config": None if retract is None else list(retract)[:6],
    }
    single_dict = load_robot_cfg_dict(single_robot_cfg)
    single_kin = single_dict["robot_cfg"]["kinematics"]
    single_kin["link_names"] = [single_robot_cfg["ee_link"]]
    tensor_args = TensorDeviceType()
    ik_cfg = IKSolverConfig.load_from_robot_config(
        single_dict,
        world_model=None,
        tensor_args=tensor_args,
        num_seeds=max(int(planner_cfg["num_ik_seeds"]), 64),
        position_threshold=float(planner_cfg["position_threshold"]),
        rotation_threshold=float(planner_cfg["rotation_threshold"]),
        self_collision_check=False,
        self_collision_opt=False,
        use_cuda_graph=False,
        regularization=True,
    )
    arm_solver = IKSolver(ik_cfg)

    # 只把它当批量关节状态 constraint probe，不调用 solve/warmup。单独创建
    # world checker 与 cost buffer，确保这里任意 N 的检查不会触碰 mg CUDA 图。
    collision_probe_cfg = IKSolverConfig.load_from_robot_config(
        combined_robot_dict,
        world_model=world,
        tensor_args=tensor_args,
        num_seeds=1,
        position_threshold=float(planner_cfg["position_threshold"]),
        rotation_threshold=float(planner_cfg["rotation_threshold"]),
        self_collision_check=True,
        self_collision_opt=False,
        use_particle_opt=False,
        use_cuda_graph=False,
        regularization=False,
        collision_activation_distance=float(
            planner_cfg["collision_activation_distance"]
        ),
        collision_checker_type=getattr(
            CollisionCheckerType,
            str(planner_cfg.get("collision_checker_type", "MESH")).upper(),
        ),
    )
    collision_probe = IKSolver(collision_probe_cfg)
    if wall_link_names:
        restrict_world_collision_to_links(
            collision_probe, list(wall_link_names), quiet=True
        )
    probe_limits = collision_probe.kinematics.get_joint_limits().position
    q_lo = _as_numpy(probe_limits[0]).astype(np.float64)
    q_hi = _as_numpy(probe_limits[1]).astype(np.float64)

    max_per_arm = int(dual_cfg.get("ik_candidates_per_arm", 8) or 0)
    max_pairs = int(dual_cfg.get("max_ik_pair_trials", 64) or 0)
    max_goals = int(dual_cfg.get("ik_goal_candidates", 4) or 0)
    collision_margin_mm = float(dual_cfg.get("collision_margin_mm", 0.0))
    if max_per_arm <= 0 or max_pairs <= 0 or max_goals <= 0:
        raise ValueError(
            "dual_arm.ik_candidates_per_arm/max_ik_pair_trials/"
            "ik_goal_candidates 必须为正整数"
        )
    if not np.isfinite(collision_margin_mm) or collision_margin_mm < 0.0:
        raise ValueError("dual_arm.collision_margin_mm 必须是有限非负数")
    return SeparateArmIkContext(
        arm_solvers=(arm_solver, arm_solver),
        root_to_arm=(np.eye(4), np.asarray(arm2_from_root, dtype=np.float64)),
        q_lo=np.asarray(q_lo, dtype=np.float64).copy(),
        q_hi=np.asarray(q_hi, dtype=np.float64).copy(),
        collision_filter=lambda q: combined_ik_collision_filter(
            collision_probe,
            q,
            str(robot_cfg.get("dual_arm_prefix", "second_")),
            collision_margin_mm,
        ),
        max_candidates_per_arm=max_per_arm,
        max_pair_trials=max_pairs,
        max_goal_solutions=max_goals,
        collision_margin_mm=collision_margin_mm,
        collision_probe=collision_probe,
    )


def _discrete_ik_context(mg: Any) -> Any:
    """测试桩可继续只提供 ``ik_solver``；真实入口必须安装独立上下文。"""
    context = getattr(mg, "_xtrainer_separate_ik", None)
    return context if context is not None else mg.ik_solver


def solve_dual_ik_solutions(
    ik: SeparateArmIkContext,
    target1: PoseSpec,
    target2: PoseSpec,
    second_ee: str,
    seed_q: Optional[np.ndarray] = None,
    max_solutions: int = 0,
    active_arms: Tuple[bool, bool] = (True, True),
    fixed_q: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, SeparateArmIkResult, Dict[str, Any]]:
    """两臂分别求 IK，Cartesian 组合后用完整 12-DOF 模型验碰。"""
    del second_ee  # 兼容旧调用签名；单臂 solver 不使用副 link 目标。
    if not isinstance(ik, SeparateArmIkContext):
        raise TypeError("离散双臂 IK 需要 SeparateArmIkContext")
    active = (bool(active_arms[0]), bool(active_arms[1]))
    if not any(active):
        raise ValueError("离散双臂 IK 至少需要一条活动臂")
    arm_dof = int(ik.arm_solvers[0].dof)
    if arm_dof <= 0 or int(ik.arm_solvers[1].dof) != arm_dof:
        raise ValueError("两条单臂 IK solver 的 DOF 不一致")
    total_dof = arm_dof * 2
    seed = None
    if seed_q is not None:
        seed = np.asarray(seed_q, dtype=np.float64).reshape(-1)
        if seed.shape != (total_dof,) or not np.isfinite(seed).all():
            raise ValueError(
                f"双臂独立 IK seed 应为有限 {total_dof} 维，实际 {seed.shape}"
            )
    fixed = seed
    if fixed_q is not None:
        fixed = np.asarray(fixed_q, dtype=np.float64).reshape(-1)
        if fixed.shape != (total_dof,) or not np.isfinite(fixed).all():
            raise ValueError(
                f"双臂独立 IK fixed_q 应为有限 {total_dof} 维，实际 {fixed.shape}"
            )
    if not all(active) and fixed is None:
        raise ValueError("存在未激活手臂时必须提供 seed_q 或 fixed_q")

    targets = (target1, target2)
    arm_solutions: List[np.ndarray] = []
    arm_results: List[Any] = []
    arm_reports: List[Dict[str, Any]] = []
    for arm_index in range(2):
        sl = slice(arm_index * arm_dof, (arm_index + 1) * arm_dof)
        if not active[arm_index]:
            arm_solutions.append(fixed[sl].reshape(1, arm_dof).copy())
            arm_results.append(None)
            arm_reports.append({
                "status": "INACTIVE_ARM_FIXED",
                "n_success": 1,
                "n_unique": 1,
                "n_returned": 1,
            })
            continue
        local_target = transform_pose(
            targets[arm_index], ik.root_to_arm[arm_index],
            f"arm{arm_index + 1}_local_",
        )
        arm_seed = None if seed is None else seed[sl]
        solutions, result, report = solve_arm_ik_solutions(
            ik.arm_solvers[arm_index],
            local_target,
            seed_q=arm_seed,
            max_solutions=ik.max_candidates_per_arm,
        )
        arm_solutions.append(solutions)
        arm_results.append(result)
        arm_reports.append(report)
        if solutions.shape[0] == 0:
            status = f"ARM{arm_index + 1}_IK_FAIL"
            compatibility = {
                "n_success": 0,
                "n_unique": 0,
                "n_returned": 0,
                "max_solutions": int(max_solutions or 0),
                "truncated": False,
                "solver_mode": "separate_6dof_then_12dof_collision_filter",
                "active_arms": list(active),
                "arm_reports": arm_reports,
            }
            combined_result = SeparateArmIkResult(
                status=status,
                arm_results=(arm_results[0] if arm_results else None,
                             arm_results[1] if len(arm_results) > 1 else None),
                report=compatibility,
            )
            return (
                np.empty((0, total_dof), dtype=np.float64),
                combined_result,
                compatibility,
            )

    combined, pair_report = combine_independent_ik_candidates(
        arm_solutions[0], arm_solutions[1], seed,
        ik.q_lo, ik.q_hi, ik.collision_filter,
        max_pair_trials=ik.max_pair_trials,
        max_solutions=max_solutions,
    )
    status = str(pair_report["status"])
    compatibility = {
        "n_success": int(pair_report.get("n_safe_pairs", 0)),
        "n_unique": int(pair_report.get("n_safe_pairs", 0)),
        "n_returned": int(combined.shape[0]),
        "max_solutions": int(max_solutions or 0),
        "truncated": bool(
            pair_report.get("output_truncated", False)
            or pair_report.get("pair_budget_exhausted", False)
            or any(report.get("truncated", False) for report in arm_reports)
        ),
        "search_incomplete": bool(
            pair_report.get("pair_budget_exhausted", False)
            or any(report.get("truncated", False) for report in arm_reports)
        ),
        "solver_mode": "separate_6dof_then_12dof_collision_filter",
        "active_arms": list(active),
        "arm_reports": arm_reports,
        "pair_report": pair_report,
    }
    combined_result = SeparateArmIkResult(
        status=status,
        arm_results=(arm_results[0], arm_results[1]),
        report=compatibility,
    )
    return combined, combined_result, compatibility


def solve_dual_ik(
    ik: SeparateArmIkContext,
    target1: PoseSpec,
    target2: PoseSpec,
    second_ee: str,
    seed_q: Optional[np.ndarray] = None,
    active_arms: Tuple[bool, bool] = (True, True),
    fixed_q: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], SeparateArmIkResult]:
    """返回离 seed 最近的一条安全独立 IK 组合。"""
    solutions, result, _ = solve_dual_ik_solutions(
        ik, target1, target2, second_ee,
        seed_q=seed_q, max_solutions=1,
        active_arms=active_arms, fixed_q=fixed_q,
    )
    return (None if solutions.shape[0] == 0 else solutions[0].copy()), result


def unique_primary_grasp_choices(asr: Dict[str, Any]) -> List[AngleChoice]:
    """每个一阶段 grasp 角只保留首个 choice。

    预抓取位姿只由 grasp 侧角度决定；去掉 place 角的重复组合可以让动态
    Home 的独立 IK 完整覆盖抓取角，同时避免无意义地重复求解。
    """
    out: List[AngleChoice] = []
    seen = set()
    for choice in arm_primary_choices(asr):
        key = round(float(choice.grasp), 12)
        if key in seen:
            continue
        seen.add(key)
        out.append(choice)
    return out


def first_pregrasp_home_pose(
    item_local: Sequence[float],
    place_local: Sequence[float],
    choice: AngleChoice,
    pp: Dict[str, Any],
    item_index: int,
    local_to_root: np.ndarray,
    arm: int,
) -> PoseSpec:
    """由某臂首件的第一个抓取侧路点构造任务 Home。"""
    sequence = make_arm_sequence(
        item_local,
        place_local,
        choice,
        pp,
        item_index,
        local_to_root,
        f"arm{int(arm)}_",
    )
    first = grasp_prefix(sequence)[0]
    return PoseSpec(
        f"arm{int(arm)}_home_first_pregrasp",
        np.asarray(first.position, dtype=np.float64).copy(),
        np.asarray(first.quat_wxyz, dtype=np.float64).copy(),
        "start",
    )


def ordered_first_home_choices(
    asr: Dict[str, Any], item: Optional[Dict[str, Any]]
) -> List[AngleChoice]:
    """让离线 prefix probe 已证明可达的 grasp 角排在 Home 根搜索前面。"""
    choices = unique_primary_grasp_choices(asr)
    if item is None:
        return choices
    preferred = item.get("_primary_grasp_choice")
    if not isinstance(preferred, dict) or "angle_grasp_deg" not in preferred:
        return choices
    preferred_grasp = float(preferred["angle_grasp_deg"])
    return sorted(
        choices,
        key=lambda choice: 0
        if abs(float(choice.grasp) - preferred_grasp) < 1e-9 else 1,
    )


def enumerate_first_pregrasp_home_roots(
    mg,
    q_seed: np.ndarray,
    arm1_item: Dict[str, Any],
    arm2_item: Optional[Dict[str, Any]],
    place1_local: Sequence[float],
    place2_local: Sequence[float],
    pp: Dict[str, Any],
    asr: Dict[str, Any],
    transforms: Tuple[np.ndarray, np.ndarray],
    ee_links: Tuple[str, str],
    idle_arm2_home: PoseSpec,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    max_angle_pair_trials: int = 0,
    max_branches_per_pair: int = 0,
    max_roots: int = 0,
) -> Tuple[List[FirstPregraspHomeRoot], Dict[str, Any]]:
    """枚举首件姿态对及每个姿态对的独立 IK 安全组合分支。"""
    seed = np.asarray(q_seed, dtype=np.float64).reshape(-1)
    lower = np.asarray(q_lo, dtype=np.float64).reshape(-1)
    upper = np.asarray(q_hi, dtype=np.float64).reshape(-1)
    if seed.shape != lower.shape or seed.shape != upper.shape:
        raise ValueError("Home root seed 与关节限位维度不一致")
    choices1 = ordered_first_home_choices(asr, arm1_item)
    choices2 = (
        ordered_first_home_choices(asr, arm2_item)
        if arm2_item is not None else []
    )
    if not choices1 or (arm2_item is not None and not choices2):
        return [], {
            "status": "NO_HOME_ANGLE_CANDIDATES",
            "n_angle_pair_trials": 0,
            "n_roots": 0,
        }

    if arm2_item is None:
        pairs: Iterable[Tuple[AngleChoice, Optional[AngleChoice]]] = (
            (choice, None) for choice in choices1
        )
    else:
        pairs = (
            (choice1, choice2)
            for choice1, choice2 in paired_angle_choices(choices1, choices2)
        )

    pair_limit = int(max_angle_pair_trials or 0)
    branch_limit = int(max_branches_per_pair or 0)
    root_limit = int(max_roots or 0)
    roots: List[FirstPregraspHomeRoot] = []
    failure_counts: Dict[str, int] = {}
    n_angle_pairs = 0
    n_raw_success_branches = 0
    n_unique_branches = 0
    n_kept_branches = 0
    n_branch_truncated_pairs = 0
    n_outside_limits = 0
    pair_budget_exhausted = False
    root_budget_exhausted = False
    for choice1, choice2 in pairs:
        if pair_limit and n_angle_pairs >= pair_limit:
            pair_budget_exhausted = True
            break
        n_angle_pairs += 1
        home1 = first_pregrasp_home_pose(
            arm1_item["position"], place1_local, choice1, pp,
            int(arm1_item.get("index", 0)), transforms[0], 1,
        )
        home2 = (
            idle_arm2_home
            if arm2_item is None or choice2 is None
            else first_pregrasp_home_pose(
                arm2_item["position"], place2_local, choice2, pp,
                int(arm2_item.get("index", 0)), transforms[1], 2,
            )
        )
        solutions, result, branch_report = solve_dual_ik_solutions(
            _discrete_ik_context(mg),
            home1,
            home2,
            ee_links[1],
            seed_q=seed,
            max_solutions=branch_limit,
            active_arms=(True, arm2_item is not None),
            fixed_q=seed,
        )
        n_raw_success_branches += int(branch_report["n_success"])
        n_unique_branches += int(branch_report["n_unique"])
        if bool(branch_report["truncated"]):
            n_branch_truncated_pairs += 1
        if solutions.shape[0] == 0:
            status = str(getattr(result, "status", "IK_FAIL"))
            failure_counts[status] = failure_counts.get(status, 0) + 1
            continue
        n_kept_branches += int(solutions.shape[0])
        for branch_index, solution in enumerate(solutions):
            if np.any(solution < lower - 1e-7) or np.any(solution > upper + 1e-7):
                n_outside_limits += 1
                continue
            roots.append(FirstPregraspHomeRoot(
                q_home=solution.copy(),
                arm1_home=home1,
                arm2_home=home2,
                arm1_choice=choice1,
                arm2_choice=choice2,
                angle_pair_trial=n_angle_pairs,
                ik_branch_index=branch_index,
                distance_to_seed=float(np.linalg.norm(solution - seed)),
            ))

    # 先试每个姿态对离 seed 最近的分支，再试每个姿态对的第二分支，
    # 依次类推。paired_angle_choices 的前一轮专门覆盖完整 grasp
    # 角范围；按 branch-rank round-robin 可以在根预算截断时保留这个性质。
    roots.sort(key=lambda root: (
        int(root.ik_branch_index),
        int(root.angle_pair_trial),
        float(root.distance_to_seed),
    ))

    # `max_roots` 是完整流水最多尝试多少个根，而不是 IK 枚举的提前退出
    # 条件。先把允许的姿态对都枚举完，再排序截断。
    n_roots_available = len(roots)
    if root_limit and n_roots_available > root_limit:
        roots = roots[:root_limit]
        root_budget_exhausted = True
    n_distinct_angle_pairs_returned = len({
        int(root.angle_pair_trial) for root in roots
    })
    max_branch_rank_returned = (
        max(int(root.ik_branch_index) for root in roots) if roots else None
    )

    status = "FIRST_PREGRASP_HOME_ROOTS_OK" if roots else "NO_FIRST_PREGRASP_HOME_IK"
    return roots, {
        "status": status,
        "arm1_source_item_index": int(arm1_item.get("index", 0)),
        "arm2_source_item_index": (
            None if arm2_item is None else int(arm2_item.get("index", 0))
        ),
        "n_angle_pair_trials": n_angle_pairs,
        "n_raw_success_ik_branches": n_raw_success_branches,
        "n_unique_ik_branches": n_unique_branches,
        "n_ik_branches_after_dedup_and_pair_cap": n_kept_branches,
        "n_branch_truncated_pairs": n_branch_truncated_pairs,
        "n_outside_limit_branches": n_outside_limits,
        "n_roots_available": n_roots_available,
        "n_roots": len(roots),
        "n_roots_returned": len(roots),
        "n_distinct_angle_pairs_returned": n_distinct_angle_pairs_returned,
        "max_branch_rank_returned": max_branch_rank_returned,
        "root_order": "ik_branch_rank_then_angle_pair_trial",
        "max_angle_pair_trials": pair_limit,
        "max_branches_per_pair": branch_limit,
        "max_roots": root_limit,
        "angle_pair_budget_exhausted": pair_budget_exhausted,
        "branch_budget_exhausted": bool(n_branch_truncated_pairs),
        "root_budget_exhausted": root_budget_exhausted,
        "search_incomplete": bool(
            pair_budget_exhausted
            or n_branch_truncated_pairs
            or root_budget_exhausted
        ),
        "failure_status_counts": failure_counts,
    }


def solve_first_pregrasp_home(
    mg,
    q_seed: np.ndarray,
    arm1_item: Dict[str, Any],
    arm2_item: Optional[Dict[str, Any]],
    place1_local: Sequence[float],
    place2_local: Sequence[float],
    pp: Dict[str, Any],
    asr: Dict[str, Any],
    transforms: Tuple[np.ndarray, np.ndarray],
    ee_links: Tuple[str, str],
    idle_arm2_home: PoseSpec,
) -> Tuple[Optional[np.ndarray], Optional[PoseSpec], Optional[PoseSpec], Dict[str, Any]]:
    """搜索“两臂各自在首件预抓取位姿”的无碰撞联合 Home IK。

    arm2 没有获配任务时没有“首件”，此时只让 arm1 使用首件预抓取 Home，
    arm2 保留配置 Home；拼接状态仍使用完整 12-DOF 模型验碰。
    """
    choices1 = ordered_first_home_choices(asr, arm1_item)
    choices2 = ordered_first_home_choices(asr, arm2_item) if arm2_item is not None else []
    if not choices1 or (arm2_item is not None and not choices2):
        return None, None, None, {
            "status": "NO_HOME_ANGLE_CANDIDATES",
            "n_trials": 0,
        }

    if arm2_item is None:
        pairs: Iterable[Tuple[AngleChoice, Optional[AngleChoice]]] = (
            (choice, None) for choice in choices1
        )
    else:
        pairs = (
            (choice1, choice2)
            for choice1, choice2 in paired_angle_choices(choices1, choices2)
        )

    counts: Dict[str, int] = {}
    n_trials = 0
    last_reason = ""
    for choice1, choice2 in pairs:
        n_trials += 1
        try:
            home1 = first_pregrasp_home_pose(
                arm1_item["position"],
                place1_local,
                choice1,
                pp,
                int(arm1_item.get("index", 0)),
                transforms[0],
                1,
            )
            home2 = (
                idle_arm2_home
                if arm2_item is None or choice2 is None
                else first_pregrasp_home_pose(
                    arm2_item["position"],
                    place2_local,
                    choice2,
                    pp,
                    int(arm2_item.get("index", 0)),
                    transforms[1],
                    2,
                )
            )
            solution, result = solve_dual_ik(
                _discrete_ik_context(mg),
                home1,
                home2,
                ee_links[1],
                seed_q=q_seed,
                active_arms=(True, arm2_item is not None),
                fixed_q=q_seed,
            )
        except Exception as exc:  # noqa: BLE001 - Home 异常必须 fail closed
            return None, None, None, {
                "status": "FIRST_PREGRASP_HOME_EXCEPTION",
                "n_trials": n_trials,
                "reason": str(exc),
            }
        if solution is not None:
            return solution, home1, home2, {
                "status": "FIRST_PREGRASP_HOME_IK_OK",
                "n_trials": n_trials,
                "arm1_source_item_index": int(arm1_item.get("index", 0)),
                "arm2_source_item_index": (
                    None if arm2_item is None
                    else int(arm2_item.get("index", 0))
                ),
                "arm1_choice": choice1.to_dict(),
                "arm2_choice": None if choice2 is None else choice2.to_dict(),
                "arm1_home": home1.to_dict(),
                "arm2_home": home2.to_dict(),
                "failure_status_counts": counts,
            }
        status = str(getattr(result, "status", "IK_FAIL"))
        counts[status] = counts.get(status, 0) + 1
        last_reason = status

    return None, None, None, {
        "status": "NO_FIRST_PREGRASP_HOME_IK",
        "n_trials": n_trials,
        "arm1_source_item_index": int(arm1_item.get("index", 0)),
        "arm2_source_item_index": (
            None if arm2_item is None else int(arm2_item.get("index", 0))
        ),
        "failure_status_counts": counts,
        "last_reason": last_reason,
    }


def prescreen_dual_schedule(
    mg, q_start: np.ndarray, seq1: Sequence[PoseSpec], seq2: Sequence[PoseSpec],
    delay: int, cr: Dict[str, Any], q_lo: np.ndarray, q_hi: np.ndarray,
    ee_links: Tuple[str, str], inactive_tolerance_deg: float,
    parked_joint_refs: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
    park_after_targets: Tuple[Optional[str], Optional[str]] = (None, None),
) -> Tuple[bool, Dict[str, Any]]:
    """逐事件独立 IK 组合预筛可达性、碰撞、限位和可选关节变化。

    返回值中的 ``ik_trace`` 保留本次预筛已经真正求出的 12-DOF
    构型：第 0 点是 q_start，之后每个点对应一次成功的 IK。
    即使该 IK 因跳变/限位/等待臂漂移被拒绝，也会以
    ``accepted=false`` 保留，便于 RViz 查看。IK 无解时不伪造关节点。

    这些是离散 IK 样本，不是连续无碰轨迹。调用方应将最佳
    trace 单独落盘，而不是把每个搜索候选都塞进 plan_failed.json。
    """
    initial1 = pose_from_fk(mg, q_start, ee_links[0], "arm1_prescreen_hold")
    initial2 = pose_from_fk(mg, q_start, ee_links[1], "arm2_prescreen_hold")
    events = scheduled_targets(seq1, seq2, initial1, initial2, delay)
    cur = np.asarray(q_start, dtype=np.float64).copy()
    ik_trace: Dict[str, Any] = {
        "joint_positions_rad": [cur.tolist()],
        "start_poses": {
            "arm1": initial1.to_dict(),
            "arm2": initial2.to_dict(),
        },
        "events": [],
        "n_events": len(events),
        "n_solved_events": 0,
        "n_accepted_events": 0,
        "deepest_attempted_event": None,
        "deepest_solved_event": None,
        "complete": False,
    }

    def with_trace(report: Dict[str, Any], complete: bool = False) -> Dict[str, Any]:
        ik_trace["n_solved_events"] = len(ik_trace["joint_positions_rad"]) - 1
        ik_trace["n_accepted_events"] = sum(
            1 for event in ik_trace["events"] if event.get("accepted") is True
        )
        ik_trace["complete"] = bool(complete)
        ik_trace["status"] = str(report.get("status", "IK_PRESCREEN_UNKNOWN"))
        return {**report, "ik_trace": ik_trace}

    parked = [
        None if ref is None else np.asarray(ref, dtype=np.float64).reshape(-1).copy()
        for ref in parked_joint_refs
    ]
    if any(ref is not None and ref.shape != cur.shape for ref in parked):
        return False, with_trace({
            "status": "IK_PRESCREEN_PARK_REFERENCE_INVALID"
        })
    arm_dof = cur.size // 2
    worst_delta = 0.0
    min_margin = float("inf")

    for index, (target1, target2, active1, active2) in enumerate(events):
        if not active1:
            target1 = pose_from_fk(
                mg, cur if parked[0] is None else parked[0], ee_links[0],
                f"arm1_prescreen_hold_{index}"
            )
        if not active2:
            target2 = pose_from_fk(
                mg, cur if parked[1] is None else parked[1], ee_links[1],
                f"arm2_prescreen_hold_{index}"
            )
        event_record: Dict[str, Any] = {
            "event": int(index),
            "targets": [target1.name, target2.name],
            "active_arms": [
                arm for arm, active in ((1, active1), (2, active2)) if active
            ],
            "target_poses": {
                "arm1": target1.to_dict(),
                "arm2": target2.to_dict(),
            },
            "solve_success": False,
            "accepted": False,
            "state_index": None,
        }
        ik_trace["deepest_attempted_event"] = int(index)
        fixed_q = cur.copy()
        for arm_idx, (active, reference) in enumerate(
            zip((active1, active2), parked)
        ):
            if active or reference is None:
                continue
            sl = slice(arm_idx * arm_dof, (arm_idx + 1) * arm_dof)
            fixed_q[sl] = reference[sl]
        try:
            ik_context = _discrete_ik_context(mg)
            goal_limit = max(
                1, int(getattr(ik_context, "max_goal_solutions", 1) or 1)
            )
            candidates, result, ik_report = solve_dual_ik_solutions(
                ik_context, target1, target2, ee_links[1],
                seed_q=cur, max_solutions=goal_limit,
                active_arms=(active1, active2), fixed_q=fixed_q,
            )
        except Exception as exc:  # noqa: BLE001
            if "CUDA error" in str(exc) or "cudaError" in str(exc):
                raise
            event_record["status"] = "IK_PRESCREEN_EXCEPTION"
            event_record["reason"] = str(exc)
            ik_trace["events"].append(event_record)
            return False, with_trace({
                "status": "IK_PRESCREEN_EXCEPTION", "event": index,
                "targets": [target1.name, target2.name], "reason": str(exc),
            })
        if candidates.shape[0] == 0:
            reason = str(getattr(result, "status", "IK 无解"))
            event_record["status"] = "IK_PRESCREEN_FAIL"
            event_record["reason"] = reason
            event_record["ik_report"] = ik_report
            ik_trace["events"].append(event_record)
            return False, with_trace({
                "status": "IK_PRESCREEN_FAIL", "event": index,
                "targets": [target1.name, target2.name],
                "reason": reason,
                "ik_report": ik_report,
            })
        candidates = np.asarray(candidates, dtype=np.float64)
        order = np.argsort(
            np.linalg.norm(candidates - cur[None, :], axis=1), kind="stable"
        )
        candidates = candidates[order]
        candidate_attempts: List[Dict[str, Any]] = []
        selected: Optional[Tuple[np.ndarray, Dict[str, Any], int]] = None
        first_rejected: Optional[Tuple[np.ndarray, Dict[str, Any]]] = None
        for rank, candidate in enumerate(candidates, start=1):
            nxt_candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
            valid, candidate_report = validate_segment(
                mg, cur, np.stack([cur, nxt_candidate]), (target1, target2),
                (active1, active2), (False, False), cr, {}, q_lo, q_hi,
                ee_links, inactive_tolerance_deg,
                enforce_joint_delta=bool(
                    cr.get("prescreen_joint_delta_check", True)
                ),
            )
            attempt: Dict[str, Any] = {
                "rank": rank,
                "status": str(candidate_report.get("status", "OK")),
            }
            if not valid:
                candidate_attempts.append({**attempt, **candidate_report})
                if first_rejected is None:
                    first_rejected = (nxt_candidate, candidate_report)
                continue

            parked_failure: Optional[Dict[str, Any]] = None
            for arm_idx, (active, reference) in enumerate(
                zip((active1, active2), parked)
            ):
                if active or reference is None:
                    continue
                sl = slice(arm_idx * arm_dof, (arm_idx + 1) * arm_dof)
                excursion = float(np.degrees(np.abs(
                    np.stack([cur, nxt_candidate])[:, sl]
                    - reference[None, sl]
                )).max())
                if excursion > inactive_tolerance_deg:
                    parked_failure = {
                        "status": "IK_PRESCREEN_PARKED_ARM_MOVED",
                        "arm": arm_idx + 1,
                        "parked_joint_excursion_deg": excursion,
                        "limit_deg": inactive_tolerance_deg,
                    }
                    break
            if parked_failure is not None:
                candidate_attempts.append({
                    "rank": rank, **parked_failure,
                })
                if first_rejected is None:
                    first_rejected = (nxt_candidate, parked_failure)
                continue

            candidate_attempts.append({"rank": rank, "status": "IK_ACCEPTED"})
            selected = (nxt_candidate, candidate_report, rank)
            break

        event_record["solve_success"] = True
        event_record["candidate_attempts"] = candidate_attempts
        event_record["ik_report"] = ik_report
        ik_trace["deepest_solved_event"] = int(index)
        if selected is None:
            # 保留离 seed 最近的被拒绝 IK，供 RViz/报告诊断；其它候选只以
            # 摘要记录，避免把候选分支误画成一条连续轨迹。
            assert first_rejected is not None
            rejected_q, rejected_report = first_rejected
            ik_trace["joint_positions_rad"].append(rejected_q.tolist())
            event_record["state_index"] = len(ik_trace["joint_positions_rad"]) - 1
            failure_status = str(rejected_report.get("status", "IK_PRESCREEN_CRITERION"))
            if failure_status == "IK_PRESCREEN_PARKED_ARM_MOVED":
                event_record.update(rejected_report)
                ik_trace["events"].append(event_record)
                return False, with_trace({
                    **rejected_report,
                    "event": index,
                    "targets": [target1.name, target2.name],
                    "candidate_attempts": candidate_attempts,
                    "ik_report": ik_report,
                })
            event_record["status"] = "IK_PRESCREEN_CRITERION"
            event_record["criterion_status"] = failure_status
            ik_trace["events"].append(event_record)
            return False, with_trace({
                "status": "IK_PRESCREEN_CRITERION", "event": index,
                "targets": [target1.name, target2.name],
                "criterion_status": failure_status,
                "candidate_attempts": candidate_attempts,
                "ik_report": ik_report,
                **{k: v for k, v in rejected_report.items() if k != "status"},
            })

        nxt, report, selected_rank = selected
        ik_trace["joint_positions_rad"].append(nxt.tolist())
        event_record["state_index"] = len(ik_trace["joint_positions_rad"]) - 1
        event_record["selected_candidate_rank"] = selected_rank
        event_record["accepted"] = True
        event_record["status"] = "IK_ACCEPTED"
        ik_trace["events"].append(event_record)
        worst_delta = max(worst_delta, max(report["arm_max_joint_delta_deg"]))
        min_margin = min(min_margin, float(report["min_limit_margin_deg"]))
        cur = nxt
        for arm_idx, (active, target) in enumerate(
            zip((active1, active2), (target1, target2))
        ):
            if active and park_after_targets[arm_idx] == target.name:
                parked[arm_idx] = cur.copy()

    return True, with_trace({
        "status": "IK_PRESCREEN_OK", "n_events": len(events),
        "max_joint_delta_deg": worst_delta,
        "min_limit_margin_deg": min_margin,
        "joint_delta_check": bool(
            cr.get("prescreen_joint_delta_check", True)
        ),
    }, complete=True)


def probe_arm_item_primary_ik(
    mg,
    q_home: np.ndarray,
    item_local: Sequence[float],
    place_local: Sequence[float],
    item_index: int,
    arm: int,
    pp: Dict[str, Any],
    asr: Dict[str, Any],
    local_to_root: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    ee_links: Tuple[str, str],
) -> Tuple[bool, Dict[str, Any]]:
    """确认一条臂至少有一个 primary grasp-prefix 独立 IK 安全组合。

    另一条臂固定在联合 home，因此探测仍包含完整 12-DOF 模型与臂间碰撞。
    这是任务开始前的必要条件，不把 trajopt/判据/碰撞失败误当成可跳过 IK。
    """
    arm_i = int(arm)
    if arm_i not in (1, 2):
        return False, {"status": "IK_PROBE_ERROR", "reason": f"非法 arm={arm}"}
    unique_grasp_choices = unique_primary_grasp_choices(asr)
    if not unique_grasp_choices:
        return False, {
            "status": "IK_PROBE_ERROR",
            "reason": "angle_search 没有 primary grasp 候选",
        }

    q_seed = np.asarray(q_home, dtype=np.float64).reshape(-1)
    lower = np.asarray(q_lo, dtype=np.float64).reshape(-1)
    upper = np.asarray(q_hi, dtype=np.float64).reshape(-1)
    if (
        q_seed.size == 0
        or q_seed.size % 2
        or lower.shape != q_seed.shape
        or upper.shape != q_seed.shape
        or not np.isfinite(q_seed).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
    ):
        return False, {
            "status": "IK_PROBE_ERROR",
            "reason": "IK probe 的 q_home/limit shape 或数值非法",
        }
    try:
        home_targets = (
            pose_from_fk(mg, q_seed, ee_links[0], "arm1_ik_filter_hold"),
            pose_from_fk(mg, q_seed, ee_links[1], "arm2_ik_filter_hold"),
        )
    except Exception as exc:  # noqa: BLE001
        return False, {"status": "IK_PROBE_ERROR", "reason": str(exc)}

    failure_counts: Dict[str, int] = {}
    selected: Optional[AngleChoice] = None
    last_failure: Optional[Dict[str, Any]] = None
    n_ik_calls = 0
    for choice in unique_grasp_choices:
        try:
            full_sequence = make_arm_sequence(
                item_local, place_local, choice, pp, item_index,
                local_to_root, f"arm{arm_i}_",
            )
            prefix = grasp_prefix(full_sequence)
        except Exception as exc:  # noqa: BLE001 - 探测异常不可当作“无解”跳过
            return False, {
                "status": "IK_PROBE_ERROR", "arm": arm_i, "reason": str(exc),
            }
        candidate_ok = True
        for event_index, pose in enumerate(prefix):
            targets = (
                (pose, home_targets[1]) if arm_i == 1
                else (home_targets[0], pose)
            )
            try:
                # 每个 target、每个角度都从同一个 q_home 重新求解，避免
                # 某条贪心 IK 链的分支选择把可达 source 误判为无解。
                solution, result = solve_dual_ik(
                    _discrete_ik_context(mg), targets[0], targets[1],
                    ee_links[1], seed_q=q_seed,
                    active_arms=(arm_i == 1, arm_i == 2), fixed_q=q_seed,
                )
                n_ik_calls += 1
            except Exception as exc:  # noqa: BLE001
                return False, {
                    "status": "IK_PROBE_ERROR", "arm": arm_i, "event": event_index,
                    "targets": [targets[0].name, targets[1].name], "reason": str(exc),
                }
            if solution is None:
                candidate_ok = False
                status = "IK_TARGET_FAIL"
                failure_counts[status] = failure_counts.get(status, 0) + 1
                last_failure = {
                    "status": status, "event": event_index,
                    "targets": [targets[0].name, targets[1].name],
                    "reason": str(getattr(result, "status", "IK 无解")),
                }
                break
            solution = np.asarray(solution, dtype=np.float64).reshape(-1)
            if (
                solution.shape != q_seed.shape
                or not np.isfinite(solution).all()
                or np.any(solution < lower - 1e-7)
                or np.any(solution > upper + 1e-7)
            ):
                return False, {
                    "status": "IK_PROBE_ERROR", "arm": arm_i, "event": event_index,
                    "targets": [targets[0].name, targets[1].name],
                    "reason": "IK solver 返回非法或越限解",
                }
        if candidate_ok:
            selected = choice
            break
    report = {
        "reachable": selected is not None,
        "n_grasp_candidates": len(unique_grasp_choices),
        "n_ik_calls": n_ik_calls,
        "selected_choice": None if selected is None else selected.to_dict(),
        "failure_status_counts": failure_counts,
        "last_failure": last_failure,
    }
    reachable = selected is not None
    return reachable, {
        "status": "IK_REACHABLE" if reachable else "NO_PRIMARY_IK",
        "arm": arm_i,
        "reachable": reachable,
        "probe": report,
    }


def probe_dual_item_primary_ik(
    mg,
    q_home: np.ndarray,
    item1_local: Sequence[float],
    item2_local: Sequence[float],
    place1_local: Sequence[float],
    place2_local: Sequence[float],
    item_index: int,
    pp: Dict[str, Any],
    asr: Dict[str, Any],
    transforms: Tuple[np.ndarray, np.ndarray],
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    ee_links: Tuple[str, str],
) -> Tuple[bool, Dict[str, Any]]:
    """兼容镜像布局：同一轮次要求两臂各自的 grasp prefix 都有 IK。"""
    results: Dict[str, Any] = {}
    for arm, item, place in (
        (1, item1_local, place1_local),
        (2, item2_local, place2_local),
    ):
        ok, detail = probe_arm_item_primary_ik(
            mg, q_home, item, place, item_index, arm, pp, asr,
            transforms[arm - 1], q_lo, q_hi, ee_links,
        )
        if str(detail.get("status")) == "IK_PROBE_ERROR":
            return False, detail
        results[f"arm{arm}"] = detail["probe"]
        results[f"arm{arm}"]["reachable"] = bool(ok)
    reachable = bool(results["arm1"]["reachable"] and results["arm2"]["reachable"])
    return reachable, {
        "status": "IK_REACHABLE" if reachable else "NO_PRIMARY_IK",
        "arm1_reachable": bool(results["arm1"]["reachable"]),
        "arm2_reachable": bool(results["arm2"]["reachable"]),
        "probe": results,
    }


def _update_secondary_metric(mg, link: str, metric) -> int:
    """CuRobo 公开 metric API 只更新主 EE；同步更新当前版本的副 link cost。"""
    count = 0
    for rollout in mg.get_all_rollout_instances():
        costs = getattr(rollout, "_link_pose_costs", {})
        if link in costs:
            costs[link].update_metric(metric, update_offset_waypoint=True)
            count += 1
        conv = getattr(rollout, "_link_pose_convergence", {})
        if link in conv:
            conv[link].update_metric(metric, update_offset_waypoint=False)
            count += 1
    return count


def make_hold_pose_metric(mg):
    """把未激活手臂的 TCP 全位姿锁在当前段起点。"""
    from curobo.rollout.cost.pose_cost import PoseCostMetric

    return PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=mg.tensor_args.to_device([1.0] * 6),
        project_to_goal_frame=False,
    )


def plan_dual_segment(
    mg, q_start: np.ndarray, target1: PoseSpec, target2: PoseSpec,
    pl: Dict[str, Any], second_ee: str, metric1=None, metric2=None,
    goal_q: Optional[np.ndarray] = None,
):
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    ta = mg.tensor_args
    start = JointState.from_position(
        ta.to_device(np.asarray(q_start, dtype=np.float64).reshape(1, -1)),
        joint_names=mg.joint_names,
    )

    def as_pose(p: PoseSpec):
        return Pose(
            position=ta.to_device(p.position.reshape(1, 3)),
            quaternion=ta.to_device(p.quat_wxyz.reshape(1, 4)),
        )

    use_joint_goal = goal_q is not None
    try:
        if use_joint_goal and (metric1 is not None or metric2 is not None):
            raise ValueError(
                "joint-space goal 不支持 Cartesian hold/直线 metric"
            )
        if metric2 is not None:
            n_patched = _update_secondary_metric(mg, second_ee, metric2)
            if n_patched == 0:
                raise RuntimeError(
                    f"找不到 {second_ee} 的 secondary pose cost，无法施加轨迹约束"
                )

        cfg = MotionGenPlanConfig(
            enable_graph=bool(pl.get("enable_graph", False)),
            enable_graph_attempt=int(pl.get("enable_graph_attempt", 3)),
            max_attempts=int(pl["max_attempts"]),
            timeout=float(pl["timeout"]),
            enable_finetune_trajopt=True,
            parallel_finetune=True,
            pose_cost_metric=metric1,
        )
        if use_joint_goal:
            goal = np.asarray(goal_q, dtype=np.float64).reshape(-1)
            if goal.shape != np.asarray(q_start).reshape(-1).shape:
                raise ValueError(
                    f"joint-space goal 维度错误: {goal.shape}"
                )
            goal_state = JointState.from_position(
                ta.to_device(goal.reshape(1, -1)),
                joint_names=mg.joint_names,
            )
            return mg.plan_single_js(start, goal_state, cfg)
        return mg.plan_single(
            start, as_pose(target1), cfg,
            link_poses={second_ee: as_pose(target2)},
        )
    finally:
        # MotionGen 正常返回时会自行 reset 主 EE；异常路径不会。显式重置
        # 两侧，确保一次失败 trial 不会把 hold/linear metric 泄漏给下一次。
        if metric1 is not None:
            mg.update_pose_cost_metric(PoseCostMetric.reset_metric())
        if metric2 is not None:
            _update_secondary_metric(mg, second_ee, PoseCostMetric.reset_metric())


def validate_segment(
    mg, q_start: np.ndarray, q: np.ndarray,
    targets: Tuple[PoseSpec, PoseSpec], active: Tuple[bool, bool],
    linear: Tuple[bool, bool], cr: Dict[str, Any], linear_cfg: Dict[str, Any],
    q_lo: np.ndarray, q_hi: np.ndarray, ee_links: Tuple[str, str],
    inactive_tolerance_deg: float,
    enforce_joint_delta: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    q = np.asarray(q, dtype=np.float64)
    q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)
    q_lo = np.asarray(q_lo, dtype=np.float64).reshape(-1)
    q_hi = np.asarray(q_hi, dtype=np.float64).reshape(-1)
    if (
        q.ndim != 2
        or q.shape[0] == 0
        or q.shape[1] == 0
        or q.shape[1] % 2 != 0
        or q_start.shape != (q.shape[1],)
        or q_lo.shape != (q.shape[1],)
        or q_hi.shape != (q.shape[1],)
    ):
        return False, {
            "status": "INVALID_TRAJECTORY_SHAPE",
            "trajectory_shape": list(q.shape),
            "start_shape": list(q_start.shape),
            "lower_limit_shape": list(q_lo.shape),
            "upper_limit_shape": list(q_hi.shape),
        }
    if not (
        np.isfinite(q).all()
        and np.isfinite(q_start).all()
        and np.isfinite(q_lo).all()
        and np.isfinite(q_hi).all()
    ):
        return False, {"status": "NONFINITE_TRAJECTORY"}

    # 两臂各自使用同一份 1-based criterion.joints。
    local_idx = [int(j) - 1 for j in cr.get("joints", [1, 2, 3, 4])]
    arm_dof = q.shape[1] // 2
    arm_delta: List[float] = []
    for offset in (0, arm_dof):
        d = np.degrees(q[:, offset:offset + arm_dof].max(axis=0)
                       - q[:, offset:offset + arm_dof].min(axis=0))
        arm_delta.append(float(d[local_idx].max()))
    max_allowed = float(cr["max_joint_delta_deg"])
    if enforce_joint_delta and max(arm_delta) > max_allowed:
        return False, {
            "status": "JOINT_DELTA_EXCEED",
            "arm_max_joint_delta_deg": arm_delta,
            "limit_deg": max_allowed,
        }

    margin = np.degrees(np.minimum(q - q_lo[None, :], q_hi[None, :] - q)).min(axis=0)
    min_margin = float(margin.min())
    threshold = float(cr.get("min_limit_margin_deg") or 0.0)
    if threshold and min_margin < threshold:
        return False, {
            "status": "LIMIT_MARGIN",
            "min_limit_margin_deg": min_margin,
            "limit_deg": threshold,
        }

    inactive_excursion: List[Optional[float]] = [None, None]
    for arm_idx, is_active in enumerate(active):
        if is_active:
            continue
        sl = slice(arm_idx * arm_dof, (arm_idx + 1) * arm_dof)
        excursion = np.degrees(np.abs(q[:, sl] - q_start[None, sl])).max()
        inactive_excursion[arm_idx] = float(excursion)
        if excursion > inactive_tolerance_deg:
            return False, {
                "status": "INACTIVE_ARM_MOVED",
                "arm": arm_idx + 1,
                "inactive_joint_excursion_deg": inactive_excursion,
                "limit_deg": inactive_tolerance_deg,
                "arm_max_joint_delta_deg": arm_delta,
                "min_limit_margin_deg": min_margin,
            }

    info: Dict[str, Any] = {
        "arm_max_joint_delta_deg": arm_delta,
        "joint_delta_check": bool(enforce_joint_delta),
        "min_limit_margin_deg": min_margin,
        "inactive_joint_excursion_deg": inactive_excursion,
    }
    if any(linear):
        fk = compute_fk(mg, q, list(ee_links))
        axis = {"x": 0, "y": 1, "z": 2}[str(linear_cfg.get("free_axis", "z"))]
        lateral = [i for i in range(3) if i != axis]
        max_dev = float(linear_cfg.get("max_deviation_mm", 3.0))
        max_rot = float(linear_cfg.get("max_rotation_deg", 5.0))
        linear_info: Dict[str, Any] = {}
        for ai, enabled in enumerate(linear):
            if not enabled:
                continue
            link, target = ee_links[ai], targets[ai]
            pos = fk[f"{link}/pos"]
            quat = fk[f"{link}/quat"]
            dev_mm = float(np.abs(pos[:, lateral] - target.position[None, lateral]).max() * 1000.0)
            rot_deg = float(max(quat_angle_deg(target.quat_wxyz, x) for x in quat))
            linear_info[f"arm{ai + 1}"] = {
                "lateral_dev_mm": dev_mm,
                "rotation_dev_deg": rot_deg,
            }
            if dev_mm > max_dev or (
                bool(linear_cfg.get("hold_rotation", True)) and rot_deg > max_rot
            ):
                return False, {
                    **info,
                    "status": "LINEAR_CONSTRAINT",
                    "linear": linear_info,
                    "max_deviation_mm": max_dev,
                    "max_rotation_deg": max_rot,
                }
        info["linear"] = linear_info
    return True, info


def validate_dual_tcp_endpoint(
    mg: Any,
    q_end: np.ndarray,
    targets: Tuple[PoseSpec, PoseSpec],
    ee_links: Tuple[str, str],
    planner_cfg: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """确认独立 IK 拼出的 q_goal 在 combined robot 上仍满足双 TCP。"""
    try:
        fk = compute_fk(
            mg, np.asarray(q_end, dtype=np.float64).reshape(1, -1),
            list(ee_links),
        )
    except Exception as exc:  # noqa: BLE001 - 端点 FK 不可用必须 fail closed
        return False, {
            "status": "IK_GOAL_TCP_ENDPOINT_CHECK_ERROR",
            "reason": str(exc),
        }

    position_limit = float(planner_cfg["position_threshold"])
    rotation_limit = float(planner_cfg["rotation_threshold"])
    per_arm: Dict[str, Any] = {}
    valid = True
    for arm_index, (link, target) in enumerate(zip(ee_links, targets)):
        pos_key, quat_key = f"{link}/pos", f"{link}/quat"
        if pos_key not in fk or quat_key not in fk:
            return False, {
                "status": "IK_GOAL_TCP_ENDPOINT_CHECK_ERROR",
                "reason": f"combined FK 缺少 {link}",
            }
        actual_pos = np.asarray(fk[pos_key][0], dtype=np.float64)
        actual_quat = np.asarray(fk[quat_key][0], dtype=np.float64)
        position_error = float(np.linalg.norm(actual_pos - target.position))
        rotation_error_deg = float(
            quat_angle_deg(target.quat_wxyz, actual_quat)
        )
        # CuRobo pose convergence 的 rotation_threshold 对应 sin(theta/2)。
        rotation_error_metric = float(
            np.sin(np.radians(rotation_error_deg) * 0.5)
        )
        arm_valid = bool(
            position_error <= position_limit + 1e-6
            and rotation_error_metric <= rotation_limit + 1e-6
        )
        valid = valid and arm_valid
        per_arm[f"arm{arm_index + 1}"] = {
            "link": link,
            "position_error_mm": position_error * 1000.0,
            "rotation_error_deg": rotation_error_deg,
            "rotation_error_metric": rotation_error_metric,
            "valid": arm_valid,
        }
    max_rotation_deg = float(
        np.degrees(2.0 * np.arcsin(np.clip(rotation_limit, 0.0, 1.0)))
    )
    return valid, {
        "status": "IK_GOAL_TCP_ENDPOINT_OK" if valid
        else "IK_GOAL_TCP_ENDPOINT_MISMATCH",
        "position_limit_mm": position_limit * 1000.0,
        "rotation_metric_limit": rotation_limit,
        "rotation_limit_deg": max_rotation_deg,
        "arms": per_arm,
    }


def plan_schedule(
    mg, q_start: np.ndarray, seq1: Sequence[PoseSpec], seq2: Sequence[PoseSpec],
    delay: int, pl: Dict[str, Any], cr: Dict[str, Any], linear_cfg: Dict[str, Any],
    q_lo: np.ndarray, q_hi: np.ndarray, ee_links: Tuple[str, str],
    inactive_tolerance_deg: float,
    parked_joint_refs: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
    park_after_targets: Tuple[Optional[str], Optional[str]] = (None, None),
) -> Tuple[bool, Dict[str, Any]]:
    hold1 = pose_from_fk(mg, q_start, ee_links[0], "arm1_hold")
    hold2 = pose_from_fk(mg, q_start, ee_links[1], "arm2_hold")
    events = scheduled_targets(seq1, seq2, hold1, hold2, delay)
    cur = np.asarray(q_start, dtype=np.float64).copy()
    parked = [
        None if ref is None else np.asarray(ref, dtype=np.float64).reshape(-1).copy()
        for ref in parked_joint_refs
    ]
    if any(ref is not None and ref.shape != cur.shape for ref in parked):
        return False, {"status": "PARK_REFERENCE_INVALID", "segments": []}
    arm_dof = cur.size // 2
    pos_l: List[np.ndarray] = []
    vel_l: List[np.ndarray] = []
    acc_l: List[np.ndarray] = []
    reports: List[Dict[str, Any]] = []
    dt: Optional[float] = None
    lin_on = bool(linear_cfg.get("enable", False))
    lin_kinds = set(linear_cfg.get("kinds") or ["place"])
    separate_context = getattr(mg, "_xtrainer_separate_ik", None)
    path_margin_mm = (
        float(separate_context.collision_margin_mm)
        if isinstance(separate_context, SeparateArmIkContext) else 0.0
    )

    if not events:
        return False, {"status": "EMPTY_SCHEDULE", "segments": reports}

    for i, (t1, t2, adv1, adv2) in enumerate(events):
        # 排程表中的 inactive target 只表示“留在上一工艺点”。这里用本段
        # 实际起始 FK 刷新它，避免前一段毫米级末端误差被下一段强行修正。
        if not adv1:
            t1 = pose_from_fk(
                mg, cur if parked[0] is None else parked[0], ee_links[0],
                f"arm1_hold_event_{i}",
            )
        if not adv2:
            t2 = pose_from_fk(
                mg, cur if parked[1] is None else parked[1], ee_links[1],
                f"arm2_hold_event_{i}",
            )
        linear = (
            bool(lin_on and adv1 and t1.kind in lin_kinds),
            bool(lin_on and adv2 and t2.kind in lin_kinds),
        )
        metric1 = (
            make_hold_pose_metric(mg) if not adv1
            else make_hold_axis_metric(
                mg, str(linear_cfg.get("free_axis", "z")),
                bool(linear_cfg.get("hold_rotation", True)),
            ) if linear[0] else None
        )
        metric2 = (
            make_hold_pose_metric(mg) if not adv2
            else make_hold_axis_metric(
                mg, str(linear_cfg.get("free_axis", "z")),
                bool(linear_cfg.get("hold_rotation", True)),
            ) if linear[1] else None
        )
        # 两臂同时运动且没有 Cartesian 直线/hold metric 时，把两臂分别
        # 求出的多个安全 IK 组合依次交给 12-DOF joint-space MotionGen。
        # 带 metric 的段仍走 pose planner，因为 plan_single_js 不接受 pose metric。
        joint_goals: Optional[np.ndarray] = None
        ik_goal_report: Optional[Dict[str, Any]] = None
        ik_goal_attempts: List[Dict[str, Any]] = []
        if metric1 is None and metric2 is None:
            try:
                ik_context = _discrete_ik_context(mg)
                goal_limit = max(
                    1, int(getattr(ik_context, "max_goal_solutions", 1) or 1)
                )
                joint_goals, ik_goal_result, ik_goal_report = (
                    solve_dual_ik_solutions(
                        ik_context, t1, t2, ee_links[1],
                        seed_q=cur, max_solutions=goal_limit,
                        active_arms=(adv1, adv2), fixed_q=cur,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 端点安全求解异常即停止
                if "CUDA error" in str(exc) or "cudaError" in str(exc):
                    raise
                return False, {
                    "failed_event": i,
                    "targets": [t1.name, t2.name],
                    "status": "SEPARATE_IK_GOAL_EXCEPTION",
                    "reason": str(exc),
                    "segments": reports,
                }
            if joint_goals.shape[0] == 0:
                return False, {
                    "failed_event": i,
                    "targets": [t1.name, t2.name],
                    "status": str(getattr(
                        ik_goal_result, "status", "SEPARATE_IK_GOAL_FAIL"
                    )),
                    "ik_goal_report": ik_goal_report,
                    "segments": reports,
                }
            # 保证调用者/测试桩即使未排序，也总从离当前 12-DOF 状态最近的
            # 分支开始尝试。
            goal_order = np.argsort(
                np.linalg.norm(joint_goals - cur[None, :], axis=1),
                kind="stable",
            )
            joint_goals = joint_goals[goal_order]

        started = time.time()
        goal_options: List[Optional[np.ndarray]] = (
            [goal.copy() for goal in joint_goals]
            if joint_goals is not None else [None]
        )
        selected: Optional[Tuple[Any, np.ndarray, np.ndarray, np.ndarray,
                                 float, Dict[str, Any]]] = None
        selected_goal: Optional[np.ndarray] = None
        selected_goal_rank: Optional[int] = None
        goal_error_deg: Optional[float] = None
        tcp_goal_report: Optional[Dict[str, Any]] = None
        selected_path_collision: Optional[Dict[str, Any]] = None
        for goal_rank, joint_goal in enumerate(goal_options, start=1):
            try:
                res = plan_dual_segment(
                    mg, cur, t1, t2, pl, ee_links[1], metric1, metric2,
                    goal_q=joint_goal,
                )
            except Exception as exc:  # noqa: BLE001 - 写入完整规划诊断
                if "CUDA error" in str(exc) or "cudaError" in str(exc):
                    raise
                attempt = {
                    "rank": goal_rank,
                    "status": "PLAN_EXCEPTION",
                    "reason": str(exc),
                }
                ik_goal_attempts.append(attempt)
                if joint_goals is not None:
                    continue
                return False, {
                    "failed_event": i, "targets": [t1.name, t2.name],
                    "status": f"EXCEPTION: {exc}",
                    "ik_goal_report": ik_goal_report,
                    "segments": reports,
                }
            plan_ok = (
                res is not None and res.success is not None
                and bool(res.success.item())
            )
            if not plan_ok:
                status = str(res.status) if res is not None else "PLAN_FAIL"
                ik_goal_attempts.append({"rank": goal_rank, "status": status})
                if joint_goals is not None:
                    continue
                return False, {
                    "failed_event": i, "targets": [t1.name, t2.name],
                    "status": status,
                    "ik_goal_report": ik_goal_report,
                    "segments": reports,
                }

            traj = res.get_interpolated_plan()
            q_candidate = traj.position.detach().cpu().numpy().astype(np.float64)
            v_candidate = (
                traj.velocity.detach().cpu().numpy().astype(np.float64)
                if traj.velocity is not None else np.zeros_like(q_candidate)
            )
            a_candidate = (
                traj.acceleration.detach().cpu().numpy().astype(np.float64)
                if traj.acceleration is not None else np.zeros_like(q_candidate)
            )
            candidate_dt = float(res.interpolation_dt)
            candidate_goal_error: Optional[float] = None
            candidate_tcp_report: Optional[Dict[str, Any]] = None
            if joint_goal is not None:
                candidate_goal_error = float(np.degrees(np.max(np.abs(
                    q_candidate[-1] - joint_goal
                ))))
                if candidate_goal_error > 1.0:
                    ik_goal_attempts.append({
                        "rank": goal_rank,
                        "status": "IK_GOAL_ENDPOINT_MISMATCH",
                        "goal_error_deg": candidate_goal_error,
                        "limit_deg": 1.0,
                    })
                    continue
                tcp_ok, candidate_tcp_report = validate_dual_tcp_endpoint(
                    mg, q_candidate[-1], (t1, t2), ee_links, pl,
                )
                if not tcp_ok:
                    ik_goal_attempts.append({
                        "rank": goal_rank,
                        **candidate_tcp_report,
                    })
                    continue

            candidate_path_collision: Optional[Dict[str, Any]] = None
            if path_margin_mm > 0.0:
                try:
                    path_safe, candidate_path_collision = (
                        separate_context.collision_filter(q_candidate)
                    )
                    path_safe_array = np.asarray(path_safe)
                    if (
                        path_safe_array.shape != (q_candidate.shape[0],)
                        or not np.issubdtype(path_safe_array.dtype, np.bool_)
                    ):
                        raise RuntimeError(
                            "轨迹 collision margin mask 非法: "
                            f"{path_safe_array.shape}/{path_safe_array.dtype}"
                        )
                    candidate_path_collision = dict(
                        candidate_path_collision or {}
                    )
                except Exception as exc:  # noqa: BLE001 - 安全检查必须 fail closed
                    if "CUDA error" in str(exc) or "cudaError" in str(exc):
                        raise
                    failure = {
                        "status": "TRAJECTORY_COLLISION_MARGIN_CHECK_ERROR",
                        "reason": str(exc),
                        "collision_margin_mm": path_margin_mm,
                    }
                    ik_goal_attempts.append({"rank": goal_rank, **failure})
                    return False, {
                        "failed_event": i,
                        "targets": [t1.name, t2.name],
                        **failure,
                        "ik_goal_report": ik_goal_report,
                        "ik_goal_attempts": ik_goal_attempts,
                        "segments": reports,
                    }
                if not bool(path_safe_array.all()):
                    rejected = np.flatnonzero(~path_safe_array)
                    failure = {
                        "status": "TRAJECTORY_COLLISION_MARGIN_POSTCHECK",
                        "collision_margin_mm": path_margin_mm,
                        "n_rejected_points": int(rejected.size),
                        "first_rejected_point": int(rejected[0]),
                        "path_collision": candidate_path_collision,
                    }
                    ik_goal_attempts.append({"rank": goal_rank, **failure})
                    if joint_goals is not None:
                        continue
                    return False, {
                        "failed_event": i,
                        "targets": [t1.name, t2.name],
                        **failure,
                        "ik_goal_report": ik_goal_report,
                        "ik_goal_attempts": ik_goal_attempts,
                        "segments": reports,
                    }

            valid, candidate_check = validate_segment(
                mg, cur, q_candidate, (t1, t2), (adv1, adv2), linear,
                cr, linear_cfg, q_lo, q_hi, ee_links,
                inactive_tolerance_deg,
            )
            if not valid:
                ik_goal_attempts.append({
                    "rank": goal_rank,
                    **candidate_check,
                })
                if joint_goals is not None:
                    continue
                return False, {
                    "failed_event": i, "targets": [t1.name, t2.name],
                    **candidate_check,
                    "ik_goal_report": ik_goal_report,
                    "segments": reports,
                }

            ik_goal_attempts.append({"rank": goal_rank, "status": "SUCCESS"})
            selected = (
                res, q_candidate, v_candidate, a_candidate,
                candidate_dt, candidate_check,
            )
            selected_goal = joint_goal
            selected_goal_rank = goal_rank if joint_goal is not None else None
            goal_error_deg = candidate_goal_error
            tcp_goal_report = candidate_tcp_report
            selected_path_collision = candidate_path_collision
            break

        if selected is None:
            return False, {
                "failed_event": i,
                "targets": [t1.name, t2.name],
                "status": "IK_GOAL_CANDIDATES_EXHAUSTED",
                "ik_goal_report": ik_goal_report,
                "ik_goal_attempts": ik_goal_attempts,
                "segments": reports,
            }
        res, q, v, a, segment_dt, check = selected
        parked_excursion: List[Optional[float]] = [None, None]
        for arm_idx, (active, reference) in enumerate(zip((adv1, adv2), parked)):
            if active or reference is None:
                continue
            sl = slice(arm_idx * arm_dof, (arm_idx + 1) * arm_dof)
            excursion = float(np.degrees(np.abs(
                q[:, sl] - reference[None, sl]
            )).max())
            parked_excursion[arm_idx] = excursion
            if excursion > inactive_tolerance_deg:
                return False, {
                    "failed_event": i,
                    "targets": [t1.name, t2.name],
                    "status": "PARKED_ARM_MOVED",
                    "arm": arm_idx + 1,
                    "parked_joint_excursion_deg": parked_excursion,
                    "limit_deg": inactive_tolerance_deg,
                    "segments": reports,
                }
        if (
            v.shape != q.shape
            or a.shape != q.shape
            or not np.isfinite(v).all()
            or not np.isfinite(a).all()
        ):
            return False, {
                "failed_event": i,
                "targets": [t1.name, t2.name],
                "status": "INVALID_DERIVATIVE_TRAJECTORY",
                "position_shape": list(q.shape),
                "velocity_shape": list(v.shape),
                "acceleration_shape": list(a.shape),
                "segments": reports,
            }
        if not np.isfinite(segment_dt) or segment_dt <= 0.0:
            return False, {
                "failed_event": i, "targets": [t1.name, t2.name],
                "status": "INVALID_INTERPOLATION_DT",
                "interpolation_dt": segment_dt, "segments": reports,
            }
        if dt is None:
            dt = segment_dt
        elif abs(segment_dt - dt) > 1e-12:
            return False, {
                "failed_event": i, "targets": [t1.name, t2.name],
                "status": "INCONSISTENT_INTERPOLATION_DT",
                "expected_dt": dt, "actual_dt": segment_dt,
                "segments": reports,
            }
        start_error = float(np.max(np.abs(q[0] - cur)))
        if start_error > 1e-5:
            return False, {
                "failed_event": i, "targets": [t1.name, t2.name],
                "status": "SEGMENT_START_DISCONTINUITY",
                "max_joint_error_rad": start_error,
                "segments": reports,
            }

        point_start = 0 if not reports else int(reports[-1]["trajectory_point_end"])
        point_end = point_start + int(q.shape[0]) - 1
        cur = q[-1].copy()
        parked_after: List[int] = []
        for arm_idx, (active, target) in enumerate(zip((adv1, adv2), (t1, t2))):
            if active and park_after_targets[arm_idx] == target.name:
                parked[arm_idx] = cur.copy()
                parked_after.append(arm_idx + 1)

        # 只在段边界去掉重复点。
        pos_l.append(q[1:] if pos_l else q)
        vel_l.append(v[1:] if vel_l else v)
        acc_l.append(a[1:] if acc_l else a)
        reports.append({
            "index": i,
            "targets": [t1.name, t2.name],
            "active_arms": [n + 1 for n, x in enumerate((adv1, adv2)) if x],
            "linear_arms": [n + 1 for n, x in enumerate(linear) if x],
            "n_points": int(q.shape[0]),
            "duration_s": float((q.shape[0] - 1) * segment_dt),
            "solve_time_s": float(time.time() - started),
            "attempts": int(res.attempts),
            "trajectory_point_start": point_start,
            "trajectory_point_end": point_end,
            "parked_arms_after_event": parked_after,
            "parked_joint_excursion_deg": parked_excursion,
            "planner_goal_mode": (
                "separate_ik_joint_goal" if selected_goal is not None
                else "dual_pose_goal_with_metric"
            ),
            "ik_goal_candidate_rank": selected_goal_rank,
            "ik_goal_candidates_available": (
                int(joint_goals.shape[0]) if joint_goals is not None else None
            ),
            "ik_goal_attempts": ik_goal_attempts,
            "ik_goal_endpoint_error_deg": goal_error_deg,
            "tcp_goal_endpoint": tcp_goal_report,
            "ik_goal_report": ik_goal_report,
            "trajectory_collision_margin": selected_path_collision,
            **check,
        })

    return True, {
        "position": np.concatenate(pos_l, axis=0),
        "velocity": np.concatenate(vel_l, axis=0),
        "acceleration": np.concatenate(acc_l, axis=0),
        "dt": float(dt),
        "q_end": cur,
        "segments": reports,
        "parked_joint_refs": tuple(parked),
    }


def try_joint_round(
    mg, q_start: np.ndarray, item1_local: Optional[Sequence[float]],
    item2_local: Optional[Sequence[float]], place1_local: Sequence[float],
    place2_local: Sequence[float], item_idx: int, pp: Dict[str, Any],
    asr: Dict[str, Any], dc: Dict[str, Any], transforms: Tuple[np.ndarray, np.ndarray],
    last_choices: Tuple[Optional[AngleChoice], Optional[AngleChoice]],
    pl: Dict[str, Any], cr: Dict[str, Any], linear_cfg: Dict[str, Any],
    q_lo: np.ndarray, q_hi: np.ndarray, ee_links: Tuple[str, str],
) -> Tuple[Optional[PlannedRound], List[Dict[str, Any]]]:
    # 与单臂版保持相同的两阶段语义：先只搜索一阶段角，得到可行基线后
    # 固定它，再搜索二阶段增量；二阶段全失败时返回已缓存的一阶段轨迹。
    # 不能把所有 a2=0 候选和非零 a2 候选平铺后遇首个成功即返回，否则
    # 只要一阶段存在解，非零 stage2 永远不会被尝试。
    c1 = arm_primary_choices(asr, last_choices[0]) if item1_local is not None else []
    c2 = arm_primary_choices(asr, last_choices[1]) if item2_local is not None else []
    max_pairs = int(dc.get("max_pair_angle_trials") or 0)
    failures: List[Dict[str, Any]] = []
    pair_trials_used = 0

    def zero_stage2(choice: Optional[AngleChoice]) -> bool:
        return choice is None or (
            abs(choice.stage2_grasp) < 1e-9
            and abs(choice.stage2_place) < 1e-9
        )

    def attempt_pairs(
        pairs: Iterable[Tuple[Optional[AngleChoice], Optional[AngleChoice]]],
        angle_stage: str,
        skip_cached_zero: bool = False,
    ) -> Optional[PlannedRound]:
        nonlocal pair_trials_used
        for choice1, choice2 in pairs:
            if skip_cached_zero and zero_stage2(choice1) and zero_stage2(choice2):
                continue
            if max_pairs and pair_trials_used >= max_pairs:
                return None
            pair_trials_used += 1
            trial = pair_trials_used
            seq1 = ([] if choice1 is None else make_arm_sequence(
                item1_local, place1_local, choice1, pp, item_idx,
                transforms[0], "arm1_"
            ))
            seq2 = ([] if choice2 is None else make_arm_sequence(
                item2_local, place2_local, choice2, pp, item_idx,
                transforms[1], "arm2_"
            ))
            if seq1 and seq2:
                delays = delay_candidates(
                    int(dc.get("start_delay_stages", 2)),
                    int(dc.get("max_start_delay_stages", 4)), len(seq1),
                )
            else:
                delays = [0]
            for delay in delays:
                c1_txt = "-" if choice1 is None else str(choice1.to_dict())
                c2_txt = "-" if choice2 is None else str(choice2.to_dict())
                print(f"    [TRY {trial} {angle_stage}] delay={delay}  "
                      f"arm1={c1_txt}  arm2={c2_txt}", flush=True)
                if bool(cr.get("prescreen_by_ik", True)):
                    pre_ok, pre = prescreen_dual_schedule(
                        mg, q_start, seq1, seq2, delay, cr, q_lo, q_hi,
                        ee_links,
                        float(dc.get("inactive_joint_tolerance_deg", 1.0)),
                    )
                    # 搜索过程可能有大量候选；不把每个候选的 12-DOF
                    # 数组重复写进 failures。连续流水入口会单独保存最佳
                    # trace，旧的单轮入口只保留标量失败摘要。
                    pre.pop("ik_trace", None)
                    if not pre_ok:
                        fail = {
                            "trial": trial, "angle_stage": angle_stage,
                            "delay_stages": delay,
                            "stage": "joint_ik_prescreen",
                            "arm1_choice": (
                                None if choice1 is None else choice1.to_dict()
                            ),
                            "arm2_choice": (
                                None if choice2 is None else choice2.to_dict()
                            ),
                            **pre,
                        }
                        failures.append(fail)
                        print(f"      -> 独立 IK 组合预筛失败: {pre.get('status')} "
                              f"@ {pre.get('targets')}")
                        continue
                    print(f"      -> 独立 IK 组合预筛通过: {pre['n_events']} 路点, "
                          f"最大关节变化 {pre['max_joint_delta_deg']:.1f}deg")
                ok, result = plan_schedule(
                    mg, q_start, seq1, seq2, delay, pl, cr, linear_cfg,
                    q_lo, q_hi, ee_links,
                    float(dc.get("inactive_joint_tolerance_deg", 1.0)),
                )
                if ok:
                    return PlannedRound(
                        position=result["position"], velocity=result["velocity"],
                        acceleration=result["acceleration"], dt=result["dt"],
                        q_end=result["q_end"], segments=result["segments"],
                        arm1_choice=choice1, arm2_choice=choice2,
                        delay_stages=delay,
                        arm1_sequence=seq1, arm2_sequence=seq2,
                    )
                fail = {
                    "trial": trial, "angle_stage": angle_stage,
                    "delay_stages": delay,
                    "arm1_choice": (
                        None if choice1 is None else choice1.to_dict()
                    ),
                    "arm2_choice": (
                        None if choice2 is None else choice2.to_dict()
                    ),
                    **{k: v for k, v in result.items() if k != "segments"},
                }
                failures.append(fail)
                print(f"      -> 失败: {fail.get('status')} "
                      f"@ {fail.get('targets')}")
        return None

    if c1 and c2:
        primary_pairs: Iterable[
            Tuple[Optional[AngleChoice], Optional[AngleChoice]]
        ] = paired_angle_choices(c1, c2)
    elif c1:
        primary_pairs = ((choice, None) for choice in c1)
    else:
        primary_pairs = ((None, choice) for choice in c2)

    base_round = attempt_pairs(primary_pairs, "primary")
    if base_round is None:
        return None, failures

    if not bool((asr.get("stage2") or {}).get("enable", False)):
        return base_round, failures
    if max_pairs and pair_trials_used >= max_pairs:
        print("    [S2] 已达到 max_pair_angle_trials，沿用第一阶段结果")
        return base_round, failures

    def stage2_choices(
        primary: Optional[AngleChoice], reuse: Optional[AngleChoice]
    ) -> List[AngleChoice]:
        if primary is None:
            return []
        secondary = list(stage2_combos(asr))
        if reuse is not None and reuse.primary == primary.primary:
            reused = (reuse.stage2_grasp, reuse.stage2_place)
            secondary = [reused] + [x for x in secondary if x != reused]
        return [
            AngleChoice(primary.grasp, primary.place, a2g, a2p)
            for a2g, a2p in secondary
        ]

    s1 = stage2_choices(base_round.arm1_choice, last_choices[0])
    s2 = stage2_choices(base_round.arm2_choice, last_choices[1])
    if s1 and s2:
        stage2_pairs: Iterable[
            Tuple[Optional[AngleChoice], Optional[AngleChoice]]
        ] = paired_stage2_choices(s1, s2)
    elif s1:
        stage2_pairs = ((choice, None) for choice in s1)
    else:
        stage2_pairs = ((None, choice) for choice in s2)

    refined_round = attempt_pairs(
        stage2_pairs, "stage2", skip_cached_zero=True
    )
    if refined_round is not None:
        return refined_round, failures
    print("    [S2] 二阶段联合候选无可行解，回退到第一阶段结果 (a2=0)")
    return base_round, failures


# ============================== 连续跨物料流水搜索 ==============================


def _stage2_variants(
    primary: AngleChoice,
    asr: Dict[str, Any],
    reuse: Optional[AngleChoice] = None,
) -> List[AngleChoice]:
    """固定一阶段角，返回非零优先、双零兜底的二阶段候选。"""
    increments = list(stage2_combos(asr))
    if reuse is not None and reuse.primary == primary.primary:
        reused = (reuse.stage2_grasp, reuse.stage2_place)
        increments = [reused] + [x for x in increments if x != reused]
    choices = [
        AngleChoice(primary.grasp, primary.place, grasp2, place2)
        for grasp2, place2 in increments
    ]
    nonzero = [
        x for x in choices
        if abs(x.stage2_grasp) > 1e-9 or abs(x.stage2_place) > 1e-9
    ]
    zero = [x for x in choices if x not in nonzero]
    return nonzero + zero


def _assemble_pipeline(
    chunks: Sequence[PlannedRound],
    arm1_choices: Sequence[AngleChoice],
    arm2_choices: Sequence[AngleChoice],
    arm1_sequences: Sequence[Sequence[PoseSpec]],
    arm2_sequences: Sequence[Sequence[PoseSpec]],
    phase: int,
    angle_stage: str,
    search_nodes: int,
    block_attempts: Dict[str, int],
    expected_start_q: Optional[np.ndarray] = None,
    arm2_terminal: Optional[PoseSpec] = None,
) -> PlannedPipeline:
    """严格校验并拼接 speculative prime/rolling chunks。"""
    if not chunks:
        raise ValueError("连续流水没有任何规划 chunk")
    if len(arm1_sequences) != len(arm1_choices) or len(arm2_sequences) != len(arm2_choices):
        raise ValueError("连续流水 choice/sequence 数量不一致")
    if not arm1_sequences or not arm1_sequences[0]:
        raise ValueError("连续流水工艺序列为空")

    sequence_len = len(arm1_sequences[0])
    if sequence_len <= 1 or not 1 <= int(phase) < sequence_len:
        raise ValueError(f"非法连续流水 phase={phase}, sequence_len={sequence_len}")
    for arm_name, sequences in (("arm1", arm1_sequences), ("arm2", arm2_sequences)):
        bad = [i for i, sequence in enumerate(sequences) if len(sequence) != sequence_len]
        if bad:
            raise ValueError(f"{arm_name} 工艺序列长度不一致: item={bad}")

    expected_width: Optional[int] = None
    dt = float(chunks[0].dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"非法 interpolation dt={dt}")
    pos_parts: List[np.ndarray] = []
    vel_parts: List[np.ndarray] = []
    acc_parts: List[np.ndarray] = []
    segments: List[Dict[str, Any]] = []
    previous_end: Optional[np.ndarray] = None
    event_index = 0
    point_count = 0
    terminal_event = (
        int(phase) + len(arm2_sequences) * sequence_len
        if arm2_terminal is not None else None
    )

    for chunk_index, chunk in enumerate(chunks):
        arrays = tuple(np.asarray(x, dtype=np.float64) for x in (
            chunk.position, chunk.velocity, chunk.acceleration,
        ))
        position, velocity, acceleration = arrays
        if (
            position.ndim != 2
            or position.shape[0] == 0
            or velocity.shape != position.shape
            or acceleration.shape != position.shape
            or not all(np.isfinite(x).all() for x in arrays)
        ):
            raise ValueError(
                f"pipeline chunk {chunk_index} 数组非法: "
                f"q={position.shape}, v={velocity.shape}, a={acceleration.shape}"
            )
        if expected_width is None:
            expected_width = position.shape[1]
        if position.shape[1] != expected_width:
            raise ValueError(f"pipeline chunk {chunk_index} DOF 不一致")
        if chunk_index == 0 and expected_start_q is not None:
            expected_start = np.asarray(expected_start_q, dtype=np.float64).reshape(-1)
            if (
                expected_start.shape != (expected_width,)
                or not np.isfinite(expected_start).all()
                or not np.allclose(
                    expected_start, position[0], atol=1e-5, rtol=1e-6
                )
            ):
                raise ValueError("pipeline 首点与 q_start 不一致")
        if not np.isfinite(float(chunk.dt)) or abs(float(chunk.dt) - dt) > 1e-12:
            raise ValueError(f"pipeline chunk {chunk_index} interpolation dt 不一致")
        q_end = np.asarray(chunk.q_end, dtype=np.float64).reshape(-1)
        if (
            q_end.shape != (expected_width,)
            or not np.isfinite(q_end).all()
            or not np.allclose(q_end, position[-1], atol=1e-7, rtol=1e-7)
        ):
            raise ValueError(f"pipeline chunk {chunk_index} q_end 与轨迹末点不一致")
        if previous_end is not None and not np.allclose(
            previous_end, position[0], atol=1e-5, rtol=1e-6
        ):
            raise ValueError(f"pipeline chunk {chunk_index - 1}->{chunk_index} 边界不连续")
        previous_end = position[-1]
        chunk_point_start = 0 if chunk_index == 0 else point_count - 1
        part_position = position if chunk_index == 0 else position[1:]
        pos_parts.append(part_position)
        vel_parts.append(velocity if chunk_index == 0 else velocity[1:])
        acc_parts.append(acceleration if chunk_index == 0 else acceleration[1:])
        point_count += int(part_position.shape[0])
        for report in chunk.segments:
            item1 = event_index // sequence_len if event_index < len(arm1_choices) * sequence_len else None
            stage1 = event_index % sequence_len if item1 is not None else None
            shifted2 = event_index - int(phase)
            item2 = shifted2 // sequence_len if 0 <= shifted2 < len(arm2_choices) * sequence_len else None
            stage2 = shifted2 % sequence_len if item2 is not None else None
            enriched = {
                **report,
                "global_event_index": event_index,
                "pipeline_chunk_index": chunk_index,
                "arm1_item_index": item1,
                "arm1_stage_index": stage1,
                "arm2_item_index": item2,
                "arm2_stage_index": stage2,
                "arm2_terminal_home": event_index == terminal_event,
            }
            if "trajectory_point_start" in report:
                enriched["trajectory_point_start"] = (
                    chunk_point_start + int(report["trajectory_point_start"])
                )
            if "trajectory_point_end" in report:
                enriched["trajectory_point_end"] = (
                    chunk_point_start + int(report["trajectory_point_end"])
                )
            segments.append(enriched)
            event_index += 1

    # 用公开的纯时间线生成器复核 rolling block 没有漏、重或重新引入 round barrier。
    timeline_arm2 = [list(sequence) for sequence in arm2_sequences]
    if arm2_terminal is not None:
        timeline_arm2.append([arm2_terminal])
    hold2 = (
        arm2_sequences[0][0] if arm2_sequences
        else arm2_terminal if arm2_terminal is not None
        else arm1_sequences[0][0]
    )
    timeline = scheduled_pipeline_targets(
        arm1_sequences, timeline_arm2,
        arm1_sequences[0][0], hold2, int(phase),
    )
    if event_index != len(timeline):
        raise ValueError(
            f"pipeline 事件数量不一致: chunks={event_index}, timeline={len(timeline)}"
        )
    for index, (target1, target2, active1, active2) in enumerate(timeline):
        report = segments[index]
        actual_active = set(int(x) for x in report.get("active_arms", []))
        expected_active = ({1} if active1 else set()) | ({2} if active2 else set())
        if actual_active != expected_active:
            raise ValueError(
                f"pipeline event {index} active arms 不一致: "
                f"actual={sorted(actual_active)}, expected={sorted(expected_active)}"
            )
        targets = list(report.get("targets") or [])
        if len(targets) != 2:
            raise ValueError(f"pipeline event {index} 缺少双 TCP target 元数据")
        if active1 and targets[0] != target1.name:
            raise ValueError(f"pipeline event {index} arm1 target 顺序错误")
        if active2 and targets[1] != target2.name:
            raise ValueError(f"pipeline event {index} arm2 target 顺序错误")

    final_parked = chunks[-1].parked_joint_refs
    if arm2_terminal is not None and final_parked[1] is None:
        raise ValueError("arm2 terminal Home 已排程但未建立固定 parked reference")

    return PlannedPipeline(
        position=np.concatenate(pos_parts, axis=0),
        velocity=np.concatenate(vel_parts, axis=0),
        acceleration=np.concatenate(acc_parts, axis=0),
        dt=dt,
        q_end=np.asarray(chunks[-1].q_end, dtype=np.float64).copy(),
        segments=segments,
        chunks=list(chunks),
        arm1_choices=list(arm1_choices),
        arm2_choices=list(arm2_choices),
        arm1_sequences=[list(x) for x in arm1_sequences],
        arm2_sequences=[list(x) for x in arm2_sequences],
        angle_stage=angle_stage,
        search_nodes=int(search_nodes),
        block_attempts=dict(block_attempts),
        arm2_terminal=arm2_terminal,
        parked_joint_refs=final_parked,
    )


def try_continuous_pipeline(
    mg,
    q_start: np.ndarray,
    arm1_items_local: Sequence[Sequence[float]],
    arm2_items_local: Sequence[Sequence[float]],
    place1_local: Sequence[float],
    place2_local: Sequence[float],
    pp: Dict[str, Any],
    asr: Dict[str, Any],
    dc: Dict[str, Any],
    transforms: Tuple[np.ndarray, np.ndarray],
    pl: Dict[str, Any],
    cr: Dict[str, Any],
    linear_cfg: Dict[str, Any],
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    ee_links: Tuple[str, str],
    ik_capture_out: Optional[Dict[str, Any]] = None,
    arm2_terminal: Optional[PoseSpec] = None,
    first_home_grasp_angles: Tuple[Optional[float], Optional[float]] = (None, None),
    first_home_joint_reference: Optional[np.ndarray] = None,
) -> Tuple[Optional[PlannedPipeline], List[Dict[str, Any]], Dict[str, Any]]:
    """用 prime + rolling blocks + DFS 回溯规划无逐物料屏障的流水线。"""
    n_arm1 = len(arm1_items_local)
    n_arm2 = len(arm2_items_local)
    if not n_arm1:
        raise ValueError("连续流水要求一号臂至少有一个物料")
    terminal_enabled = arm2_terminal is not None and n_arm2 > 0
    # 每个 block 对应 arm1 当前 item 的后半段；二号臂最后一件之后增加
    # 一个 terminal Home 事件，因此可能多出一个 home-only block。
    n_blocks = max(n_arm1, n_arm2 + (1 if terminal_enabled else 0))
    phase = int(dc.get("start_delay_stages", 3))
    max_pairs = int(dc.get("max_pair_angle_trials") or 0)
    max_nodes = int(dc.get("max_pipeline_search_nodes") or 0)
    inactive_tol = float(dc.get("inactive_joint_tolerance_deg", 1.0))
    home_branch_tol = float(dc.get("home_branch_tolerance_deg", 1.0))
    root_joint_reference: Optional[np.ndarray] = None
    if first_home_joint_reference is not None:
        root_joint_reference = np.asarray(
            first_home_joint_reference, dtype=np.float64
        ).reshape(-1)
        expected_shape = np.asarray(q_start, dtype=np.float64).reshape(-1).shape
        if (
            root_joint_reference.shape != expected_shape
            or root_joint_reference.size == 0
            or root_joint_reference.size % 2
            or not np.isfinite(root_joint_reference).all()
        ):
            raise ValueError(
                "first_home_joint_reference 必须是与 q_start 同维的有限偶数向量"
            )
        if not np.isfinite(home_branch_tol) or home_branch_tol < 0.0:
            raise ValueError("home_branch_tolerance_deg 必须是有限非负数")
        start_branch_error = float(np.degrees(np.max(np.abs(
            np.asarray(q_start, dtype=np.float64).reshape(-1)
            - root_joint_reference
        ))))
        if start_branch_error > home_branch_tol:
            raise ValueError(
                "q_start 不是选中的 Home IK 分支: "
                f"max_error={start_branch_error:.6g}deg > {home_branch_tol:g}deg"
            )
    failures: List[Dict[str, Any]] = []
    block_attempts: Dict[str, int] = {}
    search_nodes = 0
    node_budget_exhausted = False
    pair_budget_exhausted = False
    best_ik_capture: Optional[Dict[str, Any]] = None
    if ik_capture_out is not None:
        ik_capture_out.clear()

    first_primary = arm_primary_choices(asr)
    if not first_primary:
        return None, [], {"reason": "no_angle_candidates", "search_nodes": 0}
    probe = make_arm_sequence(
        arm1_items_local[0], place1_local, first_primary[0], pp, 0,
        transforms[0], "arm1_",
    )
    sequence_len = len(probe)
    if not 1 <= phase < sequence_len:
        raise ValueError(
            f"连续流水 phase 必须满足 1 <= phase < {sequence_len}，实际 {phase}"
        )
    expected_global_events = max(
        n_arm1 * sequence_len,
        phase + n_arm2 * sequence_len + (1 if terminal_enabled else 0),
    )

    def consider_ik_trace(
        trace: Optional[Dict[str, Any]],
        block_key: str,
        angle_stage: str,
        diagnostic: Optional[Dict[str, Any]],
        prefix_spans: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """只保留搜索中走得最远的一条 IK 路径。

        各 chunk 的 q_start 来自上一个 MotionGen 轨迹末点，不一定等于
        上一个 IK span 的末点；因此保留 span 边界，不伪装它们
        是一条经过验证的连续轨迹。
        """
        nonlocal best_ik_capture
        if not isinstance(trace, dict):
            return None
        raw_positions = trace.get("joint_positions_rad")
        if not isinstance(raw_positions, list) or not raw_positions:
            return None
        global_start = int((diagnostic or {}).get("global_event_start", 0))
        span = {
            "pipeline_block": block_key,
            "angle_stage": angle_stage,
            "global_event_start": global_start,
            "joint_positions_rad": raw_positions,
            "start_poses": trace.get("start_poses"),
            "events": list(trace.get("events") or []),
            "n_events": int(trace.get("n_events", 0)),
            "n_solved_events": int(trace.get("n_solved_events", 0)),
            "n_accepted_events": int(trace.get("n_accepted_events", 0)),
            "deepest_attempted_event": trace.get("deepest_attempted_event"),
            "deepest_solved_event": trace.get("deepest_solved_event"),
            "complete": bool(trace.get("complete", False)),
            "status": str(trace.get("status", "IK_PRESCREEN_UNKNOWN")),
            "diagnostic": dict(diagnostic or {}),
        }
        spans = list(prefix_spans) + [span]
        accepted_global = sorted({
            int(item["global_event_start"]) + int(event["event"])
            for item in spans
            for event in item.get("events", [])
            if event.get("accepted") is True
        })
        solved_global = sorted({
            int(item["global_event_start"]) + int(event["event"])
            for item in spans
            for event in item.get("events", [])
            if event.get("solve_success") is True
        })
        attempted_global = sorted({
            int(item["global_event_start"]) + int(event["event"])
            for item in spans
            for event in item.get("events", [])
        })
        full_ik = (
            expected_global_events > 0
            and set(range(expected_global_events)).issubset(accepted_global)
        )
        rank = (
            int(full_ik),
            accepted_global[-1] if accepted_global else -1,
            solved_global[-1] if solved_global else -1,
            attempted_global[-1] if attempted_global else -1,
            len(accepted_global),
        )
        current_rank = (
            tuple(best_ik_capture.get("_rank", ()))
            if best_ik_capture is not None else ()
        )
        if best_ik_capture is None or rank > current_rank:
            best_ik_capture = {
                "_rank": list(rank),
                "full_ik": bool(full_ik),
                "expected_global_events": int(expected_global_events),
                "deepest_accepted_global_event": (
                    accepted_global[-1] if accepted_global else None
                ),
                "deepest_solved_global_event": (
                    solved_global[-1] if solved_global else None
                ),
                "deepest_attempted_global_event": (
                    attempted_global[-1] if attempted_global else None
                ),
                "accepted_global_events": accepted_global,
                "solved_global_events": solved_global,
                "spans": spans,
            }
            if ik_capture_out is not None:
                ik_capture_out.clear()
                ik_capture_out.update(best_ik_capture)
        return span

    sequence_cache: Dict[Tuple[int, int, AngleChoice], List[PoseSpec]] = {}

    def sequence_for(arm: int, item_index: int, choice: AngleChoice) -> List[PoseSpec]:
        key = (arm, item_index, choice)
        if key not in sequence_cache:
            items = arm1_items_local if arm == 1 else arm2_items_local
            place = place1_local if arm == 1 else place2_local
            sequence_cache[key] = make_arm_sequence(
                items[item_index], place, choice, pp, item_index,
                transforms[arm - 1], f"arm{arm}_",
            )
        sequence = sequence_cache[key]
        if len(sequence) != sequence_len:
            raise ValueError(
                f"arm{arm} item{item_index} 工艺路点数 {len(sequence)} != {sequence_len}"
            )
        return sequence

    def candidates_for(
        arm: int,
        item_index: int,
        previous: Optional[AngleChoice],
        angle_stage: str,
        baseline: Optional[Tuple[Sequence[AngleChoice], Sequence[AngleChoice]]],
    ) -> List[AngleChoice]:
        if angle_stage == "primary":
            candidates = arm_primary_choices(asr, previous)
        else:
            if baseline is None:
                raise ValueError("stage2 搜索缺少 primary baseline")
            primary = baseline[arm - 1][item_index]
            candidates = _stage2_variants(primary, asr, previous)

        # shared 动态 Home 已经用首件预抓取姿态求出 q_start。首件必须沿用
        # 同一个 primary grasp 角；二阶段也不再改变首件 grasp 姿态，否则
        # “Home=首件预抓取”会在轨迹第一个事件被悄悄旋转成另一姿态。
        locked_grasp = first_home_grasp_angles[arm - 1]
        if item_index == 0 and locked_grasp is not None:
            candidates = [
                choice for choice in candidates
                if abs(float(choice.grasp) - float(locked_grasp)) < 1e-9
                and (
                    angle_stage != "stage2"
                    or abs(float(choice.stage2_grasp)) < 1e-9
                )
            ]
        return candidates

    def can_spend(block_key: str) -> bool:
        nonlocal search_nodes, node_budget_exhausted
        if max_nodes and search_nodes >= max_nodes:
            node_budget_exhausted = True
            return False
        search_nodes += 1
        block_attempts[block_key] = block_attempts.get(block_key, 0) + 1
        return True

    def check_home_branch_endpoint(
        result: Dict[str, Any], arm: int, event_index: int, context: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """检查首件锚点/终点 Home 仍为选中根的同一关节分支。"""
        if root_joint_reference is None:
            return True, {}
        position = np.asarray(result.get("position"), dtype=np.float64)
        reports = result.get("segments")
        if (
            position.ndim != 2
            or position.shape[1] != root_joint_reference.size
            or not np.isfinite(position).all()
            or not isinstance(reports, list)
            or event_index < 0
            or event_index >= len(reports)
            or not isinstance(reports[event_index], dict)
            or "trajectory_point_end" not in reports[event_index]
        ):
            return False, {
                "status": "HOME_BRANCH_CHECK_INVALID",
                "home_branch_context": context,
                "arm": int(arm),
                "reason": "轨迹或 segment 端点元数据不完整",
            }
        point = int(reports[event_index]["trajectory_point_end"])
        if point < 0 or point >= position.shape[0]:
            return False, {
                "status": "HOME_BRANCH_CHECK_INVALID",
                "home_branch_context": context,
                "arm": int(arm),
                "trajectory_point_end": point,
                "n_trajectory_points": int(position.shape[0]),
            }
        arm_dof = root_joint_reference.size // 2
        arm_index = int(arm) - 1
        sl = slice(arm_index * arm_dof, (arm_index + 1) * arm_dof)
        error = float(np.degrees(np.max(np.abs(
            position[point, sl] - root_joint_reference[sl]
        ))))
        report = {
            "home_branch_context": context,
            "arm": int(arm),
            "home_branch_error_deg": error,
            "home_branch_tolerance_deg": home_branch_tol,
            "trajectory_point_end": point,
        }
        if error > home_branch_tol:
            return False, {"status": "HOME_BRANCH_MISMATCH", **report}
        if context == "arm2_terminal_home":
            hold_error = float(np.degrees(np.max(np.abs(
                position[point:, sl] - root_joint_reference[None, sl]
            ))))
            report["home_branch_post_terminal_hold_error_deg"] = hold_error
            if hold_error > home_branch_tol:
                return False, {
                    "status": "HOME_BRANCH_HOLD_MISMATCH",
                    **report,
                }
        return True, report

    def attempt_chunk(
        block_key: str,
        angle_stage: str,
        q_begin: np.ndarray,
        seq1: Sequence[PoseSpec],
        seq2: Sequence[PoseSpec],
        choice1: Optional[AngleChoice],
        choice2: Optional[AngleChoice],
        parked_refs: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
        diagnostic: Optional[Dict[str, Any]] = None,
        ik_trace_prefix: Sequence[Dict[str, Any]] = (),
    ) -> Optional[PlannedRound]:
        if not can_spend(block_key):
            return None
        trial = block_attempts[block_key]
        print(
            f"    [PIPE {angle_stage} {block_key} TRY {trial}] "
            f"arm1={None if choice1 is None else choice1.to_dict()}  "
            f"arm2={None if choice2 is None else choice2.to_dict()}",
            flush=True,
        )
        park_after = (
            None,
            arm2_terminal.name
            if (
                arm2_terminal is not None
                and len(seq2) == 1
                and seq2[0] is arm2_terminal
            ) else None,
        )
        captured_span: Optional[Dict[str, Any]] = None
        if bool(cr.get("prescreen_by_ik", True)):
            pre_ok, pre = prescreen_dual_schedule(
                mg, q_begin, seq1, seq2, 0, cr, q_lo, q_hi,
                ee_links, inactive_tol, parked_refs, park_after,
            )
            trace = pre.pop("ik_trace", None)
            captured_span = consider_ik_trace(
                trace, block_key, angle_stage, diagnostic, ik_trace_prefix
            )
            if not pre_ok:
                pair_diag = (
                    (pre.get("ik_report") or {}).get("pair_report") or {}
                )
                collision_diag = pair_diag.get("collision") or {}
                print(
                    "      [IK PRESCREEN FAIL] "
                    f"status={pre.get('status')} event={pre.get('event')} "
                    f"reason={pre.get('reason', '')} "
                    f"pairs={pair_diag.get('n_pair_trials')} "
                    f"self={collision_diag.get('n_self_or_inter_arm_collision')} "
                    f"world={collision_diag.get('n_world_collision')} "
                    f"margin={collision_diag.get('n_inter_arm_margin_collision')}",
                    flush=True,
                )
                failures.append({
                    "pipeline_block": block_key,
                    "trial": trial,
                    "angle_stage": angle_stage,
                    "stage": "joint_ik_prescreen",
                    "arm1_choice": None if choice1 is None else choice1.to_dict(),
                    "arm2_choice": None if choice2 is None else choice2.to_dict(),
                    **(diagnostic or {}),
                    **pre,
                })
                return None
        ok, result = plan_schedule(
            mg, q_begin, seq1, seq2, 0, pl, cr, linear_cfg,
            q_lo, q_hi, ee_links, inactive_tol, parked_refs, park_after,
        )
        if not ok:
            print(
                "      [PLAN SCHEDULE FAIL] "
                f"status={result.get('status')} "
                f"event={result.get('failed_event')} "
                f"targets={result.get('targets')} ",
                flush=True,
            )
            failures.append({
                "pipeline_block": block_key,
                "trial": trial,
                "angle_stage": angle_stage,
                "arm1_choice": None if choice1 is None else choice1.to_dict(),
                "arm2_choice": None if choice2 is None else choice2.to_dict(),
                **(diagnostic or {}),
                **{k: v for k, v in result.items() if k != "segments"},
            })
            return None

        branch_checks: List[Tuple[int, int, str]] = []
        if block_key == "prime" and seq1:
            branch_checks.append((1, 0, "arm1_first_pregrasp_anchor"))
        if block_key == "block_0" and seq2:
            branch_checks.append((2, 0, "arm2_first_pregrasp_anchor"))
        if park_after[1] is not None:
            branch_checks.append((2, 0, "arm2_terminal_home"))
        for arm, event_index, context in branch_checks:
            branch_ok, branch_report = check_home_branch_endpoint(
                result, arm, event_index, context
            )
            if not branch_ok:
                failures.append({
                    "pipeline_block": block_key,
                    "trial": trial,
                    "angle_stage": angle_stage,
                    "stage": "home_branch_check",
                    "arm1_choice": (
                        None if choice1 is None else choice1.to_dict()
                    ),
                    "arm2_choice": (
                        None if choice2 is None else choice2.to_dict()
                    ),
                    **(diagnostic or {}),
                    **branch_report,
                })
                return None
            if branch_report:
                result["segments"][event_index]["home_branch_check"] = (
                    branch_report
                )
        if block_key == "prime" and root_joint_reference is not None:
            prime_position = np.asarray(result["position"], dtype=np.float64)
            arm_dof = root_joint_reference.size // 2
            arm2_slice = slice(arm_dof, 2 * arm_dof)
            waiting_error = float(np.degrees(np.max(np.abs(
                prime_position[:, arm2_slice]
                - root_joint_reference[None, arm2_slice]
            ))))
            waiting_report = {
                "home_branch_context": "arm2_pre_activation_hold",
                "arm": 2,
                "home_branch_hold_error_deg": waiting_error,
                "home_branch_tolerance_deg": home_branch_tol,
            }
            if waiting_error > home_branch_tol:
                failures.append({
                    "pipeline_block": block_key,
                    "trial": trial,
                    "angle_stage": angle_stage,
                    "stage": "home_branch_check",
                    "status": "HOME_BRANCH_HOLD_MISMATCH",
                    "arm1_choice": (
                        None if choice1 is None else choice1.to_dict()
                    ),
                    "arm2_choice": (
                        None if choice2 is None else choice2.to_dict()
                    ),
                    **(diagnostic or {}),
                    **waiting_report,
                })
                return None
            if result["segments"]:
                result["segments"][0]["arm2_pre_activation_home_hold"] = (
                    waiting_report
                )
        next_parked = result.get("parked_joint_refs", parked_refs)
        if park_after[1] is not None:
            # 后续 hold 固定到根节点的期望分支，不把姿态规划器偶然到达的
            # 另一条 IK 分支当成 Home。上面已确认实际端点在容差内。
            terminal_reference = (
                np.asarray(result["q_end"], dtype=np.float64).copy()
                if root_joint_reference is None
                else root_joint_reference.copy()
            )
            next_parked = (next_parked[0], terminal_reference)
        return PlannedRound(
            position=result["position"],
            velocity=result["velocity"],
            acceleration=result["acceleration"],
            dt=result["dt"],
            q_end=result["q_end"],
            segments=result["segments"],
            arm1_choice=choice1,
            arm2_choice=choice2,
            delay_stages=0,
            arm1_sequence=list(seq1),
            arm2_sequence=list(seq2),
            parked_joint_refs=next_parked,
            ik_trace=captured_span,
        )

    def run_search(
        angle_stage: str,
        baseline: Optional[Tuple[Sequence[AngleChoice], Sequence[AngleChoice]]] = None,
    ) -> Optional[PlannedPipeline]:
        nonlocal pair_budget_exhausted

        def search_block(
            block_index: int,
            q_begin: np.ndarray,
            current_a: Optional[AngleChoice],
            previous_b: Optional[AngleChoice],
            parked_refs: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
            chunks: List[PlannedRound],
            choices_a: List[AngleChoice],
            choices_b: List[AngleChoice],
        ) -> Optional[Tuple[List[PlannedRound], List[AngleChoice], List[AngleChoice]]]:
            nonlocal pair_budget_exhausted
            block_key = f"block_{block_index}"
            has_b_item = block_index < n_arm2
            candidates_b: List[Optional[AngleChoice]] = (
                list(candidates_for(
                    2, block_index, previous_b, angle_stage, baseline
                )) if has_b_item else [None]
            )
            has_next_a = block_index + 1 < n_arm1
            candidates_next_a = (
                candidates_for(
                    1, block_index + 1, current_a, angle_stage, baseline
                )
                if has_next_a else [None]
            )
            if has_next_a and has_b_item:
                if angle_stage == "stage2":
                    pairs: Iterable[Tuple[Optional[AngleChoice], Optional[AngleChoice]]] = (
                        paired_stage2_choices(candidates_next_a, candidates_b)
                    )
                else:
                    pairs = paired_angle_choices(candidates_next_a, candidates_b)
            elif has_next_a:
                pairs = ((choice_a, None) for choice_a in candidates_next_a)
            elif has_b_item:
                pairs = ((None, choice_b) for choice_b in candidates_b)
            else:
                pairs = ((None, None),)

            if block_index < n_arm1:
                if current_a is None:
                    raise ValueError(f"block_{block_index} 缺少 arm1 current choice")
                current_full = sequence_for(1, block_index, current_a)
            else:
                current_full = []
            # 角度对上限属于“当前上游状态下的一次 block 搜索”。回溯后
            # q_begin 已改变，必须重新给候选额度；跨分支的总成本由全局
            # max_pipeline_search_nodes 限制。block_attempts 只做累计 telemetry。
            local_pair_trials = 0
            for next_a, choice_b in pairs:
                if node_budget_exhausted:
                    return None
                if max_pairs and local_pair_trials >= max_pairs:
                    pair_budget_exhausted = True
                    break
                local_pair_trials += 1
                seq1 = list(current_full[phase:])
                if next_a is not None:
                    seq1.extend(sequence_for(1, block_index + 1, next_a)[:phase])
                if choice_b is not None:
                    seq2 = sequence_for(2, block_index, choice_b)
                elif terminal_enabled and block_index == n_arm2:
                    seq2 = [arm2_terminal]
                else:
                    seq2 = []
                planned = attempt_chunk(
                    block_key, angle_stage, q_begin, seq1, seq2,
                    next_a if next_a is not None else current_a, choice_b,
                    parked_refs,
                    diagnostic={
                        "global_event_start": phase + block_index * sequence_len,
                        "arm1_current_item_index": (
                            block_index if block_index < n_arm1 else None
                        ),
                        "arm1_current_choice": (
                            None if current_a is None else current_a.to_dict()
                        ),
                        "arm1_next_item_index": (
                            block_index + 1 if has_next_a else None
                        ),
                        "arm1_next_choice": (
                            None if next_a is None else next_a.to_dict()
                        ),
                        "arm2_item_index": block_index if has_b_item else None,
                        "arm2_item_choice": (
                            None if choice_b is None else choice_b.to_dict()
                        ),
                        "arm2_terminal_home": bool(
                            terminal_enabled and block_index == n_arm2
                        ),
                    },
                    ik_trace_prefix=[
                        chunk.ik_trace for chunk in chunks
                        if chunk.ik_trace is not None
                    ],
                )
                if planned is None:
                    continue
                next_chunks = chunks + [planned]
                next_choices_b = choices_b + ([choice_b] if choice_b is not None else [])
                next_choices_a = choices_a + ([next_a] if next_a is not None else [])
                if block_index + 1 >= n_blocks:
                    return next_chunks, next_choices_a, next_choices_b
                downstream = search_block(
                    block_index + 1,
                    planned.q_end,
                    next_a if has_next_a else None,
                    choice_b if choice_b is not None else previous_b,
                    planned.parked_joint_refs,
                    next_chunks,
                    next_choices_a,
                    next_choices_b,
                )
                if downstream is not None:
                    return downstream
                failures.append({
                    "pipeline_block": block_key,
                    "trial": block_attempts.get(block_key, 0),
                    "angle_stage": angle_stage,
                    "status": "DOWNSTREAM_BACKTRACK",
                    "global_event_start": phase + block_index * sequence_len,
                    "arm1_current_choice": (
                        None if current_a is None else current_a.to_dict()
                    ),
                    "arm1_next_choice": (
                        None if next_a is None else next_a.to_dict()
                    ),
                    "arm2_item_choice": (
                        None if choice_b is None else choice_b.to_dict()
                    ),
                    "arm2_terminal_home": bool(
                        terminal_enabled and block_index == n_arm2
                    ),
                })
            return None

        prime_candidates = candidates_for(1, 0, None, angle_stage, baseline)
        local_prime_trials = 0
        for choice_a0 in prime_candidates:
            if node_budget_exhausted:
                return None
            if max_pairs and local_prime_trials >= max_pairs:
                pair_budget_exhausted = True
                break
            local_prime_trials += 1
            full_a0 = sequence_for(1, 0, choice_a0)
            prime = attempt_chunk(
                "prime", angle_stage, q_start, full_a0[:phase], [], choice_a0, None,
                (None, None),
                diagnostic={
                    "global_event_start": 0,
                    "arm1_current_choice": choice_a0.to_dict(),
                    "arm1_next_choice": None,
                    "arm2_item_choice": None,
                },
                ik_trace_prefix=[],
            )
            if prime is None:
                continue
            found = search_block(
                0, prime.q_end, choice_a0, None,
                prime.parked_joint_refs,
                [prime], [choice_a0], [],
            )
            if found is None:
                failures.append({
                    "pipeline_block": "prime",
                    "trial": block_attempts.get("prime", 0),
                    "angle_stage": angle_stage,
                    "status": "DOWNSTREAM_BACKTRACK",
                    "global_event_start": 0,
                    "arm1_current_choice": choice_a0.to_dict(),
                    "arm1_next_choice": None,
                    "arm2_item_choice": None,
                })
                continue
            chunks, choices_a, choices_b = found
            sequences_a = [
                sequence_for(1, index, choice)
                for index, choice in enumerate(choices_a)
            ]
            sequences_b = [
                sequence_for(2, index, choice)
                for index, choice in enumerate(choices_b)
            ]
            return _assemble_pipeline(
                chunks, choices_a, choices_b, sequences_a, sequences_b,
                phase, angle_stage, search_nodes, block_attempts,
                expected_start_q=q_start,
                arm2_terminal=arm2_terminal if terminal_enabled else None,
            )
        return None

    def failure_summary() -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for failure in failures:
            status = str(failure.get("status") or failure.get("stage") or "UNKNOWN")
            counts[status] = counts.get(status, 0) + 1
        return {
            "n_failed_attempts": len(failures),
            "n_backtracks": counts.get("DOWNSTREAM_BACKTRACK", 0),
            "failure_status_counts": counts,
        }

    primary = run_search("primary")
    if primary is None:
        return None, failures, {
            "phase_stages": phase,
            "sequence_len": sequence_len,
            "n_arm1_items": n_arm1,
            "n_arm2_items": n_arm2,
            "arm2_terminal_home": terminal_enabled,
            "first_home_grasp_angles": list(first_home_grasp_angles),
            "expected_global_events": max(
                n_arm1 * sequence_len,
                phase + n_arm2 * sequence_len + (1 if terminal_enabled else 0),
            ),
            "search_nodes": search_nodes,
            "max_search_nodes": max_nodes,
            "node_budget_exhausted": node_budget_exhausted,
            "pair_budget_exhausted": pair_budget_exhausted,
            "block_attempts": dict(block_attempts),
            "stage2_status": "not_run_primary_failed",
            **failure_summary(),
        }

    chosen = primary
    stage2_on = bool((asr.get("stage2") or {}).get("enable", False))
    stage2_status = "disabled"
    if stage2_on and not node_budget_exhausted:
        stage2_status = "searching"
        baseline = (primary.arm1_choices, primary.arm2_choices)
        refined = run_search("stage2", baseline)
        if refined is not None:
            all_choices = refined.arm1_choices + refined.arm2_choices
            if any(
                abs(choice.stage2_grasp) > 1e-9
                or abs(choice.stage2_place) > 1e-9
                for choice in all_choices
            ):
                chosen = refined
                stage2_status = "selected_nonzero_refinement"
                print("    [PIPE S2] 连续流水二阶段联合细搜成功")
            else:
                stage2_status = "all_zero_refinement_kept_primary"
        elif node_budget_exhausted:
            stage2_status = "node_budget_exhausted_kept_primary"
        else:
            stage2_status = "no_feasible_refinement_kept_primary"
        if chosen is primary:
            print("    [PIPE S2] 二阶段连续流水无可行细化，沿用 primary baseline")
    elif stage2_on:
        stage2_status = "not_run_node_budget_exhausted"

    chosen.search_nodes = search_nodes
    chosen.block_attempts = dict(block_attempts)
    return chosen, failures, {
        "phase_stages": phase,
        "sequence_len": sequence_len,
        "n_arm1_items": n_arm1,
        "n_arm2_items": n_arm2,
        "arm2_terminal_home": terminal_enabled,
        "first_home_grasp_angles": list(first_home_grasp_angles),
        "expected_global_events": max(
            n_arm1 * sequence_len,
            phase + n_arm2 * sequence_len + (1 if terminal_enabled else 0),
        ),
        "search_nodes": search_nodes,
        "max_search_nodes": max_nodes,
        "node_budget_exhausted": node_budget_exhausted,
        "pair_budget_exhausted": pair_budget_exhausted,
        "block_attempts": dict(block_attempts),
        "angle_stage": chosen.angle_stage,
        "stage2_status": stage2_status,
        **failure_summary(),
    }


# ============================== 报告 ==============================


def workspace_report(points_root: np.ndarray, root_to_local: np.ndarray,
                     ws: Dict[str, Any]) -> Dict[str, Any]:
    local = transform_points(points_root, root_to_local)
    inside, violation = check_in_bounds(local, ws, margin=float(ws.get("check_margin", 0.0)))
    return {
        "checked": True,
        "n_points": int(local.shape[0]),
        "n_violation": int((~inside).sum()),
        "max_violation_mm": float(violation.max() * 1000.0) if violation.size else 0.0,
        "bbox_local_min": local.min(axis=0).tolist(),
        "bbox_local_max": local.max(axis=0).tolist(),
    }


def link_sphere_extent_report(
    mg, positions: np.ndarray, times: np.ndarray, link: str,
    root_to_local: np.ndarray, ws: Dict[str, Any], arm_name: str,
    batch_size: int = 4096,
) -> Dict[str, Any]:
    """按各臂本地 workspace 报告法兰/夹爪碰撞球包围范围。"""
    try:
        kin_cfg = mg.kinematics.kinematics_config
        name_to_idx = dict(kin_cfg.link_name_to_idx_map)
        if link not in name_to_idx:
            return {"checked": False, "link": link, "reason": "link 不在运动树"}
        idx_map = kin_cfg.link_sphere_idx_map.cpu().numpy()
        mask = idx_map == int(name_to_idx[link])
        if not mask.any():
            return {"checked": False, "link": link, "reason": "link 没有碰撞球"}

        bounds = ws["bounds"]
        lo = np.asarray([bounds[a][0] for a in ("x", "y", "z")], dtype=np.float64)
        hi = np.asarray([bounds[a][1] for a in ("x", "y", "z")], dtype=np.float64)
        bbox_lo = np.full(3, np.inf, dtype=np.float64)
        bbox_hi = np.full(3, -np.inf, dtype=np.float64)
        violation_chunks: List[np.ndarray] = []

        for start in range(0, positions.shape[0], batch_size):
            state = mg.kinematics.get_state(mg.tensor_args.to_device(
                positions[start:start + batch_size].astype(np.float32)
            ))
            spheres = (
                state.link_spheres_tensor.detach().cpu().numpy()[:, mask, :]
                .astype(np.float64)
            )
            centers = transform_points(
                spheres[..., :3].reshape(-1, 3), root_to_local
            ).reshape(spheres.shape[0], spheres.shape[1], 3)
            radius = spheres[..., 3:4]
            sphere_lo, sphere_hi = centers - radius, centers + radius
            bbox_lo = np.minimum(bbox_lo, sphere_lo.min(axis=(0, 1)))
            bbox_hi = np.maximum(bbox_hi, sphere_hi.max(axis=(0, 1)))
            under = np.maximum(lo[None, None, :] - sphere_lo, 0.0)
            over = np.maximum(sphere_hi - hi[None, None, :], 0.0)
            violation_chunks.append(np.maximum(under, over).max(axis=1))

        violation_xyz = np.concatenate(violation_chunks, axis=0)
        violation = violation_xyz.max(axis=1)
        worst = int(np.argmax(violation))
        report = {
            "checked": True,
            "arm": arm_name,
            "link": link,
            "frame": f"{arm_name}_LINK_0",
            "n_spheres": int(mask.sum()),
            "bbox_local_min": bbox_lo.tolist(),
            "bbox_local_max": bbox_hi.tolist(),
            "n_points_outside_bounds": int((violation > 1e-9).sum()),
            "max_violation_mm": float(violation[worst] * 1000.0),
            "max_violation_xyz_mm": (violation_xyz.max(axis=0) * 1000.0).tolist(),
            "max_violation_at_t": float(times[worst]),
            "note": "法兰/夹爪碰撞球范围仅报告，不改变 workspace TCP 验收语义",
        }
        print(f"[CHECK] {arm_name} {link} 碰撞球范围(local): "
              f"{report['n_points_outside_bounds']}/{positions.shape[0]} 点越界，"
              f"最大 {report['max_violation_mm']:.1f}mm")
        return report
    except Exception as exc:  # noqa: BLE001 - 信息性报告，不中断主安全检查
        return {"checked": False, "arm": arm_name, "link": link, "reason": str(exc)}


def compute_fk_batched(
    mg, positions: np.ndarray, link_names: Sequence[str], batch_size: int = 4096
) -> Dict[str, np.ndarray]:
    """分批 FK，避免长流水轨迹一次占满 GPU。"""
    pieces: Dict[str, List[np.ndarray]] = {}
    for start in range(0, positions.shape[0], batch_size):
        part = compute_fk(mg, positions[start:start + batch_size], list(link_names))
        for key, value in part.items():
            pieces.setdefault(key, []).append(value)
    return {key: np.concatenate(value, axis=0) for key, value in pieces.items()}


def inter_arm_report(
    mg, positions: np.ndarray, prefix: str, margin_mm: float,
    fk_batch_size: int = 4096,
) -> Dict[str, Any]:
    try:
        idx_map = mg.kinematics.kinematics_config.link_sphere_idx_map.cpu().numpy()
        name_to_idx = dict(mg.kinematics.kinematics_config.link_name_to_idx_map)
        idx_to_name = {int(v): k for k, v in name_to_idx.items()}
        names = [idx_to_name[int(v)] for v in idx_map]
        mask2 = np.array([n.startswith(prefix) for n in names])
        if not mask2.any() or mask2.all():
            return {
                "checked": False,
                "reason": f"前缀 {prefix!r} 无法拆分两臂碰撞球",
            }

        # GPU FK 分批；CPU 球对距离在 check_pair_collisions 内再按时间分块。
        sphere_chunks: List[np.ndarray] = []
        for start in range(0, positions.shape[0], fk_batch_size):
            state = mg.kinematics.get_state(mg.tensor_args.to_device(
                positions[start:start + fk_batch_size].astype(np.float32)
            ))
            sphere_chunks.append(
                state.link_spheres_tensor.detach().cpu().numpy().astype(np.float32)
            )
        spheres = np.concatenate(sphere_chunks, axis=0)
        rep = check_pair_collisions(
            spheres[:, ~mask2], spheres[:, mask2],
            [n for n, m in zip(names, ~mask2) if m],
            [n for n, m in zip(names, mask2) if m],
            float(margin_mm) / 1000.0,
        )
        rep.update({
            "checked": True, "margin_mm": float(margin_mm),
            "mode": "12-DOF combined trajectory (planning-time self collision)",
        })
        return rep
    except Exception as exc:  # noqa: BLE001 - 安全检查失败必须进入失败报告
        return {"checked": False, "reason": str(exc)}


def self_collision_report(
    mg, positions: np.ndarray, batch_size: int = 4096
) -> Dict[str, Any]:
    try:
        if not np.isfinite(positions).all():
            return {"checked": False, "reason": "关节轨迹含 NaN/Inf"}
        cost = None
        for rollout in mg.get_all_rollout_instances():
            for attr in ("robot_self_collision_constraint", "robot_self_collision_cost"):
                candidate = getattr(rollout, attr, None)
                if candidate is not None and getattr(candidate, "enabled", True):
                    cost = candidate
                    break
            if cost is not None:
                break
        if cost is None:
            return {"checked": False, "reason": "self collision cost disabled"}
        n_collision = 0
        for start in range(0, positions.shape[0], batch_size):
            spheres = mg.kinematics.get_state(mg.tensor_args.to_device(
                positions[start:start + batch_size].astype(np.float32)
            )).link_spheres_tensor.unsqueeze(1)
            if not bool(spheres.isfinite().all()):
                return {"checked": False, "reason": "FK 碰撞球含 NaN/Inf"}
            d = cost.forward(spheres).reshape(-1)
            if not bool(d.isfinite().all()):
                return {"checked": False, "reason": "自碰撞代价含 NaN/Inf"}
            n_collision += int((d > 0).sum().item())
        return {
            "checked": True, "n_points": int(positions.shape[0]),
            "n_collision": n_collision,
        }
    except Exception as exc:  # noqa: BLE001
        return {"checked": False, "reason": str(exc)}


IK_PLAYBACK_FRAME_SECONDS = 2.0


def save_home_root_ik_checkpoint(
    out_dir: Path,
    home_roots: Sequence[FirstPregraspHomeRoot],
    joint_names: Sequence[str],
    cfg: Dict[str, Any],
    frame_seconds: float = IK_PLAYBACK_FRAME_SECONDS,
) -> Tuple[Path, Path, int]:
    """在进入耗时的整线搜索前就保存全部 Home IK 根。"""
    if not home_roots:
        raise ValueError("没有可保存的 Home IK root")
    source = Path(out_dir) / "plan_failed.json"
    record = {
        "stage": "home_ik_checkpoint",
        "pipeline_attempts": [
            {
                "attempt": index + 1,
                "home_root_index": index,
                "home_root": root.to_dict(),
            }
            for index, root in enumerate(home_roots)
        ],
        "config": cfg,
    }
    payload, metadata = build_ik_playback_data(record, source, frame_seconds)
    actual_joint_names = [str(name) for name in joint_names]
    if len(actual_joint_names) != 12 or len(set(actual_joint_names)) != 12:
        raise ValueError(
            f"Home IK checkpoint 需要 12 个唯一实际关节名: "
            f"{actual_joint_names}"
        )
    payload["joint_names"] = np.asarray(actual_joint_names, dtype="S64")
    metadata["robot"]["joint_names"] = actual_joint_names
    metadata.update({
        "capture_kind": "home_root_candidates",
        "planning_may_still_be_running": True,
        "warning": (
            "每帧是独立的双臂 Home IK 分支；分支间没有轨迹规划或"
            "连续碰撞检查，只能用于 RViz 可视化，禁止下发机器人。"
        ),
    })
    return write_ik_playback_artifact(
        Path(out_dir) / "ik_playback", payload, metadata
    )


def save_prescreen_ik_checkpoint(
    mg,
    out_dir: Path,
    joint_names: Sequence[str],
    ee_links: Tuple[str, str],
    cfg: Dict[str, Any],
    captures: Sequence[Dict[str, Any]],
    frame_seconds: float = IK_PLAYBACK_FRAME_SECONDS,
) -> Tuple[Path, Path, int]:
    """保存所有已完成 Home root 中走得最远的独立 IK 组合路径。

    不同 chunk 保留为独立 span；播放时虽会逐帧切换，但
    metadata 明确标记所有 span 边界和未验证的转换。
    """
    frame_seconds = float(frame_seconds)
    if not np.isfinite(frame_seconds) or frame_seconds <= 0.0:
        raise ValueError("frame_seconds 必须是有限正数")
    valid_captures = [
        capture for capture in captures
        if isinstance(capture, dict) and capture.get("spans")
    ]
    if not valid_captures:
        raise ValueError("没有可保存的 prescreen IK trace")
    best = max(
        valid_captures,
        key=lambda capture: tuple(int(x) for x in capture.get("_rank", [])),
    )
    root_index_raw = best.get("home_root_index")
    root_index = -1 if root_index_raw is None else int(root_index_raw)
    positions_parts: List[np.ndarray] = []
    span_index_values: List[int] = []
    span_state_values: List[int] = []
    global_event_values: List[int] = []
    accepted_values: List[bool] = []
    solved_values: List[bool] = []
    span_meta: List[Dict[str, Any]] = []
    pose_sequences: Dict[str, List[Dict[str, Any]]] = {"arm1": [], "arm2": []}
    point_offset = 0

    for span_index, raw_span in enumerate(best.get("spans") or []):
        span = dict(raw_span)
        q = np.asarray(span.get("joint_positions_rad"), dtype=np.float64)
        if (
            q.ndim != 2
            or q.shape[0] == 0
            or q.shape[1] != len(joint_names)
            or not np.isfinite(q).all()
        ):
            raise ValueError(
                f"IK span {span_index} 关节数组非法: shape={q.shape}"
            )
        global_start = int(span.get("global_event_start", 0))
        events = list(span.get("events") or [])
        by_state = {
            int(event["state_index"]): event
            for event in events
            if event.get("state_index") is not None
        }
        positions_parts.append(q)
        span_index_values.extend([span_index] * q.shape[0])
        span_state_values.extend(range(q.shape[0]))
        # state 0 是该 chunk 的 MotionGen q_begin，不对应新的 IK 事件。
        global_event_values.append(-1)
        accepted_values.append(True)
        solved_values.append(False)
        for state_index in range(1, q.shape[0]):
            event = by_state.get(state_index)
            if event is None:
                raise ValueError(
                    f"IK span {span_index} state {state_index} 缺少 event 映射"
                )
            global_event_values.append(global_start + int(event["event"]))
            accepted_values.append(bool(event.get("accepted", False)))
            solved_values.append(bool(event.get("solve_success", False)))
        for event in events:
            target_poses = event.get("target_poses") or {}
            for arm_name in ("arm1", "arm2"):
                pose = target_poses.get(arm_name)
                if isinstance(pose, dict):
                    pose_sequences[arm_name].append(dict(pose))

        clean_span = {
            key: value for key, value in span.items()
            if key not in ("joint_positions_rad", "start_poses")
        }
        clean_span.update({
            "span_index": span_index,
            "point_start": point_offset,
            "point_end": point_offset + q.shape[0] - 1,
            "n_points": int(q.shape[0]),
            "start_poses": span.get("start_poses"),
        })
        span_meta.append(clean_span)
        point_offset += int(q.shape[0])

    positions = np.concatenate(positions_parts, axis=0)
    fk = compute_fk_batched(mg, positions, list(ee_links))
    required_fk = (
        f"{ee_links[0]}/pos", f"{ee_links[0]}/quat",
        f"{ee_links[1]}/pos", f"{ee_links[1]}/quat",
    )
    if any(key not in fk for key in required_fk):
        raise ValueError("保存 IK trace 时 FK 缺少双 TCP 数据")
    times = np.arange(positions.shape[0], dtype=np.float64) * float(frame_seconds)
    payload: Dict[str, np.ndarray] = {
        "joint_names": np.asarray([str(x) for x in joint_names], dtype="S64"),
        "positions": positions.astype(np.float64, copy=False),
        "times": times,
        "ee_positions": np.asarray(fk[required_fk[0]], dtype=np.float64),
        "ee_quats_wxyz": np.asarray(fk[required_fk[1]], dtype=np.float64),
        "second_ee_positions": np.asarray(fk[required_fk[2]], dtype=np.float64),
        "second_ee_quats_wxyz": np.asarray(fk[required_fk[3]], dtype=np.float64),
        "span_index": np.asarray(span_index_values, dtype=np.int64),
        "span_state_index": np.asarray(span_state_values, dtype=np.int64),
        "global_event_index": np.asarray(global_event_values, dtype=np.int64),
        "ik_accepted": np.asarray(accepted_values, dtype=np.bool_),
        "ik_solved": np.asarray(solved_values, dtype=np.bool_),
        "home_root_index": np.full(
            positions.shape[0], root_index, dtype=np.int64
        ),
    }
    robot = dict(cfg.get("robot") or {})
    robot["joint_names"] = [str(x) for x in joint_names]
    metadata: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_type": "dual_arm_ik_prescreen_trace",
        "artifact_type": "dual_arm_ik_prescreen_trace",
        "schema_version": 1,
        "capture_kind": "deepest_prescreen_path",
        "visualization_only": True,
        "execution_safe": False,
        "safe_to_execute": False,
        "unsafe_for_execution": True,
        "transition_checked": False,
        "warning": (
            "这是两臂独立 IK 拼接后的离散构型诊断，包含可能因碰撞/"
            "跳变/限位/漂移被拒绝"
            "的构型；跨点及跨 span 运动均未做轨迹规划/连续碰撞验证，"
            "只能用于 RViz，禁止下发机器人。"
        ),
        "ee_pose_source": "fk_recomputed_from_saved_joint_positions",
        "source_plan_failed": str((Path(out_dir) / "plan_failed.json").resolve()),
        "robot": robot,
        "workspace": cfg.get("workspace") or {},
        "config": cfg,
        "n_points": int(positions.shape[0]),
        "frame_seconds": float(frame_seconds),
        "interpolation_dt": float(frame_seconds),
        "total_duration_s": float(times[-1]) if len(times) else 0.0,
        "home_root_index": None if root_index < 0 else root_index,
        "best_trace": {
            key: value for key, value in best.items()
            if key not in ("spans", "home_root")
        },
        "home_root": best.get("home_root"),
        "n_completed_root_captures": len(valid_captures),
        "root_capture_summaries": [
            {
                "home_root_index": capture.get("home_root_index"),
                "rank": capture.get("_rank"),
                "full_ik": bool(capture.get("full_ik", False)),
                "deepest_accepted_global_event": capture.get(
                    "deepest_accepted_global_event"
                ),
                "deepest_solved_global_event": capture.get(
                    "deepest_solved_global_event"
                ),
            }
            for capture in valid_captures
        ],
        "spans": span_meta,
        "pose_sequences": pose_sequences,
        "pose_sequence": pose_sequences["arm1"],
        "checks": {
            "ik_solver_returned_each_saved_nonstart_state": True,
            "rejected_states_may_be_present": bool(
                not np.asarray(accepted_values, dtype=np.bool_).all()
            ),
            "fk_recomputed": True,
            "transition_planned": False,
            "transition_collision_checked": False,
        },
    }
    return write_ik_playback_artifact(
        Path(out_dir) / "ik_playback", payload, metadata
    )


# ============================== 主流程 ==============================


def main() -> int:
    args = dual_argparser().parse_args()
    cfg = apply_dual_cli(load_dual_config(args.config), args)
    rb, pp, ws, pl, outc, dc = (
        cfg["robot"], cfg["pick_place"], cfg["workspace"], cfg["planner"],
        cfg["output"], cfg["dual_arm"],
    )
    gg, asr, cr = pp["grasp_grid"], pp["angle_search"], pp["criterion"]

    out_dir = Path(outc["dir"])
    if not out_dir.is_absolute():
        out_dir = TASK_ROOT / out_dir
    if outc.get("add_timestamp", True):
        out_dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    # 尤其在 --no-timestamp 复用目录时，必须先关闭播放器安全门。只有本轮
    # 轨迹和全部后验检查成功写完后，函数末尾才会删除此 marker。
    dump_json({
        "stage": "planning_in_progress",
        "safe_to_play": False,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, out_dir / "plan_failed.json")
    print(f"[OUT] {out_dir}")

    task_layout = str(dc.get("task_layout", "same_local_grid"))
    if task_layout not in ("same_local_grid", "shared_global_outside_in"):
        print("[FAIL] dual_arm.task_layout 只支持 same_local_grid/"
              f"shared_global_outside_in，实际 {task_layout!r}")
        return 2
    shared_layout = task_layout == "shared_global_outside_in"
    if not bool(pl.get("self_collision_check", True)):
        print("[FAIL] 双臂协同模式必须开启 planner.self_collision_check，"
              "否则不能保证规划期臂内/臂间避碰")
        return 2
    on_fail_mode = str(pp.get("on_fail", {}).get("mode", "stop"))
    if on_fail_mode not in ("stop", "skip"):
        print(f"[FAIL] pick_place.on_fail.mode 只支持 stop/skip，实际 {on_fail_mode!r}")
        return 2
    if bool(dc.get("sequential_fallback", False)):
        print("[FAIL] 连续跨物料流水不支持 sequential_fallback：该降级会破坏固定相位。")
        return 2

    # 挂载：一号臂为联合 base，二号臂在同侧 x=-0.8m 且相对 yaw=pi。
    t12, mounts = mounts_relative_transform(rb["mounts"], "left")
    t21 = np.linalg.inv(t12)
    print(f"[MOUNT] arm2 in arm1 base: xyz={np.round(t12[:3, 3], 6).tolist()}")
    print(f"[MOUNT] R12=\n{np.round(t12[:3, :3], 6)}")
    if abs(t12[1, 3]) > 1e-4:
        print("[WARN] 二号臂仍与一号臂存在 y 方向偏置，"
              "请确认使用了同侧 mounts")
    baked_t12 = urdf_link_transform(
        rb["urdf"], rb["base_link"], rb["dual_arm_prefix"] + rb["base_link"]
    )
    mount_error = float(np.abs(baked_t12 - t12).max())
    if mount_error > 1e-6:
        print(f"[FAIL] mounts 与联合规划 URDF 中烘焙的二号臂位姿不一致 "
              f"(max error={mount_error:.3g})")
        print("       请用 build_scene_urdf.py --planning-mode independent 重生成 URDF")
        return 2

    # 任务坐标补偿以「位置 + 轴角旋转」配置，只作用于目标，不改变
    # 机器人、料台、墙或两臂物理安装。当前目标先按原逻辑构造到联合 root，再统一做
    # T_goal_effective = T_link0 @ T_goal_current。
    link0_target_config = pp.get(
        "link0_target_transform",
        {
            "position": [0.0, 0.0, 0.0],
            "rotation": {"axis": [0.0, 0.0, 1.0], "angle_deg": 0.0},
        },
    )
    try:
        link0_target_transform = parse_rigid_transform_matrix(
            link0_target_config,
            "pick_place.link0_target_transform",
        )
        target_transforms = arm_target_transforms(link0_target_transform, t12)
    except (TypeError, ValueError) as exc:
        print(f"[FAIL] LINK_0 目标变换非法: {exc}")
        return 2
    if isinstance(link0_target_config, dict):
        rotation_config = link0_target_config["rotation"]
        print(
            "[TARGET TF] axis-angle: "
            f"position={np.round(link0_target_transform[:3, 3], 6).tolist()}, "
            f"axis={rotation_config['axis']}, "
            f"angle_deg={rotation_config['angle_deg']}"
        )
    else:
        print("[TARGET TF] 使用兼容的 4x4 矩阵配置")
    print("[TARGET TF] T_goal_effective = T_link0 @ T_goal_current")
    print(f"[TARGET TF] R_link0=\n{np.round(link0_target_transform[:3, :3], 6)}")

    # shared 布局必须先展开完整世界网格再按两端分区；否则先截前 N 个会
    # 把二号臂一侧的候选意外裁掉。skip 也需要完整 pool 用于 IK 补位。
    all_local_items = build_grasp_points(
        gg, None if shared_layout or on_fail_mode == "skip" else args.max_items
    )
    if not all_local_items:
        print("[FAIL] grasp grid 为空")
        return 2
    if args.max_items is not None and int(args.max_items) <= 0:
        print("[FAIL] --max-items 必须是正整数")
        return 2
    requested_count = None if args.max_items is None else int(args.max_items)
    if shared_layout:
        # 先保留每侧完整、已 outside-in 排序的 source pool，IK 过滤时再按
        # arm quota 截取，才能让无解 source 被同侧后续点替补。
        arm1_items, arm2_items = allocate_shared_grid_outside_in(
            all_local_items, t12, None
        )
        arm_quotas: Tuple[Optional[int], Optional[int]] = (
            None if requested_count is None else (requested_count + 1) // 2,
            None if requested_count is None else requested_count // 2,
        )
    else:
        arm1_items = [
            dict(x, root_position=list(x["position"])) for x in all_local_items
        ]
        arm2_items = []
        for item in all_local_items:
            root = transform_points(np.asarray(item["position"])[None, :], t12)[0]
            arm2_items.append(dict(item, root_position=root.tolist()))
        arm_quotas = (requested_count, requested_count)
    place1_root = np.asarray(pp["place"]["position"], dtype=np.float64)
    place2_root = derive_second_place_position(place1_root, dc)
    if not np.isfinite(place1_root).all() or not np.isfinite(place2_root).all():
        print("[FAIL] 两臂 place 坐标必须是有限数")
        return 2
    place1_local = place1_root.copy()
    place2_local = transform_points(place2_root[None, :], t21)[0]
    place1_effective_root = transform_points(
        place1_root[None, :], link0_target_transform
    )[0]
    place2_effective_root = transform_points(
        place2_root[None, :], link0_target_transform
    )[0]
    place2_effective_local = transform_points(
        place2_effective_root[None, :], t21
    )[0]
    quota_text = "全部可达点" if requested_count is None else str(requested_count)
    if shared_layout:
        print(
            f"[TASK] shared world grid={len(all_local_items)}，总额度={quota_text}，"
            f"arm quotas={arm_quotas}；arm1 x大端→中间，arm2 x小端→中间，"
            f"source不重复；on_fail={on_fail_mode}"
        )
    else:
        print(f"[TASK] grid 候选={len(all_local_items)}，计划物料额度/臂={quota_text}，"
              f"on_fail={on_fail_mode}；arm2 连续相位滞后 "
              f"{dc.get('start_delay_stages', 3)} 个工艺点")
    print(
        f"[TASK] arm1 place raw(root/local)={np.round(place1_root, 4).tolist()}  "
        f"effective(root)={np.round(place1_effective_root, 4).tolist()}"
    )
    print(
        f"[TASK] arm2 place raw(root)={np.round(place2_root, 4).tolist()}  "
        f"raw(local)={np.round(place2_local, 4).tolist()}  "
        f"effective(root/local)={np.round(place2_effective_root, 4).tolist()}/"
        f"{np.round(place2_effective_local, 4).tolist()}"
    )

    inactive_tol = float(dc.get("inactive_joint_tolerance_deg", 1.0))
    home_branch_tol = float(dc.get("home_branch_tolerance_deg", 1.0))
    collision_margin_mm = float(dc.get("collision_margin_mm", 0.0))
    if not np.isfinite(inactive_tol) or inactive_tol < 0.0:
        print("[FAIL] dual_arm.inactive_joint_tolerance_deg 必须是有限非负数")
        return 2
    if not np.isfinite(collision_margin_mm) or collision_margin_mm < 0.0:
        print("[FAIL] dual_arm.collision_margin_mm 必须是有限非负数")
        return 2
    if not np.isfinite(home_branch_tol) or home_branch_tol < 0.0:
        print("[FAIL] dual_arm.home_branch_tolerance_deg 必须是有限非负数")
        return 2
    criterion_joints = [int(x) for x in cr.get("joints", [])]
    if not criterion_joints or any(x < 1 or x > 6 for x in criterion_joints):
        print(f"[FAIL] criterion.joints 必须是 1..6 的非空列表，实际 {criterion_joints}")
        return 2
    max_joint_delta = float(cr.get("max_joint_delta_deg", 0.0))
    if not np.isfinite(max_joint_delta) or max_joint_delta <= 0.0:
        print("[FAIL] criterion.max_joint_delta_deg 必须是有限正数")
        return 2
    linear_cfg = dict(pp.get("linear_move") or {})
    if linear_cfg.get("enable", False):
        bad_kinds = [
            x for x in (linear_cfg.get("kinds") or ["place"])
            if x not in ("grasp", "place")
        ]
        if bad_kinds:
            print(f"[FAIL] linear_move.kinds 只支持 grasp/place，收到 {bad_kinds}")
            return 2
        if str(pp["lift"].get("axis", "base_z")) != "base_z":
            print("[FAIL] linear_move.enable=true 时 lift.axis 必须为 base_z")
            return 2
        configured_free_axis = str(linear_cfg.get("free_axis", "z")).lower()
        if configured_free_axis != "z":
            print(
                "[FAIL] lift.axis=base_z 时 linear_move.free_axis 必须为 z，"
                f"实际 {configured_free_axis!r}"
            )
            return 2
        try:
            effective_axes = tuple(
                transformed_cardinal_axis(transform, configured_free_axis)
                for transform in target_transforms
            )
        except (TypeError, ValueError) as exc:
            print(
                "[FAIL] LINK_0 目标变换与 Cartesian 直线约束不兼容: "
                f"{exc}。当前 CuRobo metric 只能释放联合 root 的 x/y/z；"
                "请使用轴对齐旋转，或关闭 pick_place.linear_move.enable。"
            )
            return 2
        if effective_axes[0] != effective_axes[1]:
            print(
                "[FAIL] 两臂直线方向经目标/安装变换后落在不同 root 轴: "
                f"arm1={effective_axes[0]}, arm2={effective_axes[1]}"
            )
            return 2
        linear_cfg["configured_free_axis"] = configured_free_axis
        linear_cfg["free_axis"] = effective_axes[0]
        if effective_axes[0] != configured_free_axis:
            print(
                f"[TARGET TF] linear free_axis: {configured_free_axis} -> "
                f"root {effective_axes[0]}"
            )

    initial_primary = arm_primary_choices(asr)
    if not initial_primary:
        print("[FAIL] angle_search 没有生成任何候选")
        return 2
    choices_per_arm = len(initial_primary)
    trial_cap = int(dc.get("max_pair_angle_trials") or 0)
    if trial_cap < 0:
        print("[FAIL] dual_arm.max_pair_angle_trials 不能为负数")
        return 2
    node_cap = int(dc.get("max_pipeline_search_nodes") or 0)
    if node_cap < 0:
        print("[FAIL] dual_arm.max_pipeline_search_nodes 不能为负数")
        return 2
    home_angle_cap = int(dc.get("max_home_angle_pair_trials") or 0)
    home_branch_cap = int(dc.get("max_home_ik_branches_per_pair") or 0)
    home_root_cap = int(dc.get("max_home_root_trials") or 0)
    for key, value in (
        ("max_home_angle_pair_trials", home_angle_cap),
        ("max_home_ik_branches_per_pair", home_branch_cap),
        ("max_home_root_trials", home_root_cap),
    ):
        if value < 0:
            print(f"[FAIL] dual_arm.{key} 不能为负数")
            return 2
    st2_cfg = dict(asr.get("stage2") or {})
    st2_on = bool(st2_cfg.get("enable", False))
    if st2_on and str(st2_cfg.get("frame", "base")).lower() not in ("base", "tool"):
        print("[FAIL] angle_search.stage2.frame 只支持 base/tool，"
              f"实际 {st2_cfg.get('frame')!r}")
        return 2
    nominal_seq_len = len(make_arm_sequence(
        all_local_items[0]["position"], place1_local, initial_primary[0],
        pp, 0, target_transforms[0], "arm1_",
    ))
    phase = int(dc.get("start_delay_stages", 3))
    if not 1 <= phase < nominal_seq_len:
        print(f"[FAIL] 连续流水 phase 必须满足 1 <= phase < {nominal_seq_len}，"
              f"实际 {phase}")
        return 2
    configured_max_phase = int(dc.get("max_start_delay_stages", phase))
    if configured_max_phase != phase:
        print(f"[WARN] 连续流水使用固定 phase={phase}；"
              f"max_start_delay_stages={configured_max_phase} 不再用于逐轮改节拍")
    budget_text = str(trial_cap) if trial_cap else "不限"
    node_budget_text = str(node_cap) if node_cap else "不限"
    print(f"[SCHED] 每臂一阶段角度候选={choices_per_arm}，"
          f"每次 block/上游状态角度对预算={budget_text}，固定 phase={phase}，"
          f"全局搜索节点预算={node_budget_text}")
    if shared_layout:
        print(
            "[HOME ROOT] 首件姿态对预算="
            f"{home_angle_cap or '不限'}，每姿态对 IK 分支="
            f"{home_branch_cap or '不限'}，整线根节点预算="
            f"{home_root_cap or '不限'}"
        )
        if home_root_cap and node_cap:
            print(
                "[HOME ROOT] 理论最坏规划节点上限="
                f"{home_root_cap} 根 × {node_cap} 节点 = "
                f"{home_root_cap * node_cap}"
            )
    if st2_on:
        print(f"[SCHED] stage2 已开启：固定首个可行一阶段角后联合细搜；"
              f"每臂二阶段候选={len(stage2_combos(asr))}，全失败回退一阶段")
    print(
        f"[SCHED] 运动前 source IK filter="
        f"{'开' if on_fail_mode == 'skip' else '关'}；block 独立 IK 组合预筛="
        f"{'开' if cr.get('prescreen_by_ik', True) else '关'}；"
        f"IK 跳变预筛="
        f"{'开' if cr.get('prescreen_joint_delta_check', True) else '关'}；"
        f"等待臂关节漂移 <= {inactive_tol:g}deg；逐物料 barrier=关"
    )

    # 联合 robot。link_names 必须显式含副 TCP，否则 link_poses 没有 cost。
    world = make_world_config(ws)
    robot_dict = load_robot_cfg_dict(rb)
    kin = robot_dict["robot_cfg"]["kinematics"]
    from curobo.util_file import get_assets_path

    configured_urdf = Path(str(kin["urdf_path"]))
    if not configured_urdf.is_absolute():
        configured_urdf = Path(get_assets_path()) / configured_urdf
    declared_urdf = resolve_repo_path(rb["urdf"])
    if configured_urdf.resolve() != declared_urdf.resolve():
        print("[FAIL] robot.urdf 与 robot_yml 实际加载的 URDF 不一致:")
        print(f"       robot.urdf: {declared_urdf}")
        print(f"       robot_yml:  {configured_urdf}")
        return 2
    ee_links = (str(rb["ee_link"]), str(rb.get("second_ee_link") or
                                        rb.get("dual_arm_prefix", "second_") + rb["ee_link"]))
    # warmup 会把 link_names 中每个副 link 都写入持久 goal buffer。若把纯
    # FK 诊断用的 flange/gripper 也放进去，它们会被意外锁在 retract pose。
    # 因此联合规划器只声明真正作为位姿目标的两个 TCP。
    kin["link_names"] = [ee_links[0], ee_links[1]]

    wall_cfg = ws.get("wall") or {}
    wall_links: List[str] = []
    wall_report: Dict[str, Any] = {"applied": False}
    if wall_cfg.get("enable", False) and wall_cfg.get("collision_link_names"):
        base_wall = list(wall_cfg["collision_link_names"])
        wall_links = base_wall + [rb["dual_arm_prefix"] + x for x in base_wall]

    def apply_wall_filter_before_warmup(solver) -> None:
        nonlocal wall_report
        wall_report = restrict_world_collision_to_links(solver, wall_links)
        wall_report["applied"] = True

    # 所有辅助 CuRobo solver 必须在 MotionGen warmup/capture 之前创建。
    # 某些自定义 CUDA 内核含进程级缓存；capture 后再创建另一套 solver，
    # 即使对象完全独立，也可能令旧图中的内核状态失效。
    try:
        separate_ik_context = make_separate_arm_ik_context(
            robot_dict, world, wall_links, rb, pl, dc, t21,
        )
    except Exception as exc:  # noqa: BLE001 - 离散 IK 安全链缺失即停止
        print(f"[FAIL] 无法构造双臂独立 IK 上下文: {exc}")
        dump_json({
            "stage": "separate_arm_ik_setup",
            "error": str(exc),
            "config": cfg,
        }, out_dir / "plan_failed.json")
        return 2

    # Python forward wrapper 必须在 CUDA graph 捕获前安装；warmup 后再替换
    # forward 不会改变已捕获图，造成日志说已过滤、实际仍让底座撞墙。
    mg = make_motion_gen(
        robot_dict, world, pl,
        pre_warmup=apply_wall_filter_before_warmup if wall_links else None,
    )
    joint_names = list(mg.joint_names)
    expected_joint_names = (
        [f"J_{index}" for index in range(1, 7)]
        + [f"{rb['dual_arm_prefix']}J_{index}" for index in range(1, 7)]
    )
    if joint_names != expected_joint_names:
        print(f"[FAIL] 需要独立 12-DOF 联合配置，实际 joints={joint_names}")
        print(f"       期望 joints={expected_joint_names}")
        return 2
    print(f"[ROBOT] 12-DOF joints={joint_names}")

    limits = mg.kinematics.get_joint_limits().position
    q_lo = limits[0].cpu().numpy().astype(np.float64)
    q_hi = limits[1].cpu().numpy().astype(np.float64)
    if (
        separate_ik_context.q_lo.shape != q_lo.shape
        or separate_ik_context.q_hi.shape != q_hi.shape
        or not np.allclose(separate_ik_context.q_lo, q_lo, atol=1e-7)
        or not np.allclose(separate_ik_context.q_hi, q_hi, atol=1e-7)
    ):
        print("[FAIL] 独立 collision probe 与 MotionGen 的关节限位不一致")
        return 2
    mg._xtrainer_separate_ik = separate_ik_context
    print(
        "[IK] 两臂分别做 6-DOF IK；每臂最多 "
        f"{mg._xtrainer_separate_ik.max_candidates_per_arm} 条分支，"
        "拼接后使用隔离的 12-DOF limit/self/inter-arm/world collision 过滤"
    )

    # 配置 Home 只作为 IK bootstrap。shared 布局最终会在完成 source
    # 过滤后，把真正的轨迹 Home 改成两臂各自首件的预抓取位姿。
    home_cfg = pp["home"]
    configured_home1_local = PoseSpec.from_rpy_deg(
        "configured_home", home_cfg["position"], home_cfg["rpy_deg"], "start"
    )
    configured_home1 = transform_pose(
        configured_home1_local, target_transforms[0], "arm1_"
    )
    configured_home2_local = PoseSpec.from_rpy_deg(
        "configured_home", home_cfg["position"], home_cfg["rpy_deg"], "start"
    )
    configured_home2 = transform_pose(
        configured_home2_local, target_transforms[1], "arm2_"
    )
    home1 = configured_home1
    home2 = configured_home2
    if home_cfg.get("joint_deg") is not None:
        q_home = np.radians(
            np.asarray(home_cfg["joint_deg"], dtype=np.float64).reshape(-1)
        )
        if q_home.size == 6:
            q_home = np.concatenate([q_home, q_home])
        if q_home.size != 12:
            print(f"[FAIL] dual home.joint_deg 需要 6(自动复制) 或 12 个值，"
                  f"实际 {q_home.size}")
            return 3
    else:
        print("[HOME SEED] 分别求解两臂配置 Home IK，再做 12-DOF 组合验碰 ...")
        q_home, ik_result = solve_dual_ik(
            _discrete_ik_context(mg), configured_home1, configured_home2,
            ee_links[1],
        )
        if q_home is None:
            print(f"[FAIL] 双臂 Home 独立 IK/组合验碰无可用解: "
                  f"{getattr(ik_result, 'status', '')}")
            dump_json({
                "stage": "separate_arm_home_ik",
                "reason": getattr(ik_result, "status", ""),
                "ik_report": getattr(ik_result, "report", {}),
                "config": cfg,
            }, out_dir / "plan_failed.json")
            return 3
    if not np.isfinite(q_home).all():
        print("[FAIL] dual home 关节角含 NaN/Inf")
        dump_json({"stage": "dual_home_nonfinite", "config": cfg},
                  out_dir / "plan_failed.json")
        return 3
    outside_home = np.nonzero((q_home < q_lo) | (q_home > q_hi))[0]
    if outside_home.size:
        print("[FAIL] dual home 关节角超出规划限位: "
              f"{[joint_names[int(i)] for i in outside_home]}")
        dump_json({
            "stage": "dual_home_joint_limit",
            "joint_names": [joint_names[int(i)] for i in outside_home],
            "home_joint_deg": np.degrees(q_home).tolist(),
            "config": cfg,
        }, out_dir / "plan_failed.json")
        return 3
    print(f"[HOME SEED] q_deg={np.degrees(q_home).round(2).tolist()}")
    configured_q_home = np.asarray(q_home, dtype=np.float64).copy()

    def effective_item_coordinates(
        raw_root_position: Sequence[float], arm: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw_root = np.asarray(raw_root_position, dtype=np.float64).reshape(3)
        effective_root = transform_points(
            raw_root[None, :], link0_target_transform
        )[0]
        effective_local = (
            effective_root
            if int(arm) == 1
            else transform_points(effective_root[None, :], t21)[0]
        )
        return effective_root, effective_local

    def shared_item_record(
        arm: int, pipeline_index: int, item: Dict[str, Any]
    ) -> Dict[str, Any]:
        raw_root = np.asarray(item["root_position"], dtype=np.float64)
        effective_root, effective_local = effective_item_coordinates(
            raw_root, arm
        )
        return {
            "arm": int(arm),
            "pipeline_item_index": int(pipeline_index),
            "source_item_index": int(item.get("index", pipeline_index)),
            "row": item.get("row"),
            "col": item.get("col"),
            "local_position": list(item["position"]),
            "raw_root_position": raw_root.tolist(),
            "root_position": effective_root.tolist(),
            "effective_local_position": effective_local.tolist(),
        }

    def paired_item_record(
        pipeline_index: int,
        item1: Dict[str, Any],
        item2: Dict[str, Any],
    ) -> Dict[str, Any]:
        arm1_raw_root = np.asarray(item1["root_position"], dtype=np.float64)
        arm2_raw_root = np.asarray(item2["root_position"], dtype=np.float64)
        arm1_effective_root, arm1_effective_local = effective_item_coordinates(
            arm1_raw_root, 1
        )
        arm2_effective_root, arm2_effective_local = effective_item_coordinates(
            arm2_raw_root, 2
        )
        return {
            "pipeline_item_index": int(pipeline_index),
            "source_item_index": int(item1.get("index", pipeline_index)),
            "row": item1.get("row"),
            "col": item1.get("col"),
            "local_position": list(item1["position"]),
            "arm1_raw_root_position": arm1_raw_root.tolist(),
            "arm2_raw_root_position": arm2_raw_root.tolist(),
            "arm1_root_position": arm1_effective_root.tolist(),
            "arm2_root_position": arm2_effective_root.tolist(),
            "arm1_effective_local_position": arm1_effective_local.tolist(),
            "arm2_effective_local_position": arm2_effective_local.tolist(),
        }

    def current_selected_item_map() -> List[Dict[str, Any]]:
        if shared_layout:
            return [
                shared_item_record(arm, pipeline_index, item)
                for arm, items in ((1, arm1_items), (2, arm2_items))
                for pipeline_index, item in enumerate(items)
            ]
        return [
            paired_item_record(pipeline_index, item1, item2)
            for pipeline_index, (item1, item2) in enumerate(
                zip(arm1_items, arm2_items)
            )
        ]

    def enrich_skipped_item(record: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(record)
        if "arm" in enriched:
            arm = int(enriched["arm"])
            effective_root, effective_local = effective_item_coordinates(
                enriched["raw_root_position"], arm
            )
            enriched["root_position"] = effective_root.tolist()
            enriched["effective_local_position"] = effective_local.tolist()
            return enriched

        for arm in (1, 2):
            effective_root, effective_local = effective_item_coordinates(
                enriched[f"arm{arm}_raw_root_position"], arm
            )
            enriched[f"arm{arm}_root_position"] = effective_root.tolist()
            enriched[f"arm{arm}_effective_local_position"] = (
                effective_local.tolist()
            )
        return enriched

    def paired_item_ik_probe(
        source_index: int,
        item1: Dict[str, Any],
        item2: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        return probe_dual_item_primary_ik(
            mg,
            q_home,
            item1["position"],
            item2["position"],
            place1_local,
            place2_local,
            int(item1.get("index", source_index)),
            pp,
            asr,
            target_transforms,
            q_lo,
            q_hi,
            ee_links,
        )

    # 保留完整、有序的 source pool。shared 布局的两侧 source 独立探测与
    # 补位；兼容布局仍保留旧的成对探测语义。
    source_arm1_items = list(arm1_items)
    source_arm2_items = list(arm2_items)
    try:
        if shared_layout:
            def make_single_arm_probe(arm: int):
                def probe(
                    source_offset: int, item: Dict[str, Any]
                ) -> Tuple[bool, Dict[str, Any]]:
                    return probe_arm_item_primary_ik(
                        mg, q_home, item["position"],
                        place1_local if arm == 1 else place2_local,
                        int(item.get("index", source_offset)), arm, pp, asr,
                        target_transforms[arm - 1],
                        q_lo, q_hi, ee_links,
                    )
                return probe

            arm1_items, skipped1 = filter_arm_items_by_primary_ik(
                source_arm1_items, arm_quotas[0], on_fail_mode, 1,
                make_single_arm_probe(1),
            )
            arm2_items, skipped2 = filter_arm_items_by_primary_ik(
                source_arm2_items, arm_quotas[1], on_fail_mode, 2,
                make_single_arm_probe(2),
            )
            skipped_ik_items = skipped1 + skipped2
            source_cursor_by_arm = {
                "arm1": len(arm1_items) + len(skipped1),
                "arm2": len(arm2_items) + len(skipped2),
            }
            source_cursor = sum(source_cursor_by_arm.values())
        else:
            arm1_items, arm2_items, skipped_ik_items = filter_dual_items_by_primary_ik(
                source_arm1_items,
                source_arm2_items,
                requested_count,
                on_fail_mode,
                paired_item_ik_probe,
            )
            source_cursor = len(arm1_items) + len(skipped_ik_items)
            source_cursor_by_arm = {
                "arm1": source_cursor, "arm2": source_cursor,
            }
    except Exception as exc:  # noqa: BLE001 - 探测异常不可静默当作物料无解
        dump_json({
            "stage": "grasp_ik_prefilter_error",
            "error": str(exc),
            "config": cfg,
        }, out_dir / "plan_failed.json")
        print(f"[FAIL] 抓取 IK 预筛异常，未执行任何轨迹: {exc}")
        return 4
    skipped_ik_items = [
        enrich_skipped_item(item) for item in skipped_ik_items
    ]

    # shared 模式的配置 Home 到此为止只完成 bootstrap。source 只过滤一次，
    # 然后把“首件抓取姿态对 × 两臂独立 IK 安全组合”枚为流水线根节点。
    # 每个根节点都必须从自己的 q_home 重新规划整条轨迹；下游失败只会
    # 回退到下一个 Home 根，不用新分支回头重筛 source，也不继承失败 chunk。
    home_search_report: Dict[str, Any] = {
        "mode": "configured",
        "configured_home_arm1": configured_home1.to_dict(),
        "configured_home_arm2": configured_home2.to_dict(),
    }
    home_stabilization: List[Dict[str, Any]] = []
    home_root_candidates: List[FirstPregraspHomeRoot] = []
    home_root_enumeration: Dict[str, Any] = {}
    if shared_layout and arm1_items:
        first_signature = (
            int(arm1_items[0].get("index", 0)),
            None if not arm2_items else int(arm2_items[0].get("index", 0)),
        )
        try:
            home_root_candidates, home_root_enumeration = (
                enumerate_first_pregrasp_home_roots(
                    mg,
                    configured_q_home,
                    arm1_items[0],
                    arm2_items[0] if arm2_items else None,
                    place1_local,
                    place2_local,
                    pp,
                    asr,
                    target_transforms,
                    ee_links,
                    configured_home2,
                    q_lo,
                    q_hi,
                    max_angle_pair_trials=home_angle_cap,
                    max_branches_per_pair=home_branch_cap,
                    max_roots=home_root_cap,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 根节点枚举异常 fail closed
            dump_json({
                "stage": "first_pregrasp_home_root_exception",
                "error": str(exc),
                "input_first_source_indices": list(first_signature),
                "config": cfg,
            }, out_dir / "plan_failed.json")
            print(f"[FAIL] 首件 Home 根节点枚举异常: {exc}")
            return 3

        home_stabilization.append({
            "iteration": 1,
            "input_first_source_indices": list(first_signature),
            "home_root_enumeration": home_root_enumeration,
            "filter_rerun": False,
        })
        if not home_root_candidates:
            dump_json({
                "stage": "first_pregrasp_home_ik",
                "home_search": home_root_enumeration,
                "iterations": home_stabilization,
                "selected_item_map": {
                    "arm1": shared_item_record(1, 0, arm1_items[0]),
                    "arm2": (
                        None
                        if not arm2_items
                        else shared_item_record(2, 0, arm2_items[0])
                    ),
                },
                "config": cfg,
            }, out_dir / "plan_failed.json")
            print("[FAIL] 所有首件预抓取姿态对均没有可用的联合 Home IK 分支；"
                  "该失败不能安全归因到单侧 source，未继续跳过")
            return 3

        # 枚举器已做 shape/finite/limit 检查；主流程再验证一次，
        # 防止测试替身或后续接口改动将坏根节点带入轨迹规划。
        for root_index, root in enumerate(home_root_candidates):
            candidate_q = np.asarray(root.q_home, dtype=np.float64).reshape(-1)
            if (
                candidate_q.shape != configured_q_home.shape
                or not np.isfinite(candidate_q).all()
                or np.any(candidate_q < q_lo - 1e-7)
                or np.any(candidate_q > q_hi + 1e-7)
            ):
                dump_json({
                    "stage": "first_pregrasp_home_root_invalid",
                    "home_root_index": root_index,
                    "home_root": root.to_dict(),
                    "home_search": home_root_enumeration,
                    "config": cfg,
                }, out_dir / "plan_failed.json")
                print(f"[FAIL] 首件 Home 根节点 {root_index} 的关节状态非法")
                return 3

        # 这一步故意放在整线 MotionGen 搜索之前：即使后面
        # 搜索耗时较长、失败或被中断，所有已求得的 Home IK
        # 分支也已经可以独立用 RViz 查看。
        try:
            ik_npz, _ik_meta, ik_count = save_home_root_ik_checkpoint(
                out_dir, home_root_candidates, joint_names, cfg
            )
            print(
                f"[IK SAVE] Home roots={ik_count} -> {ik_npz} "
                "(仅可视化，分支间未规划/未验碰)"
            )
            print(
                "[IK PLAY] ./run_dual_ik_rviz.sh --ik-results "
                f"{ik_npz}"
            )
        except Exception as exc:  # noqa: BLE001 - 诊断落盘不改变规划结论
            print(f"[IK SAVE WARN] Home IK 诊断文件保存失败: {exc}")

        home_search_report = {
            "mode": "first_pregrasp_root_search",
            "enumeration": home_root_enumeration,
            "selected_root_index": None,
            "selected_root": None,
        }
        print(
            f"[HOME ROOT] 枚举到 {len(home_root_candidates)} 个候选："
            f"姿态对={home_root_enumeration.get('n_angle_pair_trials', 0)}，"
            f"可用分支={home_root_enumeration.get('n_roots_available', 0)}"
        )

    n_scanned_items = source_cursor
    for skipped in skipped_ik_items:
        if shared_layout:
            print(
                f"[IK SKIP] arm{skipped['arm']} source "
                f"{skipped['source_item_index']} local="
                f"{np.round(skipped['local_position'], 4).tolist()}"
            )
        else:
            print(
                f"[IK SKIP] source item {skipped['source_item_index']} "
                f"local={np.round(skipped['arm1_local_position'], 4).tolist()}  "
                f"arm1={skipped['arm1_reachable']} arm2={skipped['arm2_reachable']}"
            )
    if not arm1_items:
        dump_json({
            "stage": "no_primary_grasp_ik_items",
            "requested_max_items": args.max_items,
            "n_candidates_scanned": n_scanned_items,
            "skipped_ik_items": skipped_ik_items,
            "config": cfg,
        }, out_dir / "plan_failed.json")
        print("[FAIL] 一号臂扫描范围内没有 primary grasp IK 可达物料；"
              "无法建立以一号臂先行的连续流水")
        return 4
    if shared_layout:
        print(
            f"[HOME] mode=first_pregrasp_root_search，source="
            f"arm1:{arm1_items[0].get('index')}，"
            f"arm2:{None if not arm2_items else arm2_items[0].get('index')}"
        )
        print("[HOME] 将按候选根节点的姿态角和 12-DOF IK 分支整线回退")

    pipeline_ik_skipped_items: List[Dict[str, Any]] = []
    pipeline_attempts: List[Dict[str, Any]] = []
    started_all = time.time()
    pipeline: Optional[PlannedPipeline] = None
    pipeline_failures: List[Dict[str, Any]] = []
    pipeline_search: Dict[str, Any] = {}
    selected_item_map = current_selected_item_map()
    home_root_cursor = 0
    selected_home_root: Optional[FirstPregraspHomeRoot] = None
    selected_candidate_gate: Optional[Dict[str, Any]] = None
    # 每个 Home root 只保留一条最深的离散 IK 路径，不将数千个
    # 搜索候选的关节数组复制进 plan_failed.json。
    ik_capture_by_root: List[Dict[str, Any]] = []

    def evaluate_pipeline_candidate(candidate: PlannedPipeline) -> Dict[str, Any]:
        """计算会导致轨迹拒绝的权威后验安全门。"""
        candidate_positions = np.asarray(candidate.position, dtype=np.float64)
        if (
            candidate_positions.ndim != 2
            or candidate_positions.shape[0] == 0
            or candidate_positions.shape[1] != len(joint_names)
            or not np.isfinite(candidate_positions).all()
        ):
            raise RuntimeError(
                f"候选流水轨迹 shape/数值非法: {candidate_positions.shape}"
            )
        candidate_fk = compute_fk_batched(
            mg, candidate_positions, list(ee_links)
        )
        required_fk = (
            f"{ee_links[0]}/pos", f"{ee_links[0]}/quat",
            f"{ee_links[1]}/pos", f"{ee_links[1]}/quat",
        )
        if any(key not in candidate_fk for key in required_fk):
            raise RuntimeError("候选流水 FK 缺少双 TCP 数据")
        candidate_ee1_pos = np.asarray(
            candidate_fk[f"{ee_links[0]}/pos"], dtype=np.float64
        )
        candidate_ee1_quat = np.asarray(
            candidate_fk[f"{ee_links[0]}/quat"], dtype=np.float64
        )
        candidate_ee2_pos = np.asarray(
            candidate_fk[f"{ee_links[1]}/pos"], dtype=np.float64
        )
        candidate_ee2_quat = np.asarray(
            candidate_fk[f"{ee_links[1]}/quat"], dtype=np.float64
        )
        if bool(ws.get("check_after_plan", True)):
            candidate_ws1 = workspace_report(candidate_ee1_pos, np.eye(4), ws)
            candidate_ws2 = workspace_report(candidate_ee2_pos, t21, ws)
        else:
            candidate_ws1 = candidate_ws2 = {
                "checked": False,
                "reason": "workspace.check_after_plan=false",
            }
        candidate_self = self_collision_report(mg, candidate_positions)
        candidate_pair = inter_arm_report(
            mg, candidate_positions, rb["dual_arm_prefix"], collision_margin_mm
        )
        unavailable: List[Dict[str, Any]] = []
        if bool(pl.get("self_collision_check", True)) and not candidate_self.get(
            "checked"
        ):
            unavailable.append({
                "check": "self_collision",
                "reason": candidate_self.get("reason", "unavailable"),
            })
        if not candidate_pair.get("checked"):
            unavailable.append({
                "check": "inter_arm_collision",
                "reason": candidate_pair.get("reason", "unavailable"),
            })
        if unavailable:
            raise RuntimeError(f"候选流水安全检查不可用: {unavailable}")

        errors: List[Dict[str, Any]] = []
        if (
            bool(pl.get("self_collision_check", True))
            and int(candidate_self.get("n_collision", 0)) > 0
        ):
            errors.append({
                "check": "self_collision",
                "n_collision": int(candidate_self["n_collision"]),
            })
        if int(candidate_pair.get("n_collision_points", 0)) > 0:
            errors.append({
                "check": "inter_arm_collision",
                "n_collision_points": int(candidate_pair["n_collision_points"]),
                "margin_mm": float(candidate_pair.get("margin_mm", 0.0)),
            })
        if (
            bool(ws.get("check_after_plan", True))
            and bool(ws.get("fail_on_violation", False))
        ):
            for arm_name, report in (
                ("arm1", candidate_ws1), ("arm2", candidate_ws2)
            ):
                if not report.get("checked"):
                    raise RuntimeError(
                        f"{arm_name} workspace 安全检查不可用: "
                        f"{report.get('reason', 'unavailable')}"
                    )
                if int(report.get("n_violation", 0)) > 0:
                    errors.append({
                        "check": "workspace",
                        "arm": arm_name,
                        "n_violation": int(report["n_violation"]),
                        "max_violation_mm": float(
                            report.get("max_violation_mm", 0.0)
                        ),
                    })
        return {
            "errors": errors,
            "ee1_pos": candidate_ee1_pos,
            "ee1_quat": candidate_ee1_quat,
            "ee2_pos": candidate_ee2_pos,
            "ee2_quat": candidate_ee2_quat,
            "workspace_arm1": candidate_ws1,
            "workspace_arm2": candidate_ws2,
            "self_collision": candidate_self,
            "inter_arm_collision": candidate_pair,
        }

    while arm1_items:
        selected_item_map = current_selected_item_map()
        attempt_index = len(pipeline_attempts) + 1
        attempt_home_root: Optional[FirstPregraspHomeRoot] = None
        attempt_q_home = np.asarray(q_home, dtype=np.float64).copy()
        attempt_home1, attempt_home2 = home1, home2
        candidate_gate: Optional[Dict[str, Any]] = None
        attempt_ik_capture: Dict[str, Any] = {}
        if shared_layout:
            if home_root_cursor >= len(home_root_candidates):
                raise RuntimeError("Home root cursor 越界，根搜索状态损坏")
            attempt_home_root = home_root_candidates[home_root_cursor]
            attempt_q_home = np.asarray(
                attempt_home_root.q_home, dtype=np.float64
            ).copy()
            attempt_home1 = attempt_home_root.arm1_home
            attempt_home2 = attempt_home_root.arm2_home
        shortfall_now = (
            0 if requested_count is None
            else max(
                requested_count - (
                    len(arm1_items) + len(arm2_items)
                    if shared_layout else len(arm1_items)
                ),
                0,
            )
        )
        print(
            f"[IK FILTER] scanned={source_cursor_by_arm if shared_layout else source_cursor}, "
            f"selected=arm1:{len(arm1_items)},arm2:{len(arm2_items)}, "
            f"prefix_skipped={len(skipped_ik_items)}, "
            f"pipeline_skipped={len(pipeline_ik_skipped_items)}, "
            f"shortfall={shortfall_now}"
        )
        print(
            f"\n{'=' * 92}\n[PIPELINE attempt {attempt_index}] 一次性搜索 "
            f"arm1={len(arm1_items)}、arm2={len(arm2_items)} 件的连续联合轨迹"
            f"（phase={phase}）\n"
            f"{'=' * 92}"
        )
        if attempt_home_root is not None:
            print(
                f"[HOME ROOT {home_root_cursor + 1}/{len(home_root_candidates)}] "
                f"angle_pair={attempt_home_root.angle_pair_trial}，"
                f"ik_branch={attempt_home_root.ik_branch_index}，"
                f"grasp=(arm1:{attempt_home_root.arm1_choice.grasp:g}, "
                f"arm2:{None if attempt_home_root.arm2_choice is None else float(attempt_home_root.arm2_choice.grasp)})"
            )
            print(
                f"[HOME ROOT] q_deg="
                f"{np.degrees(attempt_q_home).round(2).tolist()}"
            )
        try:
            pipeline_args = (
                mg,
                attempt_q_home,
                [item["position"] for item in arm1_items],
                [item["position"] for item in arm2_items],
                place1_local,
                place2_local,
                pp,
                asr,
                dc,
                target_transforms,
                pl,
                cr,
                linear_cfg,
                q_lo,
                q_hi,
                ee_links,
                attempt_ik_capture,
            )
            if shared_layout:
                if attempt_home_root is None:
                    raise RuntimeError("shared 流水缺少 Home root")
                home_grasp_angles = (
                    float(attempt_home_root.arm1_choice.grasp),
                    None if attempt_home_root.arm2_choice is None else float(
                        attempt_home_root.arm2_choice.grasp
                    ),
                )
                pipeline, pipeline_failures, pipeline_search = try_continuous_pipeline(
                    *pipeline_args,
                    arm2_terminal=attempt_home2 if arm2_items else None,
                    first_home_grasp_angles=home_grasp_angles,
                    first_home_joint_reference=attempt_q_home,
                )
            else:
                pipeline, pipeline_failures, pipeline_search = try_continuous_pipeline(
                    *pipeline_args
                )
        except Exception as exc:  # noqa: BLE001 - 搜索异常绝不转成 skip
            exception_traceback = traceback.format_exc()
            dump_json({
                "stage": "continuous_pipeline_exception",
                "error": str(exc),
                "traceback": exception_traceback,
                "home_root_index": (
                    None if attempt_home_root is None else home_root_cursor
                ),
                "home_root": (
                    None if attempt_home_root is None
                    else attempt_home_root.to_dict()
                ),
                "home_search": home_search_report,
                "selected_item_map": selected_item_map,
                "skipped_ik_items": skipped_ik_items,
                "pipeline_ik_skipped_items": pipeline_ik_skipped_items,
                "pipeline_attempts": pipeline_attempts,
                "config": cfg,
            }, out_dir / "plan_failed.json")
            print(f"[FAIL] 连续流水搜索异常: {exc}")
            print(exception_traceback)
            return 5

        if attempt_ik_capture:
            attempt_ik_capture = {
                **attempt_ik_capture,
                "home_root_index": (
                    None if attempt_home_root is None else home_root_cursor
                ),
                "home_root": (
                    None if attempt_home_root is None
                    else attempt_home_root.to_dict()
                ),
            }
            ik_capture_by_root.append(attempt_ik_capture)
            # 每完成一个 Home root 就原子更新 checkpoint。只有至少
            # 一个事件真正求出 IK 时才覆盖 Home-root 预览；纯
            # IK_FAIL 仍保留前面的全部 Home 分支可视化结果。
            if attempt_ik_capture.get("solved_global_events"):
                try:
                    ik_npz, _ik_meta, ik_count = save_prescreen_ik_checkpoint(
                        mg,
                        out_dir,
                        joint_names,
                        ee_links,
                        cfg,
                        ik_capture_by_root,
                    )
                    print(
                        f"[IK SAVE] deepest prescreen states={ik_count}, "
                        f"root={attempt_ik_capture.get('home_root_index')} -> "
                        f"{ik_npz}"
                    )
                except Exception as exc:  # noqa: BLE001 - 诊断不改变规划成败
                    print(f"[IK SAVE WARN] prescreen IK checkpoint 失败: {exc}")

        if pipeline is not None and attempt_home_root is not None:
            try:
                candidate_gate = evaluate_pipeline_candidate(pipeline)
            except Exception as exc:  # noqa: BLE001 - 安全门不可用必须立即关闭
                dump_json({
                    "stage": "continuous_pipeline_validation_exception",
                    "error": str(exc),
                    "home_root_index": home_root_cursor,
                    "home_root": attempt_home_root.to_dict(),
                    "home_search": home_search_report,
                    "selected_item_map": selected_item_map,
                    "skipped_ik_items": skipped_ik_items,
                    "pipeline_ik_skipped_items": pipeline_ik_skipped_items,
                    "pipeline_attempts": pipeline_attempts,
                    "config": cfg,
                }, out_dir / "plan_failed.json")
                print(f"[FAIL] Home 根候选轨迹安全检查异常: {exc}")
                return 6
            candidate_errors = list(candidate_gate["errors"])
            if candidate_errors:
                pipeline_failures = list(pipeline_failures) + [{
                    "pipeline_block": "post_validation",
                    "angle_stage": pipeline.angle_stage,
                    "stage": "post_validation",
                    "status": "POST_VALIDATION_REJECTED",
                    "errors": candidate_errors,
                }]
                pipeline = None
                print(
                    f"[HOME ROOT] 候选整线未通过后验安全门: "
                    f"{candidate_errors}"
                )

        attempt_report: Dict[str, Any] = {
            "attempt": attempt_index,
            "selected_source_item_indices": [
                item["source_item_index"] for item in selected_item_map
            ],
            "success": pipeline is not None,
            "search": pipeline_search,
            "n_failures": len(pipeline_failures),
        }
        if candidate_gate is not None:
            attempt_report["post_validation_errors"] = list(
                candidate_gate["errors"]
            )
        if attempt_home_root is not None:
            failure_status_counts: Dict[str, int] = {}
            for failure in pipeline_failures:
                status = str(
                    failure.get("status")
                    or failure.get("stage")
                    or "UNKNOWN"
                )
                failure_status_counts[status] = (
                    failure_status_counts.get(status, 0) + 1
                )
            attempt_report.update({
                "home_root_index": home_root_cursor,
                "home_root": attempt_home_root.to_dict(),
                "failure_status_counts": failure_status_counts,
                "failure_examples": [
                    {
                        key: value for key, value in failure.items()
                        if key != "segments"
                    }
                    for failure in pipeline_failures[:3]
                ],
            })
        if pipeline is not None:
            pipeline_attempts.append(attempt_report)
            if attempt_home_root is not None:
                selected_home_root = attempt_home_root
                selected_candidate_gate = candidate_gate
                q_home = attempt_q_home.copy()
                home1, home2 = attempt_home1, attempt_home2
                pipeline_search = {
                    **pipeline_search,
                    "selected_home_root_index": home_root_cursor,
                    "selected_home_angle_pair_trial": (
                        attempt_home_root.angle_pair_trial
                    ),
                    "selected_home_ik_branch_index": (
                        attempt_home_root.ik_branch_index
                    ),
                }
                home_search_report = {
                    "mode": "first_pregrasp_root_search",
                    "enumeration": home_root_enumeration,
                    "n_root_attempts": home_root_cursor + 1,
                    "total_pipeline_search_nodes": sum(
                        int(attempt.get("search", {}).get("search_nodes", 0))
                        for attempt in pipeline_attempts
                    ),
                    "selected_root_index": home_root_cursor,
                    "selected_root": attempt_home_root.to_dict(),
                }
                print(
                    f"[HOME ROOT] 根 {home_root_cursor} 整线规划成功；"
                    f"angle_pair={attempt_home_root.angle_pair_trial}，"
                    f"ik_branch={attempt_home_root.ik_branch_index}"
                )
            break

        if attempt_home_root is not None:
            attempt_report["retry"] = (
                "next_home_root"
                if home_root_cursor + 1 < len(home_root_candidates)
                else None
            )
            pipeline_attempts.append(attempt_report)
            failed_root_index = home_root_cursor
            home_root_cursor += 1
            if home_root_cursor < len(home_root_candidates):
                print(
                    f"[HOME ROOT] 根 {failed_root_index} 不能连续到 place，"
                    f"丢弃本轮全部 chunk，从根 {home_root_cursor} "
                    "的 q_home 重新规划整条轨迹"
                )
                continue

            enumeration_incomplete = bool(
                home_root_enumeration.get("search_incomplete", False)
            )
            pipeline_incomplete = any(
                bool(attempt.get("search", {}).get("node_budget_exhausted"))
                or bool(attempt.get("search", {}).get("pair_budget_exhausted"))
                for attempt in pipeline_attempts
            )
            root_search_incomplete = enumeration_incomplete or pipeline_incomplete
            failed_home_search = {
                "mode": "first_pregrasp_root_search",
                "enumeration": home_root_enumeration,
                "n_root_attempts": len(pipeline_attempts),
                "total_pipeline_search_nodes": sum(
                    int(attempt.get("search", {}).get("search_nodes", 0))
                    for attempt in pipeline_attempts
                ),
                "selected_root_index": None,
                "selected_root": None,
                "all_attempted_roots_failed": True,
                "enumeration_incomplete": enumeration_incomplete,
                "pipeline_search_incomplete": pipeline_incomplete,
                "search_incomplete": root_search_incomplete,
                "result_status": (
                    "HOME_ROOT_SEARCH_BUDGET_EXHAUSTED"
                    if root_search_incomplete
                    else "ALL_ENUMERATED_HOME_ROOTS_FAILED"
                ),
            }
            dump_json({
                "stage": "continuous_pipeline_root_search",
                "failures": pipeline_failures,
                "search": pipeline_search,
                "home_search": failed_home_search,
                "selected_item_map": selected_item_map,
                "skipped_ik_items": skipped_ik_items,
                "pipeline_ik_skipped_items": pipeline_ik_skipped_items,
                "pipeline_attempts": pipeline_attempts,
                "config": cfg,
            }, out_dir / "plan_failed.json")
            print(
                f"[FAIL] 预算内 {len(home_root_candidates)} 个首件 Home 姿态/IK 分支"
                "均未通过完整联合流水规划；未把下游失败误当作"
                "source IK 无解跳过"
            )
            return 5

        retry = (
            classify_pipeline_ik_retry(
                pipeline_failures,
                pipeline_search,
                selected_item_map,
                phase=phase,
                sequence_length=nominal_seq_len,
            )
            if on_fail_mode == "skip"
            else None
        )
        if retry is None:
            attempt_report["retry"] = None
            pipeline_attempts.append(attempt_report)
            dump_json({
                "stage": "continuous_pipeline_search",
                "failures": pipeline_failures,
                "search": pipeline_search,
                "selected_item_map": selected_item_map,
                "skipped_ik_items": skipped_ik_items,
                "pipeline_ik_skipped_items": pipeline_ik_skipped_items,
                "pipeline_attempts": pipeline_attempts,
                "config": cfg,
            }, out_dir / "plan_failed.json")
            print("[FAIL] 没有找到完整、连续且通过联合碰撞约束的双臂流水轨迹；"
                  "失败不是可唯一归因的独立 IK/组合搜索失败，未继续跳过")
            return 5

        culprit = int(retry["pipeline_item_index"])
        if culprit < 0 or culprit >= len(arm1_items):
            raise RuntimeError("pipeline IK retry 返回越界的物料索引")
        removed1 = arm1_items.pop(culprit)
        removed2 = arm2_items.pop(culprit)
        pipeline_skip = {
            "skip_reason": "PIPELINE_JOINT_IK_SEARCH_FAILED",
            "source_item_index": int(removed1.get("index", culprit)),
            "arm1_local_position": list(removed1.get("position", [])),
            "arm1_raw_root_position": list(removed1.get("root_position", [])),
            "arm2_local_position": list(removed2.get("position", [])),
            "arm2_raw_root_position": list(removed2.get("root_position", [])),
            "retry_certificate": retry,
        }
        pipeline_skip = enrich_skipped_item(pipeline_skip)
        pipeline_ik_skipped_items.append(pipeline_skip)
        attempt_report["retry"] = retry
        pipeline_attempts.append(attempt_report)
        print(
            f"[PIPELINE IK SKIP] source item {pipeline_skip['source_item_index']} "
            f"在 configured search 的 global event "
            f"{retry['deepest_global_event']} 出现唯一可归因的独立 IK/组合搜索失败；"
            "丢弃本轮全部 chunk，从 q_home 整线重规划"
        )

        # 补入后续通过抓取 prefix 必要条件的 source；replacement 本身无
        # prefix IK 时继续向后扫描。source_cursor 单调增加，外循环必然终止。
        needed = (
            None if requested_count is None
            else max(requested_count - len(arm1_items), 0)
        )
        if source_cursor < len(source_arm1_items) and (needed is None or needed > 0):
            try:
                added1, added2, newly_skipped = filter_dual_items_by_primary_ik(
                    source_arm1_items[source_cursor:],
                    source_arm2_items[source_cursor:],
                    needed,
                    "skip",
                    paired_item_ik_probe,
                )
            except Exception as exc:  # noqa: BLE001 - replacement 探测也 fail closed
                dump_json({
                    "stage": "grasp_ik_prefilter_error",
                    "error": str(exc),
                    "selected_item_map": current_selected_item_map(),
                    "skipped_ik_items": skipped_ik_items,
                    "pipeline_ik_skipped_items": pipeline_ik_skipped_items,
                    "pipeline_attempts": pipeline_attempts,
                    "config": cfg,
                }, out_dir / "plan_failed.json")
                print(f"[FAIL] replacement 抓取 IK 预筛异常: {exc}")
                return 4
            consumed = len(added1) + len(newly_skipped)
            source_cursor += consumed
            source_cursor_by_arm = {
                "arm1": source_cursor, "arm2": source_cursor,
            }
            n_scanned_items = source_cursor
            arm1_items.extend(added1)
            arm2_items.extend(added2)
            newly_skipped = [
                enrich_skipped_item(item) for item in newly_skipped
            ]
            skipped_ik_items.extend(newly_skipped)
            for skipped in newly_skipped:
                print(
                    f"[IK SKIP] source item {skipped['source_item_index']} "
                    f"local={np.round(skipped['arm1_local_position'], 4).tolist()}  "
                    f"arm1={skipped['arm1_reachable']} "
                    f"arm2={skipped['arm2_reachable']}"
                )

    if pipeline is None:
        dump_json({
            "stage": "no_pipeline_joint_ik_items",
            "requested_max_items": requested_count,
            "n_candidates_scanned": source_cursor,
            "skipped_ik_items": skipped_ik_items,
            "pipeline_ik_skipped_items": pipeline_ik_skipped_items,
            "pipeline_attempts": pipeline_attempts,
            "config": cfg,
        }, out_dir / "plan_failed.json")
        print("[FAIL] 所有候选均因抓取 IK 或可唯一归因的流水独立 IK/组合失败被跳过；"
              "未保存任何轨迹")
        return 5

    n_scanned_items = source_cursor
    local_items = arm1_items
    selected_item_map = current_selected_item_map()
    shortfall = (
        0 if requested_count is None
        else max(
            requested_count - (
                len(arm1_items) + len(arm2_items)
                if shared_layout else len(local_items)
            ),
            0,
        )
    )

    positions = pipeline.position
    velocities = pipeline.velocity
    accelerations = pipeline.acceleration
    dt = pipeline.dt
    times = np.arange(positions.shape[0], dtype=np.float64) * dt

    pose_records: Dict[str, List[Dict[str, Any]]] = {"arm1": [], "arm2": []}
    for arm_name, sequences in (
        ("arm1", pipeline.arm1_sequences),
        ("arm2", pipeline.arm2_sequences),
    ):
        for item_index, sequence in enumerate(sequences):
            pose_records[arm_name].extend(
                {**pose.to_dict(), "item_index": item_index}
                for pose in sequence
            )

    # 保留旧版 rounds 摘要字段供已有分析脚本读取；真正的执行顺序以顶层
    # segments/global_event_index 为准，同一个 event 可同时属于不同 item。
    round_reports: List[Dict[str, Any]] = []
    if shared_layout:
        for arm, items, choices in (
            (1, arm1_items, pipeline.arm1_choices),
            (2, arm2_items, pipeline.arm2_choices),
        ):
            for item_index, (item, choice) in enumerate(zip(items, choices)):
                events = [
                    int(segment["global_event_index"])
                    for segment in pipeline.segments
                    if segment.get(f"arm{arm}_item_index") == item_index
                ]
                round_reports.append({
                    "index": len(round_reports),
                    "arm": arm,
                    "arm_item_index": item_index,
                    "source_item_index": int(item.get("index", item_index)),
                    "success": True,
                    "fallback_used": False,
                    "phase_stages": phase,
                    "choice": choice.to_dict(),
                    "global_events": events,
                })
    else:
        for item_index in range(len(local_items)):
            arm1_events = [
                int(segment["global_event_index"])
                for segment in pipeline.segments
                if segment.get("arm1_item_index") == item_index
            ]
            arm2_events = [
                int(segment["global_event_index"])
                for segment in pipeline.segments
                if segment.get("arm2_item_index") == item_index
            ]
            round_reports.append({
                "index": item_index,
                "source_item_index": selected_item_map[item_index]["source_item_index"],
                "success": True,
                "arm_success": [True, True],
                "fallback_used": False,
                "phase_stages": phase,
                "arm1_choice": pipeline.arm1_choices[item_index].to_dict(),
                "arm2_choice": pipeline.arm2_choices[item_index].to_dict(),
                "arm1_global_events": arm1_events,
                "arm2_global_events": arm2_events,
            })
    stopped = False
    print(f"[PIPELINE] OK: {len(pipeline.segments)} 个全局工艺事件，"
          f"{positions.shape[0]} 个插值点，angle_stage={pipeline.angle_stage}")

    if selected_candidate_gate is not None:
        ee1_pos = selected_candidate_gate["ee1_pos"]
        ee1_quat = selected_candidate_gate["ee1_quat"]
        ee2_pos = selected_candidate_gate["ee2_pos"]
        ee2_quat = selected_candidate_gate["ee2_quat"]
        ws1 = selected_candidate_gate["workspace_arm1"]
        ws2 = selected_candidate_gate["workspace_arm2"]
        self_coll = selected_candidate_gate["self_collision"]
        pair_coll = selected_candidate_gate["inter_arm_collision"]
    else:
        fk = compute_fk_batched(mg, positions, list(ee_links))
        ee1_pos = fk[f"{ee_links[0]}/pos"]
        ee1_quat = fk[f"{ee_links[0]}/quat"]
        ee2_pos = fk[f"{ee_links[1]}/pos"]
        ee2_quat = fk[f"{ee_links[1]}/quat"]
        if bool(ws.get("check_after_plan", True)):
            ws1 = workspace_report(ee1_pos, np.eye(4), ws)
            ws2 = workspace_report(ee2_pos, t21, ws)
        else:
            ws1 = ws2 = {
                "checked": False, "reason": "workspace.check_after_plan=false"
            }
        self_coll = self_collision_report(mg, positions)
        pair_coll = inter_arm_report(
            mg, positions, rb["dual_arm_prefix"], collision_margin_mm
        )
    if bool(ws.get("report_gripper_extent", False)):
        flange1 = str(rb.get("flange_link") or "LINK_6")
        flange2 = str(
            rb.get("second_flange_link")
            or rb["dual_arm_prefix"] + flange1
        )
        gripper_extent = {
            "arm1": link_sphere_extent_report(
                mg, positions, times, flange1, np.eye(4), ws, "arm1"
            ),
            "arm2": link_sphere_extent_report(
                mg, positions, times, flange2, t21, ws, "arm2"
            ),
        }
    else:
        gripper_extent = {
            "arm1": {"checked": False, "reason": "report_gripper_extent=false"},
            "arm2": {"checked": False, "reason": "report_gripper_extent=false"},
        }
    motion = compute_joint_motion_report(
        positions, joint_names, rb.get("cspace_distance_weight")
    )
    joint_margin = compute_joint_limit_margin_report(
        mg, positions, joint_names, float(pl.get("limit_margin_warn_deg", 1.0))
    )
    if ws1.get("checked"):
        print(f"\n[CHECK] arm1 workspace: "
              f"{ws1['n_violation']}/{ws1['n_points']} 越界")
        print(f"[CHECK] arm2 workspace(local): "
              f"{ws2['n_violation']}/{ws2['n_points']} 越界")
    else:
        print(f"\n[CHECK] workspace: skipped ({ws1.get('reason')})")
    print(f"[CHECK] combined self collision: {self_coll}")
    print(f"[CHECK] inter-arm: collision_points={pair_coll.get('n_collision_points')}  "
          f"min_clearance={pair_coll.get('min_clearance_mm')}mm")

    validation_errors: List[Dict[str, Any]] = []
    if bool(pl.get("self_collision_check", True)):
        if not self_coll.get("checked"):
            validation_errors.append({
                "check": "self_collision", "reason": self_coll.get("reason", "unavailable")
            })
        elif int(self_coll.get("n_collision", 0)) > 0:
            validation_errors.append({
                "check": "self_collision",
                "n_collision": int(self_coll["n_collision"]),
            })
    if not pair_coll.get("checked"):
        validation_errors.append({
            "check": "inter_arm_collision",
            "reason": pair_coll.get("reason", "unavailable"),
        })
    elif int(pair_coll.get("n_collision_points", 0)) > 0:
        validation_errors.append({
            "check": "inter_arm_collision",
            "n_collision_points": int(pair_coll["n_collision_points"]),
            "margin_mm": float(pair_coll.get("margin_mm", 0.0)),
        })
    if bool(ws.get("check_after_plan", True)) and bool(ws.get("fail_on_violation", False)):
        for arm_name, report in (("arm1", ws1), ("arm2", ws2)):
            if not report.get("checked"):
                validation_errors.append({
                    "check": "workspace", "arm": arm_name,
                    "reason": report.get("reason", "unavailable"),
                })
            elif int(report.get("n_violation", 0)) > 0:
                validation_errors.append({
                    "check": "workspace", "arm": arm_name,
                    "n_violation": int(report["n_violation"]),
                    "max_violation_mm": float(report.get("max_violation_mm", 0.0)),
                })
    validation = {
        "success": not validation_errors,
        "errors": validation_errors,
    }
    failure_marker_path = out_dir / "plan_failed.json"
    if validation_errors:
        print(f"[FAIL] 规划后安全校验未通过: {validation_errors}")
        # 必须先发布失败 marker，再保存诊断轨迹。这样即使 npz/meta/csv 写入
        # 中途异常，RViz 的自动选取也不会把已知不安全的目录当作成功结果。
        dump_json({
            "stage": "post_validation",
            "errors": validation_errors,
            "selected_item_map": selected_item_map,
            "skipped_ik_items": skipped_ik_items,
            "pipeline_ik_skipped_items": pipeline_ik_skipped_items,
            "pipeline_attempts": pipeline_attempts,
            "workspace_check": {"arm1": ws1, "arm2": ws2},
            "self_collision_check": self_coll,
            "dual_arm_collision_check": pair_coll,
            "trajectory": str(out_dir / "trajectory.npz"),
        }, failure_marker_path)

    skipped_rounds: List[Dict[str, Any]] = (
        list(skipped_ik_items) + list(pipeline_ik_skipped_items)
    )
    successful_rounds = round_reports
    expected_global_events = max(
        len(arm1_items) * nominal_seq_len,
        phase + len(arm2_items) * nominal_seq_len
        + (1 if shared_layout and arm2_items else 0),
    )
    parked_arm2 = pipeline.parked_joint_refs[1]
    meta: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_type": "dual_pick_place_interleaved",
        "coordination": {
            "planner": (
                "separate 6-DOF IK Cartesian pairing; 12-DOF MotionGen "
                "joint goals for unconstrained overlap and dual pose goals "
                "for Cartesian-metric segments"
            ),
            "discrete_ik": "separate_6dof_then_12dof_collision_filter",
            "collision_model": "both arms in combined self-collision at every horizon step",
            "scheduler": "prime + rolling blocks with bounded DFS backtracking",
            "task_layout": task_layout,
            "phase_stages": phase,
            "sequence_length": nominal_seq_len,
            "per_item_barrier": False,
            "expected_global_events": expected_global_events,
            "sequential_fallback": False,
            "ik_unreachable_policy": (
                "offline_independent_primary_grasp_skip"
                if shared_layout and on_fail_mode == "skip"
                else "offline_pair_skip_and_full_replan"
                if on_fail_mode == "skip"
                else "stop"
            ),
            "arm2_terminal_home": bool(shared_layout and arm2_items),
            "home_mode": home_search_report.get("mode", "configured"),
            "home_search": home_search_report,
            "selected_home_root": (
                None if selected_home_root is None
                else selected_home_root.to_dict()
            ),
            "home_filter_stabilization": home_stabilization,
            "arm2_terminal_home_pose": (
                None if pipeline.arm2_terminal is None
                else pipeline.arm2_terminal.to_dict()
            ),
            "arm2_parked_joint_deg": (
                None if parked_arm2 is None
                else np.degrees(np.asarray(parked_arm2)[6:12]).round(4).tolist()
            ),
            "parked_joint_tolerance_deg": inactive_tol,
            "home_branch_tolerance_deg": home_branch_tol,
            "home_branch_checks": [
                {
                    "global_event_index": segment.get("global_event_index"),
                    **segment["home_branch_check"],
                }
                for segment in pipeline.segments
                if isinstance(segment.get("home_branch_check"), dict)
            ],
            "home_branch_hold_checks": [
                {
                    "global_event_index": segment.get("global_event_index"),
                    **segment["arm2_pre_activation_home_hold"],
                }
                for segment in pipeline.segments
                if isinstance(
                    segment.get("arm2_pre_activation_home_hold"), dict
                )
            ],
        },
        "robot": {
            "robot_yml": rb["robot_yml"], "base_link": rb["base_link"],
            "ee_link": ee_links[0], "second_ee_link": ee_links[1],
            "dual_arm_prefix": rb["dual_arm_prefix"], "joint_names": joint_names,
            "urdf": rb["urdf"], "urdf_abs": str(resolve_repo_path(rb["urdf"])),
            "mounts": rb["mounts"],
            "arm2_in_arm1_transform": t12.tolist(),
            "link0_target_transform": link0_target_transform.tolist(),
            "arm_target_transforms": {
                "arm1": target_transforms[0].tolist(),
                "arm2": target_transforms[1].tolist(),
            },
        },
        "home_joint_deg": np.degrees(q_home).round(4).tolist(),
        "configured_home_seed_joint_deg": (
            np.degrees(configured_q_home).round(4).tolist()
        ),
        "home_poses": {
            "arm1": home1.to_dict(),
            "arm2": home2.to_dict(),
        },
        "grid": {
            "rows": gg["rows"], "cols": gg["cols"], "z": gg["z"],
            "x_range": gg["x_range"], "y_range": gg["y_range"],
            "order": gg.get("order"), "perimeter_only": gg.get("perimeter_only"),
            "requested_max_items": requested_count,
            "n_candidates_available": len(all_local_items),
            "n_candidates_available_by_arm": {
                "arm1": len(source_arm1_items), "arm2": len(source_arm2_items),
            },
            "n_candidates_scanned": n_scanned_items,
            "n_candidates_scanned_by_arm": source_cursor_by_arm,
            "n_items_per_arm": (
                {"arm1": len(arm1_items), "arm2": len(arm2_items)}
                if shared_layout else len(local_items)
            ),
            "n_items_by_arm": {
                "arm1": len(arm1_items), "arm2": len(arm2_items),
            },
            "n_ik_skipped": len(skipped_rounds),
            "n_primary_grasp_ik_skipped": len(skipped_ik_items),
            "n_pipeline_joint_ik_skipped": len(pipeline_ik_skipped_items),
            "selection_shortfall": shortfall,
            "selected_item_map": selected_item_map,
        },
        "place_positions_arm1_base": {
            "arm1": place1_effective_root.tolist(),
            "arm2": place2_effective_root.tolist(),
        },
        "place_positions_raw_arm1_base": {
            "arm1": place1_root.tolist(), "arm2": place2_root.tolist(),
        },
        "place_position_arm2_local": place2_effective_local.tolist(),
        "place_position_arm2_local_raw": place2_local.tolist(),
        "angle_search": asr, "linear_move": linear_cfg, "criterion": cr,
        "pipeline_search": pipeline_search,
        "pipeline_attempts": pipeline_attempts,
        "segments": pipeline.segments,
        "rounds": round_reports,
        "n_rounds_total": n_scanned_items,
        "n_rounds_success": len(successful_rounds),
        "n_rounds_skipped_or_partial": len(skipped_rounds),
        "skipped_or_partial_rounds": skipped_rounds,
        "ik_prefilter_skipped_items": skipped_ik_items,
        "pipeline_ik_skipped_items": pipeline_ik_skipped_items,
        "pose_sequences": pose_records,
        # 兼容旧播放器的主臂 marker 字段。
        "pose_sequence": pose_records["arm1"],
        "workspace": ws, "planner": pl, "config": cfg,
        "wall_link_restriction": wall_report,
        "workspace_check": {"arm1": ws1, "arm2": ws2},
        "gripper_extent_check": gripper_extent,
        "self_collision_check": self_coll,
        "dual_arm_collision_check": pair_coll,
        "validation": validation,
        "joint_motion": motion, "joint_limit_margin": joint_margin,
        "n_points": int(positions.shape[0]), "interpolation_dt": dt,
        "total_duration_s": float(times[-1]),
        "planning_wall_time_s": float(time.time() - started_all),
        "stopped_early": stopped,
    }
    npz_path, meta_path = save_trajectory(
        out_dir, joint_names, positions, velocities, accelerations, times,
        ee1_pos, ee1_quat, meta,
        extra_arrays={
            "second_ee_positions": ee2_pos,
            "second_ee_quats_wxyz": ee2_quat,
        },
    )
    dump_json(pair_coll, out_dir / "dual_arm_collision.json")

    csv_path = out_dir / "trajectory.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        columns = (["t"] + [f"q_{n}" for n in joint_names]
                   + ["arm1_ee_x", "arm1_ee_y", "arm1_ee_z",
                      "arm2_ee_x", "arm2_ee_y", "arm2_ee_z"])
        f.write(",".join(columns) + "\n")
        for i in range(positions.shape[0]):
            row = ([times[i]] + positions[i].tolist()
                   + ee1_pos[i].tolist() + ee2_pos[i].tolist())
            f.write(",".join(f"{float(x):.7f}" for x in row) + "\n")

    if not validation_errors:
        # --no-timestamp 会复用目录。仅在本轮所有安全检查及三个轨迹文件都
        # 成功后清理旧 marker，避免“成功返回但 RViz 仍拒绝”的假失败；
        # 更早删除则可能在本轮中断时暴露上一条不安全轨迹。
        failure_marker_path.unlink(missing_ok=True)
        skipped_path = out_dir / "plan_skipped.json"
        skipped_path.unlink(missing_ok=True)
        if skipped_rounds:
            dump_json({
                "mode": (
                    "offline_independent_primary_ik_skip"
                    if shared_layout
                    else "offline_ik_skip_and_full_pipeline_replan"
                ),
                "requested_max_items": requested_count,
                "n_candidates_scanned": n_scanned_items,
                "n_candidates_scanned_by_arm": source_cursor_by_arm,
                "n_items_planned_per_arm": (
                    {"arm1": len(arm1_items), "arm2": len(arm2_items)}
                    if shared_layout else len(local_items)
                ),
                "selection_shortfall": shortfall,
                "selected_item_map": selected_item_map,
                "skipped_ik_items": skipped_ik_items,
                "pipeline_ik_skipped_items": pipeline_ik_skipped_items,
                "pipeline_attempts": pipeline_attempts,
            }, skipped_path)

    print(f"\n[SAVE] {npz_path}\n[SAVE] {meta_path}\n[SAVE] {csv_path}")
    print(f"[SAVE] {out_dir / 'dual_arm_collision.json'}")
    if not validation_errors and skipped_rounds:
        print(f"[SAVE] {out_dir / 'plan_skipped.json'}")
    if validation_errors:
        print(f"[FAIL] 不安全轨迹仅保留用于诊断，播放器会跳过该目录: "
              f"{failure_marker_path}")
        return 6
    print(f"\n播放:\n  ./run_rviz.sh --traj {out_dir} --both-arms "
          f"--mounts {rb['mounts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
