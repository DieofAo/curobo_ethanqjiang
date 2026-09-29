#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Isolate standalone-IK versus MotionGen-IK differences on one overhead goal.

Run each ``--case`` in a fresh process.  The two cases differ only in whether a
standalone IKSolver is constructed/used before ``mg.plan_single``.  This makes
process-global CUDA/kernel state effects visible without changing the normal
planner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_pick_place import (  # noqa: E402
    build_grasp_points,
    couple_sign,
    load_pick_place_config,
    make_round_poses,
    transform_pose,
)
from plan_trajectory import (  # noqa: E402
    load_robot_cfg_dict,
    make_motion_gen,
    make_world_config,
    restrict_world_collision_to_links,
)
from xtrainer_common import (  # noqa: E402
    PoseSpec,
    dump_json,
    parse_rigid_transform_matrix,
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="absolute overhead pick/place config")
    ap.add_argument("--out-dir", required=True, help="absolute output directory")
    ap.add_argument(
        "--case",
        required=True,
        choices=("pristine_mg", "standalone_after_warmup"),
        help=(
            "pristine_mg: plan before constructing standalone IK; "
            "standalone_after_warmup: construct/use standalone IK after MG warmup but before plan"
        ),
    )
    ap.add_argument(
        "--start-joint-deg", type=float, nargs=6, required=True,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="known collision-valid home q; required to keep pristine_mg free of standalone IK",
    )
    ap.add_argument("--angle-deg", type=float, default=0.0)
    ap.add_argument(
        "--endpoint",
        choices=("grasp_lift_in", "grasp", "place_lift_in", "place"),
        default="grasp_lift_in",
    )
    ap.add_argument(
        "--wall-mask-timing",
        choices=("before_warmup", "after_warmup", "none"),
        default="before_warmup",
    )
    graph = ap.add_mutually_exclusive_group()
    graph.add_argument("--mg-cuda-graph", dest="mg_cuda_graph", action="store_true")
    graph.add_argument("--no-mg-cuda-graph", dest="mg_cuda_graph", action="store_false")
    ap.set_defaults(mg_cuda_graph=None)
    ap.add_argument(
        "--ee-only-link-output",
        action="store_true",
        help=(
            "for MotionGen only, set kinematics.link_names=[ee_link] before construction; "
            "this tests auxiliary-link goals retained by MotionGen.warmup while leaving "
            "collision_link_names and collision spheres unchanged"
        ),
    )
    ap.add_argument("--max-attempts", type=int, default=None)
    return ap


def wall_links_from_config(cfg: Dict[str, Any]) -> List[str]:
    wall = cfg["workspace"].get("wall") or {}
    if not wall.get("enable", False) or not wall.get("collision_link_names"):
        return []
    links = list(wall["collision_link_names"])
    prefix = cfg["robot"].get("dual_arm_prefix") or ""
    if prefix:
        links += [prefix + name for name in links]
    return links


def _find_pose(seq: Sequence[PoseSpec], endpoint: str) -> PoseSpec:
    suffix = {
        "grasp_lift_in": "_g_lift_in",
        "grasp": "_grasp",
        "place_lift_in": "_p_lift_in",
        "place": "_place",
    }[endpoint]
    for pose in seq:
        if pose.name.endswith(suffix):
            return pose
    # A zero lift makes the corresponding lift pose identical to the endpoint.
    fallback = "_grasp" if endpoint.startswith("grasp") else "_place"
    for pose in seq:
        if pose.name.endswith(fallback):
            return pose
    raise ValueError(f"cannot find {endpoint} in {[p.name for p in seq]}")


def build_target(cfg: Dict[str, Any], angle_deg: float, endpoint: str):
    pp = cfg["pick_place"]
    grid = copy.deepcopy(pp["grasp_grid"])
    grid["rows"] = 1
    grid["cols"] = 1
    grid["perimeter_only"] = False
    item = build_grasp_points(grid)[0]
    if not bool(pp["angle_search"].get("couple_place_to_grasp", False)):
        raise ValueError("diagnostic requires couple_place_to_grasp=true")
    place_angle = couple_sign(pp["angle_search"]["grasp"]) * float(angle_deg)
    seq = make_round_poses(
        item["position"], pp["place"]["position"],
        float(angle_deg), float(place_angle), pp, 0,
    )
    raw = _find_pose(seq, endpoint)
    correction = parse_rigid_transform_matrix(
        pp.get("link0_target_transform", np.eye(4)),
        "pick_place.link0_target_transform",
    )
    return item, float(place_angle), raw, transform_pose(raw, correction), correction


def make_standalone_solver(robot_dict, world, planner, wall_links):
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    tensor_args = TensorDeviceType()
    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_dict,
        world,
        rotation_threshold=float(planner["rotation_threshold"]),
        position_threshold=float(planner["position_threshold"]),
        num_seeds=max(int(planner["num_ik_seeds"]), 64),
        self_collision_check=bool(planner["self_collision_check"]),
        self_collision_opt=bool(planner["self_collision_opt"]),
        tensor_args=tensor_args,
        use_cuda_graph=False,
        collision_activation_distance=float(planner["collision_activation_distance"]),
        collision_checker_type=getattr(
            CollisionCheckerType,
            str(planner.get("collision_checker_type", "MESH")).upper(),
        ),
    )
    solver = IKSolver(ik_cfg)
    wall_report = {"applied": False}
    if wall_links:
        wall_report = restrict_world_collision_to_links(
            solver, list(wall_links), quiet=True
        )
        wall_report["applied"] = True
    return solver, tensor_args, wall_report


def tensor_values(value) -> Optional[List[float]]:
    if value is None:
        return None
    return value.detach().cpu().numpy().astype(np.float64).reshape(-1).tolist()


def ik_result_dict(result, q_lo: np.ndarray, q_hi: np.ndarray) -> Dict[str, Any]:
    success = result.success.detach().cpu().numpy().astype(bool)
    solution = result.solution.detach().cpu().numpy().astype(np.float64)
    pos_err = result.position_error.detach().cpu().numpy().astype(np.float64)
    rot_err = result.rotation_error.detach().cpu().numpy().astype(np.float64)
    flat_success = success.reshape(-1)
    flat_solution = solution.reshape(-1, solution.shape[-1])
    flat_pe = pos_err.reshape(-1)
    flat_re = rot_err.reshape(-1)
    indices = np.flatnonzero(flat_success)
    solutions: List[Dict[str, Any]] = []
    for index in indices:
        q = flat_solution[index]
        margin = float(np.degrees(np.minimum(q - q_lo, q_hi - q)).min())
        solutions.append(
            {
                "q_rad": q.tolist(),
                "q_deg": np.degrees(q).tolist(),
                "min_limit_margin_deg": margin,
                "position_error_m": float(flat_pe[index]),
                "rotation_error": float(flat_re[index]),
            }
        )
    return {
        "success": bool(indices.size),
        "returned_success_count": int(indices.size),
        "position_error_all_returned_m": flat_pe.tolist(),
        "rotation_error_all_returned": flat_re.tolist(),
        "successful_solutions": solutions,
        "solver_time_s": float(result.solve_time),
    }


def pose_tensor_dict(pose) -> Dict[str, Any]:
    """Serialize a cuRobo Pose without assuming a particular batch rank."""
    position = getattr(pose, "position", None)
    quaternion = getattr(pose, "quaternion", None)
    return {
        "position": tensor_values(position),
        "quaternion_wxyz": tensor_values(quaternion),
        "position_shape": None if position is None else list(position.shape),
        "quaternion_shape": None if quaternion is None else list(quaternion.shape),
    }


def ik_goal_cache_snapshot(solver) -> Dict[str, Any]:
    """Expose the private IK goal cache for this narrowly scoped diagnostic."""
    goal = getattr(solver, "_goal_buffer", None)
    solve_state = getattr(solver, "_solve_state", None)
    links = None if goal is None else getattr(goal, "links_goal_pose", None)
    link_dict = {}
    if links is not None:
        link_dict = {str(name): pose_tensor_dict(pose) for name, pose in links.items()}
    ee_link = str(solver.kinematics.ee_link)
    return {
        "solve_state": None if solve_state is None else str(solve_state),
        "main_goal": (
            None
            if goal is None or getattr(goal, "goal_pose", None) is None
            else pose_tensor_dict(goal.goal_pose)
        ),
        "links_goal_pose_is_none": links is None,
        "links_goal_pose_keys": list(link_dict),
        "non_ee_link_goal_keys": [name for name in link_dict if name != ee_link],
        "links_goal_pose": link_dict,
        "interpretation": (
            "Any non-ee key after warmup is an auxiliary pose constraint. In this cuRobo "
            "version, a later solve(link_poses=None) does not clear a cached non-None mapping."
        ),
    }


def run_standalone_probes(
    robot_dict,
    world,
    planner,
    wall_links,
    target: PoseSpec,
    start_q: np.ndarray,
    event_log: List[str],
) -> Tuple[Any, Dict[str, Any]]:
    from curobo.types.math import Pose

    event_log.append("construct_standalone_ik")
    solver, tensor_args, wall_report = make_standalone_solver(
        robot_dict, world, planner, wall_links
    )
    limits = solver.kinematics.get_joint_limits().position
    q_lo = limits[0].detach().cpu().numpy().astype(np.float64)
    q_hi = limits[1].detach().cpu().numpy().astype(np.float64)
    goal = Pose(
        position=tensor_args.to_device(target.position.reshape(1, 3)),
        quaternion=tensor_args.to_device(target.quat_wxyz.reshape(1, 4)),
    )
    n_seed = max(int(planner["num_ik_seeds"]), 64)
    all_start_seed = tensor_args.to_device(
        np.repeat(start_q.reshape(1, 1, -1), n_seed, axis=1)
    )
    event_log.append("standalone_solve_prescreen_semantics")
    prescreen = solver.solve_single(
        goal, seed_config=all_start_seed, return_seeds=8,
        num_seeds=n_seed, use_nn_seed=False,
    )
    one_start_seed = tensor_args.to_device(start_q.reshape(1, 1, -1))
    retract = tensor_args.to_device(start_q.reshape(1, -1))
    event_log.append("standalone_solve_motiongen_semantics")
    mg_semantics = solver.solve_single(
        goal, retract_config=retract, seed_config=one_start_seed,
        return_seeds=min(int(planner["num_trajopt_seeds"]), n_seed),
        num_seeds=n_seed, use_nn_seed=False,
    )
    return solver, {
        "wall_filter": wall_report,
        "default_retract_q_deg": np.degrees(
            solver.get_retract_config().detach().cpu().numpy().astype(np.float64)
        ).tolist(),
        "prescreen_semantics": ik_result_dict(prescreen, q_lo, q_hi),
        "motiongen_semantics": ik_result_dict(mg_semantics, q_lo, q_hi),
    }


def run_motiongen_ik_probes(mg, target: PoseSpec, start_q: np.ndarray):
    from curobo.types.math import Pose

    ta = mg.tensor_args
    goal = Pose(
        position=ta.to_device(target.position.reshape(1, 3)),
        quaternion=ta.to_device(target.quat_wxyz.reshape(1, 4)),
    )
    start_retract = ta.to_device(start_q.reshape(1, -1))
    one_start_seed = ta.to_device(start_q.reshape(1, 1, -1))
    mg_semantics = mg.ik_solver.solve_single(
        goal,
        retract_config=start_retract,
        seed_config=one_start_seed,
        return_seeds=int(mg.trajopt_seeds),
        num_seeds=int(mg.ik_seeds),
        use_nn_seed=False,
    )
    robot_retract = mg.ik_solver.get_retract_config().reshape(1, -1)
    all_start_seed = ta.to_device(
        np.repeat(start_q.reshape(1, 1, -1), int(mg.ik_seeds), axis=1)
    )
    prescreen_semantics = mg.ik_solver.solve_single(
        goal,
        retract_config=robot_retract,
        seed_config=all_start_seed,
        return_seeds=min(8, int(mg.ik_seeds)),
        num_seeds=int(mg.ik_seeds),
        use_nn_seed=False,
    )
    limits = mg.kinematics.get_joint_limits().position
    q_lo = limits[0].detach().cpu().numpy().astype(np.float64)
    q_hi = limits[1].detach().cpu().numpy().astype(np.float64)
    return {
        "motiongen_semantics": ik_result_dict(mg_semantics, q_lo, q_hi),
        "prescreen_semantics_inside_motiongen_solver": ik_result_dict(
            prescreen_semantics, q_lo, q_hi
        ),
    }


def run_plan(mg, target: PoseSpec, start_q: np.ndarray, planner: Dict[str, Any]):
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    ta = mg.tensor_args
    start = JointState.from_position(
        ta.to_device(start_q.reshape(1, -1)), joint_names=mg.joint_names
    )
    goal = Pose(
        position=ta.to_device(target.position.reshape(1, 3)),
        quaternion=ta.to_device(target.quat_wxyz.reshape(1, 4)),
    )
    plan_cfg = MotionGenPlanConfig(
        enable_graph=bool(planner.get("enable_graph", False)),
        enable_graph_attempt=int(planner.get("enable_graph_attempt", 3)),
        max_attempts=int(planner["max_attempts"]),
        timeout=float(planner["timeout"]),
        enable_finetune_trajopt=True,
        parallel_finetune=True,
        pose_cost_metric=None,
    )
    result = mg.plan_single(start, goal, plan_cfg)
    success = result.success is not None and bool(result.success.any().item())
    internal_ik = None
    if isinstance(result.debug_info, dict) and result.debug_info.get("ik_result") is not None:
        limits = mg.kinematics.get_joint_limits().position
        q_lo = limits[0].detach().cpu().numpy().astype(np.float64)
        q_hi = limits[1].detach().cpu().numpy().astype(np.float64)
        internal_ik = ik_result_dict(result.debug_info["ik_result"], q_lo, q_hi)
    return {
        "success": success,
        "status": str(result.status),
        "valid_query": bool(result.valid_query),
        "attempts": int(result.attempts),
        "position_error": tensor_values(result.position_error),
        "rotation_error": tensor_values(result.rotation_error),
        "solve_time_s": float(result.solve_time),
        "ik_time_s": float(result.ik_time),
        "trajopt_time_s": float(result.trajopt_time),
        "total_time_s": float(result.total_time),
        "actual_internal_ik": internal_ik,
    }


def captured(label: str, fn: Callable[[], Any], event_log: List[str]) -> Dict[str, Any]:
    event_log.append(label)
    started = time.perf_counter()
    try:
        value = fn()
        return {"completed": True, "wall_time_s": time.perf_counter() - started, "result": value}
    except Exception as exc:  # noqa: BLE001 - diagnostic must preserve later evidence
        return {
            "completed": False,
            "wall_time_s": time.perf_counter() - started,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    if not config_path.is_absolute() or not out_dir.is_absolute():
        parser.error("--config and --out-dir must be absolute")
    if not config_path.is_file():
        parser.error(f"config does not exist: {config_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_pick_place_config(str(config_path))
    planner = cfg["planner"]
    if args.mg_cuda_graph is not None:
        planner["use_cuda_graph"] = bool(args.mg_cuda_graph)
    if args.max_attempts is not None:
        planner["max_attempts"] = int(args.max_attempts)
    item, place_angle, raw_target, target, correction = build_target(
        cfg, args.angle_deg, args.endpoint
    )
    start_q = np.radians(np.asarray(args.start_joint_deg, dtype=np.float64))
    if not np.isfinite(start_q).all():
        parser.error("--start-joint-deg must contain only finite values")
    event_log: List[str] = []
    graph_tag = "graph" if bool(planner.get("use_cuda_graph", True)) else "no_graph"
    link_tag = "ee_only" if args.ee_only_link_output else "all_link_outputs"
    output = out_dir / (
        f"ik_mismatch_{args.case}_{args.wall_mask_timing}_{graph_tag}_{link_tag}.json"
    )
    latest_output = out_dir / "ik_mismatch.json"

    try:
        event_log.append("construct_world_and_robot")
        world = make_world_config(cfg["workspace"])
        robot_dict = load_robot_cfg_dict(cfg["robot"])
        mg_robot_dict = copy.deepcopy(robot_dict)
        original_link_names = list(
            robot_dict["robot_cfg"]["kinematics"].get("link_names") or []
        )
        if args.ee_only_link_output:
            mg_kin = mg_robot_dict["robot_cfg"]["kinematics"]
            mg_kin["link_names"] = [mg_kin["ee_link"]]
        motiongen_link_names = list(
            mg_robot_dict["robot_cfg"]["kinematics"].get("link_names") or []
        )
        wall_links = wall_links_from_config(cfg)
    except Exception as exc:  # noqa: BLE001 - setup failure is the evidence
        failure = {
            "schema_version": 1,
            "case": args.case,
            "stage": "world_or_robot_setup",
            "event_order": event_log,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        dump_json(failure, output)
        dump_json(failure, latest_output)
        print(f"[FAIL] setup: {exc}\n[OUT] {output}")
        return 2

    mg_wall_report: Dict[str, Any] = {"applied": False, "timing": args.wall_mask_timing}

    def pre_warmup_mask(solver) -> None:
        nonlocal mg_wall_report
        event_log.append("apply_mg_wall_filter_before_warmup")
        mg_wall_report = restrict_world_collision_to_links(solver, wall_links)
        mg_wall_report.update({"applied": True, "timing": "before_warmup"})

    try:
        event_log.append("construct_and_warmup_motiongen")
        mg = make_motion_gen(
            mg_robot_dict,
            world,
            planner,
            pre_warmup=(
                pre_warmup_mask
                if wall_links and args.wall_mask_timing == "before_warmup"
                else None
            ),
        )
        if wall_links and args.wall_mask_timing == "after_warmup":
            event_log.append("apply_mg_wall_filter_after_warmup")
            mg_wall_report = restrict_world_collision_to_links(mg, wall_links)
            mg_wall_report.update({"applied": True, "timing": "after_warmup"})
        # This flag only preserves the actual IKResult in MotionGenResult. It does not alter
        # graph capture, solver parameters, seeds, or optimization.
        mg.store_debug_in_result = True
        warmup_ik_goal_cache = ik_goal_cache_snapshot(mg.ik_solver)
    except Exception as exc:  # noqa: BLE001 - warmup/capture failure is the evidence
        failure = {
            "schema_version": 1,
            "case": args.case,
            "stage": "motiongen_construct_or_warmup",
            "event_order": event_log,
            "wall_mask_timing": args.wall_mask_timing,
            "use_cuda_graph": bool(planner.get("use_cuda_graph", True)),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        dump_json(failure, output)
        dump_json(failure, latest_output)
        print(f"[FAIL] MotionGen setup/warmup: {exc}\n[OUT] {output}")
        return 2

    standalone_solver = None
    standalone_report = None
    plan_report = None
    mg_ik_report = None

    def standalone_action():
        nonlocal standalone_solver
        standalone_solver, report = run_standalone_probes(
            robot_dict, world, planner, wall_links, target, start_q, event_log
        )
        return report

    if args.case == "standalone_after_warmup":
        standalone_report = captured(
            "begin_standalone_before_plan", standalone_action, event_log
        )
        plan_report = captured(
            "motiongen_plan_after_standalone",
            lambda: run_plan(mg, target, start_q, planner), event_log,
        )
        mg_ik_report = captured(
            "motiongen_direct_ik_after_plan",
            lambda: run_motiongen_ik_probes(mg, target, start_q), event_log,
        )
    else:
        plan_report = captured(
            "motiongen_plan_without_prior_standalone",
            lambda: run_plan(mg, target, start_q, planner), event_log,
        )
        mg_ik_report = captured(
            "motiongen_direct_ik_without_prior_standalone",
            lambda: run_motiongen_ik_probes(mg, target, start_q), event_log,
        )
        standalone_report = captured(
            "begin_standalone_after_all_motiongen_probes", standalone_action, event_log
        )

    mg_limits = mg.kinematics.get_joint_limits().position
    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "standalone IKSolver / MotionGen IK mismatch isolation",
        "fresh_process_required_per_case": True,
        "case": args.case,
        "event_order": event_log,
        "config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "planner_effective": planner,
        },
        "target": {
            "endpoint": args.endpoint,
            "center_grasp_task_position": item["position"],
            "grasp_angle_deg": float(args.angle_deg),
            "place_angle_deg": place_angle,
            "raw_task_pose": raw_target.to_dict(),
            "effective_robot_base_pose": target.to_dict(),
            "link0_target_transform_C": correction.tolist(),
        },
        "start": {
            "source": "--start-joint-deg (no standalone IK needed)",
            "q_deg": list(args.start_joint_deg),
            "q_rad": start_q.tolist(),
        },
        "solver_comparison": {
            "shared": {
                "robot_yml": cfg["robot"]["robot_yml"],
                "base_link": cfg["robot"]["base_link"],
                "ee_link": cfg["robot"]["ee_link"],
                "num_ik_seeds": int(planner["num_ik_seeds"]),
                "position_threshold_m": float(planner["position_threshold"]),
                "rotation_threshold": float(planner["rotation_threshold"]),
                "self_collision_check": bool(planner["self_collision_check"]),
                "self_collision_opt": bool(planner["self_collision_opt"]),
                "collision_activation_distance_m": float(
                    planner["collision_activation_distance"]
                ),
                "collision_checker_type": planner.get("collision_checker_type", "MESH"),
            },
            "motiongen": {
                "use_cuda_graph": bool(planner.get("use_cuda_graph", True)),
                "wall_filter": mg_wall_report,
                "ee_only_link_output": bool(args.ee_only_link_output),
                "original_link_names": original_link_names,
                "effective_link_names": motiongen_link_names,
                "collision_link_names_unchanged": (
                    robot_dict["robot_cfg"]["kinematics"].get("collision_link_names")
                    == mg_robot_dict["robot_cfg"]["kinematics"].get("collision_link_names")
                ),
                "warmup_ik_goal_cache": warmup_ik_goal_cache,
                "final_ik_goal_cache": ik_goal_cache_snapshot(mg.ik_solver),
                "ik_retract_for_probe_deg": list(args.start_joint_deg),
                "default_robot_retract_deg": np.degrees(
                    mg.ik_solver.get_retract_config().detach().cpu().numpy().astype(np.float64)
                ).tolist(),
                "joint_limit_lower_deg": np.degrees(
                    mg_limits[0].detach().cpu().numpy().astype(np.float64)
                ).tolist(),
                "joint_limit_upper_deg": np.degrees(
                    mg_limits[1].detach().cpu().numpy().astype(np.float64)
                ).tolist(),
            },
            "standalone": {
                "use_cuda_graph": False,
                "wall_filter_applied_before_solve": bool(wall_links),
                "prescreen_seed": (
                    f"{max(int(planner['num_ik_seeds']), 64)} copies of start q; "
                    "default robot retract"
                ),
                "motiongen_semantics_seed": "1 start q + generated seeds; start q retract",
                "kept_alive_during_plan_in_standalone_after_warmup_case": True,
                "constraint_note": (
                    "Standalone always uses the configured link wall mask. With "
                    "wall-mask-timing=none MotionGen intentionally does not, so that mode "
                    "is not a solver-parity comparison."
                ),
            },
        },
        "motiongen_plan": plan_report,
        "motiongen_direct_ik": mg_ik_report,
        "standalone_ik": standalone_report,
        "interpretation": {
            "case_comparison": (
                "If pristine_mg plan succeeds but standalone_after_warmup plan fails, "
                "post-warmup standalone solver creation/use is causal."
            ),
            "retract_seed_comparison": (
                "Within standalone_ik, prescreen success plus motiongen_semantics failure "
                "isolates retract/seed semantics instead of MotionGen."
            ),
            "wall_mask_comparison": (
                "Repeat with before_warmup vs after_warmup in fresh processes; a difference "
                "isolates CUDA graph capture timing."
            ),
            "warmup_aux_link_cache": (
                "If all-link-output has non_ee_link_goal_keys after warmup and fails, while "
                "--ee-only-link-output has no non-ee keys and succeeds, retained auxiliary-link "
                "goals from MotionGen warmup are causal. ee_link itself is harmless here because "
                "ArmReacher explicitly excludes it from auxiliary-link costs and convergence."
            ),
        },
    }
    dump_json(report, output)
    dump_json(report, latest_output)
    print(f"[OUT] {output}")
    print(f"[ORDER] {' -> '.join(event_log)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
