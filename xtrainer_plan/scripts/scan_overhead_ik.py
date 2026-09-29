#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch IK endpoint screen for an overhead XTrainer pick/place config.

This is deliberately an IK-only screen.  It checks collision-safe IK at home,
grasp/lift and place/lift endpoints, but it does *not* prove that a continuous
trajectory exists between them.  Confirm shortlisted configurations with the
normal pick/place planner.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_pick_place import (  # noqa: E402
    build_grasp_points,
    couple_sign,
    load_pick_place_config,
    make_round_poses,
    side_candidates,
    transform_pose,
)
from plan_trajectory import (  # noqa: E402
    load_robot_cfg_dict,
    make_world_config,
    restrict_world_collision_to_links,
)
from xtrainer_common import (  # noqa: E402
    PoseSpec,
    dump_json,
    parse_rigid_transform_matrix,
)


IK_ONLY_WARNING = (
    "IK endpoint screening only: success does not prove branch continuity, "
    "Cartesian insertion quality, or a collision-free continuous trajectory."
)


@dataclass
class GoalSpec:
    key: str
    endpoint: str
    raw_pose: PoseSpec
    effective_pose: PoseSpec
    point_index: Optional[int] = None
    angle_index: Optional[int] = None


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="absolute derived pick/place yaml")
    ap.add_argument("--out-dir", required=True, help="absolute output directory")
    ap.add_argument("--rows", type=int, default=None, help="override grasp_grid.rows")
    ap.add_argument("--cols", type=int, default=None, help="override grasp_grid.cols")
    ap.add_argument("--batch-size", type=int, choices=(32, 64), default=64)
    ap.add_argument("--return-seeds", type=int, choices=(4, 8), default=4)
    return ap


def coupled_angles(asr: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Return every configured first-stage coupled angle (ignore max_trials)."""
    if not bool(asr.get("couple_place_to_grasp", False)):
        raise ValueError(
            "scan_overhead_ik requires angle_search.couple_place_to_grasp=true"
        )
    grasp = side_candidates(
        asr["grasp"], str(asr.get("order", "abs")), "angle_search.grasp"
    )
    sign = couple_sign(asr["grasp"])
    return [(float(g), float(sign * g + 0.0)) for g in grasp]


def _named_pose(seq: Sequence[PoseSpec], suffix: str, fallback: str) -> PoseSpec:
    for pose in seq:
        if pose.name.endswith(suffix):
            return pose
    for pose in seq:
        if pose.name.endswith(fallback):
            return pose
    raise ValueError(f"pose sequence lacks {suffix} and fallback {fallback}")


def build_goals(
    items: List[Dict[str, Any]],
    angles: List[Tuple[float, float]],
    pp: Dict[str, Any],
    correction: np.ndarray,
) -> List[GoalSpec]:
    """Build unique goals while retaining the exact normal-planner pose logic."""
    if not items:
        raise ValueError("grasp grid is empty")
    goals: List[GoalSpec] = []
    home = pp["home"]
    home_raw = PoseSpec.from_rpy_deg(
        "home", home["position"], home["rpy_deg"], "start"
    )
    goals.append(GoalSpec("home", "home", home_raw, transform_pose(home_raw, correction)))

    place_pos = list(pp["place"]["position"])
    # Place goals are common to every grasp point, so solve them only once per angle.
    for ai, (ag, ap) in enumerate(angles):
        seq = make_round_poses(items[0]["position"], place_pos, ag, ap, pp, 0)
        place_lift = _named_pose(seq, "_p_lift_in", "_place")
        place = _named_pose(seq, "_place", "_place")
        for endpoint, raw in (("place_lift_in", place_lift), ("place", place)):
            goals.append(
                GoalSpec(
                    f"place/a{ai}/{endpoint}", endpoint, raw,
                    transform_pose(raw, correction), angle_index=ai,
                )
            )

    for item in items:
        pi = int(item["index"])
        for ai, (ag, ap) in enumerate(angles):
            seq = make_round_poses(item["position"], place_pos, ag, ap, pp, pi)
            grasp_lift = _named_pose(seq, "_g_lift_in", "_grasp")
            grasp = _named_pose(seq, "_grasp", "_grasp")
            for endpoint, raw in (("grasp_lift_in", grasp_lift), ("grasp", grasp)):
                goals.append(
                    GoalSpec(
                        f"point/p{pi}/a{ai}/{endpoint}", endpoint, raw,
                        transform_pose(raw, correction), point_index=pi, angle_index=ai,
                    )
                )
    return goals


def make_ik_solver(robot_dict, world, planner: Dict[str, Any]):
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    tensor_args = TensorDeviceType()
    checker = getattr(
        CollisionCheckerType,
        str(planner.get("collision_checker_type", "MESH")).upper(),
    )
    config = IKSolverConfig.load_from_robot_config(
        robot_dict,
        world,
        rotation_threshold=float(planner["rotation_threshold"]),
        position_threshold=float(planner["position_threshold"]),
        num_seeds=128,
        self_collision_check=bool(planner["self_collision_check"]),
        self_collision_opt=bool(planner["self_collision_opt"]),
        tensor_args=tensor_args,
        use_cuda_graph=False,
        collision_activation_distance=float(planner["collision_activation_distance"]),
        collision_checker_type=checker,
    )
    return IKSolver(config), tensor_args


def _margin_deg(q: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    return float(np.degrees(np.minimum(q - q_lo, q_hi - q)).min())


def solve_goals(
    ik,
    tensor_args,
    goals: List[GoalSpec],
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    min_margin_deg: float,
    batch_size: int,
    return_seeds: int,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    from curobo.types.math import Pose

    results: Dict[str, Dict[str, Any]] = {}
    batches: List[Dict[str, Any]] = []
    total = len(goals)
    t_all = time.perf_counter()
    for start in range(0, total, batch_size):
        chunk = goals[start : start + batch_size]
        position = np.stack([g.effective_pose.position for g in chunk], axis=0)
        quaternion = np.stack([g.effective_pose.quat_wxyz for g in chunk], axis=0)
        goal_pose = Pose(
            position=tensor_args.to_device(position),
            quaternion=tensor_args.to_device(quaternion),
        )
        t0 = time.perf_counter()
        solved = ik.solve_batch(
            goal_pose,
            return_seeds=return_seeds,
            num_seeds=128,
            use_nn_seed=False,
        )
        success = solved.success.detach().cpu().numpy().astype(bool)
        solution = solved.solution.detach().cpu().numpy().astype(np.float64)
        pos_err = solved.position_error.detach().cpu().numpy().astype(np.float64)
        rot_err = solved.rotation_error.detach().cpu().numpy().astype(np.float64)
        elapsed = time.perf_counter() - t0
        batches.append(
            {
                "start": start,
                "count": len(chunk),
                "wall_time_s": elapsed,
                "solver_time_s": float(solved.solve_time),
            }
        )

        for i, spec in enumerate(chunk):
            ok_idx = np.flatnonzero(success[i])
            representative = None
            margin = None
            rep_pe = None
            rep_re = None
            feasible = False
            if ok_idx.size:
                margins = np.asarray(
                    [_margin_deg(solution[i, j], q_lo, q_hi) for j in ok_idx]
                )
                # Prefer a returned solution with the largest joint-limit margin.
                local = int(np.argmax(margins))
                chosen = int(ok_idx[local])
                representative = solution[i, chosen]
                margin = float(margins[local])
                rep_pe = float(pos_err[i, chosen])
                rep_re = float(rot_err[i, chosen])
                # Match plan_pick_place.check_limit_margin: zero disables the
                # margin criterion, while the measured margin is still reported.
                feasible = min_margin_deg <= 0.0 or margin + 1e-9 >= min_margin_deg

            results[spec.key] = {
                "endpoint": spec.endpoint,
                "point_index": spec.point_index,
                "angle_index": spec.angle_index,
                "task_pose": spec.raw_pose.to_dict(),
                "effective_pose": spec.effective_pose.to_dict(),
                "ik_success": bool(ok_idx.size),
                "feasible": bool(feasible),
                "returned_success_count": int(ok_idx.size),
                "representative_q_rad": (
                    None if representative is None else representative.tolist()
                ),
                "representative_q_deg": (
                    None if representative is None
                    else np.degrees(representative).tolist()
                ),
                "representative_min_limit_margin_deg": margin,
                "representative_position_error_m": rep_pe,
                "representative_rotation_error": rep_re,
                "best_success_position_error_m": (
                    None if not ok_idx.size else float(np.min(pos_err[i, ok_idx]))
                ),
                "best_success_rotation_error": (
                    None if not ok_idx.size else float(np.min(rot_err[i, ok_idx]))
                ),
                "best_returned_attempt_position_error_m": float(np.min(pos_err[i])),
                "best_returned_attempt_rotation_error": float(np.min(rot_err[i])),
            }
        done = start + len(chunk)
        elapsed_all = time.perf_counter() - t_all
        eta = elapsed_all / done * (total - done)
        print(
            f"[IK] {done}/{total} goals, batch={len(chunk)}, "
            f"elapsed={elapsed_all:.1f}s, eta={eta:.1f}s",
            flush=True,
        )
    return results, batches


def compact_solution(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "feasible": result["feasible"],
        "ik_success": result["ik_success"],
        "q_deg": result["representative_q_deg"],
        "min_limit_margin_deg": result["representative_min_limit_margin_deg"],
    }


def assemble_report(
    cfg: Dict[str, Any],
    config_path: Path,
    correction: np.ndarray,
    items: List[Dict[str, Any]],
    angles: List[Tuple[float, float]],
    goals: List[GoalSpec],
    solved: Dict[str, Dict[str, Any]],
    batches: List[Dict[str, Any]],
    joint_names: List[str],
    wall_report: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    pp = cfg["pick_place"]
    min_margin = float(pp["criterion"].get("min_limit_margin_deg") or 0.0)
    home = solved["home"]

    angle_rows: List[Dict[str, Any]] = []
    place_ok_indices: List[int] = []
    for ai, (ag, ap) in enumerate(angles):
        lift_key = f"place/a{ai}/place_lift_in"
        place_key = f"place/a{ai}/place"
        endpoints = {
            "place_lift_in": bool(solved[lift_key]["feasible"]),
            "place": bool(solved[place_key]["feasible"]),
        }
        place_ok = all(endpoints.values())
        if place_ok:
            place_ok_indices.append(ai)
        angle_rows.append(
            {
                "angle_index": ai,
                "grasp_deg": ag,
                "place_deg": ap,
                "place_endpoint_feasible": endpoints,
                "place_and_lift_feasible": place_ok,
                "place_result_keys": {
                    "place_lift_in": lift_key,
                    "place": place_key,
                },
                "place_representative": {
                    "place_lift_in": compact_solution(solved[lift_key]),
                    "place": compact_solution(solved[place_key]),
                },
            }
        )

    point_rows: List[Dict[str, Any]] = []
    grasp_pair_ok_count = 0
    round_pair_ok_count = 0
    for item in items:
        pi = int(item["index"])
        per_angle: List[Dict[str, Any]] = []
        grasp_ok_angles: List[int] = []
        round_ok_angles: List[int] = []
        for ai, (ag, ap) in enumerate(angles):
            lift_key = f"point/p{pi}/a{ai}/grasp_lift_in"
            grasp_key = f"point/p{pi}/a{ai}/grasp"
            endpoints = {
                "grasp_lift_in": bool(solved[lift_key]["feasible"]),
                "grasp": bool(solved[grasp_key]["feasible"]),
            }
            grasp_ok = all(endpoints.values())
            round_ok = grasp_ok and ai in place_ok_indices
            if grasp_ok:
                grasp_ok_angles.append(ai)
                grasp_pair_ok_count += 1
            if round_ok:
                round_ok_angles.append(ai)
                round_pair_ok_count += 1
            per_angle.append(
                {
                    "angle_index": ai,
                    "grasp_deg": ag,
                    "place_deg": ap,
                    "grasp_endpoint_feasible": endpoints,
                    "grasp_and_lift_feasible": grasp_ok,
                    "grasp_place_endpoint_intersection": round_ok,
                    "grasp_result_keys": {
                        "grasp_lift_in": lift_key,
                        "grasp": grasp_key,
                    },
                }
            )

        # Search order is the same as the planner, so the first intersection is
        # the simple representative angle.  Home deliberately does not gate it.
        best_index = round_ok_angles[0] if round_ok_angles else (
            grasp_ok_angles[0] if grasp_ok_angles else None
        )
        best = None
        if best_index is not None:
            lift_key = f"point/p{pi}/a{best_index}/grasp_lift_in"
            grasp_key = f"point/p{pi}/a{best_index}/grasp"
            place_lift_key = f"place/a{best_index}/place_lift_in"
            place_key = f"place/a{best_index}/place"
            ag, ap = angles[best_index]
            best = {
                "angle_index": best_index,
                "grasp_deg": ag,
                "place_deg": ap,
                "includes_place_intersection": best_index in round_ok_angles,
                "representative_q_deg": {
                    "grasp_lift_in": solved[lift_key]["representative_q_deg"],
                    "grasp": solved[grasp_key]["representative_q_deg"],
                    "place_lift_in": solved[place_lift_key]["representative_q_deg"],
                    "place": solved[place_key]["representative_q_deg"],
                },
            }
        point_rows.append(
            {
                **item,
                "angles": per_angle,
                "grasp_and_lift_feasible_angle_indices": grasp_ok_angles,
                "grasp_place_endpoint_intersection_angle_indices": round_ok_angles,
                "best_angle": best,
            }
        )

    angle_coverage: List[Dict[str, Any]] = []
    for row in angle_rows:
        ai = int(row["angle_index"])
        covered = [
            int(p["index"])
            for p in point_rows
            if ai in p["grasp_place_endpoint_intersection_angle_indices"]
        ]
        row["n_grasp_points_in_endpoint_intersection"] = len(covered)
        row["grasp_point_indices_in_endpoint_intersection"] = covered
        angle_coverage.append(
            {
                "angle_index": ai,
                "grasp_deg": row["grasp_deg"],
                "place_deg": row["place_deg"],
                "place_and_lift_feasible": row["place_and_lift_feasible"],
                "n_grasp_points_in_endpoint_intersection": len(covered),
            }
        )

    best_global = None
    if angle_coverage:
        candidate = max(
            angle_coverage,
            key=lambda x: (x["n_grasp_points_in_endpoint_intersection"], -x["angle_index"]),
        )
        if candidate["n_grasp_points_in_endpoint_intersection"] > 0:
            best_global = dict(candidate)
            ai = int(candidate["angle_index"])
            best_global["place_representative_q_deg"] = {
                endpoint: solved[f"place/a{ai}/{endpoint}"]["representative_q_deg"]
                for endpoint in ("place_lift_in", "place")
            }
            best_global["grasp_representative_q_deg_by_point"] = {
                str(p["index"]): {
                    endpoint: solved[
                        f"point/p{int(p['index'])}/a{ai}/{endpoint}"
                    ]["representative_q_deg"]
                    for endpoint in ("grasp_lift_in", "grasp")
                }
                for p in point_rows
                if ai in p["grasp_place_endpoint_intersection_angle_indices"]
            }

    all_point_intersection = [
        row["angle_index"]
        for row in angle_coverage
        if row["place_and_lift_feasible"]
        and row["n_grasp_points_in_endpoint_intersection"] == len(items)
    ]
    n_points_grasp = sum(
        bool(p["grasp_and_lift_feasible_angle_indices"]) for p in point_rows
    )
    n_points_round = sum(
        bool(p["grasp_place_endpoint_intersection_angle_indices"]) for p in point_rows
    )

    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "warning": IK_ONLY_WARNING,
        "config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "resolved": cfg,
        },
        "frame": {
            "pose_rule": "T_effective = C @ T_task",
            "link0_target_transform_C": correction.tolist(),
            "grid_and_plot_coordinates": "legacy task LINK_0 coordinates",
        },
        "solver": {
            "num_seeds": 128,
            "return_seeds": int(args.return_seeds),
            "batch_size": int(args.batch_size),
            "use_cuda_graph": False,
            "position_threshold_m": float(cfg["planner"]["position_threshold"]),
            "rotation_threshold": float(cfg["planner"]["rotation_threshold"]),
            "minimum_joint_limit_margin_deg": min_margin,
            "self_collision_check": bool(cfg["planner"]["self_collision_check"]),
            "self_collision_opt": bool(cfg["planner"]["self_collision_opt"]),
            "joint_names": joint_names,
            "world_collision_link_restriction": wall_report,
            "batches": batches,
        },
        "endpoint_aliases": {
            "grasp_lift_out": "grasp_lift_in",
            "place_lift_out": "place_lift_in",
        },
        "angle_scope": (
            "all configured first-stage coupled angles; angle_search.stage2 is not screened"
        ),
        "home": compact_solution(home),
        "angles": angle_rows,
        "points": point_rows,
        "pose_results": solved,
        "summary": {
            "n_grasp_points": len(items),
            "n_coupled_angles": len(angles),
            "n_unique_pose_goals": len(goals),
            "home_feasible": bool(home["feasible"]),
            "n_place_angles_place_and_lift_feasible": len(place_ok_indices),
            "place_and_lift_feasible_angle_indices": place_ok_indices,
            "n_grasp_point_angle_pairs_grasp_and_lift_feasible": grasp_pair_ok_count,
            "n_grasp_point_angle_pairs_in_grasp_place_endpoint_intersection": round_pair_ok_count,
            "n_grasp_points_with_any_grasp_angle": n_points_grasp,
            "n_grasp_points_with_any_grasp_place_endpoint_intersection": n_points_round,
            "endpoint_screen_without_home_complete": n_points_round == len(items),
            "overall_smoke_pass_including_home": (
                bool(home["feasible"]) and n_points_round == len(items)
            ),
            "all_grasp_points_endpoint_intersection_angle_indices": all_point_intersection,
            "angle_coverage": angle_coverage,
            "best_global_angle": best_global,
            "home_is_not_a_gate_for_grasp_or_place_counts": True,
        },
    }


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    if not config_path.is_absolute() or not out_dir.is_absolute():
        parser.error("--config and --out-dir must both be absolute paths")
    if not config_path.is_file():
        parser.error(f"config does not exist: {config_path}")
    if args.rows is not None and args.rows < 1:
        parser.error("--rows must be >= 1")
    if args.cols is not None and args.cols < 1:
        parser.error("--cols must be >= 1")

    cfg = load_pick_place_config(str(config_path))
    pp = cfg["pick_place"]
    if args.rows is not None:
        pp["grasp_grid"]["rows"] = int(args.rows)
    if args.cols is not None:
        pp["grasp_grid"]["cols"] = int(args.cols)
    correction = parse_rigid_transform_matrix(
        pp.get("link0_target_transform", np.eye(4)),
        "pick_place.link0_target_transform",
    )
    angles = coupled_angles(pp["angle_search"])
    items = build_grasp_points(pp["grasp_grid"])
    goals = build_goals(items, angles, pp, correction)
    print(
        f"[SCREEN] points={len(items)}, coupled_angles={len(angles)}, "
        f"unique_goals={len(goals)}"
    )
    if len(angles) != 16:
        print(f"[WARN] expected 16 angles for [-30,0]/2deg, config produced {len(angles)}")
    print(f"[WARN] {IK_ONLY_WARNING}")

    world = make_world_config(cfg["workspace"])
    robot_dict = load_robot_cfg_dict(cfg["robot"])
    ik, tensor_args = make_ik_solver(robot_dict, world, cfg["planner"])
    wall_cfg = cfg["workspace"].get("wall") or {}
    wall_links: List[str] = []
    wall_report: Dict[str, Any] = {"applied": False}
    if wall_cfg.get("enable", False) and wall_cfg.get("collision_link_names"):
        wall_links = list(wall_cfg["collision_link_names"])
        prefix = cfg["robot"].get("dual_arm_prefix") or ""
        if prefix:
            wall_links += [prefix + name for name in wall_links]
        wall_report = restrict_world_collision_to_links(ik, wall_links)
        wall_report["applied"] = True

    limits = ik.kinematics.get_joint_limits().position
    q_lo = limits[0].detach().cpu().numpy().astype(np.float64)
    q_hi = limits[1].detach().cpu().numpy().astype(np.float64)
    min_margin = float(pp["criterion"].get("min_limit_margin_deg") or 0.0)
    solved, batches = solve_goals(
        ik, tensor_args, goals, q_lo, q_hi, min_margin,
        args.batch_size, args.return_seeds,
    )

    report = assemble_report(
        cfg, config_path, correction, items, angles, goals, solved, batches,
        list(ik.joint_names), wall_report, args,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "ik_screen.json"
    dump_json(report, output)
    summary = report["summary"]
    print(f"[HOME] feasible={summary['home_feasible']} (does not gate other results)")
    print(
        "[PLACE] place+lift angles "
        f"{summary['n_place_angles_place_and_lift_feasible']}/{len(angles)}"
    )
    print(
        "[GRASP] points with any grasp+lift angle "
        f"{summary['n_grasp_points_with_any_grasp_angle']}/{len(items)}"
    )
    print(
        "[INTERSECTION] points with any same-angle grasp/place endpoint intersection "
        f"{summary['n_grasp_points_with_any_grasp_place_endpoint_intersection']}/{len(items)}"
    )
    print(f"[OUT] {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
