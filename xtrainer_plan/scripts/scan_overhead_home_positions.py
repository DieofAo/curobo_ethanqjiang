#!/usr/bin/env python3
"""Scan only Home position; retain resolved orientation/model/collision settings.

Endpoint IK feasibility is not a continuous-trajectory or safety certificate.
The input config is never modified, and reports are created exclusively.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from plan_pick_place import load_pick_place_config, transform_pose
from plan_trajectory import load_robot_cfg_dict, make_world_config, restrict_world_collision_to_links
from scan_overhead_ik import GoalSpec, IK_ONLY_WARNING, make_ik_solver, solve_goals
from xtrainer_common import PoseSpec, parse_rigid_transform_matrix, resolve_repo_path


def build_home_goals(cfg, original_home):
    ranges = {
        "x": [-.62, -.50, -.41, -.31, -.20],
        "y": [-.10, 0., .15, .30, .40],
        "z": [.03, .10, .15, .20, .25],
    }
    pp = cfg["pick_place"]
    if pp["home"].get("joint_deg") is not None or cfg["robot"].get("dual_arm_prefix"):
        raise ValueError("Expected a Cartesian Home and single-arm config")
    correction = parse_rigid_transform_matrix(pp["link0_target_transform"])
    entries = [(f"grid/{i}", "grid", list(point)) for i, point in enumerate(
        itertools.product(ranges["x"], ranges["y"], ranges["z"]))]
    entries.extend([("original_home", "original_home", list(original_home)),
                    ("configured_home", "configured_home", list(pp["home"]["position"]))])
    goals = []
    for i, (key, source, position) in enumerate(entries):
        raw = PoseSpec.from_rpy_deg(key, position, pp["home"]["rpy_deg"], "start")
        goals.append(GoalSpec(key, source, raw, transform_pose(raw, correction), point_index=i))
    return ranges, goals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original-home", type=float, nargs=3, default=[-.31, .11, .03])
    parser.add_argument("--preferred-margin-deg", type=float, default=15.)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config_path = args.config.resolve()
    before_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    cfg = load_pick_place_config(str(config_path))
    ranges, goals = build_home_goals(cfg, args.original_home)
    print(f"[HOME_SCAN] grid=125, baseline=2, total={len(goals)}, "
          f"fixed_rpy={cfg['pick_place']['home']['rpy_deg']}", flush=True)
    print(f"[WARN] {IK_ONLY_WARNING}", flush=True)
    world = make_world_config(cfg["workspace"])
    robot_dict = load_robot_cfg_dict(cfg["robot"])
    ik, tensor_args = make_ik_solver(robot_dict, world, cfg["planner"])
    wall = cfg["workspace"].get("wall") or {}
    wall_report = {"applied": False}
    if wall.get("enable") and wall.get("collision_link_names"):
        wall_report = restrict_world_collision_to_links(ik, list(wall["collision_link_names"]))
        wall_report["applied"] = True
    limits = ik.kinematics.get_joint_limits().position.detach().cpu().numpy().astype(np.float64)
    criterion_margin = float(cfg["pick_place"]["criterion"].get("min_limit_margin_deg") or 0.)
    solved, batches = solve_goals(ik, tensor_args, goals, limits[0], limits[1],
                                 criterion_margin, batch_size=64, return_seeds=8)
    rows = []
    for goal in goals:
        row = {"key": goal.key, **solved[goal.key]}
        row["home_position"] = goal.raw_pose.position.tolist()
        row["move_from_original_home_m"] = float(np.linalg.norm(
            goal.raw_pose.position - np.asarray(args.original_home)))
        row["move_from_configured_home_m"] = float(np.linalg.norm(
            goal.raw_pose.position - np.asarray(cfg["pick_place"]["home"]["position"])))
        margin = row["representative_min_limit_margin_deg"]
        row["preferred_margin_pass"] = (row["feasible"] and margin is not None
                                        and margin >= args.preferred_margin_deg)
        rows.append(row)
    preferred = sorted((row for row in rows if row["preferred_margin_pass"]),
                       key=lambda row: (row["move_from_original_home_m"],
                                        -row["representative_min_limit_margin_deg"]))
    feasible = sorted((row for row in rows if row["feasible"]),
                      key=lambda row: row["move_from_original_home_m"])
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != before_hash:
        raise RuntimeError("Input config changed during scan")
    urdf = resolve_repo_path(cfg["robot"]["urdf"])
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "warning": IK_ONLY_WARNING,
        "config": {"path": str(config_path), "sha256": before_hash, "resolved": cfg},
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "changes": {"only_home_position_scanned": True,
                    "input_config_modified": False,
                    "home_rpy_deg_unchanged": cfg["pick_place"]["home"]["rpy_deg"],
                    "grasp_place_mount_collision_and_acceptance_unchanged": True},
        "grid_ranges": ranges, "original_home_position": args.original_home,
        "configured_home_position": cfg["pick_place"]["home"]["position"],
        "solver": {"num_seeds": 128, "return_seeds": 8, "batch_size": 64,
                   "use_cuda_graph": False, "joint_names": list(ik.joint_names),
                   "wall_link_restriction": wall_report,
                   "position_threshold": cfg["planner"]["position_threshold"],
                   "rotation_threshold": cfg["planner"]["rotation_threshold"],
                   "configured_min_margin_deg": criterion_margin},
        "summary": {"n_scanned": len(rows), "n_grid": 125,
                    "n_feasible": len(feasible), "n_preferred": len(preferred),
                    "preferred_margin_deg": args.preferred_margin_deg},
        "preferred_candidates": preferred, "feasible_candidates": feasible,
        "results": rows, "batches": batches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"[SUMMARY] {report['summary']}", flush=True)
    for row in preferred[:8]:
        print(f"[PREFERRED] position={row['home_position']} "
              f"move_original={row['move_from_original_home_m']:.6f}m "
              f"margin={row['representative_min_limit_margin_deg']:.6f}deg "
              f"seed_deg={row['representative_q_deg']}", flush=True)
    print(f"[OUT] {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
