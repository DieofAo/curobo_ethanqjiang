#!/usr/bin/env python3
"""Audit URDF-relative joint-limit clipping in fresh IK and MotionGen models.

Accept a pick/place config, or a result directory containing trajectory_meta.json.
No IK solve, MotionGen warmup, trajectory planning, or robot motion is performed.
Solver construction requires the curobo/GPU environment. All saved trajectory
samples, when present, are checked independently against the original URDF limits.
The JSON report is exclusively created; existing reports are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_urdf_limits(path: Path, joint_names: list[str]) -> np.ndarray:
    """Read revolute limits in the requested order, without loading CuRobo."""
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names must be nonempty and unique")
    joints = ET.parse(path).getroot().findall("joint")
    limits = []
    for name in joint_names:
        matches = [joint for joint in joints if joint.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one URDF joint {name!r}")
        joint = matches[0]
        if joint.get("type") != "revolute" or joint.find("mimic") is not None:
            raise ValueError(f"{name}: this radian audit requires a non-mimic revolute joint")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"{name}: missing URDF limit")
        limits.append([float(limit.attrib["lower"]), float(limit.attrib["upper"])])
    raw = np.asarray(limits, dtype=np.float64).T
    if not np.isfinite(raw).all() or np.any(raw[0] >= raw[1]):
        raise ValueError("URDF limits must be finite and lower < upper")
    return raw


def expected_limits(raw: np.ndarray, clip: float) -> np.ndarray:
    if not np.isfinite(clip) or clip < 0:
        raise ValueError("joint_limit_clip must be finite and nonnegative")
    result = np.asarray(raw, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[0] != 2 or not np.isfinite(result).all():
        raise ValueError("raw limits must have finite shape (2, DOF)")
    result[0] += clip
    result[1] -= clip
    if np.any(result[0] >= result[1]):
        raise ValueError("joint_limit_clip leaves an empty joint interval")
    return result


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def compare_limits(actual, expected: np.ndarray, tolerance: float) -> dict[str, Any]:
    actual = as_numpy(actual)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"invalid solver limit array: shape={actual.shape}")
    delta = actual - expected
    return {
        "lower_rad": actual[0].tolist(), "upper_rad": actual[1].tolist(),
        "lower_error_rad": delta[0].tolist(), "upper_error_rad": delta[1].tolist(),
        "max_abs_error_rad": float(np.abs(delta).max()),
        "matches_expected": bool(np.all(np.abs(delta) <= tolerance)),
    }


def audit_solver(solver, names: list[str], expected: np.ndarray,
                 urdf: Path, tolerance: float) -> dict[str, Any]:
    kin = solver.kinematics
    actual_urdf = Path(kin.generator_config.urdf_path).resolve()
    report = {
        "joint_names": list(solver.joint_names),
        "joint_order_matches": list(solver.joint_names) == names,
        "urdf": str(actual_urdf), "urdf_matches_config": actual_urdf == urdf,
        "kinematics": compare_limits(kin.get_joint_limits().position, expected, tolerance),
        "rollouts": [],
    }
    for index, rollout in enumerate(solver.get_all_rollout_instances()):
        constraint = getattr(rollout, "bound_constraint", None)
        if constraint is None:
            raise ValueError(f"rollout {index} has no bound_constraint")
        item = {
            "index": index, "class": type(rollout).__name__,
            "joint_names": list(rollout.joint_names),
            "joint_order_matches": list(rollout.joint_names) == names,
            "constraint_enabled": bool(constraint.enabled),
            "kinematics": compare_limits(
                rollout.kinematics.get_joint_limits().position, expected, tolerance),
            "bound_constraint": compare_limits(constraint.joint_limits.position, expected, tolerance),
        }
        item["passed"] = (item["joint_order_matches"] and item["constraint_enabled"]
                          and item["kinematics"]["matches_expected"]
                          and item["bound_constraint"]["matches_expected"])
        report["rollouts"].append(item)
    report["n_rollouts"] = len(report["rollouts"])
    report["passed"] = bool(
        report["joint_order_matches"] and report["urdf_matches_config"]
        and report["kinematics"]["matches_expected"] and report["rollouts"]
        and all(item["passed"] for item in report["rollouts"]))
    return report


def audit_samples(q, names: list[str], raw: np.ndarray,
                  clip: float, tolerance: float) -> dict[str, Any]:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(names) or not q.shape[0] or not np.isfinite(q).all():
        raise ValueError("positions must have nonempty finite shape (N, DOF)")
    effective = expected_limits(raw, clip)
    raw_margin = np.minimum(q - raw[0], raw[1] - q)
    effective_margin = np.minimum(q - effective[0], effective[1] - q)
    bad = raw_margin < clip - tolerance
    min_per_joint = raw_margin.min(axis=0)
    return {
        "n_samples": int(q.shape[0]), "required_raw_margin_rad": clip,
        "minimum_raw_margin_rad": float(raw_margin.min()),
        "minimum_raw_margin_deg": float(np.degrees(raw_margin.min())),
        "minimum_raw_margin_per_joint_rad": min_per_joint.tolist(),
        "minimum_raw_margin_per_joint_deg": np.degrees(min_per_joint).tolist(),
        "minimum_effective_margin_rad": float(effective_margin.min()),
        "minimum_effective_margin_per_joint_rad": effective_margin.min(axis=0).tolist(),
        "minimum_positions_rad": q.min(axis=0).tolist(),
        "maximum_positions_rad": q.max(axis=0).tolist(),
        "n_violation_samples": int(np.any(bad, axis=1).sum()),
        "n_violations_per_joint": bad.sum(axis=0).tolist(),
        "first_violations": [
            {"sample_index": int(i), "joint": names[j], "position_rad": float(q[i, j]),
             "raw_margin_rad": float(raw_margin[i, j])}
            for i, j in np.argwhere(bad)[:20]],
        "passed": not bool(bad.any()),
    }


def load_inputs(source: Path):
    npz_path = None
    if source.is_dir():
        meta_path = source / "trajectory_meta.json"
        with meta_path.open(encoding="utf-8") as stream:
            cfg = json.load(stream)["config"]
        source_file = meta_path
        if (source / "trajectory.npz").is_file():
            npz_path = source / "trajectory.npz"
    else:
        from plan_pick_place import load_pick_place_config
        cfg = load_pick_place_config(str(source))
        source_file = source
    provenance = {"path": str(source), "config_source": str(source_file),
                  "config_source_sha256": sha256(source_file)}
    samples = None
    if npz_path is not None:
        with np.load(npz_path, allow_pickle=False) as data:
            names = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
                     for v in data["joint_names"]]
            samples = (names, data["positions"].copy())
        provenance.update(trajectory=str(npz_path), trajectory_sha256=sha256(npz_path))
    return cfg, provenance, samples


def build_solvers(cfg, robot_dict):
    """Match pick/place construction, without warmup or graph capture."""
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    from plan_pick_place import single_ee_motion_gen_config
    from plan_trajectory import make_world_config, restrict_world_collision_to_links
    from scan_overhead_ik import make_ik_solver

    world, planner = make_world_config(cfg["workspace"]), cfg["planner"]
    ik, _ = make_ik_solver(robot_dict, world, planner)
    mg_robot = single_ee_motion_gen_config(robot_dict, cfg["robot"]["ee_link"])
    mg_cfg = MotionGenConfig.load_from_robot_config(
        mg_robot, world, TensorDeviceType(),
        interpolation_dt=float(planner["interpolation_dt"]),
        collision_checker_type=getattr(
            CollisionCheckerType, str(planner.get("collision_checker_type", "MESH")).upper()),
        collision_activation_distance=float(planner["collision_activation_distance"]),
        self_collision_check=bool(planner["self_collision_check"]),
        self_collision_opt=bool(planner["self_collision_opt"]),
        num_ik_seeds=int(planner["num_ik_seeds"]),
        num_trajopt_seeds=int(planner["num_trajopt_seeds"]),
        num_graph_seeds=int(planner["num_graph_seeds"]),
        trajopt_tsteps=int(planner["trajopt_tsteps"]),
        position_threshold=float(planner["position_threshold"]),
        rotation_threshold=float(planner["rotation_threshold"]),
        velocity_scale=float(planner["velocity_scale"]),
        acceleration_scale=float(planner["acceleration_scale"]),
        jerk_scale=float(planner["jerk_scale"]),
        use_cuda_graph=False, store_debug_in_result=False,
    )
    mg = MotionGen(mg_cfg)
    wall = cfg["workspace"].get("wall") or {}
    filters = {}
    if wall.get("enable") and wall.get("collision_link_names"):
        for label, solver in (("independent_ik", ik), ("motion_gen", mg)):
            filters[label] = restrict_world_collision_to_links(solver, list(wall["collision_link_names"]))
    return ik, mg, filters


def run_audit(cfg, report, samples, tolerance: float) -> None:
    from plan_trajectory import load_robot_cfg_dict
    from xtrainer_common import resolve_repo_path

    if cfg["robot"].get("dual_arm_prefix"):
        raise ValueError("this audit currently supports independent single-arm configurations only")
    robot_dict = load_robot_cfg_dict(cfg["robot"])
    cspace = robot_dict["robot_cfg"]["kinematics"]["cspace"]
    names = list(cspace["joint_names"])
    clip = float(cspace.get("position_limit_clip", 0.0))
    urdf = resolve_repo_path(cfg["robot"]["urdf"]).resolve()
    raw = read_urdf_limits(urdf, names)
    expected = expected_limits(raw, clip)
    report.update(
        joint_names=names, clip_rad=clip, clip_deg=float(np.degrees(clip)),
        urdf=str(urdf), urdf_sha256=sha256(urdf),
        raw_limits={"lower_rad": raw[0].tolist(), "upper_rad": raw[1].tolist()},
        expected_limits={"lower_rad": expected[0].tolist(), "upper_rad": expected[1].tolist()},
        resolved_config=cfg,
    )
    report["trajectory"] = {"checked": False, "reason": "no trajectory supplied"}
    if samples is not None:
        sample_names, q = samples
        if sample_names != names:
            raise ValueError(f"saved joint order differs: {sample_names} != {names}")
        report["trajectory"] = {"checked": True, **audit_samples(q, names, raw, clip, tolerance)}
    ik, mg, filters = build_solvers(cfg, robot_dict)
    report["world_link_filters"] = filters
    report["independent_ik"] = audit_solver(ik, names, expected, urdf, tolerance)
    report["motion_gen"] = audit_solver(mg, names, expected, urdf, tolerance)
    report["verification_completed"] = True
    report["passed"] = (report["independent_ik"]["passed"] and report["motion_gen"]["passed"]
                        and (samples is None or report["trajectory"]["passed"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="config file or result directory")
    parser.add_argument("--out", required=True, help="new JSON report path")
    parser.add_argument("--tolerance-rad", type=float, default=1e-6)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance_rad) or args.tolerance_rad < 0:
        parser.error("--tolerance-rad must be finite and nonnegative")
    source, out = Path(args.source).expanduser().resolve(), Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"refusing to replace existing report: {out}")
    started = time.perf_counter()
    report = {
        "schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
        "passed": False, "verification_completed": False, "tolerance_rad": args.tolerance_rad,
        "scope": "Fresh independent IK, MotionGen and every rollout joint-bound constraint; "
                 "all saved joint samples versus raw URDF bounds when a trajectory is supplied.",
        "solver_options": {"use_cuda_graph": False, "warmup": False, "ik_solve": False,
                           "motion_planning": False},
        "not_verified": ["collision clearance", "IK endpoint accuracy", "dynamic limits",
                         "between-sample joint extrema", "physical installation safety"],
    }
    try:
        cfg, provenance, samples = load_inputs(source)
        report["source"] = provenance
        run_audit(cfg, report, samples, args.tolerance_rad)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr)
    report["wall_time_s"] = time.perf_counter() - started
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"[JOINT LIMIT AUDIT] passed={report['passed']} out={out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
