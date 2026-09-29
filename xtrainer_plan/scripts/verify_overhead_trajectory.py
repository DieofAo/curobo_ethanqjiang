#!/usr/bin/env python3
"""Independently verify saved overhead single-arm trajectory samples.

Loads the *resolved* trajectory configuration, not today's default YAML. A fresh
IKSolver supplies kinematics and collision constraints, with CUDA graphs off;
no MotionGen object, IK solve, trajectory warmup or saved FK is reused. World
collision applies only to LINK_6/LINK_3; self collision receives every sphere
and retains the robot model's configured adjacent-link exclusions.

This checks saved discrete samples, not collisions between samples, dynamic
limits, Cartesian straightness, or physical safety of an actual installation.
The report is created exclusively and never replaces an existing report.
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
from typing import Any, Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtrainer_common import (  # noqa: E402
    parse_rigid_transform_matrix, resolve_repo_path, rpy_deg_to_matrix,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", help="result directory containing trajectory.npz/meta.json")
    parser.add_argument("--out", help="new JSON path (default: RESULT/independent_verification.json)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--fk-position-tolerance-m", type=float, default=1.0e-5)
    parser.add_argument("--fk-rotation-tolerance-deg", type=float, default=0.01)
    parser.add_argument("--joint-limit-tolerance-rad", type=float, default=1.0e-6)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quaternion_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sign-invariant quaternion angle, stable near zero (unlike acos(dot))."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    chord = np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))
    return np.degrees(4.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))


def inspect_inputs(result: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    """CPU-only input/frame/model audit; does not import torch or construct a solver."""
    meta_path = result / "trajectory_meta.json"
    npz_path = result / "trajectory.npz"
    with meta_path.open(encoding="utf-8") as stream:
        meta = json.load(stream)
    cfg = meta["config"]
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = ("joint_names", "positions", "times", "ee_positions", "ee_quats_wxyz")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"NPZ missing required arrays: {missing}")
    q = arrays["positions"]
    if q.ndim != 2 or q.shape[1] != 6 or q.shape[0] == 0:
        raise ValueError(f"single-arm positions must have nonempty shape (N,6), got {q.shape}")
    n = int(q.shape[0])
    expected_shapes = {
        "joint_names": (6,), "times": (n,), "ee_positions": (n, 3),
        "ee_quats_wxyz": (n, 4), "velocities": q.shape, "accelerations": q.shape,
    }
    for name, shape in expected_shapes.items():
        if name in arrays and arrays[name].shape != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {arrays[name].shape}")
    numeric_report: Dict[str, Any] = {}
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number):
            bad = ~np.isfinite(array)
            numeric_report[name] = {
                "shape": list(array.shape), "dtype": str(array.dtype),
                "n_nonfinite": int(bad.sum()), "finite": not bool(bad.any()),
            }
        elif name != "joint_names":
            raise ValueError(f"unexpected nonnumeric NPZ array {name}: dtype={array.dtype}")
    if not all(v["finite"] for v in numeric_report.values()):
        raise ValueError(f"nonfinite NPZ numeric array(s): {numeric_report}")
    for name in required[1:]:
        if name not in numeric_report:
            raise ValueError(f"required array {name} is not numeric")
    names = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
             for v in arrays["joint_names"]]
    quat_norm = np.linalg.norm(arrays["ee_quats_wxyz"], axis=1)
    if np.any(quat_norm < 1.0e-12):
        raise ValueError("stored EE quaternion has zero norm")

    robot = cfg["robot"]
    mount = parse_rigid_transform_matrix(robot["mount_transform"], "robot.mount_transform")
    correction = parse_rigid_transform_matrix(
        cfg["pick_place"]["link0_target_transform"], "pick_place.link0_target_transform")
    inverse_error = float(np.max(np.abs(mount @ correction - np.eye(4))))
    overhead = cfg.get("overhead")
    expected_rpy = overhead.get("mount_rpy_deg") if isinstance(overhead, dict) else None
    if (not isinstance(expected_rpy, (list, tuple)) or len(expected_rpy) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   for v in expected_rpy)
            or not np.all(np.isfinite(expected_rpy))):
        raise ValueError("overhead.mount_rpy_deg must contain three finite real numbers")
    expected_rotation = rpy_deg_to_matrix(expected_rpy)
    rotation_error = float(np.max(np.abs(mount[:3, :3] - expected_rotation)))
    base_z_error = float(np.max(np.abs(mount[:3, 2] - expected_rotation[:, 2])))
    meta_c = parse_rigid_transform_matrix(
        meta["robot"]["link0_target_transform"], "meta.robot.link0_target_transform")
    meta_c_error = float(np.max(np.abs(meta_c - correction)))

    urdf = resolve_repo_path(robot["urdf"]).resolve()
    root = ET.parse(urdf).getroot()
    tcp_joints = [joint for joint in root.findall("joint")
                  if joint.find("child") is not None
                  and joint.find("child").get("link") == robot["ee_link"]]
    if len(tcp_joints) != 1:
        raise ValueError(f"expected one TCP parent joint in {urdf}, got {len(tcp_joints)}")
    tcp_joint = tcp_joints[0]
    origin = tcp_joint.find("origin")
    if origin is None:
        raise ValueError("TCP joint is missing origin")
    tcp_xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    tcp_rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    if tcp_xyz.shape != (3,) or tcp_rpy.shape != (3,):
        raise ValueError("TCP origin must contain three xyz/rpy values")
    tcp_ok = (tcp_joint.get("type") == "fixed"
              and tcp_joint.find("parent").get("link") == "LINK_6"
              and np.allclose(tcp_xyz, [0.0, 0.0, 0.19], atol=1.0e-10, rtol=0)
              and np.allclose(tcp_rpy, 0, atol=1.0e-10, rtol=0))
    urdf_hash = sha256_file(urdf)
    recorded_hash = cfg.get("overhead", {}).get("urdf_sha256")
    wall = cfg["workspace"].get("wall") or {}
    wall_links = list(wall.get("collision_link_names") or [])
    flags = {
        "complete_metadata": not bool(meta.get("partial", False)),
        "sample_count_matches_metadata": int(meta["n_points"]) == n,
        "all_numeric_arrays_finite": True,
        "time_strictly_increasing": bool(np.all(np.diff(arrays["times"]) > 0)),
        "saved_quaternions_unit": bool(np.max(np.abs(quat_norm - 1.0)) <= 1.0e-5),
        "single_arm_six_joint_names": names == [f"J_{i}" for i in range(1, 7)]
            and not bool(robot.get("dual_arm_prefix")) and robot["base_link"] == "LINK_0",
        "metadata_joint_names_match": names == list(meta["robot"]["joint_names"]),
        "mount_times_correction_identity": inverse_error <= 1.0e-8,
        "metadata_correction_matches_config": meta_c_error <= 1.0e-8,
        "mount_rotation_matches_config": rotation_error <= 1.0e-8,
        "new_base_z_matches_config": base_z_error <= 1.0e-8,
        "tcp_0p19_from_link6": bool(tcp_ok),
        "urdf_matches_recorded_hash": recorded_hash is not None and recorded_hash == urdf_hash,
        "self_collision_enabled": bool(cfg["planner"]["self_collision_check"]),
        "world_collision_enabled_exact_links": bool(wall.get("enable"))
            and len(wall_links) == 2 and set(wall_links) == {"LINK_6", "LINK_3"},
    }
    report = {
        "source": {"result": str(result), "npz_sha256": sha256_file(npz_path),
                   "metadata_sha256": sha256_file(meta_path), "urdf": str(urdf),
                   "urdf_sha256": urdf_hash, "recorded_urdf_sha256": recorded_hash},
        "input_checks": flags,
        "numeric_arrays": numeric_report,
        "n_samples": n, "joint_names": names,
        "n_items_success": meta.get("n_items_success"),
        "n_items_total": meta.get("n_items_total"),
        "frames": {"mount_M": mount.tolist(), "correction_C": correction.tolist(),
                   "max_abs_M_times_C_minus_identity": inverse_error,
                   "max_abs_metadata_C_error": meta_c_error,
                   "expected_mount_rpy_deg": list(expected_rpy),
                   "expected_mount_rotation": expected_rotation.tolist(),
                   "max_abs_mount_rotation_error": rotation_error,
                   "base_positive_z_in_old_frame": mount[:3, 2].tolist(),
                   "expected_base_positive_z_in_old_frame": expected_rotation[:, 2].tolist(),
                   "max_abs_base_z_error": base_z_error},
        "tcp": {"joint": tcp_joint.get("name"), "xyz_m": tcp_xyz.tolist(),
                "rpy_rad": tcp_rpy.tolist()},
        "saved_quaternion_max_norm_error": float(np.max(np.abs(quat_norm - 1.0))),
    }
    return report, cfg, arrays


def _sample_values(tensor, count: int, label: str) -> np.ndarray:
    array = tensor.detach().cpu().numpy().astype(np.float64)
    if array.size != count:
        raise ValueError(f"{label}: expected one constraint value per sample, got {array.shape}")
    return array.reshape(count)


def verify_on_gpu(report, cfg, arrays, args) -> None:
    import torch
    from plan_trajectory import load_robot_cfg_dict, make_world_config, restrict_world_collision_to_links
    from scan_overhead_ik import make_ik_solver

    robot_dict = load_robot_cfg_dict(cfg["robot"])
    world = make_world_config(cfg["workspace"])
    ik, tensor_args = make_ik_solver(robot_dict, world, cfg["planner"])
    kin = ik.kinematics
    if list(ik.joint_names) != report["joint_names"]:
        raise ValueError(f"independent solver joint order mismatch: {ik.joint_names}")
    solver_urdf = Path(kin.generator_config.urdf_path).resolve()
    if solver_urdf != Path(report["source"]["urdf"]):
        raise ValueError(f"IK model URDF {solver_urdf} differs from recorded/display model")
    restriction = restrict_world_collision_to_links(ik, ["LINK_6", "LINK_3"])
    rollout = ik.rollout_fn
    self_constraint = getattr(rollout, "robot_self_collision_constraint", None)
    world_constraint = getattr(rollout, "primitive_collision_constraint", None)
    for label, constraint in (("self", self_constraint), ("world", world_constraint)):
        if constraint is None or not constraint.enabled:
            raise ValueError(f"independent {label} collision constraint is unavailable/disabled")
    if not getattr(world_constraint, "_xtrainer_link_masked", False):
        raise ValueError("world constraint did not receive required LINK_6/LINK_3 mask")
    if bool(world_constraint.use_sweep):
        raise ValueError("expected discrete IK constraint, unexpectedly configured sweep")
    limits = kin.get_joint_limits().position.detach().cpu().numpy().astype(np.float64)
    q = arrays["positions"].astype(np.float64)
    margins = np.minimum(q - limits[0], limits[1] - q)
    q_bad = np.any(margins < -args.joint_limit_tolerance_rad, axis=1)
    report["joint_position_limits"] = {
        "lower_rad": limits[0].tolist(), "upper_rad": limits[1].tolist(),
        "tolerance_rad": args.joint_limit_tolerance_rad,
        "n_violation_samples": int(q_bad.sum()),
        "first_violation_indices": np.flatnonzero(q_bad)[:20].tolist(),
        "minimum_margin_deg": float(np.degrees(margins.min())),
        "minimum_margin_per_joint_deg": np.degrees(margins.min(axis=0)).tolist(),
        "n_samples_margin_below_1deg": int(np.any(margins < np.radians(1.0), axis=1).sum()),
    }
    n = q.shape[0]
    pos_error, rot_error = np.empty(n), np.empty(n)
    self_values, world_values = np.empty(n), np.empty(n)
    spheres_count = None
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            state = kin.get_state(tensor_args.to_device(q[start:end]))
            fk_pos = state.ee_position.detach().cpu().numpy().astype(np.float64)
            fk_quat = state.ee_quaternion.detach().cpu().numpy().astype(np.float64)
            spheres = state.link_spheres_tensor.unsqueeze(1)
            if not bool(torch.isfinite(spheres).all().item()):
                raise ValueError(f"independent collision spheres nonfinite in chunk {start}:{end}")
            if not np.all(np.isfinite(fk_pos)) or not np.all(np.isfinite(fk_quat)):
                raise ValueError(f"independent FK nonfinite in chunk {start}:{end}")
            if np.any(np.linalg.norm(fk_quat, axis=1) < 1.0e-12):
                raise ValueError(f"independent FK zero quaternion in chunk {start}:{end}")
            spheres_count = int(spheres.shape[-2])
            if spheres_count != (restriction["n_spheres_under_wall"]
                                 + restriction["n_spheres_ignored_by_wall"]):
                raise ValueError("collision-sphere count disagrees with world mask")
            # The world wrapper clones spheres; confirm that no radii were altered.
            radii_before = spheres[..., 3].clone()
            self_values[start:end] = _sample_values(
                self_constraint.forward(spheres), end - start, "self")
            world_values[start:end] = _sample_values(
                world_constraint.forward(spheres, env_query_idx=None), end - start, "world")
            if not bool(torch.equal(radii_before, spheres[..., 3])):
                raise ValueError("world collision masking mutated self-collision sphere radii")
            pos_error[start:end] = np.linalg.norm(fk_pos - arrays["ee_positions"][start:end], axis=1)
            rot_error[start:end] = quaternion_error_deg(fk_quat, arrays["ee_quats_wxyz"][start:end])
            print(f"[VERIFY] independently checked {end}/{n} samples", flush=True)
    if not all(np.all(np.isfinite(a)) for a in (pos_error, rot_error, self_values, world_values)):
        raise ValueError("nonfinite independent FK error/collision constraint output")
    pos_bad = pos_error > args.fk_position_tolerance_m
    rot_bad = rot_error > args.fk_rotation_tolerance_deg
    report["independent_solver"] = {
        "type": "IKSolver constraints and kinematics only", "use_cuda_graph": False,
        "motiongen_constructed": False, "ik_solve_called": False,
        "batch_size": args.batch_size, "urdf": str(solver_urdf),
        "total_self_collision_spheres": spheres_count,
        "self_collision_links": list(kin.generator_config.collision_link_names),
        "configured_self_collision_ignore": robot_dict["robot_cfg"]["kinematics"].get("self_collision_ignore"),
        "world_link_restriction": restriction,
        "world_constraint_activation_distance": world_constraint.activation_distance.detach().cpu().tolist(),
    }
    report["fk_comparison"] = {
        "frame": cfg["robot"]["base_link"], "quaternion_order": "wxyz; q and -q equivalent",
        "position_tolerance_m": args.fk_position_tolerance_m,
        "rotation_tolerance_deg": args.fk_rotation_tolerance_deg,
        "max_position_error_m": float(pos_error.max()),
        "mean_position_error_m": float(pos_error.mean()),
        "max_rotation_error_deg": float(rot_error.max()),
        "n_position_mismatch": int(pos_bad.sum()), "n_rotation_mismatch": int(rot_bad.sum()),
        "first_mismatch_indices": np.flatnonzero(pos_bad | rot_bad)[:20].tolist(),
    }
    for label, values in (("self_collision", self_values), ("world_collision", world_values)):
        bad = values > 0.0
        report[label] = {
            "checked": True, "n_samples": n, "n_collision_samples": int(bad.sum()),
            "first_collision_indices": np.flatnonzero(bad)[:20].tolist(),
            "max_constraint_value": float(values.max()),
            "value_note": "weighted/classified constraint output, not metric penetration distance",
        }
    report["gpu_checks"] = {
        "all_derived_values_finite": True,
        "joint_positions_within_limits": not bool(q_bad.any()),
        "stored_fk_matches_independent_fk": not bool(pos_bad.any() or rot_bad.any()),
        "self_collision_free_at_samples": not bool(np.any(self_values > 0)),
        "world_collision_free_at_samples": not bool(np.any(world_values > 0)),
        "all_self_collision_spheres_retained": bool(spheres_count and spheres_count > 0),
    }


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for attr in ("fk_position_tolerance_m", "fk_rotation_tolerance_deg", "joint_limit_tolerance_rad"):
        if not np.isfinite(getattr(args, attr)) or getattr(args, attr) < 0:
            parser.error(f"--{attr.replace('_', '-')} must be finite and nonnegative")
    result = Path(args.result).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else result / "independent_verification.json"
    if out.exists():
        parser.error(f"refusing to replace existing report: {out}")
    if not result.is_dir():
        parser.error(f"result directory does not exist: {result}")
    started = time.perf_counter()
    report: Dict[str, Any] = {
        "schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
        "passed": False, "verification_completed": False,
        "scope": "All saved discrete trajectory samples; no motion planning is performed.",
        "not_verified": ["between-sample swept/continuous collision", "velocity/acceleration/jerk limits",
                         "Cartesian segment straightness", "real installation structural/physical safety"],
    }
    try:
        inputs, cfg, arrays = inspect_inputs(result)
        report.update(inputs)
        failed_inputs = [key for key, ok in report["input_checks"].items() if not ok]
        if failed_inputs:
            report["error"] = f"input audit failed; GPU audit not run: {failed_inputs}"
        else:
            verify_on_gpu(report, cfg, arrays, args)
            report["verification_completed"] = True
            report["passed"] = all(report["gpu_checks"].values())
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr)
    report["wall_time_s"] = time.perf_counter() - started
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents a late race from overwriting another run.
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"[VERIFY] passed={report['passed']} completed={report['verification_completed']}")
    print(f"[OUT] {out}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
