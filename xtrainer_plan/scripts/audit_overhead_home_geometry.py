#!/usr/bin/env python3
"""CPU-only geometric necessary-condition audit for the saved overhead Home.

This is not an IK solver or a trajectory/collision verifier. It reads a resolved
experiment JSON and the referenced URDF, and exclusively creates a new report.
No planning configuration, Home target, or existing report is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from xtrainer_common import parse_rigid_transform_matrix, resolve_repo_path, rpy_deg_to_matrix


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def urdf_origin(joint):
    origin = joint.find("origin")
    if origin is None:
        raise ValueError(f"Missing origin: {joint.get('name')}")
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([xyz, rpy]).all():
        raise ValueError(f"Invalid origin: {joint.get('name')}")
    transform = np.eye(4)
    transform[:3, 3] = xyz
    transform[:3, :3] = rpy_deg_to_matrix(np.degrees(rpy))
    return transform


def audit(config_path, sample_count=512, seed=19):
    cfg = json.loads(config_path.read_text())
    urdf = resolve_repo_path(cfg["robot"]["urdf"]).resolve()
    root = ET.parse(urdf).getroot()
    joint_names = [f"J_{i}" for i in range(1, 7)] + ["TCP_joint"]
    joints = [root.find(f"joint[@name='{name}']") for name in joint_names]
    if any(joint is None for joint in joints):
        raise ValueError("Expected six J_1..J_6 joints and TCP_joint")
    origins = [urdf_origin(joint) for joint in joints]
    unit_z = np.array([0., 0., 1.])
    close = lambda a, b: bool(np.allclose(a, b, atol=1e-10, rtol=0))
    side_offset = float(origins[3][2, 3])
    wrist_offset = float(np.dot(origins[5][:3, 3], origins[5][:3, :3] @ unit_z))
    tcp_offset = float(origins[6][2, 3])
    axial_offset = wrist_offset + tcp_offset
    # These algebraic conditions, not random sampling, establish the identities
    # below. The random FK check is a separate implementation cross-check.
    checks = {
        "expected_chain": all(
            joint.find("parent").get("link") == (f"LINK_{i}" if i < 6 else "LINK_6")
            and joint.find("child").get("link") == (f"LINK_{i+1}" if i < 6 else "TCP_LINK")
            for i, joint in enumerate(joints)),
        "six_revolute_z_axes": all(
            joint.get("type") == "revolute"
            and close(np.fromstring(joint.find("axis").get("xyz"), sep=" "), unit_z)
            for joint in joints[:6]),
        "J1_origin_on_base_axis_no_rotation": close(origins[0][:2, 3], 0)
            and close(origins[0][:3, :3], np.eye(3)),
        "J2_zero_translation_horizontal_axis": close(origins[1][:3, 3], 0)
            and abs(float((origins[1][:3, :3] @ unit_z)[2])) < 1e-10,
        "J3_J4_preserve_common_axis": all(
            close(origins[i][:3, :3] @ unit_z, unit_z) for i in (2, 3)),
        "J3_translation_perpendicular_common_axis": abs(float(origins[2][2, 3])) < 1e-10,
        "J5_translation_perpendicular_common_axis": abs(float(origins[4][2, 3])) < 1e-10,
        "J6_translation_along_final_tool_axis": close(
            origins[5][:3, 3], wrist_offset * (origins[5][:3, :3] @ unit_z)),
        "TCP_fixed_unrotated_on_link6_axis": joints[6].get("type") == "fixed"
            and close(origins[6][:3, :3], np.eye(3))
            and close(origins[6][:3, 3], tcp_offset * unit_z),
        "positive_axial_offset": axial_offset > 0,
    }
    if not all(checks.values()):
        raise ValueError(f"URDF does not support this geometric derivation: {checks}")
    rng = np.random.default_rng(seed)
    limits = np.array([[float(joint.find("limit").get(k)) for k in ("lower", "upper")]
                       for joint in joints[:6]])
    lateral_values, axial_errors, radii = [], [], []
    for q in rng.uniform(limits[:, 0], limits[:, 1], size=(sample_count, 6)):
        transform = np.eye(4)
        frames = []
        for i, origin in enumerate(origins):
            transform = transform @ origin
            if i < 6:
                c, s = math.cos(q[i]), math.sin(q[i])
                motion = np.eye(4)
                motion[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
                transform = transform @ motion
            frames.append(transform.copy())
        wrist = frames[4][:3, 3]
        lateral_values.append(float(np.dot(
            wrist - frames[0][:3, 3], frames[1][:3, 2])))
        axial_errors.append(float(np.linalg.norm(
            frames[6][:3, 3] - wrist - axial_offset * frames[6][:3, 2])))
        radii.append(float(np.linalg.norm(wrist[:2])))
    lateral_error = float(np.max(np.abs(np.asarray(lateral_values) - side_offset)))
    if lateral_error > 1e-10 or max(axial_errors) > 1e-10:
        raise AssertionError("Independent CPU FK did not reproduce the geometric identities")

    mount = parse_rigid_transform_matrix(cfg["robot"]["mount_transform"])
    correction = parse_rigid_transform_matrix(cfg["pick_place"]["link0_target_transform"])
    home = cfg["pick_place"]["home"]
    if home.get("joint_deg") is not None:
        raise ValueError("This audit requires a Cartesian Home, not explicit joint Home")
    raw_home = np.eye(4)
    raw_home[:3, 3] = home["position"]
    raw_home[:3, :3] = rpy_deg_to_matrix(home["rpy_deg"])
    base_home = correction @ raw_home
    world_home = mount @ base_home
    vertical_home = close(base_home[:2, 2], 0)
    vertical_mount = close(mount[:2, 2], 0)
    radius = float(np.linalg.norm(base_home[:2, 3]))
    required_radius = abs(side_offset)
    gap = required_radius - radius
    sweep = []
    for yaw in (0., 90., 180., 270.):
        for height in (.50, .55, .60, .65):
            candidate = np.eye(4)
            candidate[:3, :3] = rpy_deg_to_matrix([180., 0., yaw])
            candidate[:3, 3] = [*mount[:2, 3], height]
            candidate_home = np.linalg.inv(candidate) @ world_home
            sweep.append({
                "mount_xyz_m": candidate[:3, 3].tolist(), "mount_rpy_deg": [180., 0., yaw],
                "home_in_base_m": candidate_home[:3, 3].tolist(),
                "home_tool_z_in_base": candidate_home[:3, 2].tolist(),
                "home_horizontal_radius_m": float(np.linalg.norm(candidate_home[:2, 3])),
            })
    invariance_error = float(max(abs(row["home_horizontal_radius_m"] - radius) for row in sweep))
    applicable = vertical_home and vertical_mount
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"config": str(config_path), "config_sha256": sha256(config_path),
                   "urdf": str(urdf), "urdf_sha256": sha256(urdf)},
        "scope": {
            "cpu_only": True, "configuration_or_home_modified": False,
            "is_ik_or_trajectory_test": False,
            "warning": "Necessary condition for the exact vertical Home only. It does not prove "
                "IK/trajectory success, characterize toleranced IK acceptance, or imply the "
                "whole grasp region is unreachable. Joint limits and collisions may add constraints.",
        },
        "geometry": {
            "urdf_structure_checks": checks, "J4_lateral_offset_m": side_offset,
            "J6_axial_offset_m": wrist_offset, "TCP_axial_offset_m": tcp_offset,
            "J5_to_TCP_axial_offset_m": axial_offset,
            "derivation": [
                "J2/J3/J4 axes remain parallel and horizontal in base coordinates.",
                "J4 adds a fixed lateral offset along that common axis; J3 and J5 "
                "translations are perpendicular to it.",
                "Thus (p_J5 - p_J1) dot common_axis = J4_lateral_offset, "
                "so the horizontal radius of p_J5 is at least abs(J4_lateral_offset).",
                "J6 translation and fixed TCP translation are collinear with tool +Z: "
                "p_TCP - p_J5 = (J6_axial_offset + TCP_axial_offset) * tool_z.",
                "For exact vertical tool Z, p_J5 and p_TCP have the same XY. "
                "Therefore Home TCP horizontal radius must meet that lateral-offset bound.",
            ],
        },
        "cpu_fk_crosscheck": {
            "samples": sample_count, "rng_seed": seed,
            "sampling": "Uniform within URDF raw joint limits; no IK or collision filtering",
            "lateral_projection_min_max_m": [min(lateral_values), max(lateral_values)],
            "max_lateral_identity_error_m": lateral_error,
            "max_TCP_axial_identity_error_m": max(axial_errors),
            "minimum_sampled_J5_horizontal_radius_m": min(radii),
        },
        "home": {
            "physical_position_in_task_world_m": world_home[:3, 3].tolist(),
            "position_in_base_m": base_home[:3, 3].tolist(),
            "tool_z_in_base": base_home[:3, 2].tolist(),
            "vertical_home": vertical_home, "vertical_mount": vertical_mount,
            "radial_necessary_condition_applicable": applicable,
            "horizontal_radius_m": radius, "minimum_required_radius_m": required_radius,
            "radial_shortfall_m": max(0., gap) if applicable else None,
            "exact_target_violates_radial_necessary_condition": applicable and gap > 1e-10,
        },
        "yaw_height_invariance": {
            "applicable": applicable, "fixed_mount_xy_m": mount[:2, 3].tolist(),
            "description": "With a vertical base axis and fixed physical Home/mount XY, "
                "changing yaw only rotates local XY and changing mount height only changes "
                "local Z. Neither changes Home horizontal radius or removes its radial shortfall.",
            "max_radius_change_m": invariance_error, "samples": sweep,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    report = audit(args.config.resolve(), args.samples, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"[HOME] radius={report['home']['horizontal_radius_m']:.9f} m, "
          f"necessary minimum={report['home']['minimum_required_radius_m']:.9f} m")
    print(f"[FK] {args.samples} CPU samples; report={args.out.resolve()}")


if __name__ == "__main__":
    main()
