#!/usr/bin/env python3
"""CPU URDF FK analysis of LINK3 spheres relative to the original grasp area.

This measures only saved successful trajectory samples. An XY projection overlap
is NOT a physical collision; the finite grasp plane and heights are reported
separately. Object height, CAD geometry and inter-sample swept volumes are absent.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from audit_overhead_home_geometry import urdf_origin
from xtrainer_common import REPO_ROOT, parse_rigid_transform_matrix, resolve_repo_path


def cpu_fk(q, root):
    """Batched exact URDF transforms for the single six-Z-axis XTrainer chain."""
    n = len(q)
    rotation = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    position = np.zeros((n, 3))
    link3 = None
    for i, name in enumerate([f"J_{j}" for j in range(1, 7)] + ["TCP_joint"]):
        joint = root.find(f"joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"Missing URDF joint {name}")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if parent != f"LINK_{i}" or child != (f"LINK_{i+1}" if i < 6 else "TCP_LINK"):
            raise ValueError(f"Unexpected chain at {name}")
        origin = urdf_origin(joint)
        position += np.einsum("nij,j->ni", rotation, origin[:3, 3])
        rotation = rotation @ origin[:3, :3]
        if i < 6:
            if joint.get("type") != "revolute" or not np.allclose(
                    np.fromstring(joint.find("axis").get("xyz"), sep=" "), [0, 0, 1]):
                raise ValueError(f"Expected revolute Z axis at {name}")
            c, s = np.cos(q[:, i]), np.sin(q[:, i])
            motion = np.zeros((n, 3, 3))
            motion[:, 0, 0] = motion[:, 1, 1] = c
            motion[:, 0, 1] = -s
            motion[:, 1, 0] = s
            motion[:, 2, 2] = 1
            rotation = rotation @ motion
        elif joint.get("type") != "fixed":
            raise ValueError("TCP_joint must remain fixed")
        if child == "LINK_3":
            link3 = (rotation.copy(), position.copy())
    return link3, position


def sphere_rectangle_metrics(centers, radii, x_range, y_range, plane_z):
    """Arrays have shape (N, sphere_count); lengths are metres."""
    x, y, z = centers[..., 0], centers[..., 1], centers[..., 2]
    dx = np.maximum(np.maximum(x_range[0] - x, x - x_range[1]), 0)
    dy = np.maximum(np.maximum(y_range[0] - y, y - y_range[1]), 0)
    outside_distance = np.hypot(dx, dy)
    inside_depth = np.minimum.reduce([x - x_range[0], x_range[1] - x,
                                      y - y_range[0], y_range[1] - y])
    signed_center_distance = np.where(outside_distance > 0, outside_distance, -inside_depth)
    xy_clearance = signed_center_distance - radii
    overlaps_xy = outside_distance <= radii
    lowest_over_rectangle = np.where(
        overlaps_xy, z - np.sqrt(np.maximum(radii**2 - outside_distance**2, 0)), np.nan)
    return {
        "xy_signed_clearance_m": xy_clearance,
        "lowest_surface_z_m": z - radii,
        "lowest_surface_over_rectangle_z_m": lowest_over_rectangle,
        "height_above_grasp_plane_over_rectangle_m": lowest_over_rectangle - plane_z,
        "sphere_to_finite_grasp_plane_clearance_m": np.hypot(outside_distance, z - plane_z) - radii,
        "xy_projection_overlap": overlaps_xy,
    }


def analyze(result):
    meta_path, npz_path = result / "trajectory_meta.json", result / "trajectory.npz"
    meta = json.loads(meta_path.read_text())
    if meta.get("partial"):
        raise ValueError("Requires a completed result")
    cfg = meta["config"]
    if cfg["robot"].get("dual_arm_prefix") or cfg["robot"]["ee_link"] != "TCP_LINK":
        raise ValueError("Expected the single-arm TCP_LINK model")
    with np.load(npz_path, allow_pickle=False) as archive:
        q = archive["positions"].astype(np.float64)
        times = archive["times"].astype(np.float64)
        saved_tcp = archive["ee_positions"].astype(np.float64)
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in archive["joint_names"]]
    if names != [f"J_{i}" for i in range(1, 7)] or q.shape != (meta["n_points"], 6):
        raise ValueError("Unexpected saved joint order or sample count")
    mount = parse_rigid_transform_matrix(cfg["robot"]["mount_transform"])
    correction = parse_rigid_transform_matrix(cfg["pick_place"]["link0_target_transform"])
    if not np.allclose(mount @ correction, np.eye(4), atol=1e-9, rtol=0):
        raise ValueError("This rectangular task_world analysis requires M*C=I")
    urdf = resolve_repo_path(cfg["robot"]["urdf"]).resolve()
    robot_root = REPO_ROOT / "src/curobo/content/configs/robot"
    robot_yaml = robot_root / cfg["robot"]["robot_yml"]
    kin = yaml.safe_load(robot_yaml.read_text())["robot_cfg"]["kinematics"]
    sphere_ref = kin["collision_spheres"]
    spheres_path = robot_root / sphere_ref if isinstance(sphere_ref, str) else None
    sphere_dict = yaml.safe_load(spheres_path.read_text())["collision_spheres"] if spheres_path else sphere_ref
    spheres = sphere_dict["LINK_3"]
    local_centers = np.asarray([sphere["center"] for sphere in spheres], dtype=float)
    # Match load_robot_cfg_dict: a nonzero experiment override wins; zero retains YAML.
    buffer = float(cfg["robot"].get("collision_sphere_buffer") or 0)
    if buffer == 0:
        buffer = kin.get("collision_sphere_buffer", 0.)
        if isinstance(buffer, dict):
            buffer = buffer.get("LINK_3", 0.)
        buffer = float(buffer)
    radii = np.asarray([sphere["radius"] + buffer for sphere in spheres], dtype=float)
    if np.any(radii <= 0):
        raise ValueError("Analysis requires positive LINK3 sphere radii")
    (link_rotation, link_position), tcp = cpu_fk(q, ET.parse(urdf).getroot())
    fk_error = np.linalg.norm(tcp - saved_tcp, axis=1)
    if fk_error.max() > 1e-5:
        raise ValueError(f"CPU URDF FK differs from saved TCP: max {fk_error.max()}m")
    base_centers = np.einsum("nij,sj->nsi", link_rotation, local_centers) + link_position[:, None, :]
    world_centers = base_centers @ mount[:3, :3].T + mount[:3, 3]
    grid = cfg["pick_place"]["grasp_grid"]
    values = sphere_rectangle_metrics(world_centers, radii, grid["x_range"], grid["y_range"], grid["z"])
    groups = {key: [] for key in ("all_segments", "place_related", "place_transfer_p_lift_in",
                                 "place_descent", "place_lift_out", "grasp_related")}
    cursor = 0
    for item in meta["items"]:
        if not item["success"]:
            continue
        start = cursor
        for i, segment in enumerate(item["segments"]):
            lo = cursor if i == 0 else cursor - 1
            hi = lo + segment["n_points"]
            row = {"item_index": item["index"], "segment": segment["to"], "lo": lo, "hi": hi}
            groups["all_segments"].append(row)
            suffixes = {"_p_lift_in": "place_transfer_p_lift_in", "_place": "place_descent",
                        "_p_lift_out": "place_lift_out"}
            matching = [group for suffix, group in suffixes.items() if segment["to"].endswith(suffix)]
            if matching:
                groups["place_related"].append(row)
                groups[matching[0]].append(row)
            else:
                groups["grasp_related"].append(row)
            cursor = hi
        if cursor - start != item["n_points"]:
            raise ValueError("Inconsistent item/segment sample count")
    if cursor != len(q):
        raise ValueError("Segment ranges do not cover all saved samples")
    reports = {}
    for name, ranges in groups.items():
        mask = np.zeros(len(q), dtype=bool)
        for row in ranges:
            mask[row["lo"]:row["hi"]] = True
        indices = np.flatnonzero(mask)
        if not len(indices):
            reports[name] = {"n_unique_samples": 0}
            continue
        overlap_frames = values["xy_projection_overlap"][indices].any(axis=1)
        report = {"n_unique_samples": len(indices), "n_segments": len(ranges),
                  "n_samples_with_any_XY_projection_overlap": int(overlap_frames.sum()),
                  "XY_projection_overlap_fraction": float(overlap_frames.mean()),
                  "n_samples_intersecting_finite_grasp_plane": int(np.any(
                      values["sphere_to_finite_grasp_plane_clearance_m"][indices] <= 0, axis=1).sum())}
        for metric, array in values.items():
            if metric == "xy_projection_overlap":
                continue
            selected = array[indices]
            if np.isnan(selected).all():
                report["minimum_" + metric] = None
                continue
            local_i, sphere_i = np.unravel_index(np.nanargmin(selected), selected.shape)
            sample = int(indices[local_i])
            report["minimum_" + metric] = {
                "value_m": float(array[sample, sphere_i]), "sample_index": sample,
                "time_s": float(times[sample]), "sphere_index": int(sphere_i),
                "sphere_center_world_m": world_centers[sample, sphere_i].tolist(),
                "sphere_radius_m": float(radii[sphere_i]),
                "segments": [{"item_index": row["item_index"], "segment": row["segment"]}
                             for row in ranges if row["lo"] <= sample < row["hi"]]}
        reports[name] = report
    case_reports = []
    for item in meta["items"]:
        if not item["success"]:
            case_reports.append({"index": item["index"], "position_raw": item["position_raw"],
                                 "success": False, "groups": {}})
            continue
        case = {"index": item["index"], "position_raw": item["position_raw"],
                "success": True, "angle_grasp_deg": item["angle_grasp_deg"], "groups": {}}
        for name, ranges in groups.items():
            ranges = [row for row in ranges if row["item_index"] == item["index"]]
            if not ranges:
                continue
            indices = np.unique(np.concatenate([np.arange(row["lo"], row["hi"]) for row in ranges]))
            overlaps = values["xy_projection_overlap"][indices].any(axis=1)
            summary = {"n_unique_samples": len(indices),
                       "n_samples_with_any_XY_projection_overlap": int(overlaps.sum()),
                       "XY_projection_overlap_fraction": float(overlaps.mean()),
                       "n_samples_intersecting_finite_grasp_plane": int(np.any(
                           values["sphere_to_finite_grasp_plane_clearance_m"][indices] <= 0, axis=1).sum())}
            for metric, array in values.items():
                if metric != "xy_projection_overlap":
                    selected = array[indices]
                    summary["minimum_" + metric] = (None if np.isnan(selected).all()
                                                    else float(np.nanmin(selected)))
            case["groups"][name] = summary
        case_reports.append(case)
    paths = [meta_path, npz_path, urdf, robot_yaml] + ([spheres_path] if spheres_path else [])
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"result": str(result), "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                                       for path in paths}},
        "scope": {
            "successful_cases_only": True, "n_success": meta["n_items_success"],
            "n_total": meta["n_items_total"], "n_saved_samples": len(q),
            "cpu_only": True, "no_planning_constraints_modified": True,
            "warning": "Negative XY clearance means projected overlap only, NOT collision with "
                "an infinitely tall forbidden column. The finite grasp-plane distance has no "
                "object-height model. No CAD, other links, inter-sample sweeps, or dynamics are tested. "
                "Candidate comparisons depend on which cases/trajectories succeeded.",
            "place_related_definition": "Union of p_lift_in (transfer), place (descent), "
                "p_lift_out (retreat); separate from grasp segments. Boundary samples may "
                "belong to both adjacent segment groups; counts within each group are unique.",
        },
        "frame": {"name": "original_LINK_0 / task_world", "mount_transform": mount.tolist(),
                  "grasp_x_range": grid["x_range"], "grasp_y_range": grid["y_range"],
                  "grasp_plane_z": grid["z"]},
        "LINK_3_geometry": {"local_centers_m": local_centers.tolist(), "radii_m": radii.tolist(),
                            "collision_sphere_buffer_m": buffer},
        "cpu_fk_crosscheck": {"saved_TCP_max_position_error_m": float(fk_error.max()),
                              "saved_TCP_mean_position_error_m": float(fk_error.mean())},
        "metric_definitions": {
            "xy_signed_clearance_m": "Signed distance of sphere center XY to grasp rectangle minus "
                "radius. Negative means projected overlap, not a 3D collision.",
            "lowest_surface_z_m": "Minimum sphere-center world Z minus radius, irrespective of XY.",
            "lowest_surface_over_rectangle_z_m": "For XY-overlapping spheres, center Z minus "
                "sqrt(radius^2 - unsigned_center_XY_distance_to_rectangle^2): lowest sphere "
                "surface restricted to XY inside the rectangle.",
            "height_above_grasp_plane_over_rectangle_m": "Previous height minus grasp plane Z.",
            "sphere_to_finite_grasp_plane_clearance_m": "3D center distance to finite zero-thickness "
                "grasp-plane rectangle minus radius; <=0 means this sphere intersects that plane rectangle.",
        },
        "groups": reports,
        "cases": case_reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = analyze(args.result.resolve())
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(f"[OUT] {args.out.resolve()}")
    for name, group in report["groups"].items():
        print(name, "samples=", group["n_unique_samples"],
              "XY_overlap=", group.get("n_samples_with_any_XY_projection_overlap"),
              "min_height_over_plane=", group.get("minimum_height_above_grasp_plane_over_rectangle_m"))


if __name__ == "__main__":
    main()
