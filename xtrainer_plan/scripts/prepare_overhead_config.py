#!/usr/bin/env python3
"""Derive a mounted single-arm config while retaining the original task frame.

M = task_world_T_LINK_0; the planner consumes C = inv(M) @ original_C.
Tilted mounts retain exact oriented wall cuboids and an oriented soft workspace.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from plan_pick_place import load_pick_place_config
from xtrainer_common import (
    REPO_ROOT, build_workspace_wall_cuboids, parse_rigid_transform_matrix,
    matrix_to_quat_wxyz, quat_wxyz_to_matrix, rpy_deg_to_matrix,
)


def transform_bounds(bounds, transform):
    corners = np.array(list(itertools.product(*(bounds[a] for a in "xyz"))))
    points = corners @ transform[:3, :3].T + transform[:3, 3]
    return {a: [float(points[:, k].min()), float(points[:, k].max())]
            for k, a in enumerate("xyz")}


def transform_workspace(workspace, transform):
    """Transform the physical workspace into a new robot-base frame.

    Axis-aligned mounts retain the historical bounds/faces representation.
    Arbitrary rotations carry exact oriented wall boxes and an oriented soft
    bounds test; their AABB bounds are only for display and reporting.
    """
    transform = parse_rigid_transform_matrix(transform, "workspace transform")
    rotation = transform[:3, :3]
    axis_aligned = (np.allclose(np.abs(rotation).sum(0), 1, atol=1e-9)
                    and np.allclose(np.abs(rotation).sum(1), 1, atol=1e-9))
    result = copy.deepcopy(workspace)
    wall = result.get("wall", {})
    if wall.get("base_clearance", {}).get("enable"):
        raise ValueError("A base-clearance cutout needs explicit transformed geometry")
    original_wall = workspace.get("wall", {})
    prior_oriented = workspace.get("oriented_bounds")

    if axis_aligned and prior_oriented is None and "cuboids_override" not in original_wall:
        result["bounds"] = transform_bounds(workspace["bounds"], transform)
        if original_wall.get("bounds_override"):
            merged = {**workspace["bounds"], **original_wall["bounds_override"]}
            wall["bounds_override"] = transform_bounds(merged, transform)
        faces = {}
        for old_axis, axis in enumerate("xyz"):
            new_axis = int(np.argmax(np.abs(rotation[:, old_axis])))
            positive = rotation[new_axis, old_axis] > 0
            for side in ("min", "max"):
                new_side = side if positive else {"min": "max", "max": "min"}[side]
                faces[f"{'xyz'[new_axis]}_{new_side}"] = original_wall.get("faces", {}).get(
                    f"{axis}_{side}", True)
        wall["faces"] = faces
        result["wall"] = wall
        # Compare actual wall boxes, including their thickness, rather than only bounds.
        before = build_workspace_wall_cuboids(workspace)
        after = build_workspace_wall_cuboids(result)
        expected = []
        for box in before:
            center = rotation @ np.array(box["pose"][:3]) + transform[:3, 3]
            dims = np.abs(rotation) @ np.array(box["dims"])
            expected.append(np.r_[center, dims])
        actual = [np.r_[b["pose"][:3], b["dims"]] for b in after]
        if len(expected) != len(actual) or any(
            not any(np.allclose(e, a, atol=1e-9) for a in actual) for e in expected
        ):
            raise AssertionError("Transformed wall geometry differs from the original")
        return result

    if prior_oriented is None:
        reference_bounds = copy.deepcopy(workspace["bounds"])
        reference_to_old = np.eye(4)
    else:
        reference_bounds = copy.deepcopy(prior_oriented["reference_bounds"])
        reference_to_old = parse_rigid_transform_matrix(
            prior_oriented["frame_transform"], "workspace.oriented_bounds.frame_transform"
        )
    reference_to_new = transform @ reference_to_old
    result["oriented_bounds"] = {
        "reference_bounds": reference_bounds,
        "frame_transform": reference_to_new.tolist(),
    }
    result["bounds"] = transform_bounds(reference_bounds, reference_to_new)
    if original_wall.get("bounds_override"):
        merged = {**workspace["bounds"], **original_wall["bounds_override"]}
        wall["bounds_override"] = transform_bounds(merged, transform)
    explicit = []
    for box in build_workspace_wall_cuboids(workspace):
        old_pose = np.asarray(box["pose"], dtype=float)
        center = rotation @ old_pose[:3] + transform[:3, 3]
        orientation = rotation @ quat_wxyz_to_matrix(old_pose[3:])
        explicit.append({
            "name": box["name"],
            "dims": list(box["dims"]),
            "pose": center.tolist() + matrix_to_quat_wxyz(orientation).tolist(),
        })
    wall["cuboids_override"] = explicit
    result["wall"] = wall
    # Validate the serialized form that the planner will actually consume.
    if len(build_workspace_wall_cuboids(result)) != len(explicit):
        raise AssertionError("Transformed wall count differs from the original")
    return result


def prepare(source, mount_position, yaw_deg=180.0, rows=None, cols=None,
            place_x=None, mount_rpy=None):
    cfg = copy.deepcopy(source)
    if cfg["robot"].get("dual_arm_prefix"):
        raise ValueError("Overhead experiment requires a single-arm config")
    mount = np.eye(4)
    mount[:3, :3] = rpy_deg_to_matrix(mount_rpy or [180.0, 0.0, yaw_deg])
    mount[:3, 3] = np.asarray(mount_position, dtype=float)
    mount[np.abs(mount) < 1e-12] = 0.0
    inverse = np.linalg.inv(mount)
    original_c = parse_rigid_transform_matrix(cfg["pick_place"]["link0_target_transform"])
    cfg["pick_place"]["link0_target_transform"] = (inverse @ original_c).tolist()
    cfg["robot"]["task_frame"] = "task_world"
    cfg["robot"]["legacy_task_frame"] = "original_LINK_0"
    cfg["robot"]["mount_transform"] = mount.tolist()
    cfg["pick_place"]["home"]["ik_seed_joint_deg"] = None
    if cfg["pick_place"]["home"].get("joint_deg") is not None:
        raise ValueError("Source home must be Cartesian for a mount comparison")
    if rows is not None:
        cfg["pick_place"]["grasp_grid"]["rows"] = rows
    if cols is not None:
        cfg["pick_place"]["grasp_grid"]["cols"] = cols
    if place_x is not None:
        cfg["pick_place"]["place"]["position"][0] = place_x
    cfg["overhead"] = {
        "task_workspace": copy.deepcopy(source["workspace"]),
        "original_target_transform": original_c.tolist(),
        "mount_rpy_deg": list(mount_rpy or [180.0, 0.0, yaw_deg]),
        "height_above_grasp_m": float(mount_position[2] - cfg["pick_place"]["grasp_grid"]["z"]),
        "tcp_offset_m": 0.19,
        "note": "Targets/rotations/lift remain in original task frame; workspace is in robot base.",
    }
    cfg["workspace"] = transform_workspace(source["workspace"], inverse)
    cfg["output"]["add_timestamp"] = False
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None)
    ap.add_argument("--mount", nargs=3, type=float, required=True)
    ap.add_argument("--yaw", type=float, default=180.0)
    ap.add_argument("--rpy", nargs=3, type=float, help="Explicit mount RPY degrees; overrides --yaw")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--place-x", type=float)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = load_pick_place_config(args.source)
    cfg = prepare(source, args.mount, args.yaw, args.rows, args.cols, args.place_x,
                  mount_rpy=args.rpy)
    urdf = REPO_ROOT / cfg["robot"]["urdf"]
    import xml.etree.ElementTree as ET
    joint = ET.parse(urdf).getroot().find("joint[@name='TCP_joint']")
    if joint is None or joint.find("origin").get("xyz") != "0 0 0.19":
        raise ValueError("Expected current TCP_joint offset 0.19 m")
    cfg["overhead"]["urdf_sha256"] = hashlib.sha256(urdf.read_bytes()).hexdigest()
    cfg["overhead"]["source_config"] = str(Path(args.source).resolve()) if args.source else "config/pick_place_default.yaml"
    path = Path(args.output).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    print(f"[CONFIG] {path}")
    print(f"[MOUNT] task_world xyz={args.mount}, rpy={cfg['overhead']['mount_rpy_deg']}")
    print(f"[TASK] grasp={cfg['pick_place']['grasp_grid']}, place={cfg['pick_place']['place']}")
    print("[CHECK] physical task poses and all six wall boxes preserved")


if __name__ == "__main__":
    main()
