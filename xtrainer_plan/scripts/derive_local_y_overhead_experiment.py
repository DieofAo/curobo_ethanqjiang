#!/usr/bin/env python3
"""Derive one mount by rotating about the UNTILTED base's own local +Y axis.

With the baseline mount rotation R0, the candidate rotation is R0 @ Ry(theta).
The candidate translation stays at --mount-position, so rotation pivots about
that installation point. Target poses and walls remain fixed in task_world.
This is deliberately distinct from derive_overhead_experiment.py's world-Y
rotation Ry(theta) @ R0; the two experiments must retain separate provenance.
"""

import argparse
import copy
import json
import math
from pathlib import Path


def remount_from_task_local_y(source, position, local_y_tilt_deg):
    import numpy as np
    from prepare_overhead_config import transform_workspace
    from xtrainer_common import (
        matrix_to_rpy_deg, parse_rigid_transform_matrix, rpy_deg_to_matrix,
    )

    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("mount position must contain three finite coordinates")
    if not math.isfinite(local_y_tilt_deg):
        raise ValueError("local Y tilt must be finite")
    if source["robot"].get("dual_arm_prefix"):
        raise ValueError("mount experiment requires a single arm")
    overhead = source["overhead"]
    if any(key in overhead for key in ("world_y_tilt_deg", "local_y_tilt_deg", "tilt_axis")):
        raise ValueError("tilted source would compound rotations; use the untilted baseline")
    old_mount = parse_rigid_transform_matrix(source["robot"]["mount_transform"])
    old_c = parse_rigid_transform_matrix(source["pick_place"]["link0_target_transform"])
    baseline_rotation = rpy_deg_to_matrix(overhead["mount_rpy_deg"])
    if not np.allclose(old_mount[:3, :3], baseline_rotation, atol=1e-9, rtol=0):
        raise ValueError("mount rotation differs from overhead.mount_rpy_deg")
    original_c = parse_rigid_transform_matrix(
        overhead["original_target_transform"], "overhead.original_target_transform")
    if not np.allclose(old_mount @ old_c, original_c, atol=1e-9, rtol=0):
        raise ValueError("mounted correction differs from original task transform")
    tilt = rpy_deg_to_matrix([0.0, float(local_y_tilt_deg), 0.0])
    mount = np.eye(4)
    mount[:3, :3] = baseline_rotation @ tilt
    mount[:3, 3] = position
    inverse = np.linalg.inv(mount)
    cfg = copy.deepcopy(source)
    cfg["robot"]["mount_transform"] = mount.tolist()
    cfg["pick_place"]["link0_target_transform"] = (inverse @ original_c).tolist()
    linear = cfg["pick_place"]["linear_move"]
    linear["method"] = "waypoints_fk"
    linear["waypoint_step_m"] = 0.0075
    cfg["workspace"] = transform_workspace(overhead["task_workspace"], inverse)
    cfg["overhead"]["mount_rpy_deg"] = matrix_to_rpy_deg(mount[:3, :3]).tolist()
    cfg["overhead"]["local_y_tilt_deg"] = float(local_y_tilt_deg)
    cfg["overhead"]["tilt_axis"] = "original_base_local_y"
    cfg["overhead"]["height_above_grasp_m"] = float(
        position[2] - cfg["pick_place"]["grasp_grid"]["z"])
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mount-position", type=float, nargs=3, required=True,
                        help="Base origin in original task/world frame, in meters")
    parser.add_argument("--local-y-tilt-deg", type=float, required=True,
                        help="Positive right-hand rotation about original base local +Y")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    args = parser.parse_args()
    for label in ("rows", "cols"):
        value = getattr(args, label)
        if value is not None and value < 2:
            parser.error(f"--{label} must be at least 2")
    source = args.source.resolve()
    cfg = remount_from_task_local_y(json.loads(source.read_text(encoding="utf-8")),
                                    args.mount_position, args.local_y_tilt_deg)
    for label in ("rows", "cols"):
        value = getattr(args, label)
        if value is not None:
            cfg["pick_place"]["grasp_grid"][label] = value
    cfg["output"]["dir"] = str(args.run_output_dir.resolve())
    cfg["overhead"]["derived_from"] = str(source)
    cfg["overhead"]["variant_options"] = {
        "mount_position": args.mount_position,
        "local_y_tilt_deg": args.local_y_tilt_deg,
        "rows": args.rows,
        "cols": args.cols,
        "run_output_dir": str(args.run_output_dir.resolve()),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(cfg, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
