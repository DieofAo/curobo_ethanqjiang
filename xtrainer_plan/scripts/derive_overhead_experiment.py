#!/usr/bin/env python3
"""Make a recorded, non-overwriting variant of a mounted experiment config."""
import argparse
import copy
import json
import math
from pathlib import Path


def translate_mount(source, position):
    """Translate an already-mounted base; retain physical task poses and walls."""
    import numpy as np
    from prepare_overhead_config import transform_workspace
    from xtrainer_common import parse_rigid_transform_matrix, rpy_deg_to_matrix

    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("mount position must contain three finite coordinates")
    if source["robot"].get("dual_arm_prefix"):
        raise ValueError("mount translation currently requires a single arm")
    if source["pick_place"]["home"].get("joint_deg") is not None:
        raise ValueError("mount translation requires Cartesian Home to preserve its target")
    old_mount = parse_rigid_transform_matrix(source["robot"]["mount_transform"])
    old_c = parse_rigid_transform_matrix(source["pick_place"]["link0_target_transform"])
    overhead = source["overhead"]
    if not np.allclose(old_mount[:3, :3], rpy_deg_to_matrix(overhead["mount_rpy_deg"]),
                       atol=1e-9, rtol=0):
        raise ValueError("mount rotation differs from overhead.mount_rpy_deg")
    if "original_target_transform" in overhead and not np.allclose(
            old_mount @ old_c, overhead["original_target_transform"], atol=1e-9, rtol=0):
        raise ValueError("mounted correction differs from original task transform")
    new_mount = old_mount.copy()
    new_mount[:3, 3] = position
    new_from_old = np.linalg.inv(new_mount) @ old_mount
    cfg = copy.deepcopy(source)
    cfg["robot"]["mount_transform"] = new_mount.tolist()
    cfg["pick_place"]["link0_target_transform"] = (new_from_old @ old_c).tolist()
    cfg["workspace"] = transform_workspace(source["workspace"], new_from_old)
    cfg["overhead"]["height_above_grasp_m"] = float(
        position[2] - cfg["pick_place"]["grasp_grid"]["z"])
    cfg["overhead"]["translated_from_mount_transform"] = old_mount.tolist()
    # The Cartesian Home and its IK branch seed are deliberately unchanged.
    return cfg


def remount_from_task(source, position, world_y_tilt_deg):
    """Move the base and left-multiply its source rotation about task_world +Y.

    Each candidate must use the same untilted source. The mount translation is
    fixed at position; rotation happens around that installation point.
    Targets and wall geometry are rebuilt from the original task frame.
    """
    import numpy as np
    from prepare_overhead_config import transform_workspace
    from xtrainer_common import (
        matrix_to_rpy_deg, parse_rigid_transform_matrix, rpy_deg_to_matrix,
    )

    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("mount position must contain three finite coordinates")
    if not math.isfinite(world_y_tilt_deg):
        raise ValueError("world Y tilt must be finite")
    if source["robot"].get("dual_arm_prefix"):
        raise ValueError("mount experiment requires a single arm")
    overhead = source["overhead"]
    if "world_y_tilt_deg" in overhead:
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
    task_workspace = overhead["task_workspace"]
    tilt = rpy_deg_to_matrix([0.0, float(world_y_tilt_deg), 0.0])
    mount = np.eye(4)
    mount[:3, :3] = tilt @ baseline_rotation
    mount[:3, 3] = position
    inverse = np.linalg.inv(mount)
    cfg = copy.deepcopy(source)
    cfg["robot"]["mount_transform"] = mount.tolist()
    cfg["pick_place"]["link0_target_transform"] = (inverse @ original_c).tolist()
    linear = cfg["pick_place"]["linear_move"]
    linear["method"] = "waypoints_fk"
    linear["waypoint_step_m"] = 0.0075
    cfg["workspace"] = transform_workspace(task_workspace, inverse)
    cfg["overhead"]["mount_rpy_deg"] = matrix_to_rpy_deg(mount[:3, :3]).tolist()
    cfg["overhead"]["world_y_tilt_deg"] = float(world_y_tilt_deg)
    cfg["overhead"]["height_above_grasp_m"] = float(
        position[2] - cfg["pick_place"]["grasp_grid"]["z"])
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--no-prescreen", action="store_true",
                    help="Skip independent IK prescreen; retain final trajectory checks")
    ap.add_argument("--home-position", type=float, nargs=3)
    ap.add_argument("--mount-position", type=float, nargs=3,
                    help="Base position in original task/world frame; keep targets/walls fixed")
    ap.add_argument("--world-y-tilt-deg", type=float,
                    help="Left-multiply source mount by rotation about task_world +Y, "
                         "keeping the mount position fixed")
    ap.add_argument("--joint-limit-clip", type=float,
                    help="Inward margin from each raw URDF joint bound, in radians (replaces prior clip)")
    ap.add_argument("--home-seed", type=float, nargs=6)
    ap.add_argument("--place-x", type=float)
    ap.add_argument("--place-y", type=float,
                    help="Place Y in the original task frame; preserve place height and orientation")
    ap.add_argument("--grasp-angle-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                    help="Grasp candidate bounds in degrees; preserve existing place coupling")
    ap.add_argument("--grasp-angle-step", type=float)
    ap.add_argument("--search-order", choices=("abs", "asc", "desc"))
    ap.add_argument("--no-angle-reuse", action="store_true",
                    help="Start every case in the configured order, not at the last successful angle")
    ap.add_argument("--x-range", type=float, nargs=2)
    ap.add_argument("--y-range", type=float, nargs=2)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--run-output-dir", help="Planner output directory (separate from this config)")
    ap.add_argument("--timestamp-runs", action="store_true")
    args = ap.parse_args()
    source = Path(args.source).resolve()
    cfg = copy.deepcopy(json.loads(source.read_text()))
    if "overhead" not in cfg:
        raise ValueError("Expected a mounted experiment config")
    if args.world_y_tilt_deg is not None:
        if args.mount_position is None:
            ap.error("--world-y-tilt-deg requires --mount-position")
        cfg = remount_from_task(cfg, args.mount_position, args.world_y_tilt_deg)
    elif args.mount_position is not None:
        cfg = translate_mount(cfg, args.mount_position)
    pp = cfg["pick_place"]
    if args.joint_limit_clip is not None:
        if not math.isfinite(args.joint_limit_clip) or args.joint_limit_clip < 0:
            ap.error("--joint-limit-clip must be finite and nonnegative")
        cfg["robot"]["joint_limit_clip"] = args.joint_limit_clip
    if args.no_prescreen:
        pp.setdefault("criterion", {})["prescreen_by_ik"] = False
    if args.no_cuda_graph:
        cfg["planner"]["use_cuda_graph"] = False
    if args.home_position is not None:
        pp["home"]["position"] = args.home_position
        pp["home"]["ik_seed_joint_deg"] = None
        pp["home"]["joint_deg"] = None
    if args.home_seed is not None:
        pp["home"]["ik_seed_joint_deg"] = args.home_seed
    for axis, value in enumerate((args.place_x, args.place_y)):
        if value is not None:
            if not math.isfinite(value):
                ap.error("--place-x and --place-y must be finite")
            pp["place"]["position"][axis] = value
    if args.grasp_angle_range is not None:
        lo, hi = args.grasp_angle_range
        if not all(math.isfinite(v) for v in (lo, hi)) or lo > hi:
            ap.error("--grasp-angle-range must have finite MIN <= MAX")
        pp["angle_search"]["grasp"].update(min_deg=lo, max_deg=hi)
    if args.grasp_angle_step is not None:
        if not math.isfinite(args.grasp_angle_step) or args.grasp_angle_step <= 0:
            ap.error("--grasp-angle-step must be finite and positive")
        pp["angle_search"]["grasp"]["step_deg"] = args.grasp_angle_step
    if args.search_order is not None:
        pp["angle_search"]["order"] = args.search_order
    if args.no_angle_reuse:
        pp["angle_search"]["reuse_last_success"] = False
    for key in ("x_range", "y_range", "rows", "cols"):
        if getattr(args, key) is not None:
            pp["grasp_grid"][key] = getattr(args, key)
    if args.run_output_dir is not None:
        cfg["output"]["dir"] = args.run_output_dir
    if args.timestamp_runs:
        cfg["output"]["add_timestamp"] = True
    cfg["overhead"]["derived_from"] = str(source)
    cfg["overhead"]["variant_options"] = {k: v for k, v in vars(args).items()
                                             if v is not None and v is not False}
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(out)


if __name__ == "__main__":
    main()
