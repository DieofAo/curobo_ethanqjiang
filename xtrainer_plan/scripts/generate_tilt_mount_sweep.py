#!/usr/bin/env python3
"""Record a Cartesian base-translation and world-Y-tilt smoke sweep.

Every candidate is derived independently from one mounted source config.  The
source task poses, place pose, Home, angle search, collision rules, and joint
limits are preserved; only the mount, derived frame transforms, grid density,
and output path change.  This script records configs and a manifest; it does
not run a planner.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_finite(values, label, parser):
    if len(set(values)) != len(values) or not all(math.isfinite(v) for v in values):
        parser.error(f"{label} must contain unique finite values")


def runtime_inputs(repo, source):
    """Freeze the files that determine kinematics, collision and planning."""
    cfg = json.loads(source.read_text(encoding="utf-8"))
    robot_yml = repo / "src/curobo/content/configs/robot" / cfg["robot"]["robot_yml"]
    if "collision_spheres: 'spheres/xtrainer.yml'" not in robot_yml.read_text(encoding="utf-8"):
        raise ValueError("Robot YAML collision-sphere reference changed; update provenance resolver")
    files = {"source_config": source,
             "urdf": repo / cfg["robot"]["urdf"],
             "robot_yml": robot_yml,
             "collision_spheres": repo / "src/curobo/content/configs/robot/spheres/xtrainer.yml",
             "planner": repo / "xtrainer_plan/scripts/plan_pick_place.py",
             "batch_runner": repo / "xtrainer_plan/scripts/run_overhead_batch.py",
             "sweep_runner": repo / "xtrainer_plan/scripts/run_tilt_mount_sweep.py",
             "derive": repo / "xtrainer_plan/scripts/derive_overhead_experiment.py",
             "mount_geometry": repo / "xtrainer_plan/scripts/prepare_overhead_config.py",
             "robot_math": repo / "xtrainer_plan/scripts/xtrainer_common.py",
             "trajectory_planner": repo / "xtrainer_plan/scripts/plan_trajectory.py",
             "trajectory_audit": repo / "xtrainer_plan/scripts/verify_overhead_trajectory.py",
             "joint_limit_audit": repo / "xtrainer_plan/scripts/verify_joint_limit_clip.py",
             "link3_audit": repo / "xtrainer_plan/scripts/analyze_link3_grasp_clearance.py",
             "plan_summary": repo / "xtrainer_plan/scripts/summarize_overhead_plan.py",
             "sweep_summary": repo / "xtrainer_plan/scripts/summarize_tilt_mount_sweep.py",
             "summary_validation": repo / "xtrainer_plan/scripts/summarize_overhead_cartesian.py",
             "report_validation": repo / "xtrainer_plan/scripts/compare_overhead_yshift.py"}
    hashes = {key: {"path": str(path.resolve()), "sha256": sha256(path)}
              for key, path in files.items()}
    if hashes["urdf"]["sha256"] != cfg["overhead"].get("urdf_sha256"):
        raise ValueError("Current URDF hash differs from source experiment's recorded URDF")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--base-x", type=float, nargs="+", required=True)
    parser.add_argument("--base-y", type=float, nargs="+", required=True)
    parser.add_argument("--base-z", type=float, nargs="+", required=True)
    parser.add_argument("--world-y-tilt-deg", type=float, nargs="+", required=True,
                        help="Positive rotation about original LINK_0 / task_world +Y")
    parser.add_argument("--size", type=int, default=3,
                        help="Number of grasp rows and columns (default: 3)")
    args = parser.parse_args()
    for label in ("base_x", "base_y", "base_z", "world_y_tilt_deg"):
        unique_finite(getattr(args, label), label, parser)
    if args.size < 2:
        parser.error("--size must be at least 2")
    if not args.prefix or Path(args.prefix).name != args.prefix or args.prefix in (".", ".."):
        parser.error("--prefix must be one filename component")

    source, root = args.source.resolve(), args.out_root.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    repo = Path(__file__).resolve().parents[2]
    inputs = runtime_inputs(repo, source)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {root}")
    cfg_source = json.loads(source.read_text(encoding="utf-8"))
    if "overhead" not in cfg_source:
        raise ValueError("Source must be a mounted overhead config")
    grid = cfg_source["pick_place"]["grasp_grid"]
    if grid.get("perimeter_only", False):
        raise ValueError("The smoke grid must include all interior points")
    derive = Path(__file__).with_name("derive_overhead_experiment.py")
    root.mkdir(parents=True)
    rows = []
    product = itertools.product(args.base_x, args.base_y, args.base_z, args.world_y_tilt_deg)
    for index, (x, y, z, tilt) in enumerate(product):
        name = f"{args.prefix}_{index:02d}"
        config = root / "configs" / f"{name}.json"
        result = root / "runs" / name
        command = [sys.executable, str(derive), "--source", str(source),
                   "--output", str(config), "--mount-position", str(x), str(y), str(z),
                   "--world-y-tilt-deg", str(tilt), "--rows", str(args.size),
                   "--cols", str(args.size), "--run-output-dir", str(result)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        cfg = json.loads(config.read_text(encoding="utf-8"))
        mount = cfg["robot"]["mount_transform"]
        actual = [mount[i][3] for i in range(3)]
        if any(abs(a - b) > 1e-9 for a, b in zip(actual, (x, y, z))):
            raise ValueError(f"Derived mount position mismatch: {config}")
        linear = cfg["pick_place"]["linear_move"]
        if linear.get("method") != "waypoints_fk" or not math.isclose(
                linear.get("waypoint_step_m", -1), .0075, abs_tol=1e-12):
            raise ValueError(f"Derived linear waypoint method mismatch: {config}")
        if (cfg["pick_place"]["grasp_grid"]["rows"] != args.size or
                cfg["pick_place"]["grasp_grid"]["cols"] != args.size or
                Path(cfg["output"]["dir"]).resolve() != result):
            raise ValueError(f"Derived smoke grid/output mismatch: {config}")
        rows.append({"index": index, "name": name, "config": str(config),
                     "result": str(result), "base_xyz_m": [x, y, z],
                     "world_y_tilt_deg": tilt,
                     "place_xyz_m": cfg["pick_place"]["place"]["position"],
                     "config_sha256": sha256(config), "derive_command": command})
    manifest = {"schema_version": 1, "source": str(source), "source_sha256": sha256(source),
                "runtime_inputs": inputs,
                "parameters": {**vars(args), "source": str(source), "out_root": str(root)},
                "n_configs": len(rows), "candidates": rows}
    with (root / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Generated {len(rows)} candidates: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
