#!/usr/bin/env python3
"""Publish audited V67 negative-local-Y smoke maps after all 54 runs finish.

This is a report-only entry point. It never starts or resumes planning.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

from run_local_y_mount_sweep import verify_inputs


SCRIPT_DIR = Path(__file__).resolve().parent
DATE_DIR = SCRIPT_DIR.parent / "results_overhead/20260928"
DEFAULT_ROOT = DATE_DIR / "v67_near_zero_x_local_y_negative_smoke"
OUTPUT_NAMES = (
    "final_comparison.json", "final_comparison.csv", "final_comparison.png",
    "final_joint_margin.csv", "final_joint_margin.png",
    "final_observed_posture.json", "final_observed_posture_cases.csv",
    "final_observed_posture_grasps.csv", "final_observed_posture.png",
)
AXES = ((-0.20, -0.10), (0.15, 0.35, 0.45),
        (0.45, 0.55, 0.65), (-30.0, -45.0, -60.0))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_complete_smoke(root):
    manifest_path, status_path = root / "manifest.json", root / "sweep_status.json"
    manifest, status = read_json(manifest_path), read_json(status_path)
    verify_inputs(manifest, manifest_path)
    params = manifest["parameters"]
    expected = list(itertools.product(*AXES))
    if (params.get("prefix") != "v67" or params.get("size") != 3 or
            params.get("tilt_axis") != "original_base_local_y" or
            tuple(params.get("base_x", ())) != AXES[0] or
            tuple(params.get("base_y", ())) != AXES[1] or
            tuple(params.get("base_z", ())) != AXES[2] or
            tuple(params.get("local_y_tilt_deg", ())) != AXES[3] or
            manifest["n_configs"] != 54 or len(manifest["candidates"]) != 54):
        raise ValueError("Expected the original 18 XYZ × three negative-local-Y V67 angles")
    for index, (row, coord) in enumerate(zip(manifest["candidates"], expected)):
        if (row["index"] != index or row["name"] != f"v67_{index:02d}" or
                tuple(row["base_xyz_m"]) != coord[:3] or
                row["local_y_tilt_deg"] != coord[3]):
            raise ValueError(f"V67 candidate coordinates/order changed at index {index}")
    if status["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Smoke checkpoint refers to another manifest")
    names = {row["name"] for row in manifest["candidates"]}
    if set(status["outcomes"]) != names:
        raise ValueError("All 54 smoke outcomes must be recorded before publishing")
    valid = {"completed_verified", "completed_zero_success", "home_failed"}
    abnormal = {name: item["state"] for name, item in status["outcomes"].items()
                if item["state"] not in valid}
    if abnormal:
        raise ValueError(f"Resolve incomplete planning/audit outcomes before publishing: {abnormal}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.smoke_root.resolve()
    manifest = validate_complete_smoke(root)
    existing = [name for name in OUTPUT_NAMES if (root / name).exists()]
    if (root / "best_angle_3d").exists():
        existing.append("best_angle_3d")
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing reports: {existing}")
    commands = [
        [sys.executable, str(SCRIPT_DIR / "summarize_local_y_mount_sweep.py"),
         "--manifest", str(manifest), "--out", str(root / "final_comparison.json"),
         "--csv", str(root / "final_comparison.csv"),
         "--plot", str(root / "final_comparison.png")],
        [sys.executable, str(SCRIPT_DIR / "plot_local_y_sweep_joint_margin.py"),
         "--comparison", str(root / "final_comparison.json"),
         "--out", str(root / "final_joint_margin.png"),
         "--csv", str(root / "final_joint_margin.csv")],
        [sys.executable, str(SCRIPT_DIR / "analyze_grasp_posture.py"),
         "--manifest", str(manifest),
         "--output-prefix", str(root / "final_observed_posture")],
        [sys.executable, str(SCRIPT_DIR / "plot_v67_negative_best_angle_3d.py"),
         "--smoke-root", str(root)],
        [sys.executable, str(SCRIPT_DIR / "render_v67_negative_best_angle_walkthrough.py"),
         "--smoke-root", str(root)],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    print(f"Published V67 smoke maps and interactive 3D plot under {root}")


if __name__ == "__main__":
    main()
