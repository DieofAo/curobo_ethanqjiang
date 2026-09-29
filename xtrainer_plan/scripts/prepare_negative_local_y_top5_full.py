#!/usr/bin/env python3
"""Freeze the five best audited V67 negative-local-Y smoke mounts as 20×20 runs.

The smoke directory must contain all 54 recorded attempts. Selection uses only
independently verified 3×3 results: success count descending, minimum effective
joint-limit margin over all six joints descending, then original candidate index.
No planner is run here. The output manifest can be passed to
run_local_y_mount_sweep.py.
"""

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

from run_local_y_mount_sweep import verify_inputs
from summarize_local_y_mount_sweep import aggregate


BASE = Path(__file__).resolve().parents[1] / "results_overhead/20260928"
DEFAULT_REFERENCE = BASE / "v65_near_zero_x_local_ytilt60_full/configs/v65_00.json"
XYZ_AND_TILTS = {
    "base_x": [-0.20, -0.10],
    "base_y": [0.15, 0.35, 0.45],
    "base_z": [0.45, 0.55, 0.65],
    "local_y_tilt_deg": [-30.0, -45.0, -60.0],
}
ALLOWED_DIFFS = {
    "pick_place.grasp_grid.rows",
    "pick_place.grasp_grid.cols",
    "output.dir",
    "overhead.variant_options.rows",
    "overhead.variant_options.cols",
    "overhead.variant_options.run_output_dir",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def changed_fields(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        return [field for key in sorted(before.keys() | after.keys())
                for field in changed_fields(before.get(key), after.get(key),
                                            f"{prefix}.{key}" if prefix else key)]
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [prefix]
        return [field for index, (left, right) in enumerate(zip(before, after))
                for field in changed_fields(left, right, f"{prefix}[{index}]")]
    return [] if before == after else [prefix]


def check_rotation(config, source, expected_xyz, expected_tilt):
    source_mount = source["robot"]["mount_transform"]
    mount = config["robot"]["mount_transform"]
    require(len(source_mount) == len(mount) == 4 and
            all(len(row) == 4 for row in source_mount + mount),
            "Expected 4×4 mount transforms")
    angle = math.radians(expected_tilt)
    c, s = math.cos(angle), math.sin(angle)
    rotation = ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    for i in range(3):
        require(math.isclose(mount[i][3], expected_xyz[i], rel_tol=0, abs_tol=1e-9),
                "Mount translation differs from manifest XYZ")
        for j in range(3):
            expected = sum(source_mount[i][k] * rotation[k][j] for k in range(3))
            require(math.isclose(mount[i][j], expected, rel_tol=0, abs_tol=1e-9),
                    "Mount rotation differs from R0 @ Ry(negative local-Y tilt)")
    require(all(math.isclose(mount[3][j], (0.0, 0.0, 0.0, 1.0)[j],
                             rel_tol=0, abs_tol=1e-9) for j in range(4)),
            "Invalid homogeneous mount transform")


def audited_effective_margin(smoke_run):
    audit_path = smoke_run / "joint_limit_clip_audit.json"
    audit = read_json(audit_path)
    trace = audit["trajectory"]
    provenance = audit["source"]
    require(audit["passed"] and audit["verification_completed"] and
            trace["checked"] and trace["passed"],
            f"Joint-limit audit has not passed: {smoke_run}")
    require(provenance["trajectory_sha256"] == sha256(smoke_run / "trajectory.npz") and
            provenance["config_source_sha256"] == sha256(smoke_run / "trajectory_meta.json"),
            f"Joint-limit audit input hash mismatch: {smoke_run}")
    values = trace["minimum_effective_margin_per_joint_rad"]
    require(len(values) == len(audit["joint_names"]) == 6 and
            all(math.isfinite(value) and value >= 0 for value in values),
            f"Invalid six-joint effective margins: {smoke_run}")
    minimum = min(values)
    require(math.isclose(minimum, trace["minimum_effective_margin_rad"],
                         rel_tol=0, abs_tol=1e-7),
            f"Reported joint margin disagrees with six joint values: {smoke_run}")
    return minimum, {
        "joint_limit_clip_audit": str(audit_path),
        "joint_limit_clip_audit_sha256": sha256(audit_path),
        "trajectory_sha256": provenance["trajectory_sha256"],
        "trajectory_meta_sha256": provenance["config_source_sha256"],
    }


def select_and_build(smoke_root, out_root, reference_path):
    smoke_manifest_path = smoke_root / "manifest.json"
    smoke_status_path = smoke_root / "sweep_status.json"
    smoke_manifest = read_json(smoke_manifest_path)
    smoke_status = read_json(smoke_status_path)
    smoke_manifest_hash = sha256(smoke_manifest_path)
    smoke_status_hash = sha256(smoke_status_path)
    verify_inputs(smoke_manifest, smoke_manifest_path)
    require(smoke_status["manifest_sha256"] == smoke_manifest_hash,
            "Smoke status belongs to a different manifest")
    parameters = smoke_manifest["parameters"]
    require(parameters.get("prefix") == "v67" and parameters.get("size") == 3 and
            parameters.get("tilt_axis") == "original_base_local_y" and
            all(parameters.get(axis) == values for axis, values in XYZ_AND_TILTS.items()),
            "Expected the original 18 XYZ positions × -30/-45/-60° V67 smoke sweep")
    smoke_rows = smoke_manifest["candidates"]
    expected_coords = list(itertools.product(*XYZ_AND_TILTS.values()))
    require(len(smoke_rows) == smoke_manifest["n_configs"] == len(expected_coords) == 54,
            "Expected exactly 54 recorded smoke candidates")
    for index, (row, coord) in enumerate(zip(smoke_rows, expected_coords)):
        require(row["index"] == index and row["name"] == f"v67_{index:02d}" and
                tuple(row["base_xyz_m"]) == coord[:3] and
                row["local_y_tilt_deg"] == coord[3],
                f"Smoke coordinate or candidate order changed at index {index}")
    require(set(smoke_status["outcomes"]) == {row["name"] for row in smoke_rows},
            "All 54 smoke candidates must have recorded outcomes before selection")

    # Reinspect the actual trajectory and audit files rather than trusting the
    # checkpoint's success count alone. This also verifies shared task cases.
    comparison = aggregate(smoke_manifest_path)
    require(comparison["n_candidates"] == 54 and
            not comparison["missing_cartesian_coordinates"],
            "Smoke sweep is incomplete or not Cartesian")
    report_by_name = {row["name"]: row for row in comparison["candidates"]}
    baseline = read_json(Path(smoke_manifest["source"]))
    reference_full = read_json(reference_path)
    require(reference_full["pick_place"]["grasp_grid"]["rows"] == 20 and
            reference_full["pick_place"]["grasp_grid"]["cols"] == 20,
            "Reference full config is not 20×20")

    ranked = []
    selection_records = []
    for row in smoke_rows:
        name, index = row["name"], row["index"]
        report = report_by_name[name]
        outcome = smoke_status["outcomes"][name]
        require(outcome["state"] == report["status"] and
                outcome["n_success"] == report["n_success"] and
                outcome["n_total"] == report["n_total"],
                f"Smoke checkpoint disagrees with audited result: {name}")
        require(report["status"] in ("completed_verified", "completed_zero_success", "home_failed"),
                f"Smoke candidate has unresolved planning or audit status: {name}: {report['status']}")
        record = {"name": name, "index": index, "base_xyz_m": row["base_xyz_m"],
                  "local_y_tilt_deg": row["local_y_tilt_deg"],
                  "smoke_status": report["status"], "n_success": report["n_success"],
                  "n_total": report["n_total"], "ranking_eligible": report["ranking_eligible"],
                  "smoke_config": row["config"], "smoke_config_sha256": row["config_sha256"]}
        if report["ranking_eligible"]:
            require(report["status"] == "completed_verified" and
                    isinstance(report["n_success"], int) and
                    1 <= report["n_success"] <= report["n_total"] == 9,
                    f"Invalid ranked smoke coverage: {name}")
            margin, sources = audited_effective_margin(Path(row["result"]))
            record.update(minimum_effective_margin_rad=margin,
                          minimum_effective_margin_deg=math.degrees(margin),
                          **sources)
            ranked.append(record)
        selection_records.append(record)
    require(len(ranked) >= 5, "Fewer than five independently verified smoke candidates")
    ranked.sort(key=lambda record: (-record["n_success"],
                                     -record["minimum_effective_margin_rad"], record["index"]))
    for rank, record in enumerate(ranked, 1):
        record["selection_rank"] = rank
    selected = ranked[:5]
    planned = []
    for rank, record in enumerate(selected, 1):
        old_row = smoke_rows[record["index"]]
        old_config_path = Path(old_row["config"])
        smoke = read_json(old_config_path)
        require(sha256(old_config_path) == old_row["config_sha256"],
                f"Smoke config hash changed: {record['name']}")
        require(smoke["overhead"]["tilt_axis"] == "original_base_local_y" and
                smoke["overhead"]["local_y_tilt_deg"] == record["local_y_tilt_deg"] and
                smoke["robot"]["task_frame"] == "task_world",
                f"Unexpected mount metadata: {record['name']}")
        check_rotation(smoke, baseline, record["base_xyz_m"], record["local_y_tilt_deg"])
        require(smoke["pick_place"]["grasp_grid"]["rows"] == 3 and
                smoke["pick_place"]["grasp_grid"]["cols"] == 3,
                f"Source config is not 3×3: {record['name']}")
        name = record["name"]
        config_path = out_root / "configs" / f"{name}.json"
        result_path = out_root / "runs" / name
        full = copy.deepcopy(smoke)
        full["pick_place"]["grasp_grid"].update(rows=20, cols=20)
        full["output"]["dir"] = str(result_path)
        full["overhead"]["variant_options"].update(
            rows=20, cols=20, run_output_dir=str(result_path))
        require(set(changed_fields(smoke, full)) == ALLOWED_DIFFS,
                f"Unexpected full-versus-smoke difference: {name}")
        require(full["planner"] == reference_full["planner"] and
                full["pick_place"]["grasp_grid"] == reference_full["pick_place"]["grasp_grid"] and
                full["pick_place"]["home"] == reference_full["pick_place"]["home"] and
                full["overhead"]["task_workspace"] == reference_full["overhead"]["task_workspace"] and
                full["robot"]["joint_limit_clip"] == reference_full["robot"]["joint_limit_clip"],
                f"Planning protocol differs from V65 full reference: {name}")
        for key in ("place", "base_rpy", "angle_search", "criterion", "lift",
                    "linear_move", "on_fail"):
            require(full["pick_place"][key] == reference_full["pick_place"][key],
                    f"Pick/place protocol differs from V65 full reference: {name}: {key}")
        check_rotation(full, baseline, record["base_xyz_m"], record["local_y_tilt_deg"])
        planned.append((full, {"index": rank - 1, "name": name, "config": str(config_path),
                               "result": str(result_path), "base_xyz_m": record["base_xyz_m"],
                               "local_y_tilt_deg": record["local_y_tilt_deg"],
                               "tilt_axis": "original_base_local_y",
                               "place_xyz_m": full["pick_place"]["place"]["position"],
                               "source_smoke_index": record["index"],
                               "source_smoke_config": str(old_config_path),
                               "source_smoke_config_sha256": old_row["config_sha256"],
                               "source_smoke_result": old_row["result"],
                               "source_smoke_success": f"{record['n_success']}/9",
                               "source_smoke_minimum_effective_margin_deg":
                                   record["minimum_effective_margin_deg"]}))

    require(smoke_manifest_hash == sha256(smoke_manifest_path) and
            smoke_status_hash == sha256(smoke_status_path),
            "Smoke manifest or status changed while selecting the top five")
    runtime_inputs = copy.deepcopy(smoke_manifest["runtime_inputs"])
    runtime_inputs["full_generator"] = {"path": str(Path(__file__).resolve()),
                                        "sha256": sha256(Path(__file__))}
    runtime_inputs["reference_full_config"] = {"path": str(reference_path),
                                                "sha256": sha256(reference_path)}
    manifest = {"schema_version": 1, "source": smoke_manifest["source"],
                "source_sha256": smoke_manifest["source_sha256"],
                "runtime_inputs": runtime_inputs,
                "parameters": {"tilt_axis": "original_base_local_y", "grid_rows": 20,
                               "grid_cols": 20, "selection": "top five audited V67 negative-angle smoke mounts",
                               "selection_order": ["3x3 complete-trajectory success count descending",
                                                   "minimum effective joint-limit margin across J1-J6 "
                                                   "and all saved trajectory samples descending",
                                                   "original V67 candidate index ascending"],
                               "reference_full_config": str(reference_path),
                               "reference_full_config_sha256": sha256(reference_path),
                               "only_full_vs_smoke_differences": sorted(ALLOWED_DIFFS)},
                "smoke_source": {"root": str(smoke_root),
                                 "manifest": str(smoke_manifest_path),
                                 "manifest_sha256": smoke_manifest_hash,
                                 "sweep_status": str(smoke_status_path),
                                 "sweep_status_sha256": smoke_status_hash},
                "selection_ranking": sorted(
                    selection_records,
                    key=lambda record: (record.get("selection_rank") is None,
                                        record.get("selection_rank", record["index"]))),
                "n_configs": 5, "candidates": [row for _, row in planned]}
    return planned, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--reference-full-config", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    smoke_root = args.smoke_root.resolve()
    out_root = args.out_root.resolve()
    reference_path = args.reference_full_config.resolve()
    if out_root.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {out_root}")
    planned, manifest = select_and_build(smoke_root, out_root, reference_path)
    if out_root.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {out_root}")
    for full, row in planned:
        path = Path(row["config"])
        write_new_json(path, full)
        row["config_sha256"] = sha256(path)
    write_new_json(out_root / "manifest.json", manifest)
    print(f"Generated top five full configs: {out_root / 'manifest.json'}")
    for row in manifest["candidates"]:
        print(f"{row['name']}: smoke={row['source_smoke_success']} "
              f"margin={row['source_smoke_minimum_effective_margin_deg']:.3f}° "
              f"base={row['base_xyz_m']} local +Y tilt={row['local_y_tilt_deg']}°")


if __name__ == "__main__":
    main()
