#!/usr/bin/env python3
"""CPU-only, provenance-checked comparison of three original-base full runs.

Usage: compare_original_base_threeway.py RESULT_ROOT_OR_MANIFEST [--out-dir DIR]
Exclusively creates summary.json, summary.md and case_comparison.csv. No planning,
GPU, ROS, model mutation, or trajectory concatenation is performed.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from verify_joint_limit_clip import read_urdf_limits


JOINTS = [f"J_{i}" for i in range(1, 7)]
PHASES = ["g_lift_in", "grasp", "g_lift_out", "p_lift_in", "place", "p_lift_out"]
THRESHOLDS = (150, 180, 210)
REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_INPUT_CHECKS = {
    "complete_metadata", "sample_count_matches_metadata", "all_numeric_arrays_finite",
    "time_strictly_increasing", "saved_quaternions_unit", "single_arm_six_joint_names",
    "metadata_joint_names_match", "mount_times_correction_identity", "metadata_correction_matches_config",
    "mount_rotation_matches_config", "new_base_z_matches_config", "tcp_0p19_from_link6",
    "urdf_matches_recorded_hash", "self_collision_enabled", "world_collision_enabled_exact_links",
}
GPU_CHECKS = {
    "all_derived_values_finite", "joint_positions_within_limits", "stored_fk_matches_independent_fk",
    "self_collision_free_at_samples", "world_collision_free_at_samples", "all_self_collision_spheres_retained",
}
AUDIT_SCRIPTS = {
    "summarize_overhead_plan.py", "analyze_link3_grasp_clearance.py",
    "verify_overhead_trajectory.py", "verify_joint_limit_clip.py",
}
SCOPE = (
    "Saved discrete samples only. A case covers all six phases, including entry "
    "from the preceding successful case (or Home). Matched means identical grasp "
    "target, NOT identical incoming joint state or history. Joint angles are "
    "bounded revolute coordinates: no modulo-360 wrapping or unwrapping. "
    "PTP=max(q)-min(q); travel=sum(abs(diff(q))) within one complete case, once "
    "per saved interval. Full-case PTP is not an instantaneous jump, not a "
    "single-phase criterion, and not whole-run PTP. No between-sample swept "
    "collision, dynamics, CAD/installation safety or cross-run concatenation "
    "is certified. Failures are planner outcomes, not proofs of geometric "
    "unreachability. Sampled successful points do not certify intervening area."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_true_checks(checks: dict, expected: set[str], label: str):
    require(isinstance(checks, dict) and expected.issubset(checks)
            and all(value is True for value in checks.values()),
            f"{label}: missing or failed required checks")


def path_from(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def unchanged_config(cfg: dict) -> dict:
    """Only output location and the two explicitly controlled factors may differ."""
    value = copy.deepcopy(cfg)
    value.pop("output", None)
    value["pick_place"]["place"].pop("tool_z_rotation_deg", None)
    value["pick_place"]["angle_search"].pop("order", None)
    return value


def position_key(value) -> tuple:
    point = np.asarray(value, dtype=np.float64)
    require(point.shape == (3,) and np.isfinite(point).all(), "Invalid case position")
    return tuple(np.round(point, 10))


def case_metrics(meta: dict, q: np.ndarray, times: np.ndarray,
                 raw: np.ndarray, effective: np.ndarray) -> list[dict]:
    """Rebuild every half-open case/phase range; fail closed on cursor mismatch."""
    require(q.ndim == 2 and q.shape[1] == 6 and len(q) > 0 and np.isfinite(q).all(),
            "Saved joint positions must be finite, nonempty N x 6")
    require(times.shape == (len(q),) and np.isfinite(times).all()
            and np.all(np.diff(times) > 0), "Invalid saved time array")
    require(raw.shape == effective.shape == (2, 6), "Invalid joint-bound shapes")
    cursor, previous, cases = 0, None, []
    for item in meta["items"]:
        index = int(item["index"])
        row = {"index": index, "row": int(item["row"]), "col": int(item["col"]),
               "position_raw": list(map(float, item["position_raw"])),
               "success": bool(item["success"]),
               "n_angles_tried": int(item["n_angles_tried"]),
               "previous_success_index": previous}
        if not row["success"]:
            require(not item.get("n_points"), f"Failed case {index} declares saved samples")
            cases.append(row)
            continue
        start, ranges, phase_ptp = cursor, [], []
        require(len(item["segments"]) == 6, f"Case {index}: expected exactly six phases")
        for phase_index, (phase, segment) in enumerate(zip(PHASES, item["segments"])):
            require(segment["to"] == f"i{index}_{phase}" and segment["index"] == phase_index,
                    f"Case {index}: phase names/order mismatch")
            count = int(segment["n_points"])
            require(count == segment["n_points"] and count > 0,
                    f"Case {index}: invalid phase sample count")
            lo = cursor if phase_index == 0 else cursor - 1
            hi = lo + count
            require(start <= lo < hi <= len(q), f"Case {index}: phase exceeds saved samples")
            phase_span = np.degrees(np.ptp(q[lo:hi], axis=0))
            ranges.append({"phase": phase, "sample_range_half_open": [lo, hi],
                           "ptp_per_joint_deg": phase_span.tolist()})
            phase_ptp.append(phase_span)
            cursor = hi
        require(cursor - start == item["n_points"], f"Case {index}: inconsistent item n_points")
        samples = q[start:cursor]
        if start:
            require(np.max(np.abs(np.degrees(samples[0] - q[start - 1]))) <= .001,
                    f"Case {index}: saved case boundary has an uncounted joint gap over 0.001 deg")
        spans = np.degrees(np.ptp(samples, axis=0))
        travel = np.degrees(np.abs(np.diff(samples, axis=0)).sum(axis=0))
        raw_margin = np.degrees(np.minimum(samples - raw[0], raw[1] - samples).min(axis=0))
        eff_margin = np.degrees(np.minimum(samples - effective[0], effective[1] - samples).min(axis=0))
        phase_max = np.max(phase_ptp, axis=0)
        row.update({
            "angle_grasp_deg": float(item["angle_grasp_deg"]),
            "angle_place_deg": float(item["angle_place_deg"]),
            "n_points": len(samples), "sample_range_half_open": [start, cursor],
            "sampled_duration_s": float(times[cursor - 1] - times[start]),
            "recorded_duration_s": float(item["duration_s"]),
            "incoming_joint_deg": np.degrees(samples[0]).tolist(),
            "outgoing_joint_deg": np.degrees(samples[-1]).tolist(),
            "previous_saved_sample_gap_per_joint_deg": (
                np.degrees(np.abs(samples[0] - q[start - 1])).tolist() if start else None),
            "full_case_ptp_per_joint_deg": spans.tolist(),
            "max_phase_ptp_per_joint_deg": phase_max.tolist(),
            "travel_per_joint_deg": travel.tolist(),
            "raw_margin_per_joint_deg": raw_margin.tolist(),
            "effective_margin_per_joint_deg": eff_margin.tolist(),
            "full_case_max_ptp_deg": float(spans.max()),
            "full_case_J1_J5_max_ptp_deg": float(spans[:5].max()),
            "full_case_J6_ptp_deg": float(spans[5]),
            "max_phase_ptp_deg": float(phase_max.max()),
            "max_phase_J1_J5_ptp_deg": float(phase_max[:5].max()),
            "max_phase_J6_ptp_deg": float(phase_max[5]),
            "total_joint_travel_deg": float(travel.sum()),
            "min_raw_margin_deg": float(raw_margin.min()),
            "min_effective_margin_deg": float(eff_margin.min()),
            "J6_raw_margin_deg": float(raw_margin[5]),
            "J6_effective_margin_deg": float(eff_margin[5]),
            "phases": ranges,
        })
        cases.append(row)
        previous = index
    require(cursor == len(q) == meta["n_points"], "Final cursor/NPZ/metadata sample count mismatch")
    return cases


def check_hash(actual: str, expected: str, label: str):
    require(isinstance(expected, str) and actual == expected, f"Stale/mismatched hash: {label}")


def link3_reference_plane_diagnostics(cpu: dict, meta: dict) -> dict:
    """Summarize the audited CPU geometry without treating its reference plane as CAD."""
    expected = {int(item["index"]): item for item in meta["items"]}
    cases = cpu["cases"]
    require(len(cases) == len(expected) and {int(c["index"]) for c in cases} == set(expected),
            "LINK3 report case coverage differs from metadata")
    for case in cases:
        item = expected[int(case["index"])]
        require(case["success"] == item["success"]
                and position_key(case["position_raw"]) == position_key(item["position_raw"]),
                "LINK3 report case target/status differs from metadata")
    success = [case for case in cases if case["success"]]
    scopes = {}
    for name in ("all_segments", "place_related"):
        group = cpu["groups"][name]
        indices = sorted(int(case["index"]) for case in success
                         if case["groups"][name]["n_samples_intersecting_finite_grasp_plane"] > 0)
        count = int(group["n_samples_intersecting_finite_grasp_plane"])
        require(sum(case["groups"][name]["n_samples_intersecting_finite_grasp_plane"]
                    for case in success) == count, "LINK3 group/case intersection counts disagree")
        require(sum(case["groups"][name]["n_unique_samples"] for case in success)
                == group["n_unique_samples"], "LINK3 group/case sample coverage disagrees")
        minimum = group["minimum_height_above_grasp_plane_over_rectangle_m"]
        value = None if minimum is None else float(minimum["value_m"])
        require(value is None or np.isfinite(value), "Nonfinite LINK3 minimum height")
        scopes[name] = {
            "n_unique_samples": int(group["n_unique_samples"]),
            "n_samples_intersecting_finite_grasp_plane": count,
            "n_success_cases_intersecting_finite_grasp_plane": len(indices),
            "success_case_denominator": len(success),
            "success_case_fraction": len(indices) / len(success) if success else None,
            "case_indices": indices,
            "minimum_height_above_grasp_plane_over_rectangle_m": value,
            "minimum_height_above_grasp_plane_over_rectangle_mm": None if value is None else value * 1000.,
            "minimum_height_record": copy.deepcopy(minimum),
        }
    return {
        "scope": "LINK3 collision spheres versus the finite zero-thickness grasp reference-plane rectangle; saved successful-case samples only",
        "warning": "Plane intersection is a geometric warning, NOT a failure of the configured hard-wall/self-collision audit and NOT a CAD collision result. Actual objects, object height, other links, CAD and between-sample motion are not represented. Passing other audits does not establish complete safety.",
        "group_overlap_note": "place_related is a subset of all_segments; do not sum their sample/case counts",
        "groups": scopes,
    }


def load_candidate(candidate: dict, root: Path, expected_cases: int) -> dict:
    result = path_from(root, candidate["result"])
    config_path = path_from(root, candidate["config"])
    config_hash = sha256(config_path)
    check_hash(config_hash, candidate["config_sha256"], "candidate config")
    config = read_json(config_path)
    require(not (result / "plan_failed.json").exists(), f"Failed/unsafe run: {result}")
    filenames = ("trajectory_meta.json", "trajectory.npz", "run_status.json", "plan.log",
                 "independent_verification.json", "joint_limit_clip_audit.json",
                 "link3_grasp_clearance.json", "analysis_summary.json")
    hashes = {name: sha256(result / name) for name in filenames}
    meta = read_json(result / "trajectory_meta.json")
    status = read_json(result / "run_status.json")
    require(status.get("returncode") == 0 and not meta.get("partial"), f"Incomplete run: {result}")
    require(len(meta["items"]) == meta["n_items_total"] == expected_cases,
            f"{candidate['name']}: expected {expected_cases} fully recorded cases")
    require(sum(bool(i["success"]) for i in meta["items"]) == meta["n_items_success"],
            "Success count differs from metadata")
    require(meta["n_items_skipped"] == expected_cases - meta["n_items_success"],
            "Failed/skipped count differs from metadata")
    require(len({int(i["index"]) for i in meta["items"]}) == expected_cases
            and {int(i["index"]) for i in meta["items"]} == set(range(expected_cases)),
            "Original case indices must be unique and complete")
    loaded_cfg = copy.deepcopy(meta["config"])
    recorded_cfg = copy.deepcopy(config)
    loaded_cfg.pop("output", None)
    recorded_cfg.pop("output", None)
    require(loaded_cfg == recorded_cfg, "Resolved metadata differs from recorded candidate config")
    require(config["robot"].get("dual_arm_prefix") is None, "Expected single arm without mimic")
    require(np.allclose(config["robot"]["mount_transform"], np.eye(4), atol=1e-12, rtol=0)
            and np.allclose(config["pick_place"]["link0_target_transform"], np.eye(4), atol=1e-12, rtol=0),
            "This comparison requires the original LINK_0 with M=C=I")
    require(abs(float(config["robot"]["joint_limit_clip"]) - .14) < 1e-12,
            "Expected unchanged 0.14-rad joint-limit clipping")
    grid = config["pick_place"]["grasp_grid"]
    require(int(grid["rows"]) * int(grid["cols"]) == expected_cases, "Grid dimensions mismatch")
    expected_points = {
        position_key([x, y, grid["z"]])
        for x in np.linspace(*grid["x_range"], int(grid["rows"]))
        for y in np.linspace(*grid["y_range"], int(grid["cols"]))
    }
    require({position_key(i["position_raw"]) for i in meta["items"]} == expected_points,
            "Saved targets are not the complete configured grid")
    require({(int(i["row"]), int(i["col"])) for i in meta["items"]}
            == set(itertools.product(range(int(grid["rows"])), range(int(grid["cols"])))),
            "Incomplete/duplicate case row-column pairs")
    xs = np.linspace(*grid["x_range"], int(grid["rows"]))
    ys = np.linspace(*grid["y_range"], int(grid["cols"]))
    for item in meta["items"]:
        require(np.allclose(item["position_raw"], item["effective_position"], atol=1e-10, rtol=0),
                "Identity-frame raw and effective targets differ")
        require(np.allclose(item["position_raw"],
                            [xs[int(item["row"])], ys[int(item["col"])], grid["z"]],
                            atol=1e-10, rtol=0), "Row/column does not match saved physical target")
    with np.load(result / "trajectory.npz", allow_pickle=False) as archive:
        names = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
                 for v in archive["joint_names"]]
        q = np.asarray(archive["positions"], dtype=np.float64)
        times = np.asarray(archive["times"], dtype=np.float64)
    require(names == JOINTS == meta["robot"]["joint_names"], "Joint names/order mismatch")
    gpu = read_json(result / "independent_verification.json")
    clip = read_json(result / "joint_limit_clip_audit.json")
    cpu = read_json(result / "link3_grasp_clearance.json")
    analysis = read_json(result / "analysis_summary.json")
    for label, audit in (("independent verification", gpu), ("joint-limit clip", clip)):
        require(audit.get("verification_completed") is True and audit.get("passed") is True,
                f"{label}: missing complete passed audit")
    require_true_checks(gpu["input_checks"], GPU_INPUT_CHECKS, "Independent input audit")
    require_true_checks(gpu["gpu_checks"], GPU_CHECKS, "Independent GPU audit")
    for name in ("self_collision", "world_collision"):
        require(gpu[name].get("checked") is True and gpu[name]["n_samples"] == len(q)
                and gpu[name]["n_collision_samples"] == 0,
                f"Incomplete/nonzero {name} audit")
    require(gpu["fk_comparison"]["n_position_mismatch"] == 0
            and gpu["fk_comparison"]["n_rotation_mismatch"] == 0,
            "Independent saved FK mismatch")
    require(gpu["n_samples"] == len(q) and gpu["n_items_total"] == expected_cases
            and gpu["n_items_success"] == meta["n_items_success"],
            "Independent audit sample/case count mismatch")
    require(gpu["joint_names"] == clip["joint_names"] == JOINTS, "Audit joint order differs")
    check_hash(hashes["trajectory_meta.json"], gpu["source"]["metadata_sha256"], "GPU metadata")
    check_hash(hashes["trajectory.npz"], gpu["source"]["npz_sha256"], "GPU NPZ")
    check_hash(hashes["trajectory_meta.json"], clip["source"]["config_source_sha256"], "clip metadata")
    check_hash(hashes["trajectory.npz"], clip["source"]["trajectory_sha256"], "clip NPZ")
    require(Path(gpu["source"]["result"]).resolve() == result
            and Path(clip["source"]["path"]).resolve() == result
            and Path(cpu["source"]["result"]).resolve() == result
            and Path(analysis["source"]["result"]).resolve() == result,
            "Audit source belongs to a different candidate")
    for filename in ("trajectory_meta.json", "trajectory.npz"):
        check_hash(hashes[filename], cpu["source"]["sha256"].get(str(result / filename)),
                   f"CPU FK {filename}")
        check_hash(hashes[filename], analysis["source"]["sha256"].get(filename),
                   f"analysis {filename}")
    check_hash(hashes["plan.log"], analysis["source"]["sha256"].get("plan.log"), "analysis plan.log")
    for source_path, recorded_hash in cpu["source"]["sha256"].items():
        check_hash(sha256(Path(source_path)), recorded_hash, f"CPU FK input {source_path}")
    cpu_error = float(cpu["cpu_fk_crosscheck"]["saved_TCP_max_position_error_m"])
    require(np.isfinite(cpu_error) and 0 <= cpu_error <= 1e-5,
            "CPU FK cross-check missing/nonfinite/exceeds 1e-5 m")
    require(cpu["scope"]["n_saved_samples"] == len(q)
            and cpu["scope"]["n_total"] == expected_cases
            and cpu["scope"]["n_success"] == meta["n_items_success"], "CPU FK coverage mismatch")
    require(clip["trajectory"].get("checked") is True and clip["trajectory"].get("passed") is True
            and clip["trajectory"]["n_samples"] == len(q)
            and clip["trajectory"]["n_violation_samples"] == 0, "Clip audit did not check all saved samples")
    for name in ("independent_ik", "motion_gen"):
        solver = clip[name]
        require(solver.get("passed") is True and solver["joint_names"] == JOINTS
                and solver["kinematics"]["matches_expected"] is True
                and solver["n_rollouts"] == len(solver["rollouts"]) > 0,
                f"Incomplete {name} clipping audit")
        for rollout in solver["rollouts"]:
            require(rollout.get("passed") is True and rollout["constraint_enabled"] is True
                    and rollout["kinematics"]["matches_expected"] is True
                    and rollout["bound_constraint"]["matches_expected"] is True,
                    f"Failed {name} rollout clipping audit")
    require(abs(float(clip["clip_rad"]) - .14) < 1e-12, "Clip audit uses different clipping")
    urdf = Path(clip["urdf"]).resolve()
    urdf_hash = sha256(urdf)
    check_hash(urdf_hash, clip["urdf_sha256"], "clip URDF")
    check_hash(urdf_hash, gpu["source"]["urdf_sha256"], "GPU URDF")
    check_hash(urdf_hash, cpu["source"]["sha256"].get(str(urdf)), "CPU FK URDF")
    robot_dir = REPO_ROOT / "src/curobo/content/configs/robot"
    robot_yaml = path_from(robot_dir, config["robot"]["robot_yml"])
    kin = yaml.safe_load(robot_yaml.read_text(encoding="utf-8"))["robot_cfg"]["kinematics"]
    require(isinstance(kin["collision_spheres"], str), "Expected recorded external sphere YAML")
    sphere_yaml = path_from(robot_dir, kin["collision_spheres"])
    recorded_inputs = read_json(root / "manifest.json")["input_sha256"]
    for asset in (urdf, robot_yaml, sphere_yaml):
        actual_hash = sha256(asset)
        check_hash(actual_hash, cpu["source"]["sha256"].get(str(asset)), f"required CPU FK asset {asset}")
        check_hash(actual_hash, recorded_inputs.get(str(asset)), f"planning-time model asset {asset}")
    raw = read_urdf_limits(urdf, JOINTS)
    audited_raw = np.asarray([clip["raw_limits"]["lower_rad"], clip["raw_limits"]["upper_rad"]])
    effective = np.asarray([clip["expected_limits"]["lower_rad"], clip["expected_limits"]["upper_rad"]])
    expected_effective = raw + np.array([[.14], [-.14]])
    require(np.allclose(raw, audited_raw, atol=1e-12, rtol=0)
            and np.allclose(effective, expected_effective, atol=1e-12, rtol=0),
            "Audited raw/effective limits do not match URDF plus 0.14-rad clip")
    raw_margin = np.minimum(q - raw[0], raw[1] - q).min(axis=0)
    eff_margin = np.minimum(q - effective[0], effective[1] - q).min(axis=0)
    require(np.allclose(raw_margin, clip["trajectory"]["minimum_raw_margin_per_joint_rad"],
                        atol=1e-9, rtol=0), "Recomputed raw margins differ from clip audit")
    require(np.allclose(eff_margin, clip["trajectory"]["minimum_effective_margin_per_joint_rad"],
                        atol=1e-9, rtol=0), "Recomputed effective margins differ from clip audit")
    require(np.all(eff_margin >= -float(clip["tolerance_rad"])), "Effective joint limit violation")
    cases = case_metrics(meta, q, times, raw, effective)
    factor = {"place_tool_z_rotation_deg": config["pick_place"]["place"].get("tool_z_rotation_deg", 0.),
              "search_order": config["pick_place"]["angle_search"]["order"]}
    require(all(candidate[key] == value for key, value in factor.items()),
            "Manifest factors do not match candidate config")
    return {
        "name": candidate["name"], "result": str(result), "config": config,
        "factors": factor, "cases": cases, "joint_names": names,
        "n_samples": len(q), "duration_s": float(meta["total_duration_s"]),
        "plan_elapsed_s": status.get("elapsed_s"),
        "raw_limits_rad": raw.tolist(), "effective_limits_rad": effective.tolist(),
        "audit_checks": {"independent_verification_passed": True, "clip_audit_passed": True,
                         "CPU_FK_crosscheck_passed": True, "CPU_FK_max_position_error_m": cpu_error,
                         "sample_cursor_verified": True, "audit_hashes_match_current_inputs": True},
        "provenance": {"config": str(config_path), "config_sha256": config_hash,
                       "result_file_sha256": hashes, "urdf": str(urdf), "urdf_sha256": urdf_hash},
        "LINK3_reference_plane_diagnostics": link3_reference_plane_diagnostics(cpu, meta),
        "failure_classification": analysis.get("failure_classification", {}),
    }


def distribution(values) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0:
        return {"n": 0, "min": None, "p05": None, "median": None,
                "mean": None, "p95": None, "max": None}
    require(np.isfinite(values).all(), "Nonfinite statistic")
    return {"n": len(values), "min": float(values.min()),
            "p05": float(np.percentile(values, 5)), "median": float(np.median(values)),
            "mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


def population_stats(cases: list[dict]) -> dict:
    require(all(c["success"] for c in cases), "Statistics require successful cases only")
    n = len(cases)
    scalar_keys = (
        "full_case_max_ptp_deg", "full_case_J1_J5_max_ptp_deg", "full_case_J6_ptp_deg",
        "max_phase_ptp_deg", "max_phase_J1_J5_ptp_deg", "max_phase_J6_ptp_deg",
        "total_joint_travel_deg", "min_raw_margin_deg", "min_effective_margin_deg",
        "J6_raw_margin_deg", "J6_effective_margin_deg", "n_angles_tried", "recorded_duration_s",
    )
    metrics = {key: distribution(c[key] for c in cases) for key in scalar_keys}
    per_joint = {}
    for key in ("full_case_ptp_per_joint_deg", "max_phase_ptp_per_joint_deg",
                "travel_per_joint_deg", "raw_margin_per_joint_deg", "effective_margin_per_joint_deg"):
        per_joint[key] = {name: distribution(c[key][i] for c in cases)
                          for i, name in enumerate(JOINTS)}
    thresholds = {}
    for label, key in (("full_case_any_joint", "full_case_max_ptp_deg"),
                       ("full_case_J1_J5", "full_case_J1_J5_max_ptp_deg"),
                       ("full_case_J6", "full_case_J6_ptp_deg"),
                       ("max_phase_any_joint", "max_phase_ptp_deg"),
                       ("max_phase_J1_J5", "max_phase_J1_J5_ptp_deg"),
                       ("max_phase_J6", "max_phase_J6_ptp_deg")):
        thresholds[label] = {}
        for threshold in THRESHOLDS:
            count = sum(c[key] >= threshold for c in cases)
            thresholds[label][str(threshold)] = {
                "count": count, "denominator": n, "fraction": count / n if n else None,
                "indices": [c["index"] for c in cases if c[key] >= threshold],
            }
    near_limits = {}
    for threshold in (1, 5):
        near_limits[f"effective_margin_below_{threshold}deg"] = {}
        for label, key in (("any_joint", "min_effective_margin_deg"),
                           ("J6", "J6_effective_margin_deg")):
            indices = [c["index"] for c in cases if c[key] < threshold]
            near_limits[f"effective_margin_below_{threshold}deg"][label] = {
                "count": len(indices), "denominator": n,
                "fraction": len(indices) / n if n else None, "indices": indices,
            }
    return {"n_successful_cases_in_population": n, "metrics": metrics,
            "per_joint": per_joint, "ptp_greater_or_equal_thresholds_deg": thresholds,
            **near_limits}


def build_comparison(manifest_path: Path, expected_cases: int = 400) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    candidates = manifest["candidates"]
    require(len(candidates) == manifest["n_configs"] == 3, "Exactly three candidates required")
    names = [row["name"] for row in candidates]
    require(len(set(names)) == 3, "Candidate names must be unique")
    status_path = root / "sweep_status.json"
    sweep = read_json(status_path)
    require(Path(sweep["manifest"]).resolve() == manifest_path,
            "Sweep status belongs to a different manifest")
    outcomes = sweep["outcomes"]
    require([row["name"] for row in outcomes] == names, "Sweep outcomes incomplete/out of order")
    for outcome in outcomes:
        require(not outcome.get("error") and outcome.get("batch_returncode") == 0
                and outcome.get("plan_returncode") == 0, "Sweep contains an incomplete/failed run")
        post = {p["script"]: p["returncode"] for p in outcome.get("postprocessing", [])}
        require(AUDIT_SCRIPTS.issubset(post) and all(code == 0 for code in post.values()),
                "Sweep postprocessing/audits incomplete or failed")
    runs = [load_candidate(row, root, expected_cases) for row in candidates]
    baseline = unchanged_config(runs[0]["config"])
    require(all(unchanged_config(run["config"]) == baseline for run in runs),
            "Non-controlled configuration differences found")
    aligned = []
    by_position = [{position_key(c["position_raw"]): c for c in run["cases"]} for run in runs]
    for case in sorted(runs[0]["cases"], key=lambda c: c["index"]):
        key = position_key(case["position_raw"])
        members = [mapping[key] for mapping in by_position]
        require(all((c["index"], c["row"], c["col"]) == (case["index"], case["row"], case["col"])
                    for c in members), "Case index/row/column mapping differs between runs")
        aligned.append({"index": case["index"], "row": case["row"], "col": case["col"],
                        "position_raw": case["position_raw"],
                        "runs": {name: member for name, member in zip(names, members)}})
    common_indices = [c["index"] for c in aligned if all(m["success"] for m in c["runs"].values())]
    success_sets = {run["name"]: {c["index"] for c in run["cases"] if c["success"]} for run in runs}
    union = set.union(*success_sets.values())
    common = set.intersection(*success_sets.values())
    groups = []
    for run in runs:
        success = [c for c in run["cases"] if c["success"]]
        group = {k: v for k, v in run.items() if k not in ("cases", "config", "failure_classification")}
        group.update({
            "n_total": expected_cases, "n_success": len(success), "n_failed": expected_cases - len(success),
            "success_fraction": len(success) / expected_cases,
            "own_success_population": population_stats(success),
            "matched_all_three_success_population": population_stats([c for c in success if c["index"] in common]),
            "failed_cases": [{"index": c["index"], "position_raw": c["position_raw"],
                              "n_angles_tried": c["n_angles_tried"]} for c in run["cases"] if not c["success"]],
            "failure_categories_all_failed_attempts": run["failure_classification"].get(
                "all_attempt_categories_for_skipped_cases", {}),
        })
        groups.append(group)
    overlaps = []
    matched_pairs = []
    matched_metric_keys = ("full_case_max_ptp_deg", "full_case_J6_ptp_deg", "total_joint_travel_deg",
                           "min_effective_margin_deg", "J6_effective_margin_deg", "n_angles_tried")
    for a, b in itertools.combinations(names, 2):
        both = success_sets[a] & success_sets[b]
        overlaps.append({"a": a, "b": b, "both_success": sorted(both),
                         "a_only_success": sorted(success_sets[a] - success_sets[b]),
                         "b_only_success": sorted(success_sets[b] - success_sets[a]),
                         "both_failed": sorted(set(range(expected_cases)) - success_sets[a] - success_sets[b])})
        paired = [row for row in aligned if row["index"] in common]
        incoming = [float(np.max(np.abs(np.asarray(row["runs"][b]["incoming_joint_deg"])
                                        - row["runs"][a]["incoming_joint_deg"]))) for row in paired]
        matched_pairs.append({
            "a": a, "b": b, "denominator_all_three_common_success": len(common),
            "delta_definition": "b minus a on the identical all-three-success target set",
            "incoming_state_max_abs_difference_deg": distribution(incoming),
            "n_incoming_states_equal_within_0p001_deg": sum(v <= .001 for v in incoming),
            "metric_deltas_b_minus_a": {
                key: distribution(row["runs"][b][key] - row["runs"][a][key] for row in paired)
                for key in matched_metric_keys},
        })
    patterns = []
    for bits in itertools.product((False, True), repeat=3):
        indices = [row["index"] for row in aligned
                   if tuple(row["runs"][name]["success"] for name in names) == bits]
        patterns.append({"success_by_group": dict(zip(names, bits)), "count": len(indices), "indices": indices})
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE,
        "metric_definitions": {
            "raw_margin": "min(q - URDF lower, URDF upper - q), minimum over saved case samples, per joint",
            "effective_margin": "same margin after inward clipping each raw bound by 0.14 rad (8.021409 deg)",
            "near_limit_case_fraction": "case minimum effective margin strictly below 1 or 5 degrees; equality is excluded; denominators use successful cases only",
            "full_case_ptp": "peak-to-peak of all six phases together, separately for J1..J6",
            "max_phase_ptp": "largest individual-phase peak-to-peak among six phases, separately per joint",
            "travel": "sum(abs(diff(q))) in degrees over a case, no angular wrapping and no double-counted shared samples",
            "sample_ranges": "zero-based half-open [start, end); segments 1..5 include preceding endpoint; new cases do not subtract one",
            "case_boundaries": "consecutive successful cases must share joint state within 0.001 deg; larger uncounted boundary gaps are rejected",
            "own_denominator": "that run's successful cases, never all 400 cases for motion-quality ratios",
            "matched_denominator": "same targets successful in all three runs; incoming joint configurations can differ",
            "incoming_comparison": "raw bounded-joint absolute difference, not modulo 360, tolerance 0.001 deg",
        },
        "provenance": {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                       "sweep_status": str(status_path), "sweep_status_sha256": sha256(status_path),
                       "comparison_script_sha256": sha256(Path(__file__)),
                       "manifest_input_sha256": manifest.get("input_sha256", {})},
        "controlled_factors": manifest.get("controlled_factors", []),
        "unchanged_settings_verified": True, "n_total_targets": expected_cases,
        "group_order": names, "groups": groups,
        "overlap": {"all_three_success_count": len(common), "all_three_success_indices": common_indices,
                    "any_success_count": len(union), "all_three_failed_indices": sorted(set(range(expected_cases)) - union),
                    "pairwise": overlaps, "success_patterns": patterns},
        "matched_pairwise_quality": matched_pairs,
        "cases": aligned,
    }


def markdown_report(report: dict) -> str:
    lines = ["# 原始 LINK_0 三组全量对比", "",
             "仅分析已保存离散样本；三组均完成规划、独立 FK/碰撞检查、0.14 rad 内缩审计及 CPU FK 交叉核验。", "",
             "同点配对不等于相同入场关节构型：失败跳过和此前解分支会改变下一 case 的起点。", "",
             "| 组 | 成功/总数 | 自身成功 case 全程最大 PTP 中位数/最大值 | 累计六关节行程中位数 | J6 最小 raw/effective 余量 |",
             "|---|---:|---:|---:|---:|"]

    def number(value):
        return "—" if value is None else f"{value:.2f}°"

    for group in report["groups"]:
        m = group["own_success_population"]["metrics"]
        lines.append(f"| {group['name']} | {group['n_success']}/{group['n_total']} | "
                     f"{number(m['full_case_max_ptp_deg']['median'])} / {number(m['full_case_max_ptp_deg']['max'])} | "
                     f"{number(m['total_joint_travel_deg']['median'])} | "
                     f"{number(m['J6_raw_margin_deg']['min'])} / {number(m['J6_effective_margin_deg']['min'])} |")
    lines += ["", f"三组共同成功 **{report['overlap']['all_three_success_count']}** 个目标；"
              f"至少一组成功 **{report['overlap']['any_success_count']}** 个。", "",
              "## 大跨度比例（完整六阶段合起来的任一关节 PTP）", "",
              "分母仅成功 case；不是全部目标。PTP 是角度范围，不是瞬时跳变，也不是单段 170° 判据。", "",
              "| 组 / 统计集合 | 分母 | ≥150° | ≥180° | ≥210° |", "|---|---:|---:|---:|---:|"]
    for group in report["groups"]:
        for label, key in (("自身成功", "own_success_population"),
                           ("三组共同成功", "matched_all_three_success_population")):
            population = group[key]
            values = population["ptp_greater_or_equal_thresholds_deg"]["full_case_any_joint"]
            text = []
            for threshold in THRESHOLDS:
                stat = values[str(threshold)]
                text.append("—" if stat["fraction"] is None else f"{stat['count']} ({stat['fraction']:.1%})")
            lines.append(f"| {group['name']} / {label} | {population['n_successful_cases_in_population']} | "
                         + " | ".join(text) + " |")
    lines += ["", "## 靠近内缩限位的成功 case", "",
              "按整个 case 的最小 effective 余量统计，严格 <1° / <5°，恰好等于阈值不计入；不表示原 URDF 限位越界。", "",
              "| 组 / 统计集合 | 分母 | 任意关节 <1° | J6 <1° | 任意关节 <5° | J6 <5° |",
              "|---|---:|---:|---:|---:|---:|"]
    for group in report["groups"]:
        for label, key in (("自身成功", "own_success_population"),
                           ("三组共同成功", "matched_all_three_success_population")):
            population = group[key]
            values = []
            for threshold, joint in ((1, "any_joint"), (1, "J6"), (5, "any_joint"), (5, "J6")):
                stat = population[f"effective_margin_below_{threshold}deg"][joint]
                values.append("—" if stat["fraction"] is None else f"{stat['count']} ({stat['fraction']:.1%})")
            lines.append(f"| {group['name']} / {label} | {population['n_successful_cases_in_population']} | "
                         + " | ".join(values) + " |")
    lines += ["", "## LINK3 与有限 grasp 参考平面的几何诊断", "",
              "参考平面只有抓取矩形范围、零厚度；它不是实际 CAD/物体，也不是本次硬墙碰撞模型。相交是需要关注的几何现象，不能因其它审计通过而宣称完全安全。", "",
              "place相关包含转运、放置下降和抬升，是全部六阶段的子集；两行不能相加。病例分母为该组自身成功 case。", "",
              "| 组 / 阶段范围 | 相交成功case/成功分母 | 相交采样数 | 区域内球面相对grasp平面的最低高度 |",
              "|---|---:|---:|---:|"]
    for group in report["groups"]:
        diagnostics = group["LINK3_reference_plane_diagnostics"]["groups"]
        for label, key in (("全部六阶段", "all_segments"), ("place相关", "place_related")):
            stat = diagnostics[key]
            height = stat["minimum_height_above_grasp_plane_over_rectangle_mm"]
            height_text = "无XY重叠，未定义" if height is None else f"{height:.3f} mm"
            lines.append(f"| {group['name']} / {label} | "
                         f"{stat['n_success_cases_intersecting_finite_grasp_plane']}/{stat['success_case_denominator']} | "
                         f"{stat['n_samples_intersecting_finite_grasp_plane']} | {height_text} |")
    lines += ["", "## 三组共同成功点的配对质量", "",
              "差值为 B − A；关节行程/跨度越小越少运动，限位余量越大越远离边界。", "",
              "| A → B | 同入场构型/配对数（0.001°） | 最大 PTP 差中位数 | 总行程差中位数 | J6 effective 余量差中位数 |",
              "|---|---:|---:|---:|---:|"]
    for pair in report["matched_pairwise_quality"]:
        d = pair["metric_deltas_b_minus_a"]
        lines.append(f"| {pair['a']} → {pair['b']} | {pair['n_incoming_states_equal_within_0p001_deg']}/"
                     f"{pair['denominator_all_three_common_success']} | {number(d['full_case_max_ptp_deg']['median'])} | "
                     f"{number(d['total_joint_travel_deg']['median'])} | {number(d['J6_effective_margin_deg']['median'])} |")
    lines += ["", "## 失败目标", "", "下列 index 为 0 基；失败不等于几何无解。", ""]
    for group in report["groups"]:
        points = "; ".join(f"{c['index']} ({c['position_raw'][0]:.5f}, {c['position_raw'][1]:.5f})"
                           for c in group["failed_cases"])
        lines.append(f"- {group['name']}：{points or '无'}")
    lines += ["", "## 口径与限制", "",
              "- CSV 每行对应一个原始网格位置，三组状态、角度、尝试数、起点、每关节余量/PTP/累计行程逐列对齐；失败项运动指标留空。",
              "- JSON 含逐 case 六段的严格半开采样范围、每关节详细统计、两两成功/失败交集与所有输入 SHA256。",
              "- raw 余量相对原 URDF；effective 余量相对两侧各内缩 0.14 rad 的界限。",
              "- 阶段共享端点只用于各段范围；完整 case 行程不重复累计。跨 case 首点不删除。",
              "- 未证明采样间连续碰撞安全、速度/加速度/jerk 限制、CAD/真实装配安全或多组轨迹衔接。",
              "- 没有合并轨迹，也不把成功网格点之间的面积认定为已验证工作空间。", ""]
    return "\n".join(lines)


def csv_rows(report: dict) -> tuple[list[str], list[dict]]:
    basic = ["index", "row", "col", "x_m", "y_m", "z_m"]
    scalars = ["success", "angle_grasp_deg", "angle_place_deg", "n_angles_tried",
               "previous_success_index", "n_points", "recorded_duration_s", "full_case_max_ptp_deg",
               "full_case_J1_J5_max_ptp_deg", "full_case_J6_ptp_deg", "max_phase_ptp_deg",
               "total_joint_travel_deg", "min_raw_margin_deg", "min_effective_margin_deg",
               "J6_raw_margin_deg", "J6_effective_margin_deg"]
    vectors = ["incoming_joint_deg", "full_case_ptp_per_joint_deg", "max_phase_ptp_per_joint_deg",
               "travel_per_joint_deg", "raw_margin_per_joint_deg", "effective_margin_per_joint_deg"]
    fields = basic + [f"{name}.{key}" for name in report["group_order"]
                      for key in scalars + [f"{key}.{joint}" for key in vectors for joint in JOINTS]]
    rows = []
    for case in report["cases"]:
        row = {key: case[key] for key in ("index", "row", "col")}
        row.update(dict(zip(("x_m", "y_m", "z_m"), case["position_raw"])))
        for name in report["group_order"]:
            member = case["runs"][name]
            for key in scalars:
                row[f"{name}.{key}"] = member.get(key)
            for key in vectors:
                values = member.get(key, [None] * 6)
                for joint, value in zip(JOINTS, values):
                    row[f"{name}.{key}.{joint}"] = value
        rows.append(row)
    return fields, rows


def write_outputs(report: dict, out: Path):
    paths = [out / name for name in ("summary.json", "summary.md", "case_comparison.csv")]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    markdown = markdown_report(report)
    fields, rows = csv_rows(report)
    out.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x", encoding="utf-8") as stream:
        stream.write(payload)
    with paths[1].open("x", encoding="utf-8") as stream:
        stream.write(markdown)
    with paths[2].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="result root or manifest.json")
    parser.add_argument("--out-dir", type=Path, help="default: manifest parent; existing outputs are refused")
    args = parser.parse_args()
    manifest = args.source / "manifest.json" if args.source.is_dir() else args.source
    report = build_comparison(manifest)
    out = (args.out_dir or manifest.parent).resolve()
    write_outputs(report, out)
    print(json.dumps({"groups": [{"name": g["name"], "success": g["n_success"], "total": g["n_total"]}
                                 for g in report["groups"]],
                      "matched_success": report["overlap"]["all_three_success_count"],
                      "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
