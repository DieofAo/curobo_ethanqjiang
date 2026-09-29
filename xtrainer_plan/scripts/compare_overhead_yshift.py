#!/usr/bin/env python3
"""CPU-only comparison of completed overhead runs and their existing audits.

The first result is the baseline. All runs must contain the identical ordered
grasp grid. No FK, solver, planning, or ROS work is performed by this script.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def read_report(result, name):
    # Older full runs saved CPU reports immediately outside the actual result.
    path = result / name
    if not path.exists() and name in ("analysis_summary.json", "link3_grasp_clearance.json"):
        path = result.parent / name
    return json.loads(path.read_text()), path


def check_hashes(reports, hashes):
    """Reject missing/stale report sources, including old parent-directory reports."""
    analysis, link3, verification, audit = reports
    mappings = [analysis["source"]["sha256"], link3["source"]["sha256"], {
        "trajectory_meta.json": verification["source"]["metadata_sha256"],
        "trajectory.npz": verification["source"]["npz_sha256"],
    }, {
        "trajectory_meta.json": audit["source"]["config_source_sha256"],
        "trajectory.npz": audit["source"]["trajectory_sha256"],
    }]
    for mapping in mappings:
        by_name = {Path(key).name: value for key, value in mapping.items()}
        for name, expected in hashes.items():
            if by_name.get(name) != expected:
                raise ValueError(f"Missing or stale source hash for {name}")


def case_map(meta):
    items = meta["items"]
    if meta.get("partial") or len(items) != meta["n_items_total"]:
        raise ValueError("Requires complete item metadata")
    mapping = {}
    for item in items:
        index = item["index"]
        point = np.asarray(item["position_raw"], dtype=float)
        if index in mapping or point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Duplicate case index or invalid original grasp position")
        mapping[index] = item
    if sum(bool(item["success"]) for item in items) != meta["n_items_success"]:
        raise ValueError("Success count differs from item metadata")
    return mapping


def compare_cases(baseline, candidate):
    """Indices and ordering must agree: sequence can change planning branches."""
    if list(baseline) != list(candidate):
        raise ValueError("Cannot compare different grasp case indices or execution orders")
    for index in baseline:
        if not np.allclose(baseline[index]["position_raw"], candidate[index]["position_raw"],
                           atol=1e-10, rtol=0):
            raise ValueError(f"Cannot compare different original grasp points: case {index}")
    baseline_success = {i for i, item in baseline.items() if item["success"]}
    candidate_success = {i for i, item in candidate.items() if item["success"]}
    gained, lost = sorted(candidate_success - baseline_success), sorted(baseline_success - candidate_success)
    return {"identical_ordered_grasp_points": True,
            "n_new_success": len(gained), "new_success_indices": gained,
            "n_lost_success": len(lost), "lost_success_indices": lost,
            "n_common_success": len(baseline_success & candidate_success),
            "common_success_indices": sorted(baseline_success & candidate_success)}


def link3_summary(report, group_name):
    group = report["groups"][group_name]
    intersecting = sorted(row["index"] for row in report["cases"] if row["success"]
                          and row["groups"][group_name]["n_samples_intersecting_finite_grasp_plane"] > 0)
    minimum = group.get("minimum_height_above_grasp_plane_over_rectangle_m")
    min_z = group.get("minimum_lowest_surface_over_rectangle_z_m")
    return {"n_samples": group["n_unique_samples"],
            "n_plane_intersection_samples": group["n_samples_intersecting_finite_grasp_plane"],
            "n_plane_intersection_cases": len(intersecting),
            "plane_intersection_case_indices": intersecting,
            "minimum_height_above_grasp_plane_m": minimum["value_m"] if minimum else None,
            "minimum_surface_over_rectangle_z_m": min_z["value_m"] if min_z else None,
            "minimum_height_details": minimum}


def summarize(result):
    result = Path(result).resolve()
    meta_path, npz_path = result / "trajectory_meta.json", result / "trajectory.npz"
    meta = json.loads(meta_path.read_text())
    cases = case_map(meta)
    names = ("analysis_summary.json", "link3_grasp_clearance.json",
             "independent_verification.json", "joint_limit_clip_audit.json")
    loaded = [read_report(result, name) for name in names]
    analysis, link3, verification, audit = [row[0] for row in loaded]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (meta_path, npz_path)}
    check_hashes([row[0] for row in loaded], hashes)
    if not verification.get("verification_completed") or not audit.get("verification_completed"):
        raise ValueError("Independent audits have not completed")
    for key in ("n_items_total", "n_items_success"):
        if analysis["summary"][key] != meta[key] or verification[key] != meta[key]:
            raise ValueError(f"Inconsistent report {key}")
    reported_cases = {row["index"]: row for row in link3["cases"]}
    if set(reported_cases) != set(cases) or len(reported_cases) != len(link3["cases"]):
        raise ValueError("LINK3 report has inconsistent case indices")
    for index, item in cases.items():
        row = reported_cases[index]
        if row["success"] != item["success"] or not np.allclose(
                row["position_raw"], item["position_raw"], atol=1e-10, rtol=0):
            raise ValueError("LINK3 report does not match saved cases")
    with np.load(npz_path, allow_pickle=False) as archive:
        q = archive["positions"].astype(np.float64)
        joint_names = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
                       for v in archive["joint_names"]]
    if (joint_names != [f"J_{i}" for i in range(1, 7)] or
            q.shape != (meta["n_points"], 6) or len(q) < 2 or not np.isfinite(q).all()):
        raise ValueError("Invalid trajectory samples or joint order")
    if audit["joint_names"] != joint_names or verification["joint_names"] != joint_names:
        raise ValueError("Audit joint order differs from saved trajectory")
    cfg = meta["config"]
    clip = float(cfg["robot"]["joint_limit_clip"])
    if not np.isclose(clip, audit["clip_rad"], atol=1e-12, rtol=0):
        raise ValueError("Config and audited joint clips differ")
    raw = np.asarray([audit["raw_limits"][key] for key in ("lower_rad", "upper_rad")])
    effective = np.asarray([verification["joint_position_limits"][key]
                            for key in ("lower_rad", "upper_rad")])
    if raw.shape != (2, 6) or effective.shape != (2, 6):
        raise ValueError("Unexpected audit joint-limit shape")
    expected = raw + np.asarray([[clip], [-clip]])
    if not np.allclose(effective, expected, atol=1e-6, rtol=0):
        raise ValueError("Actual solver limits differ from raw URDF plus/minus configured clip")
    raw_margin = np.minimum(q[:, 5] - raw[0, 5], raw[1, 5] - q[:, 5])
    effective_margin = np.minimum(q[:, 5] - effective[0, 5], effective[1, 5] - q[:, 5])
    near = effective_margin < np.radians(1.)
    near_cases, cursor = [], 0
    for item in cases.values():
        if not item["success"]:
            continue
        stop = cursor + int(item["n_points"])
        if np.any(near[cursor:stop]):
            near_cases.append(item["index"])
        cursor = stop
    if cursor != len(q):
        raise ValueError("Per-case sample counts do not match saved trajectory")
    delta = np.degrees(np.abs(np.diff(q, axis=0)))
    flat = int(np.argmax(delta))
    frame, joint = np.unravel_index(flat, delta.shape)
    mount = np.asarray(cfg["robot"]["mount_transform"], dtype=float)
    if mount.shape != (4, 4) or not np.isfinite(mount).all():
        raise ValueError("Unexpected mount matrix")
    metrics = analysis["joint_metrics"]
    last_failures = analysis["failure_classification"]["last_attempt_categories_per_skipped_case"]
    if sum(last_failures.values()) != len(cases) - meta["n_items_success"]:
        raise ValueError("Final failure categories do not cover all failed cases")
    record = {
        "result": str(result), "source_sha256": hashes,
        "report_paths": {name: str(row[1]) for name, row in zip(names, loaded)},
        "mount_xyz_original_LINK0_m": mount[:3, 3].tolist(),
        "mount_rpy_deg": cfg["overhead"]["mount_rpy_deg"], "joint_limit_clip_rad": clip,
        "n_success": meta["n_items_success"], "n_total": meta["n_items_total"],
        "n_failed": meta["n_items_total"] - meta["n_items_success"],
        "success_fraction": meta["n_items_success"] / meta["n_items_total"],
        "n_samples": len(q), "duration_s": meta["total_duration_s"],
        "independent_verification_passed": bool(verification["passed"]),
        "joint_limit_clip_audit_passed": bool(audit["passed"]),
        "J6": {
            "minimum_raw_limit_margin_rad": float(raw_margin.min()),
            "minimum_raw_limit_margin_deg": float(np.degrees(raw_margin.min())),
            "minimum_effective_limit_margin_rad": float(effective_margin.min()),
            "minimum_effective_limit_margin_deg": float(np.degrees(effective_margin.min())),
            "n_samples_effective_margin_below_1deg": int(near.sum()),
            "n_cases_effective_margin_below_1deg": len(near_cases),
            "cases_effective_margin_below_1deg": near_cases,
            "max_segment_span_deg": metrics["max_J6_segment"]["J6_span_deg"],
            "max_segment_details": metrics["max_J6_segment"],
            "max_adjacent_sample_delta_deg": float(delta[:, 5].max()),
        },
        "max_adjacent_sample_delta": {"degrees": float(delta[frame, joint]),
                                      "joint": joint_names[joint], "from_sample": int(frame)},
        "LINK3": {group: link3_summary(link3, group)
                  for group in ("all_segments", "place_related")},
        "failure_last_attempt_categories": last_failures,
    }
    return record, cases


def compare(results):
    if not results:
        raise ValueError("At least one result is required")
    records, baseline = [], None
    for result in results:
        record, cases = summarize(result)
        if baseline is None:
            baseline = cases
        record["versus_baseline"] = compare_cases(baseline, cases)
        records.append(record)
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "baseline_result": records[0]["result"], "scope": {
                "cpu_only": True, "all_source_hashes_verified": True,
                "ordered_original_grasp_points_identical": True,
                "J6_near_limit_definition": "Distance to effective solver limits below 1 degree; J6 only.",
                "LINK3_warning": "Saved successful samples only, spherical envelope versus finite zero-thickness grasp plane. "
                    "No object-height model, CAD collision, other links, or inter-sample swept safety. "
                    "Different candidates have different successful-case sets; aggregate minima are not matched-case comparisons.",
                "joint_span_warning": "A segment's full angular range is not an instantaneous sample jump. "
                    "The existing 170-degree criterion checks only its configured joints, not necessarily J6.",
            }, "candidates": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to replace existing report: {args.out}")
    report = compare(args.results)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    for row in report["candidates"]:
        print(f"y={row['mount_xyz_original_LINK0_m'][1]:.3f}: "
              f"{row['n_success']}/{row['n_total']}; "
              f"J6 raw minimum={row['J6']['minimum_raw_limit_margin_deg']:.3f} deg; "
              f"LINK3 place intersecting cases={row['LINK3']['place_related']['n_plane_intersection_cases']}")
    print(f"[OUT] {args.out.resolve()}")


if __name__ == "__main__":
    main()
