#!/usr/bin/env python3
"""Read-only saved-sample/log summary; exclusively writes a separate JSON report."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def summarize(result):
    meta_path = result / "trajectory_meta.json"
    npz_path = result / "trajectory.npz"
    log_path = result / "plan.log"
    meta = json.loads(meta_path.read_text())
    if meta.get("partial"):
        raise ValueError("A completed trajectory is required")
    with np.load(npz_path, allow_pickle=False) as archive:
        positions = np.degrees(archive["positions"])
        times = archive["times"].copy()
        joint_names = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
                       for v in archive["joint_names"]]
    succeeded = [item for item in meta["items"] if item["success"]]
    skipped = [item for item in meta["items"] if not item["success"]]
    if len(succeeded) != meta["n_items_success"] or len(meta["items"]) != meta["n_items_total"]:
        raise ValueError("Incomplete/inconsistent item metadata")
    checked_joints = [int(j) - 1 for j in meta["criterion"]["joints"]]
    cursor, segments = 0, []
    for item in succeeded:
        start = cursor
        for j, segment in enumerate(item["segments"]):
            # Within each round the first segment retains all samples; later
            # segments share their first point with the previous endpoint.
            lo = cursor if j == 0 else cursor - 1
            hi = lo + segment["n_points"]
            samples = positions[lo:hi]
            if len(samples) != segment["n_points"]:
                raise ValueError("Segment metadata exceeds saved sample array")
            span = np.ptp(samples, axis=0)
            row = {
                "item_index": item["index"], "segment": segment["to"],
                "sample_range_half_open": [lo, hi],
                "span_per_joint_deg": span.tolist(),
                "max_checked_joint_span_deg": float(span[checked_joints].max()),
                "J6_span_deg": float(span[5]),
            }
            if "linear" in segment:
                row["linear"] = segment["linear"]
            segments.append(row)
            cursor = hi
        if cursor - start != item["n_points"]:
            raise ValueError(f"Inconsistent item sample count: {item['index']}")
    if cursor != len(positions) or cursor != meta["n_points"]:
        raise ValueError("Inconsistent total sample count")
    delta = np.abs(np.diff(positions, axis=0))
    flat = int(np.argmax(delta))
    frame, joint = np.unravel_index(flat, delta.shape)
    near = [item for item in succeeded if item["min_limit_margin_deg"] < 1.]

    failures = defaultdict(list)
    pattern = re.compile(r"规划失败: (?P<status>[^\n]+?) @ (?P<segment>i(?P<index>\d+)_\S+)")
    log = log_path.read_text(errors="replace")
    for match in pattern.finditer(log):
        status = match["status"].strip()
        category = status.split()[0]
        if category.startswith("MotionGenStatus."):
            category = category[len("MotionGenStatus."):]
        failures[int(match["index"])].append({
            "category": category, "status": status, "segment": match["segment"]})
    last_categories, all_skipped_attempts, skipped_case_category_counts = Counter(), Counter(), Counter()
    classified = []
    for item in skipped:
        attempts = failures[item["index"]]
        expected = int(item["n_angles_tried"])
        if len(attempts) != expected:
            raise ValueError(f"Log parser captured {len(attempts)}/{expected} failures "
                             f"for skipped item {item['index']}; refusing a partial classification")
        categories = Counter(row["category"] for row in attempts)
        last_categories[attempts[-1]["category"]] += 1
        all_skipped_attempts.update(categories)
        skipped_case_category_counts.update(categories.keys())
        classified.append({"index": item["index"], "position_raw": item["position_raw"],
                           "n_failed_attempts": len(attempts),
                           "last_failed_attempt": attempts[-1],
                           "all_failed_attempt_categories": dict(categories),
                           "all_failed_attempts": attempts})
    all_failures = Counter(row["category"] for rows in failures.values() for row in rows)
    linear = [row for row in segments if "linear" in row]
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"result": str(result),
                   "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (meta_path, npz_path, log_path)}},
        "scope": "Read-only analysis of saved samples and log attempts. Not GPU revalidation "
            "or a certificate of continuous collision/dynamic/physical safety.",
        "summary": {"n_items_total": meta["n_items_total"],
                    "n_items_success": len(succeeded), "n_items_failed": len(skipped),
                    "n_samples": cursor, "duration_s": meta["total_duration_s"],
                    "n_success_cases_with_margin_below_1deg": len(near),
                    "min_limit_margin_deg": min(item["min_limit_margin_deg"] for item in succeeded)},
        "joint_metrics": {
            "joint_names": joint_names, "criterion": meta["criterion"],
            "max_checked_joint_segment": max(segments, key=lambda row: row["max_checked_joint_span_deg"]),
            "max_J6_segment": max(segments, key=lambda row: row["J6_span_deg"]),
            "whole_trajectory_span_per_joint_deg": np.ptp(positions, axis=0).tolist(),
            "max_adjacent_sample_delta_per_joint_deg": delta.max(axis=0).tolist(),
            "max_adjacent_sample_delta": {
                "degrees": float(delta[frame, joint]), "joint": joint_names[joint],
                "from_sample": int(frame), "to_sample": int(frame + 1),
                "sample_dt_s": float(times[frame + 1] - times[frame])},
            "recorded_joint_limit_summary": meta["joint_limit_margin"],
            "near_limit_success_cases": [
                {"index": item["index"], "position_raw": item["position_raw"],
                 "angle_grasp_deg": item["angle_grasp_deg"],
                 "min_limit_margin_deg": item["min_limit_margin_deg"]} for item in near],
            "near_limit_note": "Below 1 degree is a warning, not a measured hard-limit violation. "
                "A case can inherit a near-limit starting state from the previous successful case. "
                "Whole-trajectory span is not one segment's span or an instantaneous frame jump.",
        },
        "linear_metrics": {
            "max_lateral_deviation_segment": max(linear, key=lambda row: row["linear"]["lateral_dev_mm"]),
            "max_rotation_deviation_segment": max(linear, key=lambda row: row["linear"]["rotation_dev_deg"])},
        "recorded_checks": {key: meta[key] for key in
                            ("workspace_check", "gripper_extent_check", "self_collision_check")},
        "failure_classification": {
            "note": "Final-attempt categories count one last attempt per skipped case; they are "
                "not the causes of all attempts. Per-case category membership overlaps.",
            "last_attempt_categories_per_skipped_case": dict(last_categories),
            "all_attempt_categories_for_skipped_cases": dict(all_skipped_attempts),
            "skipped_case_count_with_each_category_anywhere": dict(skipped_case_category_counts),
            "all_failed_attempt_categories_including_eventually_successful_cases": dict(all_failures),
            "skipped_cases": classified,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = summarize(args.result.resolve())
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"[OUT] {args.out.resolve()}")


if __name__ == "__main__":
    main()
