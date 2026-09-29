#!/usr/bin/env python3
"""Export complete min/median/max joint-span pick/place cycles for RViz.

CPU-only, no planning or ROS. Selection uses max_j(max(q_j)-min(q_j)) over
each complete saved six-segment cycle, not accumulated angular travel and not
the planner's J1--J5 per-segment criterion. Source samples are never modified.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np


PHASES = ("g_lift_in", "grasp", "g_lift_out", "p_lift_in", "place", "p_lift_out")
METRIC = "max_j(peak_to_peak(saved_q_j)) over all six complete case segments; degrees; all J1--J6"
TIE_ATOL_DEG = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_integer(value, label, minimum=1):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"{label} must be an integer >= {minimum}")
    return value


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def load_source(result):
    result = Path(result).resolve()
    require(not (result / "plan_failed.json").exists(), "Source contains plan_failed.json")
    files = {name: (result / name).read_bytes() for name in ("trajectory_meta.json", "trajectory.npz")}
    meta = json.loads(files["trajectory_meta.json"])
    with np.load(io.BytesIO(files["trajectory.npz"]), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    require(meta.get("task_type") == "pick_place_cycle" and not meta.get("partial"),
            "Requires a completed single-arm pick_place_cycle result")
    require(not meta.get("extraction"), "Use the original full result, not an existing excerpt")
    n = positive_integer(meta["n_points"], "n_points", 2)
    names = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in arrays["joint_names"]]
    require(names == [f"J_{i}" for i in range(1, 7)], "Requires ordered J_1 through J_6")
    for key, shape in (("positions", (n, 6)), ("times", (n,)),
                       ("ee_positions", (n, 3)), ("ee_quats_wxyz", (n, 4)),
                       ("velocities", (n, 6)), ("accelerations", (n, 6))):
        if key in ("velocities", "accelerations") and key not in arrays:
            continue
        require(key in arrays and arrays[key].shape == shape, f"Invalid {key} shape; expected {shape}")
    for key, values in arrays.items():
        if np.issubdtype(values.dtype, np.number):
            require(np.isfinite(values).all(), f"Non-finite values in {key}")
    times = arrays["times"]
    require(np.all(np.diff(times) >= 0) and times[-1] > times[0], "Invalid trajectory timestamps")
    require(isinstance(meta.get("config"), dict), "Missing resolved config required by the RViz model builder")
    total = positive_integer(meta["n_items_total"], "n_items_total")
    require(isinstance(meta["items"], list) and len(meta["items"]) == total,
            "Incomplete source item metadata")
    provenance = {"result": str(result), "files": {name: str(result / name) for name in files},
                  "sha256": {name: sha256(blob) for name, blob in files.items()},
                  "n_items_total": total, "n_items_success": meta["n_items_success"],
                  "n_points": n, "duration_s": float(times[-1] - times[0])}
    return meta, arrays, provenance


def rank_cases(meta, arrays):
    """Respect shared within-case endpoints, but retained cross-case endpoints."""
    q, times = arrays["positions"], arrays["times"]
    rows, seen, cursor, previous = [], set(), 0, None
    for item in meta["items"]:
        index = positive_integer(item["index"], "item index", 0)
        require(index not in seen and index < meta["n_items_total"], "Duplicate/out-of-range item index")
        seen.add(index)
        require(type(item["success"]) is bool, f"Case {index}: success must be boolean")
        if not item["success"]:
            require(item.get("n_points", 0) == 0, f"Failed case {index} contributes samples")
            continue
        start = cursor
        count = positive_integer(item["n_points"], f"Case {index}: n_points", 2)
        require(cursor + count <= len(q), f"Case {index}: sample range exceeds trajectory")
        segments = item["segments"]
        require(len(segments) == 6 and [s["to"] for s in segments] == [f"i{index}_{p}" for p in PHASES],
                f"Case {index}: requires all six ordered grasp/place segments")
        segment_rows = []
        for offset, segment in enumerate(segments):
            lo = cursor if offset == 0 else cursor - 1
            hi = lo + positive_integer(segment["n_points"], f"Case {index}: segment n_points")
            require(hi <= start + count, f"Case {index}: segment exceeds item sample range")
            spans = np.degrees(np.ptp(q[lo:hi], axis=0))
            segment_rows.append({"name": segment["to"], "phase": PHASES[offset],
                                 "source_sample_range_half_open": [lo, hi],
                                 "clip_sample_range_half_open": [lo - start, hi - start],
                                 "source_time_range_s": [float(times[lo]), float(times[hi - 1])],
                                 "duration_s": float(times[hi - 1] - times[lo]),
                                 "span_per_joint_deg": spans.tolist(),
                                 "max_joint_span_deg": float(spans.max())})
            cursor = hi
        require(cursor - start == count, f"Case {index}: segment counts disagree with n_points")
        # The first stored point is already the previous successful endpoint;
        # retain it, and do not append or delete another cross-case sample.
        if previous is not None:
            require(np.allclose(q[start], q[start - 1], atol=1e-5, rtol=0),
                    f"Case {index}: start differs from previous successful endpoint")
        positions = q[start:cursor]
        spans = np.degrees(np.ptp(positions, axis=0))
        travel = np.degrees(np.abs(np.diff(positions, axis=0)).sum(axis=0))
        max_segment = max(segment_rows, key=lambda s: s["max_joint_span_deg"])
        rows.append({"index": index, "original_case_number_1based": index + 1,
                     "position_raw": copy.deepcopy(item["position_raw"]),
                     "row": item.get("row"), "col": item.get("col"),
                     "angle_grasp_deg": item["angle_grasp_deg"], "angle_place_deg": item["angle_place_deg"],
                     "n_points": count, "duration_s": float(times[cursor - 1] - times[start]),
                     "source_sample_range_half_open": [start, cursor],
                     "source_time_range_s": [float(times[start]), float(times[cursor - 1])],
                     "previous_successful_case_index": previous,
                     "max_joint_span_deg": float(spans.max()), "span_per_joint_deg": spans.tolist(),
                     "max_span_joint": f"J_{int(np.argmax(spans)) + 1}",
                     "total_variation_per_joint_deg": travel.tolist(),
                     "total_variation_all_joints_deg": float(travel.sum()),
                     "max_segment_joint_span_deg": max_segment["max_joint_span_deg"],
                     "max_segment_name": max_segment["name"], "segments": segment_rows})
        previous = index
    require(cursor == len(q) == meta["n_points"], "Item sample counts do not cover the trajectory")
    require(len(rows) == meta["n_items_success"] and len(rows) > 0, "Success count mismatch/empty result")
    ranked = sorted(rows, key=lambda row: (row["max_joint_span_deg"], row["index"]))
    for rank, row in enumerate(ranked, 1):
        row["rank_1based"] = rank
    median = float(np.median([row["max_joint_span_deg"] for row in ranked]))
    def nearest(target):
        distances = [abs(row["max_joint_span_deg"] - target) for row in ranked]
        minimum = min(distances)
        # Round-off in rad->deg conversion must not choose the upper of two
        # equally distant middle cases instead of the documented index tie.
        tied = [row for row, distance in zip(ranked, distances) if distance <= minimum + TIE_ATOL_DEG]
        return min(tied, key=lambda row: row["index"])

    selected = {"min": nearest(ranked[0]["max_joint_span_deg"]), "median": nearest(median),
                "max": nearest(ranked[-1]["max_joint_span_deg"])}
    return ranked, selected, median


def clip_payload(meta, arrays, row, label, provenance):
    lo, hi = row["source_sample_range_half_open"]
    clipped, array_roles = {}, {}
    for key, values in arrays.items():
        sample_array = key != "joint_names" and values.ndim > 0 and values.shape[0] == len(arrays["positions"])
        clipped[key] = values[lo:hi].copy() if sample_array else values.copy()
        array_roles[key] = "sample_slice" if sample_array else "unchanged_static_array"
    clipped["times"] = clipped["times"] - arrays["times"][lo]
    original_item = next(item for item in meta["items"] if item["index"] == row["index"])
    item = copy.deepcopy(original_item)
    item["original_index"] = row["index"]
    item["duration_s"] = row["duration_s"]
    # Preserve scene/config geometry verbatim but never present full-run checks
    # or failure records as if recomputed for this isolated excerpt.
    retained = ("task_type", "robot", "rotation_convention", "quaternion_order", "grid",
                "place_position", "place_position_raw", "angle_search", "linear_move", "criterion",
                "workspace", "planner", "interpolation_dt", "wall_link_restriction", "motiongen_pose_links",
                "wall_cuboids", "config")
    clip_meta = {key: copy.deepcopy(meta[key]) for key in retained if key in meta}
    poses = [copy.deepcopy(pose) for pose in meta.get("pose_sequence", [])
             if pose.get("item_index") == row["index"]]
    require([pose["name"] for pose in poses] == [s["name"] for s in row["segments"]],
            f"Case {row['index']}: missing/inconsistent six pose markers")
    extraction = {"label": label, "source": copy.deepcopy(provenance), "selection": copy.deepcopy(row),
                  "metric": METRIC, "original_index_preserved": True,
                  "source_home_joint_deg": copy.deepcopy(meta.get("home_joint_deg")),
                  "source_item_duration_s": original_item.get("duration_s"),
                  "array_roles": array_roles, "samples_replanned": False,
                  "note": "Complete original cycle, including transfer from the previous successful endpoint. "
                          "Times rebased only. Source config/grasp grid retained for the identical RViz scene. "
                          "n_items_total retains original index space; only items listed here have clip samples. "
                          "Full-run safety summaries are omitted, not recomputed. "
                          "No motion from the real robot's current state to this clip start was planned."}
    clip_meta.update(created_at=datetime.now(timezone.utc).isoformat(), artifact_type="pick_place_case_excerpt",
                     n_items_total=meta["n_items_total"], n_items_success=1, n_items_done=1, n_items_skipped=0,
                     items=[item], skipped=[], failed=None, pose_sequence=poses, n_points=hi - lo,
                     total_duration_s=float(clipped["times"][-1]),
                     start_joint_deg=np.degrees(clipped["positions"][0]).tolist(), extraction=extraction)
    return clipped, clip_meta


def write_json_exclusive(path, payload):
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def extract(result, out):
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing to replace existing output: {out}")
    meta, arrays, source = load_source(result)
    ranked, selected, median = rank_cases(meta, arrays)
    payloads = {label: clip_payload(meta, arrays, row, label, source) for label, row in selected.items()}
    for name, digest in source["sha256"].items():
        require(sha256(Path(source["files"][name]).read_bytes()) == digest, "Source changed while extracting")
    # Do all validation before creating any outputs. A pre-existing directory,
    # even empty, is deliberately never reused or cleaned up.
    for _, clip_meta in payloads.values():
        json.dumps(clip_meta, allow_nan=False)
    out.mkdir(parents=True, exist_ok=False)
    selections = []
    for label, row in selected.items():
        target = out / label
        target.mkdir()
        clipped, clip_meta = payloads[label]
        with (target / "trajectory.npz").open("xb") as stream:
            np.savez_compressed(stream, **clipped)
        write_json_exclusive(target / "trajectory_meta.json", clip_meta)
        selection = {"label": label, "result": str(target.resolve()), **copy.deepcopy(row),
                     "output_sha256": {name: sha256((target / name).read_bytes())
                                       for name in ("trajectory.npz", "trajectory_meta.json")}}
        selections.append(selection)
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "source": source, "metric": METRIC, "median_metric_deg": median,
                "selection_tie_break": "Lowest original case index for equal extrema or equal distance to median",
                "selection_tie_tolerance_deg": TIE_ATOL_DEG,
                "scope": "Saved successful complete cycles only; all six joints and all six segments. "
                         "No angle wrapping, smoothing, interpolation, re-planning, or new safety certification.",
                "n_ranked_cases": len(ranked), "ranked_cases": ranked, "selected": selections,
                "extractor": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).read_bytes())}}
    write_json_exclusive(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = extract(args.result, args.out)
    for row in manifest["selected"]:
        print(f"{row['label']}: original index={row['index']}, max span={row['max_joint_span_deg']:.6f} deg, "
              f"duration={row['duration_s']:.2f}s, result={row['result']}")
    print(f"[OUT] {args.out.absolute() / 'manifest.json'}")


if __name__ == "__main__":
    main()
