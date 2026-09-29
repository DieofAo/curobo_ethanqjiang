#!/usr/bin/env python3
"""CPU-only sampled workspace bounds; never infer continuous reachability.

Example:
  python3 summarize_overhead_boundary.py run/ik_screen.json --out boundary.json
  python3 summarize_overhead_boundary.py run/ik_screen.json \
      --trajectory-meta plan/trajectory_meta.json --out boundary.json

IK success means that at least one *same coupled angle* has feasible grasp,
grasp-lift, place, and place-lift endpoints. Home is reported separately.
Trajectory success is the normal planner's per-item success, not a fresh audit
of its NPZ. Different scan densities are summarized separately, never merged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


TOL = 1e-8
CAUTION = (
    "Bounds describe successful sampled points only, not a proof of a continuous "
    "reachable region or a global maximum. Missing points are unverified and "
    "cannot belong to an all-success rectangle. Areas use coordinate spans "
    "without half-cell padding."
)


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def grid_axes(grid: Dict[str, Any]) -> Tuple[List[float], List[float], float]:
    def axis(name: str, size_key: str) -> List[float]:
        count = int(grid[size_key])
        if count < 1:
            raise ValueError(f"{size_key} must be positive")
        low, high = map(float, grid[f"{name}_range"])
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(f"nonfinite {name} bounds")
        if count == 1:
            return [(low + high) / 2]
        if abs(high - low) <= TOL:
            raise ValueError(f"{name}: duplicate coordinates in a multi-point axis")
        return sorted(low + (high - low) * i / (count - 1) for i in range(count))

    z = float(grid["z"])
    if not math.isfinite(z):
        raise ValueError("nonfinite grid height")
    return axis("x", "rows"), axis("y", "cols"), z


def validate_axes(xs: Sequence[float], ys: Sequence[float]) -> None:
    for name, values in (("x", xs), ("y", ys)):
        if not values or any(not math.isfinite(v) for v in values):
            raise ValueError(f"{name} axis must be nonempty and finite")
        if len(values) > 1:
            step = values[1] - values[0]
            if step <= TOL or any(
                not math.isclose(values[i] - values[i - 1], step, abs_tol=TOL, rel_tol=0)
                for i in range(1, len(values))
            ):
                raise ValueError(f"{name} coordinates must form an increasing regular grid")


def rectangle_bounds(
    xs: Sequence[float], ys: Sequence[float], r0: int, r1: int, c0: int, c1: int
) -> Dict[str, Any]:
    width = xs[r1] - xs[r0]
    height = ys[c1] - ys[c0]
    faces = {
        "x_min": r0 == 0, "x_max": r1 == len(xs) - 1,
        "y_min": c0 == 0, "y_max": c1 == len(ys) - 1,
    }
    return {
        "x_range_m": [xs[r0], xs[r1]],
        "y_range_m": [ys[c0], ys[c1]],
        "x_span_m": width,
        "y_span_m": height,
        "area_m2": width * height,
        "sample_count": (r1 - r0 + 1) * (c1 - c0 + 1),
        "sample_shape_xy": [r1 - r0 + 1, c1 - c0 + 1],
        "x_index_range_inclusive": [r0, r1],
        "y_index_range_inclusive": [c0, c1],
        "touches_scan_outer_edge": any(faces.values()),
        "touches_scan_faces": faces,
    }


def largest_success_rectangle(
    success: Sequence[Sequence[bool]], xs: Sequence[float], ys: Sequence[float]
) -> Optional[Dict[str, Any]]:
    """O(rows * cols) histogram algorithm, maximizing *physical span* area.

    A one-row/one-column region has zero area, not one padded cell's area.
    Ties prefer more samples, then larger total span, then lower x/y bounds.
    """
    validate_axes(xs, ys)
    if len(success) != len(xs) or any(len(row) != len(ys) for row in success):
        raise ValueError("success matrix shape must match x/y axes")
    heights = [0] * len(ys)
    best: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, int, float, int, int]] = None
    for r, row in enumerate(success):
        heights = [height + 1 if bool(ok) else 0 for height, ok in zip(heights, row)]
        stack: List[Tuple[int, int]] = []
        for c in range(len(ys) + 1):
            height = heights[c] if c < len(ys) else 0
            start = c
            while stack and stack[-1][1] >= height:
                start, previous_height = stack.pop()
                if previous_height == 0:
                    continue
                candidate = rectangle_bounds(xs, ys, r - previous_height + 1, r, start, c - 1)
                key = (
                    round(candidate["area_m2"], 14), candidate["sample_count"],
                    round(candidate["x_span_m"] + candidate["y_span_m"], 14),
                    -candidate["x_index_range_inclusive"][0], -start,
                )
                if best_key is None or key > best_key:
                    best, best_key = candidate, key
            stack.append((start, height))
    return best


def position_index(position: Sequence[float], xs: Sequence[float], ys: Sequence[float], z: float) -> Tuple[int, int]:
    if len(position) != 3 or any(not math.isfinite(float(v)) for v in position):
        raise ValueError(f"invalid task-frame position {position!r}")
    if not math.isclose(float(position[2]), z, abs_tol=TOL, rel_tol=0):
        raise ValueError(f"point height {position[2]} does not match grid z={z}")
    indices = []
    for value, axis in zip(position[:2], (xs, ys)):
        nearest = min(range(len(axis)), key=lambda i: abs(axis[i] - float(value)))
        if not math.isclose(axis[nearest], float(value), abs_tol=TOL, rel_tol=0):
            raise ValueError(f"position {position!r} is outside the declared regular grid")
        indices.append(nearest)
    return indices[0], indices[1]


def summarize_grid(
    records: Sequence[Dict[str, Any]], grid: Dict[str, Any], *, ik_only: bool
) -> Dict[str, Any]:
    xs, ys, z = grid_axes(grid)
    validate_axes(xs, ys)
    known = [[False] * len(ys) for _ in xs]
    success = [[False] * len(ys) for _ in xs]
    successful_indices = []
    for point in records:
        # scan_overhead_ik writes raw coordinates in position; the planner writes
        # them in position_raw. Never silently interpret planner position as raw.
        position = point.get("position") if ik_only else point.get("position_raw")
        if position is None:
            raise ValueError("trajectory item lacks explicit position_raw")
        r, c = position_index(position, xs, ys, z)
        if known[r][c]:
            raise ValueError(f"duplicate sampled coordinate {position!r}")
        known[r][c] = True
        if ik_only:
            key = "grasp_place_endpoint_intersection_angle_indices"
            if key not in point:
                raise ValueError(f"IK point lacks {key}")
            success[r][c] = bool(point[key])
        else:
            if not isinstance(point.get("success"), bool):
                raise ValueError("trajectory item success must be a JSON boolean")
            success[r][c] = point["success"]
        if success[r][c]:
            successful_indices.append((r, c))
    n_known = sum(map(sum, known))
    n_success = sum(map(sum, success))
    bbox = None
    if successful_indices:
        rs, cs = zip(*successful_indices)
        bbox = rectangle_bounds(xs, ys, min(rs), max(rs), min(cs), max(cs))
        bbox["successful_sample_count"] = n_success
        bbox["all_bounding_box_samples_successful"] = n_success == bbox["sample_count"]
        bbox["warning"] = "An outer bounding box may contain failed or unverified points."
    return {
        "evidence": "ik_endpoints_only" if ik_only else "planner_trajectory_item_success",
        "success_rule": (
            "At least one coupled angle has feasible grasp/lift and place/lift endpoints; home is NOT a gate."
            if ik_only else "Normal planner item.success=true; trajectory arrays are not re-audited by this tool."
        ),
        "coordinate_frame": "legacy task LINK_0 / task_world",
        "grid": {
            "x_coordinates_m": xs, "y_coordinates_m": ys, "z_m": z,
            "shape_xy": [len(xs), len(ys)],
            "spacing_m": {"x": None if len(xs) == 1 else xs[1] - xs[0],
                          "y": None if len(ys) == 1 else ys[1] - ys[0]},
            "scan_bounds": rectangle_bounds(xs, ys, 0, len(xs) - 1, 0, len(ys) - 1),
        },
        "counts": {
            "grid_sample_count": len(xs) * len(ys), "recorded_sample_count": n_known,
            "success": n_success, "failed": n_known - n_success,
            "unverified": len(xs) * len(ys) - n_known,
        },
        "largest_all_success_sampled_rectangle": largest_success_rectangle(success, xs, ys),
        "success_outer_bounding_box": bbox,
        "success_touches_scan_outer_edge": bool(bbox and bbox["touches_scan_outer_edge"]),
        "success_grid_xy": success,
        "verified_grid_xy": known,
        "warning": CAUTION,
    }


def compatible_configs(ik_cfg: Dict[str, Any], trajectory_cfg: Dict[str, Any]) -> None:
    """Do not group results from different mount/target/collision definitions.

    Grid extent/density/order, home/seed, and search budget may legitimately
    differ. The reports remain separate because those can affect path success.
    """
    checks = [
        ("robot", key) for key in (
            "robot_yml", "urdf", "base_link", "ee_link", "flange_link", "dual_arm_prefix",
            "mount_transform", "collision_sphere_buffer", "joint_limit_clip",
        )
    ] + [
        ("pick_place", key) for key in (
            "link0_target_transform", "place", "base_rpy", "lift", "linear_move", "criterion",
        )
    ]
    for section, key in checks:
        lhs = (ik_cfg.get(section) or {}).get(key)
        rhs = (trajectory_cfg.get(section) or {}).get(key)
        if lhs != rhs:
            raise ValueError(f"IK / trajectory mismatch in {section}.{key}")
    for key in ("workspace",):
        if ik_cfg.get(key) != trajectory_cfg.get(key):
            raise ValueError(f"IK / trajectory mismatch in {key}")
    for key in ("self_collision_check", "self_collision_opt", "position_threshold", "rotation_threshold"):
        if (ik_cfg.get("planner") or {}).get(key) != (trajectory_cfg.get("planner") or {}).get(key):
            raise ValueError(f"IK / trajectory mismatch in planner.{key}")
    lhs = (ik_cfg.get("overhead") or {}).get("urdf_sha256")
    rhs = (trajectory_cfg.get("overhead") or {}).get("urdf_sha256")
    if lhs is not None and rhs is not None and lhs != rhs:
        raise ValueError("IK / trajectory URDF content hashes differ")


def build_report(ik_path: Path, trajectory_paths: Sequence[Path] = ()) -> Dict[str, Any]:
    ik = read_json(ik_path)
    cfg = ik["config"]["resolved"]
    ik_summary = summarize_grid(ik["points"], cfg["pick_place"]["grasp_grid"], ik_only=True)
    ik_summary["home_feasible"] = bool(ik["home"]["feasible"])
    ik_summary["source"] = str(ik_path.resolve())
    ik_summary["config_source"] = ik["config"].get("path")
    trajectories = []
    for path in trajectory_paths:
        meta = read_json(path)
        compatible_configs(cfg, meta["config"])
        summary = summarize_grid(meta["items"], meta["grid"], ik_only=False)
        summary["source"] = str(path.resolve())
        summary["sequence_caution"] = (
            "Per-item success follows the recorded sequential pick/place run; "
            "each successful item's final state seeds the next item. A failed "
            "point is not proof of no IK or a geometric workspace hole. A "
            "sampled rectangle is extracted from this run, not replanned in isolation."
        )
        summary["declared_n_items_total"] = meta.get("n_items_total")
        summary["planner_reports"] = {
            key: meta.get(key) for key in ("workspace_check", "joint_limit_margin", "joint_motion")
        }
        trajectories.append(summary)
    return {
        "schema_version": 1,
        "warning": CAUTION,
        "area_definition": "(x_max_sample - x_min_sample) * (y_max_sample - y_min_sample), m^2",
        "tie_break": "maximum area, then most samples, then span sum, then lowest x/y origin",
        "ik": ik_summary,
        "trajectories": trajectories,
        "cross_report_rule": "Each grid is summarized independently. IK is not trajectory validation; no interpolation or cross-grid union is performed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ik_screen", type=Path)
    parser.add_argument("--trajectory-meta", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, help="output JSON; default is stdout")
    args = parser.parse_args()
    try:
        report = build_report(args.ik_screen, args.trajectory_meta)
        content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(content, encoding="utf-8")
            print(f"[OUT] {args.out.resolve()}")
            for label, summary in [("IK only", report["ik"])] + [("Trajectory", t) for t in report["trajectories"]]:
                print(f"[{label}] {summary['counts']}; largest={summary['largest_all_success_sampled_rectangle']}")
        else:
            print(content, end="")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
