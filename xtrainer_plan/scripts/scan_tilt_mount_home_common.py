#!/usr/bin/env python3
"""Find a fixed Cartesian Home that has collision-safe IK in every mount case.

This is an endpoint screen, not a trajectory test. First intersect complete
Home-grid reports from representative mounts, then test only the survivors in
each remaining mount. Existing configs and reports are never modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from plan_pick_place import load_pick_place_config, transform_pose
from plan_trajectory import (load_robot_cfg_dict, make_world_config,
                             restrict_world_collision_to_links)
from scan_overhead_ik import GoalSpec, IK_ONLY_WARNING, make_ik_solver, solve_goals
from xtrainer_common import PoseSpec, parse_rigid_transform_matrix


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def position_key(row):
    return tuple(float(v) for v in row["home_position"])


def seed_survivors(seed_reports, candidates):
    """Return grid positions feasible in every seed report, ranked robustly."""
    by_name = {entry["name"]: entry for entry in candidates}
    indexed = {}
    for report in seed_reports:
        source = report["config"]
        config_path = Path(source["path"])
        name = config_path.stem
        if name not in by_name:
            raise ValueError(f"Seed case is absent from manifest: {name}")
        expected = by_name[name]["config_sha256"]
        if source["sha256"] != expected or sha256(config_path) != expected:
            raise ValueError(f"Seed config hash mismatch: {name}")
        if name in indexed:
            raise ValueError(f"Duplicate seed case: {name}")
        indexed[name] = {position_key(row): row for row in report["results"]
                         if row["key"].startswith("grid/") and row["feasible"]}
    if not indexed:
        raise ValueError("At least one complete Home-grid seed report is required")
    common = set.intersection(*(set(rows) for rows in indexed.values()))
    original = np.asarray(seed_reports[0]["original_home_position"], dtype=float)
    ranked = sorted(common, key=lambda xyz: (
        -min(rows[xyz]["representative_min_limit_margin_deg"]
             for rows in indexed.values()),
        float(np.linalg.norm(np.asarray(xyz) - original)), xyz))
    return indexed, ranked


def atomic_json(path: Path, data):
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def scan_case(entry, positions):
    config_path = Path(entry["config"])
    expected = entry["config_sha256"]
    if sha256(config_path) != expected:
        raise ValueError(f"Config hash mismatch before IK: {entry['name']}")
    cfg = load_pick_place_config(str(config_path))
    home = cfg["pick_place"]["home"]
    if home.get("joint_deg") is not None or cfg["robot"].get("dual_arm_prefix"):
        raise ValueError("Expected a single-arm Cartesian Home")
    correction = parse_rigid_transform_matrix(cfg["pick_place"]["link0_target_transform"])
    goals = []
    for i, position in enumerate(positions):
        raw = PoseSpec.from_rpy_deg(f"home/c{i}", position, home["rpy_deg"], "start")
        goals.append(GoalSpec(raw.name, "home", raw,
                              transform_pose(raw, correction), point_index=i))
    world = make_world_config(cfg["workspace"])
    robot = load_robot_cfg_dict(cfg["robot"])
    ik, tensor_args = make_ik_solver(robot, world, cfg["planner"])
    wall = cfg["workspace"].get("wall") or {}
    if wall.get("enable") and wall.get("collision_link_names"):
        restrict_world_collision_to_links(ik, list(wall["collision_link_names"]))
    limits = ik.kinematics.get_joint_limits().position.detach().cpu().numpy().astype(float)
    min_margin = float(cfg["pick_place"]["criterion"].get("min_limit_margin_deg") or 0.)
    results, batches = solve_goals(ik, tensor_args, goals, limits[0], limits[1],
                                   min_margin, batch_size=32, return_seeds=8)
    if sha256(config_path) != expected:
        raise RuntimeError(f"Config changed during IK: {entry['name']}")
    return {
        "schema_version": 1, "name": entry["name"], "config": str(config_path),
        "config_sha256": expected, "home_rpy_deg": home["rpy_deg"],
        "positions": [list(x) for x in positions],
        "results": {str(i): results[f"home/c{i}"] for i in range(len(positions))},
        "batches": batches,
    }


def summarize(candidates, seed_rows, positions, out_dir):
    rows = []
    for entry in candidates:
        name = entry["name"]
        if name in seed_rows:
            by_position = seed_rows[name]
            result = [by_position.get(tuple(position)) for position in positions]
        else:
            path = out_dir / f"{name}.json"
            if not path.exists():
                continue
            case = json.loads(path.read_text())
            if case["config_sha256"] != entry["config_sha256"] or case["positions"] != positions:
                raise ValueError(f"Stale case report: {path}")
            result = [case["results"][str(i)] for i in range(len(positions))]
        rows.append({"name": name, "base_xyz_m": entry["base_xyz_m"],
                     "world_y_tilt_deg": entry["world_y_tilt_deg"],
                     "results": [{"feasible": bool(r and r["feasible"]),
                                  "q_deg": None if r is None else r["representative_q_deg"],
                                  "min_limit_margin_deg": None if r is None else r["representative_min_limit_margin_deg"]}
                                 for r in result]})
    summary = []
    for i, position in enumerate(positions):
        feasible = [row for row in rows if row["results"][i]["feasible"]]
        margins = [row["results"][i]["min_limit_margin_deg"] for row in feasible]
        summary.append({"position": position, "n_feasible": len(feasible),
                        "n_scanned": len(rows), "feasible_all_scanned": len(feasible) == len(rows),
                        "min_margin_deg": min(margins) if margins else None,
                        "mean_margin_deg": float(np.mean(margins)) if margins else None})
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "warning": IK_ONLY_WARNING, "n_expected": len(candidates),
            "n_scanned": len(rows), "complete": len(rows) == len(candidates),
            "candidate_summary": summary, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed-report", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    candidates = manifest["candidates"]
    if len(candidates) != manifest["n_configs"] or len({e["name"] for e in candidates}) != len(candidates):
        raise ValueError("Invalid manifest candidates")
    seed_reports = [json.loads(path.read_text()) for path in args.seed_report]
    seed_rows, survivors = seed_survivors(seed_reports, candidates)
    positions = [list(p) for p in survivors]
    print(f"[HOME] {len(positions)} survivors across {len(seed_rows)} seed mounts; "
          f"{len(candidates)} mounts total: {positions}", flush=True)
    if args.dry_run:
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = {"schema_version": 1, "manifest": str(args.manifest.resolve()),
                "manifest_sha256": sha256(args.manifest),
                "seed_reports": [{"path": str(p.resolve()), "sha256": sha256(p)}
                                 for p in args.seed_report],
                "positions": positions, "warning": IK_ONLY_WARNING}
    protocol_path = args.out_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing output protocol differs")
    else:
        atomic_json(protocol_path, protocol)
    for index, entry in enumerate(candidates, 1):
        if entry["name"] in seed_rows:
            continue
        report_path = args.out_dir / f"{entry['name']}.json"
        if report_path.exists():
            print(f"[SKIP] {entry['name']} report exists", flush=True)
            continue
        print(f"[SCAN] {index}/{len(candidates)} {entry['name']}", flush=True)
        atomic_json(report_path, scan_case(entry, positions))
        atomic_json(args.out_dir / "summary.json",
                    summarize(candidates, seed_rows, positions, args.out_dir))
    report = summarize(candidates, seed_rows, positions, args.out_dir)
    atomic_json(args.out_dir / "summary.json", report)
    print(f"[DONE] {report['n_scanned']}/{report['n_expected']} mounts", flush=True)
    for row in report["candidate_summary"]:
        print(f"[HOME] {row['position']} feasible={row['n_feasible']}/{row['n_scanned']} "
              f"minimum_margin_deg={row['min_margin_deg']}", flush=True)


if __name__ == "__main__":
    main()
