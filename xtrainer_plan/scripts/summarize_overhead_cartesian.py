#!/usr/bin/env python3
"""CPU-only, failure-aware summary of a base XY x place X smoke sweep.

Only successful runs with matching, completed independent trajectory and joint
limit audits enter the ranking. Home failures and unfinished runs are never
represented as tested, unreachable grasp points. No solver or ROS is started.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import textwrap

import numpy as np

from compare_overhead_yshift import case_map, compare_cases, summarize


REPORT_NAMES = ("analysis_summary.json", "link3_grasp_clearance.json",
                "independent_verification.json", "joint_limit_clip_audit.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_coordinates(cfg):
    mount = np.asarray(cfg["robot"]["mount_transform"], dtype=float)
    place = np.asarray(cfg["pick_place"]["place"]["position"], dtype=float)
    clip = float(cfg["robot"]["joint_limit_clip"])
    if (mount.shape != (4, 4) or place.shape != (3,) or
            not np.isfinite(mount).all() or not np.isfinite(place).all() or
            not np.isfinite(clip) or clip < 0):
        raise ValueError("Invalid mount, place, or joint-limit clip")
    grid = cfg["pick_place"]["grasp_grid"]
    rows, cols = int(grid["rows"]), int(grid["cols"])
    if rows < 1 or cols < 1 or rows != grid["rows"] or cols != grid["cols"]:
        raise ValueError("Invalid grasp grid dimensions")
    if grid.get("perimeter_only", False):
        expected = rows * cols - max(0, rows - 2) * max(0, cols - 2)
    else:
        expected = rows * cols
    return {"mount_xyz_original_LINK0_m": mount[:3, 3].tolist(),
            "mount_rpy_deg": cfg["overhead"]["mount_rpy_deg"],
            "place_xyz_original_LINK0_m": place.tolist(),
            "joint_limit_clip_rad": clip,
            "grasp_grid": {key: grid.get(key) for key in
                           ("rows", "cols", "x_range", "y_range", "z", "order", "perimeter_only")},
            "expected_n_total": expected}


def inspect_candidate(result):
    """Return (status record, verified cases or None), retaining source paths."""
    result = Path(result).resolve()
    row = {"result": str(result), "status": "pending", "ranking_eligible": False,
           "n_success": None, "n_total": None, "success_fraction": None,
           "n_samples": None, "run_returncode": None, "planning_elapsed_s": None,
           "provenance": {"files": {}, "sha256": {}}, "issues": []}
    provenance = row["provenance"]

    def remember(path):
        provenance["files"][path.name] = str(path)
        provenance["sha256"][path.name] = sha256(path)

    run_path, meta_path = result / "run_status.json", result / "trajectory_meta.json"
    failed_path, log_path = result / "plan_failed.json", result / "plan.log"
    status, meta, failed, cfg, cfg_origin = None, None, None, None, None
    try:
        for path in (run_path, meta_path, failed_path, log_path):
            if path.exists():
                remember(path)
        if run_path.exists():
            status = read_json(run_path)
            row["run_returncode"] = int(status["returncode"])
            row["planning_elapsed_s"] = status.get("elapsed_s")
            source = Path(status["config"])
            # The batch runner records absolute config paths. Relative paths,
            # when supplied by an external runner, are resolved from its result.
            if not source.is_absolute():
                source = result / source
            if source.exists():
                cfg, cfg_origin = read_json(source), source
                provenance["config_file"] = str(source.resolve())
                provenance["config_file_sha256"] = sha256(source)
            else:
                row["issues"].append(f"Run config file is unavailable: {source}")
        if failed_path.exists():
            failed = read_json(failed_path)
            row["failure_stage"] = failed.get("stage")
            if cfg is None and "config" in failed:
                cfg, cfg_origin = failed["config"], failed_path
        if meta_path.exists():
            meta = read_json(meta_path)
            if cfg is not None and config_coordinates(cfg) != config_coordinates(meta["config"]):
                raise ValueError("Run config and saved trajectory have different sweep parameters")
            cfg, cfg_origin = meta["config"], meta_path
        if cfg is not None:
            row.update(config_coordinates(cfg))
            provenance["parameter_source"] = str(cfg_origin)
        # A planner can save metadata before it exits. Do not mistake this race
        # (or an interrupted partial run) for a completed batch result.
        if status is None:
            row["issues"].append("No run_status.json: completion has not been established")
            return row, None
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if failed and failed.get("stage") == "home_ik":
            row["status"] = "home_failed"
            row["issues"].append("Home IK failed before the grasp grid was attempted")
            return row, None
        if failed and failed.get("stage") == "no_item_success":
            completions = re.findall(r"\[PLAN\]\s*完成\s+(\d+)/(\d+)\s*个物料", log)
            attempted = re.findall(r"^--- 物料 (\d+)/(\d+)\s+抓取点", log, re.MULTILINE)
            expected = row.get("expected_n_total")
            row["n_cases_started_from_log"] = len(attempted)
            # The planner prints 0/N even with on_fail=stop after the first
            # failed case. Require distinct per-case headers as well.
            all_attempted = (len(attempted) == expected and
                             len({index for index, _ in attempted}) == expected and
                             all(int(total) == expected for _, total in attempted))
            if (completions and int(completions[-1][0]) == 0 and
                    int(completions[-1][1]) == expected and all_attempted):
                row.update(status="completed_zero_success", n_success=0,
                           n_total=expected, success_fraction=0.0, n_samples=0)
                row["issues"].append("All smoke cases attempted; no successful trajectory to audit. "
                                     "This is not proof of geometric unreachability")
            else:
                row["status"] = "planning_failed"
                row["issues"].append("No successful trajectory; full-grid completion was not confirmed")
            return row, None
        if row["run_returncode"] != 0 or meta is None:
            row["status"] = "planning_failed"
            row["issues"].append("Planner exited unsuccessfully or complete trajectory metadata is missing")
            return row, None
        if meta.get("partial"):
            row["status"] = "planning_failed"
            row["issues"].append("Batch exited but saved metadata is partial")
            return row, None
        cases = case_map(meta)
        if meta["n_items_total"] != row["expected_n_total"]:
            raise ValueError("Saved case count does not match the configured smoke grid")
        row.update(n_success=meta["n_items_success"], n_total=meta["n_items_total"],
                   success_fraction=meta["n_items_success"] / meta["n_items_total"],
                   n_samples=meta["n_points"])
        missing = []
        for name in REPORT_NAMES:
            path = result / name
            if not path.exists() and name in REPORT_NAMES[:2]:
                path = result.parent / name
            if not path.exists():
                missing.append(name)
        if missing:
            row["status"] = "audit_missing"
            row["issues"].append("Missing required reports: " + ", ".join(missing))
            return row, None
        # This helper revalidates source hashes, case identities, all report
        # counts, trajectory shape, joint order and actual effective bounds.
        try:
            validated, cases = summarize(result)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            row["status"] = "audit_failed"
            row["issues"].append(f"Audit source/consistency validation failed: {exc}")
            return row, None
        row.update(validated)
        if not (row["independent_verification_passed"] and row["joint_limit_clip_audit_passed"]):
            row["status"] = "audit_failed"
            row["issues"].append("Independent trajectory or joint-limit audit did not pass")
            return row, None
        row.update(status="completed_verified", ranking_eligible=True)
        return row, cases
    except (ValueError, KeyError, TypeError, OSError) as exc:
        row["status"] = "planning_failed" if status is not None else "pending"
        row["issues"].append(f"Result/config metadata could not be validated: {exc}")
        return row, None


def fixed_parameters(row):
    return {"base_z_m": row["mount_xyz_original_LINK0_m"][2],
            "mount_rpy_deg": row["mount_rpy_deg"],
            "place_yz_m": row["place_xyz_original_LINK0_m"][1:],
            "joint_limit_clip_rad": row["joint_limit_clip_rad"],
            "grasp_grid": row["grasp_grid"]}


def rank_key(row):
    """Coverage first; diagnostic tie-breaks, never new acceptance constraints."""
    place = row["LINK3"]["place_related"]
    return (-row["success_fraction"], -row["n_success"],
            place["n_plane_intersection_cases"] / row["n_success"],
            -row["J6"]["minimum_effective_limit_margin_rad"],
            row["J6"]["max_segment_span_deg"], row["result"])


def aggregate(results):
    if not results:
        raise ValueError("At least one candidate is required")
    if len({str(Path(path).resolve()) for path in results}) != len(results):
        raise ValueError("Duplicate result paths")
    rows, fixed, baseline_cases, coordinates = [], None, None, set()
    for result in results:
        row, cases = inspect_candidate(result)
        if "mount_xyz_original_LINK0_m" in row:
            current = fixed_parameters(row)
            if fixed is None:
                fixed = current
            elif current != fixed:
                raise ValueError("Candidates have different fixed parameters or grasp grids; summarize separately")
            point = (*row["mount_xyz_original_LINK0_m"][:2], row["place_xyz_original_LINK0_m"][0])
            if point in coordinates:
                raise ValueError(f"Duplicate base XY / place X sweep coordinate: {point}")
            coordinates.add(point)
        if cases is not None:
            if baseline_cases is None:
                baseline_cases = cases
            compare_cases(baseline_cases, cases)
        rows.append(row)
    ranked = sorted((row for row in rows if row["ranking_eligible"]), key=rank_key)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    best = ranked[0]["success_fraction"] if ranked else None
    top = [row["result"] for row in ranked if row["success_fraction"] == best]
    axes = {"base_x_m": sorted({p[0] for p in coordinates}),
            "base_y_m": sorted({p[1] for p in coordinates}),
            "place_x_m": sorted({p[2] for p in coordinates})}
    missing_coordinates = [[x, y, p] for x in axes["base_x_m"] for y in axes["base_y_m"]
                           for p in axes["place_x_m"] if (x, y, p) not in coordinates]
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": {"cpu_only": True, "smoke_only": True,
                      "ranking_requires_independent_verification": True,
                      "ranking_order": ["success_fraction descending", "n_success descending",
                                        "LINK3 place plane-intersection case fraction ascending (diagnostic tie-break)",
                                        "J6 minimum effective-limit margin descending",
                                        "J6 maximum segment span ascending", "result path ascending"],
                      "warning": "Smoke coverage is not full-grid coverage or a proof of unreachability. "
                          "Home failures and pending runs have null success counts. Joint/collision audits cover "
                          "saved samples, not physical tracking or inter-sample swept safety. LINK3 compares "
                          "a spherical envelope with the finite zero-thickness grasp plane, not object/CAD "
                          "collision; tie-break diagnostics do not add planning constraints. Aggregate "
                          "clearances compare different successful-case sets."},
            "fixed_parameters": fixed, "sweep_axes": axes,
            "missing_cartesian_coordinates": missing_coordinates,
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "n_candidates": len(rows), "n_ranked": len(ranked),
            "best_verified_success_fraction": best,
            "best_coverage_result_paths": top,
            "ranked_result_paths": [row["result"] for row in ranked],
            "selected_result": ranked[0]["result"] if ranked else None,
            "candidates": rows}


def plot_report(report, output):
    """One panel per base X; color only independently verified success rates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing plot: {output}")
    axes = report["sweep_axes"]
    xs, ys, pxs = axes["base_x_m"], axes["base_y_m"], axes["place_x_m"]
    if not xs or not ys or not pxs:
        raise ValueError("No candidate sweep coordinates are available to plot")
    columns = min(3, len(xs))
    rows = (len(xs) + columns - 1) // columns
    fig, panels = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.8 * rows),
                               squeeze=False, layout="constrained")
    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad("#eeeeee")
    labels = {"pending": "pending", "home_failed": "HOME fail", "planning_failed": "plan fail",
              "completed_zero_success": "0/{n}\nno traj", "audit_missing": "{s}/{n}\naudit pending",
              "audit_failed": "{s}/{n}\naudit FAIL"}
    indexed = {(*row["mount_xyz_original_LINK0_m"][:2], row["place_xyz_original_LINK0_m"][0]): row
               for row in report["candidates"] if "mount_xyz_original_LINK0_m" in row}
    for panel, x in zip(panels.flat, xs):
        data = np.full((len(ys), len(pxs)), np.nan)
        for i, y in enumerate(ys):
            for j, px in enumerate(pxs):
                row = indexed.get((x, y, px))
                if row is not None and row["ranking_eligible"]:
                    data[i, j] = row["success_fraction"]
        im = panel.imshow(data, origin="lower", vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for i, y in enumerate(ys):
            for j, px in enumerate(pxs):
                row = indexed.get((x, y, px))
                if row is None:
                    label = "not supplied"
                elif row["ranking_eligible"]:
                    label = f"{row['n_success']}/{row['n_total']}"
                    if row["result"] in report["best_coverage_result_paths"]:
                        panel.add_patch(Rectangle((j - .48, i - .48), .96, .96,
                                                  fill=False, edgecolor="#cf7500", linewidth=2.5))
                else:
                    label = labels[row["status"]].format(s=row.get("n_success"),
                                                         n=row.get("n_total"))
                panel.text(j, i, label, ha="center", va="center", fontsize=9,
                           color="white" if data[i, j] > .70 else "black")
        panel.set_xticks(range(len(pxs)), [f"{value:.2f}" for value in pxs])
        panel.set_yticks(range(len(ys)), [f"{value:.2f}" for value in ys])
        panel.set_xlabel("Place X in original LINK0 (m)")
        panel.set_ylabel("Base Y in original LINK0 (m)")
        panel.set_title(f"Base X = {x:.2f} m")
    for panel in list(panels.flat)[len(xs):]:
        panel.set_visible(False)
    fixed = report["fixed_parameters"]
    title_lines = ["Complete pick/place smoke: base XY x place X",
                   f"Base Z={fixed['base_z_m']:.2f} m; place Y={fixed['place_yz_m'][0]:.2f} m; "
                   f"place Z={fixed['place_yz_m'][1]:.2f} m; raw limits inset {fixed['joint_limit_clip_rad']:.2f} rad each side",
                   "Color: independently verified success fraction; orange: tied highest smoke coverage"]
    fig.suptitle("\n".join(textwrap.fill(line, width=55 * columns) for line in title_lines), fontsize=12)
    fig.colorbar(im, ax=[panel for panel in panels.flat if panel.get_visible()],
                 label="Verified complete-trajectory success fraction", shrink=.75)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    for path in (args.out, args.plot):
        if path is not None and path.exists():
            raise FileExistsError(f"Refusing to replace existing output: {path}")
    report = aggregate(args.results)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.plot:
        plot_report(report, args.plot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps({key: report[key] for key in ("status_counts", "n_ranked",
                     "best_verified_success_fraction", "best_coverage_result_paths", "selected_result")}, indent=2))
    print(f"[OUT] {args.out.resolve()}")
    if args.plot:
        print(f"[PLOT] {args.plot.resolve()}")


if __name__ == "__main__":
    main()
