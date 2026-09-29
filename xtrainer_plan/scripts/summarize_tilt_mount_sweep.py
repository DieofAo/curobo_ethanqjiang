#!/usr/bin/env python3
"""CPU-only comparison of a recorded base XYZ × world-Y-tilt smoke sweep.

The coverage plot colors only complete pick/place results whose saved
trajectories passed the independent audits.  Other outcomes are shown with
separate status colors.  The base-to-grasp angle is simple task geometry, not
an observed robot link or joint orientation.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from compare_overhead_yshift import compare_cases
from summarize_overhead_cartesian import inspect_candidate, rank_key


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ry(degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def line_angles(base_xyz, grasp_grid):
    """Angle from vertical of base-to-target lines, independent of the robot."""
    xs = np.linspace(*grasp_grid["x_range"], int(grasp_grid["cols"]))
    ys = np.linspace(*grasp_grid["y_range"], int(grasp_grid["rows"]))
    base = np.asarray(base_xyz, dtype=float)
    targets = np.array([(x, y, grasp_grid["z"]) for y in ys for x in xs])
    delta = targets - base
    angles = np.degrees(np.arctan2(np.linalg.norm(delta[:, :2], axis=1), np.abs(delta[:, 2])))
    center = np.array([(xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2, grasp_grid["z"]])
    d = center - base
    center_angle = math.degrees(math.atan2(float(np.linalg.norm(d[:2])), abs(float(d[2]))))
    return {"center_deg": center_angle, "median_grid_deg": float(np.median(angles)),
            "min_grid_deg": float(np.min(angles)), "max_grid_deg": float(np.max(angles))}


def validate_config(source, cfg, row, result, size):
    xyz = np.asarray(row["base_xyz_m"], dtype=float)
    mount = np.asarray(cfg["robot"]["mount_transform"], dtype=float)
    baseline = np.asarray(source["robot"]["mount_transform"], dtype=float)
    tilt = float(row["world_y_tilt_deg"])
    if (mount.shape != (4, 4) or baseline.shape != (4, 4) or xyz.shape != (3,) or
            not np.isfinite(mount).all() or not np.isfinite(xyz).all() or not math.isfinite(tilt)):
        raise ValueError("Invalid mount transform or sweep coordinate")
    if not np.allclose(mount[:3, 3], xyz, atol=1e-9, rtol=0):
        raise ValueError("Manifest and config mount positions differ")
    if not np.allclose(mount[:3, :3], ry(tilt) @ baseline[:3, :3], atol=1e-9, rtol=0):
        raise ValueError("Config rotation is not source mount rotated around task_world +Y")
    if not np.allclose(mount[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("Invalid homogeneous mount row")
    original_pp, pp = source["pick_place"], cfg["pick_place"]
    for key in ("home", "place", "angle_search", "criterion"):
        if pp[key] != original_pp[key]:
            raise ValueError(f"Candidate changed fixed pick/place field: {key}")
    if cfg["robot"]["joint_limit_clip"] != source["robot"]["joint_limit_clip"]:
        raise ValueError("Candidate changed joint limit clip")
    expected_linear = dict(original_pp["linear_move"])
    expected_linear.update(method="waypoints_fk", waypoint_step_m=.0075)
    if pp["linear_move"] != expected_linear:
        raise ValueError("Candidate changed the uniform waypoint linear-move settings")
    grid = pp["grasp_grid"]
    fixed_grid = {key: value for key, value in original_pp["grasp_grid"].items()
                  if key not in ("rows", "cols")}
    if ({key: value for key, value in grid.items() if key not in ("rows", "cols")} != fixed_grid or
            grid["rows"] != size or grid["cols"] != size):
        raise ValueError("Candidate changed grasp region or grid size")
    if Path(cfg["output"]["dir"]).resolve() != result or cfg["output"]["add_timestamp"]:
        raise ValueError("Candidate output is not the recorded result path")
    return grid


def aggregate(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = read_json(manifest_path)
    source_path = Path(manifest["source"])
    if sha256(source_path) != manifest["source_sha256"]:
        raise ValueError("Source config changed since manifest generation")
    source = read_json(source_path)
    candidates = manifest["candidates"]
    if not candidates or len(candidates) != manifest["n_configs"]:
        raise ValueError("Manifest candidate count differs")
    size = int(manifest["parameters"]["size"])
    rows, case_baseline, coords = [], None, set()
    for index, item in enumerate(candidates):
        name = item["name"]
        config, result = Path(item["config"]), Path(item["result"])
        if (item["index"] != index or config != root / "configs" / f"{name}.json" or
                result != root / "runs" / name):
            raise ValueError(f"Manifest row {index} has inconsistent local paths/index")
        if sha256(config) != item["config_sha256"]:
            raise ValueError(f"Config changed since manifest generation: {config}")
        cfg = read_json(config)
        grid = validate_config(source, cfg, item, result, size)
        coord = (*item["base_xyz_m"], item["world_y_tilt_deg"])
        if coord in coords:
            raise ValueError(f"Duplicate sweep coordinates: {coord}")
        coords.add(coord)
        row, cases = inspect_candidate(result)
        row.update(index=index, name=name, config=str(config),
                   config_sha256=item["config_sha256"],
                   mount_xyz_original_LINK0_m=item["base_xyz_m"],
                   world_y_tilt_deg=item["world_y_tilt_deg"],
                   base_to_grasp_line_deviation_from_vertical_deg=line_angles(item["base_xyz_m"], grid))
        reported_config = row.get("provenance", {}).get("config_file")
        if reported_config and Path(reported_config).resolve() != config:
            row["status"] = "config_mismatch"
            row["ranking_eligible"] = False
            row["issues"].append("Run status refers to a different config than the manifest")
            cases = None
        if cases is not None:
            if case_baseline is None:
                case_baseline = cases
            else:
                compare_cases(case_baseline, cases)
        rows.append(row)
    ranked = sorted((r for r in rows if r["ranking_eligible"]), key=rank_key)
    for position, row in enumerate(ranked, 1):
        row["rank"] = position
    best = ranked[0]["success_fraction"] if ranked else None
    axes = {key: sorted({r[key] for r in rows}) for key in ("world_y_tilt_deg",)}
    axes.update({key: sorted({r["mount_xyz_original_LINK0_m"][i] for r in rows})
                 for i, key in enumerate(("base_x_m", "base_y_m", "base_z_m"))})
    expected = set(np.ndindex(*(len(axes[key]) for key in ("base_x_m", "base_y_m", "base_z_m", "world_y_tilt_deg"))))
    indexed = {(axes["base_x_m"].index(r["mount_xyz_original_LINK0_m"][0]),
                axes["base_y_m"].index(r["mount_xyz_original_LINK0_m"][1]),
                axes["base_z_m"].index(r["mount_xyz_original_LINK0_m"][2]),
                axes["world_y_tilt_deg"].index(r["world_y_tilt_deg"])) for r in rows}
    missing = [[axes["base_x_m"][i], axes["base_y_m"][j], axes["base_z_m"][k],
                axes["world_y_tilt_deg"][m]] for i, j, k, m in sorted(expected - indexed)]
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "source": str(source_path), "source_sha256": manifest["source_sha256"],
            "scope": {"smoke_only": True, "cpu_only": True,
                      "color_requires_independently_verified_saved_trajectories": True,
                      "base_to_grasp_angle_note": "Angle between vertical and the line from the mount origin "
                          "to a grasp target in original LINK_0; geometric proxy only, not an actual arm pose.",
                      "zero_success_note": "0/N means every smoke target was attempted; no saved trajectory "
                          "exists to independently audit. Home or incomplete planning failure has no success rate."},
            "sweep_axes": axes, "missing_cartesian_coordinates": missing,
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "n_candidates": len(rows), "n_ranked": len(ranked),
            "best_verified_success_fraction": best,
            "best_coverage_result_paths": [r["result"] for r in ranked if r["success_fraction"] == best],
            "ranked_result_paths": [r["result"] for r in ranked],
            "candidates": rows}


def write_csv(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("name", "base_x_m", "base_y_m", "base_z_m", "world_y_tilt_deg",
              "base_to_grasp_center_deg", "base_to_grasp_median_grid_deg", "status",
              "n_success", "n_total", "ranking_eligible", "rank", "planning_elapsed_s",
              "link3_place_intersection_cases", "j6_effective_limit_margin_deg", "result")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["candidates"]:
            x, y, z = row["mount_xyz_original_LINK0_m"]
            angle = row["base_to_grasp_line_deviation_from_vertical_deg"]
            writer.writerow({"name": row["name"], "base_x_m": x, "base_y_m": y,
                             "base_z_m": z, "world_y_tilt_deg": row["world_y_tilt_deg"],
                             "base_to_grasp_center_deg": angle["center_deg"],
                             "base_to_grasp_median_grid_deg": angle["median_grid_deg"],
                             "status": row["status"], "n_success": row["n_success"],
                             "n_total": row["n_total"], "ranking_eligible": row["ranking_eligible"],
                             "rank": row.get("rank"), "planning_elapsed_s": row["planning_elapsed_s"],
                             "link3_place_intersection_cases": row.get("LINK3", {}).get("place_related", {}).get("n_plane_intersection_cases"),
                             "j6_effective_limit_margin_deg": row.get("J6", {}).get("minimum_effective_limit_margin_deg"),
                             "result": row["result"]})


def plot_report(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch, Rectangle

    axes = report["sweep_axes"]
    xs, ys, zs, tilts = (axes[key] for key in
                         ("base_x_m", "base_y_m", "base_z_m", "world_y_tilt_deg"))
    figure, panels = plt.subplots(len(xs), len(tilts), squeeze=False,
                                  figsize=(4.6 * len(tilts), 4.0 * len(xs) + 1.2),
                                  layout="constrained")
    cmap = plt.get_cmap("YlGn")
    status_colors = {"pending": "#eeeeee", "home_failed": "#e8c3c3",
                     "planning_failed": "#d49f9f", "completed_zero_success": "#c9d5ec",
                     "audit_missing": "#f3e6aa", "audit_failed": "#e9b67b",
                     "config_mismatch": "#c87575"}
    lookup = {(*r["mount_xyz_original_LINK0_m"], r["world_y_tilt_deg"]): r
              for r in report["candidates"]}
    for i, x in enumerate(xs):
        for j, tilt in enumerate(tilts):
            ax = panels[i, j]
            for k, y in enumerate(ys):
                for m, z in enumerate(zs):
                    row = lookup.get((x, y, z, tilt))
                    if row is None:
                        face, label = "#ffffff", "missing"
                    elif row["ranking_eligible"]:
                        face = cmap(row["success_fraction"])
                        label = f"{row['n_success']}/{row['n_total']}"
                    else:
                        face = status_colors.get(row["status"], "#eeeeee")
                        labels = {"pending": "pending", "home_failed": "HOME fail",
                                  "planning_failed": "plan fail", "completed_zero_success": "0/9 tried",
                                  "audit_missing": "audit pending", "audit_failed": "audit FAIL",
                                  "config_mismatch": "config FAIL"}
                        label = labels.get(row["status"], row["status"])
                        if row["n_success"] is not None and row["status"].startswith("audit"):
                            label = f"{row['n_success']}/{row['n_total']}\n{label}"
                    ax.add_patch(Rectangle((m - .5, k - .5), 1, 1, facecolor=face,
                                           edgecolor="white", linewidth=1.2))
                    if row is not None:
                        angle = row["base_to_grasp_line_deviation_from_vertical_deg"]["center_deg"]
                        label += f"\nθ={angle:.0f}°"
                        if row["result"] in report["best_coverage_result_paths"]:
                            ax.add_patch(Rectangle((m - .48, k - .48), .96, .96,
                                                   fill=False, edgecolor="#cf7500", linewidth=2.4))
                    ax.text(m, k, label, ha="center", va="center", fontsize=7.8,
                            color="white" if row is not None and row["ranking_eligible"]
                            and row["success_fraction"] > .7 else "black")
            ax.set(xlim=(-.5, len(zs) - .5), ylim=(-.5, len(ys) - .5),
                   xticks=range(len(zs)), yticks=range(len(ys)),
                   xlabel="Base Z (m)", ylabel="Base Y (m)",
                   title=f"Base X={x:.2f} m, +Y tilt={tilt:g}°")
            ax.set_xticklabels([f"{v:.2f}" for v in zs])
            ax.set_yticklabels([f"{v:.2f}" for v in ys])
            ax.set_aspect("equal")
    figure.suptitle("3×3 full pick/place smoke by mount XYZ and task_world +Y tilt\n"
                    "θ: base-to-grasp-center line from vertical (geometry only; not an arm angle)",
                    fontsize=12)
    colorbar = figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                              ax=panels.ravel().tolist(), shrink=.62, pad=.015)
    colorbar.set_label("Verified complete-trajectory success fraction")
    legend = [Patch(facecolor=color, label=status.replace("_", " "))
              for status, color in status_colors.items()]
    legend.append(Patch(facecolor="white", edgecolor="#cf7500", linewidth=2,
                        label="top verified coverage"))
    figure.legend(handles=legend, loc="outside lower center", ncol=4, fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=170, bbox_inches="tight")
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.out, args.csv, args.plot):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite output: {path}")
    report = aggregate(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    write_csv(report, args.csv)
    plot_report(report, args.plot)
    print(json.dumps({key: report[key] for key in
                      ("status_counts", "n_ranked", "best_verified_success_fraction",
                       "best_coverage_result_paths")}, indent=2))
    print(f"[OUT] {args.out.resolve()}\n[CSV] {args.csv.resolve()}\n[PLOT] {args.plot.resolve()}")


if __name__ == "__main__":
    main()
