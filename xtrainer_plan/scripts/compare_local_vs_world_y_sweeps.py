#!/usr/bin/env python3
"""Compare matched local-base-Y and task-world-Y XTrainer smoke results."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row, axis):
    return (*row["mount_xyz_original_LINK0_m"], float(row[axis]))


def index_rows(report, axis):
    rows = {key(row, axis): row for row in report["candidates"]}
    if len(rows) != len(report["candidates"]):
        raise ValueError(f"Duplicate {axis} sweep coordinates")
    return rows


def posture_rows(report):
    rows = {row["name"]: row for row in report["candidates"]}
    if len(rows) != len(report["candidates"]):
        raise ValueError("Duplicate posture candidate")
    return rows


def margin_deg(row):
    if row["status"] == "completed_zero_success":
        return None
    if row["status"] != "completed_verified":
        raise ValueError(f"Incomplete or unaudited candidate: {row['name']}: {row['status']}")
    run = Path(row["result"])
    audit = read_json(run / "joint_limit_clip_audit.json")
    meta = run / "trajectory_meta.json"
    trajectory = run / "trajectory.npz"
    if (not audit["passed"] or not audit["verification_completed"] or
            not audit["trajectory"]["checked"] or not audit["trajectory"]["passed"] or
            audit["source"]["config_source_sha256"] != sha256(meta) or
            audit["source"]["trajectory_sha256"] != sha256(trajectory)):
        raise ValueError(f"Invalid joint audit source: {row['name']}")
    value = math.degrees(audit["trajectory"]["minimum_effective_margin_rad"])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid effective margin: {row['name']}")
    return value


def observed_median(row, posture):
    found = posture[row["name"]]
    if row["status"] == "completed_zero_success":
        if found["angle"] is not None:
            raise ValueError(f"Unexpected posture for zero-success candidate: {row['name']}")
        return None
    if found["audit"] != "verified" or found["n_success"] != row["n_success"]:
        raise ValueError(f"Unaudited or mismatched posture: {row['name']}")
    return found["angle"]["median_deg"]


def compare(world_comparison, world_posture, local_comparison, local_posture):
    world = read_json(world_comparison)
    local = read_json(local_comparison)
    wp = posture_rows(read_json(world_posture))
    lp = posture_rows(read_json(local_posture))
    if world["source_sha256"] != local["source_sha256"]:
        raise ValueError("World and local sweeps do not share a source config")
    if ("world_y_tilt_deg" not in world["sweep_axes"] or
            "local_y_tilt_deg" not in local["sweep_axes"]):
        raise ValueError("Sweep axis metadata does not match this paired comparison")
    wr = index_rows(world, "world_y_tilt_deg")
    lr = index_rows(local, "local_y_tilt_deg")
    if wr.keys() != lr.keys() or len(wr) != 72:
        raise ValueError("World and local sweeps lack the same complete 72-point grid")
    rows = []
    for coord in sorted(wr):
        w, l = wr[coord], lr[coord]
        if w["n_total"] != 9 or l["n_total"] != 9 or w["n_success"] is None or l["n_success"] is None:
            raise ValueError(f"Incomplete matched smoke: {coord}")
        w_angle, l_angle = observed_median(w, wp), observed_median(l, lp)
        w_margin, l_margin = margin_deg(w), margin_deg(l)
        rows.append({"base_x_m": coord[0], "base_y_m": coord[1], "base_z_m": coord[2],
                     "nominal_tilt_deg": coord[3], "world_name": w["name"], "local_name": l["name"],
                     "world_success": w["n_success"], "local_success": l["n_success"],
                     "delta_success_local_minus_world": l["n_success"] - w["n_success"],
                     "world_arm_median_deg": w_angle, "local_arm_median_deg": l_angle,
                     "delta_arm_angle_local_minus_world_deg": l_angle - w_angle if w_angle is not None and l_angle is not None else None,
                     "world_effective_margin_deg": w_margin, "local_effective_margin_deg": l_margin,
                     "delta_margin_local_minus_world_deg": l_margin - w_margin if w_margin is not None and l_margin is not None else None})
    return rows


def plot_grid(rows, field, label, path, suffix, fmt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    xs = sorted({r["base_x_m"] for r in rows})
    ys = sorted({r["base_y_m"] for r in rows})
    zs = sorted({r["base_z_m"] for r in rows})
    tilts = sorted({r["nominal_tilt_deg"] for r in rows})
    by_coord = {(r["base_x_m"], r["base_y_m"], r["base_z_m"], r["nominal_tilt_deg"]): r for r in rows}
    values = [abs(r[field]) for r in rows if r[field] is not None]
    bound = max(max(values, default=0), 1 if field == "delta_success_local_minus_world" else .5)
    norm, cmap = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), plt.get_cmap("RdBu")
    fig, panels = plt.subplots(len(xs), len(tilts), squeeze=False,
                               figsize=(4 * len(tilts), 3.4 * len(xs) + 1.3), layout="constrained")
    for i, x in enumerate(xs):
        for j, tilt in enumerate(tilts):
            ax = panels[i, j]
            for k, y in enumerate(ys):
                for m, z in enumerate(zs):
                    row = by_coord[(x, y, z, tilt)]
                    value = row[field]
                    ax.add_patch(Rectangle((m - .5, k - .5), 1, 1,
                                           facecolor=cmap(norm(value)) if value is not None else "#dddddd",
                                           edgecolor="white", linewidth=1.2))
                    ax.text(m, k, fmt.format(value) if value is not None else "n/a",
                            ha="center", va="center", fontsize=10,
                            color="white" if value is not None and abs(value) > .65 * bound else "black")
            ax.set(xlim=(-.5, len(zs) - .5), ylim=(-.5, len(ys) - .5),
                   xticks=range(len(zs)), yticks=range(len(ys)), xlabel="Base Z (m)",
                   ylabel="Base Y (m)", title=f"Base X={x:.2f} m, nominal tilt={tilt:g}°")
            ax.set_xticklabels([f"{z:.2f}" for z in zs])
            ax.set_yticklabels([f"{y:.2f}" for y in ys])
            ax.set_aspect("equal")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                       ax=panels.ravel().tolist(), shrink=.75, pad=.02)
    bar.set_label(label)
    fig.suptitle(f"XTrainer original-base-local +Y minus task-world +Y: {suffix}\n"
                 "Same base XYZ, nominal tilt, source config and 3×3 task grid; gray = no paired saved trajectory")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("world-comparison", "world-posture", "local-comparison", "local-posture", "output-prefix"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    prefix = args.output_prefix
    paths = [Path(str(prefix) + suffix) for suffix in
             (".csv", ".json", "_success_delta.png", "_posture_delta.png", "_margin_delta.png")]
    if any(path.exists() for path in paths):
        parser.error("Output already exists; choose a new prefix")
    rows = compare(args.world_comparison, args.world_posture,
                   args.local_comparison, args.local_posture)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"n_paired_mounts": len(rows),
               "total_world_success": sum(r["world_success"] for r in rows),
               "total_local_success": sum(r["local_success"] for r in rows),
               "local_better_coverage": sum(r["delta_success_local_minus_world"] > 0 for r in rows),
               "world_better_coverage": sum(r["delta_success_local_minus_world"] < 0 for r in rows),
               "equal_coverage": sum(r["delta_success_local_minus_world"] == 0 for r in rows),
               "rows": rows}
    paths[1].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_grid(rows, "delta_success_local_minus_world", "Verified pick/place successes out of 9: local minus world",
              paths[2], "coverage difference", "{:+.0f}")
    plot_grid(rows, "delta_arm_angle_local_minus_world_deg", "Grasp shoulder-to-wrist angle from vertical (degrees): local minus world",
              paths[3], "observed arm-angle difference", "{:+.1f}°")
    plot_grid(rows, "delta_margin_local_minus_world_deg", "Minimum effective joint margin (degrees): local minus world",
              paths[4], "audited joint-margin difference", "{:+.1f}°")
    print(json.dumps({k:v for k,v in summary.items() if k != "rows"}, indent=2))
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
