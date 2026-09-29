#!/usr/bin/env python3
"""Plot audited minimum joint-limit margin for a local-base-Y mount sweep."""

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


def collect(comparison):
    report = read_json(comparison)
    manifest_path = Path(report["manifest"])
    manifest = read_json(manifest_path)
    if (sha256(manifest_path) != report["manifest_sha256"] or
            manifest["parameters"].get("tilt_axis") != "original_base_local_y" or
            "local_y_tilt_deg" not in report["sweep_axes"]):
        raise ValueError("Expected original-base-local-Y comparison")
    rows = []
    for candidate in report["candidates"]:
        row = {"name": candidate["name"], "base_x_m": candidate["mount_xyz_original_LINK0_m"][0],
               "base_y_m": candidate["mount_xyz_original_LINK0_m"][1],
               "base_z_m": candidate["mount_xyz_original_LINK0_m"][2],
               "local_y_tilt_deg": candidate["local_y_tilt_deg"],
               "status": candidate["status"], "n_success": candidate["n_success"],
               "n_total": candidate["n_total"], "minimum_raw_margin_deg": None,
               "minimum_effective_margin_deg": None, "limiting_joint": None}
        if candidate["status"] == "completed_verified":
            run = Path(candidate["result"])
            audit_path = run / "joint_limit_clip_audit.json"
            audit = read_json(audit_path)
            source = audit["source"]
            traj = audit["trajectory"]
            if (not audit["passed"] or not audit["verification_completed"] or
                    not traj["checked"] or not traj["passed"] or
                    source["trajectory_sha256"] != sha256(run / "trajectory.npz") or
                    source["config_source_sha256"] != sha256(run / "trajectory_meta.json")):
                raise ValueError(f"Unverified joint margin source: {candidate['name']}")
            effective = traj["minimum_effective_margin_per_joint_rad"]
            if len(effective) != 6 or not all(math.isfinite(v) and v >= 0 for v in effective):
                raise ValueError(f"Invalid six-joint margin: {candidate['name']}")
            minimum = min(effective)
            if not math.isclose(minimum, traj["minimum_effective_margin_rad"], abs_tol=1e-7):
                raise ValueError(f"Inconsistent effective margin: {candidate['name']}")
            raw = traj["minimum_raw_margin_rad"]
            if not math.isclose(raw - minimum, audit["clip_rad"], abs_tol=1e-6):
                raise ValueError(f"Raw/effective margin mismatch: {candidate['name']}")
            row.update(minimum_raw_margin_deg=math.degrees(raw),
                       minimum_effective_margin_deg=math.degrees(minimum),
                       limiting_joint=audit["joint_names"][effective.index(minimum)])
        rows.append(row)
    return rows


def write_csv(rows, path):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    xs = sorted({r["base_x_m"] for r in rows})
    ys = sorted({r["base_y_m"] for r in rows})
    zs = sorted({r["base_z_m"] for r in rows})
    tilts = sorted({r["local_y_tilt_deg"] for r in rows})
    lookup = {(r["base_x_m"], r["base_y_m"], r["base_z_m"], r["local_y_tilt_deg"]): r
              for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate sweep coordinates")
    values = [r["minimum_effective_margin_deg"] for r in rows
              if r["minimum_effective_margin_deg"] is not None]
    limit = max(1., max(values, default=0.))
    cmap, norm = plt.get_cmap("viridis"), Normalize(0, limit)
    figure, panels = plt.subplots(len(xs), len(tilts), squeeze=False,
                                  figsize=(4.1 * len(tilts), 3.6 * len(xs) + 1.2),
                                  layout="constrained")
    for i, x in enumerate(xs):
        for j, tilt in enumerate(tilts):
            ax = panels[i, j]
            for k, y in enumerate(ys):
                for m, z in enumerate(zs):
                    row = lookup[(x, y, z, tilt)]
                    value = row["minimum_effective_margin_deg"]
                    face = cmap(norm(value)) if value is not None else "#e4e4e4"
                    ax.add_patch(Rectangle((m - .5, k - .5), 1, 1, facecolor=face,
                                           edgecolor="white", linewidth=1.2))
                    if value is None:
                        label = ("pending" if row["status"] == "pending" else
                                 "0/9\nno saved path" if row["n_success"] == 0 else row["status"])
                    else:
                        label = f"{value:.1f}°  {row['limiting_joint']}\n{row['n_success']}/{row['n_total']}"
                    ax.text(m, k, label, ha="center", va="center", fontsize=8,
                            color="white" if value is not None and value < .5 * limit else "black")
            ax.set(xlim=(-.5, len(zs) - .5), ylim=(-.5, len(ys) - .5),
                   xticks=range(len(zs)), yticks=range(len(ys)),
                   xlabel="Base Z (m)", ylabel="Base Y (m)",
                   title=f"Base X={x:.2f} m, local +Y tilt={tilt:g}°")
            ax.set_xticklabels([f"{v:.2f}" for v in zs])
            ax.set_yticklabels([f"{v:.2f}" for v in ys])
            ax.set_aspect("equal")
    bar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          ax=panels.ravel().tolist(), shrink=.75, pad=.02)
    bar.set_label("Minimum saved-trajectory distance to effective joint bounds (degrees)")
    figure.suptitle("XTrainer local-base-Y smoke: audited minimum over all six joints and saved samples\n"
                     "Gray: no successful saved trajectory or pending")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.out, args.csv):
        if path.exists():
            parser.error(f"Output already exists: {path}")
    rows = collect(args.comparison)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.csv)
    plot(rows, args.out)
    print(f"[ROWS] {len(rows)} [AUDITED] {sum(r['minimum_effective_margin_deg'] is not None for r in rows)}")
    print(f"[CSV] {args.csv.resolve()}\n[PLOT] {args.out.resolve()}")


if __name__ == "__main__":
    main()
