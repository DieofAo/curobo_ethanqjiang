#!/usr/bin/env python3
"""Measure observed shoulder-to-wrist posture at saved XTrainer grasp endpoints.

This is a read-only, CPU-only URDF forward-kinematics analysis.  The shoulder
is the J_2 origin (LINK_2 frame), the wrist is the J_6 origin (LINK_6 frame),
and 0 degrees means their connecting line is parallel to task_world vertical.
The measure says nothing about elbow bend, joint limits, or grasp success on
targets for which no trajectory was saved.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


REPO = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xyz_rpy_transform(origin):
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    roll, pitch, yaw = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    if len(xyz) != 3:
        raise ValueError("URDF origin xyz must have three numbers")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
    rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = xyz
    return result


def axis_rotation(axis, radians):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(radians), math.sin(radians)
    result = np.eye(4)
    result[:3, :3] = c * np.eye(3) + (1. - c) * np.outer(axis, axis) + s * np.array(
        [[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return result


class UrdfChain:
    def __init__(self, path, base_link="LINK_0"):
        self.path = Path(path).resolve()
        self.base_link = base_link
        root = ET.parse(self.path).getroot()
        self.by_child = {}
        for joint in root.findall("joint"):
            parent, child = joint.find("parent"), joint.find("child")
            if parent is None or child is None:
                raise ValueError(f"URDF joint has no parent/child: {joint.get('name')}")
            name = child.get("link")
            if name in self.by_child:
                raise ValueError(f"URDF link has multiple parents: {name}")
            self.by_child[name] = joint

    def transform(self, link, q):
        chain, seen = [], set()
        while link != self.base_link:
            if link in seen or link not in self.by_child:
                raise ValueError(f"No acyclic URDF chain from {self.base_link} to {link}")
            seen.add(link)
            joint = self.by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        transform = np.eye(4)
        for joint in reversed(chain):
            transform = transform @ xyz_rpy_transform(joint.find("origin"))
            typ = joint.get("type")
            if typ in ("revolute", "continuous"):
                name = joint.get("name")
                if name not in q:
                    raise ValueError(f"Missing trajectory joint {name}")
                axis = [float(v) for v in joint.find("axis").get("xyz").split()]
                transform = transform @ axis_rotation(axis, q[name])
            elif typ == "prismatic":
                raise ValueError("Prismatic joint in shoulder-to-wrist chain is unsupported")
            elif typ != "fixed":
                raise ValueError(f"Unsupported URDF joint type: {typ}")
        return transform


def grasp_endpoints(meta, n_samples):
    """Return (item, zero-based sample index) without assuming failed items save samples."""
    cursor, found = 0, []
    for item in meta["items"]:
        if not item["success"]:
            continue
        segments = item["segments"]
        # plan_sequence drops the duplicate first sample of every later stage.
        total = sum(int(segment["n_points"]) for segment in segments) - len(segments) + 1
        if total != int(item["n_points"]):
            raise ValueError(f"Item {item['index']} segment/sample counts disagree")
        grabs = [i for i, segment in enumerate(segments) if segment["kind"] == "grasp"]
        if len(grabs) != 1:
            raise ValueError(f"Item {item['index']} has {len(grabs)} grasp segments")
        offset = (sum(int(segment["n_points"]) for segment in segments[:grabs[0] + 1])
                  - grabs[0])
        if offset < 1 or offset > total:
            raise ValueError(f"Item {item['index']} has invalid grasp endpoint")
        found.append((item, cursor + offset - 1))
        cursor += total
    if cursor != n_samples or cursor != int(meta["n_points"]):
        raise ValueError("Trajectory and successful item sample counts disagree")
    if len(found) != int(meta["n_items_success"]):
        raise ValueError("Metadata success count disagrees with saved items")
    return found


def audit_state(run_dir, npz_path, meta_path, urdf_path):
    path = run_dir / "independent_verification.json"
    if not path.exists():
        return "missing"
    audit = read_json(path)
    if not audit.get("passed") or not audit.get("verification_completed"):
        return "failed"
    source = audit.get("source", {})
    expected = (("npz_sha256", npz_path), ("metadata_sha256", meta_path),
                ("urdf_sha256", urdf_path))
    if any(source.get(key) != sha256(file) for key, file in expected):
        return "hash_mismatch"
    return "verified"


def angle_from_vertical(delta_world):
    delta_world = np.asarray(delta_world, dtype=float)
    length = np.linalg.norm(delta_world)
    if length <= 1e-9:
        raise ValueError("Shoulder and wrist coincide")
    return math.degrees(math.atan2(float(np.linalg.norm(delta_world[:2])),
                                   abs(float(delta_world[2]))))


def summarize_angles(angles):
    if not angles:
        return None
    values = np.asarray(angles, dtype=float)
    return {"min_deg": float(np.min(values)), "p10_deg": float(np.percentile(values, 10)),
            "median_deg": float(np.median(values)), "mean_deg": float(np.mean(values)),
            "p90_deg": float(np.percentile(values, 90)), "max_deg": float(np.max(values)),
            "fraction_within_20deg_of_vertical": float(np.mean(values <= 20.))}


def analyze_run(run_dir, name=None):
    run_dir = Path(run_dir).resolve()
    npz_path, meta_path = run_dir / "trajectory.npz", run_dir / "trajectory_meta.json"
    row = {"name": name or run_dir.name, "run": str(run_dir), "status": "pending",
           "n_success": None, "n_total": None, "n_observed_grasps": 0, "angle": None,
           "audit": "unavailable", "grasps": []}
    if not npz_path.exists() or not meta_path.exists():
        if npz_path.exists() != meta_path.exists():
            row["status"] = "incomplete"
        failure_path = run_dir / "plan_failed.json"
        if failure_path.exists():
            stage = read_json(failure_path).get("stage")
            if stage == "home_ik":
                row["status"] = "home_failed"
            elif stage == "no_item_success":
                row.update(status="zero_success", n_success=0)
            else:
                row["status"] = "planning_failed"
        elif (run_dir / "run_status.json").exists():
            result = read_json(run_dir / "run_status.json")
            row["status"] = "incomplete" if result.get("returncode") == 0 else "run_failed"
        return row
    meta = read_json(meta_path)
    robot = meta["robot"]
    urdf_path = Path(robot.get("urdf_abs") or REPO / robot["urdf"]).resolve()
    urdf = UrdfChain(urdf_path, robot.get("base_link", "LINK_0"))
    mount = np.asarray(meta["config"]["robot"]["mount_transform"], dtype=float)
    if mount.shape != (4, 4) or not np.allclose(mount[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError(f"Invalid mount_transform in {meta_path}")
    if not np.allclose(mount[:3, :3].T @ mount[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError(f"Non-orthonormal mount_transform in {meta_path}")
    with np.load(npz_path) as data:
        positions = data["positions"]
        ee_positions = data["ee_positions"]
        joint_names = [v.decode() if isinstance(v, bytes) else str(v)
                       for v in data["joint_names"]]
    if joint_names != robot["joint_names"] or positions.shape[1] != len(joint_names):
        raise ValueError(f"Trajectory joint names mismatch in {run_dir}")
    endpoints = grasp_endpoints(meta, len(positions))
    grasps = []
    for item, sample_index in endpoints:
        q = dict(zip(joint_names, positions[sample_index]))
        shoulder = urdf.transform("LINK_2", q)[:3, 3]
        wrist = urdf.transform("LINK_6", q)[:3, 3]
        tcp = urdf.transform(robot["ee_link"], q)[:3, 3]
        tcp_error_mm = float(np.linalg.norm(tcp - ee_positions[sample_index]) * 1000.)
        if tcp_error_mm > 0.1:
            raise ValueError(f"CPU URDF FK differs from saved TCP FK by {tcp_error_mm:.3f} mm "
                             f"at item {item['index']} in {run_dir}")
        shoulder_world = mount[:3, :3] @ shoulder + mount[:3, 3]
        wrist_world = mount[:3, :3] @ wrist + mount[:3, 3]
        delta = wrist_world - shoulder_world
        grasps.append({"item_index": int(item["index"]), "row": int(item["row"]),
                       "col": int(item["col"]), "sample_index": int(sample_index),
                       "angle_from_world_vertical_deg": angle_from_vertical(delta),
                       "shoulder_to_wrist_length_m": float(np.linalg.norm(delta)),
                       "vertical_component_m": float(delta[2]),
                       "shoulder_task_world_m": shoulder_world.tolist(),
                       "wrist_task_world_m": wrist_world.tolist(),
                       "saved_tcp_fk_error_mm": tcp_error_mm})
    row.update(status="observed", n_success=int(meta["n_items_success"]),
               n_total=int(meta["n_items_total"]), n_observed_grasps=len(grasps),
               angle=summarize_angles([g["angle_from_world_vertical_deg"] for g in grasps]),
               audit=audit_state(run_dir, npz_path, meta_path, urdf_path),
               mount_xyz_m=mount[:3, 3].tolist(),
               grasps=grasps)
    return row


def candidate_rows(manifest):
    data = read_json(manifest)
    items = data.get("candidates", data.get("rows"))
    if items is None:
        raise ValueError("Manifest must contain candidates or rows")
    return items


def candidate_tilt(item, cfg):
    """Return the explicit tilt axis and angle, accepting legacy world-Y data."""
    overhead = cfg.get("overhead", {})
    axes = {
        "task_world_y": "world_y_tilt_deg",
        "original_base_local_y": "local_y_tilt_deg",
    }
    present = [axis for axis, key in axes.items()
               if item.get(key) is not None or overhead.get(key) is not None]
    if len(present) > 1:
        raise ValueError("Candidate mixes world-Y and local-Y tilt metadata")
    if not present:
        if item.get("tilt_axis") is not None or overhead.get("tilt_axis") is not None:
            raise ValueError("Tilt axis is recorded without a tilt angle")
        return None, None
    axis = present[0]
    key = axes[axis]
    recorded = overhead.get(key)
    requested = item.get(key)
    if recorded is None or requested is None or not math.isclose(
            float(requested), float(recorded), abs_tol=1e-9):
        raise ValueError(f"Manifest/config {axis} tilt mismatch: {item['name']}")
    declared = item.get("tilt_axis")
    config_axis = overhead.get("tilt_axis")
    if axis == "original_base_local_y" and (declared != axis or config_axis != axis):
        raise ValueError(f"Local-Y tilt axis must be explicit: {item['name']}")
    if axis == "task_world_y" and (declared not in (None, axis) or
                                    config_axis not in (None, axis)):
        raise ValueError(f"World-Y tilt axis mismatch: {item['name']}")
    return axis, float(requested)


def analyze_manifest(path, baseline_run=None):
    path = Path(path).resolve()
    candidates = []
    for item in candidate_rows(path):
        row = analyze_run(item["result"], item["name"])
        cfg = read_json(item["config"])
        mount = np.asarray(cfg["robot"]["mount_transform"], dtype=float)
        if "base_xyz_m" in item and not np.allclose(mount[:3, 3], item["base_xyz_m"]):
            raise ValueError(f"Manifest mount position mismatch: {item['name']}")
        row["mount_xyz_m"] = mount[:3, 3].tolist()
        grid = cfg["pick_place"]["grasp_grid"]
        if row["n_total"] is None and not grid.get("perimeter_only", False):
            row["n_total"] = int(grid["rows"]) * int(grid["cols"])
        axis, tilt = candidate_tilt(item, cfg)
        row["tilt_axis"] = axis
        row["tilt_deg"] = tilt
        row["world_y_tilt_deg"] = tilt if axis == "task_world_y" else None
        row["local_y_tilt_deg"] = tilt if axis == "original_base_local_y" else None
        candidates.append(row)
    return {"schema_version": 1, "metric": {
                "shoulder": "J_2 origin / LINK_2 frame origin in production URDF",
                "wrist": "J_6 origin / LINK_6 frame origin in production URDF",
                "frame": "task_world", "angle": "acute angle to world vertical; 0 deg is upright, 90 deg horizontal",
                "sampling": "last saved sample of each successful grasp segment",
                "limits": "No angle for failed grasps; ignores elbow bend and full-arm clearance."},
            "manifest": str(path), "baseline": analyze_run(baseline_run, "v57_baseline")
            if baseline_run else None, "candidates": candidates}


def write_csv(report, prefix):
    case_path = Path(str(prefix) + "_cases.csv")
    point_path = Path(str(prefix) + "_grasps.csv")
    with case_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "run", "base_x_m", "base_y_m",
            "base_z_m", "tilt_axis", "tilt_deg", "world_y_tilt_deg", "local_y_tilt_deg",
            "status", "audit", "n_success", "n_total",
            "n_observed_grasps", "median_deg", "p10_deg", "p90_deg",
            "fraction_within_20deg_of_vertical"])
        writer.writeheader()
        for row in ([report["baseline"]] if report["baseline"] else []) + report["candidates"]:
            xyz = row.get("mount_xyz_m") or [None] * 3
            angle = row["angle"] or {}
            writer.writerow({"name": row["name"], "run": row["run"],
                "base_x_m": xyz[0], "base_y_m": xyz[1], "base_z_m": xyz[2],
                "tilt_axis": row.get("tilt_axis"), "tilt_deg": row.get("tilt_deg"),
                "world_y_tilt_deg": row.get("world_y_tilt_deg"),
                "local_y_tilt_deg": row.get("local_y_tilt_deg"), "status": row["status"],
                "audit": row["audit"], "n_success": row["n_success"],
                "n_total": row["n_total"], "n_observed_grasps": row["n_observed_grasps"],
                "median_deg": angle.get("median_deg"), "p10_deg": angle.get("p10_deg"),
                "p90_deg": angle.get("p90_deg"),
                "fraction_within_20deg_of_vertical": angle.get("fraction_within_20deg_of_vertical")})
    with point_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "item_index", "row", "col",
            "sample_index", "angle_from_world_vertical_deg", "shoulder_to_wrist_length_m",
            "vertical_component_m", "saved_tcp_fk_error_mm", "audit"])
        writer.writeheader()
        for row in ([report["baseline"]] if report["baseline"] else []) + report["candidates"]:
            for grasp in row["grasps"]:
                writer.writerow({"name": row["name"], "audit": row["audit"], **{
                    key: grasp[key] for key in writer.fieldnames if key in grasp}})
    return case_path, point_path


def plot_report(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    rows = report["candidates"]
    xs = sorted({r["mount_xyz_m"][0] for r in rows})
    ys = sorted({r["mount_xyz_m"][1] for r in rows})
    zs = sorted({r["mount_xyz_m"][2] for r in rows})
    tilt_axes = {r.get("tilt_axis") for r in rows if r.get("tilt_axis") is not None}
    if len(tilt_axes) > 1:
        raise ValueError("Posture plot requires one consistent tilt axis")
    tilt_axis = next(iter(tilt_axes), None)
    axis_label = {"task_world_y": "task_world +Y",
                  "original_base_local_y": "original base local +Y"}.get(tilt_axis, "no")
    tilts = sorted({r.get("tilt_deg") for r in rows if r.get("tilt_deg") is not None})
    if not tilts:
        tilts = [None]
    lookup = {(*r["mount_xyz_m"], r.get("tilt_deg")): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Candidate plot coordinates are not unique")
    fig, panels = plt.subplots(len(xs), len(tilts), squeeze=False,
                               figsize=(3.6 * len(tilts) + 1, 3.2 * len(xs) + 1.5),
                               layout="constrained")
    cmap, norm = plt.get_cmap("viridis"), Normalize(0., 90.)
    for i, x in enumerate(xs):
        for j, tilt in enumerate(tilts):
            ax = panels[i, j]
            for k, y in enumerate(ys):
                for m, z in enumerate(zs):
                    row = lookup.get((x, y, z, tilt))
                    angle = row["angle"] if row else None
                    if angle is None:
                        face = "#e5e5e5"
                        labels = {"pending": "pending", "home_failed": "Home IK\nfailed",
                                  "zero_success": "zero success", "run_failed": "run failed",
                                  "planning_failed": "plan failed", "incomplete": "incomplete"}
                        label = labels.get(row["status"], "no saved\ngrasp") if row else "missing"
                        if row and row["status"] == "zero_success":
                            label = f"0/{row['n_total']}\ncompleted"
                    else:
                        face = cmap(norm(angle["median_deg"]))
                        label = f"{angle['median_deg']:.0f}°\n{row['n_observed_grasps']}/{row['n_total']}"
                        if row["audit"] != "verified":
                            label += "\nunverified"
                    ax.add_patch(Rectangle((m - .5, k - .5), 1, 1, facecolor=face,
                                           edgecolor="white", linewidth=1))
                    ax.text(m, k, label, ha="center", va="center", fontsize=8,
                            color="white" if angle and angle["median_deg"] < 60 else "black")
            ax.set(xlim=(-.5, len(zs) - .5), ylim=(-.5, len(ys) - .5),
                   xticks=range(len(zs)), yticks=range(len(ys)), xlabel="Base Z (m)",
                   ylabel="Base Y (m)", title=f"Base X={x:.2f} m, {axis_label} tilt={tilt}°")
            ax.set_xticklabels([f"{v:.2f}" for v in zs])
            ax.set_yticklabels([f"{v:.2f}" for v in ys])
            ax.set_aspect("equal")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            ax=panels.ravel().tolist(), shrink=.82, pad=.02)
    colorbar.set_label("J2 shoulder to J6 wrist angle from world vertical (degrees)")
    baseline = report["baseline"]
    baseline_note = ""
    if baseline and baseline["angle"]:
        baseline_note = f" | V57 median {baseline['angle']['median_deg']:.1f}° ({baseline['n_observed_grasps']}/{baseline['n_total']})"
    fig.suptitle(f"Observed XTrainer grasp posture ({axis_label} tilt): "
                 "0° upright, 90° horizontal" + baseline_note)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    output_paths = [Path(str(prefix) + suffix) for suffix in
                    (".json", "_cases.csv", "_grasps.csv", ".png")]
    if any(path.exists() for path in output_paths):
        parser.error("Output files already exist; choose a new --output-prefix")
    report = analyze_manifest(args.manifest, args.baseline_run)
    output_paths[0].write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    write_csv(report, prefix)
    plot_report(report, output_paths[-1])
    print(json.dumps({"outputs": [str(path) for path in output_paths],
                      "baseline": report["baseline"]["angle"] if report["baseline"] else None,
                      "observed_candidates": sum(row["angle"] is not None
                                                 for row in report["candidates"])},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
