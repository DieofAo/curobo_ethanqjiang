#!/usr/bin/env python3
"""Plot verified, position-indexed joint metrics for one full XTrainer run.

Only saved samples are used.  A successful item includes all six stages,
including its entry from the previous successful item (or Home).  A failed
item has no joint samples and is drawn gray, never as a zero-valued metric.

Usage:
  python plot_full_mount_joint_distributions.py RESULT [--out-dir NEW_DIRECTORY]

The independent trajectory and joint-limit audits must already have passed.
Their recorded trajectory/metadata/URDF hashes are checked before plotting.
No planning, solver, ROS, or GPU work is performed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np

from compare_original_base_threeway import PHASES, case_metrics
from plot_grasp_angle_map import pick_cjk_font
from verify_joint_limit_clip import expected_limits, read_urdf_limits


REPO_ROOT = Path(__file__).resolve().parents[2]
JOINTS = [f"J_{index}" for index in range(1, 7)]
MAIN_METRICS = (
    ("min_raw_margin_deg", "min_raw_limit_distance.png",
     "原始 URDF 限位：全程最短距离", "Raw URDF limits: minimum full-cycle distance"),
    ("min_effective_margin_deg", "min_effective_limit_margin.png",
     "规划器有效限位：全程最小余量", "Planner clipped limits: minimum full-cycle margin"),
    ("max_cycle_joint_span_deg", "max_full_cycle_single_joint_span.png",
     "完整抓放：单关节最大角跨度", "Full pick/place: largest single-joint angular span"),
)
STAGE_METRIC = ("max_stage_checked_span_deg", "max_stage_j1_j5_span.png",
                "单阶段 J1–J5 最大角跨度", "Largest J1–J5 span within one stage")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def check_hash(path: Path, recorded: str, label: str) -> str:
    actual = sha256(path)
    require(isinstance(recorded, str) and actual == recorded,
            f"{label} hash differs from passed independent audit")
    return actual


def verified_inputs(result: Path):
    """Load the two independent audits and pin all inputs to their hashes."""
    meta_path = result / "trajectory_meta.json"
    npz_path = result / "trajectory.npz"
    verification_path = result / "independent_verification.json"
    clip_path = result / "joint_limit_clip_audit.json"
    for path in (meta_path, npz_path, verification_path, clip_path):
        require(path.is_file(), f"Missing required completed-run input: {path}")
    meta = read_json(meta_path)
    verification = read_json(verification_path)
    clip_audit = read_json(clip_path)
    require(verification.get("passed") is True
            and verification.get("verification_completed") is True,
            "Independent FK/collision/limit verification did not pass")
    require(clip_audit.get("passed") is True
            and clip_audit.get("verification_completed") is True
            and clip_audit.get("trajectory", {}).get("passed") is True
            and clip_audit.get("trajectory", {}).get("checked") is True,
            "Independent joint-limit clip audit did not pass")
    require(verification.get("gpu_checks", {}).get("stored_fk_matches_independent_fk") is True,
            "Independent FK comparison is missing or failed")
    meta_hash = check_hash(meta_path, verification["source"]["metadata_sha256"], "Metadata")
    npz_hash = check_hash(npz_path, verification["source"]["npz_sha256"], "Trajectory")
    require(clip_audit["source"]["config_source_sha256"] == meta_hash
            and clip_audit["source"]["trajectory_sha256"] == npz_hash,
            "Joint-limit audit refers to different saved metadata/trajectory")
    cfg = meta["config"]
    status_path = result / "run_status.json"
    if status_path.is_file():
        status = read_json(status_path)
        require(status.get("returncode") == 0, "Planning run did not complete successfully")
        config_path = Path(status["config"])
        require(config_path.is_file() and read_json(config_path) == cfg,
                "Saved configuration differs from run input")
    robot = cfg["robot"]
    urdf_path = Path(robot["urdf"])
    if not urdf_path.is_absolute():
        urdf_path = REPO_ROOT / urdf_path
    urdf_path = urdf_path.resolve()
    urdf_hash = check_hash(urdf_path, verification["source"]["urdf_sha256"], "URDF")
    require(clip_audit["urdf_sha256"] == urdf_hash
            and cfg.get("overhead", {}).get("urdf_sha256") == urdf_hash,
            "Joint-limit audit/config references a different URDF")
    require(Path(clip_audit["urdf"]).resolve() == urdf_path,
            "Joint-limit audit references a different URDF path")
    names = list(meta["robot"]["joint_names"])
    require(names == JOINTS == clip_audit["joint_names"] == verification["joint_names"],
            "Audited joint order differs from trajectory metadata")
    clip_rad = float(robot["joint_limit_clip"])
    require(math.isfinite(clip_rad) and abs(clip_rad - float(clip_audit["clip_rad"])) < 1e-12,
            "Configured joint-limit clip differs from audited clip")
    raw = read_urdf_limits(urdf_path, names)
    effective = expected_limits(raw, clip_rad)
    for label, actual, target in (
        ("audited raw limits", clip_audit["raw_limits"], raw),
        ("audited clipped limits", clip_audit["expected_limits"], effective),
        ("independently reconstructed planner limits", verification["joint_position_limits"], effective),
    ):
        bound = np.asarray([actual["lower_rad"], actual["upper_rad"]], dtype=float)
        require(bound.shape == target.shape and np.allclose(bound, target, rtol=0, atol=1e-7),
                f"{label} do not match the URDF/config")
    with np.load(npz_path, allow_pickle=False) as archive:
        q = np.asarray(archive["positions"], dtype=np.float64)
        times = np.asarray(archive["times"], dtype=np.float64)
        saved_names = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
                       for v in archive["joint_names"]]
    require(saved_names == names, "Saved trajectory joint order differs")
    require(q.ndim == 2 and q.shape[1] == len(names) and len(q) == meta["n_points"],
            "Saved trajectory shape differs from metadata")
    require(len(q) == verification["n_samples"] == clip_audit["trajectory"]["n_samples"],
            "Audited sample counts differ")
    require(meta["n_items_total"] == verification["n_items_total"]
            and meta["n_items_success"] == verification["n_items_success"],
            "Audited item counts differ")
    provenance = {
        "result": str(result),
        "trajectory_sha256": npz_hash, "metadata_sha256": meta_hash,
        "urdf": str(urdf_path), "urdf_sha256": urdf_hash,
        "independent_verification": str(verification_path),
        "independent_verification_sha256": sha256(verification_path),
        "joint_limit_clip_audit": str(clip_path),
        "joint_limit_clip_audit_sha256": sha256(clip_path),
    }
    return meta, q, times, raw, effective, clip_rad, provenance


def grid_layout(meta: dict):
    grid = meta["grid"]
    rows, cols = int(grid["rows"]), int(grid["cols"])
    require(rows >= 2 and cols >= 2 and rows * cols == meta["n_items_total"],
            "Requires a complete rectangular grid")
    x = np.linspace(*grid["x_range"], rows)
    y = np.linspace(*grid["y_range"], cols)
    require(np.all(np.isfinite(x)) and np.all(np.isfinite(y))
            and np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0),
            "Grid coordinates must be finite and ascending")
    require(len(meta["items"]) == rows * cols, "Missing item metadata for the full grid")
    seen_index, seen_cell = set(), set()
    for item in meta["items"]:
        index, row, col = int(item["index"]), int(item["row"]), int(item["col"])
        require(index not in seen_index and 0 <= index < rows * cols,
                "Duplicate/out-of-range item index")
        require((row, col) not in seen_cell and 0 <= row < rows and 0 <= col < cols,
                "Duplicate/out-of-range grid cell")
        point = np.asarray(item["position_raw"], dtype=float)
        require(point.shape == (3,) and np.isfinite(point).all()
                and np.allclose(point, [x[row], y[col], grid["z"]], rtol=0, atol=1e-8),
                f"Case {index}: recorded grasp target differs from grid coordinates")
        seen_index.add(index)
        seen_cell.add((row, col))
    return x, y


def locate_phase(case: dict, sample: int) -> str:
    phases = [stage["phase"] for stage in case["phases"]
              if stage["sample_range_half_open"][0] <= sample < stage["sample_range_half_open"][1]]
    require(len(phases) in (1, 2), "Worst sample does not belong to a stage")
    return "/".join(phases)


def metric_rows(meta: dict, q: np.ndarray, times: np.ndarray,
                raw: np.ndarray, effective: np.ndarray, clip_rad: float) -> list[dict]:
    """Reconstruct all stages and derive per-item extrema from saved samples."""
    x, y = grid_layout(meta)
    cases = case_metrics(meta, q, times, raw, effective)
    require(len(cases) == meta["n_items_total"]
            and sum(case["success"] for case in cases) == meta["n_items_success"],
            "Reconstructed case count differs from metadata")
    checked = [int(index) - 1 for index in meta["criterion"]["joints"]]
    require(checked and len(set(checked)) == len(checked)
            and all(0 <= index < 6 for index in checked), "Invalid checked joints")
    out = []
    for item, case in zip(meta["items"], cases):
        require(item["index"] == case["index"] and item["row"] == case["row"]
                and item["col"] == case["col"] and type(item["success"]) is bool
                and item["success"] == case["success"],
                "Item/case order or success flag differs")
        row = {"index": case["index"], "case_number": case["index"] + 1,
               "row": case["row"], "col": case["col"],
               "x_m": float(x[case["row"]]), "y_m": float(y[case["col"]]),
               "z_m": float(meta["grid"]["z"]), "success": case["success"],
               "status": "success" if case["success"] else "failed_or_skipped"}
        if not case["success"]:
            out.append(row)
            continue
        lo, hi = case["sample_range_half_open"]
        samples = q[lo:hi]
        raw_gap = np.minimum(samples - raw[0], raw[1] - samples)
        effective_gap = np.minimum(samples - effective[0], effective[1] - samples)
        raw_offset, raw_joint = np.unravel_index(int(np.argmin(raw_gap)), raw_gap.shape)
        eff_offset, eff_joint = np.unravel_index(int(np.argmin(effective_gap)), effective_gap.shape)
        phase_spans = np.asarray([stage["ptp_per_joint_deg"] for stage in case["phases"]])
        stage_flat = int(np.argmax(phase_spans[:, checked]))
        stage_idx, checked_idx = np.unravel_index(stage_flat, (len(PHASES), len(checked)))
        stage_joint = checked[checked_idx]
        whole_spans = np.asarray(case["full_case_ptp_per_joint_deg"])
        cycle_joint = int(np.argmax(whole_spans))
        raw_margin_deg = float(np.degrees(raw_gap[raw_offset, raw_joint]))
        eff_margin_deg = float(np.degrees(effective_gap[eff_offset, eff_joint]))
        stage_span_deg = float(phase_spans[stage_idx, stage_joint])
        cycle_span_deg = float(whole_spans[cycle_joint])
        require(abs((raw_margin_deg - eff_margin_deg) - math.degrees(clip_rad)) <= 1e-5,
                f"Case {case['index']}: raw/effective margin difference differs from clip")
        require(abs(eff_margin_deg - float(item["min_limit_margin_deg"])) <= 1e-4,
                f"Case {case['index']}: planner effective margin differs from saved samples")
        require(abs(stage_span_deg - float(item["max_joint_delta_deg"])) <= 1e-4,
                f"Case {case['index']}: planner checked-stage span differs from saved samples")
        row.update({
            "sample_start": lo, "sample_end_exclusive": hi,
            "min_raw_margin_deg": raw_margin_deg,
            "min_raw_margin_joint": JOINTS[raw_joint],
            "min_raw_margin_sample": lo + int(raw_offset),
            "min_raw_margin_stage": locate_phase(case, lo + int(raw_offset)),
            "min_effective_margin_deg": eff_margin_deg,
            "min_effective_margin_joint": JOINTS[eff_joint],
            "min_effective_margin_sample": lo + int(eff_offset),
            "min_effective_margin_stage": locate_phase(case, lo + int(eff_offset)),
            "max_cycle_joint_span_deg": cycle_span_deg,
            "max_cycle_joint": JOINTS[cycle_joint],
            "max_stage_checked_span_deg": stage_span_deg,
            "max_stage_checked_joint": JOINTS[stage_joint],
            "max_stage_checked_stage": PHASES[stage_idx],
        })
        out.append(row)
    return out


def distribution(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": int(len(values)), "min": float(values.min()),
            "median": float(np.median(values)), "max": float(values.max())}


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    step = np.diff(centers)
    return np.r_[centers[0] - step[0] / 2,
                 centers[:-1] + step / 2,
                 centers[-1] + step[-1] / 2]


def draw_map(ax, rows: list[dict], x: np.ndarray, y: np.ndarray,
             key: str, title: str, colorbar_label: str, *, categorical=False):
    data = np.full((len(y), len(x)), np.nan)
    for row in rows:
        if row["success"]:
            value = row[key]
            data[row["col"], row["row"]] = (
                JOINTS.index(value) + 1 if categorical else value)
    mask = np.ma.masked_invalid(data)
    if categorical:
        colors = ["#3878b6", "#ee7f35", "#469c72", "#b45d9b", "#c7a23a", "#6e70b7"]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(.5, 7.5), 6)
    else:
        cmap = plt.get_cmap("magma" if "span" in key else "viridis").copy()
        values = data[np.isfinite(data)]
        upper = max(1., float(values.max())) if len(values) else 1.
        norm = plt.Normalize(vmin=0., vmax=upper)
    artist = ax.pcolormesh(centers_to_edges(x), centers_to_edges(y), mask,
                           cmap=cmap, norm=norm, shading="flat", edgecolors="none")
    dx = float(np.diff(x).mean())
    dy = float(np.diff(y).mean())
    for row in rows:
        if not row["success"]:
            px, py = row["x_m"], row["y_m"]
            ax.add_patch(Rectangle((px-dx/2, py-dy/2), dx, dy,
                                   facecolor="#d4d4d4", edgecolor="white", lw=.25))
            ax.plot(px, py, marker="x", color="#af3046", markersize=3.4, mew=.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("original task_world / LINK_0 x (m)")
    ax.set_ylabel("original task_world / LINK_0 y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.linspace(x[0], x[-1], min(6, len(x))))
    ax.set_yticks(np.linspace(y[0], y[-1], min(6, len(y))))
    ax.tick_params(labelsize=8)
    ax.grid(color="white", lw=.4, alpha=.2)
    cb = plt.colorbar(artist, ax=ax, fraction=.045, pad=.02)
    if categorical:
        cb.set_ticks(range(1, 7))
        cb.set_ticklabels([name.replace("_", "") for name in JOINTS])
    cb.set_label(colorbar_label)
    return artist


def mount_tilt(config: dict) -> tuple[str | None, float | None]:
    """Preserve the physical tilt axis in plot labels and machine-readable data."""
    overhead = config.get("overhead", {})
    world = overhead.get("world_y_tilt_deg")
    local = overhead.get("local_y_tilt_deg")
    require(not (world is not None and local is not None),
            "Mount cannot declare both world-Y and local-Y tilt")
    if local is not None:
        require(overhead.get("tilt_axis") == "original_base_local_y",
                "Local-Y tilt must identify original_base_local_y axis")
        return "original_base_local_y", float(local)
    if world is not None:
        require(overhead.get("tilt_axis") in (None, "task_world_y"),
                "World-Y tilt axis metadata differs")
        return "task_world_y", float(world)
    return None, None


def write_outputs(out: Path, meta: dict, rows: list[dict], clip_rad: float,
                  provenance: dict) -> list[Path]:
    require(not out.exists(), f"Refusing to overwrite existing output directory: {out}")
    x, y = grid_layout(meta)
    success = [row for row in rows if row["success"]]
    require(success, "No successful items with saved trajectories to plot")
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if font else (lambda zh, en: en)
    mount = np.asarray(meta["config"]["robot"]["mount_transform"], dtype=float)[:3, 3]
    tilt_axis, tilt = mount_tilt(meta["config"])
    mount_text = f"LINK_0=({mount[0]:+.2f}, {mount[1]:+.2f}, {mount[2]:+.2f}) m"
    if tilt is not None:
        axis_label = ("original base local +Y" if tilt_axis == "original_base_local_y"
                      else "task_world +Y")
        mount_text += f", {axis_label}={tilt:g}°"
    heading = (f"{mount_text} | {len(success)}/{len(rows)} "
               + L("个抓放成功", "successful pick/place cases"))
    note = L("灰色 × 为规划失败/跳过，无关节样本；每成功点含完整六阶段和入场段。",
             "Gray × means failed/skipped, with no joint samples; each success includes six stages and entry.")
    stats = {key: distribution(row[key] for row in success)
             for key in ("min_raw_margin_deg", "min_effective_margin_deg",
                         "max_cycle_joint_span_deg", "max_stage_checked_span_deg")}
    report = {
        "schema_version": 1, "provenance": provenance,
        "mount_xyz_task_world_m": mount.tolist(), "tilt_axis": tilt_axis,
        "tilt_deg": tilt,
        "world_y_tilt_deg": tilt if tilt_axis == "task_world_y" else None,
        "local_y_tilt_deg": tilt if tilt_axis == "original_base_local_y" else None,
        "grid": meta["grid"], "n_total": len(rows), "n_success": len(success),
        "n_failed_or_skipped": len(rows) - len(success),
        "joint_names": JOINTS, "joint_limit_clip_rad": clip_rad,
        "joint_limit_clip_deg": math.degrees(clip_rad),
        "definitions": {
            "coordinates": "Original task_world/LINK_0 grasp target x,y; place is fixed and each plotted item uses its originating grasp position.",
            "min_raw_margin_deg": "Minimum over all saved samples of all six stages and J1–J6 of min(q-URDF_lower, URDF_upper-q), in degrees.",
            "min_effective_margin_deg": "Same minimum relative to planner limits URDF_lower+clip and URDF_upper-clip. With uniform clip this equals raw margin minus clip_deg.",
            "max_cycle_joint_span_deg": "For each J1–J6, max(q)-min(q) over complete six-stage pick/place item; take largest joint. Bounded angles are not wrapped modulo 360.",
            "max_stage_checked_span_deg": "For each of six external stages and configured checked joints, max(q)-min(q) within that stage; take largest stage/joint. This is the planner joint-span criterion, not the full-cycle span.",
            "failed_or_skipped": "No saved samples; metrics absent/null, never zero. A planning failure does not prove geometric unreachability.",
            "scope": "Saved discrete samples only; passed source audits cover those samples and independent FK. No between-sample collision or physical installation guarantee.",
        },
        "statistics_deg": stats, "cases": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    created = []
    with tempfile.TemporaryDirectory(prefix=".joint_distributions_", dir=out.parent) as scratch:
        staging = Path(scratch) / "data"
        staging.mkdir()
        for key, filename, zh, en in (*MAIN_METRICS, STAGE_METRIC):
            fig, ax = plt.subplots(figsize=(10.5, 8.2), constrained_layout=True)
            draw_map(ax, rows, x, y, key, L(zh, en), L("角度 (°)", "Degrees (°)"))
            fig.suptitle(heading, fontsize=12)
            fig.text(.5, .012, note, ha="center", fontsize=8)
            fig.savefig(staging / filename, dpi=190)
            plt.close(fig)
            created.append(filename)
        fig, ax = plt.subplots(figsize=(10.5, 8.2), constrained_layout=True)
        draw_map(ax, rows, x, y, "min_effective_margin_joint",
                 L("距离有效限位最近的关节", "Joint nearest its clipped limit"),
                 L("关节编号", "Joint ID"), categorical=True)
        fig.suptitle(heading, fontsize=12)
        fig.text(.5, .012, note, ha="center", fontsize=8)
        filename = "nearest_limit_joint.png"
        fig.savefig(staging / filename, dpi=190)
        plt.close(fig)
        created.append(filename)
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
        for ax, (key, _, zh, en) in zip(axes.flat, (*MAIN_METRICS, STAGE_METRIC)):
            draw_map(ax, rows, x, y, key, L(zh, en), L("角度 (°)", "Degrees (°)"))
        fig.suptitle(heading, fontsize=16)
        fig.text(.5, .005, note, ha="center", fontsize=10)
        filename = "joint_distribution_comparison.png"
        fig.savefig(staging / filename, dpi=170)
        plt.close(fig)
        created.append(filename)
        with (staging / "joint_distribution_cases.csv").open("x", newline="", encoding="utf-8") as stream:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        created.append("joint_distribution_cases.csv")
        (staging / "joint_distribution_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        created.append("joint_distribution_summary.json")
        os.rename(staging, out)
    return [out / name for name in created]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    result = args.result.resolve()
    out = (args.out_dir or result / "joint_distributions").resolve()
    require(not out.exists(), f"Refusing to overwrite existing output directory: {out}")
    meta, q, times, raw, effective, clip_rad, provenance = verified_inputs(result)
    rows = metric_rows(meta, q, times, raw, effective, clip_rad)
    for path in write_outputs(out, meta, rows, clip_rad, provenance):
        print(f"[ok] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
