#!/usr/bin/env python3
"""Draw V65 raw joint-limit distance in the v64_47 per-cell map style.

Run from the repository root::

    python3 xtrainer_plan/scripts/plot_v65_raw_limit_distance_cell_map.py

Use --out /tmp/v65_raw_cell_map.png to reproduce without replacing the published PNG.
The figure is based on the published, independently audited V65 per-case CSV/JSON;
it does not rerun planning or alter the original distribution plot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

from plot_grasp_angle_map import pick_cjk_font


ROOT = Path(__file__).resolve().parents[1] / "results_overhead/20260928/v65_near_zero_x_local_ytilt60_full"
METRIC = "min_raw_margin_deg"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contrast_color(rgba: tuple[float, ...]) -> str:
    rgb = np.asarray(rgba[:3], dtype=float)
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = float(np.dot(linear, [.2126, .7152, .0722]))
    return "#16202b" if luminance > .30 else "#ffffff"


def load_checked_data(root: Path) -> tuple[list[dict], dict, dict]:
    report_dir = root / "joint_distributions"
    csv_path = report_dir / "joint_distribution_cases.csv"
    summary = json.loads((report_dir / "joint_distribution_summary.json").read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    cases = summary["cases"]
    grid = summary["grid"]
    require(grid["rows"] == grid["cols"] == 20 and len(cases) == len(csv_rows) == 400,
            "Expected 400 V65 cells on a 20×20 grid")
    require(summary["n_total"] == 400 and summary["n_success"] == 383
            and summary["n_failed_or_skipped"] == 17, "Unexpected published V65 counts")
    require(summary["tilt_axis"] == "original_base_local_y"
            and math.isclose(summary["tilt_deg"], 60., abs_tol=1e-12)
            and np.allclose(summary["mount_xyz_task_world_m"], [-.2, .15, .65], rtol=0, atol=1e-12),
            "Unexpected V65 mount")

    provenance = summary["provenance"]
    for name, recorded in (("trajectory.npz", provenance["trajectory_sha256"]),
                           ("trajectory_meta.json", provenance["metadata_sha256"])):
        path = root / "runs/v65_00" / name
        require(sha256(path) == recorded, f"Published V65 {name} hash has changed")
    for label in ("independent_verification", "joint_limit_clip_audit"):
        path = Path(provenance[label])
        require(sha256(path) == provenance[f"{label}_sha256"],
                f"Published V65 {label} hash has changed")
        audit = json.loads(path.read_text(encoding="utf-8"))
        require(audit.get("passed") is True and audit.get("verification_completed") is True,
                f"Published V65 {label} did not pass")
    meta = json.loads((root / "runs/v65_00/trajectory_meta.json").read_text(encoding="utf-8"))
    require(meta["n_items_total"] == 400 and meta["n_items_success"] == 383,
            "V65 metadata count differs from published report")
    mount = np.asarray(meta["config"]["robot"]["mount_transform"], dtype=float)
    require(mount.shape == (4, 4)
            and np.allclose(mount[:3, 3], summary["mount_xyz_task_world_m"], atol=1e-12),
            "V65 mount differs from metadata")
    place = meta["place_position_raw"]
    require(np.allclose(place, [-.36, -.12, .10], rtol=0, atol=1e-12),
            "Unexpected V65 fixed place position")

    xs = np.linspace(*grid["x_range"], 20)
    ys = np.linspace(*grid["y_range"], 20)
    seen = set()
    values = []
    success_count = 0
    for case, csv_row in zip(cases, csv_rows):
        index, row, col = (int(case[key]) for key in ("index", "row", "col"))
        require((row, col) not in seen and 0 <= row < 20 and 0 <= col < 20,
                f"Duplicate/out-of-range V65 cell {index}")
        seen.add((row, col))
        require(index == int(csv_row["index"]) and index + 1 == int(csv_row["case_number"])
                and row == int(csv_row["row"]) and col == int(csv_row["col"]),
                f"V65 CSV and JSON case index differ at {index}")
        for key, target in (("x_m", xs[row]), ("y_m", ys[col]), ("z_m", grid["z"])):
            require(math.isclose(float(case[key]), float(csv_row[key]), abs_tol=1e-12)
                    and math.isclose(float(case[key]), float(target), abs_tol=1e-8),
                    f"V65 {key} differs at case {index}")
        success = case["success"]
        require(type(success) is bool and csv_row["success"] == str(success),
                f"V65 success status differs at case {index}")
        if success:
            value = float(case[METRIC])
            require(math.isfinite(value) and value >= 0
                    and math.isclose(value, float(csv_row[METRIC]), rel_tol=0, abs_tol=1e-10),
                    f"V65 raw margin differs at case {index}")
            values.append(value)
            success_count += 1
        else:
            require(METRIC not in case and csv_row[METRIC] == "",
                    f"Failed V65 case {index} has a margin value")
    require(len(seen) == 400 and success_count == 383 and len(values) == 383,
            "Unexpected V65 cell coverage")
    stats = summary["statistics_deg"][METRIC]
    require(stats["count"] == 383
            and np.allclose([min(values), np.median(values), max(values)],
                            [stats["min"], stats["median"], stats["max"]], rtol=0, atol=1e-10),
            "V65 metric distribution differs from the published summary")
    return cases, summary, meta


def draw(cases: list[dict], summary: dict, meta: dict, output: Path) -> None:
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")
    grid = summary["grid"]
    xs = np.linspace(*grid["x_range"], 20)
    ys = np.linspace(*grid["y_range"], 20)
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    values = np.asarray([case[METRIC] for case in cases if case["success"]], dtype=float)
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = plt.get_cmap("viridis")
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(16.8, 14.2), dpi=240)
    fig.subplots_adjust(left=.105, right=.82, bottom=.115, top=.88)
    for case in cases:
        x, y = float(case["x_m"]), float(case["y_m"])
        if case["success"]:
            value = float(case[METRIC])
            rgba = cmap(norm(value))
            fill, label, color, edge = rgba, f"{value:.1f}", contrast_color(rgba), "#ffffff"
            line_width = .38
        else:
            fill, label, color, edge = "#fbfaf8", "×", "#d4383a", "#e1464b"
            line_width = .85
        ax.add_patch(Rectangle((x - dx/2, y - dy/2), dx, dy,
                               facecolor=fill, edgecolor=edge, linewidth=line_width, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=6.2,
                color=color, zorder=3)

    base = summary["mount_xyz_task_world_m"]
    place = meta["place_position_raw"]
    # V65 sits on the grid's right boundary and halfway between two Y rows.
    # A smaller marker preserves the neighbouring numeric cell labels.
    ax.scatter([base[0]], [base[1]], marker="*", s=110, facecolor="#f2ae00",
               edgecolor="#1e2530", linewidth=1.1, zorder=6)
    ax.scatter([place[0]], [place[1]], marker="X", s=150, facecolor="#d32971",
               edgecolor="#211b27", linewidth=1.2, zorder=6)
    ax.scatter([0], [0], marker="o", s=105, facecolor="#f8f8f8",
               edgecolor="#454a54", linewidth=1.5, zorder=6)
    ax.scatter([0], [0], marker="+", s=63, color="#454a54",
               linewidth=1.1, zorder=7)

    ax.set_xlim(-.655, .057)
    ax.set_ylim(-.16, .445)
    ax.set_aspect("equal", adjustable="box")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(.02))
        axis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=6.8)
    ax.tick_params(axis="y", labelsize=7.1)
    ax.grid(color="#cbd2d9", linewidth=.43, linestyle=":", alpha=.68, zorder=0)
    ax.set_xlabel("task_world / 原始 LINK_0  X (m)", fontsize=11)
    ax.set_ylabel("task_world / 原始 LINK_0  Y (m)", fontsize=11)
    fig.suptitle("V65 / v65_00  ·  原始 URDF 关节限位最近距离\n"
                 "整条轨迹 · J1–J6 全关节最小值\n"
                 f"数值范围 {values.min():.2f}–{values.max():.2f}°  |  成功 383/400",
                 fontsize=15.5, y=.97)
    handles = [
        Line2D([], [], marker="*", linestyle="None", markerfacecolor="#f2ae00",
               markeredgecolor="#1e2530", markersize=12,
               label="当前 V65 安装基座 XY 投影  (-0.20, +0.15) m；Z = 0.65 m"),
        Line2D([], [], marker="X", linestyle="None", markerfacecolor="#d32971",
               markeredgecolor="#211b27", markersize=9,
               label="固定 place XY 投影  (-0.36, -0.12) m；Z = 0.10 m"),
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="white",
               markeredgecolor="#454a54", markersize=8,
               label="原始 LINK_0 基座 / task_world 原点  (0, 0)"),
        Patch(facecolor="#fbfaf8", edgecolor="#e1464b", label="失败/跳过：无保存轨迹"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              facecolor="white", framealpha=.98, edgecolor="#b3bac2")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=.046, pad=.034, shrink=.85)
    cb.set_label("角度 (°)", fontsize=11)
    cb.ax.tick_params(labelsize=9)
    fig.text(.50, .055,
             "grasp 网格 Z = 0.03 m；颜色只表示当前图的指标。失败格无关节数值，红叉不代表已证明几何不可达。",
             ha="center", fontsize=9, color="#46505b")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="V65 result root")
    parser.add_argument("--out", type=Path, help="New PNG path")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.out or root / "joint_distributions/min_raw_limit_distance_cell_map.png").resolve()
    cases, summary, meta = load_checked_data(root)
    draw(cases, summary, meta, output)
    print(f"[ok] {output}: 400 cells, 383 success, 17 failed, "
          f"{summary['statistics_deg'][METRIC]['min']:.4f}–"
          f"{summary['statistics_deg'][METRIC]['max']:.4f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
