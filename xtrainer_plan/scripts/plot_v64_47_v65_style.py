#!/usr/bin/env python3
"""Draw the V66 v64_47 row as a V65-style 2x2 figure, without replanning.

The source is the four independently audited V66 runs.  Every plotted cell is
checked against the published V66 comparison CSV; metric color limits are the
same four-candidate limits used by ``full_candidate_comparison.png``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np

from compare_full_mount_candidates import NAMES, METRICS, load_candidate, matched_data
from plot_full_mount_joint_distributions import centers_to_edges, require, sha256
from plot_grasp_angle_map import pick_cjk_font


NAME = "v64_47"
METRIC_TITLES = {
    "min_raw_margin_deg": ("原始 URDF 限位：全程最短距离", "Raw URDF limits: minimum full-cycle distance"),
    "min_effective_margin_deg": ("规划器有效限位：全程最小余量", "Planner clipped limits: minimum full-cycle margin"),
    "max_cycle_joint_span_deg": ("完整抓放：单关节最大角跨度", "Full pick/place: largest single-joint angular span"),
}


def check_published_csv(path: Path, records: list[dict]) -> None:
    """Make the pixels traceable to exactly the data in the V66 overview."""
    require(path.is_file(), f"Missing V66 comparison CSV: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(records[0]), "Published CSV columns differ")
        published = list(reader)
    require(len(published) == len(records) == 400, "Expected 400 same-point rows")
    for expected, saved in zip(records, published):
        for key, value in expected.items():
            actual = saved[key]
            if type(value) is bool:
                require(actual == str(value), f"CSV case {expected['case_number']} {key} differs")
            elif type(value) is int:
                require(actual == str(value), f"CSV case {expected['case_number']} {key} differs")
            elif type(value) is float:
                require(math.isfinite(float(actual)) and math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-10),
                        f"CSV case {expected['case_number']} {key} differs")
            else:
                require(actual == str(value), f"CSV case {expected['case_number']} {key} differs")


def draw_failed_marks(ax, rows: list[dict], dx: float, dy: float) -> None:
    for case in rows:
        if case[f"{NAME}_success"]:
            continue
        x, y = case["x_m"], case["y_m"]
        ax.add_patch(Rectangle((x-dx/2, y-dy/2), dx, dy,
                               facecolor="#d4d4d4", edgecolor="white", lw=.25))
        ax.plot(x, y, marker="x", color="#af3046", markersize=3.4, mew=.8)


def draw(root: Path, out: Path) -> None:
    require(not out.exists(), f"Refusing to overwrite: {out}")
    candidates = [load_candidate(root, name) for name in NAMES]
    records, summary = matched_data(candidates)
    source_csv = root / "reports/comparison/same_point_metrics.csv"
    check_published_csv(source_csv, records)
    candidate = next(c for c in candidates if c["name"] == NAME)
    require(candidate["joint"]["n_success"] == 376
            and sum(row[f"{NAME}_success"] for row in records) == 376,
            "v64_47 success count differs from audited 376/400")
    require(sum(not row[f"{NAME}_success"] for row in records) == 24,
            "v64_47 gray-cell count differs from 24")
    require(candidate["joint"]["tilt_axis"] == "original_base_local_y",
            "Expected original-base local +Y tilt")

    grid = summary["grid"]
    xs = np.linspace(*grid["x_range"], grid["rows"])
    ys = np.linspace(*grid["y_range"], grid["cols"])
    xe, ye = centers_to_edges(xs), centers_to_edges(ys)
    dx, dy = float(np.diff(xs).mean()), float(np.diff(ys).mean())
    limits = {}
    for key, *_ in METRICS:
        values = [float(row[f"{name}_{key}"]) for row in records
                  for name in NAMES if row[f"{name}_success"]]
        require(values and all(math.isfinite(value) for value in values), f"No valid {key} values")
        limits[key] = [min(values), max(values)]

    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if font else (lambda zh, en: en)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    coverage = np.zeros((len(ys), len(xs)), dtype=int)
    for row in records:
        coverage[int(row["col"]), int(row["row"])] = int(row[f"{NAME}_success"])
    ax = axes[0, 0]
    ax.pcolormesh(xe, ye, coverage,
                  cmap=ListedColormap(["#d4d4d4", "#45b7ab"]),
                  vmin=0, vmax=1, shading="flat")
    draw_failed_marks(ax, records, dx, dy)
    ax.set_title(L("完整抓放成功分布", "Complete pick/place coverage"), fontsize=12)

    for ax, (key, _, _, palette) in zip((axes[0, 1], axes[1, 0], axes[1, 1]), METRICS):
        data = np.full((len(ys), len(xs)), np.nan)
        for row in records:
            if row[f"{NAME}_success"]:
                data[int(row["col"]), int(row["row"])] = row[f"{NAME}_{key}"]
        cmap = plt.get_cmap(palette).copy()
        cmap.set_bad("#d4d4d4")
        artist = ax.pcolormesh(xe, ye, np.ma.masked_invalid(data), cmap=cmap,
                               norm=Normalize(*limits[key]), shading="flat", edgecolors="none")
        draw_failed_marks(ax, records, dx, dy)
        ax.set_title(L(*METRIC_TITLES[key]), fontsize=12)
        cb = fig.colorbar(artist, ax=ax, fraction=.045, pad=.02)
        cb.set_label(L("角度 (°)", "Degrees (°)"))

    for ax in axes.flat:
        ax.set_xlabel("original task_world / LINK_0 x (m)")
        ax.set_ylabel("original task_world / LINK_0 y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.linspace(xs[0], xs[-1], 6))
        ax.set_yticks(np.linspace(ys[0], ys[-1], 6))
        ax.tick_params(labelsize=8)
        ax.grid(color="white", lw=.4, alpha=.2)
    base = candidate["joint"]["mount_xyz_task_world_m"]
    tilt = candidate["joint"]["tilt_deg"]
    heading = (f"{NAME} | LINK_0=({base[0]:+.2f}, {base[1]:+.2f}, {base[2]:+.2f}) m, "
               f"original base local +Y={tilt:g}° | 376/400 "
               + L("个抓放成功", "successful pick/place cases"))
    fig.suptitle(heading, fontsize=16)
    note = L("灰色 ×：规划失败/跳过，无关节样本；关节指标色标沿用 V66 四组总图。",
             "Gray ×: failed/skipped, no joint samples; joint-metric scales match the V66 four-mount overview.")
    fig.text(.5, .005, note, ha="center", fontsize=10)

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v64_47_v65_style_", dir=out.parent) as scratch:
        stage = Path(scratch) / "data"
        stage.mkdir()
        filename = "v64_47_v65_style_four_metrics.png"
        fig.savefig(stage / filename, dpi=170, facecolor="white")
        plt.close(fig)
        metadata = {
            "candidate": NAME, "n_total": 400, "n_success": 376, "n_failed": 24,
            "source_comparison_csv": str(source_csv),
            "source_comparison_csv_sha256": sha256(source_csv),
            "candidate_trajectory_sha256": candidate["joint"]["provenance"]["trajectory_sha256"],
            "metric_limits_across_all_four_candidates_deg": limits,
            "candidate_names_for_color_limits": list(NAMES),
            "source_cell_validation": "All 400 rows and every CSV field match independently audited V66 candidate reports.",
        }
        (stage / "plot_data_summary.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "README.md").write_text(
            "# v64_47 四项分布图（V65 布局）\n\n"
            "`v64_47_v65_style_four_metrics.png` 将 V66 四组总图中 **v64_47** 的一行单独绘成与 V65 最终关节分布图一致的 2×2 布局。"
            "四格依次为完整抓放成功分布、距原始 URDF 关节限位最近距离、规划有效限位最小余量、完整抓放单关节最大角跨度。"
            "基座 `(-0.10, 0.15, 0.65) m`，绕原始基座局部 `+Y` 轴转 `60°`；成功 `376/400`，灰色红叉 `24/400`。\n\n"
            "**来源和口径：**图中每格来自 V66 `reports/comparison/same_point_metrics.csv` 对应的 `v64_47_*` 列；"
            "脚本还逐字段对照四组已通过独立轨迹及关节限位审核的报告。失败格无保存轨迹，关节指标为空，不能当作零。"
            "关节指标使用 V66 四组总图的共同色标范围（见 `plot_data_summary.json`）；"
            "V65 现有图按自身数据单独定色标，所以两图色深不能直接比较。指标只覆盖已保存的离散轨迹采样。\n\n"
            "**名称区别：**此图是 `v64_47`，不是 `v64_11`。`v64_11` 是 V65_00 全量实验所采用的布置，"
            "基座 `(-0.20, 0.15, 0.65) m`，同为局部 `+Y 60°`。\n\n"
            "复现：在仓库根目录运行\n\n"
            "```bash\n"
            "python3 xtrainer_plan/scripts/plot_v64_47_v65_style.py "
            "xtrainer_plan/results_overhead/20260928/v66_v64_8of9_local_y_full "
            "--out-dir /tmp/v64_47_v65_style_rebuild\n"
            "```\n",
            encoding="utf-8")
        os.rename(stage, out)
    print(f"[ok] {out / filename}: 376 successes, 24 failed, V66 CSV matched")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="V66 root containing runs/ and reports/")
    parser.add_argument("--out-dir", type=Path, help="New output directory; existing directories are never overwritten")
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out_dir or root / "reports" / NAME / "comparison_v65_style").resolve()
    draw(root, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
