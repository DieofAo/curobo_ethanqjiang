#!/usr/bin/env python3
"""Draw audited 20x20 per-cell maps for any one full XTrainer mount.

Four maps match the V66/v64_47 cell-map layout.  --task-style also writes a
binary success map in the V65 task-map layout.  This script only reads saved
full-run reports; it neither replans nor launches ROS.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

from compare_full_mount_candidates import load_candidate
from plot_full_mount_joint_distributions import require, sha256
from plot_grasp_angle_map import pick_cjk_font


METRICS = (
    ("min_raw_margin_deg", "raw_limit_distance_cell_map.png",
     "原始 URDF 关节限位最近距离", "整条轨迹 · J1–J6 全关节最小值", "viridis"),
    ("min_effective_margin_deg", "effective_limit_margin_cell_map.png",
     "规划器有效关节限位最小余量", "整条轨迹 · J1–J6 全关节最小值", "cividis"),
    ("max_cycle_joint_span_deg", "max_joint_span_cell_map.png",
     "完整抓放周期单关节最大角跨度", "每条轨迹取 J1–J6 中最大的 max(q)−min(q)", "RdYlBu_r"),
)


def source_data(root: Path, name: str) -> tuple[dict, list[dict], Path]:
    require(bool(re.fullmatch(r"[A-Za-z0-9_]+", name)), "Invalid candidate name")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = {row["name"]: row for row in manifest["candidates"]}
    require(name in entries, f"Candidate not in manifest: {name}")
    require(Path(entries[name]["result"]).resolve() == root / "runs" / name,
            "Manifest run path differs")
    candidate = load_candidate(root, name)
    joint, scene = candidate["joint"], candidate["scene"]
    require(joint["grid"] == scene["grid"], "Joint and scene grasp grids differ")
    grid = joint["grid"]
    require(int(grid["rows"]) == int(grid["cols"]) == 20,
            "Expected the audited 20x20 grasp grid")
    cases = [candidate["rows"][index] for index in range(400)]
    xs = np.linspace(*grid["x_range"], 20)
    ys = np.linspace(*grid["y_range"], 20)
    for case in cases:
        r, c = int(case["row"]), int(case["col"])
        require(0 <= r < 20 and 0 <= c < 20, "Invalid grid cell index")
        require(np.allclose([case["x_m"], case["y_m"], case["z_m"]],
                            [xs[r], ys[c], grid["z"]], atol=1e-9, rtol=0),
                f"Case {case['case_number']} coordinates differ from grid")
    csv_path = root / "reports" / name / "joint_distributions" / "joint_distribution_cases.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, "Missing joint CSV header")
        csv_rows = list(reader)
    require(len(csv_rows) == 400, "Joint CSV must contain exactly 400 cases")
    for case, row in zip(cases, csv_rows):
        for key in reader.fieldnames:
            value = case.get(key)
            actual = row[key]
            if value is None:
                require(actual == "", f"Case {case['case_number']}: unexpected CSV {key}")
            elif isinstance(value, bool):
                require(actual == str(value), f"Case {case['case_number']}: CSV {key} differs")
            elif isinstance(value, float):
                require(actual != "" and math.isclose(float(actual), value, abs_tol=1e-9, rel_tol=1e-10),
                        f"Case {case['case_number']}: CSV {key} differs")
            else:
                require(actual == str(value), f"Case {case['case_number']}: CSV {key} differs")
    require(sum(case["success"] is True for case in cases) == joint["n_success"] == scene["n_success"],
            "Success count differs across audited sources")
    return candidate, cases, csv_path


def contrast_color(rgba: tuple[float, ...]) -> str:
    rgb = np.asarray(rgba[:3], dtype=float)
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = float(np.dot(linear, [.2126, .7152, .0722]))
    return "#16202b" if luminance > .30 else "#ffffff"


def bounds(cases: list[dict], base: list[float], place: list[float], dx: float, dy: float):
    xs = [case["x_m"] for case in cases]
    ys = [case["y_m"] for case in cases]
    return (min(min(xs)-dx/2, base[0], place[0], 0)-.04,
            max(max(xs)+dx/2, base[0], place[0], 0)+.055,
            min(min(ys)-dy/2, base[1], place[1], 0)-.04,
            max(max(ys)+dy/2, base[1], place[1], 0)+.04)


def markers(ax, base: list[float], place: list[float], *, task_style: bool = False) -> list:
    if task_style:
        ax.scatter([0], [0], marker="+", s=180, linewidths=2,
                   color="#d4a600", zorder=7)
        ax.scatter([base[0]], [base[1]], marker="v", s=190,
                   facecolors="none", edgecolors="#1565c0", linewidths=2, zorder=8)
        ax.annotate(f"实际基座\nz={base[2]:.2f}m", xy=base[:2], xytext=(6, 7),
                    textcoords="offset points", fontsize=7.5, color="#0d47a1", zorder=9)
        ax.scatter([place[0]], [place[1]], marker="X", s=125,
                   color="#d81b60", edgecolors="black", linewidths=.7, zorder=8)
        return [
            Line2D([0], [0], marker="v", ls="none", markersize=10,
                   markerfacecolor="none", markeredgecolor="#1565c0", markeredgewidth=2,
                   label="实际基座投影"),
            Line2D([0], [0], marker="+", ls="none", markersize=11,
                   markeredgewidth=2, color="#d4a600", label="原始坐标原点"),
            Line2D([0], [0], marker="X", ls="none", markersize=9,
                   markerfacecolor="#d81b60", markeredgecolor="black", label="固定 place"),
        ]
    ax.scatter([base[0]], [base[1]], marker="*", s=270, facecolor="#f2ae00",
               edgecolor="#1e2530", linewidth=1.4, zorder=6)
    ax.scatter([place[0]], [place[1]], marker="X", s=150, facecolor="#d32971",
               edgecolor="#211b27", linewidth=1.2, zorder=6)
    ax.scatter([0], [0], marker="o", s=105, facecolor="#f8f8f8",
               edgecolor="#454a54", linewidth=1.5, zorder=6)
    ax.scatter([0], [0], marker="+", s=63, color="#454a54", linewidth=1.1, zorder=7)
    return [
        Line2D([], [], marker="*", ls="none", markerfacecolor="#f2ae00",
               markeredgecolor="#1e2530", markersize=12,
               label=f"当前安装基座  ({base[0]:+.2f}, {base[1]:+.2f}, {base[2]:+.2f}) m"),
        Line2D([], [], marker="X", ls="none", markerfacecolor="#d32971",
               markeredgecolor="#211b27", markersize=9,
               label=f"固定 place  ({place[0]:+.2f}, {place[1]:+.2f}, {place[2]:+.2f}) m"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="white",
               markeredgecolor="#454a54", markersize=8,
               label="原始 LINK_0 基座 / task_world 原点  (0, 0)"),
    ]


def draw_one(name: str, candidate: dict, cases: list[dict], key: str | None,
             title: str, subtitle: str, palette: str | None, output: Path) -> dict:
    grid = candidate["joint"]["grid"]
    dx = (grid["x_range"][1] - grid["x_range"][0]) / 19
    dy = (grid["y_range"][1] - grid["y_range"][0]) / 19
    success = [case for case in cases if case["success"]]
    values = np.asarray([case[key] for case in success], dtype=float) if key else None
    if key:
        require(len(values) == candidate["joint"]["n_success"] and np.isfinite(values).all(),
                "Invalid success metric values")
        low, high = float(values.min()), float(values.max())
        norm = Normalize(vmin=low if low < high else low-1,
                         vmax=high if low < high else high+1)
        cmap = plt.get_cmap(palette)
    fig, ax = plt.subplots(figsize=(16.8, 14.2), dpi=240)
    fig.subplots_adjust(left=.105, right=.82, bottom=.115, top=.88)
    for case in cases:
        x, y = case["x_m"], case["y_m"]
        if not case["success"]:
            fill, label, color, edge, width = "#fbfaf8", "×", "#d4383a", "#e1464b", .85
        elif key:
            value = float(case[key])
            fill = cmap(norm(value))
            label, color, edge, width = f"{value:.1f}", contrast_color(fill), "white", .38
        else:
            fill, label, color, edge, width = "#157f86", "✓", "white", "white", .38
        ax.add_patch(Rectangle((x-dx/2, y-dy/2), dx, dy, facecolor=fill,
                               edgecolor=edge, linewidth=width, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=6.2,
                color=color, zorder=3)
    base = candidate["joint"]["mount_xyz_task_world_m"]
    place = candidate["scene"]["place_position_m"]
    handles = markers(ax, base, place)
    ax.set_xlim(*bounds(cases, base, place, dx, dy)[:2])
    ax.set_ylim(*bounds(cases, base, place, dx, dy)[2:])
    ax.set_aspect("equal", adjustable="box")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(.02))
        axis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=6.8)
    ax.tick_params(axis="y", labelsize=7.1)
    ax.grid(color="#cbd2d9", linewidth=.43, linestyle=":", alpha=.68, zorder=0)
    ax.set_xlabel("task_world / 原始 LINK_0  X (m)", fontsize=11)
    ax.set_ylabel("task_world / 原始 LINK_0  Y (m)", fontsize=11)
    n_success = len(success)
    n_total = len(cases)
    range_text = (f"数值范围 {values.min():.2f}–{values.max():.2f}°  |  成功 {n_success}/{n_total}"
                  if key else f"完整抓放成功 {n_success}/{n_total}  |  失败或跳过 {n_total-n_success}/{n_total}")
    ax_tilt = candidate["joint"]["tilt_deg"]
    fig.suptitle(f"{name}  ·  {title}\n{subtitle}\n{range_text}  |  局部 +Y {ax_tilt:g}°",
                 fontsize=15.5, y=.97)
    handles.append(Patch(facecolor="#fbfaf8", edgecolor="#e1464b",
                         label="失败/跳过：无保存轨迹"))
    if key:
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                                fraction=.046, pad=.034, shrink=.85)
        colorbar.set_label("角度 (°)", fontsize=11)
        colorbar.ax.tick_params(labelsize=9)
    else:
        handles.insert(0, Patch(facecolor="#157f86", edgecolor="white",
                                label="成功：有完整轨迹"))
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              facecolor="white", framealpha=.98, edgecolor="#b3bac2")
    fig.text(.50, .055,
             f"grasp 网格 Z = {grid['z']:.3f} m；颜色只表示当前图的指标。失败格无关节数值，红叉不代表已证明几何不可达。",
             ha="center", fontsize=9, color="#46505b")
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)
    if key:
        return {"minimum_deg": float(values.min()), "maximum_deg": float(values.max()),
                "cell_labels": [f"{float(case[key]):.1f}" if case["success"] else "×"
                                for case in cases]}
    return {"n_success": n_success, "n_failed": n_total-n_success}


def draw_task_style(name: str, candidate: dict, cases: list[dict], output: Path) -> None:
    grid = candidate["joint"]["grid"]
    dx = (grid["x_range"][1] - grid["x_range"][0]) / 19
    dy = (grid["y_range"][1] - grid["y_range"][0]) / 19
    base = candidate["joint"]["mount_xyz_task_world_m"]
    place = candidate["scene"]["place_position_m"]
    fig, ax = plt.subplots(figsize=(8.8, 7.5), constrained_layout=True)
    for case in cases:
        x, y = case["x_m"], case["y_m"]
        ok = case["success"]
        ax.add_patch(Rectangle((x-dx*.43, y-dy*.43), dx*.86, dy*.86,
                               facecolor="#4053ba" if ok else "#e1e1e1",
                               edgecolor="#303030" if ok else "#c62828",
                               linewidth=.45 if ok else .9, zorder=2))
        if not ok:
            ax.text(x, y, "×", ha="center", va="center", fontsize=5.25,
                    fontweight="bold", color="#b2182b", zorder=4)
    handles = markers(ax, base, place, task_style=True)
    ax.set_xlim(*bounds(cases, base, place, dx, dy)[:2])
    ax.set_ylim(*bounds(cases, base, place, dx, dy)[2:])
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(.02))
    ax.yaxis.set_major_locator(MultipleLocator(.02))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=5.5)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.grid(True, which="major", linestyle=":", linewidth=.5, alpha=.58)
    ax.set_axisbelow(True)
    ax.set_xlabel("task_world x (m)")
    ax.set_ylabel("task_world y (m)")
    n_success = candidate["joint"]["n_success"]
    ax.set_title(f"{name}\n成功 {n_success}/400  失败 {400-n_success}\n"
                 f"mount xyz=({base[0]:+.2f}, {base[1]:+.2f}, {base[2]:+.2f}) m  "
                 f"原始基座局部 +Y 倾角 {candidate['joint']['tilt_deg']:g}°  "
                 f"grid z={grid['z']:.3f} m", fontsize=10.2, pad=8)
    fig.suptitle("XTrainer 单臂规划成功分布（保留的 task/world 坐标系）", fontsize=14)
    colorbar = fig.colorbar(ScalarMappable(norm=BoundaryNorm([-.5, .5, 1.5], 2),
                                           cmap=ListedColormap(["#e1e1e1", "#4053ba"])),
                            ax=ax, fraction=.022, pad=.015, shrink=.84, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["无完整轨迹", "完整轨迹"])
    colorbar.set_label("规划结果")
    fig.legend(handles=[Patch(facecolor="#4053ba", edgecolor="#303030", label="规划成功"),
                        Patch(facecolor="#e1e1e1", edgecolor="#c62828", label="无完整轨迹"),
                        *handles], loc="lower center", bbox_to_anchor=(.5, -.055),
               ncol=5, fontsize=8.2, framealpha=.94)
    fig.text(.5, -.095,
             "按记录顺序连续抓放；红叉表示无保存的完整轨迹，不证明几何不可达。刻度间隔 2 cm，不代表采样间隔。",
             ha="center", va="top", fontsize=8, color="#555555")
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw(root: Path, name: str, out: Path, task_style: bool) -> None:
    require(not out.exists(), f"Refusing to overwrite existing output: {out}")
    candidate, cases, csv_path = source_data(root, name)
    joint, scene = candidate["joint"], candidate["scene"]
    n_success = joint["n_success"]
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}_cell_maps_", dir=out.parent) as scratch:
        stage = Path(scratch) / "data"
        stage.mkdir()
        figures = {"success_cell_map.png": draw_one(
            name, candidate, cases, None, "grasp 区域成功分布",
            "20×20 目标网格 · 绿色成功 / 红叉失败", None,
            stage / "success_cell_map.png")}
        for key, filename, title, subtitle, palette in METRICS:
            figures[filename] = draw_one(name, candidate, cases, key, title, subtitle,
                                         palette, stage / filename)
        if task_style:
            draw_task_style(name, candidate, cases, stage / "success_task_style.png")
            figures["success_task_style.png"] = {"n_success": n_success,
                                                  "n_failed": 400-n_success}
        audit = {
            "candidate": name, "source_joint_csv": str(csv_path),
            "source_joint_csv_sha256": sha256(csv_path),
            "source_joint_json_sha256": sha256(root / "reports" / name / "joint_distributions" /
                                               "joint_distribution_summary.json"),
            "source_scene_json_sha256": sha256(root / "reports" / name / "scene_distribution" /
                                               "task_scene_distribution.json"),
            "trajectory_sha256": joint["provenance"]["trajectory_sha256"],
            "independent_verification_passed": True,
            "joint_limit_clip_audit_passed": True,
            "n_grid_cells": 400, "n_success": n_success,
            "n_failed_or_skipped": 400-n_success,
            "mount_task_world_m": joint["mount_xyz_task_world_m"],
            "original_base_xy_task_world_m": [0.0, 0.0],
            "place_task_world_m": scene["place_position_m"],
            "tilt_axis": joint["tilt_axis"], "tilt_deg": joint["tilt_deg"],
            "grid": joint["grid"], "figures": figures,
            "figure_sha256": {filename: sha256(stage / filename) for filename in figures},
        }
        (stage / "plot_data_summary.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        clip = joint["joint_limit_clip_rad"]
        readme = [
            f"# {name} 逐格分布图", "",
            "四张 20×20 图片：`success_cell_map.png`、`raw_limit_distance_cell_map.png`、"
            "`effective_limit_margin_cell_map.png`、`max_joint_span_cell_map.png`。"
            "三张关节图的成功格标注一位小数（度）；成功分布图用勾号；无完整轨迹格用浅色红框和红叉。",
            f"当前安装基座为 `{joint['mount_xyz_task_world_m']}` m，绕原始基座局部 `+Y` 轴转 "
            f"`{joint['tilt_deg']:g}°`；固定 place 为 `{scene['place_position_m']}` m。"
            "原始 LINK_0 基座与 `task_world` 原点为 `(0,0)`。全部坐标为任务世界坐标。", "",
            "原始限位最近距离与有效限位最小余量均取每条成功轨迹所有保存采样、J1–J6 六关节中的最小值。"
            f"有效限位是规划器把原始限位每侧向内收缩 `{clip:.3f} rad` 后的限位。"
            "完整周期最大单关节跨度先算每个关节 `max(q)−min(q)`，再取最大值。"
            "指标包含六阶段抓放及该 case 的入场段；失败格没有关节数值，不代表几何不可达。"
            "各图使用本组局部色标，跨布置比较请看五组共享色标总图。", "",
            f"逐格数据来自已通过独立轨迹和限位审核的报告；`joint_distribution_cases.csv` "
            f"的 400 行已与 JSON 逐字段核对。成功 `{n_success}/400`，无完整轨迹 `{400-n_success}/400`。"
            "`plot_data_summary.json` 记录输入哈希和每格标注。", "",
        ]
        if task_style:
            readme += ["另有 `success_task_style.png`：按 V65 task map 版式绘制的二值成功图；"
                       "统一蓝色表示成功，浅灰红叉表示无完整轨迹。它不编码抓取搜索角度。", ""]
        readme += ["复现：", "", "```bash",
                   f"python3 xtrainer_plan/scripts/plot_full_mount_cell_maps.py {root} {name}"
                   + (" --task-style" if task_style else ""),
                   "```", ""]
        (stage / "README.md").write_text("\n".join(readme), encoding="utf-8")
        os.rename(stage, out)
    print(f"[ok] {out}: {len(figures)} figures, 400 cells, {n_success} success; audited CSV matched")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Full-run result root with runs/ and reports/")
    parser.add_argument("name", help="Candidate name in ROOT/manifest.json")
    parser.add_argument("--out-dir", type=Path, help="New output directory; default reports/NAME/cell_maps")
    parser.add_argument("--task-style", action="store_true",
                        help="Also draw the V65-style binary task-map success plot")
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out_dir or root / "reports" / args.name / "cell_maps").resolve()
    draw(root, args.name, out, args.task_style)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
