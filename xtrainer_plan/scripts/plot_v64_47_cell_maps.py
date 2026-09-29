#!/usr/bin/env python3
"""Draw readable per-cell V66/v64_47 maps from independently audited data.

Rebuild from the V66 result root, without planning or changing earlier figures:
  python3 xtrainer_plan/scripts/plot_v64_47_cell_maps.py V66_ROOT --out-dir /tmp/v64_47_cells
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.patches import Patch, Rectangle
import numpy as np

from compare_full_mount_candidates import NAMES, load_candidate, matched_data
from plot_full_mount_joint_distributions import require, sha256
from plot_grasp_angle_map import pick_cjk_font
from plot_v64_47_v65_style import check_published_csv


NAME = "v64_47"
METRICS = (
    ("min_raw_margin_deg", "raw_limit_distance_cell_map.png",
     "原始 URDF 关节限位最近距离", "整条轨迹 · J1–J6 全关节最小值", "viridis"),
    ("min_effective_margin_deg", "effective_limit_margin_cell_map.png",
     "规划器有效关节限位最小余量", "整条轨迹 · J1–J6 全关节最小值", "cividis"),
    ("max_cycle_joint_span_deg", "max_joint_span_cell_map.png",
     "完整抓放周期单关节最大角跨度", "每条轨迹取 J1–J6 中最大的 max(q)−min(q)", "RdYlBu_r"),
)


def contrast_color(rgba: tuple[float, ...]) -> str:
    """Black on bright cells, white on dark cells."""
    rgb = np.asarray(rgba[:3], dtype=float)
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = float(np.dot(linear, [.2126, .7152, .0722]))
    return "#16202b" if luminance > .30 else "#ffffff"


def draw_cell(ax, x: float, y: float, dx: float, dy: float, *, fill: str | tuple,
              label: str, color: str, failed: bool = False) -> None:
    ax.add_patch(Rectangle((x - dx/2, y - dy/2), dx, dy,
                           facecolor=fill, edgecolor="#e1464b" if failed else "#ffffff",
                           linewidth=.85 if failed else .38, zorder=2))
    ax.text(x, y, label, ha="center", va="center", fontsize=6.2,
            color=color, zorder=3)


def draw_one(records: list[dict], summary: dict, candidate: dict,
             key: str | None, title: str, subtitle: str, palette: str | None,
             output: Path) -> dict:
    grid = summary["grid"]
    xs = np.linspace(*grid["x_range"], int(grid["rows"]))
    ys = np.linspace(*grid["y_range"], int(grid["cols"]))
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    success = [row for row in records if row[f"{NAME}_success"]]
    values = np.asarray([float(row[f"{NAME}_{key}"]) for row in success], dtype=float) if key else None
    if key:
        require(len(values) == 376 and np.isfinite(values).all(), "Invalid cell metric values")
        norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
        cmap = plt.get_cmap(palette)
    fig, ax = plt.subplots(figsize=(16.8, 14.2), dpi=240)
    fig.subplots_adjust(left=.105, right=.82, bottom=.115, top=.88)
    for row in records:
        x, y = float(row["x_m"]), float(row["y_m"])
        if not row[f"{NAME}_success"]:
            draw_cell(ax, x, y, dx, dy, fill="#fbfaf8", label="×",
                      color="#d4383a", failed=True)
        elif key:
            value = float(row[f"{NAME}_{key}"])
            rgba = cmap(norm(value))
            draw_cell(ax, x, y, dx, dy, fill=rgba, label=f"{value:.1f}",
                      color=contrast_color(rgba))
        else:
            draw_cell(ax, x, y, dx, dy, fill="#157f86", label="✓", color="white")

    base = candidate["joint"]["mount_xyz_task_world_m"]
    place = summary["place_position_m"]
    require(np.allclose(base, [-.10, .15, .65], rtol=0, atol=1e-9), "Unexpected v64_47 mount")
    require(np.allclose(place, [-.36, -.12, .10], rtol=0, atol=1e-9), "Unexpected fixed place")
    ax.scatter([base[0]], [base[1]], marker="*", s=270, facecolor="#f2ae00",
               edgecolor="#1e2530", linewidth=1.4, zorder=6)
    ax.scatter([place[0]], [place[1]], marker="X", s=150, facecolor="#d32971",
               edgecolor="#211b27", linewidth=1.2, zorder=6)
    ax.scatter([0], [0], marker="o", s=105, facecolor="#f8f8f8",
               edgecolor="#454a54", linewidth=1.5, zorder=6)
    ax.scatter([0], [0], marker="+", s=63, color="#454a54",
               linewidth=1.1, zorder=7)

    # The 20 sampled x and y coordinates are the actual grid centres.  The
    # current mount and the original base lie outside the grasp rectangle.
    ax.set_xlim(-.655, .057)
    ax.set_ylim(-.16, .445)
    ax.set_aspect("equal", adjustable="box")
    # Coordinate ticks are every 0.02 m, independent of the exact 20×20
    # sample centres (whose spacings are 0.42/19 and 0.50/19 metres).
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(.02))
        axis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=6.8)
    ax.tick_params(axis="y", labelsize=7.1)
    ax.grid(color="#cbd2d9", linewidth=.43, linestyle=":", alpha=.68, zorder=0)
    ax.set_xlabel("task_world / 原始 LINK_0  X (m)", fontsize=11)
    ax.set_ylabel("task_world / 原始 LINK_0  Y (m)", fontsize=11)
    if key:
        range_text = f"数值范围 {values.min():.2f}–{values.max():.2f}°  |  成功 376/400"
    else:
        range_text = "完整抓放成功 376/400  |  失败或跳过 24/400"
    fig.suptitle(f"v64_47  ·  {title}\n{subtitle}\n{range_text}", fontsize=15.5, y=.97)
    failed_handle = Patch(facecolor="#fbfaf8", edgecolor="#e1464b", label="失败/跳过：无保存轨迹")
    handles = [
        Line2D([], [], marker="*", linestyle="None", markerfacecolor="#f2ae00",
               markeredgecolor="#1e2530", markersize=12,
               label="当前 v64_47 安装基座  (-0.10, +0.15, +0.65) m"),
        Line2D([], [], marker="X", linestyle="None", markerfacecolor="#d32971",
               markeredgecolor="#211b27", markersize=9,
               label="固定 place  (-0.36, -0.12, +0.10) m"),
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="white",
               markeredgecolor="#454a54", markersize=8,
               label="原始 LINK_0 基座 / task_world 原点  (0, 0)"),
        failed_handle,
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              facecolor="white", framealpha=.98, edgecolor="#b3bac2")
    if key:
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          fraction=.046, pad=.034, shrink=.85)
        cb.set_label("角度 (°)", fontsize=11)
        cb.ax.tick_params(labelsize=9)
    else:
        ax.legend(handles=[Patch(facecolor="#157f86", edgecolor="white", label="成功：有完整轨迹"),
                           *handles], loc="upper right", fontsize=9,
                  facecolor="white", framealpha=.98, edgecolor="#b3bac2")
    fig.text(.50, .055,
             "grasp 网格 Z = 0.03 m；颜色只表示当前图的指标。失败格无关节数值，红叉不代表已证明几何不可达。",
             ha="center", fontsize=9, color="#46505b")
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)
    return {"minimum_deg": float(values.min()), "maximum_deg": float(values.max()),
            "cell_labels": [f"{float(row[f'{NAME}_{key}']):.1f}" if row[f"{NAME}_success"] else "×"
                            for row in records]} if key else {"n_success": 376, "n_failed": 24}


def draw(root: Path, out: Path) -> None:
    require(not out.exists(), f"Refusing to overwrite existing output: {out}")
    candidates = [load_candidate(root, name) for name in NAMES]
    records, summary = matched_data(candidates)
    csv_path = root / "reports/comparison/same_point_metrics.csv"
    check_published_csv(csv_path, records)
    candidate = next(c for c in candidates if c["name"] == NAME)
    require(len(records) == 400 and sum(row[f"{NAME}_success"] for row in records) == 376
            and sum(not row[f"{NAME}_success"] for row in records) == 24,
            "Expected complete audited v64_47 20×20 grid with 376 successes")
    require(candidate["joint"]["tilt_axis"] == "original_base_local_y"
            and math.isclose(candidate["joint"]["tilt_deg"], 60., abs_tol=1e-9),
            "Expected v64_47 local +Y 60° mount")

    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v64_47_cell_maps_", dir=out.parent) as scratch:
        stage = Path(scratch) / "data"
        stage.mkdir()
        file_info = {}
        file_info["success_cell_map.png"] = draw_one(
            records, summary, candidate, None, "grasp 区域成功分布",
            "20×20 目标网格 · 绿色成功 / 红叉失败", None,
            stage / "success_cell_map.png")
        for key, filename, title, subtitle, palette in METRICS:
            file_info[filename] = draw_one(records, summary, candidate,
                                           key, title, subtitle, palette,
                                           stage / filename)
        audit = {
            "candidate": NAME,
            "source_comparison_csv": str(csv_path),
            "source_comparison_csv_sha256": sha256(csv_path),
            "trajectory_sha256": candidate["joint"]["provenance"]["trajectory_sha256"],
            "independent_verification_passed": True,
            "joint_limit_clip_audit_passed": True,
            "n_grid_cells": len(records),
            "n_success": 376,
            "n_failed_or_skipped": 24,
            "mount_task_world_m": candidate["joint"]["mount_xyz_task_world_m"],
            "original_base_xy_task_world_m": [0., 0.],
            "place_task_world_m": summary["place_position_m"],
            "tilt_axis": "original_base_local_y",
            "tilt_deg": 60.,
            "grid": summary["grid"],
            "figures": file_info,
        }
        (stage / "plot_data_summary.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
        (stage / "README.md").write_text(
            "# v64_47 逐格分布图\n\n"
            "四张 20×20 独立图片：`success_cell_map.png`、`raw_limit_distance_cell_map.png`、"
            "`effective_limit_margin_cell_map.png`、`max_joint_span_cell_map.png`。"
            "三张关节指标图在每个成功格标一位小数（度），成功分布图以勾号标记成功；失败/跳过均为浅色红框和红叉，没有关节数值。"
            "原始 LINK_0 基座 `(0,0)` 与当前 v64_47 安装基座 `(-0.10,0.15,0.65) m` 单独标记，"
            "固定 place 是 `(-0.36,-0.12,0.10) m`。所有 XY 坐标均为 `task_world` 中的原始位置，"
            "grasp 高度为 `0.03 m`。坐标轴每 `0.02 m` 一格只是读数刻度；"
            "20×20 格子的中心保留真实采样坐标（X 间距 `0.42/19 m`，Y 间距 `0.50/19 m`）。"
            "当前安装绕原始基座局部 `+Y` 轴转 `60°`。\n\n"
            "原始限位距离和有效限位余量均为每条已保存完整轨迹所有离散采样、J1–J6 六关节中的**最小**距离；"
            "有效限位是规划器将 URDF 原始上下限向内各收缩 `0.14 rad` 后的限位。"
            "跨度是每条完整抓放轨迹中，J1–J6 各自 `max(q)−min(q)` 的**最大**值，单位度；"
            "各图色标只按 v64_47 当前指标范围设置，适合看本组内部差异，不能直接以色深比较其他布置。"
            "失败红叉表示规划未保存完整轨迹，不等于几何不可达。指标只覆盖保存的离散采样。\n\n"
            "绘图读取 V66 四组经独立轨迹审核和关节限位审核的报告，重新核对源文件哈希，"
            "并逐字段核对 `reports/comparison/same_point_metrics.csv` 的 400 行。"
            "`plot_data_summary.json` 保存来源哈希、数值范围及每格实际标注。\n\n"
            "复现（在仓库根目录）：\n\n"
            "```bash\n"
            "python3 xtrainer_plan/scripts/plot_v64_47_cell_maps.py "
            "xtrainer_plan/results_overhead/20260928/v66_v64_8of9_local_y_full "
            "--out-dir /tmp/v64_47_cell_maps_rebuild\n"
            "```\n",
            encoding="utf-8")
        os.rename(stage, out)
    print(f"[ok] {out}: 4 figures, 400 cells, 376 success, 24 failed; audited V66 CSV matched")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="V66 results root")
    parser.add_argument("--out-dir", type=Path, help="New output directory")
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out_dir or root / "reports" / NAME / "cell_maps").resolve()
    draw(root, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
