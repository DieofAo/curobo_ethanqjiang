#!/usr/bin/env python3
"""Draw the v64_47 binary success grid in the V65 task-map layout.

The V65 reference colours successful cells by adopted grasp angle. This plot
keeps the meaning of success_cell_map.png: one colour for all successes and a
red-bordered cross for every run without a saved complete trajectory.

From the repository root:
  python3 xtrainer_plan/scripts/plot_v64_47_success_task_style.py \
    --out /tmp/v64_47_success_task_style.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

from plot_overhead_results import _cell_basis, _cjk_font, _panel_extent, load_result


DEFAULT_ROOT = (Path(__file__).resolve().parents[1] / "results_overhead" /
                "20260928" / "v66_v64_8of9_local_y_full")
SUCCESS_COLOR = "#4053ba"
FAIL_COLOR = "#e1e1e1"
FAIL_EDGE = "#c62828"


def check_source(root: Path):
    run = root / "runs" / "v64_47"
    result = load_result(run)
    scene_path = root / "reports" / "v64_47" / "scene_distribution" / "task_scene_distribution.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    assert scene["n_total"] == 400 and scene["n_success"] == 376
    assert scene["tilt_axis"] == "original_base_local_y" and math.isclose(scene["tilt_deg"], 60.)
    assert np.allclose(scene["mount_origin_m"], [-.10, .15, .65], rtol=0, atol=1e-9)
    assert np.allclose(scene["place_position_m"], [-.36, -.12, .10], rtol=0, atol=1e-9)
    assert np.allclose(result.mount[:3, 3], scene["mount_origin_m"], rtol=0, atol=1e-9)
    assert np.allclose(result.place_raw, scene["place_position_m"], rtol=0, atol=1e-9)
    assert result.grid == scene["grid"] and result.task_frame == "task_world"
    assert result.declared_total == 400 and len(result.items) == 400
    assert result.n_success == 376 and result.n_failed == 24

    by_index = {int(item["index"]): item for item in result.items}
    assert len(by_index) == 400 and set(by_index) == set(range(400))
    seen_cells = set()
    csv_path = root / "reports" / "comparison" / "same_point_metrics.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 400
    for row in rows:
        index = int(row["index"])
        item = by_index[index]
        cell = int(row["row"]), int(row["col"])
        assert cell == (int(item["row"]), int(item["col"]))
        assert cell not in seen_cells
        seen_cells.add(cell)
        assert row["v64_47_success"] in ("True", "False")
        assert (row["v64_47_success"] == "True") is bool(item["success"])
        assert np.allclose([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                           item["position_raw"], rtol=0, atol=1e-9)
    assert seen_cells == {(r, c) for r in range(20) for c in range(20)}
    return result


def draw(result, output: Path) -> None:
    cjk = _cjk_font()
    if cjk:
        plt.rcParams["font.family"] = cjk
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8.8, 7.5), constrained_layout=True)
    row_step, col_step = _cell_basis(result)
    cell_row, cell_col = .86 * row_step, .86 * col_step

    for item in result.items:
        center = np.asarray(item["position_raw"][:2], dtype=float)
        success = bool(item["success"])
        corners = np.array([
            center - cell_row / 2 - cell_col / 2,
            center + cell_row / 2 - cell_col / 2,
            center + cell_row / 2 + cell_col / 2,
            center - cell_row / 2 + cell_col / 2,
        ])
        ax.add_patch(Polygon(corners, closed=True,
                             facecolor=SUCCESS_COLOR if success else FAIL_COLOR,
                             edgecolor="#303030" if success else FAIL_EDGE,
                             linewidth=.45 if success else .9, zorder=2))
        if not success:
            ax.text(center[0], center[1], "×", ha="center", va="center",
                    fontsize=5.25, fontweight="bold", color="#b2182b", zorder=4)

    mount = result.mount[:3, 3]
    place = result.place_raw
    ax.scatter([0], [0], marker="+", s=180, linewidths=2,
               color="#d4a600", zorder=7)
    ax.scatter([mount[0]], [mount[1]], marker="v", s=190,
               facecolors="none", edgecolors="#1565c0", linewidths=2, zorder=8)
    ax.annotate(f"实际基座\nz={mount[2]:.2f}m", xy=mount[:2], xytext=(6, 7),
                textcoords="offset points", fontsize=7.5, color="#0d47a1", zorder=9)
    ax.scatter([place[0]], [place[1]], marker="X", s=125,
               color="#d81b60", edgecolors="black", linewidths=.7, zorder=8)

    extent = _panel_extent([result])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
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
    ax.set_title("v64_47\n成功 376/400  失败 24\n"
                 "mount xyz=[-0.10, 0.15, 0.65] m  原始基座局部 +Y 倾角 60°  "
                 "grid z=0.030 m", fontsize=10.2, pad=8)
    fig.suptitle("XTrainer 单臂规划成功分布（保留的 task/world 坐标系）", fontsize=14)

    colourmap = ListedColormap([FAIL_COLOR, SUCCESS_COLOR])
    scalar = plt.cm.ScalarMappable(norm=BoundaryNorm([-.5, .5, 1.5], 2),
                                   cmap=colourmap)
    cb = fig.colorbar(scalar, ax=ax, fraction=.022, pad=.015, shrink=.84,
                      ticks=[0, 1])
    cb.ax.set_yticklabels(["无完整轨迹", "完整轨迹"])
    cb.set_label("规划结果")
    handles = [
        Patch(facecolor=SUCCESS_COLOR, edgecolor="#303030", label="规划成功"),
        Patch(facecolor=FAIL_COLOR, edgecolor=FAIL_EDGE, label="无完整轨迹"),
        Line2D([0], [0], marker="v", linestyle="none", markersize=10,
               markerfacecolor="none", markeredgecolor="#1565c0", markeredgewidth=2,
               label="实际基座投影"),
        Line2D([0], [0], marker="+", linestyle="none", markersize=11,
               markeredgewidth=2, color="#d4a600", label="保留的旧坐标原点"),
        Line2D([0], [0], marker="X", linestyle="none", markersize=9,
               markerfacecolor="#d81b60", markeredgecolor="black",
               label="place（原 task 坐标）"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.055),
               ncol=5, fontsize=8.2, framealpha=.94)
    fig.text(.5, -.095,
             "按记录顺序连续抓放；红叉表示无保存的完整轨迹，不证明几何不可达。刻度间隔 2 cm，不代表采样间隔。",
             ha="center", va="top", fontsize=8, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), f"Refusing to overwrite {output}"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path,
                        default=DEFAULT_ROOT / "reports/v64_47/cell_maps/success_task_style.png")
    args = parser.parse_args()
    result = check_source(args.root.resolve())
    draw(result, args.out.resolve())
    print(f"[ok] {args.out}: 400 cells, 376 success, 24 without saved trajectory; CSV matched")


if __name__ == "__main__":
    main()
