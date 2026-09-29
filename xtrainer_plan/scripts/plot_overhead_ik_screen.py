#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare overhead mount candidates from ``scan_overhead_ik.py`` reports.

The colored task-plane cells show the first configured grasp angle for which
all four relevant endpoints are IK-feasible: grasp lift, grasp, place lift and
place.  A failed cell has no same-angle endpoint intersection and is marked
with ``x``.  Home feasibility is reported separately and never changes a
cell's color.

This is deliberately *not* a trajectory-success plot.  Endpoint IK does not
establish branch continuity, acceptable joint jumps, Cartesian insertion
quality, or collision-free continuous motion.  Shortlisted mounts must still
be run through the normal pick/place planner.

Inputs may be individual ``ik_screen.json`` files, result directories that
contain one, or a batch directory whose descendants contain the reports::

    python3 scripts/plot_overhead_ik_screen.py \
        results_overhead/20260915/v4_ik_screen \
        --out results_overhead/20260915/v4_ik_screen/comparison.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtrainer_common import (  # noqa: E402
    matrix_to_rpy_deg,
    parse_rigid_transform_matrix,
)


IK_ONLY_EN = (
    "ENDPOINT IK ONLY — not trajectory success: no branch-continuity, "
    "joint-jump, Cartesian-path, or continuous-collision guarantee."
)
IK_ONLY_ZH = (
    "仅为端点 IK 筛选，不代表轨迹成功：尚未验证 IK 分支连续性、关节跳变、"
    "笛卡尔插拔质量或连续轨迹碰撞。"
)


def _pick_cjk_font() -> str | None:
    preferred = [
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Source Han Sans CN",
        "SimHei",
        "Microsoft YaHei",
        "AR PL UMing CN",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in preferred if name in installed), None)


@dataclass
class Screen:
    path: Path
    name: str
    report: Dict[str, Any]
    config: Dict[str, Any]
    summary: Dict[str, Any]
    points: List[Dict[str, Any]]
    angle_by_index: Dict[int, Dict[str, Any]]
    mount: np.ndarray
    grid: Dict[str, Any]
    place: np.ndarray | None
    task_frame: str
    warnings: List[str] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def chosen_angles(self) -> List[float | None]:
        return [_chosen_grasp_angle(self, point) for point in self.points]

    @property
    def n_endpoint_covered(self) -> int:
        return sum(value is not None for value in self.chosen_angles)

    @property
    def n_grasp_covered(self) -> int:
        return sum(
            bool(point.get("grasp_and_lift_feasible_angle_indices"))
            for point in self.points
        )


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _resolve_input_paths(raw_inputs: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for raw in raw_inputs:
        path = Path(raw).resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir() and (path / "ik_screen.json").is_file():
            candidates = [path / "ik_screen.json"]
        elif path.is_dir():
            candidates = sorted(path.rglob("ik_screen.json"))
            if not candidates:
                raise ValueError(f"no ik_screen.json below {path}")
        else:
            raise ValueError(f"input does not exist: {path}")
        found.extend(candidate.resolve() for candidate in candidates)

    unique: List[Path] = []
    seen: set[Path] = set()
    for path in found:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    if not unique:
        raise ValueError("no ik_screen.json inputs")
    return unique


def _valid_position(point: Dict[str, Any], source: Path) -> np.ndarray:
    raw = point.get("position_raw", point.get("position"))
    position = np.asarray(raw, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(
            f"{source}: point index={point.get('index')} has invalid task position"
        )
    return position


def _mount_label(mount: np.ndarray) -> str:
    rotation = mount[:3, :3]
    rx = np.diag([1.0, -1.0, -1.0])
    ry = np.diag([-1.0, 1.0, -1.0])
    if np.allclose(rotation, rx, rtol=0.0, atol=1.0e-7):
        return "Rx180"
    if np.allclose(rotation, ry, rtol=0.0, atol=1.0e-7):
        return "Ry180"
    rpy = matrix_to_rpy_deg(rotation)
    return "rpy=" + ",".join(f"{value:.0f}" for value in rpy)


def load_screen(path: Path) -> Screen:
    report = _read_json(path)
    if int(report.get("schema_version", -1)) != 1:
        raise ValueError(
            f"{path}: unsupported schema_version={report.get('schema_version')!r}"
        )
    config_doc = report.get("config") or {}
    config = config_doc.get("resolved") or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config.resolved must be an object")
    robot = config.get("robot") or {}
    pick_place = config.get("pick_place") or {}
    mount = parse_rigid_transform_matrix(
        robot.get("mount_transform"),
        f"{path}: config.resolved.robot.mount_transform",
    )

    raw_points = report.get("points") or []
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError(f"{path}: points must be a non-empty list")
    points: List[Dict[str, Any]] = []
    seen_points: set[int] = set()
    for offset, raw_point in enumerate(raw_points):
        if not isinstance(raw_point, dict):
            raise ValueError(f"{path}: point #{offset} is not an object")
        point = dict(raw_point)
        index = int(point.get("index", offset))
        if index in seen_points:
            raise ValueError(f"{path}: duplicate point index={index}")
        seen_points.add(index)
        point["index"] = index
        point["position_raw"] = _valid_position(point, path).tolist()
        points.append(point)

    raw_angles = report.get("angles") or []
    if not isinstance(raw_angles, list) or not raw_angles:
        raise ValueError(f"{path}: angles must be a non-empty list")
    angle_by_index: Dict[int, Dict[str, Any]] = {}
    for row in raw_angles:
        index = int(row["angle_index"])
        if index in angle_by_index:
            raise ValueError(f"{path}: duplicate angle_index={index}")
        angle_by_index[index] = dict(row)

    summary = report.get("summary") or {}
    warnings: List[str] = []
    expected_points = int(summary.get("n_grasp_points") or len(points))
    if expected_points != len(points):
        warnings.append(
            f"summary n_grasp_points={expected_points}, points={len(points)}"
        )

    correction_raw = (report.get("frame") or {}).get(
        "link0_target_transform_C"
    )
    if correction_raw is not None:
        correction = parse_rigid_transform_matrix(
            correction_raw, f"{path}: frame.link0_target_transform_C"
        )
        inverse_error = float(
            np.max(np.abs(correction @ mount - np.eye(4)))
        )
        if inverse_error > 1.0e-6:
            warnings.append(f"C @ mount != I (max residual {inverse_error:.3g})")

    grid = dict(pick_place.get("grasp_grid") or {})
    place_value = (pick_place.get("place") or {}).get("position")
    place = None
    if place_value is not None:
        place = np.asarray(place_value, dtype=np.float64)
        if place.shape != (3,) or not np.all(np.isfinite(place)):
            raise ValueError(f"{path}: invalid raw place position")

    task_frame = str(
        robot.get("task_frame")
        or robot.get("legacy_task_frame")
        or "task_world"
    )
    screen = Screen(
        path=path,
        name=path.parent.name,
        report=report,
        config=config,
        summary=summary,
        points=points,
        angle_by_index=angle_by_index,
        mount=mount,
        grid=grid,
        place=place,
        task_frame=task_frame,
        warnings=warnings,
    )

    computed = screen.n_endpoint_covered
    recorded = summary.get(
        "n_grasp_points_with_any_grasp_place_endpoint_intersection"
    )
    if recorded is not None and int(recorded) != computed:
        screen.warnings.append(
            f"summary endpoint coverage={recorded}, recomputed={computed}"
        )
    return screen


def _chosen_grasp_angle(screen: Screen, point: Dict[str, Any]) -> float | None:
    raw_indices = point.get(
        "grasp_place_endpoint_intersection_angle_indices"
    ) or []
    indices = [int(index) for index in raw_indices]
    if not indices:
        return None

    # The scanner writes these indices in the configured planner search order.
    # Prefer its explicit best-angle record when it is an intersection result,
    # otherwise use the first feasible index from that ordered list.
    best = point.get("best_angle") or {}
    best_index = best.get("angle_index")
    if bool(best.get("includes_place_intersection")) and best_index is not None:
        index = int(best_index)
        if index not in indices:
            raise ValueError(
                f"{screen.path}: point index={point['index']} best angle is not "
                "in its endpoint-intersection list"
            )
    else:
        index = indices[0]
    if index not in screen.angle_by_index:
        raise ValueError(
            f"{screen.path}: point index={point['index']} references "
            f"unknown angle_index={index}"
        )
    return float(screen.angle_by_index[index]["grasp_deg"])


def _median_vector(vectors: List[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if vectors:
        value = np.median(np.stack(vectors), axis=0)
        if np.linalg.norm(value) > 1.0e-10:
            return value
    return fallback


def _cell_basis(screen: Screen) -> tuple[np.ndarray, np.ndarray]:
    indexed: Dict[tuple[int, int], np.ndarray] = {}
    for point in screen.points:
        if point.get("row") is None or point.get("col") is None:
            continue
        indexed[(int(point["row"]), int(point["col"]))] = np.asarray(
            point["position_raw"][:2], dtype=np.float64
        )

    row_vectors: List[np.ndarray] = []
    col_vectors: List[np.ndarray] = []
    for (row, col), position in indexed.items():
        if (row + 1, col) in indexed:
            row_vectors.append(indexed[(row + 1, col)] - position)
        if (row, col + 1) in indexed:
            col_vectors.append(indexed[(row, col + 1)] - position)

    rows = max(1, int(screen.grid.get("rows") or 1))
    cols = max(1, int(screen.grid.get("cols") or 1))
    x_range = screen.grid.get("x_range") or [0.0, 0.0]
    y_range = screen.grid.get("y_range") or [0.0, 0.0]
    dx = (
        (float(x_range[1]) - float(x_range[0])) / (rows - 1)
        if rows > 1 else 0.02
    )
    dy = (
        (float(y_range[1]) - float(y_range[0])) / (cols - 1)
        if cols > 1 else 0.02
    )
    row_step = _median_vector(row_vectors, np.array([dx or 0.02, 0.0]))
    col_step = _median_vector(col_vectors, np.array([0.0, dy or 0.02]))

    # A 3x3 smoke grid has very sparse samples.  Cap each cell at 4 cm so the
    # picture does not imply that an entire 20-30 cm area was checked.  At a
    # final 2 cm scan resolution cells naturally shrink to about 1.7 cm.
    def cell_vector(step: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(step))
        if length <= 1.0e-12:
            return np.array([0.02, 0.0])
        cell_length = min(0.84 * length, 0.04)
        return step / length * cell_length

    return cell_vector(row_step), cell_vector(col_step)


def _angle_range(screens: Sequence[Screen]) -> tuple[float, float]:
    values = [
        float(row["grasp_deg"])
        for screen in screens
        for row in screen.angle_by_index.values()
    ]
    low, high = min(values), max(values)
    if math.isclose(low, high, abs_tol=1.0e-12):
        return low - 1.0, high + 1.0
    return low, high


def _shared_extent(screens: Sequence[Screen]) -> tuple[float, float, float, float]:
    coordinates: List[np.ndarray] = [np.array([0.0, 0.0])]
    for screen in screens:
        coordinates.extend(
            np.asarray(point["position_raw"][:2], dtype=np.float64)
            for point in screen.points
        )
        coordinates.append(screen.mount[:2, 3].copy())
        if screen.place is not None:
            coordinates.append(screen.place[:2].copy())
    xy = np.stack(coordinates)
    padding = max(0.04, 0.07 * float(np.ptp(xy, axis=0).max()))
    quantum = 0.02
    return (
        math.floor((float(xy[:, 0].min()) - padding) / quantum) * quantum,
        math.ceil((float(xy[:, 0].max()) + padding) / quantum) * quantum,
        math.floor((float(xy[:, 1].min()) - padding) / quantum) * quantum,
        math.ceil((float(xy[:, 1].max()) + padding) / quantum) * quantum,
    )


def _draw_summary(axis: plt.Axes, screens: Sequence[Screen], language) -> None:
    L = language
    count = len(screens)
    x = np.arange(count, dtype=np.float64)
    totals = np.asarray([max(screen.n_points, 1) for screen in screens])
    grasp_rates = np.asarray(
        [screen.n_grasp_covered for screen in screens], dtype=np.float64
    ) / totals * 100.0
    endpoint_rates = np.asarray(
        [screen.n_endpoint_covered for screen in screens], dtype=np.float64
    ) / totals * 100.0
    width = 0.36
    axis.bar(
        x - width / 2.0, grasp_rates, width,
        color="#90caf9", edgecolor="#1565c0", linewidth=0.8,
        label=L("grasp+lift 端点", "grasp+lift endpoints"),
    )
    axis.bar(
        x + width / 2.0, endpoint_rates, width,
        color="#66bb6a", edgecolor="#1b5e20", linewidth=0.8,
        label=L("同角度 grasp/place 端点交集", "same-angle grasp/place endpoints"),
    )

    labels: List[str] = []
    for index, screen in enumerate(screens):
        home_ok = bool(screen.summary.get("home_feasible"))
        marker = "o" if home_ok else "x"
        color = "#2e7d32" if home_ok else "#c62828"
        axis.scatter(
            [index], [108.0], marker=marker, s=55, color=color,
            linewidths=1.8, zorder=5,
        )
        place_count = int(
            screen.summary.get("n_place_angles_place_and_lift_feasible") or 0
        )
        angle_count = int(screen.summary.get("n_coupled_angles") or 0)
        best = screen.summary.get("best_global_angle") or {}
        best_text = (
            "--" if best.get("grasp_deg") is None
            else f"{float(best['grasp_deg']):+.0f}°"
        )
        axis.text(
            index,
            min(endpoint_rates[index] + 3.0, 98.0),
            f"{screen.n_endpoint_covered}/{screen.n_points}\n"
            f"best {best_text}",
            ha="center", va="bottom", fontsize=7.2,
        )
        labels.append(
            f"{screen.name}\n{_mount_label(screen.mount)} "
            f"z={screen.mount[2, 3]:.2f}\n"
            f"place {place_count}/{angle_count}"
        )

    axis.axhline(100.0, color="0.35", linestyle=":", linewidth=0.8)
    axis.set_ylim(0.0, 116.0)
    axis.set_xlim(-0.65, count - 0.35)
    axis.set_ylabel(L("采样点端点 IK 覆盖率 (%)", "sampled-point endpoint IK coverage (%)"))
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=7.2)
    axis.grid(axis="y", linestyle=":", alpha=0.4)
    axis.set_axisbelow(True)
    axis.set_title(
        L(
            "安装方案端点 IK 比较（顶部 ○/× = home 端点 IK 成功/失败）",
            "Mount comparison by endpoint IK (top o/x = home endpoint IK pass/fail)",
        ),
        fontsize=11,
    )
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.31),
        fontsize=8, ncol=2,
    )


def _draw_map(
    axis: plt.Axes,
    screen: Screen,
    *,
    cmap,
    norm,
    extent: tuple[float, float, float, float],
    annotate: bool,
    language,
) -> None:
    L = language
    cell_row, cell_col = _cell_basis(screen)
    chosen = screen.chosen_angles
    for point, angle in zip(screen.points, chosen):
        center = np.asarray(point["position_raw"][:2], dtype=np.float64)
        vertices = np.array([
            center - cell_row / 2.0 - cell_col / 2.0,
            center + cell_row / 2.0 - cell_col / 2.0,
            center + cell_row / 2.0 + cell_col / 2.0,
            center - cell_row / 2.0 + cell_col / 2.0,
        ])
        feasible = angle is not None
        face = cmap(norm(float(angle))) if feasible else "#e0e0e0"
        axis.add_patch(Polygon(
            vertices,
            closed=True,
            facecolor=face,
            edgecolor="#303030" if feasible else "#c62828",
            linewidth=0.65 if feasible else 1.0,
            zorder=3,
        ))
        if feasible and annotate:
            rgba = cmap(norm(float(angle)))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(
                center[0], center[1], f"{float(angle):+.0f}",
                ha="center", va="center", fontsize=7.3,
                fontweight="bold", color="black" if luminance > 0.52 else "white",
                zorder=11,
            )
        elif not feasible:
            axis.text(
                center[0], center[1], "×",
                ha="center", va="center", fontsize=11,
                fontweight="bold", color="#b2182b", zorder=11,
            )

    x_range = screen.grid.get("x_range")
    y_range = screen.grid.get("y_range")
    if x_range is not None and y_range is not None:
        x0, x1 = sorted(float(value) for value in x_range)
        y0, y1 = sorted(float(value) for value in y_range)
        axis.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor="0.3", linestyle="--", linewidth=0.9,
            zorder=2,
        ))

    axis.scatter(
        [0.0], [0.0], marker="+", s=155, linewidths=2.0,
        color="#d4a600", zorder=8,
    )
    mount_xy = screen.mount[:2, 3]
    axis.scatter(
        [mount_xy[0]], [mount_xy[1]], marker="v", s=175,
        facecolors="none", edgecolors="#1565c0", linewidths=2.0, zorder=9,
    )
    if screen.place is not None:
        axis.scatter(
            [screen.place[0]], [screen.place[1]], marker="X", s=110,
            color="#d81b60", edgecolors="black", linewidths=0.6, zorder=9,
        )

    covered = screen.n_endpoint_covered
    summary = screen.summary
    home = L("通过", "PASS") if summary.get("home_feasible") else L("失败", "FAIL")
    place_ok = int(summary.get("n_place_angles_place_and_lift_feasible") or 0)
    n_angles = int(summary.get("n_coupled_angles") or len(screen.angle_by_index))
    title = L(
        f"{screen.name}  |  同角度四端点 {covered}/{screen.n_points}"
        f"  home={home}  place角={place_ok}/{n_angles}\n"
        f"mount={_mount_label(screen.mount)}  "
        f"xyz={np.round(screen.mount[:3, 3], 3).tolist()}",
        f"{screen.name}  |  same-angle four endpoints {covered}/{screen.n_points}"
        f"  home={home}  place angles={place_ok}/{n_angles}\n"
        f"mount={_mount_label(screen.mount)}  "
        f"xyz={np.round(screen.mount[:3, 3], 3).tolist()}",
    )
    axis.set_title(title, fontsize=9.3, pad=7)
    axis.set_xlabel(f"{screen.task_frame} x (m)")
    axis.set_ylabel(f"{screen.task_frame} y (m)")
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal", adjustable="box")

    # Coarse labels remain readable; minor ticks/grid provide the requested
    # exact 2 cm spatial scale.
    axis.xaxis.set_major_locator(MultipleLocator(0.10))
    axis.yaxis.set_major_locator(MultipleLocator(0.10))
    axis.xaxis.set_minor_locator(MultipleLocator(0.02))
    axis.yaxis.set_minor_locator(MultipleLocator(0.02))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.tick_params(axis="both", which="major", labelsize=7, length=4)
    axis.tick_params(axis="both", which="minor", length=2)
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.52)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.42, alpha=0.45)
    axis.set_axisbelow(True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="ik_screen.json files, result directories, or batch directories",
    )
    parser.add_argument("--out", required=True, help="output PNG/PDF/SVG")
    parser.add_argument(
        "--no-angle-labels", action="store_true",
        help="color feasible samples by angle without writing angle numbers",
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    try:
        paths = _resolve_input_paths(args.inputs)
        screens = [load_screen(path) for path in paths]
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    cjk = _pick_cjk_font()
    if cjk:
        plt.rcParams["font.family"] = cjk
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    count = len(screens)
    columns = 1 if count == 1 else min(3, count)
    map_rows = int(math.ceil(count / columns))
    figure = plt.figure(
        figsize=(7.0 * columns, 3.8 + 6.2 * map_rows)
    )
    grid_spec = figure.add_gridspec(
        map_rows + 1,
        columns,
        height_ratios=[0.58] + [1.0] * map_rows,
        hspace=0.48,
        wspace=0.27,
    )
    summary_axis = figure.add_subplot(grid_spec[0, :])
    _draw_summary(summary_axis, screens, L)

    low, high = _angle_range(screens)
    norm = plt.Normalize(vmin=low, vmax=high)
    cmap = plt.get_cmap("coolwarm")
    extent = _shared_extent(screens)
    map_axes: List[plt.Axes] = []
    for index, screen in enumerate(screens):
        row, col = divmod(index, columns)
        axis = figure.add_subplot(grid_spec[row + 1, col])
        map_axes.append(axis)
        _draw_map(
            axis,
            screen,
            cmap=cmap,
            norm=norm,
            extent=extent,
            annotate=not args.no_angle_labels,
            language=L,
        )
        for warning in screen.warnings:
            print(f"[warn] {screen.name}: {warning}", file=sys.stderr)
    for index in range(count, map_rows * columns):
        row, col = divmod(index, columns)
        axis = figure.add_subplot(grid_spec[row + 1, col])
        axis.set_visible(False)

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    # A dedicated axes keeps the shared colorbar out of the rightmost map;
    # ``Figure.colorbar(..., ax=map_axes)`` can overlap GridSpec panels after
    # a later ``subplots_adjust`` on older Matplotlib releases.
    colorbar_axis = figure.add_axes([0.947, 0.20, 0.012, 0.51])
    colorbar = figure.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label(
        L(
            "首个同角度四端点 IK 可行的 grasp 角 (°)",
            "first same-angle four-endpoint IK-feasible grasp angle (deg)",
        )
    )

    legend = [
        Patch(
            facecolor="#e0e0e0", edgecolor="#c62828",
            label=L("无同角度四端点 IK 交集", "no same-angle four-endpoint IK intersection"),
        ),
        Line2D(
            [0], [0], marker="v", linestyle="none", markersize=9,
            markerfacecolor="none", markeredgecolor="#1565c0",
            markeredgewidth=2.0,
            label=L("上方基座投影", "overhead-base projection"),
        ),
        Line2D(
            [0], [0], marker="+", linestyle="none", markersize=10,
            markeredgewidth=2.0, color="#d4a600",
            label=L("保留的任务原点", "preserved task origin"),
        ),
        Line2D(
            [0], [0], marker="X", linestyle="none", markersize=8,
            markerfacecolor="#d81b60", markeredgecolor="black",
            label=L("place（task坐标）", "place (task coordinates)"),
        ),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.038),
        ncol=4,
        fontsize=8,
        framealpha=0.94,
    )
    figure.suptitle(
        L(
            "XTrainer 倒装单臂安装方案与任务平面端点 IK 覆盖",
            "XTrainer overhead mount comparison and task-plane endpoint IK coverage",
        ),
        fontsize=14,
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        L(IK_ONLY_ZH, IK_ONLY_EN)
        + L("  图中细网格间距=2cm；色块仅代表离散采样点。",
            "  Minor grid=2cm; colored cells are discrete samples only."),
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#b71c1c",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff3e0", "edgecolor": "#ef6c00"},
    )
    figure.subplots_adjust(top=0.93, bottom=0.105, left=0.065, right=0.925)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"[ok] {output.resolve()}")
    print(f"[info] reports={len(screens)}, angle_range=[{low:g}, {high:g}]deg")
    for screen in screens:
        print(
            f"[info] {screen.name}: endpoint_intersection="
            f"{screen.n_endpoint_covered}/{screen.n_points}, "
            f"home={'PASS' if screen.summary.get('home_feasible') else 'FAIL'}, "
            f"mount={_mount_label(screen.mount)} "
            f"xyz={np.round(screen.mount[:3, 3], 6).tolist()}"
        )
    print(f"[warn] {IK_ONLY_EN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
