#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot overhead single-arm pick/place results in the preserved task frame.

Unlike ``plot_grasp_angle_map.py``, this utility always draws the coordinates
from before ``pick_place.link0_target_transform`` was applied.  The plotted
frame is therefore the preserved task/world frame, while the physical robot
base projection comes from ``meta.config.robot.mount_transform``.

Examples::

    python3 scripts/plot_overhead_results.py results_pick_place/smoke_h055 \
        --out results_pick_place/smoke_h055/overhead_task_map.png

    python3 scripts/plot_overhead_results.py \
        results_pick_place/smoke_h055 results_pick_place/smoke_h065 \
        results_pick_place/smoke_h075 --compare --out /tmp/heights.png

    python3 scripts/plot_overhead_results.py results_pick_place/full \
        --replan-result-dir results_pick_place/full_failed_grasp_to_0 \
        --out results_pick_place/full/overhead_task_map_combined.png

For one source result, every ``--replan-result-dir`` is applied in order.  For
multiple source results, either omit replans or provide exactly one replan per
source (in the same order).  A completed all-failed run is supported through
``plan_failed.json`` plus ``plan_skipped.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtrainer_common import (  # noqa: E402
    matrix_to_rpy_deg,
    parse_rigid_transform_matrix,
)


TASK_FRAME_DEFAULT = "task_world"


def _cjk_font() -> str | None:
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
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    return next((name for name in preferred if name in installed), None)


@dataclass
class ResultData:
    path: Path
    config: Dict[str, Any]
    items: List[Dict[str, Any]]
    grid: Dict[str, Any]
    place_raw: np.ndarray | None
    mount: np.ndarray
    declared_total: int
    task_frame: str
    warnings: List[str] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return sum(bool(item.get("success")) for item in self.items)

    @property
    def n_failed(self) -> int:
        return sum(not bool(item.get("success")) for item in self.items)

    @property
    def n_unknown(self) -> int:
        return max(0, self.declared_total - len(self.items))


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _mount_from_config(config: Dict[str, Any], meta: Dict[str, Any]) -> np.ndarray:
    robot_cfg = config.get("robot") or {}
    raw = robot_cfg.get("mount_transform")
    # Keep a metadata-level fallback so diagnostic results remain plottable if
    # a future writer promotes the resolved matrix into ``meta.robot``.
    if raw is None:
        raw = (meta.get("robot") or {}).get("mount_transform")
    if raw is None:
        raise ValueError(
            "missing meta.config.robot.mount_transform (physical ^task T_base)"
        )
    return parse_rigid_transform_matrix(raw, "robot.mount_transform")


def _raw_position(item: Dict[str, Any], mount: np.ndarray) -> np.ndarray:
    raw = item.get("position_raw")
    if raw is not None:
        position = np.asarray(raw, dtype=np.float64)
    else:
        # Older/interrupted metadata may only retain the effective planner-base
        # coordinate.  Since mount = ^task T_base, recover task coordinates as
        # p_task = R_task_base p_base + t_task_base.
        effective = item.get("effective_position", item.get("position"))
        if effective is None:
            raise ValueError(f"item index={item.get('index')} has no position")
        p_base = np.asarray(effective, dtype=np.float64)
        position = mount[:3, :3] @ p_base + mount[:3, 3]
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(
            f"item index={item.get('index')} has invalid raw position {raw!r}"
        )
    return position


def _normalise_items(
    raw_items: Sequence[Dict[str, Any]], mount: np.ndarray, source: Path
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for offset, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"{source}: item #{offset} is not an object")
        item = dict(raw_item)
        index = int(item.get("index", offset))
        if index in seen:
            raise ValueError(f"{source}: duplicate item index={index}")
        seen.add(index)
        item["index"] = index
        item["success"] = bool(item.get("success", False))
        item["position_raw"] = _raw_position(item, mount).tolist()
        items.append(item)
    return items


def load_result(path_like: str | Path) -> ResultData:
    path = Path(path_like).resolve()
    meta_path = path / "trajectory_meta.json"
    warnings: List[str] = []

    if meta_path.is_file():
        meta = _read_json(meta_path)
        config = meta.get("config") or {}
        raw_items = list(meta.get("items") or [])
        if not raw_items:
            raise ValueError(f"{meta_path} contains no items")
        grid = dict(meta.get("grid") or {})
        declared_total = int(meta.get("n_items_total") or len(raw_items))
        place_raw_value = meta.get("place_position_raw")
    else:
        failed_path = path / "plan_failed.json"
        skipped_path = path / "plan_skipped.json"
        failed = _read_json(failed_path)
        skipped = _read_json(skipped_path)
        config = failed.get("config") or {}
        raw_items = list(skipped.get("skipped") or [])
        if not raw_items:
            raise ValueError(
                f"{path}: all-failed fallback contains no skipped items"
            )
        for item in raw_items:
            item.setdefault("success", False)
            item.setdefault("n_angles_tried", item.get("n_combos_tried", 0))
        pick_place = config.get("pick_place") or {}
        grid = dict(pick_place.get("grasp_grid") or {})
        declared_total = int(
            skipped.get("n_items_total")
            or (int(grid.get("rows", 0)) * int(grid.get("cols", 0)))
            or len(raw_items)
        )
        place_raw_value = (pick_place.get("place") or {}).get("position")
        done = int(skipped.get("n_items_done") or len(raw_items))
        if done != declared_total or len(raw_items) != declared_total:
            warnings.append(
                f"incomplete all-failed record: done={done}, "
                f"recorded={len(raw_items)}, declared={declared_total}"
            )
        meta = failed

    if not isinstance(config, dict):
        raise ValueError(f"{path}: config is not an object")
    mount = _mount_from_config(config, meta)
    items = _normalise_items(raw_items, mount, path)

    if not grid:
        grid = dict(((config.get("pick_place") or {}).get("grasp_grid") or {}))

    if place_raw_value is None:
        place_raw_value = (
            ((config.get("pick_place") or {}).get("place") or {}).get("position")
        )
    place_raw: np.ndarray | None = None
    if place_raw_value is not None:
        place_raw = np.asarray(place_raw_value, dtype=np.float64)
        if place_raw.shape != (3,) or not np.all(np.isfinite(place_raw)):
            raise ValueError(f"{path}: invalid place_position_raw")

    robot_cfg = config.get("robot") or {}
    task_frame = str(
        robot_cfg.get("task_frame")
        or robot_cfg.get("task_frame_name")
        or TASK_FRAME_DEFAULT
    )

    # The resolved target transform should be the inverse of the physical
    # mount.  Warn rather than refusing to plot: the picture is useful for
    # diagnosing precisely this kind of configuration drift.
    correction_raw = (config.get("pick_place") or {}).get(
        "link0_target_transform"
    )
    if correction_raw is not None:
        correction = parse_rigid_transform_matrix(
            correction_raw, "pick_place.link0_target_transform"
        )
        residual = float(np.max(np.abs(correction @ mount - np.eye(4))))
        if residual > 1.0e-6:
            warnings.append(f"C @ mount != I (max residual {residual:.3g})")

    return ResultData(
        path=path,
        config=config,
        items=items,
        grid=grid,
        place_raw=place_raw,
        mount=mount,
        declared_total=max(declared_total, len(items)),
        task_frame=task_frame,
        warnings=warnings,
    )


def merge_replans(base: ResultData, replans: Iterable[ResultData]) -> ResultData:
    items = [dict(item) for item in base.items]
    by_index = {int(item["index"]): offset for offset, item in enumerate(items)}
    warnings = list(base.warnings)

    for replan in replans:
        mount_error = float(np.max(np.abs(replan.mount - base.mount)))
        if mount_error > 1.0e-6:
            raise ValueError(
                f"replan {replan.path} uses a different mount "
                f"(max delta {mount_error:.3g})"
            )
        for candidate in replan.items:
            index = int(candidate["index"])
            if index not in by_index:
                raise ValueError(
                    f"replan {replan.path}: index={index} is absent from {base.path}"
                )
            current = items[by_index[index]]
            if not np.allclose(
                np.asarray(current["position_raw"], dtype=np.float64),
                np.asarray(candidate["position_raw"], dtype=np.float64),
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError(
                    f"replan {replan.path}: raw position mismatch at index={index}"
                )
            if bool(candidate.get("success")) and not bool(current.get("success")):
                replacement = dict(candidate)
                # Preserve the source grid bookkeeping if a retry writer did
                # not repeat row/column fields.
                replacement.setdefault("row", current.get("row"))
                replacement.setdefault("col", current.get("col"))
                replacement["position_raw"] = current["position_raw"]
                items[by_index[index]] = replacement
        warnings.extend(replan.warnings)

    return ResultData(
        path=base.path,
        config=base.config,
        items=items,
        grid=base.grid,
        place_raw=base.place_raw,
        mount=base.mount,
        declared_total=base.declared_total,
        task_frame=base.task_frame,
        warnings=warnings,
    )


def _median_step(vectors: List[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if not vectors:
        return fallback
    result = np.median(np.stack(vectors), axis=0)
    return result if np.linalg.norm(result) > 1.0e-10 else fallback


def _cell_basis(data: ResultData) -> tuple[np.ndarray, np.ndarray]:
    positions: Dict[tuple[int, int], np.ndarray] = {}
    for item in data.items:
        if item.get("row") is None or item.get("col") is None:
            continue
        positions[(int(item["row"]), int(item["col"]))] = np.asarray(
            item["position_raw"][:2], dtype=np.float64
        )

    row_steps: List[np.ndarray] = []
    col_steps: List[np.ndarray] = []
    for (row, col), xy in positions.items():
        if (row + 1, col) in positions:
            row_steps.append(positions[(row + 1, col)] - xy)
        if (row, col + 1) in positions:
            col_steps.append(positions[(row, col + 1)] - xy)

    rows = max(1, int(data.grid.get("rows", 1)))
    cols = max(1, int(data.grid.get("cols", 1)))
    x_range = data.grid.get("x_range") or [0.0, 0.0]
    y_range = data.grid.get("y_range") or [0.0, 0.0]
    dx = (
        (float(x_range[1]) - float(x_range[0])) / (rows - 1)
        if rows > 1 else 0.02
    )
    dy = (
        (float(y_range[1]) - float(y_range[0])) / (cols - 1)
        if cols > 1 else 0.02
    )
    return (
        _median_step(row_steps, np.array([dx or 0.02, 0.0])),
        _median_step(col_steps, np.array([0.0, dy or 0.02])),
    )


def _angle_limit(results: Sequence[ResultData]) -> float:
    angles = [
        abs(float(item["angle_grasp_deg"]))
        for result in results
        for item in result.items
        if item.get("success") and item.get("angle_grasp_deg") is not None
    ]
    configured: List[float] = []
    for result in results:
        grasp = (
            (((result.config.get("pick_place") or {}).get("angle_search") or {}).get("grasp"))
            or {}
        )
        for key in ("min_deg", "max_deg"):
            if grasp.get(key) is not None:
                configured.append(abs(float(grasp[key])))
    return max([1.0, *angles, *configured])


def _near_limit_count(data: ResultData, threshold: float) -> int:
    return sum(
        bool(item.get("success"))
        and item.get("min_limit_margin_deg") is not None
        and float(item["min_limit_margin_deg"]) < threshold
        for item in data.items
    )


def _panel_extent(results: Sequence[ResultData]) -> tuple[float, float, float, float]:
    points: List[np.ndarray] = [np.array([0.0, 0.0])]
    for result in results:
        points.extend(
            np.asarray(item["position_raw"][:2], dtype=np.float64)
            for item in result.items
        )
        points.append(result.mount[:2, 3].copy())
        if result.place_raw is not None:
            points.append(result.place_raw[:2].copy())
    xy = np.stack(points)
    pad = max(0.04, 0.08 * float(np.ptp(xy, axis=0).max()))
    step = 0.02
    x0 = math.floor((float(xy[:, 0].min()) - pad) / step) * step
    x1 = math.ceil((float(xy[:, 0].max()) + pad) / step) * step
    y0 = math.floor((float(xy[:, 1].min()) - pad) / step) * step
    y1 = math.ceil((float(xy[:, 1].max()) + pad) / step) * step
    return x0, x1, y0, y1


def _draw_panel(
    ax: plt.Axes,
    data: ResultData,
    *,
    cmap,
    norm,
    annotate: str,
    near_limit_deg: float,
    language,
) -> None:
    L = language
    row_step, col_step = _cell_basis(data)
    cell_row = 0.86 * row_step
    cell_col = 0.86 * col_step

    for item in data.items:
        center = np.asarray(item["position_raw"][:2], dtype=np.float64)
        success = bool(item.get("success"))
        angle = item.get("angle_grasp_deg")
        finite_angle = success and angle is not None and np.isfinite(float(angle))
        face = cmap(norm(float(angle))) if finite_angle else "#e1e1e1"
        vertices = np.array([
            center - cell_row / 2.0 - cell_col / 2.0,
            center + cell_row / 2.0 - cell_col / 2.0,
            center + cell_row / 2.0 + cell_col / 2.0,
            center - cell_row / 2.0 + cell_col / 2.0,
        ])
        ax.add_patch(Polygon(
            vertices,
            closed=True,
            facecolor=face,
            edgecolor="#303030" if success else "#c62828",
            linewidth=0.45 if success else 0.9,
            zorder=2,
        ))

        tried = int(item.get("n_angles_tried", item.get("n_combos_tried", 0)) or 0)
        if not success:
            label = "×" if annotate != "tried" else str(tried)
            if annotate == "both":
                label = f"×\n({tried})"
            color = "#b2182b"
        elif annotate == "none":
            label, color = "", "black"
        else:
            angle_label = "?" if angle is None else f"{float(angle):+.0f}"
            if annotate == "angle":
                label = angle_label
            elif annotate == "tried":
                label = str(tried)
            else:
                label = f"{angle_label}\n({tried})"
            shade = abs(float(angle)) / max(abs(norm.vmin), abs(norm.vmax), 1.0) \
                if angle is not None else 0.0
            color = "white" if shade > 0.55 else "black"
        if label:
            rows = max(1, int(data.grid.get("rows", 1)))
            cols = max(1, int(data.grid.get("cols", 1)))
            size = max(3.2, min(9.0, 105.0 / max(rows, cols)))
            ax.text(
                center[0], center[1], label,
                ha="center", va="center", fontsize=size,
                fontweight="bold", color=color, zorder=4,
            )

    # Preserved task origin, physical base projection, and raw place.
    ax.scatter(
        [0.0], [0.0], marker="+", s=180, linewidths=2.0,
        color="#d4a600", zorder=7,
    )
    mount_xy = data.mount[:2, 3]
    ax.scatter(
        [mount_xy[0]], [mount_xy[1]], marker="v", s=190,
        facecolors="none", edgecolors="#1565c0", linewidths=2.0, zorder=8,
    )
    ax.annotate(
        L(f"实际基座\nz={data.mount[2, 3]:.2f}m",
          f"physical base\nz={data.mount[2, 3]:.2f}m"),
        xy=mount_xy, xytext=(6, 7), textcoords="offset points",
        fontsize=7.5, color="#0d47a1", zorder=9,
    )
    if data.place_raw is not None:
        ax.scatter(
            [data.place_raw[0]], [data.place_raw[1]], marker="X", s=125,
            color="#d81b60", edgecolors="black", linewidths=0.7, zorder=8,
        )

    near = _near_limit_count(data, near_limit_deg)
    grid_z = data.grid.get("z")
    rpy = matrix_to_rpy_deg(data.mount[:3, :3])
    subtitle = L(
        f"成功 {data.n_success}/{data.declared_total}  失败 {data.n_failed}"
        f"  近有效限位成功 {near} (<{near_limit_deg:g}°)\n"
        f"mount xyz={np.round(data.mount[:3, 3], 3).tolist()}  "
        f"rpy={np.round(rpy, 1).tolist()}°"
        + (f"  grid z={float(grid_z):.3f}m" if grid_z is not None else ""),
        f"success {data.n_success}/{data.declared_total}  failed {data.n_failed}"
        f"  near effective-limit {near} (<{near_limit_deg:g} deg)\n"
        f"mount xyz={np.round(data.mount[:3, 3], 3).tolist()}  "
        f"rpy={np.round(rpy, 1).tolist()} deg"
        + (f"  grid z={float(grid_z):.3f}m" if grid_z is not None else ""),
    )
    if data.n_unknown:
        subtitle += L(f"  未记录 {data.n_unknown}", f"  unrecorded {data.n_unknown}")
    name = data.path.name
    if len(name) > 52:
        name = name[:24] + "…" + name[-24:]
    ax.set_title(f"{name}\n{subtitle}", fontsize=10.2, pad=8)
    ax.set_xlabel(L(f"{data.task_frame} x (m)", f"{data.task_frame} x (m)"))
    ax.set_ylabel(L(f"{data.task_frame} y (m)", f"{data.task_frame} y (m)"))
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=5.5)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.58)
    ax.set_axisbelow(True)

    for warning in data.warnings:
        print(f"[warn] {data.path.name}: {warning}", file=sys.stderr)


def _bind_replans(
    base_paths: Sequence[str], replan_paths: Sequence[str], parser: argparse.ArgumentParser
) -> List[List[str]]:
    groups: List[List[str]] = [[] for _ in base_paths]
    if not replan_paths:
        return groups
    if len(base_paths) == 1:
        groups[0].extend(replan_paths)
        return groups
    if len(replan_paths) != len(base_paths):
        parser.error(
            "多个源结果使用重规划层时，--replan-result-dir 数量必须与源结果数量相同"
        )
    for index, path in enumerate(replan_paths):
        groups[index].append(path)
    return groups


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "result_dirs", nargs="+",
        help="one or more pick/place result directories",
    )
    parser.add_argument("--out", required=True, help="output PNG/PDF/SVG path")
    parser.add_argument(
        "--replan-result-dir", action="append", default=[],
        help="successful retry layer to merge; may be repeated",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="comparison mode (automatically enabled for multiple result dirs)",
    )
    parser.add_argument(
        "--annotate", choices=["none", "angle", "tried", "both"],
        default="angle", help="cell annotation",
    )
    parser.add_argument(
        "--near-limit-deg", type=float, default=1.0,
        help="threshold relative to clipped model bounds, used only for the near-limit count",
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    if args.near_limit_deg < 0.0:
        parser.error("--near-limit-deg must be non-negative")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    replan_groups = _bind_replans(
        args.result_dirs, args.replan_result_dir, parser
    )
    results: List[ResultData] = []
    try:
        for source, replan_group in zip(args.result_dirs, replan_groups):
            base = load_result(source)
            replans = [load_result(path) for path in replan_group]
            results.append(merge_replans(base, replans))
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    cjk = _cjk_font()
    if cjk:
        plt.rcParams["font.family"] = cjk
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    count = len(results)
    columns = 2 if count == 4 else min(3, count)
    rows = int(math.ceil(count / columns))
    fig, axes_array = plt.subplots(
        rows,
        columns,
        figsize=(8.8 * columns, 7.5 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    axes = list(axes_array.flat)
    vmax = _angle_limit(results)
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    extent = _panel_extent(results)

    for axis, result in zip(axes, results):
        _draw_panel(
            axis,
            result,
            cmap=cmap,
            norm=norm,
            annotate=args.annotate,
            near_limit_deg=float(args.near_limit_deg),
            language=L,
        )
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
    for axis in axes[count:]:
        axis.set_visible(False)

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar, ax=axes[:count], fraction=0.022, pad=0.015, shrink=0.84
    )
    colorbar.set_label(
        L("成功解采用的 grasp 角度 (°)", "adopted grasp angle for successes (deg)")
    )
    legend = [
        Patch(facecolor="#e1e1e1", edgecolor="#c62828", label=L("规划失败", "failed")),
        Line2D([0], [0], marker="v", linestyle="none", markersize=10,
               markerfacecolor="none", markeredgecolor="#1565c0",
               markeredgewidth=2.0,
               label=L("实际基座投影", "physical base projection")),
        Line2D([0], [0], marker="+", linestyle="none", markersize=11,
               markeredgewidth=2, color="#d4a600",
               label=L("保留的旧坐标原点", "preserved task origin")),
        Line2D([0], [0], marker="X", linestyle="none", markersize=9,
               markerfacecolor="#d81b60", markeredgecolor="black",
               label=L("place（原 task 坐标）", "place (raw task coordinates)")),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=4,
        fontsize=8.5,
        framealpha=0.94,
    )
    fig.suptitle(
        L(
            "XTrainer 单臂规划结果（保留的 task/world 坐标系）",
            "XTrainer single-arm results (preserved task/world frame)",
        ),
        fontsize=14,
    )
    fig.text(
        0.5, -0.095,
        L(
            (
                "多批次成功采样合并；尚未验证批次间轨迹衔接。刻度间隔 2 cm，不代表规划采样间隔。"
                if any(replan_groups) else
                "按记录顺序连续抓放的采样结果；失败不等于目标无 IK。刻度间隔 2 cm，不代表规划采样间隔。"
            ),
            (
                "Union of successful samples from separate runs; transitions between runs are unverified. "
                if any(replan_groups) else
                "Sampled sequential pick/place results; failure does not prove no IK. "
            ) + "Ticks: 2 cm, not necessarily the planning grid spacing.",
        ),
        ha="center", va="top", fontsize=8, color="#555555",
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] {output.resolve()}")
    for result in results:
        near = _near_limit_count(result, float(args.near_limit_deg))
        print(
            f"[info] {result.path.name}: success={result.n_success}/"
            f"{result.declared_total}, failed={result.n_failed}, "
            f"near_limit_success={near}, mount_xyz="
            f"{np.round(result.mount[:3, 3], 6).tolist()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
