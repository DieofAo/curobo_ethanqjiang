#!/usr/bin/env python3
"""Build the requested left/right symmetric grasp-angle map.

The source map is the 20 x 20 LINK_0 z=-7 degree run plus its two
single-arm retry layers.  Stored ``position`` coordinates are rotated by
Rz(+7 degrees) and checked against ``position_raw`` before any filtering or
mirroring is performed.

For the symmetric map, the axis lies between a configurable pair of adjacent
x cells counted from the right (11/12 by default).  Original and mirrored
cells form a union.  Where both sides occupy the same cell, the ordinary
numeric minimum of the finite angles is used; one finite value wins over one
failure, and two failures stay failed.

Run from anywhere inside or outside the repository:

    python3 xtrainer_plan/scripts/plot_symmetric_grasp_angle_map.py

    python3 xtrainer_plan/scripts/plot_symmetric_grasp_angle_map.py \
        --axis-right-cells 12 13 \
        --png-name grasp_angle_map_symmetric_right12_13.png \
        --csv-name grasp_angle_map_symmetric_right12_13.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "xtrainer_plan" / "results_pick_place"

DEFAULT_ORIGINAL_META = (
    RESULTS_ROOT
    / "link0_z_m7_20260909_000757"
    / "trajectory_meta.json"
)
DEFAULT_REPLAN_METAS = (
    RESULTS_ROOT
    / "link0_z_m7_20260909_000757_failed_single_arm_replan"
    / "trajectory_meta.json",
    RESULTS_ROOT
    / "link0_z_m7_20260909_000757_failed_single_arm_replan_grasp_to_0"
    / "trajectory_meta.json",
)
DEFAULT_OUTPUT_DIR = DEFAULT_REPLAN_METAS[-1].parent
DEFAULT_PNG_NAME = "grasp_angle_map_symmetric_min_overlap_y_ge_m012.png"
DEFAULT_CSV_NAME = "grasp_angle_map_symmetric_min_overlap_y_ge_m012.csv"

ROTATE_TO_RAW_DEG = 7.0
Y_CENTER_MIN_M = -0.12
DEFAULT_AXIS_RIGHT_CELLS = (11, 12)
X_LIMITS_M = (-0.96, 0.04)
Y_LIMITS_M = (-0.25, 0.50)
ANGLE_LIMIT_DEG = 30.0
AXIS_GRID_STEP_M = 0.02
CELL_SCALE = 0.9
DEFAULT_GREEN_STAR_X_M = -0.92
GREEN_STAR_Y_M = 0.0
YELLOW_ARM_COLOR = "#f2c500"
GREEN_ARM_COLOR = "#00b83e"

EXPECTED_SOURCE_TOTAL = 400
EXPECTED_SOURCE_FINITE = 373


@dataclass(frozen=True)
class SourceCell:
    index: int
    row: int
    col: int
    x: float
    y: float
    angle: float | None
    source_layer: str

    @property
    def success(self) -> bool:
        return self.angle is not None


@dataclass(frozen=True)
class PlotCell:
    x_index: int
    y_index: int
    x: float
    y: float
    angle: float | None
    merge_case: str
    selected_side: str
    direct: SourceCell | None
    mirrored: SourceCell | None

    @property
    def success(self) -> bool:
        return self.angle is not None


@dataclass(frozen=True)
class SymmetrySpec:
    right_cells: tuple[int, int]
    axis_x: float
    mirror_index_sum: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-meta",
        type=Path,
        default=DEFAULT_ORIGINAL_META,
        help="Original 20x20 trajectory_meta.json",
    )
    parser.add_argument(
        "--replan-meta",
        type=Path,
        action="append",
        default=None,
        help=(
            "Retry trajectory_meta.json in merge order; repeat for each layer. "
            "The two known retry layers are used by default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated PNG and CSV",
    )
    parser.add_argument("--png-name", default=DEFAULT_PNG_NAME)
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME)
    parser.add_argument(
        "--axis-right-cells",
        type=int,
        nargs=2,
        default=DEFAULT_AXIS_RIGHT_CELLS,
        metavar=("FIRST", "SECOND"),
        help=(
            "Adjacent 1-based x-cell numbers counted from the right whose "
            "shared boundary is the symmetry axis (default: 11 12)"
        ),
    )
    parser.add_argument(
        "--green-star-x",
        type=float,
        default=DEFAULT_GREEN_STAR_X_M,
        help="Green star x coordinate in metres (default: -0.92)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def read_meta(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing source file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("items"), list):
        raise ValueError(f"{path} does not contain an items list")
    return document


def finite_angle(item: dict[str, Any]) -> float | None:
    value = item.get("angle_grasp_deg")
    if not item.get("success") or value is None:
        return None
    angle = float(value)
    if not math.isfinite(angle):
        raise ValueError(f"item index={item.get('index')} has non-finite angle {value!r}")
    return angle


def validate_item_identity(
    reference: dict[str, Any], candidate: dict[str, Any], source_path: Path
) -> None:
    index = int(reference["index"])
    for field in ("row", "col"):
        if int(reference[field]) != int(candidate[field]):
            raise ValueError(
                f"{source_path}: index={index} has inconsistent {field}: "
                f"{candidate[field]} != {reference[field]}"
            )
    for field in ("position", "position_raw"):
        reference_position = np.asarray(reference.get(field), dtype=float)
        candidate_position = np.asarray(candidate.get(field), dtype=float)
        if reference_position.shape != (3,) or candidate_position.shape != (3,):
            raise ValueError(f"{source_path}: index={index} has invalid {field}")
        if not np.allclose(
            reference_position, candidate_position, rtol=0.0, atol=1e-10
        ):
            raise ValueError(
                f"{source_path}: index={index} has inconsistent {field}"
            )


def merge_source_layers(
    original_path: Path, replan_paths: list[Path]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[tuple[str, int, int]]]:
    original_meta = read_meta(original_path)
    original_items = [dict(item) for item in original_meta["items"]]
    if len(original_items) != EXPECTED_SOURCE_TOTAL:
        raise ValueError(
            f"expected {EXPECTED_SOURCE_TOTAL} original cells, got {len(original_items)}"
        )

    offsets: dict[int, int] = {}
    occupied: set[tuple[int, int]] = set()
    for offset, item in enumerate(original_items):
        index = int(item["index"])
        cell = (int(item["row"]), int(item["col"]))
        if index in offsets:
            raise ValueError(f"duplicate original item index={index}")
        if cell in occupied:
            raise ValueError(f"duplicate original grid cell={cell}")
        offsets[index] = offset
        occupied.add(cell)
        item["_source_layer"] = "original"

    layer_stats: list[tuple[str, int, int]] = [
        (
            "original",
            len(original_items),
            sum(finite_angle(item) is not None for item in original_items),
        )
    ]
    merged_items = original_items
    for layer_number, path in enumerate(replan_paths, start=1):
        meta = read_meta(path)
        seen: set[int] = set()
        recovered = 0
        source_layer = path.parent.name
        for candidate in meta["items"]:
            index = int(candidate["index"])
            if index in seen:
                raise ValueError(f"{path}: duplicate item index={index}")
            seen.add(index)
            if index not in offsets:
                raise ValueError(f"{path}: unknown item index={index}")
            offset = offsets[index]
            validate_item_identity(original_items[offset], candidate, path)
            candidate_angle = finite_angle(candidate)
            if candidate_angle is None:
                continue
            if finite_angle(merged_items[offset]) is not None:
                raise ValueError(
                    f"{path}: successful index={index} was already successful "
                    "in an earlier layer"
                )
            replacement = dict(candidate)
            replacement["_source_layer"] = source_layer
            merged_items[offset] = replacement
            recovered += 1
        layer_stats.append((f"replan_{layer_number}", len(meta["items"]), recovered))

    merged_finite = sum(finite_angle(item) is not None for item in merged_items)
    if merged_finite != EXPECTED_SOURCE_FINITE:
        raise ValueError(
            f"expected {EXPECTED_SOURCE_FINITE} finite merged source cells, "
            f"got {merged_finite}"
        )
    return merged_items, original_meta, layer_stats


def rotate_and_index_source(
    items: list[dict[str, Any]],
    axis_right_cells: tuple[int, int],
) -> tuple[
    dict[tuple[int, int], SourceCell],
    dict[int, float],
    dict[int, float],
    SymmetrySpec,
    float,
]:
    theta = math.radians(ROTATE_TO_RAW_DEG)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=float,
    )
    transformed_by_cell: dict[tuple[int, int], np.ndarray] = {}
    max_raw_error = 0.0
    for item in items:
        row = int(item["row"])
        col = int(item["col"])
        position = np.asarray(item.get("position"), dtype=float)
        position_raw = np.asarray(item.get("position_raw"), dtype=float)
        if position.shape != (3,) or position_raw.shape != (3,):
            raise ValueError(f"index={item.get('index')} has invalid position data")
        transformed = rotation @ position[:2]
        raw_error = float(np.max(np.abs(transformed - position_raw[:2])))
        max_raw_error = max(max_raw_error, raw_error)
        if raw_error > 1e-10:
            raise ValueError(
                f"Rz(+{ROTATE_TO_RAW_DEG:g} deg) does not recover position_raw "
                f"for index={item.get('index')}; max error={raw_error:.3g} m"
            )
        transformed_by_cell[(row, col)] = transformed

    rows = sorted({row for row, _ in transformed_by_cell})
    cols = sorted({col for _, col in transformed_by_cell})
    if rows != list(range(20)) or cols != list(range(20)):
        raise ValueError(f"expected row/col indices 0..19, got rows={rows}, cols={cols}")

    # Average repeated coordinates after the explicit rotation.  This both
    # suppresses roundoff at ~1e-16 m and verifies that row maps to x while
    # column maps to y in this data set.
    x_by_row = {
        row: float(np.mean([xy[0] for (r, _), xy in transformed_by_cell.items() if r == row]))
        for row in rows
    }
    y_by_col = {
        col: float(np.mean([xy[1] for (_, c), xy in transformed_by_cell.items() if c == col]))
        for col in cols
    }
    for (row, col), xy in transformed_by_cell.items():
        if not np.allclose(
            xy, [x_by_row[row], y_by_col[col]], rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"transformed grid is not axis-aligned at row={row}, col={col}")

    first_from_right, second_from_right = axis_right_cells
    if (
        first_from_right < 1
        or second_from_right != first_from_right + 1
        or second_from_right > len(rows)
    ):
        raise ValueError(
            "--axis-right-cells must be two adjacent, ascending 1-based cell "
            f"numbers within 1..{len(rows)}; got {axis_right_cells}"
        )
    rows_by_x_descending = sorted(rows, key=x_by_row.__getitem__, reverse=True)
    first_row = rows_by_x_descending[first_from_right - 1]
    second_row = rows_by_x_descending[second_from_right - 1]
    symmetry = SymmetrySpec(
        right_cells=(first_from_right, second_from_right),
        axis_x=(x_by_row[first_row] + x_by_row[second_row]) / 2.0,
        mirror_index_sum=first_row + second_row,
    )

    source: dict[tuple[int, int], SourceCell] = {}
    for item in items:
        row = int(item["row"])
        col = int(item["col"])
        source[(row, col)] = SourceCell(
            index=int(item["index"]),
            row=row,
            col=col,
            x=x_by_row[row],
            y=y_by_col[col],
            angle=finite_angle(item),
            source_layer=str(item["_source_layer"]),
        )
    return source, x_by_row, y_by_col, symmetry, max_raw_error


def combine_candidates(
    direct: SourceCell | None, mirrored: SourceCell | None
) -> tuple[float | None, str, str]:
    direct_angle = direct.angle if direct is not None else None
    mirrored_angle = mirrored.angle if mirrored is not None else None

    if direct is not None and mirrored is not None:
        if direct_angle is not None and mirrored_angle is not None:
            angle = min(direct_angle, mirrored_angle)
            if direct_angle < mirrored_angle:
                selected = "direct"
            elif mirrored_angle < direct_angle:
                selected = "mirrored"
            else:
                selected = "both_equal"
            return angle, "overlap_numeric_min", selected
        if direct_angle is not None:
            return direct_angle, "overlap_finite_over_failed", "direct"
        if mirrored_angle is not None:
            return mirrored_angle, "overlap_finite_over_failed", "mirrored"
        return None, "overlap_both_failed", "failed"

    if direct is not None:
        return direct_angle, "direct_only", "direct" if direct.success else "failed"
    if mirrored is not None:
        return (
            mirrored_angle,
            "mirrored_only",
            "mirrored" if mirrored.success else "failed",
        )
    raise AssertionError("plot cell has neither a direct nor a mirrored source")


def build_symmetric_cells(
    source: dict[tuple[int, int], SourceCell],
    x_by_row: dict[int, float],
    y_by_col: dict[int, float],
    symmetry: SymmetrySpec,
) -> list[PlotCell]:
    kept_cols = [
        col
        for col, y in sorted(y_by_col.items())
        if y >= Y_CENTER_MIN_M - 1e-12
    ]
    candidates: dict[tuple[int, int], dict[str, SourceCell]] = {}
    for (row, col), cell in source.items():
        if col not in kept_cols:
            continue
        candidates.setdefault((row, col), {})["direct"] = cell
        mirrored_row = symmetry.mirror_index_sum - row
        candidates.setdefault((mirrored_row, col), {})["mirrored"] = cell

    plot_cells: list[PlotCell] = []
    for (x_index, y_index), sides in sorted(candidates.items(), key=lambda pair: pair[0]):
        direct = sides.get("direct")
        mirrored = sides.get("mirrored")
        angle, merge_case, selected_side = combine_candidates(direct, mirrored)

        if x_index in x_by_row:
            x = x_by_row[x_index]
        else:
            reflected_source_row = symmetry.mirror_index_sum - x_index
            x = 2.0 * symmetry.axis_x - x_by_row[reflected_source_row]
        y = y_by_col[y_index]

        if mirrored is not None:
            reflected_x = 2.0 * symmetry.axis_x - mirrored.x
            if not math.isclose(x, reflected_x, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"mirrored coordinate mismatch at x_index={x_index}: "
                    f"{x:.17g} != {reflected_x:.17g}"
                )

        plot_cells.append(
            PlotCell(
                x_index=x_index,
                y_index=y_index,
                x=x,
                y=y,
                angle=angle,
                merge_case=merge_case,
                selected_side=selected_side,
                direct=direct,
                mirrored=mirrored,
            )
        )
    validate_output(plot_cells, symmetry)
    return plot_cells


def validate_output(cells: list[PlotCell], symmetry: SymmetrySpec) -> None:
    if not cells:
        raise ValueError("symmetric output is empty")
    x_indices = sorted({cell.x_index for cell in cells})
    y_indices = sorted({cell.y_index for cell in cells})
    expected_total = len(x_indices) * len(y_indices)
    if len(cells) != expected_total:
        raise ValueError(
            f"symmetric output is not rectangular: expected {expected_total} "
            f"cells from {len(x_indices)}x{len(y_indices)}, got {len(cells)}"
        )
    by_index = {(cell.x_index, cell.y_index): cell for cell in cells}
    for cell in cells:
        reflected_key = (
            symmetry.mirror_index_sum - cell.x_index,
            cell.y_index,
        )
        reflected = by_index.get(reflected_key)
        if reflected is None:
            raise ValueError(f"missing reflected output cell for {reflected_key}")
        if cell.angle != reflected.angle:
            raise ValueError(
                f"asymmetric merged angle at {(cell.x_index, cell.y_index)} "
                f"and {reflected_key}: {cell.angle} != {reflected.angle}"
            )
        if not math.isclose(
            cell.x + reflected.x,
            2.0 * symmetry.axis_x,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"asymmetric coordinates at {(cell.x_index, cell.y_index)} "
                f"and {reflected_key}"
            )


def optional(value: Any) -> Any:
    return "" if value is None else value


def write_csv(cells: list[PlotCell], path: Path, symmetry: SymmetrySpec) -> None:
    fieldnames = [
        "plot_x_index",
        "plot_y_index",
        "x_m",
        "y_m",
        "success",
        "angle_grasp_deg",
        "merge_case",
        "selected_side",
        "symmetry_axis_x_m",
        "direct_source_index",
        "direct_source_layer",
        "direct_success",
        "direct_angle_grasp_deg",
        "mirrored_source_index",
        "mirrored_source_layer",
        "mirrored_success",
        "mirrored_angle_grasp_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for cell in sorted(cells, key=lambda item: (item.y_index, item.x_index)):
            direct = cell.direct
            mirrored = cell.mirrored
            writer.writerow(
                {
                    "plot_x_index": cell.x_index,
                    "plot_y_index": cell.y_index,
                    "x_m": f"{cell.x:.15g}",
                    "y_m": f"{cell.y:.15g}",
                    "success": cell.success,
                    "angle_grasp_deg": optional(cell.angle),
                    "merge_case": cell.merge_case,
                    "selected_side": cell.selected_side,
                    "symmetry_axis_x_m": f"{symmetry.axis_x:.15g}",
                    "direct_source_index": optional(direct.index if direct else None),
                    "direct_source_layer": optional(direct.source_layer if direct else None),
                    "direct_success": optional(direct.success if direct else None),
                    "direct_angle_grasp_deg": optional(direct.angle if direct else None),
                    "mirrored_source_index": optional(mirrored.index if mirrored else None),
                    "mirrored_source_layer": optional(
                        mirrored.source_layer if mirrored else None
                    ),
                    "mirrored_success": optional(mirrored.success if mirrored else None),
                    "mirrored_angle_grasp_deg": optional(
                        mirrored.angle if mirrored else None
                    ),
                }
            )


def pick_cjk_font() -> str | None:
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
    available = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in preferred if name in available), None)


def plot(
    cells: list[PlotCell],
    output_path: Path,
    z_m: float,
    dpi: int,
    symmetry: SymmetrySpec,
    green_star_x: float,
) -> None:
    cjk = pick_cjk_font()
    if cjk:
        plt.rcParams["font.family"] = cjk
    plt.rcParams["axes.unicode_minus"] = False
    label = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    x_values = sorted({cell.x for cell in cells})
    y_values = sorted({cell.y for cell in cells})
    finite_count = sum(cell.success for cell in cells)
    failed_count = len(cells) - finite_count
    cell_width = float(np.median(np.diff(x_values))) * CELL_SCALE
    cell_height = float(np.median(np.diff(y_values))) * CELL_SCALE

    fig, axis = plt.subplots(figsize=(9.5, 7.5))
    norm = plt.Normalize(vmin=-ANGLE_LIMIT_DEG, vmax=ANGLE_LIMIT_DEG)
    cmap = plt.get_cmap("coolwarm")
    text_size = max(3.5, min(10.5, 120.0 / max(len(x_values), len(y_values))))

    for cell in cells:
        if cell.success:
            facecolor = cmap(norm(cell.angle))
            if (
                cell.merge_case == "overlap_finite_over_failed"
                and cell.direct is not None
                and not cell.direct.success
            ):
                # 黄星（原始数据）一侧本来失败，最终由镜像侧成功值覆盖。
                edgecolor = YELLOW_ARM_COLOR
                linewidth = 1.8
                zorder = 3
            elif (
                cell.merge_case == "overlap_finite_over_failed"
                and cell.mirrored is not None
                and not cell.mirrored.success
            ):
                # 绿星（镜像数据）一侧本来失败，最终由原始侧成功值覆盖。
                edgecolor = GREEN_ARM_COLOR
                linewidth = 1.8
                zorder = 3
            else:
                edgecolor = "0.25"
                linewidth = 0.35
                zorder = 2
        else:
            facecolor = "0.88"
            edgecolor = "red"
            linewidth = 0.8
            zorder = 3
        axis.add_patch(
            Rectangle(
                (cell.x - cell_width / 2.0, cell.y - cell_height / 2.0),
                cell_width,
                cell_height,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=zorder,
            )
        )
        if cell.success:
            annotation = f"{cell.angle:+.0f}"
            text_color = "white" if abs(cell.angle) / ANGLE_LIMIT_DEG > 0.55 else "black"
        else:
            annotation = "×"
            text_color = "#b2182b"
        axis.text(
            cell.x,
            cell.y,
            annotation,
            ha="center",
            va="center",
            fontsize=text_size,
            fontweight="bold",
            color=text_color,
            zorder=4,
        )

    axis.axvline(
        symmetry.axis_x,
        color="0.28",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        zorder=1,
        label=label(
            f"轴 x={symmetry.axis_x:.4f}m",
            f"axis x={symmetry.axis_x:.4f}m",
        ),
    )
    axis.scatter(
        [green_star_x],
        [GREEN_STAR_Y_M],
        s=210,
        marker="*",
        color=GREEN_ARM_COLOR,
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
        label=label(
            f"绿星 ({green_star_x:g}, 0)",
            f"green star ({green_star_x:g}, 0)",
        ),
    )
    axis.scatter(
        [0.0],
        [0.0],
        s=200,
        marker="*",
        color=YELLOW_ARM_COLOR,
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label=label("黄星 (0, 0)", "yellow star (0, 0)"),
    )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(np.asarray([cell.angle for cell in cells if cell.success]))
    colorbar = fig.colorbar(scalar_mappable, ax=axis, pad=0.02)
    colorbar.set_label(label("绕工具 X 轴转角 (度)", "rotation about tool X (deg)"))

    axis.set_title(
        label(
            "XTrainer 对称抓取位姿角分布 (grasp)\n"
            f"对称并集 {len(x_values)}x{len(y_values)} @ z={z_m:.2f}m   "
            f"有效 {finite_count}/{len(cells)}\n"
            f"轴 x={symmetry.axis_x:.6f}m   重叠取数值较小者   "
            "点位左乘 Rz(+7°)",
            "XTrainer symmetric grasp angle map\n"
            f"symmetric union {len(x_values)}x{len(y_values)} @ z={z_m:.2f}m   "
            f"finite {finite_count}/{len(cells)}\n"
            f"axis x={symmetry.axis_x:.6f}m   numeric min on overlap   "
            "Rz(+7 deg)",
        ),
        fontsize=12.5,
        pad=12,
    )
    axis.set_xlabel("LINK_0 x (m)")
    axis.set_ylabel("LINK_0 y (m)")
    axis.set_xlim(*X_LIMITS_M)
    axis.set_ylim(*Y_LIMITS_M)
    axis.set_aspect("equal", adjustable="box")
    axis.xaxis.set_major_locator(MultipleLocator(AXIS_GRID_STEP_M))
    axis.yaxis.set_major_locator(MultipleLocator(AXIS_GRID_STEP_M))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.tick_params(axis="x", labelrotation=45, labelsize=7)
    axis.tick_params(axis="y", labelsize=7)
    axis.grid(True, which="major", linestyle=":", linewidth=0.55, alpha=0.55)
    axis.set_axisbelow(True)

    failed_patch = Patch(
        facecolor="0.88",
        edgecolor="red",
        linewidth=0.8,
        label=label(f"规划失败 ({failed_count})", f"failed ({failed_count})"),
    )
    yellow_covered_count = sum(
        cell.merge_case == "overlap_finite_over_failed"
        and cell.direct is not None
        and not cell.direct.success
        for cell in cells
    )
    green_covered_count = sum(
        cell.merge_case == "overlap_finite_over_failed"
        and cell.mirrored is not None
        and not cell.mirrored.success
        for cell in cells
    )
    yellow_covered_patch = Patch(
        facecolor="none",
        edgecolor=YELLOW_ARM_COLOR,
        linewidth=1.8,
        label=label(
            f"黄框：黄星侧失败→覆盖 ({yellow_covered_count})",
            f"yellow: yellow side failed→covered ({yellow_covered_count})",
        ),
    )
    green_covered_patch = Patch(
        facecolor="none",
        edgecolor=GREEN_ARM_COLOR,
        linewidth=1.8,
        label=label(
            f"绿框：绿星侧失败→覆盖 ({green_covered_count})",
            f"green: green side failed→covered ({green_covered_count})",
        ),
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        [failed_patch, yellow_covered_patch, green_covered_patch, *handles],
        [
            failed_patch.get_label(),
            yellow_covered_patch.get_label(),
            green_covered_patch.get_label(),
            *labels,
        ],
        loc="upper right",
        fontsize=6.5,
        borderpad=0.35,
        labelspacing=0.3,
        handlelength=1.7,
        handletextpad=0.45,
        markerscale=0.75,
        framealpha=0.9,
    )
    fig.text(
        0.5,
        0.015,
        label(
            "标注 = 对称合并后的抓取角；重叠格取两侧有限值中的普通数值较小者",
            "labels = merged grasp angles; overlap uses the smaller finite numeric value",
        ),
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    replan_paths = list(args.replan_meta or DEFAULT_REPLAN_METAS)
    items, original_meta, layer_stats = merge_source_layers(
        args.original_meta.resolve(), [path.resolve() for path in replan_paths]
    )
    source, x_by_row, y_by_col, symmetry, max_raw_error = rotate_and_index_source(
        items, tuple(args.axis_right_cells)
    )
    cells = build_symmetric_cells(source, x_by_row, y_by_col, symmetry)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / args.png_name
    csv_path = output_dir / args.csv_name
    write_csv(cells, csv_path, symmetry)
    z_m = float((original_meta.get("grid") or {}).get("z", 0.03))
    plot(
        cells,
        png_path,
        z_m=z_m,
        dpi=args.dpi,
        symmetry=symmetry,
        green_star_x=args.green_star_x,
    )

    frequency = Counter(cell.angle for cell in cells)
    ordered_frequency = {
        ("failed" if angle is None else f"{angle:g}"): frequency[angle]
        for angle in sorted(frequency, key=lambda value: (value is None, value or 0.0))
    }
    print(f"[ok] PNG: {png_path}")
    print(f"[ok] CSV: {csv_path}")
    print(f"[info] source layers (name, items, contributed finite): {layer_stats}")
    print(
        f"[info] Rz(+{ROTATE_TO_RAW_DEG:g} deg) -> raw grid; "
        f"max coordinate error={max_raw_error:.3e} m"
    )
    print(
        f"[info] symmetry axis between right-counted cells "
        f"{symmetry.right_cells[0]}/{symmetry.right_cells[1]}: "
        f"x={symmetry.axis_x:.17g} m; "
        f"output={len({cell.x_index for cell in cells})}x"
        f"{len({cell.y_index for cell in cells})}={len(cells)}, "
        f"finite={sum(cell.success for cell in cells)}, "
        f"failed={sum(not cell.success for cell in cells)}"
    )
    print(f"[info] angle frequency: {ordered_frequency}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
