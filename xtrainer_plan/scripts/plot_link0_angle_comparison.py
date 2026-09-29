#!/usr/bin/env python3
"""Compare pick/place success distributions for several LINK_0 rotations.

Each run is supplied as ``--run=ANGLE=RESULT_DIR``.  Successful runs are read
from ``trajectory_meta.json``.  A completed all-failure run is reconstructed
from ``plan_skipped.json`` and checked against ``plan_failed.json``.

Example:

    python3 xtrainer_plan/scripts/plot_link0_angle_comparison.py \
        --run=-14=xtrainer_plan/results_pick_place/link0_z_m14_retry1 \
        --run=-15=xtrainer_plan/results_pick_place/link0_z_m15_retry1 \
        --run=-16=xtrainer_plan/results_pick_place/link0_z_m16_retry1 \
        --output-dir xtrainer_plan/experiments/my_sweep/plots
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Point:
    row: int
    col: int
    x: float
    y: float
    success: bool


@dataclass(frozen=True)
class RunResult:
    angle_deg: float
    result_dir: Path
    points: tuple[Point, ...]
    total: int
    successes: int
    coordinate_mode: str

    @property
    def success_rate(self) -> float:
        return 100.0 * self.successes / self.total if self.total else 0.0


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def parse_run_spec(spec: str) -> tuple[float, Path]:
    try:
        angle_text, directory = spec.split("=", 1)
        angle_text = angle_text.strip()
        if angle_text.endswith("°"):
            angle_text = angle_text[:-1]
        if angle_text.endswith("deg"):
            angle_text = angle_text[:-3]
        angle = float(angle_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid run {spec!r}; expected ANGLE=RESULT_DIR"
        ) from exc
    if not directory:
        raise argparse.ArgumentTypeError(
            f"invalid run {spec!r}; RESULT_DIR must not be empty"
        )
    return angle, Path(directory).expanduser()


def normalize_items(items: list[dict[str, Any]]) -> tuple[tuple[Point, ...], str]:
    if not items:
        raise ValueError("result contains no grid items")

    use_raw_coordinates = all(
        isinstance(item.get("position_raw"), list)
        and len(item["position_raw"]) >= 2
        for item in items
    )
    coordinate_mode = "raw" if use_raw_coordinates else "grid"
    points: list[Point] = []
    occupied_cells: set[tuple[int, int]] = set()

    for item in items:
        try:
            row = int(item["row"])
            col = int(item["col"])
            if use_raw_coordinates:
                x = float(item["position_raw"][0])
                y = float(item["position_raw"][1])
            else:
                x = float(col)
                y = float(row)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"invalid grid item: {item!r}") from exc

        cell = (row, col)
        if cell in occupied_cells:
            raise ValueError(f"duplicate grid cell row={row}, col={col}")
        occupied_cells.add(cell)
        points.append(
            Point(row=row, col=col, x=x, y=y, success=bool(item.get("success")))
        )

    points.sort(key=lambda point: (point.row, point.col))
    return tuple(points), coordinate_mode


def load_run(angle_deg: float, result_dir: Path) -> RunResult:
    result_dir = result_dir.resolve()
    meta_path = result_dir / "trajectory_meta.json"

    if meta_path.is_file():
        document = read_json(meta_path)
        items = list(document.get("items") or [])
        total = int(document.get("n_items_total") or len(items))
    else:
        skipped_path = result_dir / "plan_skipped.json"
        failed_path = result_dir / "plan_failed.json"
        skipped_document = read_json(skipped_path)
        failed_document = read_json(failed_path)
        skipped = list(skipped_document.get("skipped") or [])
        total = int(skipped_document.get("n_items_total") or len(skipped))
        done = int(skipped_document.get("n_items_done") or len(skipped))

        if failed_document.get("stage") != "no_item_success":
            raise ValueError(
                f"{result_dir} has no trajectory_meta.json and is not marked "
                "as a no_item_success run"
            )
        if done != total or len(skipped) != total:
            raise ValueError(
                f"{result_dir} is incomplete: done={done}, skipped={len(skipped)}, "
                f"total={total}"
            )

        items = []
        for skipped_item in skipped:
            item = dict(skipped_item)
            item["success"] = False
            items.append(item)

    if total <= 0:
        raise ValueError(f"{result_dir} has invalid total={total}")
    if len(items) != total:
        raise ValueError(
            f"{result_dir} has {len(items)} plotted items but reports total={total}"
        )

    points, coordinate_mode = normalize_items(items)
    successes = sum(point.success for point in points)
    reported_successes = document.get("n_items_success") if meta_path.is_file() else 0
    if reported_successes is not None and int(reported_successes) != successes:
        raise ValueError(
            f"{result_dir} reports {reported_successes} successes but its items "
            f"contain {successes}"
        )

    return RunResult(
        angle_deg=angle_deg,
        result_dir=result_dir,
        points=points,
        total=total,
        successes=successes,
        coordinate_mode=coordinate_mode,
    )


def angle_label(angle_deg: float) -> str:
    return f"{angle_deg:g}°"


def padded_limits(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = high - low
    padding = 0.04 * span if span else 0.5
    return low - padding, high + padding


def plot_spatial(runs: list[RunResult], output_path: Path, dpi: int) -> None:
    n_runs = len(runs)
    fig, axes = plt.subplots(
        1,
        n_runs,
        figsize=(5.15 * n_runs, 5.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_row = axes[0]

    all_x = [point.x for run in runs for point in run.points]
    all_y = [point.y for run in runs for point in run.points]
    x_limits = padded_limits(all_x)
    y_limits = padded_limits(all_y)
    raw_coordinates = runs[0].coordinate_mode == "raw"

    for axis, run in zip(axes_row, runs):
        failures = [point for point in run.points if not point.success]
        successes = [point for point in run.points if point.success]

        # A neutral marker makes every sampled grid position visible.  Successes
        # cover it in green; failures retain the grey base and receive a red x.
        axis.scatter(
            [point.x for point in run.points],
            [point.y for point in run.points],
            s=35,
            marker="o",
            facecolor="#d8d8d8",
            edgecolor="#a8a8a8",
            linewidth=0.35,
            zorder=1,
        )
        if successes:
            axis.scatter(
                [point.x for point in successes],
                [point.y for point in successes],
                s=30,
                marker="o",
                facecolor="#2ca25f",
                edgecolor="#187a43",
                linewidth=0.35,
                zorder=2,
            )
        if failures:
            axis.scatter(
                [point.x for point in failures],
                [point.y for point in failures],
                s=30,
                marker="x",
                color="#d73027",
                linewidth=0.9,
                zorder=3,
            )

        axis.set_title(
            f"LINK_0 rotation {angle_label(run.angle_deg)}\n"
            f"{run.successes}/{run.total} succeeded ({run.success_rate:.2f}%)",
            fontsize=11,
        )
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#ececec", linewidth=0.6, zorder=0)
        axis.set_xlabel("Raw x (m)" if raw_coordinates else "Grid column")

    axes_row[0].set_ylabel("Raw y (m)" if raw_coordinates else "Grid row")
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            markerfacecolor="#2ca25f", markeredgecolor="#187a43",
            label="Success",
        ),
        Line2D(
            [0], [0], marker="x", linestyle="none", markersize=7,
            markeredgewidth=1.2, color="#d73027", label="Failure",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
    )
    fig.suptitle(
        "Pick/place spatial outcome by LINK_0 target rotation",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 0.93))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_success_rates(
    runs: list[RunResult], output_path: Path, dpi: int
) -> None:
    labels = [angle_label(run.angle_deg) for run in runs]
    rates = [run.success_rate for run in runs]
    fig, axis = plt.subplots(figsize=(max(6.5, 1.6 * len(runs)), 5.2), constrained_layout=True)
    bars = axis.bar(labels, rates, width=0.62, color="#2ca25f", edgecolor="#187a43")

    axis.set_ylim(0.0, 100.0)
    axis.set_xlabel("LINK_0 target rotation")
    axis.set_ylabel("Success rate (%)")
    axis.set_title("Pick/place success rate by LINK_0 target rotation")
    axis.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    axis.set_axisbelow(True)

    for bar, run in zip(bars, runs):
        annotation_y = max(run.success_rate, 1.0) + 1.5
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            annotation_y,
            f"{run.success_rate:.2f}%\n({run.successes}/{run.total})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def default_output_dir(run_specs: list[tuple[float, Path]]) -> Path:
    resolved_dirs = [str(directory.resolve()) for _, directory in run_specs]
    common_parent = Path(os.path.commonpath(resolved_dirs))
    if common_parent in [directory.resolve() for _, directory in run_specs]:
        common_parent = common_parent.parent
    return common_parent / "link0_angle_comparison_plots"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
        metavar="ANGLE=RESULT_DIR",
        help=(
            "angle and result directory; repeat for every run. For negative "
            "angles use --run=-14=/path/to/result"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: a comparison directory beside the runs)",
    )
    parser.add_argument("--prefix", default="link0_angle_comparison")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if len(args.run) < 2:
        parser.error("at least two --run arguments are required")

    angles = [angle for angle, _ in args.run]
    if len(set(angles)) != len(angles):
        parser.error("each --run angle must be unique")

    try:
        runs = [load_run(angle, directory) for angle, directory in args.run]
    except ValueError as exc:
        parser.error(str(exc))

    coordinate_modes = {run.coordinate_mode for run in runs}
    if len(coordinate_modes) != 1:
        parser.error("runs do not use the same coordinate representation")

    output_dir = (args.output_dir or default_output_dir(args.run)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_path = output_dir / f"{args.prefix}_spatial.png"
    rate_path = output_dir / f"{args.prefix}_success_rate.png"

    plot_spatial(runs, spatial_path, args.dpi)
    plot_success_rates(runs, rate_path, args.dpi)

    for run in runs:
        print(
            f"{angle_label(run.angle_deg):>6}: "
            f"{run.successes}/{run.total} ({run.success_rate:.2f}%)  "
            f"{run.result_dir}"
        )
    print(f"spatial plot: {spatial_path}")
    print(f"rate plot:    {rate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
