#!/usr/bin/env python3
"""Plot CSV outputs from analyze_relative_ee.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create plots from a relative_ee_analysis results directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show", action="store_true", help="Open interactive plot windows.")
    return parser.parse_args()


def parse_value(value: str) -> float:
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    if value == "":
        return float("nan")
    return float(value)


def read_csv_numeric(path: Path) -> Dict[str, np.ndarray]:
    with path.open(newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = list(reader)
    if not rows:
        return {name: np.asarray([], dtype=np.float64) for name in reader.fieldnames or []}

    data: Dict[str, List[float]] = {name: [] for name in reader.fieldnames or []}
    for row in rows:
        for name, value in row.items():
            data[name].append(parse_value(value))
    return {name: np.asarray(values, dtype=np.float64) for name, values in data.items()}


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def grouped_mean(
    group_index: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_index = group_index.astype(np.int64)
    unique = np.unique(group_index)
    means = np.full(unique.shape, np.nan, dtype=np.float64)
    counts = np.zeros(unique.shape, dtype=np.float64)
    for i, group in enumerate(unique):
        group_values = finite(values[group_index == group])
        counts[i] = group_values.size
        if group_values.size:
            means[i] = np.mean(group_values)
    return unique, means, counts


def grouped_failure_count(group_index: np.ndarray, success: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    group_index = group_index.astype(np.int64)
    unique = np.unique(group_index)
    failure_count = np.zeros(unique.shape, dtype=np.float64)
    success_bool = success.astype(bool)
    for i, group in enumerate(unique):
        failure_count[i] = np.count_nonzero(~success_bool[group_index == group])
    return unique, failure_count


def savefig(fig, output_dir: Path, name: str, show: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if show:
        fig.show()
    print(f"wrote {path}")


def plot_dual_axis_by_q(
    q_index: np.ndarray,
    trans: np.ndarray,
    rot: np.ndarray,
    output_dir: Path,
    output_name: str,
    title: str,
    show: bool,
    trans_label: str = "translation [mm]",
    rot_label: str = "rotation [deg]",
    failed_q: Optional[np.ndarray] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(q_index, trans, label="translation", color="tab:blue")
    ax1.set_xlabel("q index")
    ax1.set_ylabel(trans_label, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(q_index, rot, label="rotation", color="tab:orange")
    ax2.set_ylabel(rot_label, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    if failed_q is not None and failed_q.size:
        for q in failed_q:
            ax1.axvline(q, color="tab:red", alpha=0.22, linewidth=1.0)
        ax2.scatter(
            failed_q,
            np.interp(failed_q, q_index, rot),
            color="tab:red",
            marker="x",
            s=30,
            label="q has IK failure",
            zorder=3,
        )
        ax2.legend()

    ax1.set_title(title)
    savefig(fig, output_dir, output_name, show)


def plot_q_samples(q_csv: Path, output_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    data = read_csv_numeric(q_csv)
    if not data:
        return
    q_index = data["q_index"]
    joint_names = [
        name
        for name in data
        if name.startswith("J_") or (name.startswith("q") and name[1:].isdigit())
    ]
    if not joint_names:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in joint_names:
        ax.plot(q_index, data[name], linewidth=1.2, label=name)
    ax.set_title("Joint Positions From MCAP")
    ax.set_xlabel("q index")
    ax.set_ylabel("joint position [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(3, len(joint_names)))
    savefig(fig, output_dir, "q_samples_joints.png", show)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in joint_names:
        values = data[name]
        ax.plot(q_index, values - values[0], linewidth=1.2, label=name)
    ax.set_title("Joint Position Change From First Sample")
    ax.set_xlabel("q index")
    ax.set_ylabel("delta joint position [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(3, len(joint_names)))
    savefig(fig, output_dir, "q_samples_joint_deltas.png", show)

    fig, axes = plt.subplots(len(joint_names), 1, figsize=(11, 1.8 * len(joint_names)), sharex=True)
    if len(joint_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, joint_names):
        ax.plot(q_index, data[name], linewidth=1.0)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Joint Positions, One Axis Per Joint")
    axes[-1].set_xlabel("q index")
    savefig(fig, output_dir, "q_samples_joints_separate_axes.png", show)


def plot_target1(target1_csv: Path, output_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    data = read_csv_numeric(target1_csv)
    q_index = data["q_index"]
    trans = data["translation_error_mm"]
    rot = data["rotation_error_deg"]

    if "relative_index" in data:
        plot_relative_target(data, output_dir, show, target_num=1)
        return

    plot_dual_axis_by_q(
        q_index,
        trans,
        rot,
        output_dir,
        "target1_fk_errors_by_q.png",
        "Target 1 Raw FK Error By q",
        show,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(finite(trans), bins=40, color="tab:blue", alpha=0.85)
    axes[0].set_title("Target 1 Raw FK Translation Error")
    axes[0].set_xlabel("translation [mm]")
    axes[0].set_ylabel("count")
    axes[1].hist(finite(rot), bins=40, color="tab:orange", alpha=0.85)
    axes[1].set_title("Target 1 Raw FK Rotation Error")
    axes[1].set_xlabel("rotation [deg]")
    savefig(fig, output_dir, "target1_fk_error_histograms.png", show)


def plot_relative_target(
    data: Dict[str, np.ndarray], output_dir: Path, show: bool, target_num: int
) -> None:
    import matplotlib.pyplot as plt

    sample = data["sample_index"]
    q_index = data["q_index"]
    trans = data["translation_error_mm"]
    rot = data["rotation_error_deg"]
    prefix = f"target{target_num}"

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(sample, trans, linewidth=0.8)
    axes[0].set_ylabel("translation [mm]")
    axes[0].set_title(f"Target {target_num} Relative FK Error By Relative Target")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(sample, rot, linewidth=0.8, color="tab:orange")
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("rotation [deg]")
    axes[1].grid(True, alpha=0.3)
    savefig(fig, output_dir, f"{prefix}_relative_errors_by_sample.png", show)

    q_unique, trans_mean, _ = grouped_mean(q_index, trans)
    _, rot_mean, _ = grouped_mean(q_index, rot)
    plot_dual_axis_by_q(
        q_unique,
        trans_mean,
        rot_mean,
        output_dir,
        f"{prefix}_relative_errors_by_q.png",
        f"Target {target_num} Relative FK Mean Error Per q",
        show,
        trans_label="mean translation [mm]",
        rot_label="mean rotation [deg]",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(finite(trans), bins=40, color="tab:blue", alpha=0.85)
    axes[0].set_title(f"Target {target_num} Relative FK Translation Error")
    axes[0].set_xlabel("translation [mm]")
    axes[0].set_ylabel("count")
    axes[1].hist(finite(rot), bins=40, color="tab:orange", alpha=0.85)
    axes[1].set_title(f"Target {target_num} Relative FK Rotation Error")
    axes[1].set_xlabel("rotation [deg]")
    savefig(fig, output_dir, f"{prefix}_relative_error_histograms.png", show)


def plot_target2(target2_csv: Path, output_dir: Path, show: bool) -> None:
    data = read_csv_numeric(target2_csv)
    if "gt_eval_translation_error_mm" in data:
        plot_ik_target(data, output_dir, show, target_num=2)
        return
    plot_relative_target(data, output_dir, show, target_num=2)


def plot_target2_extra_error_vs_target1(
    target1_csv: Path,
    target2_csv: Path,
    output_dir: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    target1 = read_csv_numeric(target1_csv)
    target2 = read_csv_numeric(target2_csv)
    if "gt_eval_translation_error_mm" in target2 or "relative_index" not in target2:
        return

    q_unique, target2_trans_mean, _ = grouped_mean(
        target2["q_index"], target2["translation_error_mm"]
    )
    _, target2_rot_mean, _ = grouped_mean(target2["q_index"], target2["rotation_error_deg"])

    target1_by_q = {
        int(q): (float(trans), float(rot))
        for q, trans, rot in zip(
            target1["q_index"],
            target1["translation_error_mm"],
            target1["rotation_error_deg"],
        )
    }
    target1_trans = np.asarray(
        [target1_by_q.get(int(q), (np.nan, np.nan))[0] for q in q_unique], dtype=np.float64
    )
    target1_rot = np.asarray(
        [target1_by_q.get(int(q), (np.nan, np.nan))[1] for q in q_unique], dtype=np.float64
    )

    extra_trans = target2_trans_mean - target1_trans
    extra_rot = target2_rot_mean - target1_rot

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(q_unique, extra_trans, color="tab:blue", linewidth=1.0)
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    trans_mean = finite(extra_trans)
    if trans_mean.size:
        axes[0].axhline(
            np.mean(trans_mean),
            color="tab:blue",
            linestyle="--",
            linewidth=1.0,
            label=f"mean {np.mean(trans_mean):.4g} mm",
        )
        axes[0].legend()
    axes[0].set_ylabel("extra translation [mm]")
    axes[0].set_title("Target 2 Relative FK Mean Error Minus Target 1 Raw FK Error")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(q_unique, extra_rot, color="tab:orange", linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    rot_mean = finite(extra_rot)
    if rot_mean.size:
        axes[1].axhline(
            np.mean(rot_mean),
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
            label=f"mean {np.mean(rot_mean):.4g} deg",
        )
        axes[1].legend()
    axes[1].set_xlabel("q index")
    axes[1].set_ylabel("extra rotation [deg]")
    axes[1].grid(True, alpha=0.3)
    savefig(fig, output_dir, "target2_extra_error_minus_target1_by_q.png", show)


def plot_ik_target(
    data: Dict[str, np.ndarray], output_dir: Path, show: bool, target_num: int
) -> None:
    import matplotlib.pyplot as plt

    sample = data["sample_index"]
    q_index = data["q_index"]
    trans = data["gt_eval_translation_error_mm"]
    rot = data["gt_eval_rotation_error_deg"]
    both_success = data["both_ik_success"]
    prefix = f"target{target_num}"
    failed = both_success < 0.5

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(sample, trans, linewidth=0.8)
    if np.any(failed):
        axes[0].scatter(
            sample[failed],
            trans[failed],
            color="tab:red",
            marker="x",
            s=28,
            label="IK failed",
            zorder=3,
        )
        axes[0].legend()
    axes[0].set_ylabel("GT eval trans [mm]")
    axes[0].set_title(f"Target {target_num} GT-Evaluated IK Result Difference")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(sample, rot, linewidth=0.8, color="tab:orange")
    if np.any(failed):
        axes[1].scatter(
            sample[failed],
            rot[failed],
            color="tab:red",
            marker="x",
            s=28,
            label="IK failed",
            zorder=3,
        )
        axes[1].legend()
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("GT eval rot [deg]")
    axes[1].grid(True, alpha=0.3)
    savefig(fig, output_dir, f"{prefix}_ik_errors_by_sample.png", show)

    q_unique, trans_mean, _ = grouped_mean(q_index, trans)
    _, rot_mean, _ = grouped_mean(q_index, rot)
    failure_q, failure_count = grouped_failure_count(q_index, both_success)
    failed_q = failure_q[failure_count > 0]
    plot_dual_axis_by_q(
        q_unique,
        trans_mean,
        rot_mean,
        output_dir,
        f"{prefix}_ik_metrics_by_q.png",
        f"Target {target_num} IK Mean Metrics Per q",
        show,
        trans_label="mean trans [mm]",
        rot_label="mean rot [deg]",
        failed_q=failed_q,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(finite(trans), bins=40, color="tab:blue", alpha=0.85)
    axes[0].set_title(f"Target {target_num} GT Eval Translation")
    axes[0].set_xlabel("translation [mm]")
    axes[0].set_ylabel("count")
    axes[1].hist(finite(rot), bins=40, color="tab:orange", alpha=0.85)
    axes[1].set_title(f"Target {target_num} GT Eval Rotation")
    axes[1].set_xlabel("rotation [deg]")
    savefig(fig, output_dir, f"{prefix}_ik_error_histograms.png", show)


def plot_target3(target3_csv: Path, output_dir: Path, show: bool) -> None:
    plot_ik_target(read_csv_numeric(target3_csv), output_dir, show, target_num=3)


def plot_fk_sample(fk_sample_csv: Path, output_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    data = read_csv_numeric(fk_sample_csv)
    sample = data["sample_index"]
    trans = data["translation_error_mm"]
    rot = data["rotation_error_deg"]

    plot_dual_axis_by_q(
        sample,
        trans,
        rot,
        output_dir,
        "fk_sample_errors_by_sample.png",
        "Joint-Limit FK Sampling Error By Sample",
        show,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(finite(trans), bins=60, color="tab:blue", alpha=0.85)
    axes[0].set_title("FK Sampling Translation Error")
    axes[0].set_xlabel("translation [mm]")
    axes[0].set_ylabel("count")
    axes[1].hist(finite(rot), bins=60, color="tab:orange", alpha=0.85)
    axes[1].set_title("FK Sampling Rotation Error")
    axes[1].set_xlabel("rotation [deg]")
    savefig(fig, output_dir, "fk_sample_error_histograms.png", show)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else result_dir / "plots"

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    q_csv = result_dir / "q_samples.csv"
    target1_csv = result_dir / "target1_metrics.csv"
    target2_csv = result_dir / "target2_metrics.csv"
    target3_csv = result_dir / "target3_metrics.csv"
    fk_sample_csv = result_dir / "fk_sample_metrics.csv"

    if q_csv.exists():
        plot_q_samples(q_csv, output_dir, args.show)
    if target1_csv.exists():
        plot_target1(target1_csv, output_dir, args.show)
    if target2_csv.exists():
        plot_target2(target2_csv, output_dir, args.show)
    if target1_csv.exists() and target2_csv.exists():
        plot_target2_extra_error_vs_target1(target1_csv, target2_csv, output_dir, args.show)
    if target3_csv.exists():
        plot_target3(target3_csv, output_dir, args.show)
    if fk_sample_csv.exists():
        plot_fk_sample(fk_sample_csv, output_dir, args.show)

    print(f"plots saved under {output_dir}")


if __name__ == "__main__":
    main()
