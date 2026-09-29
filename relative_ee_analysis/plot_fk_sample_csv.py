#!/usr/bin/env python3
"""Plot FK sampling curves from fk_sample_metrics.csv or fk_sample_results.npz."""

from __future__ import annotations

import argparse
import csv
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.lib import format as npy_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot curves from a FK sampling CSV or NPZ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_or_result_dir",
        type=Path,
        help=(
            "Path to fk_sample_metrics.csv, fk_sample_results.npz, or to a result "
            "directory containing either file."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Maximum points to keep for line/scatter plots. Use <=0 to plot all rows.",
    )
    parser.add_argument("--point-size", type=float, default=2.0, help="Scatter point size.")
    parser.add_argument("--alpha", type=float, default=0.18, help="Scatter point alpha.")
    parser.add_argument("--show", action="store_true", help="Open interactive plot windows.")
    return parser.parse_args()


def resolve_input_path(path: Path) -> Tuple[Path, str]:
    path = path.resolve()
    if path.is_dir():
        csv_path = path / "fk_sample_metrics.csv"
        npz_path = path / "fk_sample_results.npz"
        if csv_path.is_file():
            return csv_path, "csv"
        if npz_path.is_file():
            return npz_path, "npz"
        raise FileNotFoundError(
            f"Could not find fk_sample_metrics.csv or fk_sample_results.npz under {path}"
        )
    if path.suffix == ".csv" and path.is_file():
        return path, "csv"
    if path.suffix == ".npz" and path.is_file():
        return path, "npz"
    raise FileNotFoundError(f"Could not find FK sample CSV/NPZ: {path}")


def parse_float(row: Dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value else float("nan")


def update_stat(stats: Dict[str, Dict[str, float]], name: str, value: float, sample_index: int) -> None:
    if not math.isfinite(value):
        return
    item = stats.setdefault(
        name,
        {
            "count": 0,
            "sum": 0.0,
            "min": float("inf"),
            "min_sample_index": -1,
            "max": -float("inf"),
            "max_sample_index": -1,
        },
    )
    item["count"] += 1
    item["sum"] += value
    if value < item["min"]:
        item["min"] = value
        item["min_sample_index"] = sample_index
    if value > item["max"]:
        item["max"] = value
        item["max_sample_index"] = sample_index


def count_and_stats(csv_path: Path) -> Tuple[int, Dict[str, Dict[str, float]]]:
    stats: Dict[str, Dict[str, float]] = {}
    row_count = 0
    with csv_path.open(newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            sample_index = int(float(row["sample_index"]))
            update_stat(stats, "translation_error_mm", parse_float(row, "translation_error_mm"), sample_index)
            update_stat(stats, "rotation_error_deg", parse_float(row, "rotation_error_deg"), sample_index)
            if "gt_ee_radius_mm" in row:
                update_stat(stats, "gt_ee_radius_mm", parse_float(row, "gt_ee_radius_mm"), sample_index)
            row_count += 1

    for item in stats.values():
        item["mean"] = item["sum"] / item["count"] if item["count"] else float("nan")
    return row_count, stats


def read_plot_arrays(csv_path: Path, stride: int) -> Dict[str, np.ndarray]:
    data: Dict[str, List[float]] = {
        "sample_index": [],
        "translation_error_mm": [],
        "rotation_error_deg": [],
        "gt_ee_radius_mm": [],
    }

    with csv_path.open(newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        has_radius = "gt_ee_radius_mm" in (reader.fieldnames or [])
        for row_idx, row in enumerate(reader):
            if row_idx % stride != 0:
                continue
            data["sample_index"].append(parse_float(row, "sample_index"))
            data["translation_error_mm"].append(parse_float(row, "translation_error_mm"))
            data["rotation_error_deg"].append(parse_float(row, "rotation_error_deg"))
            if has_radius:
                data["gt_ee_radius_mm"].append(parse_float(row, "gt_ee_radius_mm"))

    return {name: np.asarray(values, dtype=np.float64) for name, values in data.items()}


def npz_array_names(npz_path: Path) -> List[str]:
    with zipfile.ZipFile(npz_path) as zf:
        return [Path(name).stem for name in zf.namelist() if name.endswith(".npy")]


def read_exact(file_obj, byte_count: int) -> bytes:
    chunks = []
    remaining = byte_count
    while remaining > 0:
        chunk = file_obj.read(remaining)
        if not chunk:
            raise EOFError(f"Expected {byte_count} bytes, got {byte_count - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_npy_header(file_obj) -> Tuple[Tuple[int, ...], bool, np.dtype]:
    version = npy_format.read_magic(file_obj)
    if version == (1, 0):
        shape, fortran_order, dtype = npy_format.read_array_header_1_0(file_obj)
    elif version in ((2, 0), (3, 0)):
        shape, fortran_order, dtype = npy_format.read_array_header_2_0(file_obj)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    return tuple(int(dim) for dim in shape), bool(fortran_order), np.dtype(dtype)


def npz_array_info(npz_path: Path, array_name: str) -> Tuple[Tuple[int, ...], np.dtype]:
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open(f"{array_name}.npy") as file_obj:
            shape, fortran_order, dtype = read_npy_header(file_obj)
            if fortran_order and len(shape) > 1:
                raise ValueError(f"{array_name} is Fortran-ordered; this script expects C-order arrays")
            return shape, dtype


def iter_npz_array_chunks(
    npz_path: Path,
    array_name: str,
    chunk_values: int = 1_000_000,
):
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open(f"{array_name}.npy") as file_obj:
            shape, fortran_order, dtype = read_npy_header(file_obj)
            if fortran_order and len(shape) > 1:
                raise ValueError(f"{array_name} is Fortran-ordered; this script expects C-order arrays")
            total = int(np.prod(shape))
            start = 0
            while start < total:
                count = min(chunk_values, total - start)
                raw = read_exact(file_obj, count * dtype.itemsize)
                values = np.frombuffer(raw, dtype=dtype, count=count)
                if not values.dtype.isnative:
                    values = values.byteswap().newbyteorder()
                yield start, values.reshape(-1)
                start += count


def update_stat_from_values(
    stats: Dict[str, Dict[str, float]],
    name: str,
    values: np.ndarray,
    start_index: int,
) -> None:
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return

    finite_values = values[finite_mask]
    finite_indices = np.nonzero(finite_mask)[0]
    item = stats.setdefault(
        name,
        {
            "count": 0,
            "sum": 0.0,
            "min": float("inf"),
            "min_sample_index": -1,
            "max": -float("inf"),
            "max_sample_index": -1,
        },
    )
    item["count"] += int(finite_values.size)
    item["sum"] += float(np.sum(finite_values, dtype=np.float64))

    local_min_idx = int(np.argmin(finite_values))
    local_min = float(finite_values[local_min_idx])
    if local_min < item["min"]:
        item["min"] = local_min
        item["min_sample_index"] = int(start_index + finite_indices[local_min_idx])

    local_max_idx = int(np.argmax(finite_values))
    local_max = float(finite_values[local_max_idx])
    if local_max > item["max"]:
        item["max"] = local_max
        item["max_sample_index"] = int(start_index + finite_indices[local_max_idx])


def read_npz_scaled_array(
    npz_path: Path,
    array_name: str,
    output_name: str,
    scale: float,
    stride: int,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    sampled: List[np.ndarray] = []
    stats: Dict[str, Dict[str, float]] = {}
    for start, chunk in iter_npz_array_chunks(npz_path, array_name):
        values = chunk.astype(np.float64, copy=False) * scale
        update_stat_from_values(stats, output_name, values, start)
        local_indices = np.arange(chunk.size, dtype=np.int64)
        keep = ((start + local_indices) % stride) == 0
        if np.any(keep):
            sampled.append(values[keep].copy())

    item = stats.get(output_name)
    if item is not None:
        item["mean"] = item["sum"] / item["count"] if item["count"] else float("nan")

    if sampled:
        return np.concatenate(sampled, axis=0), stats
    return np.asarray([], dtype=np.float64), stats


def read_npz_plot_data(
    npz_path: Path,
    max_points: int,
) -> Tuple[int, int, Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    available = set(npz_array_names(npz_path))
    required = {"translation_error_m", "rotation_error_rad"}
    missing = sorted(required - available)
    if missing:
        raise KeyError(f"{npz_path} is missing required arrays: {missing}")

    shape, _ = npz_array_info(npz_path, "translation_error_m")
    row_count = int(np.prod(shape))
    stride = 1
    if max_points > 0 and row_count > max_points:
        stride = int(math.ceil(row_count / max_points))

    data: Dict[str, np.ndarray] = {
        "sample_index": np.arange(0, row_count, stride, dtype=np.float64)
    }
    stats: Dict[str, Dict[str, float]] = {}

    trans, trans_stats = read_npz_scaled_array(
        npz_path, "translation_error_m", "translation_error_mm", 1000.0, stride
    )
    rot, rot_stats = read_npz_scaled_array(
        npz_path, "rotation_error_rad", "rotation_error_deg", 180.0 / math.pi, stride
    )
    data["translation_error_mm"] = trans
    data["rotation_error_deg"] = rot
    stats.update(trans_stats)
    stats.update(rot_stats)

    if "gt_ee_radius_m" in available:
        radius, radius_stats = read_npz_scaled_array(
            npz_path, "gt_ee_radius_m", "gt_ee_radius_mm", 1000.0, stride
        )
        data["gt_ee_radius_mm"] = radius
        stats.update(radius_stats)
    else:
        data["gt_ee_radius_mm"] = np.asarray([], dtype=np.float64)

    return row_count, stride, stats, data


def finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def savefig(fig, output_dir: Path, name: str, show: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if show:
        fig.show()
    print(f"wrote {path}")


def add_stat_lines(ax, item: Dict[str, float], orientation: str) -> None:
    if not item:
        return
    specs = [
        ("min", "lower", "tab:green", ":"),
        ("mean", "mean", "black", "-"),
        ("max", "upper", "tab:red", "--"),
    ]
    for key, label, color, linestyle in specs:
        value = item.get(key)
        if value is None or not math.isfinite(value):
            continue
        line_label = f"{label} {value:.4g}"
        if orientation == "horizontal":
            ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.2, label=line_label)
        else:
            ax.axvline(value, color=color, linestyle=linestyle, linewidth=1.2, label=line_label)
    ax.legend(loc="best")


def plot_sample_curves(
    data: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, float]],
    output_dir: Path,
    show: bool,
    point_size: float,
    alpha: float,
) -> None:
    import matplotlib.pyplot as plt

    sample = data["sample_index"]
    trans = data["translation_error_mm"]
    rot = data["rotation_error_deg"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].scatter(sample, trans, s=point_size, alpha=alpha, color="tab:blue", rasterized=True)
    axes[0].set_ylabel("translation [mm]")
    axes[0].set_title("FK Sampling Error By Sample")
    axes[0].grid(True, alpha=0.3)
    add_stat_lines(axes[0], stats.get("translation_error_mm", {}), "horizontal")
    axes[1].scatter(sample, rot, s=point_size, alpha=alpha, color="tab:orange", rasterized=True)
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("rotation [deg]")
    axes[1].grid(True, alpha=0.3)
    add_stat_lines(axes[1], stats.get("rotation_error_deg", {}), "horizontal")
    savefig(fig, output_dir, "fk_sample_errors_by_sample.png", show)


def plot_histograms(
    data: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, float]],
    output_dir: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    trans = data["translation_error_mm"]
    rot = data["rotation_error_deg"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(trans[np.isfinite(trans)], bins=70, color="tab:blue", alpha=0.85)
    axes[0].set_title("Translation Error")
    axes[0].set_xlabel("translation [mm]")
    axes[0].set_ylabel("count")
    add_stat_lines(axes[0], stats.get("translation_error_mm", {}), "vertical")
    axes[1].hist(rot[np.isfinite(rot)], bins=70, color="tab:orange", alpha=0.85)
    axes[1].set_title("Rotation Error")
    axes[1].set_xlabel("rotation [deg]")
    add_stat_lines(axes[1], stats.get("rotation_error_deg", {}), "vertical")
    savefig(fig, output_dir, "fk_sample_error_histograms.png", show)


def plot_radius_curves(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    show: bool,
    point_size: float,
    alpha: float,
) -> None:
    import matplotlib.pyplot as plt

    radius = data.get("gt_ee_radius_mm")
    if radius is None or radius.size == 0:
        return

    radius_t, trans = finite_pair(radius, data["translation_error_mm"])
    radius_r, rot = finite_pair(radius, data["rotation_error_deg"])
    if radius_t.size == 0 or radius_r.size == 0:
        return
    if float(np.max(radius_t)) <= float(np.min(radius_t)):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(radius_t, trans, s=point_size, alpha=alpha, color="tab:blue", rasterized=True)
    axes[0].set_title("Translation Error vs Reach")
    axes[0].set_xlabel("GT EE radius [mm]")
    axes[0].set_ylabel("translation [mm]")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(radius_r, rot, s=point_size, alpha=alpha, color="tab:orange", rasterized=True)
    axes[1].set_title("Rotation Error vs Reach")
    axes[1].set_xlabel("GT EE radius [mm]")
    axes[1].set_ylabel("rotation [deg]")
    axes[1].grid(True, alpha=0.3)
    savefig(fig, output_dir, "fk_sample_error_vs_reach_scatter.png", show)

    bins = np.linspace(float(np.min(radius_t)), float(np.max(radius_t)), 60)
    centers = 0.5 * (bins[:-1] + bins[1:])
    trans_min = np.full(centers.shape, np.nan, dtype=np.float64)
    trans_mean = np.full(centers.shape, np.nan, dtype=np.float64)
    trans_max = np.full(centers.shape, np.nan, dtype=np.float64)
    rot_min = np.full(centers.shape, np.nan, dtype=np.float64)
    rot_mean = np.full(centers.shape, np.nan, dtype=np.float64)
    rot_max = np.full(centers.shape, np.nan, dtype=np.float64)
    trans_bins = np.digitize(radius_t, bins) - 1
    rot_bins = np.digitize(radius_r, bins) - 1
    for idx in range(centers.size):
        trans_values = trans[trans_bins == idx]
        rot_values = rot[rot_bins == idx]
        if trans_values.size:
            trans_min[idx] = np.min(trans_values)
            trans_mean[idx] = np.mean(trans_values)
            trans_max[idx] = np.max(trans_values)
        if rot_values.size:
            rot_min[idx] = np.min(rot_values)
            rot_mean[idx] = np.mean(rot_values)
            rot_max[idx] = np.max(rot_values)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(centers, trans_min, color="tab:green", linestyle=":", label="lower")
    axes[0].plot(centers, trans_mean, color="black", label="mean")
    axes[0].plot(centers, trans_max, color="tab:red", linestyle="--", label="upper")
    axes[0].set_ylabel("translation [mm]")
    axes[0].set_title("FK Error Binned By Reach")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(centers, rot_min, color="tab:green", linestyle=":", label="lower")
    axes[1].plot(centers, rot_mean, color="black", label="mean")
    axes[1].plot(centers, rot_max, color="tab:red", linestyle="--", label="upper")
    axes[1].set_xlabel("GT EE radius [mm]")
    axes[1].set_ylabel("rotation [deg]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    savefig(fig, output_dir, "fk_sample_error_vs_reach_binned.png", show)


def print_stats(stats: Dict[str, Dict[str, float]]) -> None:
    for name in ("translation_error_mm", "rotation_error_deg", "gt_ee_radius_mm"):
        item = stats.get(name)
        if not item:
            continue
        print(
            f"{name}: min={item['min']:.6g}, mean={item['mean']:.6g}, "
            f"max={item['max']:.6g}, min_sample_index={int(item['min_sample_index'])}, "
            f"max_sample_index={int(item['max_sample_index'])}, count={int(item['count'])}"
        )


def main() -> None:
    args = parse_args()
    input_path, input_kind = resolve_input_path(args.csv_or_result_dir)
    output_dir = args.output_dir.resolve() if args.output_dir else input_path.parent / "fk_sample_plots"

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    if input_kind == "csv":
        row_count, stats = count_and_stats(input_path)
        stride = 1
        if args.max_points > 0 and row_count > args.max_points:
            stride = int(math.ceil(row_count / args.max_points))
        data = read_plot_arrays(input_path, stride)
    else:
        row_count, stride, stats, data = read_npz_plot_data(input_path, args.max_points)

    plot_sample_curves(data, stats, output_dir, args.show, args.point_size, args.alpha)
    plot_histograms(data, stats, output_dir, args.show)
    plot_radius_curves(data, output_dir, args.show, args.point_size, args.alpha)

    print_stats(stats)
    print(f"read {row_count:,} {input_kind.upper()} rows, plotted every {stride} row(s)")
    print(f"plots saved under {output_dir}")


if __name__ == "__main__":
    main()
