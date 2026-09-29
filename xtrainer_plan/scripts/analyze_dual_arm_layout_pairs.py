#!/usr/bin/env python3
"""Score two-location XTrainer candidates from the audited ten-case full grid.

This is a report-only calculation. A point is individually usable by an arm
when its saved full trajectory succeeds and satisfies the chosen effective
joint-margin and full-cycle maximum-joint-span thresholds. Pair overlap means
either arm could be selected for that point, not that two arms may safely move
there at the same time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np

DATE_ROOT = Path(__file__).resolve().parents[1] / "results_overhead" / "20260928"
SOURCE = DATE_ROOT / "signed_local_y_full_comparison" / "reports"
OUTPUT = DATE_ROOT / "dual_arm_layout_analysis" / "reports"
THRESHOLDS = (
    ("q20_170", 20.0, 170.0),
    ("q20_150", 20.0, 150.0),
    ("q30_150", 30.0, 150.0),
)
DISPLAY_THRESHOLDS = THRESHOLDS[1:]
DISPLAY_PAIRS = (
    ("v64_47", "v67_24"),
    ("v64_39", "v67_52"),
    ("v64_47", "v67_44"),
)
COLORS = ("#e3e7ea", "#3973a2", "#df8b3d", "#3caa8b")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source(folder: Path) -> tuple[dict, list[dict], dict]:
    hashes = json.loads((folder / "artifact_sha256.json").read_text(encoding="utf-8"))
    paths = {
        "same_point_metrics.csv": folder / "same_point_metrics.csv",
        "comparison_summary.json": folder / "comparison_summary.json",
    }
    for filename, path in paths.items():
        require(path.is_file() and sha256(path) == hashes[filename],
                f"Audited source missing or changed: {path}")
    summary = json.loads(paths["comparison_summary.json"].read_text(encoding="utf-8"))
    with paths["same_point_metrics.csv"].open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    candidates = {candidate["name"]: candidate for candidate in summary["candidates"]}
    require(len(candidates) == 10 and len(rows) == 400,
            "Expected ten candidate metadata records and 400 grasp rows")
    require(summary["coordinate_frame"] == "task_world" and
            summary["grid"]["rows"] == summary["grid"]["cols"] == 20,
            "Unexpected world frame or full-grid size")
    require(summary["n_any_of_ten_success"] == 400,
            "Audited ten-case coverage count changed")
    require({int(row["index"]) for row in rows} == set(range(400)),
            "Missing or duplicate grasp index")
    require({(int(row["row"]), int(row["col"])) for row in rows} ==
            set(itertools.product(range(20), repeat=2)),
            "Missing or duplicate grasp grid cell")
    grid = summary["grid"]
    xs = np.linspace(*grid["x_range"], 20)
    ys = np.linspace(*grid["y_range"], 20)
    for row in rows:
        i, j = int(row["row"]), int(row["col"])
        require(abs(float(row["x_m"]) - xs[i]) < 1e-8 and
                abs(float(row["y_m"]) - ys[j]) < 1e-8 and
                abs(float(row["z_m"]) - grid["z"]) < 1e-8,
                "Grasp coordinate differs from audited grid")
    for name, candidate in candidates.items():
        n_success = 0
        for row in rows:
            require(row[f"{name}_success"] in ("True", "False"),
                    f"Invalid success flag for {name}")
            success = row[f"{name}_success"] == "True"
            n_success += success
            for metric in ("min_effective_margin_deg", "max_cycle_joint_span_deg"):
                value = row[f"{name}_{metric}"]
                require((value != "") == success,
                        f"Missing success metric or failed point has metric for {name}")
                if success:
                    require(math.isfinite(float(value)) and float(value) >= 0,
                            f"Invalid {metric} for {name}")
        require(n_success == candidate["n_success"],
                f"Case count differs from audited summary for {name}")
    provenance = {name: {"path": str(path.resolve()), "sha256": hashes[name]}
                  for name, path in paths.items()}
    provenance["source_protocol_sha256"] = summary["shared_protocol_sha256"]
    provenance["source_urdf_sha256"] = summary["shared_urdf_sha256"]
    return candidates, rows, {"grid": grid, "provenance": provenance}


def flags(rows: list[dict], name: str,
          margin: float | None = None, span: float | None = None) -> np.ndarray:
    successful = np.array([row[f"{name}_success"] == "True" for row in rows], dtype=bool)
    if margin is None:
        return successful
    return np.array([
        bool(successful[i] and
             float(row[f"{name}_min_effective_margin_deg"]) >= margin and
             float(row[f"{name}_max_cycle_joint_span_deg"]) <= span)
        for i, row in enumerate(rows)
    ], dtype=bool)


def largest_component(mask: np.ndarray, rows: list[dict]) -> int:
    cells = {(int(row["row"]), int(row["col"])) for row, good in zip(rows, mask) if good}
    largest = 0
    while cells:
        start = cells.pop()
        frontier = [start]
        size = 1
        while frontier:
            i, j = frontier.pop()
            for neighbor in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if neighbor in cells:
                    cells.remove(neighbor)
                    frontier.append(neighbor)
                    size += 1
        largest = max(largest, size)
    return largest


def pair_stats(a: str, b: str, candidates: dict, rows: list[dict],
               arrays: dict) -> dict:
    base_a = candidates[a]["base_xyz_task_world_m"]
    base_b = candidates[b]["base_xyz_task_world_m"]
    result = {
        "arm_a": a, "arm_b": b,
        "arm_a_tilt_deg": candidates[a]["original_base_local_y_tilt_deg"],
        "arm_b_tilt_deg": candidates[b]["original_base_local_y_tilt_deg"],
        "arm_a_base_xyz_m": base_a, "arm_b_base_xyz_m": base_b,
        "base_origin_distance_m": math.dist(base_a, base_b),
        "base_y_distance_m": abs(base_a[1] - base_b[1]),
    }
    lower = np.array([float(row["y_m"]) <= 0.15 for row in rows], dtype=bool)
    upper = ~lower
    for key, _, _ in (("success", None, None), *THRESHOLDS):
        aa, bb = arrays[a][key], arrays[b][key]
        union, overlap = aa | bb, aa & bb
        result[f"{key}_a"] = int(aa.sum())
        result[f"{key}_b"] = int(bb.sum())
        result[f"{key}_union"] = int(union.sum())
        result[f"{key}_overlap"] = int(overlap.sum())
        result[f"{key}_a_only"] = int((aa & ~bb).sum())
        result[f"{key}_b_only"] = int((bb & ~aa).sum())
        result[f"{key}_neither"] = int((~union).sum())
        result[f"{key}_union_lower_y"] = int((union & lower).sum())
        result[f"{key}_union_upper_y"] = int((union & upper).sum())
        result[f"{key}_overlap_lower_y"] = int((overlap & lower).sum())
        result[f"{key}_overlap_upper_y"] = int((overlap & upper).sum())
        result[f"{key}_a_only_lower_y"] = int((aa & ~bb & lower).sum())
        result[f"{key}_a_only_upper_y"] = int((aa & ~bb & upper).sum())
        result[f"{key}_b_only_lower_y"] = int((bb & ~aa & lower).sum())
        result[f"{key}_b_only_upper_y"] = int((bb & ~aa & upper).sum())
        result[f"{key}_neither_lower_y"] = int((~union & lower).sum())
        result[f"{key}_neither_upper_y"] = int((~union & upper).sum())
        result[f"{key}_overlap_largest_4_neighbor_component"] = largest_component(
            overlap, rows)
        require(result[f"{key}_union"] + result[f"{key}_neither"] == 400 and
                result[f"{key}_overlap"] + result[f"{key}_a_only"] +
                result[f"{key}_b_only"] == result[f"{key}_union"],
                f"Invalid pair partition for {a} / {b}, {key}")
    return result


def flat(record: dict) -> dict:
    out = record.copy()
    for arm in ("arm_a", "arm_b"):
        out.pop(f"{arm}_base_xyz_m")
    return out


def draw_maps(pairs: list[dict], candidates: dict, rows: list[dict],
              arrays: dict, grid: dict, destination: Path) -> None:
    by_pair = {(p["arm_a"], p["arm_b"]): p for p in pairs}
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), 4)
    xs = np.linspace(*grid["x_range"], 20)
    ys = np.linspace(*grid["y_range"], 20)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    extent = (xs[0] - dx / 2, xs[-1] + dx / 2,
              ys[0] - dy / 2, ys[-1] + dy / 2)
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 15), constrained_layout=True)
    for row_number, pair in enumerate(DISPLAY_PAIRS):
        a, b = pair
        score = by_pair[pair]
        for col_number, (key, margin, span) in enumerate(DISPLAY_THRESHOLDS):
            ax = axes[row_number, col_number]
            aa, bb = arrays[a][key], arrays[b][key]
            state = np.zeros((20, 20), dtype=int)
            for source_row, good_a, good_b in zip(rows, aa, bb):
                state[int(source_row["row"]), int(source_row["col"])] = (
                    int(good_a) + 2 * int(good_b))
            ax.imshow(state.T, origin="lower", extent=extent,
                      interpolation="nearest", cmap=cmap, norm=norm, aspect="equal")
            ax.scatter(*candidates[a]["base_xyz_task_world_m"][:2],
                       marker="v", s=115, facecolors="#533d9b", edgecolors="white",
                       linewidths=0.8, zorder=5)
            ax.scatter(*candidates[b]["base_xyz_task_world_m"][:2],
                       marker="v", s=115, facecolors="#b93d6b", edgecolors="white",
                       linewidths=0.8, zorder=5)
            ax.scatter(-0.36, -0.12, marker="X", s=85, color="#111827",
                       edgecolors="white", linewidths=0.5, zorder=5)
            ax.set(xlim=(-0.66, -0.05), ylim=(-0.16, 0.50),
                   xlabel="task_world grasp x (m)",
                   ylabel="task_world grasp y (m)")
            ax.grid(color="#ffffff", alpha=0.20, linewidth=0.4)
            ax.set_title(
                f"{a} + {b}  |  margin ≥{margin:g}°, span ≤{span:g}°\n"
                f"Either {score[key + '_union']}/400  •  both {score[key + '_overlap']}/400  "
                f"•  base origins {score['base_origin_distance_m']:.3f} m",
                fontsize=10)
    handles = [
        Line2D([], [], marker="s", linestyle="None", color=color, markersize=11,
               label=label)
        for color, label in zip(COLORS, ("Neither", "Arm A only", "Arm B only", "Both"))
    ] + [
        Line2D([], [], marker="v", linestyle="None", color=color, markersize=10,
               label=label)
        for color, label in (("#533d9b", "Arm A base"), ("#b93d6b", "Arm B base"))
    ] + [
        Line2D([], [], marker="X", linestyle="None", color="#111827",
               markersize=9, label="Shared place")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Saved single-arm trajectories: two-location quality maps\n"
                 "Shared green cells permit arm choice; simultaneous safety is untested",
                 fontsize=15)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_tradeoff(pairs: list[dict], destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    highlights = set(DISPLAY_PAIRS)
    for ax, (key, margin, span) in zip(axes, DISPLAY_THRESHOLDS):
        x = [p[f"{key}_union"] for p in pairs]
        y = [p[f"{key}_overlap"] for p in pairs]
        colors = [p["base_origin_distance_m"] for p in pairs]
        scatter = ax.scatter(x, y, c=colors, cmap="viridis",
                             vmin=0.2, vmax=0.38, s=80,
                             edgecolors="#293241", linewidths=0.6)
        label_offsets = {
            ("v64_47", "v67_24"): (6, -13),
            ("v64_39", "v67_52"): (6, 6),
            ("v64_47", "v67_44"): (6, 9),
        }
        for p in pairs:
            pair_key = (p["arm_a"], p["arm_b"])
            if pair_key in highlights:
                ax.annotate(f"{p['arm_a']}+{p['arm_b']}",
                            (p[f"{key}_union"], p[f"{key}_overlap"]),
                            xytext=label_offsets[pair_key], textcoords="offset points",
                            fontsize=8)
        ax.set(xlabel="Either arm qualifies / 400",
               ylabel="Both arms qualify / 400",
               title=f"Effective margin ≥{margin:g}°, cycle span ≤{span:g}°")
        ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes, label="Base-origin distance (m)", shrink=0.87)
    fig.suptitle("24 sampled two-location pairs  •  points sharing the same 20×20 grid",
                 fontsize=13)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    candidates, rows, context = load_source(args.source)
    near = [name for name, candidate in candidates.items()
            if math.isclose(candidate["base_xyz_task_world_m"][1], 0.15,
                            abs_tol=1e-9)]
    far = [name for name, candidate in candidates.items()
           if candidate["base_xyz_task_world_m"][1] >= 0.35 - 1e-9]
    require(len(near) == 4 and len(far) == 6 and
            set(near).isdisjoint(far), "Expected four y=0.15 and six y≥0.35 mounts")
    arrays = {}
    for name in candidates:
        arrays[name] = {"success": flags(rows, name)}
        for key, margin, span in THRESHOLDS:
            arrays[name][key] = flags(rows, name, margin, span)
    pairs = [pair_stats(a, b, candidates, rows, arrays)
             for a, b in itertools.product(near, far)]
    pairs.sort(key=lambda p: (-p["q20_150_union"], -p["q20_150_overlap"],
                              -p["q30_150_union"], -p["base_origin_distance_m"],
                              p["arm_a"], p["arm_b"]))
    require(len(pairs) == 24, "Expected 24 cross-Y pairs")
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "pair_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat(pairs[0])))
        writer.writeheader()
        writer.writerows(flat(pair) for pair in pairs)
    point_path = args.output / "pair_point_states.csv"
    with point_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["arm_a", "arm_b", "index", "case_number", "x_m", "y_m",
                  "success_state", "q20_170_state", "q20_150_state", "q30_150_state"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            a, b = pair["arm_a"], pair["arm_b"]
            for i, row in enumerate(rows):
                values = {"arm_a": a, "arm_b": b, "index": row["index"],
                          "case_number": row["case_number"],
                          "x_m": row["x_m"], "y_m": row["y_m"]}
                for key in ("success", "q20_170", "q20_150", "q30_150"):
                    values[f"{key}_state"] = (
                        int(arrays[a][key][i]) + 2 * int(arrays[b][key][i]))
                writer.writerow(values)
    draw_maps(pairs, candidates, rows, arrays, context["grid"],
              args.output / "pair_quality_maps.png")
    draw_tradeoff(pairs, args.output / "pair_tradeoff_scatter.png")
    first_success = {}
    for name in candidates:
        first_index = int(np.flatnonzero(arrays[name]["success"])[0])
        first_success[name] = {
            "case_number": int(rows[first_index]["case_number"]),
            "full_cycle_max_joint_span_deg": float(
                rows[first_index][f"{name}_max_cycle_joint_span_deg"]),
            "effective_margin_deg": float(
                rows[first_index][f"{name}_min_effective_margin_deg"]),
        }
    report = {
        "schema_version": 1,
        "description": "Cross-Y pairing of ten audited saved single-arm mount results",
        "source": context["provenance"],
        "coordinate_frame": "task_world",
        "grid": context["grid"],
        "thresholds_exploratory": {
            key: {"effective_margin_at_least_deg": margin,
                  "full_cycle_max_single_joint_span_at_most_deg": span}
            for key, margin, span in THRESHOLDS
        },
        "arm_a_base_y_m": 0.15,
        "arm_b_base_y_m": [0.35, 0.45],
        "pair_count": len(pairs),
        "candidate_first_saved_success": first_success,
        "candidate_quality_counts": {
            name: {key: int(arrays[name][key].sum())
                   for key in ("success", "q20_170", "q20_150", "q30_150")}
            for name in candidates
        },
        "pairs": pairs,
        "definitions": {
            "union": "At least one sampled arm has a saved trajectory meeting this point's criteria.",
            "overlap": "Both sampled arms independently meet the criteria for this grasp point; only one-arm-at-a-time alternation is implied.",
            "largest_4_neighbor_component": "Largest edge-connected region of both-arm-qualified cells on the 20x20 grasp grid.",
            "lower_y": "Grasp y <= 0.15 m, 200 cells.",
            "upper_y": "Grasp y > 0.15 m, 200 cells.",
            "full_cycle_span": "Includes entry from Home or prior successful task; result depends on case order.",
            "base_distance": "Euclidean distance between origin points, not robot-body clearance.",
        },
        "limitations": [
            "Each trajectory was planned for a single robot without a second robot installed.",
            "There is no swept-volume, simultaneous two-arm, parked-arm, or mounting-hardware collision audit.",
            "Both arms share one fixed place target; serialized place access is required even for an overlapping grasp point.",
            "The sampled y=0.15 and y=0.35/0.45 mounts are not established as physically left/right mounting surfaces.",
            "Thresholds screen saved sampled trajectories, not a guarantee of robust replanning or execution.",
        ],
    }
    json_path = args.output / "pair_summary.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    hashes = {
        path.name: sha256(path)
        for path in (csv_path, point_path,
                     args.output / "pair_quality_maps.png",
                     args.output / "pair_tradeoff_scatter.png", json_path)
    }
    if (args.output / "README.md").is_file():
        hashes["README.md"] = sha256(args.output / "README.md")
    (args.output / "artifact_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("Wrote", args.output)
    for pair in pairs[:5]:
        print(pair["arm_a"], pair["arm_b"],
              "distance", round(pair["base_origin_distance_m"], 3),
              "success union/overlap", pair["success_union"], pair["success_overlap"],
              "q20/150 union/overlap", pair["q20_150_union"], pair["q20_150_overlap"],
              "q30/150 union/overlap", pair["q30_150_union"], pair["q30_150_overlap"])


if __name__ == "__main__":
    main()
