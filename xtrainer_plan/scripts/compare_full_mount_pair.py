#!/usr/bin/env python3
"""Plot matched full-grid coverage and joint-metric deltas for two audited runs.

The first summary is subtracted by the second.  Joint deltas are defined only
where both runs saved independently audited complete trajectories.
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
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np

from plot_full_mount_joint_distributions import require, sha256
from plot_grasp_angle_map import pick_cjk_font


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def load(summary_path: Path) -> dict:
    report = read_json(summary_path)
    source = report["provenance"]
    run = Path(source["result"]).resolve()
    for key, filename in (("trajectory_sha256", "trajectory.npz"),
                          ("metadata_sha256", "trajectory_meta.json"),
                          ("independent_verification_sha256", "independent_verification.json"),
                          ("joint_limit_clip_audit_sha256", "joint_limit_clip_audit.json")):
        require(sha256(run / filename) == source[key],
                f"{run.name}: {filename} changed since audited plot")
    require(sha256(Path(source["urdf"])) == source["urdf_sha256"],
            f"{run.name}: URDF changed")
    for filename in ("independent_verification.json", "joint_limit_clip_audit.json"):
        audit = read_json(run / filename)
        require(audit.get("passed") is True and audit.get("verification_completed") is True,
                f"{run.name}: {filename} not fully passed")
    meta = read_json(run / "trajectory_meta.json")
    require(report["n_total"] == len(report["cases"]) == meta["n_items_total"] == 400
            and report["n_success"] == meta["n_items_success"],
            f"{run.name}: report/meta counts differ")
    cases = {int(case["index"]): case for case in report["cases"]}
    require(len(cases) == 400 and set(cases) == set(range(400)),
            f"{run.name}: missing or duplicate grasp index")
    return {"run": run, "report": report, "meta": meta, "cases": cases}


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    array = np.asarray(values)
    return {"count": len(values), "min": float(array.min()),
            "median": float(np.median(array)), "max": float(array.max())}


def match(first: dict, second: dict) -> tuple[list[dict], dict]:
    a, b = first["report"], second["report"]
    require(a["grid"] == b["grid"] and a["n_total"] == b["n_total"] == 400,
            "The two full grids differ")
    require(a["provenance"]["urdf_sha256"] == b["provenance"]["urdf_sha256"]
            and math.isclose(a["joint_limit_clip_rad"], b["joint_limit_clip_rad"], abs_tol=1e-12),
            "URDF or planner joint-limit clip differs")
    require(np.allclose(first["meta"]["place_position_raw"],
                        second["meta"]["place_position_raw"], atol=1e-9),
            "Fixed place target differs")
    records = []
    margin, span = [], []
    status_counts = {"neither": 0, "first_only": 0, "second_only": 0, "both": 0}
    for index in range(400):
        ca, cb = first["cases"][index], second["cases"][index]
        require((ca["row"], ca["col"]) == (cb["row"], cb["col"])
                and np.allclose([ca["x_m"], ca["y_m"], ca["z_m"]],
                                [cb["x_m"], cb["y_m"], cb["z_m"]], atol=1e-8),
                f"Case {index+1}: grasp coordinates differ")
        sa, sb = ca["success"], cb["success"]
        require(type(sa) is bool and type(sb) is bool, "Invalid success flag")
        status = "both" if sa and sb else "first_only" if sa else "second_only" if sb else "neither"
        status_counts[status] += 1
        record = {"index": index, "case_number": index+1,
                  "row": ca["row"], "col": ca["col"],
                  "x_m": ca["x_m"], "y_m": ca["y_m"], "z_m": ca["z_m"],
                  "coverage": status, "first_success": sa, "second_success": sb,
                  "effective_margin_delta_deg": "", "max_joint_span_delta_deg": ""}
        if status == "both":
            dm = float(ca["min_effective_margin_deg"] - cb["min_effective_margin_deg"])
            ds = float(ca["max_cycle_joint_span_deg"] - cb["max_cycle_joint_span_deg"])
            record["effective_margin_delta_deg"] = dm
            record["max_joint_span_delta_deg"] = ds
            margin.append(dm)
            span.append(ds)
        records.append(record)
    result = {
        "schema_version": 1, "coordinate_frame": "task_world",
        "first": {"run": str(first["run"]), "mount_xyz_m": a["mount_xyz_task_world_m"],
                  "tilt_axis": a["tilt_axis"], "tilt_deg": a["tilt_deg"],
                  "n_success": a["n_success"], "trajectory_sha256": a["provenance"]["trajectory_sha256"]},
        "second": {"run": str(second["run"]), "mount_xyz_m": b["mount_xyz_task_world_m"],
                   "tilt_axis": b["tilt_axis"], "tilt_deg": b["tilt_deg"],
                   "n_success": b["n_success"], "trajectory_sha256": b["provenance"]["trajectory_sha256"]},
        "grid": a["grid"], "place_position_m": first["meta"]["place_position_raw"],
        "coverage": status_counts,
        "first_minus_second_effective_margin_deg": distribution(margin),
        "first_minus_second_max_joint_span_deg": distribution(span),
        "n_first_higher_margin": sum(v > 0 for v in margin),
        "n_first_lower_span": sum(v < 0 for v in span),
        "definitions": {
            "margin": "Minimum planner-effective joint-limit margin over saved full-cycle samples; positive first-minus-second favors the first run.",
            "span": "Largest single-joint max(q)-min(q) across J1-J6 and the full pick/place cycle; negative first-minus-second means less angular motion in the first run.",
            "missing": "Joint deltas exist only where both full trajectories were saved; a failed case has no joint metric, never a zero.",
        },
    }
    return records, result


def edges(values: np.ndarray) -> np.ndarray:
    d = np.diff(values)
    return np.r_[values[0]-d[0]/2, values[:-1]+d/2, values[-1]+d[-1]/2]


def draw(records: list[dict], summary: dict, png: Path, first_name: str, second_name: str) -> None:
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if font else (lambda zh, en: en)
    grid = summary["grid"]
    xs = np.linspace(*grid["x_range"], grid["rows"])
    ys = np.linspace(*grid["y_range"], grid["cols"])
    xe, ye = edges(xs), edges(ys)
    coverage = np.zeros((len(ys), len(xs)), dtype=int)
    margin = np.full_like(coverage, np.nan, dtype=float)
    span = np.full_like(coverage, np.nan, dtype=float)
    codes = {"neither": 0, "first_only": 1, "second_only": 2, "both": 3}
    for row in records:
        i, j = int(row["row"]), int(row["col"])
        coverage[j, i] = codes[row["coverage"]]
        if row["coverage"] == "both":
            margin[j, i] = row["effective_margin_delta_deg"]
            span[j, i] = row["max_joint_span_delta_deg"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)
    cmap = ListedColormap(["#d1d1d1", "#4ac1ad", "#f4a146", "#3f6db3"])
    axes[0].pcolormesh(xe, ye, coverage, cmap=cmap,
                       norm=BoundaryNorm(np.arange(-.5, 4.5), 4), shading="flat")
    axes[0].legend(handles=[
        Patch(facecolor="#4ac1ad", label=L(f"仅 {first_name} 成功", f"{first_name} only")),
        Patch(facecolor="#f4a146", label=L(f"仅 {second_name} 成功", f"{second_name} only")),
        Patch(facecolor="#3f6db3", label=L("两组都成功", "Both successful")),
        Patch(facecolor="#d1d1d1", label=L("两组都失败", "Neither successful")),
    ], loc="upper right", fontsize=7.5)
    axes[0].set_title(L("同点覆盖区域", "Matched-point coverage"))
    for ax, data, zh, en, color in (
        (axes[1], margin, "有效关节限位余量差", "Effective joint-limit margin delta", "RdBu_r"),
        (axes[2], span, "完整抓放单关节最大跨度差", "Full-cycle max joint-span delta", "RdBu_r"),
    ):
        values = data[np.isfinite(data)]
        bound = max(abs(float(values.min())), abs(float(values.max())), 1.0)
        palette = plt.get_cmap(color).copy()
        palette.set_bad("#d1d1d1")
        artist = ax.pcolormesh(xe, ye, np.ma.masked_invalid(data),
                               cmap=palette, norm=TwoSlopeNorm(vcenter=0, vmin=-bound, vmax=bound),
                               shading="flat")
        cb = fig.colorbar(artist, ax=ax, fraction=.045, pad=.02)
        cb.set_label("°")
        ax.set_title(L(zh, en))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("task_world X (m)")
        ax.set_ylabel("task_world Y (m)")
        ax.tick_params(labelsize=8)
    fig.suptitle(f"{first_name} − {second_name} | "
                 + L(f"共同成功 {summary['coverage']['both']}/400 点，灰色指标缺失",
                     f"{summary['coverage']['both']}/400 both successful; gray means no pair metric"),
                 fontsize=15)
    fig.savefig(png, dpi=190, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_summary", type=Path)
    parser.add_argument("second_summary", type=Path)
    parser.add_argument("--first-name", required=True)
    parser.add_argument("--second-name", required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    first = load(args.first_summary)
    second = load(args.second_summary)
    rows, summary = match(first, second)
    out = args.out_prefix.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    targets = [out.with_suffix(ext) for ext in (".png", ".csv", ".json")]
    require(not any(path.exists() for path in targets),
            f"Refusing to overwrite existing pair output: {out}")
    draw(rows, summary, targets[0], args.first_name, args.second_name)
    with targets[1].open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    targets[2].write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"[ok] {out}: both={summary['coverage']['both']}, "
          f"first_only={summary['coverage']['first_only']}, "
          f"second_only={summary['coverage']['second_only']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
