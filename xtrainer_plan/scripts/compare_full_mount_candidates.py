#!/usr/bin/env python3
"""Compare audited full XTrainer mount candidates at the same grasp coordinates.

Each candidate must already have been processed by
``plot_full_mount_joint_distributions.py`` and
``plot_full_mount_task_scene.py``.  Failed cases remain missing rather than
being assigned zero joint margin or zero angular span.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np

from plot_full_mount_joint_distributions import require, sha256
from plot_grasp_angle_map import pick_cjk_font


NAMES = ("v64_39", "v64_10", "v64_47", "v64_52")
COUNT_ZH = {4: "四", 5: "五"}
METRICS = (
    ("min_raw_margin_deg", "原始限位最短距离", "Raw URDF limit distance", "viridis"),
    ("min_effective_margin_deg", "有效限位最小余量", "Effective limit margin", "viridis"),
    ("max_cycle_joint_span_deg", "完整抓放单关节最大角跨度", "Full-cycle max joint span", "magma"),
)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Missing report: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def load_candidate(root: Path, name: str) -> dict:
    run = (root / "runs" / name).resolve()
    report_dir = root / "reports" / name
    joint = read_json(report_dir / "joint_distributions" / "joint_distribution_summary.json")
    scene = read_json(report_dir / "scene_distribution" / "task_scene_distribution.json")
    require(Path(joint["provenance"]["result"]).resolve() == run
            and Path(scene["provenance"]["result"]).resolve() == run,
            f"{name}: reports refer to another run")
    for report in (joint, scene):
        source = report["provenance"]
        for key, filename in (("trajectory_sha256", "trajectory.npz"),
                              ("metadata_sha256", "trajectory_meta.json"),
                              ("independent_verification_sha256", "independent_verification.json"),
                              ("joint_limit_clip_audit_sha256", "joint_limit_clip_audit.json")):
            require(sha256(run / filename) == source[key],
                    f"{name}: {filename} changed after report generation")
        require(sha256(Path(source["urdf"])) == source["urdf_sha256"],
                f"{name}: source URDF changed after report generation")
    for filename in ("independent_verification.json", "joint_limit_clip_audit.json"):
        audit = read_json(run / filename)
        require(audit.get("passed") is True and audit.get("verification_completed") is True,
                f"{name}: {filename} did not pass fully")
    require(joint["n_total"] == scene["n_total"] == 400
            and joint["n_success"] == scene["n_success"],
            f"{name}: layout and joint report counts differ")
    require(joint["tilt_axis"] == scene["tilt_axis"] == "original_base_local_y"
            and math.isclose(joint["tilt_deg"], scene["tilt_deg"], abs_tol=1e-9),
            f"{name}: local-Y tilt labels differ")
    require(np.allclose(joint["mount_xyz_task_world_m"], scene["mount_origin_m"], atol=1e-9),
            f"{name}: joint and layout base origins differ")
    meta = read_json(run / "trajectory_meta.json")
    require(meta["n_items_success"] == joint["n_success"]
            and meta["n_points"] == read_json(run / "independent_verification.json")["n_samples"],
            f"{name}: saved metadata and independent audit counts differ")
    rows = {int(case["index"]): case for case in joint["cases"]}
    require(len(rows) == 400 and set(rows) == set(range(400)),
            f"{name}: missing or duplicate grasp case")
    require(sum(case["success"] is True for case in rows.values()) == joint["n_success"],
            f"{name}: case count differs")
    for case in rows.values():
        require(type(case["success"]) is bool, f"{name}: invalid success flag")
        for key, *_ in METRICS:
            require((key in case) == case["success"],
                    f"{name}: failed case has a metric or successful case lacks one")
            if case["success"]:
                require(math.isfinite(float(case[key])), f"{name}: non-finite metric")
    return {"name": name, "run": run, "joint": joint, "scene": scene,
            "meta": meta, "rows": rows}


def metric_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {"count": len(values), "min": float(array.min()),
            "median": float(np.median(array)), "max": float(array.max())}



def span_extremes_excluding_first_saved(candidate: dict) -> dict:
    saved = [case for _, case in sorted(candidate["rows"].items()) if case["success"]]
    require(len(saved) >= 3, f"{candidate['name']}: too few successful cases for extremes")
    eligible = saved[1:]

    def detail(case: dict) -> dict:
        return {
            "case_number": int(case["case_number"]),
            "index": int(case["index"]),
            "grasp_task_world_m": [case["x_m"], case["y_m"], case["z_m"]],
            "joint": case["max_cycle_joint"],
            "span_deg": float(case["max_cycle_joint_span_deg"]),
            "sample_range_half_open": [int(case["sample_start"]), int(case["sample_end_exclusive"])],
        }

    return {
        "excluded_first_saved_case_number": int(saved[0]["case_number"]),
        "n_eligible": len(eligible),
        "maximum": detail(max(eligible, key=lambda case: case["max_cycle_joint_span_deg"])),
        "minimum": detail(min(eligible, key=lambda case: case["max_cycle_joint_span_deg"])),
    }


def matched_data(candidates: list[dict]) -> tuple[list[dict], dict]:
    first = candidates[0]
    grid = first["joint"]["grid"]
    xs = np.linspace(*grid["x_range"], grid["rows"])
    ys = np.linspace(*grid["y_range"], grid["cols"])
    for candidate in candidates[1:]:
        require(candidate["joint"]["grid"] == grid,
                f"{candidate['name']}: grasp grid differs")
        require(np.allclose(candidate["scene"]["place_position_m"],
                            first["scene"]["place_position_m"], atol=1e-9),
                f"{candidate['name']}: place position differs")
        require(candidate["joint"]["provenance"]["urdf_sha256"]
                == first["joint"]["provenance"]["urdf_sha256"],
                f"{candidate['name']}: URDF differs")
        require(math.isclose(candidate["joint"]["joint_limit_clip_rad"],
                             first["joint"]["joint_limit_clip_rad"], abs_tol=1e-12),
                f"{candidate['name']}: joint-limit clip differs")
    records = []
    all_success = any_success = 0
    for index in range(400):
        source = first["rows"][index]
        row, col = int(source["row"]), int(source["col"])
        require(np.allclose([source["x_m"], source["y_m"]],
                            [xs[row], ys[col]], atol=1e-8),
                f"Case {index}: source coordinates differ from grid")
        record = {"index": index, "case_number": index+1, "row": row, "col": col,
                  "x_m": source["x_m"], "y_m": source["y_m"], "z_m": source["z_m"]}
        successes = []
        for candidate in candidates:
            name, case = candidate["name"], candidate["rows"][index]
            require((int(case["row"]), int(case["col"])) == (row, col)
                    and np.allclose([case["x_m"], case["y_m"], case["z_m"]],
                                    [source["x_m"], source["y_m"], source["z_m"]],
                                    atol=1e-8),
                    f"{name}: case {index} is not the same grasp point")
            record[f"{name}_success"] = case["success"]
            successes.append(case["success"])
            for key, *_ in METRICS:
                record[f"{name}_{key}"] = case[key] if case["success"] else ""
        count = sum(successes)
        record["success_count"] = count
        all_success += count == len(candidates)
        any_success += count > 0
        records.append(record)
    pairwise = {}
    for i, left in enumerate(candidates):
        for right in candidates[i+1:]:
            key = f"{left['name']}_minus_{right['name']}"
            matched = [index for index in range(400)
                       if left["rows"][index]["success"] and right["rows"][index]["success"]]
            pairwise[key] = {
                "n_both_success": len(matched),
                "effective_margin_delta_deg": metric_stats([
                    left["rows"][index]["min_effective_margin_deg"]
                    - right["rows"][index]["min_effective_margin_deg"] for index in matched]),
                "max_joint_span_delta_deg": metric_stats([
                    left["rows"][index]["max_cycle_joint_span_deg"]
                    - right["rows"][index]["max_cycle_joint_span_deg"] for index in matched]),
            }
    summary = {
        "schema_version": 1, "coordinate_frame": "task_world",
        "grid": grid, "place_position_m": first["scene"]["place_position_m"],
        "n_all_candidates_success": all_success, "n_any_candidate_success": any_success,
        "n_none_success": 400-any_success, "pairwise_same_point": pairwise,
        "candidates": [{
            "name": c["name"], "run": str(c["run"]),
            "n_success": c["joint"]["n_success"],
            "mount_xyz_task_world_m": c["joint"]["mount_xyz_task_world_m"],
            "tilt_axis": c["joint"]["tilt_axis"],
            "tilt_deg": c["joint"]["tilt_deg"],
            "statistics_deg": c["joint"]["statistics_deg"],
            "n_trajectory_samples": c["meta"]["n_points"],
            "tcp_workspace_check": c["meta"].get("workspace_check"),
            "link6_gripper_extent_check": c["meta"].get("gripper_extent_check"),
            "span_extremes_excluding_first_saved": span_extremes_excluding_first_saved(c),
            "trajectory_sha256": c["joint"]["provenance"]["trajectory_sha256"],
            "metadata_sha256": c["joint"]["provenance"]["metadata_sha256"],
        } for c in candidates],
        "definitions": {
            "same_point": "All candidates use the same original task_world grasp coordinate, grid index and fixed place target.",
            "failed": "No saved full trajectory; joint metrics are blank, never zero. Planning failure does not prove geometric unreachability.",
            "min_raw_margin_deg": "Minimum distance to original URDF joint limits over all saved samples in the full pick/place cycle, over J1-J6.",
            "min_effective_margin_deg": "Minimum distance to planner limits clipped inward by joint_limit_clip_rad.",
            "max_cycle_joint_span_deg": "Largest max(q)-min(q) for a single J1-J6 joint across all six stages including entry. Values are not wrapped modulo 360.",
            "tcp_workspace_check": "Saved TCP points against configured workspace bounds; a zero count does not cover all robot links.",
            "link6_gripper_extent_check": "Nine LINK_6 collision spheres against original task workspace bounds; informational only, not the wall-collision test or success criterion.",
            "scope": "Only saved discrete samples and passed independent audits; no hardware or continuous swept-volume guarantee.",
        },
    }
    return records, summary


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    step = np.diff(values)
    return np.r_[values[0]-step[0]/2, values[:-1]+step/2, values[-1]+step[-1]/2]


def draw_comparison(candidates: list[dict], summary: dict, png: Path) -> None:
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if font else (lambda zh, en: en)
    grid = summary["grid"]
    xs = np.linspace(*grid["x_range"], grid["rows"])
    ys = np.linspace(*grid["y_range"], grid["cols"])
    xe, ye = centers_to_edges(xs), centers_to_edges(ys)
    fig, axes = plt.subplots(len(candidates), 1+len(METRICS),
                             figsize=(21, 18 * len(candidates) / 4),
                             sharex=True, sharey=True, constrained_layout=True)
    limits = {}
    for key, *_ in METRICS:
        values = [float(case[key]) for candidate in candidates
                  for case in candidate["rows"].values() if case["success"]]
        require(values, f"No successful values for {key}")
        limits[key] = Normalize(vmin=min(values), vmax=max(values))
    metric_mappables = {}
    for r, candidate in enumerate(candidates):
        name = candidate["name"]
        cases = candidate["rows"]
        success_map = np.zeros((len(ys), len(xs)), dtype=int)
        for case in cases.values():
            success_map[int(case["col"]), int(case["row"])] = int(case["success"])
        ax = axes[r, 0]
        ax.pcolormesh(xe, ye, success_map, cmap=ListedColormap(["#d2d2d2", "#45b7ab"]),
                      vmin=0, vmax=1, shading="flat")
        mount = candidate["joint"]["mount_xyz_task_world_m"]
        ax.scatter(mount[0], mount[1], marker="v", s=80, facecolors="white",
                   edgecolors="#1565c0", lw=1.5, zorder=4)
        ax.scatter(*summary["place_position_m"][:2], marker="X", s=75,
                   color="#d81b60", edgecolors="black", lw=.4, zorder=5)
        ax.set_ylabel(f"{name}\n({mount[0]:+.2f}, {mount[1]:+.2f}, {mount[2]:+.2f}) m\n"
                      f"{candidate['joint']['tilt_deg']:g}° | {candidate['joint']['n_success']}/400\n"
                      "task_world Y (m)", fontsize=10)
        for c, (key, zh, en, color) in enumerate(METRICS, start=1):
            ax = axes[r, c]
            data = np.full((len(ys), len(xs)), np.nan)
            for case in cases.values():
                if case["success"]:
                    data[int(case["col"]), int(case["row"])] = case[key]
            cmap = plt.get_cmap(color).copy()
            cmap.set_bad("#d2d2d2")
            metric_mappables[key] = ax.pcolormesh(
                xe, ye, np.ma.masked_invalid(data), cmap=cmap,
                norm=limits[key], shading="flat")
        for ax in axes[r]:
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=8)
            ax.grid(color="white", lw=.3, alpha=.35)
    axes[0, 0].set_title(L("完整抓放成功分布", "Complete pick/place coverage"), fontsize=12)
    for c, (key, zh, en, _) in enumerate(METRICS, start=1):
        axes[0, c].set_title(L(zh, en), fontsize=12)
        colorbar = fig.colorbar(metric_mappables[key], ax=axes[:, c], shrink=.82,
                                fraction=.025, pad=.02)
        colorbar.set_label("°")
    for ax in axes[-1]:
        ax.set_xlabel("task_world X (m)")
    legacy = tuple(candidate["name"] for candidate in candidates) == NAMES
    count = len(candidates)
    title_zh = ("四组 8/9 冒烟候选的 20×20 同点全量比较" if legacy else
                f"{COUNT_ZH[count]}组候选的 20×20 同点全量比较")
    title_en = ("Four 8/9 smoke candidates: matched 20×20 full-grid comparison" if legacy else
                f"{count} mount candidates: matched 20×20 full-grid comparison")
    fig.suptitle(L(title_zh, title_en), fontsize=18, fontweight="bold")
    fig.legend(handles=[Patch(facecolor="#45b7ab", label=L("有完整轨迹", "Complete trajectory")),
                        Patch(facecolor="#d2d2d2", label=L("无完整轨迹；指标缺失", "No trajectory; metric missing"))],
               loc="lower center", ncol=2, bbox_to_anchor=(.5, -.005), frameon=False)
    fig.savefig(png, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)



def draw_shared_scale_singles(candidates: list[dict], summary: dict, out: Path) -> None:
    """One four-panel image per mount, with metric colors fixed across mounts."""
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if font else (lambda zh, en: en)
    grid = summary["grid"]
    xs = np.linspace(*grid["x_range"], grid["rows"])
    ys = np.linspace(*grid["y_range"], grid["cols"])
    xe, ye = centers_to_edges(xs), centers_to_edges(ys)
    limits = {}
    for key, *_ in METRICS:
        values = [float(case[key]) for candidate in candidates
                  for case in candidate["rows"].values() if case["success"]]
        limits[key] = Normalize(vmin=min(values), vmax=max(values))
    for candidate in candidates:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11.5), constrained_layout=True)
        cases = candidate["rows"]
        coverage = np.zeros((len(ys), len(xs)), dtype=int)
        for case in cases.values():
            coverage[int(case["col"]), int(case["row"])] = int(case["success"])
        ax = axes[0, 0]
        ax.pcolormesh(xe, ye, coverage,
                      cmap=ListedColormap(["#d2d2d2", "#45b7ab"]),
                      vmin=0, vmax=1, shading="flat")
        ax.set_title(L("完整抓放成功分布", "Complete pick/place coverage"))
        base = candidate["joint"]["mount_xyz_task_world_m"]
        ax.scatter(base[0], base[1], marker="v", s=90, facecolors="white",
                   edgecolors="#1565c0", lw=1.7, zorder=4)
        ax.scatter(*summary["place_position_m"][:2], marker="X", s=90,
                   color="#d81b60", edgecolors="black", lw=.5, zorder=5)
        for ax, (key, zh, en, color) in zip((axes[0, 1], axes[1, 0], axes[1, 1]), METRICS):
            data = np.full((len(ys), len(xs)), np.nan)
            for case in cases.values():
                if case["success"]:
                    data[int(case["col"]), int(case["row"])] = case[key]
            cmap = plt.get_cmap(color).copy()
            cmap.set_bad("#d2d2d2")
            artist = ax.pcolormesh(xe, ye, np.ma.masked_invalid(data),
                                   cmap=cmap, norm=limits[key], shading="flat")
            ax.set_title(L(zh, en))
            cb = fig.colorbar(artist, ax=ax, fraction=.045, pad=.02)
            cb.set_label("°")
        for ax in axes.flat:
            ax.set_xlabel("task_world X (m)")
            ax.set_ylabel("task_world Y (m)")
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=8)
        fig.suptitle(f"{candidate['name']} | base ({base[0]:+.2f}, {base[1]:+.2f}, {base[2]:+.2f}) m | "
                     f"local +Y {candidate['joint']['tilt_deg']:g}° | "
                     f"{candidate['joint']['n_success']}/400", fontsize=15)
        fig.savefig(out / f"{candidate['name']}_shared_scale_distributions.png",
                    dpi=170, facecolor="white")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root containing runs/ and reports/")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--names", nargs="+", default=list(NAMES))
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out_dir or root / "reports" / "comparison").resolve()
    require(not out.exists(), f"Refusing to overwrite existing output directory: {out}")
    require(len(args.names) in COUNT_ZH and len(args.names) == len(set(args.names)),
            "Exactly four or five distinct candidate names are required")
    candidates = [load_candidate(root, name) for name in args.names]
    records, summary = matched_data(candidates)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".candidate_comparison_", dir=out.parent) as scratch:
        stage = Path(scratch) / "data"
        stage.mkdir()
        draw_comparison(candidates, summary, stage / "full_candidate_comparison.png")
        draw_shared_scale_singles(candidates, summary, stage)
        with (stage / "same_point_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (stage / "comparison_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
            encoding="utf-8")
        legacy = tuple(args.names) == NAMES
        count_zh = COUNT_ZH[len(args.names)]
        heading = ("四组 8/9 冒烟候选" if legacy else f"{count_zh}组候选")
        image_pattern = ("v64_XX_shared_scale_distributions.png" if legacy else
                         "<候选名>_shared_scale_distributions.png")
        (stage / "README.md").write_text(
            f"# {heading}的全量同点比较\n\n"
            f"- `full_candidate_comparison.png`：每行一组，依次显示完整轨迹成功分布、原始 URDF 限位最短距离、规划有效限位最小余量、完整抓放单关节最大角跨度。相同指标{count_zh}行共用色标。灰格无保存的完整轨迹，指标为空。\n"
            f"- `{image_pattern}`：各组单独四宫格，{count_zh}张图的相同指标也共用同一色标。\n"
            f"- `same_point_metrics.csv`：400 个相同 task_world 抓取点的{count_zh}组成功状态与指标。\n"
            "- `comparison_summary.json`：覆盖交并集、各组统计、两两共同成功点的指标差值。差值为名称左组减右组。\n\n"
            "所有指标只来自已通过独立轨迹及关节限位审计的保存采样；规划失败不等于已证明不可达。\n",
            encoding="utf-8")
        artifact_paths = [stage / "full_candidate_comparison.png",
                          stage / "same_point_metrics.csv", stage / "README.md"]
        artifact_paths += [stage / f"{name}_shared_scale_distributions.png"
                           for name in args.names]
        (stage / "artifact_sha256.json").write_text(
            json.dumps({path.name: sha256(path) for path in artifact_paths},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(stage, out)
    print(f"[ok] {out}: {[c['joint']['n_success'] for c in candidates]} /400; "
          f"all {len(candidates)} success={summary['n_all_candidates_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
