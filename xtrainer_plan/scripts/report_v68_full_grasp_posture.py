#!/usr/bin/env python3
"""Audit and map V68 grasp-end shoulder-to-wrist posture on five 20×20 grids.

This report-only script refuses to run until all five full candidates are
completed_verified. It reuses analyze_grasp_posture.py's independent URDF FK
measurement and checks every saved grasp against its task-world grid position.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np

from analyze_grasp_posture import analyze_manifest, write_csv
from run_local_y_mount_sweep import verify_inputs


DEFAULT_ROOT = (Path(__file__).resolve().parents[1] / "results_overhead/20260928"
                / "v68_v67_negative_local_y_top5_full")
EXPECTED_NAMES = ("v67_16", "v67_44", "v67_39", "v67_24", "v67_52")
POSTURE_THRESHOLD_DEG = 20.0


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def font_setup():
    names = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
        if family in names:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False


def completed_manifest(root):
    manifest_path = root / "manifest.json"
    status_path = root / "sweep_status.json"
    manifest = read_json(manifest_path)
    status = read_json(status_path)
    verify_inputs(manifest, manifest_path)
    rows = manifest["candidates"]
    require(manifest["n_configs"] == len(rows) == 5 and
            tuple(row["name"] for row in rows) == EXPECTED_NAMES,
            "Expected the frozen V68 five-candidate manifest")
    params = manifest["parameters"]
    require(params["tilt_axis"] == "original_base_local_y" and
            params["grid_rows"] == params["grid_cols"] == 20 and
            all(row["local_y_tilt_deg"] in (-30.0, -45.0, -60.0) for row in rows),
            "Expected 20×20 negative-local-Y V68 planning inputs")
    require(status["manifest_sha256"] == sha256(manifest_path) and
            set(status["outcomes"]) == set(EXPECTED_NAMES) and
            all(status["outcomes"][name]["state"] == "completed_verified"
                for name in EXPECTED_NAMES),
            "Wait until all five V68 runs are completed_verified")
    return manifest_path, manifest, status_path, status


def audited_grids(manifest, status, report):
    by_name = {row["name"]: row for row in report["candidates"]}
    require(len(by_name) == 5 and set(by_name) == set(EXPECTED_NAMES),
            "Posture analysis does not contain the five V68 candidates")
    grids, summaries, positions = {}, [], []
    for item in manifest["candidates"]:
        name = item["name"]
        row = by_name[name]
        outcome = status["outcomes"][name]
        require(row["status"] == "observed" and row["audit"] == "verified" and
                row["n_total"] == outcome["n_total"] == 400 and
                row["n_success"] == row["n_observed_grasps"] == outcome["n_success"] and
                len(row["grasps"]) == row["n_success"],
                f"Missing or unaudited grasp endpoints: {name}")
        config = read_json(Path(item["config"]))
        grid = config["pick_place"]["grasp_grid"]
        require(grid["rows"] == grid["cols"] == 20 and not grid["perimeter_only"] and
                config["overhead"]["tilt_axis"] == "original_base_local_y" and
                config["overhead"]["local_y_tilt_deg"] == item["local_y_tilt_deg"],
                f"Unexpected full grid/tilt metadata: {name}")
        run = Path(item["result"])
        meta = read_json(run / "trajectory_meta.json")
        cases = {int(case["index"]): case for case in meta["items"]}
        require(len(cases) == 400 and len(meta["items"]) == 400 and
                set(cases) == set(range(400)),
                f"Expected 400 uniquely indexed full-grid cases: {name}")
        xs = np.linspace(*grid["x_range"], 20)
        ys = np.linspace(*grid["y_range"], 20)
        case_cells = set()
        for index, case in cases.items():
            r, c = int(case["row"]), int(case["col"])
            expected_position = (float(xs[r]), float(ys[c]), float(grid["z"]))
            require(0 <= r < 20 and 0 <= c < 20 and (r, c) not in case_cells and
                    np.allclose(case["position_raw"], expected_position,
                                rtol=0, atol=1e-8),
                    f"Full-grid case differs from task-world cell: {name}, {index}")
            case_cells.add((r, c))
        require(len(case_cells) == 400,
                f"Full-grid cases do not cover every task-world cell: {name}")
        # The planner's row indexes X and col indexes Y; image rows index Y.
        matrix = np.full((20, 20), np.nan)
        seen = set()
        for grasp in row["grasps"]:
            index, r, c = (int(grasp[key]) for key in ("item_index", "row", "col"))
            require(0 <= index < 400 and 0 <= r < 20 and 0 <= c < 20 and
                    (r, c) not in seen, f"Repeated or invalid grasp location: {name}, {index}")
            case = cases[index]
            expected_position = (float(xs[r]), float(ys[c]), float(grid["z"]))
            require(case["success"] and int(case["row"]) == r and
                    int(case["col"]) == c and
                    np.allclose(case["position_raw"], expected_position,
                                rtol=0, atol=1e-8),
                    f"Saved grasp differs from task-world grid cell: {name}, {index}")
            angle = float(grasp["angle_from_world_vertical_deg"])
            require(math.isfinite(angle) and 0 <= angle <= 90 and
                    0 <= grasp["saved_tcp_fk_error_mm"] <= 0.1,
                    f"Invalid posture angle or TCP FK error: {name}, {index}")
            matrix[c, r] = angle
            seen.add((r, c))
            positions.append({"name": name, "item_index": index,
                              "case": index + 1, "row": r, "col": c,
                              "grasp_x_m": expected_position[0],
                              "grasp_y_m": expected_position[1],
                              "grasp_z_m": expected_position[2],
                              "shoulder_wrist_from_vertical_deg": angle,
                              "within_20deg_of_vertical": angle <= POSTURE_THRESHOLD_DEG})
        values = matrix[np.isfinite(matrix)]
        require(len(values) == row["n_success"] and
                math.isclose(float(np.median(values)), row["angle"]["median_deg"],
                             rel_tol=0, abs_tol=1e-8) and
                math.isclose(float(np.mean(values <= POSTURE_THRESHOLD_DEG)),
                             row["angle"]["fraction_within_20deg_of_vertical"],
                             rel_tol=0, abs_tol=1e-8),
                f"Position map disagrees with audited posture summary: {name}")
        grids[name] = {"matrix": matrix, "xs": xs, "ys": ys,
                       "base_xyz_m": item["base_xyz_m"],
                       "local_y_tilt_deg": item["local_y_tilt_deg"]}
        summaries.append({"name": name, "base_xyz_m": item["base_xyz_m"],
                          "local_y_tilt_deg": item["local_y_tilt_deg"],
                          "n_success": row["n_success"], "n_total": 400,
                          "angle_min_deg": row["angle"]["min_deg"],
                          "angle_p10_deg": row["angle"]["p10_deg"],
                          "angle_median_deg": row["angle"]["median_deg"],
                          "angle_p90_deg": row["angle"]["p90_deg"],
                          "angle_max_deg": row["angle"]["max_deg"],
                          "n_within_20deg_of_vertical": int(np.sum(values <= POSTURE_THRESHOLD_DEG)),
                          "fraction_within_20deg_of_vertical":
                              row["angle"]["fraction_within_20deg_of_vertical"],
                          "independent_audit": "verified"})
    return grids, summaries, positions


def color_map():
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e3e8ed")
    return cmap


def draw_map(ax, item, *, annotate=False):
    matrix, xs, ys = item["matrix"], item["xs"], item["ys"]
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    image = ax.imshow(matrix, origin="lower", cmap=color_map(), vmin=0, vmax=90,
                      extent=(xs[0] - dx/2, xs[-1] + dx/2,
                              ys[0] - dy/2, ys[-1] + dy/2), aspect="equal")
    near = np.argwhere(np.isfinite(matrix) & (matrix <= POSTURE_THRESHOLD_DEG))
    if len(near):
        ax.scatter(xs[near[:, 1]], ys[near[:, 0]], s=30 if not annotate else 48,
                   marker="o", facecolors="none", edgecolors="#e2354a",
                   linewidths=1.3, label="距竖直 ≤20°")
    if annotate:
        for r, c in np.ndindex(matrix.shape):
            value = matrix[r, c]
            if math.isfinite(value):
                ax.text(xs[c], ys[r], f"{value:.0f}", ha="center", va="center",
                        fontsize=5.5, color="white" if value < 52 else "#162635")
        ax.set_xticks(xs[::2])
        ax.set_yticks(ys[::2])
        ax.tick_params(axis="x", labelrotation=45)
    else:
        ax.set_xticks(np.linspace(xs[0], xs[-1], 5))
        ax.set_yticks(np.linspace(ys[0], ys[-1], 6))
    ax.set_xlabel("grasp X / m")
    ax.set_ylabel("grasp Y / m")
    ax.tick_params(labelsize=8)
    return image


def render_plots(grids, summaries, out):
    font_setup()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    for ax, summary in zip(axes.flat, summaries):
        name = summary["name"]
        item = grids[name]
        draw_map(ax, item)
        ax.set_title(f"{name} · 基座 {tuple(item['base_xyz_m'])} m · "
                     f"{item['local_y_tilt_deg']:g}°\n"
                     f"{summary['n_success']}/400 成功 · 姿态中位 {summary['angle_median_deg']:.1f}° · "
                     f"≤20° {summary['n_within_20deg_of_vertical']} 点", fontsize=10.5)
    axes.flat[-1].axis("off")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 90), cmap=color_map()),
                       ax=axes.ravel().tolist(), shrink=.8, pad=.02)
    bar.set_label("抓取末端肩到腕连线距竖直方向 (°)")
    fig.suptitle("V68 五组全量抓取姿态 · 0° 直立，90° 水平；灰色无完整轨迹",
                 fontsize=15, fontweight="bold")
    fig.savefig(out / "all_five_grasp_posture.png", dpi=170, facecolor="white")
    plt.close(fig)
    for summary in summaries:
        name = summary["name"]
        item = grids[name]
        fig, ax = plt.subplots(figsize=(13, 11), layout="constrained")
        image = draw_map(ax, item, annotate=True)
        fig.colorbar(image, ax=ax, shrink=.86,
                     label="肩到腕连线距竖直方向 (°)")
        fig.suptitle(f"{name} · 基座 {tuple(item['base_xyz_m'])} m · "
                     f"局部 +Y 倾角 {item['local_y_tilt_deg']:g}°\n"
                     f"成功 {summary['n_success']}/400 · 姿态中位 {summary['angle_median_deg']:.1f}° · "
                     f"距竖直 ≤20°：{summary['n_within_20deg_of_vertical']} 点",
                     fontsize=14, fontweight="bold")
        fig.savefig(out / f"{name}_grasp_posture_map.png", dpi=180, facecolor="white")
        plt.close(fig)


def write_outputs(root, stage, manifest_path, manifest, status_path, report, summaries, positions, grids):
    report_path = stage / "full_grasp_posture.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    write_csv(report, stage / "full_grasp_posture")
    summary_csv = stage / "summary.csv"
    with summary_csv.open("x", newline="", encoding="utf-8") as stream:
        fields = ["name", "base_x_m", "base_y_m", "base_z_m", "local_y_tilt_deg",
                  "n_success", "n_total", "angle_min_deg", "angle_p10_deg",
                  "angle_median_deg", "angle_p90_deg", "angle_max_deg",
                  "n_within_20deg_of_vertical", "fraction_within_20deg_of_vertical",
                  "independent_audit"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            x, y, z = row["base_xyz_m"]
            writer.writerow({**{key: value for key, value in row.items() if key != "base_xyz_m"},
                             "base_x_m": x, "base_y_m": y, "base_z_m": z})
    with (stage / "grasp_positions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(positions[0]))
        writer.writeheader()
        writer.writerows(positions)
    provenance = {"schema_version": 1, "scope": "saved successful grasp endpoints only",
                  "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                  "sweep_status": str(status_path), "sweep_status_sha256": sha256(status_path),
                  "urdf_fk_method": "analyze_grasp_posture.py: J2/LINK_2 shoulder to J6/LINK_6 wrist in task_world",
                  "upright_threshold_deg": POSTURE_THRESHOLD_DEG,
                  "analysis_script_sha256": sha256(Path(__file__)),
                  "core_posture_script_sha256": sha256(Path(__file__).with_name("analyze_grasp_posture.py")),
                  "per_run": {item["name"]: {"run": item["result"],
                              "trajectory_sha256": sha256(Path(item["result"]) / "trajectory.npz"),
                              "trajectory_meta_sha256": sha256(Path(item["result"]) / "trajectory_meta.json"),
                              "independent_verification_sha256": sha256(
                                  Path(item["result"]) / "independent_verification.json")}
                              for item in manifest["candidates"]}}
    (stage / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_plots(grids, summaries, stage)
    lines = ["# V68 五组全量抓取姿态", "",
             "`all_five_grasp_posture.png` 比较五个安装布置的 20×20 抓取位置分布；每组还有带逐格角度数字的单独 PNG。灰格为没有保存完整抓放轨迹的位置，不能赋予姿态角。红圈标记肩到腕连线距竖直方向 ≤20° 的成功位置。",
             "", "J2/LINK_2 是第二关节处肩部，J6/LINK_6 是第六关节处腕部；姿态角取两者连线与任务世界 `task_world` 竖直方向的锐角，0° 为直立、90° 为水平。这个指标在保存轨迹的抓取阶段末端计算，不描述肘部弯曲，也不代表整条轨迹都保持同一姿态。",
             "", "数据由独立 URDF（机器人描述文件）正向运动学重算，并核对轨迹、元数据和 URDF 的独立审核哈希及 TCP（工具中心点）误差。`summary.csv` 给出每组中位角、10%/90% 分位数与 ≤20° 占比；`grasp_positions.csv` 给出每个成功位置的角度和坐标。失败位置没有姿态测量。任务网格的 `row` 沿 X、`col` 沿 Y；图片显示时 X 为横轴、Y 为纵轴，因此绘图矩阵已按此转置。配置中的 `order=ring` 仅在 `perimeter_only=true` 时产生环形顺序，本轮 20×20 全网格是逐行顺序。",
             "", "复现：`python3 xtrainer_plan/scripts/report_v68_full_grasp_posture.py --root "
             "xtrainer_plan/results_overhead/20260928/v68_v67_negative_local_y_top5_full`。脚本要求五组全部 `completed_verified`，拒绝覆盖已有报告。", ""]
    (stage / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path, manifest, status_path, status = completed_manifest(root)
    target = root / "reports/posture"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing posture report: {target}")
    report = analyze_manifest(manifest_path)
    grids, summaries, positions = audited_grids(manifest, status, report)
    require(len(positions) == sum(row["n_success"] for row in summaries),
            "Per-position report does not cover every successful grasp")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".posture_", dir=target.parent))
    try:
        write_outputs(root, stage, manifest_path, manifest, status_path,
                      report, summaries, positions, grids)
        if target.exists():
            raise FileExistsError(target)
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"[POSTURE] {target}: {len(summaries)} audited full runs, "
          f"{len(positions)} successful grasp endpoints")


if __name__ == "__main__":
    main()
