#!/usr/bin/env python3
"""Create audited grasp-end posture maps for four V66 and one V65 full runs."""

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
from matplotlib.colors import Normalize
import numpy as np

from analyze_grasp_posture import analyze_run, write_csv
from report_v68_full_grasp_posture import color_map, draw_map, font_setup
from run_local_y_mount_sweep import verify_inputs


ROOT = Path(__file__).resolve().parents[1] / "results_overhead/20260928"
V66 = ROOT / "v66_v64_8of9_local_y_full"
V65 = ROOT / "v65_near_zero_x_local_ytilt60_full"
TARGET = ROOT / "positive_local_y_five_full_summary/posture"
NAMES = ("v64_39", "v64_10", "v64_47", "v64_52", "v65_00")
POSTURE_THRESHOLD_DEG = 20.0


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def get_source_rows():
    v66_manifest_path = V66 / "manifest.json"
    v66_status_path = V66 / "sweep_status.json"
    v65_manifest_path = V65 / "posture_manifest.json"
    v66_manifest = read_json(v66_manifest_path)
    v66_status = read_json(v66_status_path)
    v65_manifest = read_json(v65_manifest_path)
    verify_inputs(v66_manifest, v66_manifest_path)
    require(v66_status["manifest_sha256"] == sha256(v66_manifest_path),
            "V66 status/manifest hash mismatch")
    v66_rows = v66_manifest["candidates"]
    require(tuple(row["name"] for row in v66_rows) == NAMES[:4] and
            v66_manifest["n_configs"] == 4, "Unexpected V66 full candidates")
    require(all(v66_status["outcomes"][name]["state"] == "completed_verified"
                for name in NAMES[:4]), "A V66 candidate is not completed_verified")
    v65_rows = v65_manifest["candidates"]
    require(len(v65_rows) == 1 and v65_rows[0]["name"] == NAMES[4],
            "Unexpected V65 posture manifest")
    rows = [*v66_rows, *v65_rows]
    require([row["local_y_tilt_deg"] for row in rows] ==
            [60.0, 45.0, 60.0, 0.0, 60.0],
            "Unexpected local +Y angles")
    outcomes = dict(v66_status["outcomes"])
    outcomes["v65_00"] = {"n_total": 400, "n_success": 383}
    sources = {
        "v66_manifest": v66_manifest_path,
        "v66_sweep_status": v66_status_path,
        "v65_posture_manifest": v65_manifest_path,
    }
    return rows, outcomes, sources


def audited_grids(manifest, status, report):
    by_name = {row["name"]: row for row in report["candidates"]}
    require(len(by_name) == 5 and set(by_name) == set(NAMES),
            "Posture analysis does not contain the five positive/zero-angle candidates")
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


def render(grids, summaries, out):
    font_setup()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    for ax, summary in zip(axes.flat, summaries):
        item = grids[summary["name"]]
        draw_map(ax, item)
        ax.set_title(
            f"{summary['name']} · 基座 {tuple(item['base_xyz_m'])} m · "
            f"{item['local_y_tilt_deg']:g}°\n"
            f"{summary['n_success']}/400 成功 · 姿态中位 {summary['angle_median_deg']:.1f}° · "
            f"≤20° {summary['n_within_20deg_of_vertical']} 点",
            fontsize=10.5)
    axes.flat[-1].axis("off")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 90), cmap=color_map()),
                       ax=axes.ravel().tolist(), shrink=.8, pad=.02)
    bar.set_label("抓取末端肩到腕连线距竖直方向 (°)")
    fig.suptitle("正向局部 +Y 倾角全量布置及 0° 对照 · 0° 直立，90° 水平；灰色无完整轨迹",
                 fontsize=14, fontweight="bold")
    fig.savefig(out / "all_five_grasp_posture.png", dpi=170, facecolor="white")
    plt.close(fig)
    for summary in summaries:
        name = summary["name"]
        item = grids[name]
        fig, ax = plt.subplots(figsize=(13, 11), layout="constrained")
        image = draw_map(ax, item, annotate=True)
        fig.colorbar(image, ax=ax, shrink=.86, label="肩到腕连线距竖直方向 (°)")
        fig.suptitle(
            f"{name} · 基座 {tuple(item['base_xyz_m'])} m · "
            f"局部 +Y 倾角 {item['local_y_tilt_deg']:g}°\n"
            f"成功 {summary['n_success']}/400 · 姿态中位 {summary['angle_median_deg']:.1f}° · "
            f"距竖直 ≤20°：{summary['n_within_20deg_of_vertical']} 点",
            fontsize=14, fontweight="bold")
        fig.savefig(out / f"{name}_grasp_posture_map.png", dpi=180, facecolor="white")
        plt.close(fig)


def write_report(stage, rows, sources, report, grids, summaries, positions):
    (stage / "full_grasp_posture.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    write_csv(report, stage / "full_grasp_posture")
    with (stage / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        fields = [
            "name", "base_x_m", "base_y_m", "base_z_m", "local_y_tilt_deg",
            "n_success", "n_total", "angle_min_deg", "angle_p10_deg",
            "angle_median_deg", "angle_p90_deg", "angle_max_deg",
            "n_within_20deg_of_vertical", "fraction_within_20deg_of_vertical",
            "independent_audit"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            x, y, z = summary["base_xyz_m"]
            writer.writerow({
                **{key: value for key, value in summary.items() if key != "base_xyz_m"},
                "base_x_m": x, "base_y_m": y, "base_z_m": z})
    with (stage / "grasp_positions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(positions[0]))
        writer.writeheader()
        writer.writerows(positions)
    provenance = {
        "schema_version": 1,
        "scope": "last saved sample of grasp stage in successful full trajectories",
        "source_documents": {key: {"path": str(path), "sha256": sha256(path)}
                             for key, path in sources.items()},
        "method": "analyze_grasp_posture.py independent URDF FK; J2/LINK_2 to J6/LINK_6 in task_world",
        "upright_threshold_deg": 20.0,
        "report_script_sha256": sha256(Path(__file__)),
        "core_posture_script_sha256": sha256(Path(__file__).with_name("analyze_grasp_posture.py")),
        "map_script_sha256": sha256(Path(__file__).with_name("report_v68_full_grasp_posture.py")),
        "per_run": {
            item["name"]: {
                "run": item["result"],
                "config_sha256": sha256(Path(item["config"])),
                "trajectory_sha256": sha256(Path(item["result"]) / "trajectory.npz"),
                "trajectory_meta_sha256": sha256(Path(item["result"]) / "trajectory_meta.json"),
                "independent_verification_sha256": sha256(
                    Path(item["result"]) / "independent_verification.json"),
            } for item in rows
        },
    }
    (stage / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render(grids, summaries, stage)
    lines = [
        "# 五组正向局部 +Y 布置的抓取末端姿态",
        "",
        "五组包括 V66 四组和 V65 一组。v64_52 倾角为 0°，是前一轮 8/9 冒烟筛选中的零倾角对照；其余四组为正向局部 +Y 倾角。图上灰格表示该位置没有保存完整抓放轨迹，红圈标记肩到腕连线距离世界竖直方向 ≤20° 的成功位置。所有图使用同一 0°–90° 色标。",
        "",
        "J2/LINK_2 是第二关节处肩部，J6/LINK_6 是第六关节处腕部；姿态角是两者连线与任务世界 task_world 竖直方向的锐角，0° 为直立、90° 为水平。测量取成功轨迹的抓取阶段最后一个保存采样点。它不表示肘部弯曲程度、整条轨迹的姿态、关节余量或双臂碰撞距离。",
        "",
        "独立 URDF（机器人描述文件）正向运动学重算每个抓取点，核对保存的 TCP（工具中心点）位置、轨迹/元数据/URDF 独立审核哈希，并验证 20×20 网格的每个原始抓取位置及 case 编号。summary.csv 给出每组统计，grasp_positions.csv 给出成功位置的世界坐标与姿态角；失败点无姿态角。网格 row 沿 X、col 沿 Y，绘图时矩阵已转置以使 X 水平、Y 垂直。",
        "",
        "复现命令：python3 xtrainer_plan/scripts/report_positive_full_grasp_posture.py。脚本要求 V66 四组 completed_verified、五组独立审核哈希通过，且拒绝覆盖已有报告。",
        "",
    ]
    (stage / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    rows, outcomes, sources = get_source_rows()
    analyses = []
    gridspec = None
    for item in rows:
        row = analyze_run(item["result"], item["name"])
        require(row["status"] == "observed" and row["audit"] == "verified",
                f"Unaudited or missing saved run: {item['name']}")
        config = read_json(Path(item["config"]))
        grid = config["pick_place"]["grasp_grid"]
        signature = (
            tuple(grid["x_range"]), tuple(grid["y_range"]), float(grid["z"]),
            int(grid["rows"]), int(grid["cols"]), bool(grid["perimeter_only"]))
        require(signature[-3:] == (20, 20, False),
                f"Unexpected task grid: {item['name']}")
        if gridspec is None:
            gridspec = signature
        require(signature == gridspec, f"Task grid differs: {item['name']}")
        require(row["n_success"] == outcomes[item["name"]]["n_success"] and
                row["n_total"] == 400 and row["n_observed_grasps"] == row["n_success"],
                f"Saved-success count mismatch: {item['name']}")
        analyses.append(row)
    v65_existing = read_json(V65 / "observed_posture.json")["candidates"][0]
    require(math.isclose(analyses[-1]["angle"]["median_deg"],
                         v65_existing["angle"]["median_deg"], abs_tol=1e-8),
            "New V65 posture differs from existing audited posture")
    report = {
        "schema_version": 1,
        "metric": {
            "shoulder": "J2/LINK_2 origin in production URDF",
            "wrist": "J6/LINK_6 origin in production URDF",
            "frame": "task_world",
            "angle": "acute angle to world vertical; 0 deg upright, 90 deg horizontal",
            "sampling": "last saved sample of successful grasp segment",
            "limitations": "No failed-point angle, elbow bend, full-path posture, or dual-arm clearance",
        },
        "sources": {key: str(value) for key, value in sources.items()},
        "baseline": None,
        "candidates": analyses,
    }
    grids, summaries, positions = audited_grids(
        {"candidates": rows}, {"outcomes": outcomes}, report)
    require(len(positions) == sum(row["n_success"] for row in summaries),
            "Position report does not cover all saved successful grasps")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {TARGET}")
    stage = Path(tempfile.mkdtemp(prefix=".posture_", dir=TARGET.parent))
    try:
        write_report(stage, rows, sources, report, grids, summaries, positions)
        os.rename(stage, TARGET)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    for row in summaries:
        print(f"{row['name']}: {row['n_success']}/400, "
              f"median={row['angle_median_deg']:.3f} deg, "
              f"within 20 deg={row['n_within_20deg_of_vertical']}")
    print(f"Report: {TARGET}")


if __name__ == "__main__":
    main()
