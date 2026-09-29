#!/usr/bin/env python3
"""Audit and compare the ten saved full-grid local-Y XTrainer mount candidates.

This is report-only. It never launches the planner or modifies source runs.
Failed cases retain missing joint/posture values; all numeric comparisons are
matched by the original 20x20 task_world grasp index.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

from plot_grasp_angle_map import pick_cjk_font


DATE_ROOT = Path(__file__).resolve().parents[1] / "results_overhead" / "20260928"
POSITIVE_ROOT = DATE_ROOT / "v66_v64_8of9_local_y_full"
V65_ROOT = DATE_ROOT / "v65_near_zero_x_local_ytilt60_full"
NEGATIVE_ROOT = DATE_ROOT / "v68_v67_negative_local_y_top5_full"
POSITIVE_POSTURE = DATE_ROOT / "positive_local_y_five_full_summary" / "posture"
NEGATIVE_POSTURE = NEGATIVE_ROOT / "reports" / "posture"
CANDIDATES = (
    ("v64_39", POSITIVE_ROOT, "positive"),
    ("v64_10", POSITIVE_ROOT, "positive"),
    ("v64_47", POSITIVE_ROOT, "positive"),
    ("v64_52", POSITIVE_ROOT, "zero_control"),
    ("v65_00", V65_ROOT, "positive"),
    ("v67_16", NEGATIVE_ROOT, "negative"),
    ("v67_44", NEGATIVE_ROOT, "negative"),
    ("v67_39", NEGATIVE_ROOT, "negative"),
    ("v67_24", NEGATIVE_ROOT, "negative"),
    ("v67_52", NEGATIVE_ROOT, "negative"),
)
METRICS = ("min_raw_margin_deg", "min_effective_margin_deg", "max_cycle_joint_span_deg")
POSTURE_KEY = "grasp_shoulder_wrist_from_vertical_deg"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    require(path.is_file(), f"Missing source file: {path}")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Missing JSON: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(obj, dict), f"Expected JSON object: {path}")
    return obj


def normalize_grid(grid: dict) -> dict:
    return {key: grid[key] for key in ("x_range", "y_range", "z", "rows", "cols", "order", "perimeter_only")}


def stats(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {"count": 0, "min": None, "median": None, "max": None}
    require(bool(np.all(np.isfinite(array))), "Nonfinite statistic input")
    return {"count": len(array), "min": float(array.min()),
            "median": float(np.median(array)), "max": float(array.max())}


def load_posture_sources() -> tuple[dict, dict]:
    result = {}
    provenance = {}
    for label, folder in (("positive", POSITIVE_POSTURE), ("negative", NEGATIVE_POSTURE)):
        path = folder / "full_grasp_posture_grasps.csv"
        prov = read_json(folder / "provenance.json")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        by_name = {}
        for row in rows:
            name = row["name"]
            index = int(row["item_index"])
            require(row["audit"] == "verified", f"Unverified posture: {name}/{index}")
            angle = float(row["angle_from_world_vertical_deg"])
            require(math.isfinite(angle) and 0 <= angle <= 90 + 1e-8,
                    f"Invalid posture angle: {name}/{index}")
            require(index not in by_name.setdefault(name, {}), f"Duplicate posture: {name}/{index}")
            by_name[name][index] = angle
        result[label] = by_name
        provenance[label] = {"path": str(folder / "provenance.json"),
                             "sha256": digest(folder / "provenance.json"),
                             "grasp_csv": str(path), "grasp_csv_sha256": digest(path),
                             "document": prov}
    return result, provenance


def load_candidate(name: str, root: Path, group: str, posture: dict, posture_prov: dict) -> dict:
    run = (root / "runs" / name).resolve()
    joint_path = (root / "joint_distributions" / "joint_distribution_summary.json"
                  if name == "v65_00" else
                  root / "reports" / name / "joint_distributions" / "joint_distribution_summary.json")
    joint = read_json(joint_path)
    scene_path = (root / "scene_distribution" / "layout_data.json"
                  if name == "v65_00" else
                  root / "reports" / name / "scene_distribution" / "task_scene_distribution.json")
    scene = read_json(scene_path)
    meta = read_json(run / "trajectory_meta.json")
    source = joint["provenance"]
    require(Path(source["result"]).resolve() == run, f"{name}: joint report points to another run")
    for key, filename in (("trajectory_sha256", "trajectory.npz"),
                          ("metadata_sha256", "trajectory_meta.json"),
                          ("independent_verification_sha256", "independent_verification.json"),
                          ("joint_limit_clip_audit_sha256", "joint_limit_clip_audit.json")):
        require(digest(run / filename) == source[key], f"{name}: {filename} changed")
    require(digest(Path(source["urdf"])) == source["urdf_sha256"],
            f"{name}: source URDF changed")
    for filename in ("independent_verification.json", "joint_limit_clip_audit.json"):
        audit = read_json(run / filename)
        require(audit.get("passed") is True and audit.get("verification_completed") is True,
                f"{name}: {filename} did not pass")
    require(joint["n_total"] == 400 and joint["n_success"] == meta["n_items_success"]
            and meta["n_items_total"] == 400, f"{name}: inconsistent run counts")
    require(joint["tilt_axis"] == "original_base_local_y", f"{name}: wrong tilt axis")
    require(math.isclose(float(joint["joint_limit_clip_rad"]), 0.14, abs_tol=1e-12),
            f"{name}: wrong joint clip")
    tilt = float(joint["tilt_deg"])
    require((group == "positive" and tilt > 0) or
            (group == "negative" and tilt < 0) or
            (group == "zero_control" and tilt == 0), f"{name}: angle/sign label mismatch")
    if name == "v65_00":
        require(scene["coordinate_frame"] == "task_world"
                and scene["n_grasp_total"] == 400
                and scene["n_complete_saved"] == joint["n_success"]
                and normalize_grid(scene["grasp_grid"]) == normalize_grid(joint["grid"])
                and digest(root / scene["config"]) == scene["config_sha256"]
                and digest(root / scene["trajectory_meta"]) == scene["trajectory_meta_sha256"],
                f"{name}: scene layout mismatch")
        base = scene["mount_origin_m"]
        place = scene["place_position_m"]
    else:
        require(scene["coordinate_frame"] == "task_world"
                and scene["n_total"] == 400
                and scene["n_success"] == joint["n_success"]
                and Path(scene["provenance"]["result"]).resolve() == run,
                f"{name}: scene layout mismatch")
        for key, filename in (("trajectory_sha256", "trajectory.npz"),
                              ("metadata_sha256", "trajectory_meta.json"),
                              ("independent_verification_sha256", "independent_verification.json"),
                              ("joint_limit_clip_audit_sha256", "joint_limit_clip_audit.json")):
            require(digest(run / filename) == scene["provenance"][key],
                    f"{name}: scene source {filename} changed")
        require(normalize_grid(scene["grid"]) == normalize_grid(joint["grid"]) and
                math.isclose(float(scene["tilt_deg"]), tilt, abs_tol=1e-9),
                f"{name}: grid or tilt differs between reports")
        base = scene["mount_origin_m"]
        place = scene["place_position_m"]
    require(np.allclose(base, joint["mount_xyz_task_world_m"], atol=1e-9),
            f"{name}: base differs between reports")
    posture_group = "negative" if group == "negative" else "positive"
    angles = posture[posture_group].get(name, {})
    provenance_row = posture_prov[posture_group]["document"]["per_run"][name]
    require(Path(provenance_row["run"]).resolve() == run, f"{name}: posture run differs")
    for key, filename in (("trajectory_sha256", "trajectory.npz"),
                          ("trajectory_meta_sha256", "trajectory_meta.json"),
                          ("independent_verification_sha256", "independent_verification.json")):
        require(digest(run / filename) == provenance_row[key],
                f"{name}: posture source {filename} changed")
    rows = {int(row["index"]): row for row in joint["cases"]}
    require(len(rows) == 400 and set(rows) == set(range(400)),
            f"{name}: incomplete or duplicate grasp indices")
    successes = {index for index, row in rows.items() if row["success"] is True}
    require(len(successes) == joint["n_success"] and set(angles) == successes,
            f"{name}: saved grasp and posture point sets differ")
    for index, row in rows.items():
        require(type(row["success"]) is bool, f"{name}/{index}: invalid success flag")
        for key in METRICS:
            require((key in row) == row["success"],
                    f"{name}/{index}: failed point has joint metrics or successful point lacks them")
            if row["success"]:
                require(math.isfinite(float(row[key])), f"{name}/{index}: nonfinite {key}")
    # Both V66 and V68 manifests pin each full config byte-for-byte.
    if name != "v65_00":
        manifest = read_json(root / "manifest.json")
        matches = [c for c in manifest["candidates"] if c["name"] == name]
        require(len(matches) == 1 and digest(Path(matches[0]["config"])) == matches[0]["config_sha256"],
                f"{name}: full config changed since manifest")
    return {"name": name, "group": group, "root": root, "run": run, "joint": joint,
            "scene_path": scene_path, "scene": scene, "joint_path": joint_path,
            "meta": meta, "base": base, "place": place, "rows": rows,
            "successes": successes, "angles": angles}


def protocol(meta: dict) -> dict:
    config = meta["config"]
    return {
        "robot": {key: val for key, val in config["robot"].items() if key != "mount_transform"},
        "pick_place": {key: val for key, val in config["pick_place"].items()
                       if key != "link0_target_transform"},
        "planner": config["planner"],
        "original_task_workspace": config["overhead"]["task_workspace"],
        "original_target_transform": config["overhead"]["original_target_transform"],
        "home_policy": config["overhead"]["home_policy"],
        "tcp_offset_m": config["overhead"]["tcp_offset_m"],
        "tilt_axis": config["overhead"]["tilt_axis"],
    }


def checked_candidates(posture: dict, posture_prov: dict) -> list[dict]:
    candidates = [load_candidate(*spec, posture, posture_prov) for spec in CANDIDATES]
    first = candidates[0]
    common_protocol = protocol(first["meta"])
    common_grid = normalize_grid(first["joint"]["grid"])
    for candidate in candidates[1:]:
        name = candidate["name"]
        require(protocol(candidate["meta"]) == common_protocol,
                f"{name}: task or planner protocol differs")
        require(normalize_grid(candidate["joint"]["grid"]) == common_grid,
                f"{name}: grasp grid differs")
        require(np.allclose(candidate["place"], first["place"], atol=1e-9),
                f"{name}: place target differs")
        require(candidate["joint"]["provenance"]["urdf_sha256"] ==
                first["joint"]["provenance"]["urdf_sha256"],
                f"{name}: URDF differs")
    return candidates


def compare(candidates: list[dict], posture_prov: dict) -> tuple[list[dict], dict]:
    first = candidates[0]
    grid = normalize_grid(first["joint"]["grid"])
    xs = np.linspace(*grid["x_range"], grid["rows"])
    ys = np.linspace(*grid["y_range"], grid["cols"])
    records = []
    groups = {"positive_including_zero": [c for c in candidates if c["group"] != "negative"],
              "strictly_positive": [c for c in candidates if c["group"] == "positive"],
              "negative": [c for c in candidates if c["group"] == "negative"]}
    for index in range(400):
        reference = first["rows"][index]
        row, col = int(reference["row"]), int(reference["col"])
        xyz = [float(reference[key]) for key in ("x_m", "y_m", "z_m")]
        require(np.allclose(xyz, [xs[row], ys[col], grid["z"]], atol=1e-8),
                f"First candidate index {index} differs from grid")
        item = {"index": index, "case_number": index + 1, "row": row, "col": col,
                "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]}
        for candidate in candidates:
            name = candidate["name"]
            case = candidate["rows"][index]
            require((case["row"], case["col"]) == (row, col) and
                    np.allclose([case["x_m"], case["y_m"], case["z_m"]], xyz, atol=1e-8),
                    f"{name}: case {index} differs from original task point")
            item[f"{name}_success"] = case["success"]
            for key in METRICS:
                item[f"{name}_{key}"] = case[key] if case["success"] else ""
            item[f"{name}_{POSTURE_KEY}"] = candidate["angles"][index] if case["success"] else ""
        item["n_success_ten"] = sum(item[f"{c['name']}_success"] for c in candidates)
        for group, members in groups.items():
            item[f"any_{group}_success"] = any(item[f"{c['name']}_success"] for c in members)
        records.append(item)
    out_candidates = []
    for candidate in candidates:
        name = candidate["name"]
        good = [candidate["rows"][index] for index in sorted(candidate["successes"])]
        spans = [float(case["max_cycle_joint_span_deg"]) for case in good]
        angles = [candidate["angles"][index] for index in sorted(candidate["successes"])]
        out_candidates.append({
            "name": name, "angle_group": candidate["group"],
            "base_xyz_task_world_m": candidate["base"],
            "original_base_local_y_tilt_deg": float(candidate["joint"]["tilt_deg"]),
            "n_success": len(good), "n_failed": 400 - len(good),
            "raw_margin_deg": stats(case["min_raw_margin_deg"] for case in good),
            "effective_margin_deg": stats(case["min_effective_margin_deg"] for case in good),
            "max_cycle_joint_span_deg": stats(spans),
            "n_span_strictly_over_170deg": sum(value > 170 for value in spans),
            "n_span_strictly_over_200deg": sum(value > 200 for value in spans),
            "grasp_shoulder_wrist_from_vertical_deg": stats(angles),
            "n_grasp_within_20deg_of_vertical": sum(value <= 20 for value in angles),
            "n_trajectory_samples": int(candidate["meta"]["n_points"]),
            "source": {
                "run": str(candidate["run"]),
                "joint_report": str(candidate["joint_path"]),
                "joint_report_sha256": digest(candidate["joint_path"]),
                "scene_report": str(candidate["scene_path"]),
                "scene_report_sha256": digest(candidate["scene_path"]),
                "trajectory_sha256": candidate["joint"]["provenance"]["trajectory_sha256"],
                "trajectory_meta_sha256": candidate["joint"]["provenance"]["metadata_sha256"],
                "independent_verification_sha256":
                    candidate["joint"]["provenance"]["independent_verification_sha256"],
                "joint_limit_clip_audit_sha256":
                    candidate["joint"]["provenance"]["joint_limit_clip_audit_sha256"],
            },
        })
    positive_set = set.union(*(c["successes"] for c in groups["positive_including_zero"]))
    negative_set = set.union(*(c["successes"] for c in groups["negative"]))
    strictly_positive_set = set.union(*(c["successes"] for c in groups["strictly_positive"]))
    pairwise = {}
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            intersection = sorted(left["successes"] & right["successes"])
            pairwise[f"{left['name']}__{right['name']}"] = {
                "n_both_success": len(intersection),
                "n_either_success": len(left["successes"] | right["successes"]),
                "effective_margin_delta_left_minus_right_deg": stats(
                    left["rows"][index]["min_effective_margin_deg"] -
                    right["rows"][index]["min_effective_margin_deg"] for index in intersection),
                "max_cycle_joint_span_delta_left_minus_right_deg": stats(
                    left["rows"][index]["max_cycle_joint_span_deg"] -
                    right["rows"][index]["max_cycle_joint_span_deg"] for index in intersection),
            }
    summary = {
        "schema_version": 1, "coordinate_frame": "task_world",
        "grid": grid, "fixed_place_position_m": first["place"],
        "joint_limit_clip_rad": 0.14,
        "shared_urdf_sha256": first["joint"]["provenance"]["urdf_sha256"],
        "shared_protocol_sha256": hashlib.sha256(json.dumps(protocol(first["meta"]), sort_keys=True,
                                                       separators=(",", ":")).encode()).hexdigest(),
        "n_any_of_ten_success": sum(bool(item["n_success_ten"]) for item in records),
        "n_all_ten_success": sum(item["n_success_ten"] == 10 for item in records),
        "n_any_positive_including_zero_success": len(positive_set),
        "n_any_strictly_positive_success": len(strictly_positive_set),
        "n_any_negative_success": len(negative_set),
        "n_any_both_positive_including_zero_and_negative_success": len(positive_set & negative_set),
        "n_neither_sign_success": 400 - len(positive_set | negative_set),
        "posture_sources": {key: {k: v for k, v in value.items() if k != "document"}
                            for key, value in posture_prov.items()},
        "candidates": out_candidates,
        "pairwise_same_point": pairwise,
        "definitions": {
            "zero_control": "v64_52 uses 0° local-Y tilt; it is retained among five earlier selected full candidates, not counted as a positive angle.",
            "success": "Saved complete pick/place trajectory for the original task_world grid point and shared fixed place target.",
            "failed_metrics": "Joint and posture metrics are blank when no full trajectory was saved; failure does not prove geometric unreachability.",
            "min_raw_margin_deg": "Smallest distance over saved J1-J6 samples to original URDF joint limits.",
            "min_effective_margin_deg": "Smallest distance over saved J1-J6 samples to planner limits clipped inward by 0.14 rad.",
            "max_cycle_joint_span_deg": "Largest max(q)-min(q) of one J1-J6 joint over the full saved cycle including entry, without 360° wrapping.",
            POSTURE_KEY: "Acute angle from task_world vertical of J2 shoulder to J6 wrist line at the last saved grasp-stage sample; 0° is upright.",
            "threshold_counts": "Span counts use strict >170° and >200°; upright count uses inclusive <=20°. All are among successful trajectories.",
            "scope": "Audited discrete saved trajectories from ten independent single-arm layouts; no simultaneous dual-arm collision conclusion.",
        },
    }
    return records, summary


def centers_to_edges(points: np.ndarray) -> np.ndarray:
    step = np.diff(points)
    return np.r_[points[0] - step[0] / 2, points[:-1] + step / 2,
                 points[-1] + step[-1] / 2]


def draw(candidates: list[dict], summary: dict, png: Path) -> None:
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if font else (lambda zh, en: en)
    grid = summary["grid"]
    xs = np.linspace(*grid["x_range"], grid["rows"])
    ys = np.linspace(*grid["y_range"], grid["cols"])
    xe, ye = centers_to_edges(xs), centers_to_edges(ys)
    configs = (
        ("success", L("完整轨迹覆盖", "Complete trajectory coverage"), None, None),
        ("min_effective_margin_deg", L("最小有效限位余量", "Minimum effective joint margin"), "viridis", None),
        ("max_cycle_joint_span_deg", L("最大单关节全周期跨度", "Max full-cycle single-joint span"), "magma", None),
        (POSTURE_KEY, L("抓取时肩腕离竖直角", "Grasp shoulder-wrist angle from vertical"), "cividis", Normalize(0, 90)),
    )
    norm = {}
    for key, _, _, fixed in configs[1:]:
        vals = [float(c["rows"][i][key]) if key != POSTURE_KEY else c["angles"][i]
                for c in candidates for i in c["successes"]]
        norm[key] = fixed or Normalize(0, max(vals))
    fig, axes = plt.subplots(len(candidates), len(configs), figsize=(20, 29),
                             sharex=True, sharey=True, constrained_layout=True)
    mappables = {}
    for row, c in enumerate(candidates):
        for col, (key, title, color, _) in enumerate(configs):
            ax = axes[row, col]
            arr = np.full((len(ys), len(xs)), np.nan)
            for index, case in c["rows"].items():
                r, k = int(case["row"]), int(case["col"])
                if key == "success":
                    arr[k, r] = int(case["success"])
                elif case["success"]:
                    arr[k, r] = c["angles"][index] if key == POSTURE_KEY else case[key]
            if key == "success":
                artist = ax.pcolormesh(xe, ye, arr,
                                       cmap=ListedColormap(["#d0d4d8", "#319f93"]),
                                       vmin=0, vmax=1, shading="flat")
            else:
                cmap = plt.get_cmap(color).copy()
                cmap.set_bad("#d0d4d8")
                artist = ax.pcolormesh(xe, ye, np.ma.masked_invalid(arr),
                                       cmap=cmap, norm=norm[key], shading="flat")
                mappables[key] = artist
            if col == 0:
                ax.scatter(c["base"][0], c["base"][1], marker="v", s=80,
                           facecolor="white", edgecolor="#176ba0", lw=1.5, zorder=4)
                ax.scatter(*c["place"][:2], marker="X", s=65,
                           color="#e01c74", edgecolors="black", lw=.4, zorder=5)
                group_label = L("零角对照", "0° control") if c["group"] == "zero_control" else ""
                ax.set_ylabel(f"{c['name']} {group_label}\n"
                              f"{c['joint']['tilt_deg']:+g}°  {c['joint']['n_success']}/400\n"
                              f"({c['base'][0]:+.2f}, {c['base'][1]:+.2f}, {c['base'][2]:+.2f}) m\n"
                              "task_world Y (m)", fontsize=8)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=7)
            ax.grid(color="white", lw=.25, alpha=.35)
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold")
            if row == len(candidates) - 1:
                ax.set_xlabel("task_world X (m)", fontsize=9)
    for col, (key, _, _, _) in enumerate(configs[1:], start=1):
        cb = fig.colorbar(mappables[key], ax=axes[:, col], shrink=.75,
                          fraction=.018, pad=.01)
        cb.set_label("°", fontsize=10)
    fig.suptitle(L("局部 +Y 倾角：正向／零角／负向十组 20×20 同点全量对照",
                   "Local +Y tilt: ten signed-angle full-grid candidates"),
                 fontsize=19, fontweight="bold")
    fig.legend(handles=[Patch(facecolor="#319f93", label=L("有完整轨迹", "Complete trajectory")),
                        Patch(facecolor="#d0d4d8", label=L("无完整轨迹，指标缺失", "No saved trajectory"))],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.004))
    fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=DATE_ROOT / "signed_local_y_full_comparison" / "reports")
    args = parser.parse_args()
    output = args.out_dir.resolve()
    require(not output.exists(), f"Refusing to overwrite existing report directory: {output}")
    posture, posture_prov = load_posture_sources()
    candidates = checked_candidates(posture, posture_prov)
    records, summary = compare(candidates, posture_prov)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".signed_full_comparison_", dir=output.parent) as scratch:
        stage = Path(scratch) / "reports"
        stage.mkdir()
        chart = stage / "full_ten_shared_grid.png"
        draw(candidates, summary, chart)
        csv_path = stage / "same_point_metrics.csv"
        with csv_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        summary_path = stage / "comparison_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                           allow_nan=False) + "\n", encoding="utf-8")
        readme = stage / "README.md"
        readme.write_text(
            "# 正向、零角与负向全量同点对照的数据\n\n"
            "运行 `python3 xtrainer_plan/scripts/compare_signed_full_mounts.py` 可从已审核的 V66、V65、V68 保存结果重新生成本目录；不会重跑规划。\n\n"
            "- `full_ten_shared_grid.png`：十组共用坐标与各指标色标；每行依次为完整轨迹覆盖、有效关节限位余量、完整周期最大单关节跨度、抓取末端肩腕连线与竖直方向的夹角。蓝色倒三角是基座，粉色叉号是固定放置点。灰色格子无保存轨迹，因此没有关节或姿态值。\n"
            "- `same_point_metrics.csv`：400 个相同的 `task_world` 抓取点、十组逐点成功状态、关节数值和抓取姿态角。布尔值为 `True`／`False`，失败点的数值字段留空。\n"
            "- `comparison_summary.json`：来源哈希、协议一致性、覆盖交并集、逐组统计和两两同点比较。`v64_52` 的局部 +Y 倾角是 0° 对照，严格正角集合不含它。\n"
            "- `artifact_sha256.json`：上述三个文件和本说明的 SHA-256 摘要。\n\n"
            "有效余量按规划器向内收紧 0.14 rad 的限位计算；全周期角跨度包含入场段，且不按 360° 折返。图中单臂成功不能据此判断两臂同时运行时无碰撞。\n",
            encoding="utf-8")
        (stage / "artifact_sha256.json").write_text(
            json.dumps({p.name: digest(p) for p in (chart, csv_path, summary_path, readme)},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(stage, output)
    print(json.dumps({"out_dir": str(output),
                      "candidate_success_counts": {c["name"]: len(c["successes"]) for c in candidates},
                      "n_any_of_ten_success": summary["n_any_of_ten_success"],
                      "n_all_ten_success": summary["n_all_ten_success"],
                      "n_positive_any": summary["n_any_positive_including_zero_success"],
                      "n_negative_any": summary["n_any_negative_success"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
