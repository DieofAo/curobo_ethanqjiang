#!/usr/bin/env python3
"""Side-by-side actual joint-angle maps for the three audited original-base runs.

All angle panels share [-180, 180] degrees. All effective-margin panels share
one linear color scale. Failed cases are missing values, never zero angles.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_original_base_threeway import distribution, read_json, require, sha256
from plot_grasp_angle_map import pick_cjk_font
from plot_original_base_joint_maps import METRICS, draw_panel, load_rows


def validate_alignment(datasets):
    require(len(datasets) == 3, "Expected exactly three runs")
    reference = datasets[0][1]
    reference_by_id = {row["index"]: row for row in reference}
    require(len(reference_by_id) == len(reference), "Duplicate case index")
    for meta, rows, report in datasets:
        by_id = {row["index"]: row for row in rows}
        require(len(rows) == len(reference) == len(by_id) and by_id.keys() == reference_by_id.keys(),
                "Case index coverage differs")
        require(meta["grid"] == datasets[0][0]["grid"], "Configured grids differ")
        require(np.allclose(meta["place_position_raw"], datasets[0][0]["place_position_raw"], atol=1e-10, rtol=0),
                "Place positions differ")
        require(report["joint"] == datasets[0][2]["joint"], "Joint names differ")
        for key in ("raw_limits_deg", "effective_limits_deg"):
            require(np.allclose(report[key], datasets[0][2][key], atol=1e-10, rtol=0), "Joint limits differ")
        for index, row in by_id.items():
            other = reference_by_id[index]
            require(row["row"] == other["row"] and row["col"] == other["col"]
                    and np.allclose([row[k] for k in ("x_m", "y_m", "z_m")],
                                    [other[k] for k in ("x_m", "y_m", "z_m")], atol=1e-10, rtol=0),
                    f"Case {index}: target or row/column differs")
    return sorted(set.intersection(*[{r["index"] for r in rows if r["success"]}
                                     for _, rows, _ in datasets]))


def population(rows):
    successes = [r for r in rows if r["success"]]
    return {"n_success": len(successes),
            "statistics_deg": {k: distribution(r[k] for r in successes)
                               for k in METRICS + ["effective_margin_deg", "raw_margin_deg"]},
            "effective_margin_below_1deg": {
                "count": sum(r["effective_margin_deg"] < 1 for r in successes),
                "denominator": len(successes),
                "indices": [r["index"] for r in successes if r["effective_margin_deg"] < 1],
            }}


def render(datasets, keys, out, joint_label, language, margin_max):
    L = language
    labels = {
        "grasp_deg": L("grasp 到位角度", "grasp endpoint angle"),
        "place_deg": L("place 到位角度", "place endpoint angle"),
        "cycle_min_deg": L("完整六阶段最小角度", "full-cycle minimum angle"),
        "cycle_max_deg": L("完整六阶段最大角度", "full-cycle maximum angle"),
        "effective_margin_deg": L("完整六阶段最小有效限位余量", "full-cycle minimum effective margin"),
    }
    fig, axes = plt.subplots(len(keys), 3, figsize=(34, 8.6 * len(keys)), squeeze=False, constrained_layout=True)
    for row_index, key in enumerate(keys):
        is_margin = key == "effective_margin_deg"
        norm = plt.Normalize(0, margin_max) if is_margin else plt.Normalize(-180, 180)
        cmap = plt.get_cmap("viridis" if is_margin else "coolwarm")
        for col, (meta, rows, report) in enumerate(datasets):
            factor = report["factors"]
            start = "−30°" if factor["search_order"] == "asc" else "+30°"
            extra = factor["place_tool_z_rotation_deg"]
            label = chr(ord("A") + col)
            title = L(f"{label}：place 额外 {extra:g}°，{start} 起搜 | {report['n_success']}/{report['n_total']} 成功",
                      f"{label}: place extra {extra:g} deg, start {start} | {report['n_success']}/{report['n_total']} success")
            draw_panel(axes[row_index, col], rows, meta, key, title + "\n" + labels[key], joint_label, norm, cmap, L)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[row_index]),
                          shrink=.85, fraction=.02, pad=.01)
        cb.set_label(L("有效限位余量 (°)", "effective limit margin (deg)") if is_margin else f"{joint_label} (°)")
    heading = L(f"{joint_label} 三组同位置对比：左 A / 中 B / 右 C；各行使用统一色标；刻度间隔 2 cm",
                f"{joint_label} three-way maps: A / B / C, shared color scale within each row; ticks 2 cm")
    if keys == ["effective_margin_deg"]:
        heading += L("\n0°代表内缩后的边界，距原始限位仍有0.14 rad（8.02°）",
                     "\n0 deg means the clipped boundary; the original limit is still 0.14 rad (8.02 deg) away")
    else:
        heading += L("\n颜色是实际关节角，不是 grasp 搜索角；灰底红叉为失败点，无保存角度",
                     "\nColors are actual joint angles, not tool search angles; gray/red crosses have no saved joint angle")
    fig.suptitle(heading, fontsize=17)
    fig.text(.5, -.005, L("按原始 grasp XY 排列（place 图同样如此）。全程含进入本 case 的运动；同一目标不代表同一入场构型。",
                        "All maps use originating grasp XY, including place. Cycle includes entry; same target does not imply same incoming joint state."),
             ha="center", fontsize=10)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--joint", default="J_6")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = read_json(root / "summary.json")
    datasets = [load_rows(Path(g["result"]).resolve(), args.joint) for g in summary["groups"]]
    common = validate_alignment(datasets)
    require(len(common) == summary["overlap"]["all_three_success_count"], "Common-success count differs from summary")
    prefix = args.joint.replace("_", "").lower()
    out = args.out_dir or root / "joint_angle_comparison"
    keys = METRICS + ["effective_margin_deg"]
    suffixes = ["grasp_threeway", "place_threeway", "cycle_min_threeway", "cycle_max_threeway", "min_limit_margin_threeway"]
    outputs = [out / f"{prefix}_{s}.png" for s in suffixes]
    outputs += [out / f"{prefix}_overview_threeway.png", out / f"{prefix}_threeway.csv", out / f"{prefix}_threeway.json"]
    require(not any(p.exists() for p in outputs), "Refusing to overwrite existing comparison outputs")
    out.mkdir(parents=True, exist_ok=True)
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if font else (lambda zh, en: en)
    margin_max = float(np.ceil(max(r["effective_margin_deg"] for _, rows, _ in datasets
                                   for r in rows if r["success"]) / 10) * 10)
    margin_max = max(10., margin_max)
    for key, path in zip(keys, outputs[:5]):
        render(datasets, [key], path, prefix.upper(), L, margin_max)
    render(datasets, METRICS, outputs[5], prefix.upper(), L, margin_max)
    metrics = keys + ["raw_margin_deg", "cycle_span_deg", "grasp_sample_index", "place_sample_index"]
    fields = ["index", "case_number", "row", "col", "x_m", "y_m", "z_m", "all_three_success"]
    names = [Path(report["result"]).name for _, _, report in datasets]
    fields += [f"{name}.{key}" for name in names for key in ["success"] + metrics]
    aligned = [{r["index"]: r for r in rows} for _, rows, _ in datasets]
    with outputs[6].open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in sorted(aligned[0]):
            row = {key: aligned[0][index][key] for key in fields[:7]}
            row["all_three_success"] = index in common
            for name, by_id in zip(names, aligned):
                for key in ["success"] + metrics:
                    row[f"{name}.{key}"] = by_id[index].get(key)
            writer.writerow(row)
    report = {
        "joint": args.joint, "group_order": names, "n_targets": len(aligned[0]),
        "all_three_success_count": len(common), "all_three_success_indices": common,
        "angle_color_limits_deg": [-180, 180], "margin_color_limits_deg": [0, margin_max],
        "definitions": datasets[0][2]["definitions"],
        "warning": "Same target does not imply same incoming joint state. Failed cases have no angle. Per-run standalone margin maps autoscale; these three-way panels share a single margin scale.",
        "source_summary_sha256": sha256(root / "summary.json"), "script_sha256": sha256(Path(__file__)),
        "groups": [{"name": name, "factors": data[2]["factors"], "provenance": data[2]["provenance"],
                    "own_success_population": population(data[1]),
                    "common_success_population": population([r for r in data[1] if r["index"] in common])}
                   for name, data in zip(names, datasets)],
    }
    with outputs[7].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"[ok] {outputs[6]}\n[ok] {outputs[7]}")
    print(json.dumps({"common_success": len(common), "margin_color_limits_deg": [0, margin_max]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
