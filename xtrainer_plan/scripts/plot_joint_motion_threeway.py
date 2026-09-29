#!/usr/bin/env python3
"""Compare per-joint worst complete-case spans in the audited V60 three-way data.

Main metric: max over successful cases of (max(q_j)-min(q_j)) over all six
phases. Also export the different whole-run envelope metric and a spatial map
of each case's largest span among J1..J6. No angular wrapping, GPU, or replanning.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_original_base_threeway import JOINTS, case_metrics, check_hash, read_json, require, sha256
from plot_grasp_angle_map import pick_cjk_font
from plot_original_base_joint_maps import draw_panel


def worst_cases(cases, q, indices=None):
    successes = [c for c in cases if c["success"] and (indices is None or c["index"] in indices)]
    require(bool(successes), "No successful cases in selected population")
    records = []
    for j, joint in enumerate(JOINTS):
        case = max(successes, key=lambda c: c["full_case_ptp_per_joint_deg"][j])
        lo, hi = case["sample_range_half_open"]
        samples = q[lo:hi, j]
        minimum, maximum = lo + int(np.argmin(samples)), lo + int(np.argmax(samples))
        records.append({
            "joint": joint, "max_span_deg": float(np.degrees(np.ptp(samples))),
            "case_index": case["index"], "case_number": case["index"] + 1,
            "x_m": case["position_raw"][0], "y_m": case["position_raw"][1], "z_m": case["position_raw"][2],
            "minimum_deg": float(np.degrees(q[minimum, j])), "maximum_deg": float(np.degrees(q[maximum, j])),
            "min_sample_index": minimum, "max_sample_index": maximum,
            "min_phases": [p["phase"] for p in case["phases"] if p["sample_range_half_open"][0] <= minimum < p["sample_range_half_open"][1]],
            "max_phases": [p["phase"] for p in case["phases"] if p["sample_range_half_open"][0] <= maximum < p["sample_range_half_open"][1]],
            "n_success_in_population": len(successes),
        })
    return records


def map_rows(cases):
    rows = []
    for case in cases:
        row = {"index": case["index"], "row": case["row"], "col": case["col"],
               "x_m": case["position_raw"][0], "y_m": case["position_raw"][1],
               "z_m": case["position_raw"][2], "success": case["success"]}
        if case["success"]:
            values = case["full_case_ptp_per_joint_deg"]
            j = int(np.argmax(values))
            row.update(max_joint_span_deg=float(values[j]), max_joint=JOINTS[j],
                       spans_deg=values, case_number=case["index"] + 1)
        rows.append(row)
    return rows


def load_data(root):
    summary = read_json(root / "summary.json")
    common = set(summary["overlap"]["all_three_success_indices"])
    datasets = []
    for group in summary["groups"]:
        result = Path(group["result"])
        for file in ("trajectory.npz", "trajectory_meta.json"):
            check_hash(sha256(result / file), group["provenance"]["result_file_sha256"][file], file)
        meta = read_json(result / "trajectory_meta.json")
        with np.load(result / "trajectory.npz", allow_pickle=False) as archive:
            q, times = np.asarray(archive["positions"]), np.asarray(archive["times"])
            names = [v.decode() if isinstance(v, bytes) else str(v) for v in archive["joint_names"]]
        require(names == JOINTS == meta["robot"]["joint_names"], "Joint order differs")
        cases = case_metrics(meta, q, times, np.asarray(group["raw_limits_rad"]), np.asarray(group["effective_limits_rad"]))
        require(len(cases) == group["n_total"] and sum(c["success"] for c in cases) == group["n_success"], "Case counts differ")
        old = {c["index"]: c["runs"][group["name"]] for c in summary["cases"]}
        require(len({c["index"] for c in cases}) == len(cases) == len(old), "Duplicate/missing case index")
        for case in cases:
            reference = old[case["index"]]
            require(case["success"] == reference["success"] and case["position_raw"] == reference["position_raw"], "Case target/status differs")
            if case["success"]:
                require(np.allclose(case["full_case_ptp_per_joint_deg"], reference["full_case_ptp_per_joint_deg"], atol=1e-9, rtol=0), "Case span differs from audited summary")
        record = {"name": group["name"], "factors": group["factors"], "n_success": group["n_success"],
                  "own_success_worst_cases": worst_cases(cases, q),
                  "common_success_worst_cases": worst_cases(cases, q, common),
                  "whole_run_min_deg": np.degrees(q.min(axis=0)).tolist(),
                  "whole_run_max_deg": np.degrees(q.max(axis=0)).tolist(),
                  "whole_run_span_deg": np.degrees(np.ptp(q, axis=0)).tolist(),
                  "provenance": group["provenance"], "cases": map_rows(cases)}
        datasets.append((meta, record))
    require(len(datasets) == 3, "Expected three results")
    return summary, datasets


def bar_plot(datasets, key, title, note, path, language):
    L = language
    fig, ax = plt.subplots(figsize=(13, 7.4))
    colors = ["#2878b5", "#e68432", "#369c6c"]
    x = np.arange(6)
    for offset, (_, group) in enumerate(datasets):
        factor = group["factors"]
        start = "−30°" if factor["search_order"] == "asc" else "+30°"
        label = f"{chr(65 + offset)}: place +{factor['place_tool_z_rotation_deg']:g}°, grasp {start}"
        values = group[key] if key == "whole_run_span_deg" else [r["max_span_deg"] for r in group[key]]
        bars = ax.bar(x + (offset - 1) * .26, values, width=.25, color=colors[offset], label=label)
        ax.bar_label(bars, labels=[f"{v:.2f}°" for v in values], fontsize=8, padding=4)
    ax.set_xticks(x, [j.replace("_", "") for j in JOINTS], fontsize=12)
    ax.set_ylabel(L("角度跨度：max(q) − min(q) (°)", "Angular span: max(q) - min(q) (deg)"))
    ax.set_ylim(0, 310)
    ax.grid(axis="y", linestyle=":", alpha=.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=15, pad=16)
    fig.text(.5, .02, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[ok] {path}")


def spatial_plot(datasets, path, language):
    L = language
    fig, axes = plt.subplots(1, 3, figsize=(34, 9.2), constrained_layout=True)
    norm, cmap = plt.Normalize(0, 250), plt.get_cmap("YlOrRd")
    for col, (meta, group) in enumerate(datasets):
        factor = group["factors"]
        start = "−30°" if factor["search_order"] == "asc" else "+30°"
        title = L(f"{chr(65 + col)}：place额外{factor['place_tool_z_rotation_deg']:g}°，{start}起搜 | {group['n_success']}/400成功",
                  f"{chr(65 + col)}: place extra {factor['place_tool_z_rotation_deg']:g} deg, start {start} | {group['n_success']}/400 success")
        rows = group["cases"]
        draw_panel(axes[col], rows, meta, "max_joint_span_deg", title,
                   L("六关节中的最大跨度", "largest joint span"), norm, cmap, L)
        # draw_panel creates exactly one data text per input row; replace those
        # labels in the same order, retaining its shared geometry and styling.
        require(len(axes[col].texts) == len(rows), "Unexpected panel text layout")
        for artist, row in zip(axes[col].texts, rows):
            if row["success"]:
                artist.set_text(f"{row['max_joint'].replace('_', '')}\n{row['max_joint_span_deg']:.1f}°")
                artist.set_fontsize(5.8)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), shrink=.85, fraction=.02, pad=.01).set_label(
        L("最大关节跨度 (°)", "largest joint span (deg)"))
    fig.suptitle(L("每个位置完整六阶段：六个关节中变化最大的关节及其跨度\n左 A / 中 B / 右 C；统一色标；2 cm刻度；失败点不填0",
                   "Each full six-phase case: joint with largest span and its value\nA / B / C; shared color scale; ticks 2 cm; failures are not zero"), fontsize=17)
    fig.text(.5, -.005, L("含上一case末态或Home进入本case的运动；不是相邻采样瞬时跳变，也不是累计往返行程。",
                         "Includes entry from prior case or Home; not adjacent-sample jumps or cumulative angular travel."), ha="center", fontsize=10)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    summary, data = load_data(root)
    out = root / "joint_motion_comparison"
    names = ["joint_max_cycle_span_threeway.png", "joint_max_cycle_span_common_threeway.png",
             "joint_whole_run_span_threeway.png", "max_joint_span_map_threeway.png",
             "joint_maxima.csv", "joint_spans_by_case.csv", "joint_motion_summary.json"]
    require(not any((out / name).exists() for name in names), "Refusing to overwrite outputs")
    out.mkdir(parents=True, exist_ok=True)
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if font else (lambda zh, en: en)
    note = L("每关节分别取各成功case六阶段跨度的最大值；含入场段/Home。不是瞬时跳变、累计行程或单段170°判据。",
             "Per joint: largest six-phase case span, including entry/Home. Not instantaneous jump, cumulative travel or single-phase criterion.")
    bar_plot(data, "own_success_worst_cases", L("各关节完整抓放变化量的最大值（三组自身成功case）", "Worst full-cycle span by joint (each run's successful cases)"), note, out / names[0], L)
    bar_plot(data, "common_success_worst_cases", L(f"各关节最大完整抓放跨度（共同成功{summary['overlap']['all_three_success_count']}点）", "Worst full-cycle span by joint (common successful targets)"),
             note + L("\n同一目标不保证入场关节构型相同。", "\nSame target does not imply same incoming state."), out / names[1], L)
    bar_plot(data, "whole_run_span_deg", L("另一口径：整条全量轨迹的关节角范围", "Different metric: whole-run joint-angle envelope"),
             L("全run的max(q)−min(q)，两端可以来自不同case；不能当作一次抓放的变化量。", "Whole-run max(q)-min(q); extrema can belong to different cases, not one pick/place motion."), out / names[2], L)
    spatial_plot(data, out / names[3], L)
    rows = []
    for _, group in data:
        for population in ("own_success_worst_cases", "common_success_worst_cases"):
            for row in group[population]:
                rows.append({"group": group["name"], "population": population,
                             **{k: v for k, v in row.items() if not isinstance(v, list)}})
    with (out / names[4]).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_id = [{r["index"]: r for r in g["cases"]} for _, g in data]
    fields = ["index", "case_number", "x_m", "y_m", "z_m"]
    fields += [f"{g['name']}.{k}" for _, g in data for k in ["success", "max_joint", "max_joint_span_deg"] + JOINTS]
    with (out / names[5]).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in sorted(by_id[0]):
            r = by_id[0][index]
            row = {"index": index, "case_number": index + 1, **{k: r[k] for k in ("x_m", "y_m", "z_m")}}
            for (_, group), mapping in zip(data, by_id):
                case = mapping[index]
                for key in ("success", "max_joint", "max_joint_span_deg"):
                    row[f"{group['name']}.{key}"] = case.get(key)
                if case["success"]:
                    row.update({f"{group['name']}.{j}": v for j, v in zip(JOINTS, case["spans_deg"])})
            writer.writerow(row)
    report = {"metric": "max_successful_case(max_six_phase_q_j - min_six_phase_q_j), degrees, bounded joints without wrapping",
              "scope": "Includes entry from previous successful case/Home; saved samples only. Not cumulative travel or adjacent jumps. Common targets may have different entry states.",
              "source_summary_sha256": sha256(root / "summary.json"), "script_sha256": sha256(Path(__file__)),
              "common_success_count": summary["overlap"]["all_three_success_count"], "groups": [g for _, g in data]}
    with (out / names[6]).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    for name in names[4:]:
        print(f"[ok] {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
