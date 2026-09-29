#!/usr/bin/env python3
"""Plot actual joint angles from an audited original-base three-way result.

Reads saved trajectories only; no IK, planning, ROS or model changes. The parent
comparison summary supplies recorded input hashes and raw/effective limits.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

from compare_original_base_threeway import case_metrics, check_hash, distribution, read_json, require, sha256
from plot_grasp_angle_map import pick_cjk_font


METRICS = ["grasp_deg", "place_deg", "cycle_min_deg", "cycle_max_deg"]


def extract_rows(cases, q, joint_index, raw_limits, effective_limits):
    """Use the LAST saved sample of grasp/place, not their incoming endpoint."""
    rows = []
    for case in cases:
        row = {"index": case["index"], "case_number": case["index"] + 1,
               "row": case["row"], "col": case["col"],
               "x_m": case["position_raw"][0], "y_m": case["position_raw"][1],
               "z_m": case["position_raw"][2], "success": case["success"]}
        if case["success"]:
            lo, hi = case["sample_range_half_open"]
            samples = q[lo:hi, joint_index]
            row.update(sample_start=lo, sample_end_exclusive=hi,
                       grasp_search_angle_deg=case["angle_grasp_deg"])
            for phase in case["phases"]:
                if phase["phase"] in ("grasp", "place"):
                    end = phase["sample_range_half_open"][1] - 1
                    row[f"{phase['phase']}_sample_index"] = end
                    row[f"{phase['phase']}_deg"] = float(np.degrees(q[end, joint_index]))
            margins = np.minimum(samples - effective_limits[0, joint_index],
                                 effective_limits[1, joint_index] - samples)
            worst = lo + int(np.argmin(margins))
            phase_names = [p["phase"] for p in case["phases"]
                           if p["sample_range_half_open"][0] <= worst < p["sample_range_half_open"][1]]
            row.update(
                cycle_min_deg=float(np.degrees(samples.min())),
                cycle_max_deg=float(np.degrees(samples.max())),
                cycle_span_deg=float(np.degrees(np.ptp(samples))),
                raw_margin_deg=float(np.degrees(np.minimum(
                    samples - raw_limits[0, joint_index], raw_limits[1, joint_index] - samples).min())),
                effective_margin_deg=float(np.degrees(margins.min())),
                nearest_limit_sample_index=worst,
                nearest_limit_angle_deg=float(np.degrees(q[worst, joint_index])),
                nearest_limit_phases="/".join(phase_names),
            )
            require(np.isclose(row["effective_margin_deg"],
                               case["effective_margin_per_joint_deg"][joint_index], atol=1e-9),
                    "Joint margin differs from reconstructed case metrics")
        rows.append(row)
    return rows


def load_rows(result, joint):
    root = result.parent.parent
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    groups = [g for g in summary["groups"] if Path(g["result"]).resolve() == result]
    require(len(groups) == 1, "Result must be one run in the audited three-way summary")
    group = groups[0]
    for name in ("trajectory_meta.json", "trajectory.npz"):
        check_hash(sha256(result / name), group["provenance"]["result_file_sha256"][name], name)
    meta = read_json(result / "trajectory_meta.json")
    with np.load(result / "trajectory.npz", allow_pickle=False) as archive:
        q, times = np.array(archive["positions"]), np.array(archive["times"])
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in archive["joint_names"]]
    require(names == group["joint_names"] == meta["robot"]["joint_names"], "Joint order differs")
    require(joint in names, f"Unknown joint: {joint}")
    raw = np.asarray(group["raw_limits_rad"])
    effective = np.asarray(group["effective_limits_rad"])
    cases = case_metrics(meta, q, times, raw, effective)
    require(len(cases) == group["n_total"] and sum(c["success"] for c in cases) == group["n_success"],
            "Case counts differ from audited summary")
    rows = extract_rows(cases, q, names.index(joint), raw, effective)
    success = [r for r in rows if r["success"]]
    require(bool(success), "No saved successful trajectory to plot")
    report = {
        "joint": joint, "result": str(result), "factors": group["factors"],
        "n_total": len(rows), "n_success": len(success), "n_failed": len(rows) - len(success),
        "raw_limits_deg": np.degrees(raw[:, names.index(joint)]).tolist(),
        "effective_limits_deg": np.degrees(effective[:, names.index(joint)]).tolist(),
        "definitions": {
            "grasp_deg": "Joint angle at last saved sample of the grasp descent phase",
            "place_deg": "Joint angle at last saved sample of place descent, plotted at its originating grasp XY",
            "cycle_min_max": "Minimum/maximum over all six saved phases, including entry from previous successful case or Home",
            "failed": "Failed planning case: no saved joint angle; blank in CSV and gray/red cross in maps, never zero",
            "coordinates": "Original LINK_0 grasp target XY, same sampling points; axis ticks every 0.02 m are not grid spacing",
            "angle": "Actual bounded revolute joint position in degrees, no modulo/wrapping; NOT the grasp tool-angle search parameter",
            "effective_margin": "Minimum distance to either clipped bound over saved full-case samples; raw margin uses original URDF bounds",
            "scope": "Saved discrete trajectories only, no rerun or additional continuous-motion safety claim",
        },
        "provenance": {"summary": str(summary_path), "summary_sha256": sha256(summary_path),
                       "source": group["provenance"], "script_sha256": sha256(Path(__file__))},
        "statistics_deg": {key: distribution(r[key] for r in success)
                           for key in METRICS + ["cycle_span_deg", "effective_margin_deg", "raw_margin_deg"]},
        "nearest_limit_case": min(success, key=lambda r: r["effective_margin_deg"]),
        "cases": rows,
    }
    return meta, rows, report


def draw_panel(ax, rows, meta, key, title, joint_label, norm, cmap, language):
    L = language
    grid = meta["grid"]
    dx = abs(np.diff(grid["x_range"])[0]) / max(1, int(grid["rows"]) - 1)
    dy = abs(np.diff(grid["y_range"])[0]) / max(1, int(grid["cols"]) - 1)
    dx, dy = max(dx, .005) * .9, max(dy, .005) * .9
    for row in rows:
        x, y = row["x_m"], row["y_m"]
        ok = row["success"]
        color = cmap(norm(row[key])) if ok else (.88, .88, .88, 1.)
        ax.add_patch(Rectangle((x - dx / 2, y - dy / 2), dx, dy,
                              facecolor=color, edgecolor="#444444" if ok else "#c62828",
                              linewidth=.45 if ok else .85, zorder=2))
        luminance = .299 * color[0] + .587 * color[1] + .114 * color[2]
        label = f"{row[key]:+.1f}" if ok and key in METRICS else f"{row[key]:.1f}" if ok else "×"
        ax.text(x, y, label, ha="center", va="center", fontsize=6.1, zorder=3,
                color=("white" if luminance < .52 else "black") if ok else "#b2182b")
    place = meta["place_position_raw"]
    ax.scatter([0], [0], marker="*", s=140, color="gold", edgecolors="black", zorder=4,
               label=L("原始 LINK_0 基座", "original LINK_0 base"))
    ax.scatter([place[0]], [place[1]], marker="X", s=100, color="#d81b60", edgecolors="black", zorder=4,
               label=L("固定 place 位置", "fixed place position"))
    values = [r[key] for r in rows if r["success"]]
    ax.set_title(title + "\n" + L(f"{joint_label} 范围 {min(values):.2f}° ~ {max(values):.2f}°",
                                 f"{joint_label} range {min(values):.2f} to {max(values):.2f} deg"), fontsize=12)
    ax.set_xlabel("LINK_0 x (m)")
    ax.set_ylabel("LINK_0 y (m)")
    ax.set_xlim(min(r["x_m"] for r in rows) - .035, .045)
    ax.set_ylim(min(min(r["y_m"] for r in rows), place[1]) - .04,
                max(r["y_m"] for r in rows) + .045)
    ax.set_aspect("equal")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(.02))
        axis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, linestyle=":", linewidth=.5, alpha=.55)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [Patch(facecolor=".88", edgecolor="#c62828")],
              labels + [L("规划失败：无保存角度", "failed: no saved angle")],
              loc="upper right", fontsize=7)


def write_outputs(out, meta, rows, report):
    joint_label = report["joint"].replace("_", "")
    prefix = joint_label.lower()
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if font else (lambda zh, en: en)
    titles = {
        "grasp_deg": L("抓取到位时的实际关节角", "Actual joint angle at grasp endpoint"),
        "place_deg": L("对应放置到位时的实际关节角", "Actual joint angle at corresponding place endpoint"),
        "cycle_min_deg": L("完整六阶段：最小关节角", "Full six-phase cycle: minimum joint angle"),
        "cycle_max_deg": L("完整六阶段：最大关节角", "Full six-phase cycle: maximum joint angle"),
        "effective_margin_deg": L("完整六阶段：最小有效限位余量", "Full six-phase cycle: minimum effective margin"),
    }
    suffixes = ["grasp_angle_map", "place_angle_map", "cycle_min_angle_map", "cycle_max_angle_map", "min_limit_margin_map"]
    outputs = [out / f"{prefix}_{suffix}.png" for suffix in suffixes]
    outputs += [out / f"{prefix}_angle_comparison.png", out / f"{prefix}_angles.csv", out / f"{prefix}_angles.json"]
    require(not any(p.exists() for p in outputs), "Refusing to overwrite existing map/data outputs")
    out.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(-180, 180)
    extra = report["factors"]["place_tool_z_rotation_deg"]
    order = "−30° → +30°" if report["factors"]["search_order"] == "asc" else "+30° → −30°"
    heading = L(f"{joint_label} 实际关节角分布 | place 额外旋转 {extra:g}° | grasp 搜索 {order}",
                f"{joint_label} actual joint-angle maps | place extra {extra:g} deg | grasp {order}")
    subtitle = L(f"{report['n_success']}/{report['n_total']} 成功；色标为关节角，不是 grasp 搜索角；刻度间隔 2 cm",
                 f"{report['n_success']}/{report['n_total']} successful; colors are joint angles, not grasp search angles; ticks 2 cm")
    note = L("每格对应同一个原始 grasp 位置；place 图也按来源 grasp 位置排列。全程极值包含进入本 case 的运动。",
             "Each cell is an original grasp target (also for place). Cycle extrema include entry into that case.")
    for key, path in zip(METRICS + ["effective_margin_deg"], outputs[:5]):
        fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
        margin = key == "effective_margin_deg"
        local_norm = plt.Normalize(0, max(r[key] for r in rows if r["success"])) if margin else norm
        local_cmap = plt.get_cmap("viridis") if margin else cmap
        draw_panel(ax, rows, meta, key, titles[key], joint_label, local_norm, local_cmap, L)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=local_norm, cmap=local_cmap), ax=ax, shrink=.85, pad=.02)
        cb.set_label(L("有效余量 (°)，相对双侧内缩 0.14 rad 后的边界", "Effective margin (deg), bounds clipped by 0.14 rad")
                     if margin else f"{joint_label} (°)")
        fig.suptitle(heading + "\n" + subtitle, fontsize=12)
        fig.text(.5, -.005, note, ha="center", fontsize=8)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(24, 18), constrained_layout=True)
    for ax, key in zip(axes.flat, METRICS):
        draw_panel(ax, rows, meta, key, titles[key], joint_label, norm, cmap, L)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes.flat),
                 shrink=.88, fraction=.025, pad=.015).set_label(f"{joint_label} (°)")
    fig.suptitle(heading + "\n" + subtitle, fontsize=17)
    fig.text(.5, -.005, note, ha="center", fontsize=11)
    fig.savefig(outputs[5], dpi=160, bbox_inches="tight")
    plt.close(fig)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with outputs[6].open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with outputs[7].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    for path in outputs:
        print(f"[ok] {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--joint", default="J_6")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    result = args.result_dir.resolve()
    meta, rows, report = load_rows(result, args.joint)
    write_outputs(args.out_dir or result / "joint_angle_maps", meta, rows, report)
    print(json.dumps(report["statistics_deg"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
