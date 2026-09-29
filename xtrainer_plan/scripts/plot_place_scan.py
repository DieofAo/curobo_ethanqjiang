#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 scan_place_region.py 的三阶段结果画成图。

出图清单:
  stage12_grasp_ik.png        阶段1: 每个 grasp 位置 IK 是否可行(绿/红)
  stage12_place_ik.png        阶段2: 每个 place 位置 IK 是否可行(绿/红)
  place_rank_map.png          阶段3: place 区域热力图, 每格 = 规划成功的 grasp 数
  stage3_top<k>_place<i>.png  阶段3: Top-N 各自的 grasp 可达图(绿=能到, 红=不能到)
  stage3_summary.png          阶段3: Top-N 成功数对比 + 失败原因分解

用法:
    python3 plot_place_scan.py results_place_scan/<时间戳>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle


def pick_cjk_font() -> Optional[str]:
    prefer = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
              "WenQuanYi Micro Hei", "Source Han Sans CN", "SimHei",
              "Microsoft YaHei", "AR PL UMing CN"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in prefer:
        if n in have:
            return n
    return None


CJK = pick_cjk_font()
if CJK:
    plt.rcParams["font.family"] = CJK
plt.rcParams["axes.unicode_minus"] = False
L = (lambda zh, en: zh) if CJK else (lambda zh, en: en)


def ok_fail_map(pts: List[Dict[str, Any]], ok_idx, step: float, out: Path,
                title: str, sub: str, xlabel: str, ylabel: str,
                marks: Optional[List[Dict[str, Any]]] = None) -> None:
    """通用「可行/不可行」方格图。ok_idx = 可行点的 index 集合。"""
    oks = set(int(i) for i in ok_idx)
    fig, ax = plt.subplots(figsize=(10.2, 8.2))
    for p in pts:
        x, y = p["position"][0], p["position"][1]
        good = p["index"] in oks
        ax.add_patch(Rectangle((x - step / 2, y - step / 2), step, step,
                               facecolor="#2ca02c" if good else "#d62728",
                               alpha=0.85, edgecolor="0.7", linewidth=0.3,
                               zorder=2))
    n_ok = len([p for p in pts if p["index"] in oks])
    handles = [
        Patch(facecolor="#2ca02c", label=L(f"可行 ({n_ok})", f"ok ({n_ok})")),
        Patch(facecolor="#d62728",
              label=L(f"不可行 ({len(pts) - n_ok})", f"failed ({len(pts) - n_ok})")),
    ]
    ax.scatter([0], [0], s=260, marker="*", color="gold", edgecolors="black",
               linewidths=1.2, zorder=6, label=L("LINK_0 原点", "LINK_0 origin"))
    for m in (marks or []):
        ax.scatter([m["pos"][0]], [m["pos"][1]], s=210, marker="X",
                   color="magenta", edgecolors="black", linewidths=1.2, zorder=6,
                   label=m["label"])
    h2, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + h2, loc="upper left", fontsize=9, framealpha=0.92)
    ax.set_title(f"{title}\n{sub}", fontsize=12.5, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.08)
    ax.grid(True, ls=":", alpha=0.35, zorder=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


def place_heatmap(s3: Dict[str, Any], s12: Dict[str, Any], out: Path) -> None:
    """place 区域热力图: 颜色 = 阶段3 规划成功的 grasp 数。"""
    res = s3["results"]
    if not res:
        print("[skip] stage3 无结果")
        return
    x = np.array([r["position"][0] for r in res])
    y = np.array([r["position"][1] for r in res])
    v = np.array([float(r["n_success"]) for r in res])
    top = list(s3.get("top_place_indices") or [])
    step = float(s12["place_grid"].get("step", 0.01))

    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    vmin, vmax = float(v.min()), float(v.max())
    norm = plt.Normalize(vmin, vmax if vmax > vmin else vmin + 1.0)
    cmap = plt.get_cmap("viridis")
    for xi, yi, vi in zip(x, y, v):
        ax.add_patch(Rectangle((xi - step / 2, yi - step / 2), step, step,
                               facecolor=cmap(norm(vi)), edgecolor="0.75",
                               linewidth=0.4, zorder=2))
    # 阶段2 就被跳过的 place(不在 stage3 结果里) 用灰色标出
    r_idx = {r["place_index"] for r in res}
    for p in s12["places"]:
        if p["index"] in r_idx:
            continue
        ax.add_patch(Rectangle((p["position"][0] - step / 2,
                                p["position"][1] - step / 2), step, step,
                               facecolor="0.82", edgecolor="0.7",
                               linewidth=0.4, hatch="///", zorder=2))
    for k, pidx in enumerate(top):
        r = next((q for q in res if q["place_index"] == pidx), None)
        if r is None:
            continue
        px, py = r["position"][0], r["position"][1]
        ax.add_patch(Rectangle((px - step / 2, py - step / 2), step, step,
                               facecolor="none", edgecolor="red",
                               linewidth=1.8, zorder=4))
        ax.text(px, py, str(k + 1), ha="center", va="center", zorder=5,
                fontsize=8, fontweight="bold", color="red")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label(L("阶段3 规划成功的 grasp 点数", "stage3 planned-ok grasp count"))

    ng = s3["grasp_grid"]["n_points"]
    n_skip = len(s12["places"]) - len(res)
    sub = L(f"place 网格 {s12['place_grid']['nx']}x{s12['place_grid']['ny']} "
            f"step={step * 100:g}cm; 阶段3 每个 place 对 {ng} 个 grasp 点规划; "
            f"灰色斜纹={n_skip} 个在阶段2 被跳过; 红框=Top{len(top)}",
            f"place grid, {ng} grasp points each; grey=skipped at stage2; "
            f"red=Top{len(top)}")
    ax.set_title(L("place 候选位置的可用性排名（阶段3 轨迹规划结果）",
                   "place ranking (stage3 trajopt)") + f"\n{sub}",
                 fontsize=12.5, pad=12)
    ax.set_xlabel(L("place 位置 x (m)  [LINK_0 系]", "place x (m)"))
    ax.set_ylabel(L("place 位置 y (m)  [LINK_0 系]", "place y (m)"))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.06)
    ax.grid(True, ls=":", alpha=0.35, zorder=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


def stage3_summary(s3: Dict[str, Any], out: Path) -> None:
    res = [r for r in s3["results"] if r["all_stages_ok"]]
    res = sorted(res, key=lambda r: -r["n_success"])[: max(int(s3.get("top_n", 10)), 1)]
    if not res:
        print("[skip] stage3 没有三阶段全成功的 place")
        return
    lbl = [f"#{r['place_index']}\n{r['position'][0]:.2f},{r['position'][1]:.2f}"
           for r in res]
    xs = np.arange(len(res))
    tot = res[0]["n_grasp_total"]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(max(9.0, len(res) * 1.15), 8.8))
    ax.bar(xs, [r["n_success"] for r in res], color="#2ca02c", alpha=0.85,
           label=L(f"规划成功 (总 {tot} 个 grasp 点)",
                   f"planned ok (of {tot})"))
    ax.axhline(tot, ls="--", lw=1.0, color="0.5",
               label=L("grasp 点总数", "total grasp points"))
    for i, r in enumerate(res):
        ax.text(i, r["n_success"], str(r["n_success"]), ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(lbl, fontsize=8)
    ax.set_ylabel(L("成功 grasp 点数", "success count"))
    ax.set_title(L("阶段3 Top-N place 排名（按成功 grasp 点数）",
                   "Stage3 Top-N ranking"), fontsize=12.5)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)

    # 失败原因分解: 阶段1/2 就无 IK / 规划失败 / 关节变化超限
    b1 = np.array([r["n_skip_no_ik"] for r in res], dtype=float)
    b2 = np.array([r["n_fail_plan"] for r in res], dtype=float)
    b3 = np.array([r["n_fail_joint_delta"] for r in res], dtype=float)
    ax2.bar(xs, b1, color="0.6",
            label=L("跳过: 阶段1/2 无可行 IK", "skipped: no IK"))
    ax2.bar(xs, b2, bottom=b1, color="#ff7f0e",
            label=L("规划失败", "plan failed"))
    ax2.bar(xs, b3, bottom=b1 + b2, color="#9467bd",
            label=L("关节变化超限", "joint delta over limit"))
    ax2.set_xticks(xs)
    ax2.set_xticklabels(lbl, fontsize=8)
    ax2.set_ylabel(L("失败/跳过次数", "fail count"))
    ax2.set_title(L("失败原因分解（注: 规划失败次数按角度尝试计, 可能多于点数）",
                    "failure breakdown"), fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", ls=":", alpha=0.4)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir")
    ap.add_argument("--top-maps", type=int, default=None,
                    help="给前几名各画一张 grasp 可达图, 默认全部 Top-N")
    args = ap.parse_args()

    rd = Path(args.result_dir)
    p12, p3 = rd / "stage12_result.json", rd / "stage3_result.json"

    s12: Optional[Dict[str, Any]] = None
    if p12.is_file():
        s12 = json.loads(p12.read_text(encoding="utf-8"))
        gstep = float(s12["grasp_grid"].get("step", 0.01))
        pstep = float(s12["place_grid"].get("step", 0.01))
        g_ok = [int(k) for k in s12["grasp_feasible"]]
        p_ok = [int(k) for k in s12["place_feasible"]]
        st1, st2 = s12["stage1_stats"], s12["stage2_stats"]
        ok_fail_map(
            s12["grasps"], g_ok, gstep, rd / "stage12_grasp_ik.png",
            L("阶段1: grasp 位置的 IK 可行性", "Stage1: grasp IK feasibility"),
            L(f"{st1['n_feasible_points']}/{st1['n_points']} 个位置可行, "
              f"每个位置试了 {st1['n_angles']} 个角度; "
              f"IK 调用 {st1['n_ik_calls']} 次, 用时 {st1['elapsed_s']:.0f}s"
              + ("  [只扫外围一圈]" if s12["grasp_grid"].get("perimeter_only") else ""),
              f"{st1['n_feasible_points']}/{st1['n_points']} feasible"),
            L("grasp 位置 x (m)  [LINK_0 系]", "grasp x (m)"),
            L("grasp 位置 y (m)  [LINK_0 系]", "grasp y (m)"))
        ok_fail_map(
            s12["places"], p_ok, pstep, rd / "stage12_place_ik.png",
            L("阶段2: place 位置的 IK 可行性", "Stage2: place IK feasibility"),
            L(f"{st2['n_feasible_points']}/{st2['n_points']} 个位置可行, "
              f"每个位置试了 {st2['n_angles']} 个角度; "
              f"IK 调用 {st2['n_ik_calls']} 次, 用时 {st2['elapsed_s']:.0f}s",
              f"{st2['n_feasible_points']}/{st2['n_points']} feasible"),
            L("place 位置 x (m)  [LINK_0 系]", "place x (m)"),
            L("place 位置 y (m)  [LINK_0 系]", "place y (m)"))
    else:
        print(f"[skip] 没有 {p12}")

    if p3.is_file() and s12 is not None:
        s3 = json.loads(p3.read_text(encoding="utf-8"))
        place_heatmap(s3, s12, rd / "place_rank_map.png")
        gstep3 = float(s3["grasp_grid"].get("step", 0.03))
        ranked = sorted([r for r in s3["results"] if r["all_stages_ok"]],
                        key=lambda r: -r["n_success"])
        top = int(s3.get("top_n", 10))
        if args.top_maps:
            top = min(top, args.top_maps)
        for k, r in enumerate(ranked[:top]):
            ok_fail_map(
                s3["grasps"], r["success_indices"], gstep3,
                rd / f"stage3_top{k + 1}_place{r['place_index']}.png",
                L(f"阶段3: place #{r['place_index']} 的 grasp 可达性 (第 {k + 1} 名)",
                  f"Stage3: place #{r['place_index']} (rank {k + 1})"),
                L(f"place={np.round(r['position'], 3).tolist()}  "
                  f"成功 {r['n_success']}/{r['n_grasp_total']} "
                  f"({r['success_ratio'] * 100:.1f}%)  最远 {r['max_dist']:.3f}m  "
                  f"[完整 trajopt 验证, 含关节变化判据]",
                  f"place={np.round(r['position'], 3).tolist()} "
                  f"{r['n_success']}/{r['n_grasp_total']}"),
                L("grasp 位置 x (m)  [LINK_0 系]", "grasp x (m)"),
                L("grasp 位置 y (m)  [LINK_0 系]", "grasp y (m)"),
                marks=[{"pos": r["position"],
                        "label": L(f"place 点 z={r['position'][2]:.3f}",
                                   f"place z={r['position'][2]:.3f}")}])
        stage3_summary(s3, rd / "stage3_summary.png")
    elif not p3.is_file():
        print(f"[skip] 没有 {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
