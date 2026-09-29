#!/usr/bin/env python3
"""Draw the physical task layout for an independently verified full mount run.

The grasp grid, base origin and place target are all in the saved task_world
frame.  A gray/red cell has no saved complete trajectory; it is not assigned a
joint metric and does not establish geometric unreachability.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from plot_full_mount_joint_distributions import (
    grid_layout, mount_tilt, require, verified_inputs,
)
from plot_grasp_angle_map import pick_cjk_font


def scene_data(meta: dict) -> dict:
    """Validate the saved physical layout against its full-grid metadata."""
    x, y = grid_layout(meta)
    cfg = meta["config"]
    grid = cfg["pick_place"]["grasp_grid"]
    for key in ("rows", "cols", "x_range", "y_range", "z", "perimeter_only"):
        require(grid[key] == meta["grid"][key], f"Config/meta grasp grid differs: {key}")
    mount = np.asarray(cfg["robot"]["mount_transform"], dtype=float)
    require(mount.shape == (4, 4) and np.isfinite(mount).all(), "Invalid mount transform")
    require(np.allclose(mount[3], [0, 0, 0, 1], atol=1e-10), "Invalid mount last row")
    rotation = mount[:3, :3]
    require(np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8)
            and abs(np.linalg.det(rotation) - 1) < 1e-8,
            "Mount rotation is not a proper rotation")
    axis, tilt = mount_tilt(cfg)
    require(axis == "original_base_local_y" and tilt is not None,
            "This layout requires the original-base-local-Y candidate family")
    radians = math.radians(tilt)
    expected_y = np.array([1., 0., 0.])
    expected_z = np.array([0., math.sin(radians), -math.cos(radians)])
    require(np.allclose(rotation[:, 1], expected_y, atol=1e-8)
            and np.allclose(rotation[:, 2], expected_z, atol=1e-8),
            "Mounted local axes disagree with the declared original-base-local-Y tilt")
    place = np.asarray(cfg["pick_place"]["place"]["position"], dtype=float)
    saved_place = np.asarray(meta["place_position_raw"], dtype=float)
    require(place.shape == (3,) and np.allclose(place, saved_place, atol=1e-8),
            "Config/meta place targets differ")
    require(len(meta["items"]) == len(x) * len(y) == int(meta["n_items_total"]),
            "Expected a complete 20x20 full-grid result")
    require(len(x) == len(y) == 20, "Expected the approved 20x20 full grid")
    success = sum(item["success"] is True for item in meta["items"])
    require(success == int(meta["n_items_success"]), "Saved success count differs")
    return {
        "grid": meta["grid"], "mount": mount, "place": place,
        "x": x, "y": y, "items": meta["items"],
        "n_success": success, "tilt_axis": axis, "tilt_deg": tilt,
    }


def draw(scene: dict, title: str, png: Path) -> None:
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if font else (lambda zh, en: en)
    x, y = scene["x"], scene["y"]
    mount, place = scene["mount"], scene["place"]
    base, direction = mount[:3, 3], mount[:3, 2]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 7.1),
                             gridspec_kw={"width_ratios": [1.12, 1]},
                             constrained_layout=True)
    plan, side = axes
    plan.add_patch(Rectangle((x[0]-dx/2, y[0]-dy/2),
                             x[-1]-x[0]+dx, y[-1]-y[0]+dy,
                             facecolor="#eaf4f4", edgecolor="#178a86", lw=1.7))
    for item in scene["items"]:
        px, py, _ = item["position_raw"]
        ok = item["success"] is True
        plan.add_patch(Rectangle((px-dx*.43, py-dy*.43), dx*.86, dy*.86,
                                 facecolor="#49b9ae" if ok else "#eb8a92",
                                 edgecolor="white", lw=.2, zorder=2))
        if not ok:
            plan.plot(px, py, "x", color="#a72b38", ms=3.4, mew=.8, zorder=3)
    plan.scatter(base[0], base[1], marker="v", s=175, facecolors="white",
                 edgecolors="#1565c0", lw=2.5, zorder=6)
    plan.scatter(place[0], place[1], marker="X", s=170, color="#d81b60",
                 edgecolors="black", lw=.6, zorder=7)
    plan.scatter(0, 0, marker="+", s=130, color="#bc9a00", lw=1.8, zorder=7)
    plan.annotate("", xy=(base[0]+direction[0]*.17, base[1]+direction[1]*.17),
                  xytext=base[:2], arrowprops={"arrowstyle": "->", "lw": 2.1,
                                                "color": "#1565c0"})
    plan.set_xlim(min(x[0]-.06, base[0]-.06, place[0]-.06),
                  max(x[-1]+.19, base[0]+.19, place[0]+.06, .08))
    plan.set_ylim(min(y[0]-.09, base[1]-.06, place[1]-.06),
                  max(y[-1]+.06, base[1]+.23, place[1]+.06))
    plan.set_aspect("equal", adjustable="box")
    plan.grid(linestyle=":", color="#aab5bd", lw=.6)
    plan.set_axisbelow(True)
    plan.set_xlabel("task_world X (m)")
    plan.set_ylabel("task_world Y (m)")
    plan.set_title(L(f"俯视：grasp 成功 {scene['n_success']}/400",
                     f"Top view: {scene['n_success']}/400 complete grasps"))
    plan.legend(handles=[
        Patch(facecolor="#49b9ae", label=L("完整轨迹", "Complete trajectory")),
        Patch(facecolor="#eb8a92", label=L("无完整轨迹", "No complete trajectory")),
        Line2D([0], [0], marker="v", ls="none", ms=9, mfc="white",
               mec="#1565c0", mew=2, label=L("基座原点", "Base origin")),
        Line2D([0], [0], marker="X", ls="none", ms=9, color="#d81b60",
               label="place"),
        Line2D([0], [0], marker="+", ls="none", ms=9, color="#bc9a00",
               label="task_world 0"),
    ], loc="upper right", fontsize=8.4)

    grasp_z = float(scene["grid"]["z"])
    side.axhline(0, color="#676d73", lw=1)
    side.add_patch(Rectangle((y[0]-dy/2, grasp_z-.012), y[-1]-y[0]+dy,
                             .024, facecolor="#9adbd6", edgecolor="#178a86"))
    side.scatter(place[1], place[2], marker="X", s=170, color="#d81b60",
                 edgecolors="black", lw=.6, zorder=5)
    side.scatter(base[1], base[2], marker="v", s=170, facecolors="white",
                 edgecolors="#1565c0", lw=2.5, zorder=6)
    length = .27
    side.annotate("", xy=(base[1]+direction[1]*length,
                          base[2]+direction[2]*length),
                  xytext=(base[1], base[2]),
                  arrowprops={"arrowstyle": "->", "lw": 2.8,
                              "color": "#1565c0", "mutation_scale": 16})
    side.plot([base[1], base[1]], [grasp_z, base[2]], ":", color="#8a929b", lw=1)
    side.set_xlim(min(y[0]-.10, place[1]-.08, base[1]-.09),
                  max(y[-1]+.12, base[1]+direction[1]*length+.12))
    side.set_ylim(-.04, max(.77, base[2]+.12))
    side.set_aspect("equal", adjustable="box")
    side.grid(linestyle=":", color="#aab5bd", lw=.6)
    side.set_axisbelow(True)
    side.set_xlabel("task_world Y (m)")
    side.set_ylabel("task_world Z (m)")
    side.set_title(L(f"沿 X 轴看 Y–Z：基座局部 +Z，倾角 {scene['tilt_deg']:g}°",
                     f"Y-Z side view: mounted local +Z, tilt {scene['tilt_deg']:g}°"))
    fig.suptitle(f"{title} | base ({base[0]:+.2f}, {base[1]:+.2f}, {base[2]:+.2f}) m | "
                 f"place ({place[0]:+.2f}, {place[1]:+.2f}, {place[2]:+.2f}) m",
                 fontsize=14, fontweight="bold")
    fig.savefig(png, dpi=175, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    result = args.result.resolve()
    out = (args.out_dir or result.parent.parent / "scene_distributions" / result.name).resolve()
    require(not out.exists(), f"Refusing to overwrite existing output directory: {out}")
    meta, _, _, _, _, _, provenance = verified_inputs(result)
    scene = scene_data(meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".task_scene_", dir=out.parent) as scratch:
        stage = Path(scratch) / "data"
        stage.mkdir()
        draw(scene, result.name, stage / "task_scene_distribution.png")
        report = {
            "schema_version": 1, "provenance": provenance,
            "coordinate_frame": "task_world", "grid": scene["grid"],
            "n_total": len(scene["items"]), "n_success": scene["n_success"],
            "n_without_saved_trajectory": len(scene["items"])-scene["n_success"],
            "mount_transform": scene["mount"].tolist(),
            "mount_origin_m": scene["mount"][:3, 3].tolist(),
            "mounted_local_z_in_task_world": scene["mount"][:3, 2].tolist(),
            "tilt_axis": scene["tilt_axis"], "tilt_deg": scene["tilt_deg"],
            "place_position_m": scene["place"].tolist(),
            "note": "Failed cells have no saved full trajectory; no metric is assigned and reachability is not disproven.",
        }
        (stage / "task_scene_distribution.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(stage, out)
    print(f"[ok] {out}: {scene['n_success']}/{len(scene['items'])} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
