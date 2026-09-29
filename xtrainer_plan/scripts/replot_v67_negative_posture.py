#!/usr/bin/env python3
"""Replace only V67's overlapping-title posture PNG from its audited JSON/CSV.

The original PNG is archived before atomic replacement. No planning or source
metric files are changed.
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
from matplotlib.colors import Normalize
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
from PIL import Image


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checked(root):
    report_path = root / "final_observed_posture.json"
    csv_path = root / "final_observed_posture_cases.csv"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if (manifest["n_configs"] != 54 or len(manifest["candidates"]) != 54 or
            len(report["candidates"]) != 54 or len(rows) != 54 or
            Path(report["manifest"]).resolve() != (root / "manifest.json").resolve()):
        raise ValueError("Expected matching V67 manifest and 54 posture records")
    axes = manifest["parameters"]
    if (axes.get("prefix") != "v67" or axes.get("size") != 3 or
            axes.get("local_y_tilt_deg") != [-30.0, -45.0, -60.0] or
            axes.get("tilt_axis") != "original_base_local_y"):
        raise ValueError("Expected V67 negative local-Y 3×3 smoke protocol")
    csv_by_name = {row["name"]: row for row in rows}
    data_by_name = {row["name"]: row for row in report["candidates"]}
    names = {row["name"] for row in manifest["candidates"]}
    if (len(csv_by_name) != 54 or len(data_by_name) != 54 or
            names != set(csv_by_name) or names != set(data_by_name)):
        raise ValueError("Posture JSON/CSV names differ from experiment manifest")
    for name in names:
        item, flat = data_by_name[name], csv_by_name[name]
        if (item["tilt_axis"] != "original_base_local_y" or
                float(flat["local_y_tilt_deg"]) != item["tilt_deg"] or
                flat["status"] != item["status"]):
            raise ValueError(f"Posture axis/status mismatch: {name}")
        if item["status"] == "observed":
            angle = item["angle"]["median_deg"]
            if (item["audit"] != "verified" or not math.isfinite(angle) or
                    not math.isclose(float(flat["median_deg"]), angle, abs_tol=1e-9) or
                    int(flat["n_success"]) != item["n_success"]):
                raise ValueError(f"Unaudited or changed posture value: {name}")
        elif item["status"] != "zero_success" or item["angle"] is not None:
            raise ValueError(f"Unexpected posture status: {name}")
    return report, {"json": sha256(report_path), "csv": sha256(csv_path)}


def render(report, output):
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
        if family in available:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False
    rows = report["candidates"]
    xs = sorted({row["mount_xyz_m"][0] for row in rows})
    ys = sorted({row["mount_xyz_m"][1] for row in rows})
    zs = sorted({row["mount_xyz_m"][2] for row in rows})
    tilts = sorted({row["tilt_deg"] for row in rows})
    if (xs != [-.2, -.1] or ys != [.15, .35, .45] or
            zs != [.45, .55, .65] or tilts != [-60., -45., -30.]):
        raise ValueError("Unexpected V67 posture plot axes")
    lookup = {(*row["mount_xyz_m"], row["tilt_deg"]): row for row in rows}
    if len(lookup) != 54:
        raise ValueError("Duplicate posture plot coordinate")
    fig, panels = plt.subplots(2, 3, figsize=(16.5, 8.6), layout="constrained")
    cmap, norm = plt.get_cmap("viridis"), Normalize(0, 90)
    for i, x in enumerate(xs):
        for j, tilt in enumerate(tilts):
            ax = panels[i, j]
            for yi, y in enumerate(ys):
                for zi, z in enumerate(zs):
                    item = lookup[(x, y, z, tilt)]
                    angle = item["angle"]
                    if angle is None:
                        face, label, ink = "#e4e9ed", "0/9\n无完整轨迹", "#364454"
                    else:
                        value = angle["median_deg"]
                        face = cmap(norm(value))
                        label = f"{value:.0f}°\n{item['n_success']}/9"
                        ink = "white" if value < 55 else "#152330"
                    ax.add_patch(Rectangle((zi - .5, yi - .5), 1, 1,
                                           facecolor=face, edgecolor="white", linewidth=1.5))
                    ax.text(zi, yi, label, ha="center", va="center",
                            fontsize=11, fontweight="semibold", color=ink)
            ax.set_xlim(-.5, 2.5)
            ax.set_ylim(-.5, 2.5)
            ax.set_xticks(range(3), [f"{value:.2f}" for value in zs])
            ax.set_yticks(range(3), [f"{value:.2f}" for value in ys])
            ax.set_xlabel("基座 Z (m)" if i == 1 else "")
            ax.set_ylabel("基座 Y (m)" if j == 0 else "")
            ax.set_title(f"X = {x:+.2f} m  ·  倾角 {tilt:g}°", fontsize=13, pad=11)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=10)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            ax=panels.ravel().tolist(), shrink=.82, pad=.02)
    colorbar.set_label("肩到腕连线距竖直方向的中位角 (°)", fontsize=11)
    fig.suptitle("V67 · 负角基座抓取姿态分布\n"
                 "绕原始基座局部 +Y 轴旋转；0° 表示直立，90° 表示水平",
                 fontsize=16, fontweight="bold")
    fig.legend(handles=[Patch(facecolor="#e4e9ed", edgecolor="#aab5be",
                              label="已尝试 0/9，未保存完整轨迹")],
               loc="outside lower center", frameon=False, fontsize=10)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Completed V67 negative-local-Y smoke result root")
    args = parser.parse_args()
    root = args.root.resolve()
    original = root / "final_observed_posture.png"
    archive = root / "final_observed_posture_before_title_fix.png"
    if not original.is_file() or archive.exists():
        raise FileExistsError("Original posture PNG must exist and archive must not yet exist")
    report, source_hashes = load_checked(root)
    fd, tmp_name = tempfile.mkstemp(prefix=".v67_posture_", suffix=".png", dir=root)
    os.close(fd)
    temporary = Path(tmp_name)
    try:
        render(report, temporary)
        with Image.open(temporary) as image:
            image.verify()
        shutil.copy2(original, archive)
        os.replace(temporary, original)
    finally:
        temporary.unlink(missing_ok=True)
    if source_hashes != {"json": sha256(root / "final_observed_posture.json"),
                         "csv": sha256(root / "final_observed_posture_cases.csv")}:
        raise RuntimeError("Source JSON/CSV changed while redrawing the report")
    print(f"[ARCHIVE] {archive}\n[PLOT] {original}\n[SOURCES] unchanged {source_hashes}")


if __name__ == "__main__":
    main()
