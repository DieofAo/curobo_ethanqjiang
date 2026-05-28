#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工作空间数据 quick-summary（纯文本，无依赖图形栈）。

用法：
    python viz_summary.py
    python viz_summary.py --out-dir <dir>

输出：
  - reachable voxel 数 / 总数
  - 平均 / 中位数 / 最大 SO(3) 覆盖率
  - dexterous workspace（覆盖率>=0.8）体素数
  - 工作空间包络：xyz min/max / 球壳半径分布
"""
import argparse, json
from pathlib import Path
import numpy as np


def load(out_dir: Path):
    data = np.load(out_dir / "workspace_data.npz")
    meta_path = out_dir / "workspace_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    return data, meta


def summarize(data, meta):
    centers = data["voxel_centers"]
    so3_hit = data["so3_hit"].astype(np.int32)
    npix = int(data["npix"])
    voxel = float(data["voxel"])
    cov = so3_hit.astype(np.float64) / float(npix)
    n_vox = centers.shape[0]

    # 1) basic stats
    print("=" * 60)
    print(f"[VOXEL] size = {voxel*100:.2f} cm,  HEALPix npix = {npix}")
    print(f"[VOXEL] reachable voxels = {n_vox:,}")
    if "grid_dims" in data.files:
        gd = data["grid_dims"].tolist()
        total_grid = int(np.prod(gd))
        print(f"[GRID]  dims = {gd},  total = {total_grid:,},  fill ratio = {n_vox/total_grid*100:.2f} %")
    if "bbox_lo" in data.files and "bbox_hi" in data.files:
        print(f"[BBOX]  lo = {data['bbox_lo'].tolist()},  hi = {data['bbox_hi'].tolist()}")

    # 2) coverage stats
    print("-" * 60)
    print(f"[COV]   mean   = {cov.mean():.4f}")
    print(f"[COV]   median = {np.median(cov):.4f}")
    print(f"[COV]   max    = {cov.max():.4f}")
    for thr in (0.1, 0.3, 0.5, 0.8, 0.95):
        n = int((cov >= thr).sum())
        print(f"[COV]   >= {thr:>4.2f}  : {n:>10,} voxels  ({n/n_vox*100:6.2f} %)")

    # 3) envelope stats
    print("-" * 60)
    xyz_min = centers.min(axis=0); xyz_max = centers.max(axis=0)
    print(f"[ENV]   x: [{xyz_min[0]:+.3f}, {xyz_max[0]:+.3f}]  span={xyz_max[0]-xyz_min[0]:.3f} m")
    print(f"[ENV]   y: [{xyz_min[1]:+.3f}, {xyz_max[1]:+.3f}]  span={xyz_max[1]-xyz_min[1]:.3f} m")
    print(f"[ENV]   z: [{xyz_min[2]:+.3f}, {xyz_max[2]:+.3f}]  span={xyz_max[2]-xyz_min[2]:.3f} m")
    r = np.linalg.norm(centers, axis=1)
    print(f"[ENV]   r from base : min={r.min():.3f}  max={r.max():.3f}  mean={r.mean():.3f} m")

    # 4) z-layer histogram (top 10 layers by voxel count)
    print("-" * 60)
    print("[Z-LAYER] top layers by voxel count (z value, count, mean cov):")
    z_round = np.round(centers[:, 2] / voxel) * voxel
    z_keys, inv = np.unique(z_round, return_inverse=True)
    counts = np.bincount(inv)
    cov_sum = np.bincount(inv, weights=cov)
    cov_mean = cov_sum / np.maximum(counts, 1)
    top = np.argsort(-counts)[:10]
    for k in top:
        print(f"  z={z_keys[k]:+.3f} m  : {counts[k]:>8,} voxels   mean_cov={cov_mean[k]:.3f}")

    # 5) save text summary alongside data
    return {
        "voxel": voxel, "npix": npix, "reachable_voxels": int(n_vox),
        "cov_mean": float(cov.mean()), "cov_median": float(np.median(cov)), "cov_max": float(cov.max()),
        "dexterous_voxels_ge_0p8": int((cov >= 0.8).sum()),
        "envelope_xyz_min": xyz_min.tolist(), "envelope_xyz_max": xyz_max.tolist(),
        "r_min": float(r.min()), "r_max": float(r.max()), "r_mean": float(r.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/workspace/curobo_ethanqjiang/workspace_rviz")
    ap.add_argument("--save", action="store_true", help="保存 summary.json")
    args = ap.parse_args()
    out = Path(args.out_dir)
    data, meta = load(out)
    summary = summarize(data, meta)
    if args.save:
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[SAVE] {out / 'summary.json'}")


if __name__ == "__main__":
    main()
