#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线 matplotlib 可视化（适配 1cm 体素 + 48 SO(3) bin 的正式数据）：
  1) 整体 3D 散点 + 色温 = 姿态覆盖率 (so3_hit / NPIX)
  2) 多 z 切片色温图（默认 z = -0.5/-0.25/0.0/0.25/0.5/0.75 m）
  3) 三正交投影：每个 (x,y) / (x,z) / (y,z) 取该列体素的最大覆盖率（"max-projection"）
图保存到 workspace_rviz/figs/。
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def load(out_dir: Path):
    data = np.load(out_dir / "workspace_data.npz")
    meta_path = out_dir / "workspace_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    return data, meta


def fig_scatter3d(centers, cov, fig_path, max_points=300000):
    n = centers.shape[0]
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)
        centers, cov = centers[idx], cov[idx]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                    c=cov, cmap="turbo", s=1.5, alpha=0.55, vmin=0, vmax=1)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"JAKA workspace ({n:,} voxels), color = SO(3) coverage [0,1]")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7); cbar.set_label("coverage")
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout(); fig.savefig(fig_path, dpi=140); plt.close(fig)
    print(f"[SAVE] {fig_path}")


def fig_z_slices(centers, cov, voxel, fig_path, z_values):
    n_layers = len(z_values)
    cols = 3; rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows), squeeze=False)
    half = voxel * 0.6
    for i, z0 in enumerate(z_values):
        ax = axes[i // cols][i % cols]
        mask = np.abs(centers[:, 2] - z0) <= half
        c, v = centers[mask], cov[mask]
        if c.shape[0] == 0:
            ax.set_title(f"z = {z0:+.2f} m  (empty)")
            ax.set_aspect("equal"); continue
        sc = ax.scatter(c[:, 0], c[:, 1], c=v, cmap="turbo", s=4, vmin=0, vmax=1, marker="s")
        ax.set_aspect("equal"); ax.set_title(f"z = {z0:+.2f} m  ({c.shape[0]:,} vox)")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="coverage")
    # 隐藏多余子图
    for j in range(n_layers, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"JAKA workspace XY slices, voxel={voxel*100:.1f} cm")
    fig.tight_layout(); fig.savefig(fig_path, dpi=140); plt.close(fig)
    print(f"[SAVE] {fig_path}")


def _max_projection(centers, cov, voxel, drop_axis):
    """drop_axis: 0=>沿X积， 输出 (Y,Z) 平面；同理 1->XZ, 2->XY"""
    keep = [a for a in range(3) if a != drop_axis]
    coords = centers[:, keep]
    # 用整数化 key 做 argmax
    ij = np.round(coords / voxel).astype(np.int64)
    i_min = ij.min(axis=0); ij = ij - i_min
    H = int(ij[:, 0].max() + 1); W = int(ij[:, 1].max() + 1)
    grid = np.full((H, W), -1.0, dtype=np.float32)
    flat = ij[:, 0] * W + ij[:, 1]
    # np.maximum.at 处理重复
    np.maximum.at(grid.reshape(-1), flat, cov.astype(np.float32))
    extent = [
        i_min[1] * voxel - voxel/2, (i_min[1] + W) * voxel - voxel/2,
        i_min[0] * voxel - voxel/2, (i_min[0] + H) * voxel - voxel/2,
    ]
    return grid, extent, keep


def fig_max_projections(centers, cov, voxel, fig_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for k, drop in enumerate([2, 1, 0]):  # 沿 Z, Y, X 投影 -> XY, XZ, YZ 平面
        grid, extent, keep = _max_projection(centers, cov, voxel, drop)
        ax = axes[k]
        # -1 为空白
        masked = np.ma.masked_where(grid < 0, grid)
        im = ax.imshow(masked, origin="lower", extent=extent, cmap="turbo",
                       vmin=0, vmax=1, aspect="equal", interpolation="nearest")
        names = "XYZ"
        ax.set_title(f"max-cov projection along {names[drop]}-axis "
                     f"-> plane {names[keep[1]]}{names[keep[0]]}")
        ax.set_xlabel(names[keep[1]] + " (m)"); ax.set_ylabel(names[keep[0]] + " (m)")
        plt.colorbar(im, ax=ax, shrink=0.8, label="max coverage")
    fig.suptitle(f"JAKA workspace max-coverage projections (voxel={voxel*100:.1f} cm)")
    fig.tight_layout(); fig.savefig(fig_path, dpi=140); plt.close(fig)
    print(f"[SAVE] {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/workspace/curobo_ethanqjiang/workspace_rviz")
    ap.add_argument("--z-list", type=float, nargs="+",
                    default=[-0.40, -0.20, 0.00, 0.20, 0.40, 0.60])
    args = ap.parse_args()
    out = Path(args.out_dir)
    fig_dir = out / "figs"; fig_dir.mkdir(exist_ok=True)
    data, meta = load(out)
    centers = data["voxel_centers"]
    cov = data["so3_hit"].astype(np.float32) / float(data["npix"])
    voxel = float(data["voxel"])
    print(f"[INFO] reachable voxels = {centers.shape[0]:,}, voxel = {voxel*100:.1f} cm,"
          f" npix = {int(data['npix'])}")
    fig_scatter3d(centers, cov, fig_dir / "scatter3d.png")
    fig_z_slices(centers, cov, voxel, fig_dir / "z_slices.png", args.z_list)
    fig_max_projections(centers, cov, voxel, fig_dir / "max_projections.png")


if __name__ == "__main__":
    main()
