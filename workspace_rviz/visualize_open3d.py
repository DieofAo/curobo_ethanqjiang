#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Open3D 交互式可视化：
  - 左键拖动旋转，右键拖动平移，滚轮缩放
  - 按 T 切换显示阈值（按覆盖率过滤）
  - 默认色温 = SO(3) 覆盖率 (so3_hit / NPIX)，turbo 配色
"""
import argparse, json
from pathlib import Path
import numpy as np
import open3d as o3d
from matplotlib import cm


def load(out_dir: Path):
    data = np.load(out_dir / "workspace_data.npz")
    return data


def colormap(values01, name="turbo"):
    cmap = cm.get_cmap(name)
    return cmap(values01)[:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/workspace/curobo_ethanqjiang/workspace_rviz")
    ap.add_argument("--min-cov", type=float, default=0.0,
                    help="最小覆盖率阈值（默认 0=全部显示）")
    ap.add_argument("--cube", action="store_true",
                    help="把每个体素画成立方体而不是点（更慢，更直观）")
    args = ap.parse_args()
    data = load(Path(args.out_dir))
    centers = data["voxel_centers"]
    cov = data["so3_hit"].astype(np.float32) / float(data["npix"])
    voxel = float(data["voxel"])

    keep = cov >= args.min_cov
    centers, cov = centers[keep], cov[keep]
    print(f"[INFO] showing {centers.shape[0]} voxels (min_cov={args.min_cov})")

    colors = colormap(cov)
    geos = []

    if args.cube:
        # 用 VoxelGrid 显示，速度快
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel)
        geos.append(vg)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geos.append(pcd)

    # 坐标轴 + 机器人 base
    geos.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    print("[HINT] 左键旋转，Shift+左键平移，滚轮缩放，按 Q 退出")
    o3d.visualization.draw_geometries(geos,
        window_name="JAKA workspace (color=SO(3) coverage)",
        width=1280, height=860)


if __name__ == "__main__":
    main()
