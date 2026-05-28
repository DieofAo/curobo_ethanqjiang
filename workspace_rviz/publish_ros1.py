#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS1 发布：把工作空间体素以 PointCloud2 (rgb 编码 SO(3) 覆盖率) 发布出去。

适配 noetic / Python3。在没装 rospy 的机器上不会被 import 触发；只在 main 里 import。

启动：
  rosrun ... 或 python publish_ros1.py --topic /jaka/workspace --frame world --rate 1
RViz：
  Add -> PointCloud2 -> Topic = /jaka/workspace ; Color Transformer = RGB8

可选：--cube 用 MarkerArray 立方体（颜色更直观，但点多时慢）。
"""
import argparse, json, struct, time
from pathlib import Path
import numpy as np


def colormap_rgb(values01):
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("turbo")
    rgba = cmap(values01)
    rgb = (rgba[:, :3] * 255.0).astype(np.uint8)
    return rgb


def save_pcd_ascii(path, points, rgb):
    """导出 PCL 兼容的 ASCII .pcd（rgb 打包成单 float），不依赖 PCL 库。"""
    n = points.shape[0]
    rgb_uint32 = (rgb[:, 0].astype(np.uint32) << 16) \
               | (rgb[:, 1].astype(np.uint32) << 8) \
               | (rgb[:, 2].astype(np.uint32))
    rgb_float = rgb_uint32.view(np.float32)
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\nDATA ascii\n")
        for i in range(n):
            f.write(f"{points[i,0]:.5f} {points[i,1]:.5f} {points[i,2]:.5f} {rgb_float[i]:.7e}\n")
    print(f"[SAVE] {path}  ({n:,} points)")


def make_pointcloud2(points, rgb, frame_id):
    """生成 sensor_msgs/PointCloud2，xyz + rgb (packed float)。"""
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    n = points.shape[0]
    # 按行打包 [x y z rgb] 4*float32
    rgb_uint32 = (rgb[:, 0].astype(np.uint32) << 16) \
               | (rgb[:, 1].astype(np.uint32) << 8) \
               | (rgb[:, 2].astype(np.uint32))
    rgb_float = rgb_uint32.view(np.float32)  # 不变bit的reinterpret
    cloud = np.zeros((n, 4), dtype=np.float32)
    cloud[:, 0:3] = points
    cloud[:, 3] = rgb_float
    msg = PointCloud2()
    msg.header = Header(); msg.header.frame_id = frame_id
    msg.height = 1; msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.is_dense = True
    msg.data = cloud.tobytes()
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/workspace/curobo_ethanqjiang/workspace_rviz")
    ap.add_argument("--topic", default="/jaka/workspace")
    ap.add_argument("--frame", default="world")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--min-cov", type=float, default=0.0)
    ap.add_argument("--max-points", type=int, default=5_000_000,
                    help="超过则随机下采样，避免 RViz 卡死")
    ap.add_argument("--save-pcd", default="",
                    help="若指定路径，则离线导出 .pcd（不启动 rospy）")
    # bbox 过滤（米），不传则不限制
    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)
    # 薄切片：在指定轴上 [at-thick/2, at+thick/2]
    ap.add_argument("--slice-axis", choices=["x", "y", "z"], default=None,
                    help="启用薄切片模式，例如 z 轴")
    ap.add_argument("--slice-at", type=float, default=0.0,
                    help="切片中心位置（米）")
    ap.add_argument("--slice-thick", type=float, default=0.02,
                    help="切片厚度（米），默认 2cm（与 voxel 同尺度）")
    args = ap.parse_args()

    data = np.load(Path(args.out_dir) / "workspace_data.npz")
    centers = data["voxel_centers"]
    cov = data["so3_hit"].astype(np.float32) / float(data["npix"])
    keep = cov >= args.min_cov
    centers, cov = centers[keep], cov[keep]

    # ---- 空间过滤：bbox + 薄切片 ----
    n0 = centers.shape[0]
    mask = np.ones(n0, dtype=bool)
    axis_idx = {"x": 0, "y": 1, "z": 2}
    bbox = {
        "x": (args.xmin, args.xmax),
        "y": (args.ymin, args.ymax),
        "z": (args.zmin, args.zmax),
    }
    for ax, (lo, hi) in bbox.items():
        col = centers[:, axis_idx[ax]]
        if lo is not None:
            mask &= col >= lo
        if hi is not None:
            mask &= col <= hi
    if args.slice_axis is not None:
        col = centers[:, axis_idx[args.slice_axis]]
        half = args.slice_thick * 0.5
        mask &= (col >= args.slice_at - half) & (col <= args.slice_at + half)
        print(f"[SLICE] axis={args.slice_axis} at={args.slice_at} thick={args.slice_thick}")
    if not mask.all():
        centers, cov = centers[mask], cov[mask]
        print(f"[FILTER] {n0:,} -> {centers.shape[0]:,} points after bbox/slice")
    if centers.shape[0] == 0:
        print("[WARN] 过滤后 0 个点，请放宽 bbox/切片参数")
        return

    if centers.shape[0] > args.max_points:
        idx = np.random.default_rng(0).choice(centers.shape[0], args.max_points, replace=False)
        centers, cov = centers[idx], cov[idx]
    rgb = colormap_rgb(cov)

    # 离线 PCD 导出分支（不需要 ROS 环境）
    if args.save_pcd:
        save_pcd_ascii(args.save_pcd, centers.astype(np.float32), rgb)
        return

    print(f"[INFO] publishing {centers.shape[0]} points on {args.topic} (frame={args.frame})")
    import rospy
    rospy.init_node("jaka_workspace_publisher", anonymous=True)
    from sensor_msgs.msg import PointCloud2
    pub = rospy.Publisher(args.topic, PointCloud2, queue_size=1, latch=True)
    rate = rospy.Rate(args.rate)
    msg = make_pointcloud2(centers.astype(np.float32), rgb, args.frame)
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
