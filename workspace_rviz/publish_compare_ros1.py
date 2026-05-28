#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比可视化：同时发布 baseline / target / diff 三个 PointCloud2 话题。
RViz 中可分别勾选三个 topic 叠加显示：
  /jaka/workspace_baseline  (蓝色)
  /jaka/workspace_target    (绿色)
  /jaka/workspace_diff      (红色, 仅 baseline 才能到达 = 被自碰撞裁掉的)

支持 bbox 过滤、薄切片、最大点数下采样。

用法：
  python3 publish_compare_ros1.py \
    --baseline runs/no_collision_step0p2/workspace_data.npz \
    --target   runs/self_collision_step0p2/workspace_data.npz \
    --frame world --rate 1
"""
import argparse
from pathlib import Path
import numpy as np


SOLID_COLOR = {
    "baseline": np.array([60, 120, 220], dtype=np.uint8),  # 蓝
    "target":   np.array([60, 200, 100], dtype=np.uint8),  # 绿
    "diff":     np.array([230, 60, 60],  dtype=np.uint8),  # 红
}


def filter_points(centers, args):
    n0 = centers.shape[0]
    mask = np.ones(n0, dtype=bool)
    axis_idx = {"x": 0, "y": 1, "z": 2}
    for ax, lo, hi in [("x", args.xmin, args.xmax),
                       ("y", args.ymin, args.ymax),
                       ("z", args.zmin, args.zmax)]:
        col = centers[:, axis_idx[ax]]
        if lo is not None: mask &= col >= lo
        if hi is not None: mask &= col <= hi
    if args.slice_axis is not None:
        col = centers[:, axis_idx[args.slice_axis]]
        half = args.slice_thick * 0.5
        mask &= (col >= args.slice_at - half) & (col <= args.slice_at + half)
    return mask


def downsample(centers, max_points, seed=0):
    if centers.shape[0] <= max_points:
        return centers, np.arange(centers.shape[0])
    idx = np.random.default_rng(seed).choice(centers.shape[0], max_points, replace=False)
    return centers[idx], idx


def make_pc2(points, rgb_solid, frame_id):
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    n = points.shape[0]
    rgb_uint32 = (int(rgb_solid[0]) << 16) | (int(rgb_solid[1]) << 8) | int(rgb_solid[2])
    rgb_float = np.frombuffer(np.uint32(rgb_uint32).tobytes(), dtype=np.float32)[0]
    cloud = np.zeros((n, 4), dtype=np.float32)
    cloud[:, 0:3] = points
    cloud[:, 3] = rgb_float
    msg = PointCloud2()
    msg.header = Header(); msg.header.frame_id = frame_id
    msg.height = 1; msg.width = n
    msg.fields = [
        PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.is_dense = True
    msg.data = cloud.tobytes()
    return msg


def load_centers(npz_path):
    d = np.load(npz_path)
    return d["voxel_centers"].astype(np.float32), d


def compute_diff(a_npz, b_npz):
    """返回 baseline 中存在但 target 中不存在的 voxel 中心。"""
    A = np.load(a_npz); B = np.load(b_npz)
    dims = tuple(int(x) for x in A["dims"])
    iA = A["voxel_index"].astype(np.int64)
    iB = B["voxel_index"].astype(np.int64)
    kA = iA[:,0]*(dims[1]*dims[2]) + iA[:,1]*dims[2] + iA[:,2]
    kB = iB[:,0]*(dims[1]*dims[2]) + iB[:,1]*dims[2] + iB[:,2]
    only_a = ~np.isin(kA, kB, assume_unique=True)
    return A["voxel_centers"][only_a].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--target",   required=True)
    ap.add_argument("--frame", default="world")
    ap.add_argument("--rate",  type=float, default=1.0)
    ap.add_argument("--max-points", type=int, default=2_000_000)
    ap.add_argument("--xmin", type=float); ap.add_argument("--xmax", type=float)
    ap.add_argument("--ymin", type=float); ap.add_argument("--ymax", type=float)
    ap.add_argument("--zmin", type=float); ap.add_argument("--zmax", type=float)
    ap.add_argument("--slice-axis", choices=["x","y","z"])
    ap.add_argument("--slice-at",    type=float, default=0.0)
    ap.add_argument("--slice-thick", type=float, default=0.02)
    ap.add_argument("--topic-prefix", default="/jaka/workspace")
    args = ap.parse_args()

    print(f"[LOAD] baseline = {args.baseline}")
    base_pts, _ = load_centers(args.baseline)
    print(f"[LOAD] target   = {args.target}")
    targ_pts, _ = load_centers(args.target)
    print(f"[CALC] diff (only baseline) ...")
    diff_pts = compute_diff(args.baseline, args.target)
    print(f"  baseline={len(base_pts)}  target={len(targ_pts)}  diff={len(diff_pts)}")

    layers = {"baseline": base_pts, "target": targ_pts, "diff": diff_pts}
    for name in list(layers.keys()):
        pts = layers[name]
        m = filter_points(pts, args)
        pts = pts[m]
        pts, _ = downsample(pts, args.max_points)
        layers[name] = pts
        print(f"  [{name:8s}] after filter/ds = {len(pts)}")

    import rospy
    from sensor_msgs.msg import PointCloud2
    rospy.init_node("jaka_workspace_compare", anonymous=True)
    pubs = {}
    msgs = {}
    for name, pts in layers.items():
        topic = f"{args.topic_prefix}_{name}"
        pubs[name] = rospy.Publisher(topic, PointCloud2, queue_size=1, latch=True)
        msgs[name] = make_pc2(pts, SOLID_COLOR[name], args.frame) if len(pts) > 0 else None
        print(f"  topic {topic}  -> {len(pts)} points  color={SOLID_COLOR[name].tolist()}")

    rate = rospy.Rate(args.rate)
    while not rospy.is_shutdown():
        for name, msg in msgs.items():
            if msg is None: continue
            msg.header.stamp = rospy.Time.now()
            pubs[name].publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
