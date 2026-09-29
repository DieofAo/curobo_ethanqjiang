#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
place 扫描 top1 结果点云发布器 (ROS1)

从 results_place_scan/<时间戳> 读取 stage12_result.json + stage3_result.json,
发布两个 PointCloud2 话题 (frame: LINK_0, 同一话题内不分颜色):
  1. /place_scan/top1_points    top1 place 及其轨迹规划成功的 grasp 点
  2. /place_scan/stage12_points 阶段1/2 可行但阶段3 未覆盖的 grasp/place 点

筛选口径 (与扫描实验的三阶段一致):
  * 基本盘: 阶段1/2 IK 判定可行的 grasp/place 点;
  * 再剔除: 阶段3 对 top1 place 轨迹规划失败的 grasp 点
    (含 IK 失败与关节变化超限; 阶段1 就不可行的点根本进不了 detail, 自动剔除);
  * 只处理阶段3 做过 x 范围筛选的子集 (stage3_result.json 本身就是筛选后的结果);
  * place 点同理: 阶段2 可行 ∩ x 筛选 ∩ 阶段3 all_stages_ok。

若目录里还没有 stage3_result.json (只跑了 --stage 12), 退化为发布
阶段1/2 的可行点, 并按 config.stage3 的 x 上限过滤, 与正式口径保持一致。

必须在 ROS noetic 的 python3 (系统 /usr/bin/python3) 下运行, 不需要 curobo。

用法:
  python3 publish_scan_top1_cloud.py                       # 自动用最新一次扫描
  python3 publish_scan_top1_cloud.py --scan-dir results_place_scan/20260901_181723
  python3 publish_scan_top1_cloud.py --rank 3              # 看第3名
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header

TASK_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = TASK_ROOT / "results_place_scan"


# ============================== 数据加载 ==============================


def latest_scan_dir() -> Path:
    """找 results_place_scan 下最新的、至少含 stage12_result.json 的目录。"""
    cands = sorted(
        p.parent for p in SCAN_ROOT.glob("*/stage12_result.json")
    )
    if not cands:
        raise FileNotFoundError(f"{SCAN_ROOT} 下没有 stage12_result.json")
    return cands[-1]


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def points_from_stage3(s3: dict, rank: int) -> Tuple[List[list], List[list], dict]:
    """返回 (成功的 grasp 点, 通过的 place 点, top place 记录)。"""
    top_indices = s3.get("top_place_indices") or s3.get("ranking")
    if not top_indices:
        raise ValueError("stage3_result.json 缺少 top_place_indices/ranking")
    if not 1 <= rank <= len(top_indices):
        raise ValueError(f"--rank 需在 1..{len(top_indices)} 之间")
    top = top_indices[rank - 1]

    res0 = next(r for r in s3["results"] if r["place_index"] == top)
    gpos = {g["index"]: g["position"] for g in s3["grasps"]}
    # ok=True 即「阶段1 可行 (基本盘) 且 阶段3 规划成功」; 两类失败都被剔除
    grasps = [gpos[d["grasp_index"]] for d in res0["detail"] if d.get("ok")]
    places = [r["position"] for r in s3["results"] if r.get("all_stages_ok")]
    return grasps, places, res0


def _stage12_feasible_idx(s12: dict) -> Tuple[set, set]:
    g_ok = {int(k) for k, v in s12.get("grasp_feasible", {}).items() if v}
    p_ok = {int(k) for k, v in s12.get("place_feasible", {}).items() if v}
    return g_ok, p_ok


def stage12_split(s12: dict) -> Tuple[Tuple[List[list], List[list]],
                                      Tuple[List[list], List[list]]]:
    """阶段1/2 可行点按 config.stage3 的 x 上限拆成 (覆盖内, 覆盖外)。

    只跑了阶段1/2 时的退化路径用: 覆盖内的点进 top1 话题, 覆盖外的进
    stage12 话题 (等价于「阶段3 若跑起来会覆盖不到的那部分」)。
    """
    st3cfg = s12.get("config", {}).get("stage3", {})
    gx_max = st3cfg.get("grasp_x_max", float("inf"))
    px_max = st3cfg.get("place_x_max", float("inf"))
    g_ok, p_ok = _stage12_feasible_idx(s12)
    gin = [g["position"] for g in s12["grasps"]
           if g["index"] in g_ok and g["position"][0] <= gx_max + 1e-9]
    gout = [g["position"] for g in s12["grasps"]
            if g["index"] in g_ok and g["position"][0] > gx_max + 1e-9]
    pin = [p["position"] for p in s12["places"]
           if p["index"] in p_ok and p["position"][0] <= px_max + 1e-9]
    pout = [p["position"] for p in s12["places"]
            if p["index"] in p_ok and p["position"][0] > px_max + 1e-9]
    return (gin, pin), (gout, pout)


def stage12_only_points(s12: dict, s3: dict) -> Tuple[List[list], List[list]]:
    """阶段1/2 可行但阶段3 未覆盖的点 (grasp 按位置匹配, 容差 1e-4)。

    stage3 的 grasp 网格更密且按 x 上限过滤, stage1 的粗网格点是其子集;
    stage3 的 place_index 直接沿用 stage12 的 place 索引。
    """
    g_ok, p_ok = _stage12_feasible_idx(s12)
    s3_g = {tuple(round(c, 4) for c in g["position"]) for g in s3["grasps"]}
    s3_p = {r["place_index"] for r in s3["results"]}
    grasps = [g["position"] for g in s12["grasps"]
              if g["index"] in g_ok
              and tuple(round(c, 4) for c in g["position"]) not in s3_g]
    places = [p["position"] for p in s12["places"]
              if p["index"] in p_ok and p["index"] not in s3_p]
    return grasps, places


# ============================== 主流程 ==============================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="把 place 扫描 top-N 的 grasp/place 点发布为 PointCloud2")
    ap.add_argument("--scan-dir", type=str, default=None,
                    help="扫描结果目录, 默认取 results_place_scan 下最新一次")
    ap.add_argument("--rank", type=int, default=1, help="看第几名 (默认 top1)")
    ap.add_argument("--topic", type=str, default="/place_scan/top1_points")
    ap.add_argument("--stage12-topic", type=str,
                    default="/place_scan/stage12_points",
                    help="阶段1/2 可行但阶段3 未覆盖点的发布话题")
    ap.add_argument("--base-frame", type=str, default="LINK_0")
    ap.add_argument("--rate-hz", type=float, default=1.0,
                    help="重发频率, 0 表示只发一次 (话题本身已 latch)")
    args, unknown = ap.parse_known_args()
    # 经 roslaunch launch-prefix 启动时尾部会被追加占位 node 路径与 __name:= 等参数
    bad = [a for a in unknown
           if not (a.startswith("__") or "robot_state_publisher" in a)]
    if bad:
        ap.error(f"无法识别的参数: {bad}")

    scan_dir = Path(args.scan_dir) if args.scan_dir else latest_scan_dir()
    if not scan_dir.is_absolute():
        scan_dir = (Path.cwd() / scan_dir).resolve()

    s12 = load_json(scan_dir / "stage12_result.json")
    s3_path = scan_dir / "stage3_result.json"
    if s3_path.exists():
        s3 = load_json(s3_path)
        grasps, places, res0 = points_from_stage3(s3, args.rank)
        ex_g, ex_p = stage12_only_points(s12, s3)
        n_total = res0["n_grasp_total"]
        src = f"stage3 top{args.rank} (place#{res0['place_index']} " \
              f"pos={[round(p, 3) for p in res0['position']]})"
        src12 = "阶段1/2 可行但阶段3 未覆盖"
    else:
        (grasps, places), (ex_g, ex_p) = stage12_split(s12)
        n_total = len(grasps)
        src = "stage12 (未跑阶段3, 仅 IK 可行点)"
        src12 = "阶段1/2 可行点在 x 筛选外 (阶段3 未跑)"
        if args.rank != 1:
            print(f"[WARN] 无 stage3 结果, --rank {args.rank} 被忽略",
                  file=sys.stderr)

    pts = [tuple(map(float, p)) for p in grasps] + \
          [tuple(map(float, p)) for p in places]
    pts12 = [tuple(map(float, p)) for p in ex_g] + \
            [tuple(map(float, p)) for p in ex_p]
    if not pts and not pts12:
        raise RuntimeError(f"{scan_dir} 筛完没有可发布的点")

    rospy.init_node("place_scan_cloud")
    pub = rospy.Publisher(args.topic, PointCloud2, queue_size=1, latch=True)
    pub12 = rospy.Publisher(args.stage12_topic, PointCloud2,
                            queue_size=1, latch=True)
    header = Header()
    header.frame_id = args.base_frame
    msg = create_cloud_xyz32(header, pts)
    header12 = Header()
    header12.frame_id = args.base_frame
    msg12 = create_cloud_xyz32(header12, pts12)

    print(f"[scan_cloud] dir={scan_dir}")
    print(f"[scan_cloud] {src}: grasp {len(grasps)} 点 + place {len(places)} 点 "
          f"(grasp 共 {n_total} 点, 失败已剔除)")
    print(f"[scan_cloud] 发布 {len(pts)} 点 -> {args.topic} "
          f"(frame={args.base_frame}, latched)")
    print(f"[scan_cloud] {src12}: grasp {len(ex_g)} 点 + place {len(ex_p)} 点")
    print(f"[scan_cloud] 发布 {len(pts12)} 点 -> {args.stage12_topic} "
          f"(frame={args.base_frame}, latched)")

    if args.rate_hz <= 0:
        pub.publish(msg)
        pub12.publish(msg12)
        rospy.spin()
        return
    rate = rospy.Rate(args.rate_hz)
    while not rospy.is_shutdown():
        stamp = rospy.Time.now()
        msg.header.stamp = stamp
        msg12.header.stamp = stamp
        pub.publish(msg)
        pub12.publish(msg12)
        rate.sleep()


if __name__ == "__main__":
    main()
