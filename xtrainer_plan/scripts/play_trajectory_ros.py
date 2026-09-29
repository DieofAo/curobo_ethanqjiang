#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XTrainer 轨迹 ROS1 播放器

读取 plan_trajectory.py 导出的 trajectory.npz，在 rviz 中播放并发布 tf：
  1. /joint_states            -> robot_state_publisher 驱动整臂运动
  2. tf: <base>/xtrainer_start, xtrainer_goal, xtrainer_wp_*   (静态, 起始/目标/路点位姿)
  3. tf: <base>/xtrainer_arm{1,2}_place  (双臂固定 place 位置)
  4. tf: <base>/xtrainer_ee_cmd     (运动过程中 ee 的实时位姿)
  5. tf: <base>/xtrainer_ee_planned (同上, 由规划时 curobo FK 给出, 用于交叉验证)
  6. MarkerArray /xtrainer_plan/markers -> 工作空间线框盒 + 墙体 + 位姿轴 + ee 轨迹线

必须在 ROS noetic 的 python3 (系统 /usr/bin/python3) 下运行，不需要 curobo。

用法:
  python3 play_trajectory_ros.py --traj ../results/20260828_120000
  python3 play_trajectory_ros.py --traj <dir> --speed 0.5 --loop
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtrainer_common import (  # noqa: E402
    build_place_position_tf_specs,
    build_workspace_wall_cuboids,
    load_trajectory,
    parse_rigid_transform_matrix,
    quat_wxyz_to_xyzw,
)
from playback_timing import iter_playback_frames, validate_playback_timing  # noqa: E402
from playback_start import resolve_playback_start  # noqa: E402

import rospy  # noqa: E402
import tf2_ros  # noqa: E402
from geometry_msgs.msg import Point, TransformStamped  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import ColorRGBA  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402


# ============================== tf 辅助 ==============================


def make_tf(parent: str, child: str, pos, quat_wxyz, stamp) -> TransformStamped:
    q = quat_wxyz_to_xyzw(quat_wxyz)
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = float(pos[0])
    t.transform.translation.y = float(pos[1])
    t.transform.translation.z = float(pos[2])
    t.transform.rotation.x = float(q[0])
    t.transform.rotation.y = float(q[1])
    t.transform.rotation.z = float(q[2])
    t.transform.rotation.w = float(q[3])
    return t


# ============================== Marker 辅助 ==============================


def _color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=r, g=g, b=b, a=a)


def wireframe_box_marker(
    frame: str, mid: int, lo, hi, color: ColorRGBA, width: float, ns: str
) -> Marker:
    """用 LINE_LIST 画一个线框长方体。"""
    m = Marker()
    m.header.frame_id = frame
    m.ns = ns
    m.id = mid
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = width
    m.color = color
    m.pose.orientation.w = 1.0
    c = [
        (lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
        (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        m.points.append(Point(x=c[a][0], y=c[a][1], z=c[a][2]))
        m.points.append(Point(x=c[b][0], y=c[b][1], z=c[b][2]))
    return m


def cube_marker(frame: str, mid: int, center, dims, color: ColorRGBA, ns: str,
                quat_wxyz=None) -> Marker:
    m = Marker()
    m.header.frame_id = frame
    m.ns = ns
    m.id = mid
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = float(center[0])
    m.pose.position.y = float(center[1])
    m.pose.position.z = float(center[2])
    q = quat_wxyz_to_xyzw(
        [1.0, 0.0, 0.0, 0.0] if quat_wxyz is None else quat_wxyz)
    m.pose.orientation.x = float(q[0])
    m.pose.orientation.y = float(q[1])
    m.pose.orientation.z = float(q[2])
    m.pose.orientation.w = float(q[3])
    m.scale.x = max(float(dims[0]), 1e-4)
    m.scale.y = max(float(dims[1]), 1e-4)
    m.scale.z = max(float(dims[2]), 1e-4)
    m.color = color
    return m


def line_strip_marker(
    frame: str, mid: int, pts: np.ndarray, color: ColorRGBA, width: float, ns: str
) -> Marker:
    m = Marker()
    m.header.frame_id = frame
    m.ns = ns
    m.id = mid
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = width
    m.color = color
    m.pose.orientation.w = 1.0
    for p in pts:
        m.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
    return m


def text_marker(frame: str, mid: int, pos, text: str, color: ColorRGBA, ns: str) -> Marker:
    m = Marker()
    m.header.frame_id = frame
    m.ns = ns
    m.id = mid
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = float(pos[0])
    m.pose.position.y = float(pos[1])
    m.pose.position.z = float(pos[2]) + 0.045
    m.pose.orientation.w = 1.0
    m.scale.z = 0.025
    m.color = color
    m.text = text
    return m


def sphere_marker(frame: str, mid: int, pos, r: float, color: ColorRGBA, ns: str) -> Marker:
    m = Marker()
    m.header.frame_id = frame
    m.ns = ns
    m.id = mid
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = float(pos[0])
    m.pose.position.y = float(pos[1])
    m.pose.position.z = float(pos[2])
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = 2.0 * r
    m.color = color
    return m


KIND_COLOR = {
    "start": _color(0.1, 0.9, 0.1, 0.95),
    "goal": _color(0.95, 0.15, 0.15, 0.95),
    "lift": _color(0.2, 0.55, 1.0, 0.9),
    "extra": _color(1.0, 0.75, 0.1, 0.9),
    # pick & place 任务的路点类型
    "grasp": _color(0.1, 0.95, 0.4, 0.95),   # 绿：抓取点
    "place": _color(1.0, 0.2, 0.6, 0.95),    # 品红：放置点
    "waypoint": _color(0.8, 0.8, 0.8, 0.9),
}

# 第二臂用蓝/橙色系，与一号臂的绿/红/品红色系区分。
ARM2_KIND_COLOR = {
    "start": _color(0.1, 0.75, 1.0, 0.95),
    "goal": _color(1.0, 0.55, 0.05, 0.95),
    "lift": _color(0.35, 0.55, 1.0, 0.9),
    "extra": _color(0.1, 0.9, 0.95, 0.9),
    "grasp": _color(0.05, 0.65, 1.0, 0.95),
    "place": _color(1.0, 0.65, 0.05, 0.95),
    "waypoint": _color(0.45, 0.75, 1.0, 0.9),
}


def build_static_markers(
    meta: Dict[str, Any],
    base: str,
    ee_pos: np.ndarray,
    args: argparse.Namespace,
    second_ee_pos: Optional[np.ndarray] = None,
) -> MarkerArray:
    """工作空间、墙体、双臂位姿标记和规划 ee 轨迹线。"""
    arr = MarkerArray()
    ws = meta.get("workspace") or {}
    mid = 0

    # 工作空间线框。倾斜安装时画真实旋转盒子，而非仅供报告的 AABB。
    if ws.get("bounds"):
        oriented = ws.get("oriented_bounds")
        b = oriented["reference_bounds"] if oriented else ws["bounds"]
        lo = (b["x"][0], b["y"][0], b["z"][0])
        hi = (b["x"][1], b["y"][1], b["z"][1])
        marker = wireframe_box_marker(
            base, mid, lo, hi, _color(0.1, 1.0, 1.0, 0.9), 0.006, "workspace")
        label_pos = np.asarray(hi, dtype=float)
        if oriented:
            frame = parse_rigid_transform_matrix(
                oriented["frame_transform"], "workspace.oriented_bounds.frame_transform")
            for point in marker.points:
                xyz = frame[:3, :3] @ np.array([point.x, point.y, point.z]) + frame[:3, 3]
                point.x, point.y, point.z = map(float, xyz)
            label_pos = frame[:3, :3] @ label_pos + frame[:3, 3]
        arr.markers.append(marker)
        mid += 1
        arr.markers.append(
            text_marker(
                base, mid, label_pos,
                f"workspace x[{b['x'][0]},{b['x'][1]}] y[{b['y'][0]},{b['y'][1]}] "
                f"z[{b['z'][0]},{b['z'][1]}]" + (" task frame" if oriented else ""),
                _color(0.1, 1.0, 1.0, 0.9), "workspace",
            )
        )
        mid += 1

    # 墙体
    if args.show_walls:
        walls = meta.get("wall_cuboids") or []
        if not walls and ws.get("bounds"):
            walls = build_workspace_wall_cuboids(ws)
        for w in walls:
            arr.markers.append(
                cube_marker(
                    base, mid, w["pose"][:3], w["dims"],
                    _color(0.6, 0.6, 0.65, float(args.wall_alpha)), "walls",
                    quat_wxyz=w["pose"][3:],
                )
            )
            mid += 1

    # 位姿点 + 名字
    for p in meta.get("pose_sequence") or []:
        col = KIND_COLOR.get(p.get("kind", "waypoint"), KIND_COLOR["waypoint"])
        arr.markers.append(sphere_marker(base, mid, p["position"], 0.012, col, "poses"))
        mid += 1
        arr.markers.append(text_marker(base, mid, p["position"], p["name"], col, "poses"))
        mid += 1

    # 新 12-DOF 轨迹在 pose_sequences.arm2 中保存二号臂路点。
    # 旧轨迹没有 pose_sequences，此分支自然为空，原 marker 保持不变。
    pose_sequences = meta.get("pose_sequences") or {}
    arm2_poses = (
        pose_sequences.get("arm2") or []
        if isinstance(pose_sequences, dict) else []
    )
    for p in arm2_poses:
        col = ARM2_KIND_COLOR.get(
            p.get("kind", "waypoint"), ARM2_KIND_COLOR["waypoint"]
        )
        arr.markers.append(
            sphere_marker(base, mid, p["position"], 0.012, col, "poses_arm2")
        )
        mid += 1
        arr.markers.append(
            text_marker(base, mid, p["position"], p["name"], col, "poses_arm2")
        )
        mid += 1

    # 规划出的 ee 轨迹线
    if ee_pos is not None and len(ee_pos) > 1:
        arr.markers.append(
            line_strip_marker(
                base, mid, ee_pos, _color(1.0, 0.4, 0.9, 0.9), 0.004, "ee_path"
            )
        )
        mid += 1
    if second_ee_pos is not None and len(second_ee_pos) > 1:
        arr.markers.append(
            line_strip_marker(
                base, mid, second_ee_pos,
                _color(0.05, 0.7, 1.0, 0.95), 0.004, "ee_path_arm2",
            )
        )
        mid += 1
    return arr


# ============================== 主流程 ==============================


def main() -> int:
    ap = argparse.ArgumentParser(
        description="XTrainer 轨迹 ROS1 播放器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--traj", type=str, required=True, help="轨迹目录或 trajectory.npz 路径")
    ap.add_argument("--joint-topic", type=str, default="/joint_states")
    ap.add_argument("--marker-topic", type=str, default="/xtrainer_plan/markers")
    ap.add_argument("--base-frame", type=str, default=None, help="默认取 meta 中的 base_link")
    ap.add_argument("--speed", type=float, default=1.0, help="播放倍速, <1 更慢")
    ap.add_argument("--rate-hz", type=float, default=0.0,
                    help="0 = 按轨迹时间戳播放; >0 = 固定频率逐点播放")
    ap.add_argument("--display-hz", type=float, default=50.0,
                    help="时间戳模式的显示刷新上限；高倍速跳过显示帧，不修改轨迹")
    start_group = ap.add_mutually_exclusive_group()
    start_group.add_argument("--start-percent", type=float,
                             help="从原轨迹时间进度0–100%%开始，例如50；每次循环从此处开始")
    start_group.add_argument("--start-time", type=float,
                             help="从原轨迹起点后的指定秒数开始（不随speed换算）")
    start_group.add_argument("--start-frame", type=int,
                             help="从指定轨迹采样帧开始，编号从0开始")
    start_group.add_argument("--start-item", type=int,
                             help="从原物料case的循环起点开始，编号从1开始；失败case不能播放")
    ap.add_argument("--loop", action="store_true", help="循环播放")
    ap.add_argument("--loop-pause", type=float, default=1.0, help="循环间停顿(秒)")
    ap.add_argument("--hold", type=float, default=2.0, help="播完保持末端姿态的秒数")
    ap.add_argument("--start-hold", type=float, default=1.5, help="播放前先保持起始姿态的秒数")
    ap.add_argument("--no-walls", dest="show_walls", action="store_false", help="不显示墙体方块")
    ap.add_argument("--wall-alpha", type=float, default=0.12, help="墙体透明度")
    ap.add_argument("--ee-frame", type=str, default="xtrainer_ee_cmd")
    ap.add_argument("--second-ee-frame", type=str, default="xtrainer_second_ee_cmd",
                    help="NPZ 带第二臂 TCP 轨迹时发布的 TF child frame")
    ap.add_argument("--node-name", type=str, default="xtrainer_traj_player")
    ap.add_argument("--mirror-joint-prefix", type=str, default="",
                    help="非空时, /joint_states 里同时带上 <前缀>原名 的关节 "
                         "(值与主臂相同), 驱动场景里第二条臂做相同动作")
    ap.add_argument("--collision-json", type=str, default="",
                    help="双臂碰撞检测报告(dual_arm_collision.json), 提供后 "
                         "播到首个碰撞帧即暂停不动")
    ap.add_argument("--collision-margin-mm", type=float, default=None,
                    help="在报告基础上再加余量(按逐点最小间隙重判), "
                         "不传则用报告自身的判定")
    # 通过 roslaunch 的 launch-prefix 启动时, roslaunch 会把占位 node 的可执行文件路径
    # 与 __name:= / __log:= 追加到命令尾部, 因此必须忽略这些多余参数。
    args, unknown = ap.parse_known_args()
    ros_injected = [a for a in unknown if a.startswith("__") or "robot_state_publisher" in a]
    bad = [a for a in unknown if a not in ros_injected]
    if bad:
        ap.error(f"unrecognized arguments: {' '.join(bad)}")

    data, meta = load_trajectory(args.traj)
    joint_names: List[str] = data["joint_names"]
    positions = data["positions"]
    velocities = data.get("velocities")
    times = data["times"]
    ee_pos = data["ee_positions"]
    ee_quat = data["ee_quats_wxyz"]
    n = positions.shape[0]
    try:
        relative_times = validate_playback_timing(
            times, speed=args.speed, rate_hz=args.rate_hz, display_hz=args.display_hz,
        )
        start_index = resolve_playback_start(
            times, meta, start_percent=args.start_percent, start_time=args.start_time,
            start_frame=args.start_frame, start_item=args.start_item,
        )
    except ValueError as exc:
        ap.error(str(exc))

    # 12-DOF 双臂规划会额外存第二个 TCP 轨迹。两个数组必须成对且
    # 长度与关节轨迹一致；损坏的可选数据不应影响关节轨迹本身的播放。
    second_ee_pos = data.get("second_ee_positions")
    second_ee_quat = data.get("second_ee_quats_wxyz")
    second_ee_error = ""
    if (second_ee_pos is None) != (second_ee_quat is None):
        second_ee_error = "second_ee_positions/second_ee_quats_wxyz 必须成对存在"
        second_ee_pos = second_ee_quat = None
    elif second_ee_pos is not None:
        second_ee_pos = np.asarray(second_ee_pos)
        second_ee_quat = np.asarray(second_ee_quat)
        if second_ee_pos.shape != (n, 3) or second_ee_quat.shape != (n, 4):
            second_ee_error = (
                f"第二臂 TCP 数组形状错误: positions={second_ee_pos.shape}, "
                f"quats={second_ee_quat.shape}, 期望 ({n}, 3)/({n}, 4)"
            )
            second_ee_pos = second_ee_quat = None

    base = args.base_frame or (meta.get("robot") or {}).get("base_link", "LINK_0")
    ee_link = (meta.get("robot") or {}).get("ee_link", "LINK_6")

    rospy.init_node(args.node_name, anonymous=False)
    rospy.loginfo("[xtrainer] traj=%s", args.traj)
    if bool(meta.get("visualization_only", False)):
        rospy.logwarn(
            "[xtrainer] === 仅可视化的离散 IK 诊断；"
            "帧间未规划/未连续验碰，禁止下发机器人 ==="
        )
    rospy.loginfo("[xtrainer] %d points, %.2fs, dt=%s, base=%s, ee=%s",
                  n, float(times[-1]), meta.get("interpolation_dt"), base, ee_link)
    rospy.loginfo("[xtrainer] joints=%s", joint_names)
    start_progress = (
        100.0 * float(relative_times[start_index]) / float(relative_times[-1])
        if relative_times[-1] > 0 else 0.0
    )
    rospy.loginfo(
        "[xtrainer] playback start: frame=%d/%d (zero-based), time=%.3fs, progress=%.2f%%",
        start_index, n - 1, float(relative_times[start_index]), start_progress,
    )
    if args.start_item is not None:
        selected_item = next(item for item in meta["items"]
                             if item["index"] == args.start_item - 1)
        rospy.loginfo("[xtrainer] start item=%d (meta index=%d), task position=%s",
                      args.start_item, selected_item["index"],
                      selected_item.get("position_raw", selected_item.get("position")))
    if start_index:
        rospy.logwarn("[xtrainer] 从保存的中间构型直接预览；未规划当前姿态到该构型的运动，禁止直接用于实机")
    if second_ee_error:
        rospy.logwarn("[xtrainer] %s; 已忽略第二臂 TCP TF", second_ee_error)
    elif second_ee_pos is not None:
        second_ee_link = (meta.get("robot") or {}).get("second_ee_link", "second_TCP_LINK")
        rospy.loginfo(
            "[xtrainer] second ee=%s, tf=%s, start=[%+.4f %+.4f %+.4f], "
            "end=[%+.4f %+.4f %+.4f]",
            second_ee_link, args.second_ee_frame,
            *second_ee_pos[0], *second_ee_pos[-1],
        )
    for s in meta.get("segments") or []:
        rospy.loginfo("[xtrainer] seg %d: %s -> %s  %d pts  %.2fs  err %.2fmm/%.2fdeg",
                      s["index"], s["from"]["name"], s["to"]["name"], s["n_points"],
                      s["duration_s"], s["position_error_mm"], s["rotation_error_deg"])
    wc = meta.get("workspace_check") or {}
    if wc.get("checked"):
        rospy.loginfo("[xtrainer] workspace check: %d/%d violation, max %.1fmm",
                      wc.get("n_violation", 0), wc.get("n_points", 0),
                      wc.get("max_violation_mm", 0.0))
    sc = meta.get("self_collision_check") or {}
    if sc.get("checked"):
        rospy.loginfo("[xtrainer] self-collision check: %d/%d in collision",
                      sc.get("n_collision", 0), sc.get("n_points", 0))

    js_pub = rospy.Publisher(args.joint_topic, JointState, queue_size=10)
    mk_pub = rospy.Publisher(args.marker_topic, MarkerArray, queue_size=2, latch=True)
    tf_bc = tf2_ros.TransformBroadcaster()
    # Keep this publisher alive for late subscribers (e.g. RViz starting later).
    static_tf_bc = tf2_ros.StaticTransformBroadcaster()
    mirror_prefix = args.mirror_joint_prefix.strip()
    trajectory_has_second = any(name.startswith("second_") for name in joint_names)
    trajectory_has_mirror_prefix = bool(mirror_prefix) and any(
        name.startswith(mirror_prefix) for name in joint_names
    )
    mirror_enabled = bool(mirror_prefix) and not (
        trajectory_has_second or trajectory_has_mirror_prefix
    )
    if mirror_prefix and not mirror_enabled:
        rospy.loginfo(
            "[xtrainer] 轨迹已含第二臂独立关节, "
            "不再追加镜像 %sJ_*", mirror_prefix,
        )
    elif mirror_enabled:
        rospy.loginfo("[xtrainer] 镜像关节: 同时发布 %sJ_* (与主臂同值)",
                      mirror_prefix)

    # ---------- 双臂碰撞点预载: 播到首个碰撞帧即暂停 ----------
    collision_idx: set = set()
    collision_pause = False  # 命中后置 True, 之后所有帧都停在原位置
    collision_first: Optional[int] = None
    if args.collision_json:
        try:
            with open(args.collision_json, "r") as f:
                coll_rep = json.load(f)
            if args.collision_margin_mm is None:
                collision_idx = set(coll_rep.get("collision_indices") or [])
                margin_txt = f"margin={coll_rep.get('margin_mm', 0):g}mm(报告原判)"
            else:
                mm = float(args.collision_margin_mm)
                collision_idx = {
                    i for i, c in enumerate(
                        coll_rep.get("clearance_min_per_point_mm") or [])
                    if c < mm
                }
                margin_txt = f"margin={mm:g}mm(重判)"
            collision_idx = {i for i in collision_idx if 0 <= i < n}
            if collision_idx:
                collision_first = min(collision_idx)
                rospy.logwarn(
                    "[xtrainer] 双臂碰撞点 %d 个(%s), 首个 @ 帧 %d (t=%.2fs), "
                    "播到即暂停", len(collision_idx), margin_txt,
                    collision_first, float(times[collision_first]))
            else:
                rospy.loginfo("[xtrainer] 双臂碰撞检测报告: 无碰撞点(%s)",
                              margin_txt)
        except Exception as e:  # noqa: BLE001  报告坏了不挡播放
            rospy.logerr("[xtrainer] 读取碰撞报告失败, 不启用碰撞暂停: %s", e)
            collision_idx = set()

    if collision_first is not None and collision_first < start_index:
        ap.error(
            f"requested start frame {start_index} would bypass the first recorded "
            f"collision at frame {collision_first}; choose an earlier start"
        )

    # ---- 起始/目标/路点 tf ----
    # 固定路点一次性发布到 /tf_static；latch 按 publisher 连接保存，与
    # robot_state_publisher 的固定关节共存。一次发送完整列表，晚加入的
    # RViz 也能收到全部路点，不必每帧重建/发送数千个固定 TF。
    pose_tf_specs = []
    for p in meta.get("pose_sequence") or []:
        kind = p.get("kind", "waypoint")
        if kind == "start":
            child = "xtrainer_start"
        elif kind == "goal":
            child = "xtrainer_goal"
        else:
            child = "xtrainer_wp_" + p["name"]
        pose_tf_specs.append((child, p["position"], p["quat_wxyz"]))
        rospy.logdebug("[xtrainer] pose tf %s <- %s  xyz=%s rpy_deg=%s",
                      base, child,
                      np.round(p["position"], 4).tolist(),
                      np.round(p["rpy_deg"], 2).tolist())

    # 单/双臂 place 都由 metadata 以规划 root（通常为 LINK_0）表示。
    # 单独发布任务级 alias，便于在 RViz 中查看固定放置位置；frame 朝向仅
    # 显示已记录的 LINK_0 任务修正，逐件实际 TCP 姿态仍看 waypoint。
    place_tf_specs = build_place_position_tf_specs(meta)
    pose_tf_specs.extend(place_tf_specs)
    for child, position, _quat in place_tf_specs:
        rospy.loginfo(
            "[xtrainer] place position tf %s <- %s  xyz=%s",
            base, child, np.round(position, 4).tolist(),
        )
    if pose_tf_specs:
        stamp = rospy.Time.now()
        static_tf_bc.sendTransform([
            make_tf(base, child, pos, quat, stamp)
            for child, pos, quat in pose_tf_specs
        ])
    rospy.loginfo("[xtrainer] fixed pose TF: %d, published once on /tf_static",
                  len(pose_tf_specs))

    markers = build_static_markers(meta, base, ee_pos, args, second_ee_pos)
    mk_pub.publish(markers)

    speed = float(args.speed)
    fixed_rate = float(args.rate_hz) > 0.0
    expected_motion_s = (
        (n - 1 - start_index) / float(args.rate_hz) / speed
        if fixed_rate else float(relative_times[-1] - relative_times[start_index]) / speed
    )
    rospy.loginfo(
        "[xtrainer] timing=%s, expected motion=%.3fs (holds excluded), display=%.1fHz",
        "fixed-rate/every-frame" if fixed_rate else "wall-clock/latest-due-frame",
        expected_motion_s, float(args.display_hz),
    )

    def publish_index(i: int) -> None:
        """发布第 i 个点的 joint_states 和动态 ee tf。

        启用了双臂碰撞暂停时, 命中首个碰撞帧后所有后续调用都发该碰撞帧,
        保持姿态不动。
        """
        nonlocal collision_pause, collision_first
        if collision_pause and collision_first is not None:
            i = collision_first
        elif collision_first is not None and i >= collision_first:
            # Even if a caller skips display frames, never jump over the first
            # recorded collision. The timing iterator also clamps to this frame.
            collision_pause = True
            i = collision_first
            rospy.logwarn(
                "[xtrainer] === 双臂碰撞帧 %d (t=%.2fs), 暂停播放 ===",
                i, float(times[i]))
        now = rospy.Time.now()
        js = JointState()
        js.header.stamp = now
        js.name = list(joint_names)
        js.position = [float(v) for v in positions[i]]
        if velocities is not None:
            js.velocity = [float(v) for v in velocities[i]]
        if mirror_enabled:
            # 第二条臂: 同名加前缀, 关节值与主臂完全一致
            js.name += [mirror_prefix + n for n in joint_names]
            js.position += js.position[:len(joint_names)]
            if velocities is not None:
                js.velocity += js.velocity[:len(joint_names)]
        js_pub.publish(js)
        # 运动过程中的 ee tf。cmd 与 planned 同源, 保留两个 frame 便于后续接真机对比
        tfs = [
            make_tf(base, args.ee_frame, ee_pos[i], ee_quat[i], now),
            make_tf(base, "xtrainer_ee_planned", ee_pos[i], ee_quat[i], now),
        ]
        if second_ee_pos is not None:
            tfs.extend([
                make_tf(base, args.second_ee_frame,
                        second_ee_pos[i], second_ee_quat[i], now),
                make_tf(base, "xtrainer_second_ee_planned",
                        second_ee_pos[i], second_ee_quat[i], now),
            ])
        tf_bc.sendTransform(tfs)

    def hold(seconds: float, idx: int) -> None:
        """保持某个构型一段时间, 期间持续发布, 保证 rviz/tf 不过期。

        使用真实时间，保持时间不随 --speed 缩放；inf 表示无限期保持。
        """
        if seconds <= 0:
            return
        t_end = time.monotonic() + seconds
        while not rospy.is_shutdown() and time.monotonic() < t_end:
            publish_index(idx)
            time.sleep(max(0.0, min(0.02, t_end - time.monotonic())))

    time.sleep(0.5)  # 等 publisher 建立连接, 否则前几帧会丢
    loop_i = 0
    while not rospy.is_shutdown():
        loop_i += 1
        rospy.loginfo("[xtrainer] ---- play #%d (speed=%.2fx) ----", loop_i, speed)
        hold(float(args.start_hold), start_index)

        t_start = time.monotonic()
        displayed = 0
        next_progress = start_index
        for i in iter_playback_frames(
            times, speed=speed, rate_hz=args.rate_hz, display_hz=args.display_hz,
            collision_index=collision_first, start_index=start_index,
            is_shutdown=rospy.is_shutdown,
        ):
            if collision_pause:
                break
            publish_index(i)
            displayed += 1
            if i >= next_progress or i == n - 1:
                next_progress = i + max(1, n // 10)
                if second_ee_pos is None:
                    rospy.loginfo(
                        "[xtrainer]   t=%.2fs  %d/%d  ee=[%+.4f %+.4f %+.4f]",
                        float(times[i]), i + 1, n, *ee_pos[i]
                    )
                else:
                    rospy.loginfo(
                        "[xtrainer]   t=%.2fs  %d/%d  "
                        "ee1=[%+.4f %+.4f %+.4f]  ee2=[%+.4f %+.4f %+.4f]",
                        float(times[i]), i + 1, n, *ee_pos[i], *second_ee_pos[i]
                    )
            if collision_pause:
                break
        if rospy.is_shutdown():
            break
        if collision_pause and collision_first is not None:
            # 已撞到双臂碰撞帧: 不继续走 hold/loop, 就停在碰撞帧上不动
            rospy.logwarn(
                "[xtrainer] 双臂碰撞, 停在帧 %d; Ctrl-C 退出", collision_first)
            hold(float("inf"), collision_first)
            break
        rospy.loginfo(
            "[xtrainer] motion elapsed=%.3fs, expected=%.3fs; displayed=%d/%d frames",
            time.monotonic() - t_start, expected_motion_s, displayed, n - start_index,
        )
        hold(float(args.hold), n - 1)
        if second_ee_pos is None:
            rospy.loginfo("[xtrainer] play #%d done, ee end=[%+.4f %+.4f %+.4f]",
                          loop_i, *ee_pos[-1])
        else:
            rospy.loginfo(
                "[xtrainer] play #%d done, ee1 end=[%+.4f %+.4f %+.4f], "
                "ee2 end=[%+.4f %+.4f %+.4f]",
                loop_i, *ee_pos[-1], *second_ee_pos[-1],
            )
        if not args.loop:
            break
        hold(float(args.loop_pause), n - 1)

    # 碰撞暂停过的, 最终保持帧也是碰撞帧 (publish_index 内部已强制,
    # 这里显式选对齐日志与语义)
    final_idx = (collision_first
                 if collision_pause and collision_first is not None else n - 1)
    rospy.loginfo("[xtrainer] finished; holding pose at frame %d. "
                  "Ctrl-C to exit.", final_idx)
    hold(float("inf"), final_idx)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(0)
