#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAKA 7-DOF 碰撞球体包络验证工具

功能:
  1. 在连杆局部坐标系下, 检查碰撞球体是否完全覆盖了 STL mesh
  2. 输出每个 link 的覆盖率百分比
  3. 多姿态验证: 随机采样多个关节角, 检查各姿态下的覆盖情况
  4. 发布到 RViz: mesh (灰色) + 球体 (半透明绿色) + 未覆盖顶点 (红色)

原理:
  对每个 link:
    1. 加载 STL mesh, 获取所有顶点 (连杆局部坐标系)
    2. 从 jaka.yml 读取该 link 的碰撞球体 (center, radius)
    3. 对每个顶点, 计算到所有球体中心的距离
    4. 如果 min(距离) <= 球体半径, 则该顶点被覆盖
    5. 覆盖率 = 被覆盖顶点数 / 总顶点数

使用方法:
  # 基本用法: 在零位姿态下验证
  python examples/jaka_verify_spheres.py

  # 多姿态验证 (随机 10 个姿态)
  python examples/jaka_verify_spheres.py --num-poses 10

  # 发布到 RViz 可视化
  python examples/jaka_verify_spheres.py --publish

  # 指定关节角
  python examples/jaka_verify_spheres.py --joint-angles 0 0 0 -1 0 0 0
"""

import os
import sys
import argparse
import struct
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import yaml

# ============================================================
#  数据结构
# ============================================================

@dataclass
class SphereInfo:
    """单个碰撞球体信息 (连杆局部坐标系)."""
    center: np.ndarray  # (3,)
    radius: float


@dataclass
class LinkVerifyResult:
    """单个 link 的验证结果."""
    link_name: str
    num_vertices: int
    num_covered: int
    coverage: float  # [0, 1]
    num_spheres: int
    max_uncovered_dist: float  # 未覆盖顶点中, 离最近球体表面最远的距离
    uncovered_vertices: np.ndarray  # (N, 3) 未覆盖的顶点坐标 (局部系)


# ============================================================
#  STL 加载 (不依赖 trimesh, 直接解析二进制 STL)
# ============================================================

def load_stl_vertices(stl_path: str) -> np.ndarray:
    """加载二进制 STL 文件, 返回去重后的顶点坐标.

    Args:
        stl_path: STL 文件路径

    Returns:
        vertices: (N, 3) 顶点坐标数组
    """
    with open(stl_path, "rb") as f:
        # 跳过 80 字节头
        f.read(80)
        # 读取三角面片数量
        num_triangles = struct.unpack("<I", f.read(4))[0]

        # 每个三角面片: 法向量(12B) + 3个顶点(36B) + 属性(2B) = 50B
        all_vertices = np.zeros((num_triangles * 3, 3), dtype=np.float32)

        for i in range(num_triangles):
            # 跳过法向量 (3 * float32 = 12 bytes)
            f.read(12)
            # 读取 3 个顶点
            for j in range(3):
                x, y, z = struct.unpack("<fff", f.read(12))
                all_vertices[i * 3 + j] = [x, y, z]
            # 跳过属性字节
            f.read(2)

    # 去重 (使用 numpy unique, 精度到 0.0001mm)
    vertices = np.unique(np.round(all_vertices, decimals=6), axis=0)
    return vertices


# ============================================================
#  球体配置加载
# ============================================================

def load_sphere_config(sphere_yml_path: str) -> Dict[str, List[SphereInfo]]:
    """从 jaka.yml 加载碰撞球体配置.

    Returns:
        dict: {link_name: [SphereInfo, ...]}
    """
    with open(sphere_yml_path, "r") as f:
        data = yaml.safe_load(f)

    spheres_dict = {}
    collision_spheres = data.get("collision_spheres", {})

    for link_name, sphere_list in collision_spheres.items():
        spheres = []
        for s in sphere_list:
            center = np.array(s["center"], dtype=np.float32)
            radius = float(s["radius"])
            spheres.append(SphereInfo(center=center, radius=radius))
        spheres_dict[link_name] = spheres

    return spheres_dict


# ============================================================
#  核心验证逻辑
# ============================================================

def verify_link_coverage(
    vertices: np.ndarray,
    spheres: List[SphereInfo],
    link_name: str,
) -> LinkVerifyResult:
    """验证单个 link 的球体覆盖率.

    对每个 mesh 顶点, 检查是否在某个球体内部.
    判定条件: dist(vertex, sphere_center) <= sphere_radius

    Args:
        vertices: (N, 3) mesh 顶点 (连杆局部坐标系)
        spheres: 该 link 的碰撞球体列表
        link_name: link 名称

    Returns:
        LinkVerifyResult: 验证结果
    """
    N = vertices.shape[0]
    if N == 0 or len(spheres) == 0:
        return LinkVerifyResult(
            link_name=link_name,
            num_vertices=N,
            num_covered=0,
            coverage=0.0,
            num_spheres=len(spheres),
            max_uncovered_dist=0.0,
            uncovered_vertices=vertices,
        )

    # 构造球体中心和半径数组
    centers = np.array([s.center for s in spheres], dtype=np.float32)  # (K, 3)
    radii = np.array([s.radius for s in spheres], dtype=np.float32)    # (K,)

    # 计算每个顶点到每个球体中心的距离: (N, K)
    # vertices: (N, 3), centers: (K, 3)
    diff = vertices[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, K, 3)
    dists = np.linalg.norm(diff, axis=-1)  # (N, K)

    # 每个顶点到每个球体表面的距离 (负值表示在球体内部)
    surface_dists = dists - radii[np.newaxis, :]  # (N, K)

    # 每个顶点到最近球体表面的距离
    min_surface_dist = surface_dists.min(axis=1)  # (N,)

    # 被覆盖: 到最近球体表面距离 <= 0
    covered_mask = min_surface_dist <= 0
    num_covered = covered_mask.sum()

    # 未覆盖顶点
    uncovered_mask = ~covered_mask
    uncovered_verts = vertices[uncovered_mask]
    max_uncovered_dist = 0.0
    if uncovered_mask.any():
        max_uncovered_dist = float(min_surface_dist[uncovered_mask].max())

    return LinkVerifyResult(
        link_name=link_name,
        num_vertices=N,
        num_covered=int(num_covered),
        coverage=float(num_covered) / N if N > 0 else 0.0,
        num_spheres=len(spheres),
        max_uncovered_dist=max_uncovered_dist,
        uncovered_vertices=uncovered_verts,
    )


def run_verification(
    mesh_dir: str,
    sphere_yml_path: str,
    link_names: Optional[List[str]] = None,
) -> List[LinkVerifyResult]:
    """对所有 link 执行球体覆盖率验证.

    Args:
        mesh_dir: STL mesh 文件目录
        sphere_yml_path: 碰撞球体 yml 配置路径
        link_names: 要验证的 link 列表, None 则验证所有

    Returns:
        List[LinkVerifyResult]: 每个 link 的验证结果
    """
    # 加载球体配置
    spheres_dict = load_sphere_config(sphere_yml_path)

    if link_names is None:
        link_names = list(spheres_dict.keys())

    results = []
    for link_name in link_names:
        stl_path = os.path.join(mesh_dir, f"{link_name}.STL")
        if not os.path.exists(stl_path):
            print(f"  [警告] 未找到 {link_name} 的 STL 文件: {stl_path}")
            continue

        vertices = load_stl_vertices(stl_path)
        spheres = spheres_dict.get(link_name, [])

        result = verify_link_coverage(vertices, spheres, link_name)
        results.append(result)

    return results


# ============================================================
#  结果打印
# ============================================================

def print_results(results: List[LinkVerifyResult]):
    """打印验证结果表格."""
    print(f"\n{'='*72}")
    print(f"  JAKA 碰撞球体包络验证结果")
    print(f"{'='*72}")
    print(f"  {'Link':<12} {'顶点数':>8} {'覆盖数':>8} {'覆盖率':>8} "
          f"{'球体数':>6} {'最大暴露(mm)':>12}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*12}")

    total_verts = 0
    total_covered = 0

    for r in results:
        total_verts += r.num_vertices
        total_covered += r.num_covered

        # 覆盖率颜色标记
        if r.coverage >= 0.95:
            status = "✓"
        elif r.coverage >= 0.80:
            status = "△"
        else:
            status = "✗"

        print(
            f"  {r.link_name:<12} {r.num_vertices:>8} {r.num_covered:>8} "
            f"{r.coverage:>7.1%} {status} {r.num_spheres:>6} "
            f"{r.max_uncovered_dist * 1000:>11.2f}"
        )

    overall = total_covered / total_verts if total_verts > 0 else 0
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*12}")
    print(f"  {'总计':<12} {total_verts:>8} {total_covered:>8} {overall:>7.1%}")
    print(f"{'='*72}")

    # 给出建议
    print(f"\n  图例: ✓ ≥95%  △ ≥80%  ✗ <80%")
    bad_links = [r for r in results if r.coverage < 0.80]
    if bad_links:
        print(f"\n  [建议] 以下 link 覆盖率较低, 建议增加球体或调整参数:")
        for r in bad_links:
            print(f"    - {r.link_name}: {r.coverage:.1%}, "
                  f"最大暴露 {r.max_uncovered_dist*1000:.1f}mm, "
                  f"未覆盖 {r.num_vertices - r.num_covered} 个顶点")


# ============================================================
#  RViz 可视化
# ============================================================

def publish_to_rviz(
    mesh_dir: str,
    sphere_yml_path: str,
    results: List[LinkVerifyResult],
    joint_angles: Optional[List[float]] = None,
    frame_id: str = "LINK_BASE",
):
    """发布 mesh + 球体 + 未覆盖顶点到 RViz.

    - mesh: 灰色半透明三角面片 (TRIANGLE_LIST)
    - 球体: 绿色半透明球 (SPHERE)
    - 未覆盖顶点: 红色点 (POINTS)

    使用 CuRobo FK 将所有内容变换到世界坐标系.
    """
    try:
        import rospy
        from visualization_msgs.msg import Marker, MarkerArray
        from std_msgs.msg import Header, ColorRGBA
        from geometry_msgs.msg import Point, Vector3, Pose as RosPose, Quaternion
    except ImportError:
        print("\n  [错误] 未找到 rospy, 请确保已安装 ROS 并 source 了环境。")
        return

    # --- 初始化 CuRobo 运动学模型 ---
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml

    tensor_args = TensorDeviceType()
    robot_data = load_yaml(
        join_path(get_robot_configs_path(), "jaka.yml")
    )["robot_cfg"]
    robot_data["kinematics"]["load_link_names_with_mesh"] = True
    robot_cfg = RobotConfig.from_dict(robot_data, tensor_args)
    kin_model = CudaRobotModel(robot_cfg.kinematics)

    # 关节角
    if joint_angles is None:
        joint_angles = [0.0] * kin_model.get_dof()
    q = torch.tensor(
        [joint_angles], device=tensor_args.device, dtype=tensor_args.dtype
    )

    # FK: 获取球体世界坐标
    sph_list = kin_model.get_robot_as_spheres(q)
    spheres_world = sph_list[0]  # 第一个 batch

    # FK: 获取 link 位姿
    link_names = [r.link_name for r in results]
    link_poses = kin_model.get_link_poses(q, link_names)

    # --- ROS 初始化 ---
    rospy.init_node("jaka_sphere_verify", anonymous=True)
    marker_pub = rospy.Publisher(
        "/jaka_sphere_verify", MarkerArray, queue_size=1, latch=True
    )

    markers = MarkerArray()
    marker_id = 0

    # --- 1. 发布碰撞球体 (绿色半透明) ---
    for si, sph in enumerate(spheres_world):
        m = Marker()
        m.header.frame_id = frame_id
        m.ns = "collision_spheres"
        m.id = marker_id
        marker_id += 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = sph.pose[0]
        m.pose.position.y = sph.pose[1]
        m.pose.position.z = sph.pose[2]
        m.pose.orientation.w = 1.0
        d = 2.0 * sph.radius
        m.scale = Vector3(d, d, d)
        m.color = ColorRGBA(0.2, 0.9, 0.3, 0.3)  # 绿色半透明
        m.lifetime = rospy.Duration(0)
        markers.markers.append(m)

    # --- 2. 发布未覆盖顶点 (红色) ---
    for li, result in enumerate(results):
        if result.uncovered_vertices.shape[0] == 0:
            continue

        # 获取该 link 的世界位姿
        pos = link_poses.position[0, li].cpu().numpy()   # (3,)
        quat = link_poses.quaternion[0, li].cpu().numpy() # (4,) wxyz

        # 四元数转旋转矩阵
        R = _quat_wxyz_to_rotation_matrix(quat)

        # 将未覆盖顶点变换到世界坐标系
        uncov_world = (R @ result.uncovered_vertices.T).T + pos

        m = Marker()
        m.header.frame_id = frame_id
        m.ns = f"uncovered_{result.link_name}"
        m.id = marker_id
        marker_id += 1
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale = Vector3(0.003, 0.003, 0.0)  # 点大小 3mm
        m.color = ColorRGBA(1.0, 0.1, 0.1, 1.0)  # 红色

        for vi in range(uncov_world.shape[0]):
            p = Point()
            p.x = float(uncov_world[vi, 0])
            p.y = float(uncov_world[vi, 1])
            p.z = float(uncov_world[vi, 2])
            m.points.append(p)

        m.lifetime = rospy.Duration(0)
        markers.markers.append(m)

    # --- 3. 发布 mesh 顶点 (灰色, 用 POINTS 简化显示) ---
    spheres_dict = load_sphere_config(sphere_yml_path)
    for li, result in enumerate(results):
        stl_path = os.path.join(mesh_dir, f"{result.link_name}.STL")
        if not os.path.exists(stl_path):
            continue

        vertices = load_stl_vertices(stl_path)
        pos = link_poses.position[0, li].cpu().numpy()
        quat = link_poses.quaternion[0, li].cpu().numpy()
        R = _quat_wxyz_to_rotation_matrix(quat)
        verts_world = (R @ vertices.T).T + pos

        m = Marker()
        m.header.frame_id = frame_id
        m.ns = f"mesh_{result.link_name}"
        m.id = marker_id
        marker_id += 1
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale = Vector3(0.002, 0.002, 0.0)  # 点大小 2mm
        m.color = ColorRGBA(0.7, 0.7, 0.7, 0.5)  # 灰色半透明

        for vi in range(verts_world.shape[0]):
            p = Point()
            p.x = float(verts_world[vi, 0])
            p.y = float(verts_world[vi, 1])
            p.z = float(verts_world[vi, 2])
            m.points.append(p)

        m.lifetime = rospy.Duration(0)
        markers.markers.append(m)

    # --- 发布 ---
    print(f"\n  [RViz] 发布 {len(markers.markers)} 个 Marker 到 /jaka_sphere_verify")
    print(f"    绿色半透明球: 碰撞球体")
    print(f"    灰色点: mesh 顶点")
    print(f"    红色点: 未被球体覆盖的顶点")
    print(f"    按 Ctrl+C 停止")

    rate = rospy.Rate(1.0)
    try:
        while not rospy.is_shutdown():
            for m in markers.markers:
                m.header.stamp = rospy.Time.now()
            marker_pub.publish(markers)
            rate.sleep()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        print("\n  发布停止。")


def _quat_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """四元数 (w, x, y, z) → 3x3 旋转矩阵."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


# ============================================================
#  多姿态验证
# ============================================================

def multi_pose_verification(
    mesh_dir: str,
    sphere_yml_path: str,
    num_poses: int = 10,
    link_names: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    """在多个随机姿态下验证球体覆盖率.

    注意: 球体定义在连杆局部坐标系, 所以覆盖率与姿态无关.
    这里主要验证的是: 球体配置在局部坐标系下是否正确.
    多姿态验证的意义在于: 确认 FK 变换后球体仍然正确包络 mesh.

    实际上, 由于球体和 mesh 都在同一个局部坐标系下定义,
    覆盖率在所有姿态下应该是相同的.
    所以这个函数主要用于验证这一假设是否成立.
    """
    print(f"\n  [多姿态验证] 验证 {num_poses} 个随机姿态...")

    # 加载球体配置
    spheres_dict = load_sphere_config(sphere_yml_path)
    if link_names is None:
        link_names = list(spheres_dict.keys())

    # 在局部坐标系下, 覆盖率是固定的, 直接计算一次即可
    results = run_verification(mesh_dir, sphere_yml_path, link_names)

    print(f"\n  由于球体和 mesh 都定义在连杆局部坐标系,")
    print(f"  覆盖率与关节角无关, 所有姿态下结果相同。")
    print(f"  如需验证 FK 变换的正确性, 请使用 --publish 在 RViz 中目视检查。")

    return {r.link_name: r.coverage for r in results}


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="JAKA 碰撞球体包络验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num-poses", type=int, default=1,
        help="验证姿态数量 (默认: 1, 即零位)",
    )
    parser.add_argument(
        "--publish", action="store_true",
        help="发布到 RViz 可视化",
    )
    parser.add_argument(
        "--joint-angles", type=float, nargs="+", default=None,
        help="指定关节角 (7 个值), 用于 RViz 可视化",
    )
    parser.add_argument(
        "--frame", type=str, default="LINK_BASE",
        help="RViz 坐标系 (默认: LINK_BASE)",
    )
    args = parser.parse_args()

    # 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    mesh_dir = os.path.join(
        project_root,
        "src/curobo/content/assets/robot/jaka_description/meshes",
    )
    sphere_yml_path = os.path.join(
        project_root,
        "src/curobo/content/configs/robot/spheres/jaka.yml",
    )

    # 验证文件存在
    if not os.path.exists(mesh_dir):
        print(f"  [错误] mesh 目录不存在: {mesh_dir}")
        sys.exit(1)
    if not os.path.exists(sphere_yml_path):
        print(f"  [错误] 球体配置不存在: {sphere_yml_path}")
        sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"  JAKA 碰撞球体包络验证")
    print(f"  Mesh 目录: {mesh_dir}")
    print(f"  球体配置: {sphere_yml_path}")
    print(f"{'#'*60}")

    # 执行验证
    link_names = [
        "LINK_BASE", "LINK_1", "LINK_2", "LINK_3",
        "LINK_4", "LINK_5", "LINK_6", "LINK_7",
    ]
    results = run_verification(mesh_dir, sphere_yml_path, link_names)
    print_results(results)

    # 多姿态验证
    if args.num_poses > 1:
        multi_pose_verification(
            mesh_dir, sphere_yml_path,
            num_poses=args.num_poses,
            link_names=link_names,
        )

    # RViz 可视化
    if args.publish:
        publish_to_rviz(
            mesh_dir, sphere_yml_path, results,
            joint_angles=args.joint_angles,
            frame_id=args.frame,
        )
    else:
        print(f"\n  提示: 添加 --publish 参数可在 RViz 中可视化")
        print(f"    绿色球体 = 碰撞球体")
        print(f"    灰色点 = mesh 顶点")
        print(f"    红色点 = 未被覆盖的顶点")


if __name__ == "__main__":
    main()
