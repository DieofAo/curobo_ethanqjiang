#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 sample.txt 中 jaka_arm_left success 数据的 FK 一致性。

对每条 success 数据:
  1. 取其关节角 (字段 11-17)
  2. 用 CuRobo FK 计算末端位姿
  3. 与记录的目标 pose (字段 4-10) 对比
  4. 统计位置误差和旋转误差

数据格式 (每行17个字段):
  字段1:   时间戳
  字段2:   机械臂名称 (jaka_arm_left / jaka_arm_right)
  字段3:   状态 (success / solver_failed / large_joint_change)
  字段4-6:  目标位置 (x, y, z)
  字段7-10: 目标四元数 (x, y, z, w)
  字段11-17: 7个关节角度

使用方法:
  # 默认读取项目根目录下的 sample.txt，验证 jaka_arm_left
  python examples/jaka_fk_verify_sample.py

  # 指定轨迹文件
  python examples/jaka_fk_verify_sample.py --file ./xiyiye/left_umi_20260417_145950.txt

  # 指定轨迹文件和机械臂名称
  python examples/jaka_fk_verify_sample.py --file ./data/my_traj.txt --arm jaka_arm_right
"""

import argparse
import os
import time
import math

import torch
import numpy as np

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def quat_angular_distance(q1, q2):
    """计算两个四元数之间的角度距离 (rad)。

    q1, q2: shape (..., 4), 格式 [w, x, y, z]
    返回: shape (...,), 角度距离 (rad)
    """
    # 归一化
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
    # 内积
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(max=1.0)
    # 角度距离 = 2 * arccos(|dot|)
    angle = 2.0 * torch.acos(dot)
    return angle


def parse_success_data(filepath: str, arm_name: str = "jaka_arm_left"):
    """解析轨迹数据文件，提取指定机械臂的 success 数据。

    Args:
        filepath: 数据文件路径
        arm_name: 机械臂名称，用于过滤数据行

    Returns:
        timestamps: list[float]
        positions: np.ndarray (N, 3)   - 目标位置
        quaternions: np.ndarray (N, 4) - 目标四元数 [w, x, y, z] (已从 xyzw 转换)
        joints: np.ndarray (N, 7)      - 关节角度
    """
    timestamps = []
    positions = []
    quaternions = []
    joints_list = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 17:
                continue
            if parts[1] != arm_name or parts[2] != "success":
                continue

            timestamps.append(float(parts[0]))
            positions.append([float(parts[3]), float(parts[4]), float(parts[5])])
            # 数据文件中四元数格式为 (x, y, z, w)，转换为 CuRobo 的 (w, x, y, z)
            qx, qy, qz, qw = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])
            quaternions.append([qw, qx, qy, qz])
            joints_list.append([float(parts[i]) for i in range(10, 17)])

    return (
        timestamps,
        np.array(positions, dtype=np.float32),
        np.array(quaternions, dtype=np.float32),
        np.array(joints_list, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(
        description='验证轨迹数据中 success 数据的 FK 一致性')
    parser.add_argument('--file', type=str, default=None,
                        help='轨迹数据文件路径（默认: 项目根目录下的 sample.txt）')
    parser.add_argument('--arm', type=str, default='jaka_arm_left',
                        help='机械臂名称，用于过滤数据行（默认: jaka_arm_left）')
    args = parser.parse_args()

    # ========== 1. 解析数据 ==========
    if args.file:
        sample_path = os.path.abspath(args.file)
    else:
        sample_path = os.path.join(os.path.dirname(__file__), "..", "sample.txt")
        sample_path = os.path.abspath(sample_path)
    print(f"读取数据文件: {sample_path}")

    timestamps, positions, quaternions, joints = parse_success_data(
        sample_path, arm_name=args.arm)
    total = len(timestamps)
    print(f"{args.arm} success 数据量: {total}")

    if total == 0:
        print("没有 success 数据，退出。")
        return

    # ========== 2. 初始化 CuRobo FK 模型 ==========
    print("\n初始化 CuRobo FK 模型...")
    tensor_args = TensorDeviceType()

    robot_file = "jaka.yml"
    config = load_yaml(join_path(get_robot_configs_path(), robot_file))
    robot_cfg = RobotConfig.from_dict(config["robot_cfg"])
    kin_model = CudaRobotModel(robot_cfg.kinematics)

    print(f"FK 模型初始化完成")
    print(f"  关节名称: {kin_model.kinematics_config.joint_names}")
    print(f"  自由度: {kin_model.get_dof()}")
    print(f"  ee_link: {kin_model.ee_link}")

    # ========== 3. 批量 FK 计算 ==========
    print(f"\n开始批量 FK 计算 ({total} 条)...")

    # 将关节角转为 tensor
    q_tensor = torch.tensor(joints, device=tensor_args.device, dtype=tensor_args.dtype)

    # 批量 FK
    t_start = time.time()
    state = kin_model.get_state(q_tensor)
    torch.cuda.synchronize()
    fk_time = time.time() - t_start

    # 提取 FK 结果
    fk_pos = state.ee_position  # (N, 3)
    fk_quat = state.ee_quaternion  # (N, 4) [w, x, y, z]

    print(f"FK 计算完成，耗时: {fk_time*1000:.2f}ms ({total/fk_time:.0f}Hz)")

    # ========== 4. 计算误差 ==========
    # 目标 pose 转为 tensor
    target_pos = torch.tensor(positions, device=tensor_args.device, dtype=tensor_args.dtype)
    target_quat = torch.tensor(quaternions, device=tensor_args.device, dtype=tensor_args.dtype)

    # 位置误差 (欧氏距离, 单位: m)
    pos_error = torch.norm(fk_pos - target_pos, dim=-1)  # (N,)

    # 旋转误差 (角度距离, 单位: rad)
    rot_error = quat_angular_distance(fk_quat, target_quat)  # (N,)

    # 转为 numpy
    pos_error_np = pos_error.cpu().numpy()
    rot_error_np = rot_error.cpu().numpy()

    # ========== 5. 统计结果 ==========
    print("\n" + "=" * 70)
    print("        JAKA Left Arm FK 验证统计结果 (success 数据)")
    print("=" * 70)

    print(f"\n【基本信息】")
    print(f"  验证数据量:     {total} 条")
    print(f"  FK 计算耗时:    {fk_time*1000:.2f}ms (总) / {fk_time/total*1e6:.1f}μs (每条)")

    # 位置误差统计
    print(f"\n【位置误差 (mm)】")
    print(f"  均值:     {np.mean(pos_error_np)*1000:.4f}")
    print(f"  中位数:   {np.median(pos_error_np)*1000:.4f}")
    print(f"  标准差:   {np.std(pos_error_np)*1000:.4f}")
    print(f"  最小值:   {np.min(pos_error_np)*1000:.4f}")
    print(f"  最大值:   {np.max(pos_error_np)*1000:.4f}")
    print(f"  P90:      {np.percentile(pos_error_np, 90)*1000:.4f}")
    print(f"  P95:      {np.percentile(pos_error_np, 95)*1000:.4f}")
    print(f"  P99:      {np.percentile(pos_error_np, 99)*1000:.4f}")

    # 旋转误差统计
    print(f"\n【旋转误差 (deg)】")
    rot_error_deg = rot_error_np * 180.0 / math.pi
    print(f"  均值:     {np.mean(rot_error_deg):.4f}")
    print(f"  中位数:   {np.median(rot_error_deg):.4f}")
    print(f"  标准差:   {np.std(rot_error_deg):.4f}")
    print(f"  最小值:   {np.min(rot_error_deg):.4f}")
    print(f"  最大值:   {np.max(rot_error_deg):.4f}")
    print(f"  P90:      {np.percentile(rot_error_deg, 90):.4f}")
    print(f"  P95:      {np.percentile(rot_error_deg, 95):.4f}")
    print(f"  P99:      {np.percentile(rot_error_deg, 99):.4f}")

    # 分段统计
    print(f"\n【位置误差分布】")
    thresholds_mm = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for th in thresholds_mm:
        count = np.sum(pos_error_np * 1000 <= th)
        print(f"  ≤ {th:5.1f}mm: {count:6d} / {total} ({count/total*100:6.2f}%)")
    over_10 = np.sum(pos_error_np * 1000 > 10.0)
    print(f"  > 10.0mm: {over_10:6d} / {total} ({over_10/total*100:6.2f}%)")

    print(f"\n【旋转误差分布】")
    thresholds_deg = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for th in thresholds_deg:
        count = np.sum(rot_error_deg <= th)
        print(f"  ≤ {th:5.1f}°:  {count:6d} / {total} ({count/total*100:6.2f}%)")
    over_10d = np.sum(rot_error_deg > 10.0)
    print(f"  > 10.0°:  {over_10d:6d} / {total} ({over_10d/total*100:6.2f}%)")

    # 打印误差最大的前10条
    print(f"\n【位置误差最大的前10条】")
    top_pos_idx = np.argsort(pos_error_np)[-10:][::-1]
    for rank, idx in enumerate(top_pos_idx):
        print(
            f"  [{rank+1:2d}] t={timestamps[idx]:.6f}  "
            f"pos_err={pos_error_np[idx]*1000:.4f}mm  "
            f"rot_err={rot_error_deg[idx]:.4f}°  "
            f"target=({positions[idx][0]:.4f}, {positions[idx][1]:.4f}, {positions[idx][2]:.4f})  "
            f"fk=({fk_pos[idx][0].item():.4f}, {fk_pos[idx][1].item():.4f}, {fk_pos[idx][2].item():.4f})"
        )

    print(f"\n【旋转误差最大的前10条】")
    top_rot_idx = np.argsort(rot_error_np)[-10:][::-1]
    for rank, idx in enumerate(top_rot_idx):
        print(
            f"  [{rank+1:2d}] t={timestamps[idx]:.6f}  "
            f"pos_err={pos_error_np[idx]*1000:.4f}mm  "
            f"rot_err={rot_error_deg[idx]:.4f}°  "
            f"target_q=({quaternions[idx][0]:.4f}, {quaternions[idx][1]:.4f}, "
            f"{quaternions[idx][2]:.4f}, {quaternions[idx][3]:.4f})  "
            f"fk_q=({fk_quat[idx][0].item():.4f}, {fk_quat[idx][1].item():.4f}, "
            f"{fk_quat[idx][2].item():.4f}, {fk_quat[idx][3].item():.4f})"
        )

    # 综合评估
    print(f"\n{'=' * 70}")
    mean_pos_mm = np.mean(pos_error_np) * 1000
    mean_rot_deg = np.mean(rot_error_deg)
    if mean_pos_mm < 1.0 and mean_rot_deg < 1.0:
        print(f"  ✅ FK 验证通过: 平均位置误差 {mean_pos_mm:.4f}mm, 平均旋转误差 {mean_rot_deg:.4f}°")
        print(f"     关节角与记录的 pose 高度一致，数据质量良好。")
    elif mean_pos_mm < 5.0 and mean_rot_deg < 5.0:
        print(f"  ⚠️  FK 验证: 平均位置误差 {mean_pos_mm:.4f}mm, 平均旋转误差 {mean_rot_deg:.4f}°")
        print(f"     存在一定偏差，可能是传感器噪声或时间戳不同步导致。")
    else:
        print(f"  ❌ FK 验证: 平均位置误差 {mean_pos_mm:.4f}mm, 平均旋转误差 {mean_rot_deg:.4f}°")
        print(f"     偏差较大，请检查 URDF 模型或数据来源是否一致。")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
