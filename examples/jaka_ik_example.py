#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAKA 7自由度机器人 CuRobo 逆运动学(IK)示例程序

功能说明:
  1. demo_basic_ik()        - 基础IK求解（无碰撞检测）
  2. demo_collision_free_ik() - 带碰撞检测的IK求解
  3. demo_single_pose_ik()  - 单目标位姿IK求解示例

使用方法:
  python examples/jaka_ik_example.py
"""

# 标准库
import time

# 第三方库
import torch
import numpy as np

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# 启用 cuDNN benchmark 以加速 CUDA 运算
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def demo_basic_ik():
    """基础IK求解演示 - 无碰撞检测

    从 JAKA 机器人配置文件中加载 URDF，使用 RobotConfig.from_basic() 创建
    简化的机器人配置，然后进行批量IK求解。
    """
    print("=" * 60)
    print("JAKA 基础IK求解（无碰撞检测）")
    print("=" * 60)

    tensor_args = TensorDeviceType()

    # 加载 JAKA 机器人配置
    config_file = load_yaml(join_path(get_robot_configs_path(), "jaka.yml"))
    urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

    # 从基础参数创建机器人配置
    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

    # 配置IK求解器
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,  # 无世界碰撞配置
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    # 批量IK求解测试
    print("\n运行批量IK求解（每批100个目标位姿）...")
    for i in range(10):
        # 随机采样关节配置，通过正运动学获取目标位姿
        q_sample = ik_solver.sample_configs(100)
        kin_state = ik_solver.fk(q_sample)
        goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()

        success_rate = torch.count_nonzero(result.success).item() / len(q_sample)
        hz = q_sample.shape[0] / (time.time() - st_time)
        print(
            f"  第{i+1}轮: 成功率={success_rate:.2%}, "
            f"求解时间={result.solve_time:.4f}s, "
            f"频率={hz:.1f}Hz, "
            f"位置误差={torch.mean(result.position_error).item()*1000:.2f}mm, "
            f"旋转误差={torch.mean(result.rotation_error).item():.4f}rad"
        )


def demo_collision_free_ik():
    """带碰撞检测的IK求解演示

    使用完整的机器人配置（包含碰撞球体），并加载世界碰撞环境，
    进行无碰撞的IK求解。
    """
    print("\n" + "=" * 60)
    print("JAKA 碰撞安全IK求解")
    print("=" * 60)

    tensor_args = TensorDeviceType()

    # 加载完整的机器人配置
    robot_file = "jaka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    # 加载世界碰撞环境（桌面场景）
    world_file = "collision_table.yml"
    world_cfg = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), world_file))
    )

    # 配置IK求解器（启用自碰撞检测）
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    # 单目标IK求解测试
    print("\n运行单目标IK求解...")
    for i in range(10):
        q_sample = ik_solver.sample_configs(1)
        kin_state = ik_solver.fk(q_sample)
        goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        total_time = (time.time() - st_time) / q_sample.shape[0]

        print(
            f"  第{i+1}轮: 成功={result.success.item()}, "
            f"求解时间={result.solve_time:.4f}s, "
            f"总时间={total_time:.4f}s, "
            f"频率={1.0/total_time:.1f}Hz, "
            f"位置误差={torch.mean(result.position_error).item()*1000:.2f}mm, "
            f"旋转误差={torch.mean(result.rotation_error).item():.4f}rad"
        )

    # 批量IK求解测试
    print("\n运行批量IK求解（10个目标位姿）...")
    q_sample = ik_solver.sample_configs(10)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

    for i in range(3):
        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        success_rate = torch.count_nonzero(result.success).item() / len(q_sample)
        print(
            f"  第{i+1}轮: 成功率={success_rate:.2%}, "
            f"求解时间={result.solve_time:.4f}s, "
            f"总时间={time.time()-st_time:.4f}s"
        )


def demo_single_pose_ik():
    """单目标位姿IK求解示例

    演示如何为 JAKA 机器人求解一个指定的目标位姿。
    """
    print("\n" + "=" * 60)
    print("JAKA 单目标位姿IK求解")
    print("=" * 60)

    tensor_args = TensorDeviceType()

    # 加载完整的机器人配置
    robot_file = "jaka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    # 配置IK求解器
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    # 定义目标位姿: 位置 [x, y, z] 和四元数 [w, x, y, z]
    # 这里先通过正运动学获取一个可达的目标位姿
    retract_config = robot_cfg.kinematics.cspace.retract_config
    if retract_config is not None:
        q_retract = torch.tensor(
            retract_config, device=tensor_args.device, dtype=tensor_args.dtype
        ).unsqueeze(0)
    else:
        q_retract = ik_solver.sample_configs(1)

    kin_state = ik_solver.fk(q_retract)
    target_position = kin_state.ee_position
    target_quaternion = kin_state.ee_quaternion

    print(f"\n目标位置 (x, y, z): {target_position.cpu().numpy().flatten()}")
    print(f"目标四元数 (w, x, y, z): {target_quaternion.cpu().numpy().flatten()}")

    # 求解IK
    goal = Pose(target_position, target_quaternion)
    result = ik_solver.solve_batch(goal)
    torch.cuda.synchronize()

    if result.success.item():
        q_solution = result.js_solution.position
        print(f"\nIK求解成功!")
        print(f"关节角度 (rad): {q_solution.cpu().numpy().flatten()}")
        print(f"关节角度 (deg): {np.degrees(q_solution.cpu().numpy().flatten())}")
        print(f"位置误差: {result.position_error.item()*1000:.4f} mm")
        print(f"旋转误差: {result.rotation_error.item():.6f} rad")
    else:
        print("\nIK求解失败，目标位姿可能不可达。")

    # 也可以直接指定一个目标位姿进行求解
    print("\n--- 自定义目标位姿求解 ---")
    # 示例: 在机器人前方的一个位置
    custom_position = torch.tensor(
        [[0.3, 0.0, 0.5]], device=tensor_args.device, dtype=tensor_args.dtype
    )
    custom_quaternion = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], device=tensor_args.device, dtype=tensor_args.dtype
    )

    goal = Pose(custom_position, custom_quaternion)
    result = ik_solver.solve_batch(goal)
    torch.cuda.synchronize()

    if result.success.item():
        q_solution = result.js_solution.position
        print(f"自定义目标IK求解成功!")
        print(f"目标位置: {custom_position.cpu().numpy().flatten()}")
        print(f"关节角度 (rad): {q_solution.cpu().numpy().flatten()}")
        print(f"位置误差: {result.position_error.item()*1000:.4f} mm")
        print(f"旋转误差: {result.rotation_error.item():.6f} rad")
    else:
        print(f"自定义目标位姿 {custom_position.cpu().numpy().flatten()} 不可达，")
        print("请尝试调整目标位姿到机器人工作空间内。")


if __name__ == "__main__":
    # 运行基础IK演示
    demo_basic_ik()

    # # 运行带碰撞检测的IK演示
    # demo_collision_free_ik()

    # # 运行单目标位姿IK演示
    # demo_single_pose_ik()
