#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从mcap中提取的pose_command数据，使用CuRobo进行IK求解（含自碰撞检测），
然后通过ROS发布关节角到标准话题 /joint_states。

使用方法:
  1. 先运行 extract_pose_commands.py 提取数据
  2. (可选) 运行 test_ik_solve.py 预计算IK解
  3. 启动 roscore
  4. 启动 rviz 加载机械臂模型
  5. 运行本脚本: conda run -n curobo python3 mcap_ik_publisher.py

pose_command 是 gripper_link 相对于 robot_stand 的位姿。
使用 seed_config 将前一帧的解作为初始种子，保证帧间连续性。
"""

import os
import time
import numpy as np
import torch

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============ 配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, "left_arm_pose_commands.npz")
IK_CACHE_PATH = os.path.join(SCRIPT_DIR, "ik_solutions.npz")
ROBOT_CONFIG_FILE = "marvin_left_arm.yml"
NUM_SEEDS = 32
POSITION_THRESHOLD = 0.005  # 5mm
ROTATION_THRESHOLD = 0.05   # rad
BATCH_SIZE = 64

# 关节名称（与URDF一致）
JOINT_NAMES = [
    "Joint1_L", "Joint2_L", "Joint3_L", "Joint4_L",
    "Joint5_L", "Joint6_L", "Joint7_L",
]


def create_ik_solver():
    """创建带自碰撞检测的IK求解器"""
    tensor_args = TensorDeviceType()
    cfg_dict = load_yaml(join_path(get_robot_configs_path(), ROBOT_CONFIG_FILE))
    robot_cfg = RobotConfig.from_dict(cfg_dict["robot_cfg"])

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=ROTATION_THRESHOLD,
        position_threshold=POSITION_THRESHOLD,
        num_seeds=NUM_SEEDS,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    )
    return IKSolver(ik_config), tensor_args


def solve_ik_sequential(ik_solver, positions, quaternions_wxyz, tensor_args, batch_size=BATCH_SIZE):
    """
    顺序批量IK求解，使用前一批次最后一帧的解作为下一批次的seed，保证连续性。
    """
    N = len(positions)
    dof = 7
    all_solutions = np.zeros((N, dof), dtype=np.float32)
    all_success = np.zeros(N, dtype=bool)

    pos_tensor = torch.from_numpy(positions.astype(np.float32)).to(tensor_args.device)
    quat_tensor = torch.from_numpy(quaternions_wxyz.astype(np.float32)).to(tensor_args.device)

    prev_solution = None

    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        goal = Pose(pos_tensor[i:j], quat_tensor[i:j])
        actual_batch = j - i

        seed_config = None
        if prev_solution is not None:
            # seed_config 形状: (batch, n_seeds, dof)
            seed = prev_solution.unsqueeze(0).unsqueeze(0).expand(actual_batch, 1, -1)
            seed_config = seed

        result = ik_solver.solve_batch(goal, seed_config=seed_config)
        solutions = result.solution.squeeze(1)
        success = result.success.squeeze(1)

        all_solutions[i:j] = solutions.cpu().numpy()
        all_success[i:j] = success.cpu().numpy()

        success_mask = success.bool()
        if success_mask.any():
            last_success_idx = torch.where(success_mask)[0][-1]
            prev_solution = solutions[last_success_idx].clone()
        else:
            prev_solution = solutions[-1].clone()

        n_success = success.sum().item()
        print(f"  批次 [{i}:{j}] 成功: {n_success}/{actual_batch}")

    return all_solutions, all_success


def smooth_solutions(solutions, success, max_joint_jump=0.5):
    """平滑处理：插值失败帧 + 修复关节角跳变"""
    N = len(solutions)
    result = solutions.copy()

    success_indices = np.where(success)[0]
    if len(success_indices) == 0:
        print("  警告: 所有帧IK求解均失败!")
        return result

    # 对失败帧进行插值
    for i in range(N):
        if success[i]:
            continue
        prev_idx = success_indices[success_indices < i]
        next_idx = success_indices[success_indices > i]
        if len(prev_idx) == 0 and len(next_idx) == 0:
            continue
        elif len(prev_idx) == 0:
            result[i] = solutions[next_idx[0]]
        elif len(next_idx) == 0:
            result[i] = solutions[prev_idx[-1]]
        else:
            p, n = prev_idx[-1], next_idx[0]
            alpha = (i - p) / (n - p)
            result[i] = (1 - alpha) * solutions[p] + alpha * solutions[n]

    # 修复关节角跳变
    jumps = np.abs(np.diff(result, axis=0))
    max_jumps = jumps.max(axis=1)
    n_jumps = (max_jumps > max_joint_jump).sum()
    if n_jumps > 0:
        print(f"  检测到 {n_jumps} 帧存在关节角跳变 (>{max_joint_jump:.2f} rad)")
        for i in range(1, N):
            if np.max(np.abs(result[i] - result[i-1])) > max_joint_jump:
                result[i] = result[i-1]

    return result


def main():
    print("=" * 60)
    print("  Marvin 左臂 MCAP IK 求解 + ROS 发布")
    print("=" * 60)

    # ========== 1. 加载数据 ==========
    print(f"\n[1/4] 加载pose数据: {NPZ_PATH}")
    data = np.load(NPZ_PATH)
    timestamps_ns = data['timestamps_ns']
    positions = data['positions']
    quaternions_xyzw = data['quaternions_xyzw']
    N = len(timestamps_ns)
    print(f"  总帧数: {N}")

    # ========== 2 & 3. IK求解（优先加载缓存） ==========
    if os.path.exists(IK_CACHE_PATH):
        print(f"\n[2/4] 加载预计算的IK解: {IK_CACHE_PATH}")
        ik_data = np.load(IK_CACHE_PATH)
        solutions = ik_data['solutions']
        success = ik_data['success']
        n_success = int(success.sum())
        print(f"  成功: {n_success}/{N} ({100.0*n_success/N:.1f}%)")
        print(f"\n[3/4] 平滑处理...")
        solutions = smooth_solutions(solutions, success)
    else:
        print(f"\n[2/4] 未找到缓存，初始化CuRobo IK求解器 (自碰撞检测: ON)...")
        # 转换四元数: xyzw -> wxyz
        quaternions_wxyz = np.zeros_like(quaternions_xyzw)
        quaternions_wxyz[:, 0] = quaternions_xyzw[:, 3]
        quaternions_wxyz[:, 1] = quaternions_xyzw[:, 0]
        quaternions_wxyz[:, 2] = quaternions_xyzw[:, 1]
        quaternions_wxyz[:, 3] = quaternions_xyzw[:, 2]

        ik_solver, tensor_args = create_ik_solver()
        print(f"  关节名: {ik_solver.joint_names}")
        print(f"  DOF: {ik_solver.dof}")

        print(f"\n[3/4] 顺序IK求解 (seed传递保证连续性)...")
        t_start = time.time()
        solutions, success = solve_ik_sequential(
            ik_solver, positions, quaternions_wxyz, tensor_args, batch_size=BATCH_SIZE
        )
        torch.cuda.synchronize()
        t_elapsed = time.time() - t_start

        n_success = int(success.sum())
        print(f"\n  IK求解完成:")
        print(f"    成功: {n_success}/{N} ({100.0*n_success/N:.1f}%)")
        print(f"    耗时: {t_elapsed:.2f}s ({N/t_elapsed:.0f} poses/s)")

        # 保存缓存
        np.savez_compressed(IK_CACHE_PATH, solutions=solutions, success=success,
                            timestamps_ns=timestamps_ns)
        print(f"  IK解已缓存到: {IK_CACHE_PATH}")

        # 平滑处理
        print(f"  平滑处理...")
        solutions = smooth_solutions(solutions, success)

    # ========== 4. ROS发布 ==========
    print(f"\n[4/4] 初始化ROS节点并发布关节角...")
    rospy.init_node('marvin_left_arm_ik_publisher', anonymous=True)
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

    # 等待连接
    rospy.sleep(0.5)

    # 计算时间间隔
    dt_ns = np.diff(timestamps_ns)
    dt_s = dt_ns / 1e9

    print(f"  开始发布 {N} 帧关节角...")
    print(f"  时间间隔: mean={dt_s.mean()*1000:.2f}ms, 频率约{1.0/dt_s.mean():.1f}Hz")
    print(f"  预计总时长: {dt_s.sum():.2f}s")
    print(f"  按 Ctrl+C 停止")

    try:
        for i in range(N):
            if rospy.is_shutdown():
                break

            js = JointState()
            js.header = Header()
            js.header.stamp = rospy.Time.now()
            js.name = JOINT_NAMES
            js.position = solutions[i].tolist()
            js.velocity = []
            js.effort = []

            pub.publish(js)

            # 按mcap中的时间间隔sleep
            if i < N - 1:
                sleep_time = dt_s[i]
                if sleep_time > 0:
                    rospy.sleep(sleep_time)

            if (i + 1) % 500 == 0:
                print(f"    已发布 {i+1}/{N} 帧")

    except rospy.ROSInterruptException:
        pass

    print(f"\n  发布完成!")


if __name__ == "__main__":
    main()
