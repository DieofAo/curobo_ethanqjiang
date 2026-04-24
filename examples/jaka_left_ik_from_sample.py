#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 sample.txt 中提取 jaka_arm_left 的目标位姿数据，
使用 CuRobo IK 求解器进行批量逆运动学求解，并统计结果。
支持通过 --publish 参数将求解结果发布到 ROS JointState 话题，在 RViz 中可视化。

数据格式 (每行17个字段):
  字段1:  时间戳
  字段2:  机械臂名称 (jaka_arm_left / jaka_arm_right)
  字段3:  状态 (solver_failed / large_joint_change)
  字段4-6:  目标位置 (x, y, z)
  字段7-10: 目标四元数 (x, y, z, w)
  字段11-17: 7个关节角度 (seed)

使用方法:
  # 仅求解并统计
  python examples/jaka_left_ik_from_sample.py

  # 求解后发布到 ROS 话题 (需要先启动 roscore 和 robot_state_publisher)
  python examples/jaka_left_ik_from_sample.py --publish

  # 指定发布速率倍数 (默认1.0x，即按原始时间戳间隔回放)
  python examples/jaka_left_ik_from_sample.py --publish --speed 2.0

  # 循环回放
  python examples/jaka_left_ik_from_sample.py --publish --loop

RViz 可视化前置步骤:
  1. 启动 roscore
  2. 启动 robot_state_publisher:
     rosrun robot_state_publisher robot_state_publisher \
       robot_description:=$(cat /path/to/left_jaka.urdf)
     或者:
     roslaunch your_pkg jaka_display.launch
  3. 在 RViz 中添加 RobotModel 显示，Fixed Frame 设为 LINK_BASE
"""

import os
import time
import argparse

import torch
import numpy as np

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# JAKA 7-DOF 关节名称 (与 URDF 中定义一致)
JAKA_JOINT_NAMES = ["J_1", "J_2", "J_3", "J_4", "J_5", "J_6", "J_7"]


def parse_sample_file(filepath: str):
    """解析 sample.txt，提取 jaka_arm_left 的数据。

    Returns:
        timestamps: list[float]  - 时间戳
        statuses: list[str]      - 原始状态 (solver_failed / large_joint_change)
        positions: np.ndarray    - 目标位置 (N, 3)
        quaternions: np.ndarray  - 目标四元数 (N, 4) [w, x, y, z] (已从 xyzw 转换)
        seed_joints: np.ndarray  - seed 关节角度 (N, 7)
    """
    timestamps = []
    statuses = []
    positions = []
    quaternions = []
    seed_joints = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 17:
                continue
            if parts[1] != "jaka_arm_left":
                continue

            timestamps.append(float(parts[0]))
            statuses.append(parts[2])
            # 位置: x, y, z
            pos = [float(parts[3]), float(parts[4]), float(parts[5])]
            # 四元数: 数据文件中为 x, y, z, w，转换为 CuRobo 的 w, x, y, z
            qx, qy, qz, qw = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])
            quat = [qw, qx, qy, qz]
            # 关节角度: 7个
            joints = [float(parts[i]) for i in range(10, 17)]

            positions.append(pos)
            quaternions.append(quat)
            seed_joints.append(joints)

    return (
        timestamps,
        statuses,
        np.array(positions, dtype=np.float32),
        np.array(quaternions, dtype=np.float32),
        np.array(seed_joints, dtype=np.float32),
    )


def main():
    # ========== 0. 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description="JAKA Left Arm IK 求解 & ROS 可视化")
    parser.add_argument(
        "--file", type=str, default=None,
        help="指定输入数据文件路径 (默认为脚本上级目录的 sample.txt)"
    )
    parser.add_argument(
        "--publish", action="store_true",
        help="求解后将关节角发布到 ROS /joint_states 话题"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="回放速率倍数 (默认 1.0x，2.0 表示两倍速)"
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="循环回放"
    )
    parser.add_argument(
        "--topic", type=str, default="/joint_states",
        help="发布的 ROS 话题名 (默认 /joint_states)"
    )
    parser.add_argument(
        "--focus-failed", action="store_true",
        help="仅播放 IK 失败片段及其前后正常轨迹 (慢速模式)"
    )
    parser.add_argument(
        "--context-size", type=int, default=5,
        help="--focus-failed 模式下，失败片段前后各保留的正常帧数 (默认 5)"
    )
    args = parser.parse_args()

    # ========== 1. 解析数据 ==========
    if args.file:
        sample_path = os.path.abspath(args.file)
    else:
        sample_path = os.path.join(os.path.dirname(__file__), "..", "sample.txt")
        sample_path = os.path.abspath(sample_path)
    print(f"读取数据文件: {sample_path}")

    timestamps, statuses, positions, quaternions, seed_joints = parse_sample_file(sample_path)
    total_count = len(timestamps)
    print(f"jaka_arm_left 总数据量: {total_count}")

    status_counts = {}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"原始状态分布: {status_counts}")

    # ========== 2. 初始化 CuRobo IK 求解器 ==========
    print("\n初始化 CuRobo IK 求解器...")
    tensor_args = TensorDeviceType()

    robot_file = "jaka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,  # 无世界碰撞配置
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=500,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    print(f"IK 求解器初始化完成，关节名称: {ik_solver.joint_names}")

    # ========== 3. 批量 IK 求解 ==========
    # 由于 cuda_graph 启用后 batch_size 不能变化，
    # 我们逐条求解（batch_size=1），并传入 seed_config
    print(f"\n开始逐条 IK 求解 ({total_count} 条)...")

    results_success = []
    results_pos_error = []
    results_rot_error = []
    results_solve_time = []
    results_joint_solutions = []

    # 预热：先用第一条数据做一次求解，让 cuda_graph 完成捕获
    warmup_pos = torch.tensor(
        positions[0:1], device=tensor_args.device, dtype=tensor_args.dtype
    )
    warmup_quat = torch.tensor(
        quaternions[0:1], device=tensor_args.device, dtype=tensor_args.dtype
    )
    warmup_seed = torch.tensor(
        seed_joints[0:1], device=tensor_args.device, dtype=tensor_args.dtype
    ).unsqueeze(0)  # shape: (1, 1, 7)
    warmup_goal = Pose(warmup_pos, warmup_quat)
    _ = ik_solver.solve_single(warmup_goal, seed_config=warmup_seed)
    torch.cuda.synchronize()
    print("CUDA Graph 预热完成\n")

    total_start = time.time()

    for i in range(total_count):
        # 构造目标位姿
        pos_tensor = torch.tensor(
            positions[i : i + 1], device=tensor_args.device, dtype=tensor_args.dtype
        )
        quat_tensor = torch.tensor(
            quaternions[i : i + 1], device=tensor_args.device, dtype=tensor_args.dtype
        )
        goal = Pose(pos_tensor, quat_tensor)

        # 构造 seed_config: shape (1, 1, 7) — (num_seeds_provided, batch, dof)
        seed = torch.tensor(
            seed_joints[i : i + 1], device=tensor_args.device, dtype=tensor_args.dtype
        ).unsqueeze(0)

        # 求解
        st = time.time()
        result = ik_solver.solve_single(goal, seed_config=seed)
        torch.cuda.synchronize()
        solve_t = time.time() - st

        success = result.success.item()
        pos_err = result.position_error.item()
        rot_err = result.rotation_error.item()

        results_success.append(success)
        results_pos_error.append(pos_err)
        results_rot_error.append(rot_err)
        results_solve_time.append(solve_t)

        if success:
            results_joint_solutions.append(
                result.js_solution.position.cpu().numpy().flatten()
            )
        else:
            results_joint_solutions.append(None)

        # 每100条打印一次进度
        if (i + 1) % 100 == 0 or (i + 1) == total_count:
            current_success = sum(results_success[: i + 1])
            print(
                f"  进度: {i+1}/{total_count}, "
                f"当前成功: {current_success}/{i+1} "
                f"({current_success/(i+1)*100:.1f}%)"
            )

    total_time = time.time() - total_start

    # ========== 4. 统计结果 ==========
    print("\n" + "=" * 70)
    print("                    JAKA Left Arm IK 求解统计结果")
    print("=" * 70)

    success_count = sum(results_success)
    fail_count = total_count - success_count
    success_rate = success_count / total_count * 100

    print(f"\n【总体统计】")
    print(f"  总数据量:       {total_count}")
    print(f"  求解成功:       {success_count} ({success_rate:.2f}%)")
    print(f"  求解失败:       {fail_count} ({100 - success_rate:.2f}%)")
    print(f"  总耗时:         {total_time:.2f}s")
    print(f"  平均每条耗时:   {total_time/total_count*1000:.2f}ms")
    print(f"  求解频率:       {total_count/total_time:.1f}Hz")

    # 按原始状态分组统计
    print(f"\n【按原始状态分组统计】")
    for status in sorted(status_counts.keys()):
        indices = [j for j, s in enumerate(statuses) if s == status]
        sub_success = sum(results_success[j] for j in indices)
        sub_total = len(indices)
        print(
            f"  {status:25s}: {sub_success}/{sub_total} 成功 "
            f"({sub_success/sub_total*100:.2f}%)"
        )

    # 成功样本的误差统计
    success_indices = [j for j in range(total_count) if results_success[j]]
    if success_indices:
        success_pos_errors = [results_pos_error[j] for j in success_indices]
        success_rot_errors = [results_rot_error[j] for j in success_indices]
        success_times = [results_solve_time[j] for j in success_indices]

        print(f"\n【成功样本误差统计】(共 {len(success_indices)} 条)")
        print(f"  位置误差 (mm):")
        print(f"    均值:   {np.mean(success_pos_errors)*1000:.4f}")
        print(f"    最大:   {np.max(success_pos_errors)*1000:.4f}")
        print(f"    最小:   {np.min(success_pos_errors)*1000:.4f}")
        print(f"    中位数: {np.median(success_pos_errors)*1000:.4f}")
        print(f"  旋转误差 (rad):")
        print(f"    均值:   {np.mean(success_rot_errors):.6f}")
        print(f"    最大:   {np.max(success_rot_errors):.6f}")
        print(f"    最小:   {np.min(success_rot_errors):.6f}")
        print(f"    中位数: {np.median(success_rot_errors):.6f}")
        print(f"  求解时间 (ms):")
        print(f"    均值:   {np.mean(success_times)*1000:.2f}")
        print(f"    最大:   {np.max(success_times)*1000:.2f}")
        print(f"    最小:   {np.min(success_times)*1000:.2f}")

    # 失败样本的误差统计
    fail_indices = [j for j in range(total_count) if not results_success[j]]
    if fail_indices:
        fail_pos_errors = [results_pos_error[j] for j in fail_indices]
        fail_rot_errors = [results_rot_error[j] for j in fail_indices]

        print(f"\n【失败样本误差统计】(共 {len(fail_indices)} 条)")
        print(f"  位置误差 (mm):")
        print(f"    均值:   {np.mean(fail_pos_errors)*1000:.4f}")
        print(f"    最大:   {np.max(fail_pos_errors)*1000:.4f}")
        print(f"    最小:   {np.min(fail_pos_errors)*1000:.4f}")
        print(f"  旋转误差 (rad):")
        print(f"    均值:   {np.mean(fail_rot_errors):.6f}")
        print(f"    最大:   {np.max(fail_rot_errors):.6f}")
        print(f"    最小:   {np.min(fail_rot_errors):.6f}")

    # 打印前10条失败样本的详细信息
    if fail_indices:
        print(f"\n【前10条失败样本详情】")
        for idx, j in enumerate(fail_indices[:10]):
            print(
                f"  [{idx+1}] 时间={timestamps[j]:.6f}, "
                f"状态={statuses[j]}, "
                f"位置=({positions[j][0]:.4f}, {positions[j][1]:.4f}, {positions[j][2]:.4f}), "
                f"位置误差={results_pos_error[j]*1000:.4f}mm, "
                f"旋转误差={results_rot_error[j]:.6f}rad"
            )

    print("\n" + "=" * 70)
    print("求解完成")
    print("=" * 70)

    # ========== 5. ROS 发布 (可选) ==========
    if args.publish:
        publish_to_ros(
            timestamps=timestamps,
            statuses=statuses,
            positions=positions,
            quaternions=quaternions,
            seed_joints=seed_joints,
            results_success=results_success,
            results_joint_solutions=results_joint_solutions,
            topic=args.topic,
            speed=args.speed,
            loop=args.loop,
            focus_failed=args.focus_failed,
            context_size=args.context_size,
        )


def _build_focus_failed_segments(statuses, results_success, context_size, total):
    """构建 focus-failed 模式下需要播放的帧索引列表和片段信息。

    找出所有 CuRobo IK 求解失败的帧，将其及前后 context_size 帧合并为连续片段。
    返回:
        segments: list[dict]  每个片段包含 start, end, failed_indices
        play_indices_set: set  所有需要播放的帧索引
    """
    # 找出所有 CuRobo 求解失败的帧索引
    failed_indices = [i for i in range(total) if not results_success[i]]
    if not failed_indices:
        return [], set()

    # 为每个失败帧扩展前后 context_size 帧，合并重叠区间
    raw_ranges = []
    for fi in failed_indices:
        start = max(0, fi - context_size)
        end = min(total - 1, fi + context_size)
        raw_ranges.append((start, end))

    # 合并重叠/相邻区间
    merged = [raw_ranges[0]]
    for s, e in raw_ranges[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e + 1:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))

    # 构建片段信息
    segments = []
    play_indices_set = set()
    for seg_start, seg_end in merged:
        seg_failed = [fi for fi in failed_indices if seg_start <= fi <= seg_end]
        segments.append({
            "start": seg_start,
            "end": seg_end,
            "failed_indices": seg_failed,
        })
        for idx in range(seg_start, seg_end + 1):
            play_indices_set.add(idx)

    return segments, play_indices_set


def publish_to_ros(
    timestamps,
    statuses,
    positions,
    quaternions,
    seed_joints,
    results_success,
    results_joint_solutions,
    topic="/joint_states",
    speed=1.0,
    loop=False,
    focus_failed=False,
    context_size=5,
):
    """将求解结果通过 ROS sensor_msgs/JointState 话题发布，用于 RViz 可视化。

    两种播放模式:
      1. 全量模式 (默认): 播放所有帧
      2. 失败片段模式 (--focus-failed): 仅播放 CuRobo IK 失败帧及前后 context_size 帧，
         自动慢速 (speed * 0.3)，对 txt 中标记失败但 CuRobo 成功的 case 重点提醒

    对于求解成功的样本，发布 CuRobo 求解的关节角；
    对于求解失败的样本，发布原始 seed 关节角（终端标注 [SEED]）。
    """
    try:
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Header
        from geometry_msgs.msg import PoseStamped
    except ImportError:
        print("\n[错误] 未找到 rospy / sensor_msgs，请确保已安装 ROS 并 source 了环境。")
        print("  例如: source /opt/ros/noetic/setup.bash")
        return

    rospy.init_node("jaka_left_ik_visualizer", anonymous=True)
    joint_pub = rospy.Publisher(topic, JointState, queue_size=10)
    pose_pub = rospy.Publisher("/ik_target_pose", PoseStamped, queue_size=10)

    total = len(timestamps)

    # ---- 构建播放列表 ----
    if focus_failed:
        segments, play_set = _build_focus_failed_segments(
            statuses, results_success, context_size, total
        )
        if not segments:
            print("\n[提示] 所有样本均求解成功，无失败片段可播放。")
            return
        play_indices = sorted(play_set)
        # focus-failed 模式自动降速
        effective_speed = speed * 0.3
        mode_name = "失败片段模式 (focus-failed)"
    else:
        play_indices = list(range(total))
        segments = None
        effective_speed = speed
        mode_name = "全量模式"

    # ---- 统计 txt 中失败但 CuRobo 成功的 case ----
    txt_failed_curobo_success = []
    for i in range(total):
        if statuses[i] in ("solver_failed", "large_joint_change") and results_success[i]:
            txt_failed_curobo_success.append(i)
    txt_failed_curobo_success_set = set(txt_failed_curobo_success)

    print(f"\n{'='*70}")
    print(f"  播放模式: {mode_name}")
    print(f"  发布话题: {topic}")
    print(f"  回放速率: {effective_speed:.2f}x | 循环: {loop}")
    if focus_failed:
        print(f"  失败片段数: {len(segments)}, 总播放帧数: {len(play_indices)}/{total}")
        print(f"  上下文帧数: 前后各 {context_size} 帧")
    print(f"  成功样本: 发布 CuRobo 求解关节角")
    print(f"  失败样本: 发布原始 seed 关节角 (终端标注 [SEED])")
    if txt_failed_curobo_success:
        print(f"  ★ txt标记失败但CuRobo成功: {len(txt_failed_curobo_success)} 条 (终端高亮 ★)")
    print(f"  目标位姿同步发布到: /ik_target_pose")
    print(f"  按 Ctrl+C 停止")
    print(f"{'='*70}\n")

    rate_hz = 50
    ros_rate = rospy.Rate(rate_hz)

    # ANSI 颜色码
    COLOR_RED = "\033[91m"      # IK 失败
    COLOR_GREEN = "\033[92m"    # IK 成功
    COLOR_YELLOW = "\033[93m"   # txt失败但CuRobo成功 (重点提醒)
    COLOR_CYAN = "\033[96m"     # 片段分隔
    COLOR_RESET = "\033[0m"

    try:
        while not rospy.is_shutdown():
            current_segment_idx = 0

            for play_pos, i in enumerate(play_indices):
                if rospy.is_shutdown():
                    break

                # focus-failed 模式下，打印片段分隔线
                if focus_failed and segments:
                    seg = segments[current_segment_idx]
                    if i == seg["start"]:
                        n_failed_in_seg = len(seg["failed_indices"])
                        print(
                            f"\n{COLOR_CYAN}{'─'*70}\n"
                            f"  ▶ 片段 {current_segment_idx + 1}/{len(segments)}  "
                            f"帧范围 [{seg['start']+1}~{seg['end']+1}]  "
                            f"含 {n_failed_in_seg} 个失败帧\n"
                            f"{'─'*70}{COLOR_RESET}"
                        )
                    if i > seg["end"] and current_segment_idx + 1 < len(segments):
                        current_segment_idx += 1

                # 确定要发布的关节角
                if results_success[i] and results_joint_solutions[i] is not None:
                    joint_values = results_joint_solutions[i].tolist()
                    source_tag = "IK"
                else:
                    joint_values = seed_joints[i].tolist()
                    source_tag = "SEED"

                # 构造 JointState 消息
                js_msg = JointState()
                js_msg.header = Header()
                js_msg.header.stamp = rospy.Time.now()
                js_msg.name = JAKA_JOINT_NAMES
                js_msg.position = joint_values

                # 构造目标位姿消息
                pose_msg = PoseStamped()
                pose_msg.header.stamp = js_msg.header.stamp
                pose_msg.header.frame_id = "LINK_BASE"
                pose_msg.pose.position.x = float(positions[i][0])
                pose_msg.pose.position.y = float(positions[i][1])
                pose_msg.pose.position.z = float(positions[i][2])
                pose_msg.pose.orientation.w = float(quaternions[i][0])
                pose_msg.pose.orientation.x = float(quaternions[i][1])
                pose_msg.pose.orientation.y = float(quaternions[i][2])
                pose_msg.pose.orientation.z = float(quaternions[i][3])

                # 发布
                joint_pub.publish(js_msg)
                pose_pub.publish(pose_msg)

                # ---- 终端打印 ----
                status_str = statuses[i]
                joint_str = ", ".join(f"{v:.3f}" for v in joint_values)

                # 判断是否为 "txt失败但CuRobo成功" 的重点 case
                is_highlight = (i in txt_failed_curobo_success_set)

                if is_highlight:
                    # 重点提醒：txt 标记失败但 CuRobo 求解成功
                    color = COLOR_YELLOW
                    prefix = "★ TXT失败/CuRobo成功"
                    success_mark = "✓★"
                elif not results_success[i]:
                    color = COLOR_RED
                    prefix = ""
                    success_mark = "✗"
                else:
                    color = COLOR_GREEN
                    prefix = ""
                    success_mark = "✓"

                line = (
                    f"  [{i+1:4d}/{total}] [{source_tag:4s}] {success_mark} "
                    f"状态={status_str:20s} "
                    f"关节=[{joint_str}]"
                )
                if prefix:
                    line += f"  {prefix}"

                print(f"{color}{line}{COLOR_RESET}")

                # ---- 控制回放速度 ----
                if play_pos + 1 < len(play_indices):
                    next_i = play_indices[play_pos + 1]
                    dt = timestamps[next_i] - timestamps[i]
                    if dt > 0:
                        sleep_time = dt / effective_speed
                        sleep_time = min(sleep_time, 2.0)
                        rospy.sleep(sleep_time)
                    else:
                        ros_rate.sleep()
                else:
                    ros_rate.sleep()

            if not loop:
                print("\n回放完成。")
                break
            else:
                print(f"\n--- 循环回放，重新开始 ---\n")

    except rospy.ROSInterruptException:
        print("\nROS 节点被中断。")
    except KeyboardInterrupt:
        print("\n用户中断，停止发布。")


if __name__ == "__main__":
    main()
