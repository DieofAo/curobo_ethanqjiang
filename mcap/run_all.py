#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键完成：mcap数据提取 -> CuRobo IK求解(按机器人配置是否启用碰撞检测) -> ROS发布关节角

使用方法:
  1. 启动 roscore
  2. 启动 rviz 加载机械臂模型
  3. 运行: conda run -n curobo python3 run_all.py [--robot marvin|jaka|both] [--stream] [--loop] [--mcap xxx.mcap]

模式:
  批处理模式(默认): 先提取全部pose -> 批量IK求解 -> 发布 (支持缓存)
  流式模式(--stream): 边读mcap边IK求解边发布，无需等待全部加载完成

Marvin pose_command 是 gripper_link 相对于 robot_stand 的位姿。
JAKA pose_command 是 gripper_link 相对于 LINK_BASE 的位姿。
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from mcap_ros1.reader import read_ros1_messages

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
DEFAULT_MCAP = os.path.join(SCRIPT_DIR, "20260130-154646_optimized.mcap")
TOPIC = "/robot/data/jaka_arm_left/action"
NUM_SEEDS = 4
RETURN_SEEDS = NUM_SEEDS
LOCAL_SEED_NOISE_STD = 0.05
JOINT_DISTANCE_METRIC = "l2"  # "l1" = sum(abs(dq)), "l2" = norm(dq), "weighted_l2" = sqrt(sum(w*dq^2))
JOINT_DISTANCE_WEIGHTS = None
POSITION_THRESHOLD = 0.005  # 5mm
ROTATION_THRESHOLD = 0.05   # rad
BATCH_SIZE = 64

# 对mcap中读取的pose左乘一个纯平移变换 (无旋转)
# T_new = T_offset * T_original => pos_new = pos_original + offset (因为R_offset=I)
MARVIN_POSE_OFFSET = np.array([0.20, 0.0, 0.130], dtype=np.float64)
JAKA_POSE_OFFSET = np.array([0.0, 0.0, 0], dtype=np.float64)

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
ROBOT_PROFILES = {
    "marvin": {
        "display_name": "Marvin 左臂",
        "robot_config_file": "marvin_left_arm.yml",
        "base_link": "robot_stand",
        "ee_link": "gripper_link",
        "joint_names": [
            "Joint1_L", "Joint2_L", "Joint3_L", "Joint4_L",
            "Joint5_L", "Joint6_L", "Joint7_L",
        ],
        "pose_offset": MARVIN_POSE_OFFSET,
        "joint_state_prefix": "marvin_",
        "node_name": "marvin_left_arm_ik_publisher",
        "self_collision_check": False,
        "self_collision_opt": False,
    },
    "jaka": {
        "display_name": "JAKA 左臂",
        "robot_config_file": "jaka.yml",
        "urdf_path": os.path.join(REPO_ROOT, "jaka", "left_jaka.urdf"),
        "asset_root_path": os.path.join(REPO_ROOT, "jaka"),
        "base_link": "LINK_BASE",
        "ee_link": "gripper_link",
        "joint_names": ["J_1", "J_2", "J_3", "J_4", "J_5", "J_6", "J_7"],
        "pose_offset": JAKA_POSE_OFFSET,
        "joint_state_prefix": "jaka_",
        "node_name": "jaka_left_arm_ik_publisher",
        "self_collision_check": False,
        "self_collision_opt": False,
    },
}

# 流式模式的缓冲区大小（攒够多少帧后批量求解一次）
STREAM_BUFFER_SIZE = 32

# 运行时由main()设置
MCAP_PATH = DEFAULT_MCAP

# 缓存文件路径（避免重复计算，按mcap文件名区分）
def _cache_safe(text):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text).strip("_")


def get_pose_cache_path(mcap_path):
    """根据mcap文件名生成原始pose缓存路径。pose_offset在IK前按机器人单独应用。"""
    basename = os.path.splitext(os.path.basename(mcap_path))[0]
    topic_tag = _cache_safe(TOPIC)
    return os.path.join(SCRIPT_DIR, f"pose_cache_raw_{topic_tag}_{basename}.npz")


def get_ik_cache_path(mcap_path, robot_name, profile):
    """根据mcap文件名和机器人配置生成IK缓存路径。"""
    basename = os.path.splitext(os.path.basename(mcap_path))[0]
    topic_tag = _cache_safe(TOPIC)
    collision_tag = "collision" if profile["self_collision_check"] else "nocollision"
    offset_tag = "_".join(f"{v:.3f}" for v in profile["pose_offset"])
    return os.path.join(
        SCRIPT_DIR,
        f"ik_cache_{robot_name}_{topic_tag}_{collision_tag}_prevseed_nearest_"
        f"seeds{NUM_SEEDS}_offset_{offset_tag}_{basename}.npz",
    )


def load_robot_config(profile, tensor_args):
    """加载并按当前 profile 覆盖 base/ee/URDF 设置。"""
    cfg_dict = load_yaml(join_path(get_robot_configs_path(), profile["robot_config_file"]))
    kinematics = cfg_dict["robot_cfg"]["kinematics"]
    kinematics["base_link"] = profile["base_link"]
    kinematics["ee_link"] = profile["ee_link"]
    if "urdf_path" in profile:
        kinematics["urdf_path"] = profile["urdf_path"]
    if "asset_root_path" in profile:
        kinematics["asset_root_path"] = profile["asset_root_path"]
    return RobotConfig.from_dict(cfg_dict["robot_cfg"], tensor_args=tensor_args)


# ============================================================
# IK求解器初始化（共用）
# ============================================================
def create_ik_solver(profile):
    """创建CuRobo IK求解器"""
    print(f"  初始化CuRobo IK求解器 (自碰撞检测: {profile['self_collision_check']})...")
    tensor_args = TensorDeviceType()
    robot_cfg = load_robot_config(profile, tensor_args)

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg, None,
        rotation_threshold=ROTATION_THRESHOLD,
        position_threshold=POSITION_THRESHOLD,
        num_seeds=NUM_SEEDS,
        self_collision_check=profile["self_collision_check"],
        self_collision_opt=profile["self_collision_opt"],
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    print(f"  关节: {ik_solver.joint_names}")
    print(f"  DOF: {ik_solver.dof}")
    return ik_solver, tensor_args


def select_continuous_solution(
    result,
    prev_solution,
    batch_index=0,
    metric=JOINT_DISTANCE_METRIC,
    weights=JOINT_DISTANCE_WEIGHTS,
):
    """从IK返回候选中选一个解：有上一帧时选关节距离最近的成功解。"""
    candidates = result.solution[batch_index]
    success = result.success[batch_index].bool()

    if success.any():
        success_indices = torch.where(success)[0]
        success_solutions = candidates[success_indices]

        if prev_solution is None:
            selected_idx = success_indices[0]
        else:
            diff = success_solutions - prev_solution.unsqueeze(0)
            if metric == "weighted_l2":
                if weights is None:
                    w = torch.ones(diff.shape[-1], device=diff.device, dtype=diff.dtype)
                else:
                    w = torch.as_tensor(weights, device=diff.device, dtype=diff.dtype)
                distances = torch.sqrt(torch.sum(w.unsqueeze(0) * diff * diff, dim=-1))
            elif metric == "l2":
                distances = torch.linalg.norm(diff, dim=-1)
            else:
                distances = torch.sum(torch.abs(diff), dim=-1)
            selected_idx = success_indices[torch.argmin(distances)]

        return candidates[selected_idx], True

    return candidates[0], False


def make_local_seed_config(prev_solution, num_seeds=NUM_SEEDS, noise_std=LOCAL_SEED_NOISE_STD):
    """以上一帧成功解为中心构造多个IK初值，形状为 (batch=1, num_seeds, dof)。"""
    if prev_solution is None:
        return None

    seeds = prev_solution.unsqueeze(0).repeat(num_seeds, 1)
    if num_seeds > 1 and noise_std > 0.0:
        seeds[1:] += torch.randn_like(seeds[1:]) * noise_std
    return seeds.unsqueeze(0)


def sleep_for_mcap_dt(dt_s, speed=1.0):
    """按MCAP时间间隔sleep，speed>1加速播放。"""
    if dt_s > 0 and dt_s < 1.0:
        rospy.sleep(dt_s / speed)


# ============================================================
# 流式模式：边读边解边发布
# ============================================================
def run_stream_mode(mcap_path, profile, buffer_size=32, loop=False, speed=1.0):
    """流式模式：从mcap逐步读取pose，批量IK求解，实时发布"""
    print(f"\n  [流式模式] 边读取 -> 边IK求解 -> 边发布")
    print(f"  缓冲区大小: {buffer_size} 帧")
    print(f"  播放速度: {speed}x")
    print(f"  循环播放: {loop}")

    # 初始化ROS
    rospy.init_node(profile["node_name"], anonymous=True)
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rospy.sleep(0.5)

    # 初始化IK求解器
    ik_solver, tensor_args = create_ik_solver(profile)

    # 状态变量
    prev_solution = None
    prev_timestamp_ns = None
    total_count = 0
    success_count = 0
    loop_count = 0
    t_start = time.time()

    # 缓冲区
    buf_positions = []
    buf_quaternions_xyzw = []
    buf_timestamps_ns = []

    def flush_buffer():
        """处理缓冲区中的数据：IK求解 + 发布"""
        nonlocal prev_solution, prev_timestamp_ns, total_count, success_count

        if not buf_positions:
            return

        n = len(buf_positions)
        positions = np.array(buf_positions, dtype=np.float32)
        quats_xyzw = np.array(buf_quaternions_xyzw, dtype=np.float64)
        timestamps = buf_timestamps_ns[:]

        # 左乘平移变换: z偏移-0.13
        positions += profile["pose_offset"].astype(np.float32)

        # 转换四元数: xyzw -> wxyz
        quats_wxyz = np.zeros_like(quats_xyzw, dtype=np.float32)
        quats_wxyz[:, 0] = quats_xyzw[:, 3]  # w
        quats_wxyz[:, 1] = quats_xyzw[:, 0]  # x
        quats_wxyz[:, 2] = quats_xyzw[:, 1]  # y
        quats_wxyz[:, 3] = quats_xyzw[:, 2]  # z

        pos_tensor = torch.from_numpy(positions).to(tensor_args.device)
        quat_tensor = torch.from_numpy(quats_wxyz).to(tensor_args.device)

        # 逐帧发布
        for k in range(n):
            if rospy.is_shutdown():
                return

            goal = Pose(pos_tensor[k:k + 1], quat_tensor[k:k + 1])
            seed_config = make_local_seed_config(prev_solution)
            result = ik_solver.solve_batch(
                goal,
                seed_config=seed_config,
                return_seeds=RETURN_SEEDS,
                num_seeds=NUM_SEEDS,
            )
            solution, frame_success = select_continuous_solution(
                result,
                prev_solution,
            )
            if frame_success:
                prev_solution = solution.clone()

            solution_np = solution.cpu().numpy()

            total_count += 1
            if frame_success:
                success_count += 1

            # 按时间间隔等待
            ts_ns = timestamps[k]
            if prev_timestamp_ns is not None:
                dt_s = (ts_ns - prev_timestamp_ns) / 1e9
                sleep_for_mcap_dt(dt_s, speed)
            prev_timestamp_ns = ts_ns

            # 发布
            js = JointState()
            js.header = Header()
            js.header.stamp = rospy.Time.now()
            js.name = profile["joint_names"]
            js.position = solution_np.tolist()
            js.velocity = []
            js.effort = []
            pub.publish(js)

        if total_count % 200 < buffer_size:
            elapsed = time.time() - t_start
            print(f"    已处理 {total_count} 帧, 成功率 {100.0*success_count/total_count:.1f}%, 耗时 {elapsed:.1f}s")

        # 清空缓冲区
        buf_positions.clear()
        buf_quaternions_xyzw.clear()
        buf_timestamps_ns.clear()

    # 流式读取mcap
    print(f"\n  开始流式读取: {mcap_path}")
    print(f"  按 Ctrl+C 停止\n")

    try:
        while not rospy.is_shutdown():
            loop_count += 1
            prev_timestamp_ns = None
            print(f"  播放第 {loop_count} 轮")

            for msg in read_ros1_messages(mcap_path, topics=[TOPIC]):
                if rospy.is_shutdown():
                    break

                decoded_msg = msg.ros_msg
                p = decoded_msg.pose_command.pose.position
                o = decoded_msg.pose_command.pose.orientation

                lt = msg.log_time
                if hasattr(lt, 'timestamp'):
                    ts_ns = int(lt.timestamp() * 1e9)
                else:
                    ts_ns = int(lt)

                buf_positions.append([p.x, p.y, p.z])
                buf_quaternions_xyzw.append([o.x, o.y, o.z, o.w])
                buf_timestamps_ns.append(ts_ns)

                # 缓冲区满，批量处理
                if len(buf_positions) >= buffer_size:
                    flush_buffer()

            # 处理剩余数据
            if buf_positions and not rospy.is_shutdown():
                flush_buffer()

            if not loop:
                break
            if not rospy.is_shutdown():
                print(f"\n  --- 循环播放，重新开始 ---\n")

    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - t_start
    print(f"\n  流式处理完成!")
    print(f"  总帧数: {total_count}, 成功率: {100.0*success_count/max(total_count,1):.1f}%")
    print(f"  总耗时: {elapsed:.1f}s")


def solve_stream_frame(ik_solver, tensor_args, profile, position, quat_xyzw, prev_solution):
    """流式求解单帧IK。失败时返回上一帧成功解，避免对不可达点插值。"""
    position = np.array(position, dtype=np.float32) + profile["pose_offset"].astype(np.float32)
    quat_xyzw = np.array(quat_xyzw, dtype=np.float32)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

    pos_tensor = torch.from_numpy(position.reshape(1, 3)).to(tensor_args.device)
    quat_tensor = torch.from_numpy(quat_wxyz.reshape(1, 4)).to(tensor_args.device)
    goal = Pose(pos_tensor, quat_tensor)

    seed_config = make_local_seed_config(prev_solution)
    result = ik_solver.solve_batch(
        goal,
        seed_config=seed_config,
        return_seeds=RETURN_SEEDS,
        num_seeds=NUM_SEEDS,
    )
    solution, success = select_continuous_solution(result, prev_solution)

    if success:
        return solution, solution, True
    if prev_solution is not None:
        return prev_solution, prev_solution, False
    return solution, prev_solution, False


def publish_dual_frame(pub, robot_states, stamp=None):
    """发布双臂当前帧。robot_states: [(profile, solution_tensor), ...]"""
    joint_names = []
    positions = []
    for profile, solution in robot_states:
        joint_names.extend(get_publish_joint_names(profile, prefix=True))
        positions.extend(solution.cpu().numpy().tolist())

    js = JointState()
    js.header = Header()
    js.header.stamp = stamp if stamp is not None else rospy.Time.now()
    js.name = joint_names
    js.position = positions
    js.velocity = []
    js.effort = []
    pub.publish(js)


def run_dual_stream_mode(mcap_path, robot_names, loop=False, speed=1.0):
    """双臂流式模式：逐帧IK并发布，失败帧保持上一帧成功解，不做插值。"""
    print(f"\n  [双臂流式模式] 边读取 -> 双IK求解 -> 合并发布")
    print(f"  播放速度: {speed}x")
    print(f"  循环播放: {loop}")
    print(f"  失败帧策略: 保持该机器人上一帧成功IK解，不插值不可达位置")

    rospy.init_node("marvin_jaka_dual_ik_publisher", anonymous=True)
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rospy.sleep(0.5)

    solvers = []
    for robot_name in robot_names:
        profile = ROBOT_PROFILES[robot_name]
        ik_solver, tensor_args = create_ik_solver(profile)
        solvers.append({
            "name": robot_name,
            "profile": profile,
            "ik_solver": ik_solver,
            "tensor_args": tensor_args,
            "prev_solution": None,
            "has_published": False,
            "success_count": 0,
            "publish_count": 0,
            "hold_count": 0,
            "skip_count": 0,
        })

    prev_timestamp_ns = None
    total_count = 0
    loop_count = 0
    t_start = time.time()

    try:
        while not rospy.is_shutdown():
            loop_count += 1
            prev_timestamp_ns = None
            if loop_count > 1:
                for state in solvers:
                    state["prev_solution"] = None
                    state["has_published"] = False
            print(f"  播放第 {loop_count} 轮")

            for msg in read_ros1_messages(mcap_path, topics=[TOPIC]):
                if rospy.is_shutdown():
                    break

                decoded_msg = msg.ros_msg
                p = decoded_msg.pose_command.pose.position
                o = decoded_msg.pose_command.pose.orientation
                position = [p.x, p.y, p.z]
                quat_xyzw = [o.x, o.y, o.z, o.w]

                lt = msg.log_time
                if hasattr(lt, 'timestamp'):
                    ts_ns = int(lt.timestamp() * 1e9)
                else:
                    ts_ns = int(lt)

                if prev_timestamp_ns is not None:
                    dt_s = (ts_ns - prev_timestamp_ns) / 1e9
                    sleep_for_mcap_dt(dt_s, speed)
                prev_timestamp_ns = ts_ns

                frame_states = []
                for state in solvers:
                    publish_solution, next_prev, success = solve_stream_frame(
                        state["ik_solver"],
                        state["tensor_args"],
                        state["profile"],
                        position,
                        quat_xyzw,
                        state["prev_solution"],
                    )
                    if success:
                        state["prev_solution"] = next_prev
                        state["has_published"] = True
                        state["success_count"] += 1
                    elif not state["has_published"]:
                        state["skip_count"] += 1
                        frame_states = []
                        break
                    else:
                        state["hold_count"] += 1

                    frame_states.append((state["profile"], publish_solution))

                total_count += 1
                if frame_states:
                    publish_dual_frame(pub, frame_states)
                    for state in solvers:
                        state["publish_count"] += 1

                if total_count % 200 == 0:
                    elapsed = time.time() - t_start
                    stats = ", ".join(
                        f"{s['name']} 成功 {100.0*s['success_count']/max(s['publish_count'],1):.1f}%"
                        f" 保持{s['hold_count']} 跳过{s['skip_count']}"
                        for s in solvers
                    )
                    print(f"    已处理 {total_count} 帧, {stats}, 耗时 {elapsed:.1f}s")

            if not loop:
                break
            if not rospy.is_shutdown():
                print(f"\n  --- 循环播放，重新开始 ---\n")

    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - t_start
    print(f"\n  双臂流式处理完成!")
    for state in solvers:
        print(
            f"  {state['name']}: 发布帧 {state['publish_count']}, "
            f"成功 {state['success_count']}, 保持 {state['hold_count']}, 跳过 {state['skip_count']}"
        )
    print(f"  总帧数: {total_count}, 总耗时: {elapsed:.1f}s")


# ============================================================
# Step 1: 从mcap提取pose数据（批处理模式）
# ============================================================
def extract_pose_from_mcap(mcap_path, pose_cache_path):
    """从mcap文件提取原始pose_command数据，有缓存则直接加载"""
    if os.path.exists(pose_cache_path):
        print(f"  [缓存] 加载已提取的pose数据: {pose_cache_path}")
        data = np.load(pose_cache_path)
        return data['timestamps_ns'], data['positions'], data['quaternions_xyzw']

    print(f"  读取mcap文件: {mcap_path}")
    print(f"  话题: {TOPIC}")

    timestamps_ns = []
    positions = []
    quaternions_xyzw = []

    count = 0
    for msg in read_ros1_messages(mcap_path, topics=[TOPIC]):
        decoded_msg = msg.ros_msg
        p = decoded_msg.pose_command.pose.position
        o = decoded_msg.pose_command.pose.orientation

        lt = msg.log_time
        if hasattr(lt, 'timestamp'):
            timestamps_ns.append(int(lt.timestamp() * 1e9))
        else:
            timestamps_ns.append(int(lt))
        positions.append([p.x, p.y, p.z])
        quaternions_xyzw.append([o.x, o.y, o.z, o.w])

        count += 1
        if count % 1000 == 0:
            print(f"    已读取 {count} 条消息...")

    print(f"  总共提取 {count} 条pose消息")

    timestamps_ns = np.array(timestamps_ns, dtype=np.int64)
    positions = np.array(positions, dtype=np.float64)
    quaternions_xyzw = np.array(quaternions_xyzw, dtype=np.float64)

    # 缓存
    np.savez_compressed(pose_cache_path,
        timestamps_ns=timestamps_ns, positions=positions,
        quaternions_xyzw=quaternions_xyzw)
    print(f"  已缓存到: {pose_cache_path}")

    return timestamps_ns, positions, quaternions_xyzw


# ============================================================
# Step 2: CuRobo IK求解（批处理模式）
# ============================================================
def solve_ik(positions, quaternions_xyzw, timestamps_ns, ik_cache_path, profile):
    """使用CuRobo进行IK求解（按profile决定是否碰撞检测），有缓存则直接加载"""
    if os.path.exists(ik_cache_path):
        print(f"  [缓存] 加载已计算的IK解: {ik_cache_path}")
        ik_data = np.load(ik_cache_path)
        solutions = ik_data['solutions']
        success = ik_data['success']
        n_success = int(success.sum())
        N = len(solutions)
        print(f"  成功率: {n_success}/{N} ({100.0*n_success/N:.1f}%)")
        return solutions, success

    N = len(positions)
    solve_positions = positions.astype(np.float32) + profile["pose_offset"].astype(np.float32)

    # 转换四元数: xyzw -> wxyz (CuRobo格式)
    quaternions_wxyz = np.zeros_like(quaternions_xyzw)
    quaternions_wxyz[:, 0] = quaternions_xyzw[:, 3]  # w
    quaternions_wxyz[:, 1] = quaternions_xyzw[:, 0]  # x
    quaternions_wxyz[:, 2] = quaternions_xyzw[:, 1]  # y
    quaternions_wxyz[:, 3] = quaternions_xyzw[:, 2]  # z

    # 初始化IK求解器
    ik_solver, tensor_args = create_ik_solver(profile)

    # 小批量读取张量，逐帧IK；每帧以上一帧成功解附近的多个seed搜索。
    print(f"  开始IK求解 (chunk_size={BATCH_SIZE}, 上一帧附近多seed搜索, 共{N}帧)...")
    pos_tensor = torch.from_numpy(solve_positions).to(tensor_args.device)
    quat_tensor = torch.from_numpy(quaternions_wxyz.astype(np.float32)).to(tensor_args.device)

    all_solutions = np.zeros((N, len(profile["joint_names"])), dtype=np.float32)
    all_success = np.zeros(N, dtype=bool)
    prev_solution = None

    t_start = time.time()
    for i in range(0, N, BATCH_SIZE):
        j = min(i + BATCH_SIZE, N)
        n_s = 0
        for k in range(i, j):
            goal = Pose(pos_tensor[k:k + 1], quat_tensor[k:k + 1])
            seed_config = make_local_seed_config(prev_solution)
            result = ik_solver.solve_batch(
                goal,
                seed_config=seed_config,
                return_seeds=RETURN_SEEDS,
                num_seeds=NUM_SEEDS,
            )
            solution, frame_success = select_continuous_solution(
                result,
                prev_solution,
            )

            all_solutions[k] = solution.cpu().numpy()
            all_success[k] = frame_success
            if frame_success:
                prev_solution = solution.clone()

            if frame_success:
                n_s += 1

        print(f"    帧 [{i}:{j}] 成功: {n_s}/{j - i}")

    torch.cuda.synchronize()
    t_elapsed = time.time() - t_start

    n_success = int(all_success.sum())
    avg_ms = 1000.0 * t_elapsed / max(N, 1)
    print(
        f"  IK求解完成: {n_success}/{N} ({100.0*n_success/N:.1f}%), "
        f"耗时{t_elapsed:.2f}s, 平均{avg_ms:.2f}ms/帧"
    )

    # 缓存
    np.savez_compressed(ik_cache_path,
        solutions=all_solutions, success=all_success, timestamps_ns=timestamps_ns)
    print(f"  已缓存到: {ik_cache_path}")

    return all_solutions, all_success


def smooth_solutions(solutions, success, max_joint_jump=0.5):
    """平滑处理：插值失败帧 + 修复关节角跳变"""
    N = len(solutions)
    result = solutions.copy()

    success_indices = np.where(success)[0]
    if len(success_indices) == 0:
        print("  警告: 所有帧IK求解均失败!")
        return result

    # 对失败帧进行线性插值
    n_failed = N - int(success.sum())
    if n_failed > 0:
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
        print(f"  已插值 {n_failed} 个失败帧")

    # 修复关节角跳变
    jumps = np.abs(np.diff(result, axis=0))
    max_jumps = jumps.max(axis=1)
    n_jumps = (max_jumps > max_joint_jump).sum()
    if n_jumps > 0:
        print(f"  修复 {n_jumps} 帧关节角跳变 (>{max_joint_jump:.2f} rad)")
        for i in range(1, N):
            if np.max(np.abs(result[i] - result[i-1])) > max_joint_jump:
                result[i] = result[i-1]

    return result


# ============================================================
# Step 3: ROS发布（批处理模式）
# ============================================================
def publish_joint_states(solutions, timestamps_ns, profile, loop=False, speed=1.0):
    """通过ROS /joint_states话题按原始时间间隔发布关节角"""
    N = len(solutions)

    rospy.init_node(profile["node_name"], anonymous=True)
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rospy.sleep(0.5)  # 等待连接

    # 计算时间间隔
    dt_ns = np.diff(timestamps_ns)
    dt_s = dt_ns / 1e9

    print(f"  发布 {N} 帧, 频率约{1.0/dt_s.mean():.1f}Hz, 总时长{dt_s.sum():.1f}s")
    print(f"  播放速度: {speed}x")
    print(f"  循环播放: {loop}")
    print(f"  按 Ctrl+C 停止")

    try:
        loop_count = 0
        while not rospy.is_shutdown():
            loop_count += 1
            print(f"  播放第 {loop_count} 轮")

            for i in range(N):
                if rospy.is_shutdown():
                    break

                js = JointState()
                js.header = Header()
                js.header.stamp = rospy.Time.now()
                js.name = profile["joint_names"]
                js.position = solutions[i].tolist()
                js.velocity = []
                js.effort = []

                pub.publish(js)

                if i < N - 1:
                    sleep_time = dt_s[i]
                    sleep_for_mcap_dt(sleep_time, speed)

                if (i + 1) % 500 == 0:
                    print(f"    已发布 {i+1}/{N} 帧")

            if not loop:
                break
            if not rospy.is_shutdown():
                print(f"\n  --- 循环播放，重新开始 ---\n")

    except rospy.ROSInterruptException:
        pass

    print(f"  发布完成!")


def get_publish_joint_names(profile, prefix=False):
    """返回发布到/joint_states的关节名，双臂模式使用前缀避免URDF重名。"""
    if not prefix:
        return profile["joint_names"]
    return [profile["joint_state_prefix"] + name for name in profile["joint_names"]]


def publish_dual_joint_states(robot_results, timestamps_ns, loop=False, speed=1.0):
    """发布双臂合并后的 /joint_states。"""
    N = len(timestamps_ns)

    rospy.init_node("marvin_jaka_dual_ik_publisher", anonymous=True)
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rospy.sleep(0.5)

    dt_ns = np.diff(timestamps_ns)
    dt_s = dt_ns / 1e9 if len(dt_ns) > 0 else np.array([], dtype=np.float64)
    mean_hz = 1.0 / dt_s.mean() if len(dt_s) > 0 and dt_s.mean() > 0 else 0.0
    total_duration = dt_s.sum() if len(dt_s) > 0 else 0.0

    joint_names = []
    for _, profile, _ in robot_results:
        joint_names.extend(get_publish_joint_names(profile, prefix=True))

    print(f"  双臂发布 {N} 帧, 频率约{mean_hz:.1f}Hz, 总时长{total_duration:.1f}s")
    print(f"  发布关节数: {len(joint_names)}")
    print(f"  播放速度: {speed}x")
    print(f"  循环播放: {loop}")
    print(f"  按 Ctrl+C 停止")

    try:
        loop_count = 0
        while not rospy.is_shutdown():
            loop_count += 1
            print(f"  播放第 {loop_count} 轮")

            for i in range(N):
                if rospy.is_shutdown():
                    break

                positions = []
                for _, _, solutions in robot_results:
                    positions.extend(solutions[i].tolist())

                js = JointState()
                js.header = Header()
                js.header.stamp = rospy.Time.now()
                js.name = joint_names
                js.position = positions
                js.velocity = []
                js.effort = []
                pub.publish(js)

                if i < N - 1 and len(dt_s) > i:
                    sleep_time = dt_s[i]
                    sleep_for_mcap_dt(sleep_time, speed)

                if (i + 1) % 500 == 0:
                    print(f"    已发布 {i+1}/{N} 帧")

            if not loop:
                break
            if not rospy.is_shutdown():
                print(f"\n  --- 循环播放，重新开始 ---\n")

    except rospy.ROSInterruptException:
        pass

    print(f"  双臂发布完成!")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MCAP提取 -> CuRobo IK求解 -> ROS发布")
    robot_choices = sorted(list(ROBOT_PROFILES.keys()) + ["both"])
    parser.add_argument("--robot", choices=robot_choices, default="marvin",
                        help="选择机器人配置: marvin、jaka 或 both (默认: marvin)")
    parser.add_argument("--mcap", type=str, default=DEFAULT_MCAP,
                        help=f"mcap文件路径 (默认: {DEFAULT_MCAP})")
    parser.add_argument("--stream", action="store_true",
                        help="启用流式模式: 边读mcap边IK求解边发布，无需等待全部加载")
    parser.add_argument("--buffer-size", type=int, default=STREAM_BUFFER_SIZE,
                        help=f"流式模式缓冲区大小 (默认: {STREAM_BUFFER_SIZE})")
    parser.add_argument("--loop", action="store_true",
                        help="循环播放: 播完mcap后从头开始，直到 Ctrl+C 停止")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="播放速度倍数，按MCAP时间间隔除以该值sleep (默认: 1.0)")
    # 过滤掉ROS添加的参数（如__name:=xxx）
    args, _ = parser.parse_known_args()

    mcap_path = os.path.abspath(args.mcap)
    if not os.path.exists(mcap_path):
        print(f"错误: mcap文件不存在: {mcap_path}")
        sys.exit(1)
    if args.speed <= 0:
        print(f"错误: --speed 必须大于0，当前为 {args.speed}")
        sys.exit(1)

    selected_robot_names = ["marvin", "jaka"] if args.robot == "both" else [args.robot]
    for robot_name in selected_robot_names:
        profile = ROBOT_PROFILES[robot_name]
        if "urdf_path" in profile and not os.path.exists(profile["urdf_path"]):
            print(f"错误: URDF文件不存在: {profile['urdf_path']}")
            sys.exit(1)

    profile = ROBOT_PROFILES[selected_robot_names[0]]

    print("=" * 60)
    title = "Marvin + JAKA 双臂" if args.robot == "both" else profile["display_name"]
    print(f"  {title}: MCAP提取 -> IK求解 -> ROS发布")
    print("=" * 60)
    print(f"  MCAP文件: {mcap_path}")
    print(f"  机器人: {args.robot}")
    print(f"  播放速度: {args.speed}x")
    for robot_name in selected_robot_names:
        p = ROBOT_PROFILES[robot_name]
        print(f"  [{robot_name}] IK frame: {p['base_link']} -> {p['ee_link']}")
        print(f"  [{robot_name}] 关节: {p['joint_names']}")
        print(f"  [{robot_name}] pose offset: {p['pose_offset'].tolist()}")
        print(f"  [{robot_name}] 自碰撞检测: {p['self_collision_check']}")
    print(f"  循环播放: {args.loop}")

    if args.stream:
        # ========== 流式模式 ==========
        stream_buffer = args.buffer_size
        print(f"  模式: 流式 (buffer={stream_buffer})")
        if args.robot == "both":
            run_dual_stream_mode(mcap_path, selected_robot_names, loop=args.loop, speed=args.speed)
        else:
            run_stream_mode(mcap_path, profile, stream_buffer, loop=args.loop, speed=args.speed)
    else:
        # ========== 批处理模式 ==========
        print(f"  模式: 批处理 (支持缓存)")
        pose_cache_path = get_pose_cache_path(mcap_path)

        # Step 1
        print(f"\n{'='*20} Step 1: 提取pose数据 {'='*20}")
        timestamps_ns, positions, quaternions_xyzw = extract_pose_from_mcap(mcap_path, pose_cache_path)
        N = len(timestamps_ns)
        print(f"  总帧数: {N}")

        robot_results = []
        for robot_name in selected_robot_names:
            p = ROBOT_PROFILES[robot_name]
            ik_cache_path = get_ik_cache_path(mcap_path, robot_name, p)

            print(f"\n{'='*20} Step 2: {robot_name} IK求解 {'='*20}")
            solutions, success = solve_ik(positions, quaternions_xyzw, timestamps_ns, ik_cache_path, p)

            if args.robot == "both":
                print(f"\n  {robot_name} 双臂模式: 跳过失败帧插值，避免补不可达IK点")
            else:
                print(f"\n  {robot_name} 平滑处理...")
                solutions = smooth_solutions(solutions, success)
            robot_results.append((robot_name, p, solutions))

        print(f"\n{'='*20} Step 3: ROS发布 {'='*20}")
        if args.robot == "both":
            publish_dual_joint_states(robot_results, timestamps_ns, loop=args.loop, speed=args.speed)
        else:
            _, p, solutions = robot_results[0]
            publish_joint_states(solutions, timestamps_ns, p, loop=args.loop, speed=args.speed)

    print(f"\n{'='*60}")
    print(f"  全部完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
