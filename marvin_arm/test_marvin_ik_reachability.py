#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Marvin 左臂 IK 可达性测试

测试条件:
  - 位置: x=0, y=0, z ∈ [0.2, 0.776] (相对 Base_L)
  - 旋转: 固定为 [0.500, -0.500, 0.500, -0.500] (w,x,y,z)
  - 运动链: Base_L → Link7_L (7-DOF)
  - 使用 CuRobo IK 求解器
  - 统计操纵度 (Yoshikawa manipulability)

用法:
  python test_marvin_ik_reachability.py
  python test_marvin_ik_reachability.py --n-pos 20 --num-seeds 32
  python test_marvin_ik_reachability.py --self-collision
"""

import argparse
import time

import numpy as np
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============ 操纵度计算工具函数 ============

def quat_wxyz_to_z_axis(q: torch.Tensor) -> torch.Tensor:
    """从 (w,x,y,z) 四元数提取局部 z 轴在世界系下的方向向量。
    q: [..., 4], 返回 [..., 3]"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    zx = 2.0 * (x * z + w * y)
    zy = 2.0 * (y * z - w * x)
    zz = 1.0 - 2.0 * (x * x + y * y)
    return torch.stack([zx, zy, zz], dim=-1)


def compute_geometric_jacobian_batch(
    link_positions: torch.Tensor,
    link_quaternions: torch.Tensor,
    ee_position: torch.Tensor,
    joint_link_indices: list,
) -> torch.Tensor:
    """
    几何法计算 6×DOF 雅可比矩阵 (全部在 GPU 上批量计算)。

    Args:
        link_positions: [B, n_links, 3] 各 link 在世界坐标系下的位置
        link_quaternions: [B, n_links, 4] 各 link 四元数 (w,x,y,z)
        ee_position: [B, 3] 末端执行器位置
        joint_link_indices: 长度为 dof 的列表, 每个元素是该关节对应 link 的索引

    Returns:
        J: [B, 6, dof] 几何雅可比矩阵
    """
    B = ee_position.shape[0]
    dof = len(joint_link_indices)
    device = ee_position.device

    J = torch.zeros(B, 6, dof, device=device, dtype=ee_position.dtype)

    for i, li in enumerate(joint_link_indices):
        z_i = quat_wxyz_to_z_axis(link_quaternions[:, li, :])  # [B, 3]
        p_i = link_positions[:, li, :]  # [B, 3]
        r = ee_position - p_i  # [B, 3]
        # 线速度: z_i × r
        J[:, 0, i] = z_i[:, 1] * r[:, 2] - z_i[:, 2] * r[:, 1]
        J[:, 1, i] = z_i[:, 2] * r[:, 0] - z_i[:, 0] * r[:, 2]
        J[:, 2, i] = z_i[:, 0] * r[:, 1] - z_i[:, 1] * r[:, 0]
        # 角速度: z_i
        J[:, 3, i] = z_i[:, 0]
        J[:, 4, i] = z_i[:, 1]
        J[:, 5, i] = z_i[:, 2]

    return J


def compute_manipulability(J: torch.Tensor) -> torch.Tensor:
    """计算 Yoshikawa 操纵度 w = sqrt(det(J·Jᵀ))。
    J: [B, 6, dof], 返回 [B]。
    w 越大表示离奇异越远; w → 0 表示接近奇异。"""
    JJT = J @ J.transpose(-1, -2)  # [B, 6, 6]
    det_val = torch.linalg.det(JJT)  # [B]
    # det 可能因数值误差为负, clamp 到 0
    return det_val.clamp_min(0.0).sqrt()


# ============ 随机旋转采样 ============

def random_quaternions(n: int, seed: int = 42) -> np.ndarray:
    """在 SO(3) 上均匀采样 n 个四元数 (w, x, y, z)。
    使用 Shoemake 方法。"""
    rng = np.random.default_rng(seed)
    u1 = rng.random(n)
    u2 = rng.random(n) * 2 * np.pi
    u3 = rng.random(n) * 2 * np.pi

    sqrt_u1 = np.sqrt(u1)
    sqrt_1mu1 = np.sqrt(1.0 - u1)

    w = sqrt_1mu1 * np.sin(u2)
    x = sqrt_1mu1 * np.cos(u2)
    y = sqrt_u1 * np.sin(u3)
    z = sqrt_u1 * np.cos(u3)

    # 归一化 (理论上已经是单位四元数, 但数值安全)
    quats = np.stack([w, x, y, z], axis=-1).astype(np.float32)
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / norms
    return quats


def main():
    ap = argparse.ArgumentParser(description="Marvin 左臂 IK 可达性测试")
    ap.add_argument("--n-pos", type=int, default=13,
                    help="z 方向采样点数 (在 [0.2, 0.8] 均匀分布)")
    ap.add_argument("--n-rot", type=int, default=1,
                    help="每个位置的重复次数 (旋转固定, 此参数仅控制重复验证次数)")
    ap.add_argument("--num-seeds", type=int, default=32,
                    help="IK 求解器 seed 数量 (越多越容易找到解)")
    ap.add_argument("--self-collision", action="store_true",
                    help="启用自碰撞检测")
    ap.add_argument("--pos-threshold", type=float, default=0.005,
                    help="位置误差阈值 (m)")
    ap.add_argument("--rot-threshold", type=float, default=0.05,
                    help="旋转误差阈值 (rad)")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子")
    args = ap.parse_args()

    print("=" * 60)
    print("  Marvin 左臂 IK 可达性测试 (含操纵度统计)")
    print("=" * 60)
    print(f"  位置范围: x=0, y=0, z=[0.2, 0.776] (相对 Base_L)")
    print(f"  旋转固定: [0.500, -0.500, 0.500, -0.500] (w,x,y,z)")
    print(f"  位置采样数: {args.n_pos}")
    print(f"  每位置重复次数: {args.n_rot}")
    print(f"  总测试数: {args.n_pos * args.n_rot}")
    print(f"  IK seeds: {args.num_seeds}")
    print(f"  自碰撞检测: {'ON' if args.self_collision else 'OFF'}")
    print(f"  位置阈值: {args.pos_threshold} m")
    print(f"  旋转阈值: {args.rot_threshold} rad")
    print("=" * 60)

    # ========== 1. 初始化 IK 求解器 + FK 模型 ==========
    print("\n[1/4] 初始化 CuRobo IK 求解器 + FK 模型...")
    tensor_args = TensorDeviceType()

    robot_file = "marvin_left_arm.yml"
    cfg_dict = load_yaml(join_path(get_robot_configs_path(), robot_file))

    # 修改配置: base_link=Base_L, ee_link=Link7_L
    cfg_dict["robot_cfg"]["kinematics"]["base_link"] = "Base_L"
    cfg_dict["robot_cfg"]["kinematics"]["ee_link"] = "Link7_L"

    # Marvin 左臂 7-DOF 各关节 child link (用于雅可比计算)
    joint_child_links = [
        "Link1_L", "Link2_L", "Link3_L", "Link4_L",
        "Link5_L", "Link6_L", "Link7_L",
    ]
    # 配置 link_names 以获取中间 link 位姿 (FK 输出)
    cfg_dict["robot_cfg"]["kinematics"]["link_names"] = joint_child_links

    robot_cfg = RobotConfig.from_dict(cfg_dict["robot_cfg"])

    # IK 求解器
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,  # 无环境障碍物
        rotation_threshold=args.rot_threshold,
        position_threshold=args.pos_threshold,
        num_seeds=args.num_seeds,
        self_collision_check=args.self_collision,
        self_collision_opt=args.self_collision,
        tensor_args=tensor_args,
        use_cuda_graph=False,  # 关闭以支持动态 batch size
    )
    ik_solver = IKSolver(ik_config)
    print(f"  关节名: {ik_solver.joint_names}")
    print(f"  DOF: {ik_solver.dof}")

    # 独立 FK 模型 (用于操纵度计算)
    km = CudaRobotModel(robot_cfg.kinematics)
    # 确定各关节 child link 在 link_names 中的索引
    joint_link_indices = []
    for jl_name in joint_child_links:
        idx = km.link_names.index(jl_name)
        joint_link_indices.append(idx)
    # ee_link 在 link_names 中的索引 (Link7_L)
    ee_link_idx = km.link_names.index("Link7_L")
    print(f"  FK link_names: {km.link_names}")
    print(f"  joint_link_indices: {joint_link_indices}")
    print(f"  ee_link_idx: {ee_link_idx}")
    print(f"  IK 求解器 + FK 模型初始化完成")

    # ========== 2. 生成目标位姿 ==========
    print("\n[2/4] 生成目标位姿...")

    # z 方向均匀采样 (上限 0.776)
    z_values = np.linspace(0.2, 0.776, args.n_pos, dtype=np.float32)
    print(f"  z 采样值: {z_values}")

    # 固定旋转四元数 (w, x, y, z)
    fixed_quat = np.array([0.500, -0.500, 0.500, -0.500], dtype=np.float32)
    print(f"  固定旋转: {fixed_quat}")

    # 为每个位置生成目标
    all_positions = []
    all_quaternions = []
    pos_labels = []  # 记录每个目标对应的 z 值

    for z_val in z_values:
        positions = np.zeros((args.n_rot, 3), dtype=np.float32)
        positions[:, 2] = z_val  # z 分量
        quats = np.tile(fixed_quat, (args.n_rot, 1))  # 重复固定四元数

        all_positions.append(positions)
        all_quaternions.append(quats)
        pos_labels.extend([z_val] * args.n_rot)

    all_positions = np.concatenate(all_positions, axis=0)  # [N, 3]
    all_quaternions = np.concatenate(all_quaternions, axis=0)  # [N, 4] (w,x,y,z)
    N = len(all_positions)
    print(f"  总目标数: {N}")

    # 转为 torch tensor
    pos_tensor = torch.from_numpy(all_positions).to(tensor_args.device)
    quat_tensor = torch.from_numpy(all_quaternions).to(tensor_args.device)

    # ========== 3. 批量 IK 求解 ==========
    print("\n[3/4] 批量 IK 求解...")

    # 分批求解 (避免显存不足)
    batch_size = 100
    all_success = []
    all_pos_err = []
    all_rot_err = []
    all_solutions = []

    t_start = time.time()
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        batch_goal = Pose(pos_tensor[i:j], quat_tensor[i:j])
        result = ik_solver.solve_batch(batch_goal)
        all_success.append(result.success.cpu())
        all_pos_err.append(result.position_error.cpu())
        all_rot_err.append(result.rotation_error.cpu())
        all_solutions.append(result.solution.cpu())

    torch.cuda.synchronize()
    t_elapsed = time.time() - t_start

    success = torch.cat(all_success, dim=0).squeeze()  # [N]
    pos_err = torch.cat(all_pos_err, dim=0).squeeze()  # [N]
    rot_err = torch.cat(all_rot_err, dim=0).squeeze()  # [N]
    solutions = torch.cat(all_solutions, dim=0).squeeze()  # [N, dof]

    n_success = int(success.sum().item())
    print(f"  IK 求解完成: {n_success}/{N} 成功 ({100.0*n_success/N:.1f}%)")
    print(f"  求解耗时: {t_elapsed:.3f}s ({N / t_elapsed:.0f} poses/s)")

    # ========== 4. 操纵度计算 (对 IK 成功的解) ==========
    print("\n[4/4] 计算操纵度...")

    # 初始化操纵度数组 (全部为 NaN, 失败的保持 NaN)
    manip_all = torch.full((N,), float('nan'))

    if n_success > 0:
        success_mask = success.bool()
        success_solutions = solutions[success_mask]  # [n_success, dof]

        # 分批 FK + 操纵度计算
        fk_batch_size = 512
        manip_list = []
        n_s = success_solutions.shape[0]

        for i in range(0, n_s, fk_batch_size):
            j = min(i + fk_batch_size, n_s)
            q_batch = success_solutions[i:j].to(tensor_args.device)

            # FK 获取各 link 位姿
            st = km.get_state(q_batch)
            link_pos = st.links_position  # [B, n_links, 3]
            link_quat = st.links_quaternion  # [B, n_links, 4]
            ee_pos = st.ee_position  # [B, 3]

            # 计算几何雅可比
            J = compute_geometric_jacobian_batch(
                link_pos, link_quat, ee_pos, joint_link_indices
            )
            # 计算操纵度
            manip = compute_manipulability(J)
            manip_list.append(manip.cpu())

        manip_success = torch.cat(manip_list, dim=0)  # [n_success]
        manip_all[success_mask] = manip_success

        print(f"  操纵度计算完成 ({n_success} 个成功解)")
    else:
        manip_success = torch.tensor([])
        print(f"  无成功解, 跳过操纵度计算")

    # ========== 5. 统计结果 ==========
    print("\n" + "=" * 60)
    print("  结果统计")
    print("=" * 60)

    print(f"\n  总体成功率: {n_success}/{N} = {100.0 * n_success / N:.2f}%")
    print(f"  求解耗时: {t_elapsed:.3f}s ({N / t_elapsed:.0f} poses/s)")

    if n_success > 0:
        print(f"\n  成功样本统计:")
        print(f"    位置误差 mean: {pos_err[success.bool()].mean().item() * 1000:.3f} mm")
        print(f"    位置误差 max:  {pos_err[success.bool()].max().item() * 1000:.3f} mm")
        print(f"    旋转误差 mean: {rot_err[success.bool()].mean().item():.5f} rad")
        print(f"    旋转误差 max:  {rot_err[success.bool()].max().item():.5f} rad")

        # 操纵度统计
        print(f"\n  操纵度统计 (Yoshikawa w = sqrt(det(J·Jᵀ))):")
        print(f"    mean:   {manip_success.mean().item():.6f}")
        print(f"    std:    {manip_success.std().item():.6f}")
        print(f"    min:    {manip_success.min().item():.6f}")
        print(f"    max:    {manip_success.max().item():.6f}")
        print(f"    median: {manip_success.median().item():.6f}")
        # 接近奇异的比例
        near_singular = (manip_success < 0.001).sum().item()
        print(f"    接近奇异 (w < 0.001): {near_singular}/{n_success} "
              f"({100.0*near_singular/n_success:.1f}%)")

    # 按 z 值分组统计 (含操纵度 + 关节角)
    print(f"\n  按 z 值分组统计:")
    print(f"  {'z值':>8s} | {'成功数':>6s}/{args.n_rot} | {'成功率':>7s} | "
          f"{'平均位置误差':>12s} | {'平均操纵度':>10s} | {'最小操纵度':>10s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*7}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

    pos_labels_arr = np.array(pos_labels)
    for z_val in z_values:
        mask = pos_labels_arr == z_val
        mask_t = torch.from_numpy(mask)
        z_success = success[mask_t]
        z_pos_err = pos_err[mask_t]
        z_manip = manip_all[mask_t]
        z_solutions = solutions[mask_t]
        n_z_success = int(z_success.sum().item())
        rate = 100.0 * n_z_success / args.n_rot

        if n_z_success > 0:
            avg_pe = z_pos_err[z_success.bool()].mean().item() * 1000
            # 操纵度 (只取成功的, 即非 NaN 的)
            z_manip_valid = z_manip[z_success.bool()]
            avg_manip = z_manip_valid.mean().item()
            min_manip = z_manip_valid.min().item()
            print(f"  {z_val:8.3f} | {n_z_success:6d}/{args.n_rot} | {rate:6.1f}% | "
                  f"{avg_pe:10.3f} mm | {avg_manip:10.6f} | {min_manip:10.6f}")
            # 打印成功解的关节角 (rad + deg)
            z_sol_success = z_solutions[z_success.bool()]  # [n_z_success, dof]
            for si in range(z_sol_success.shape[0]):
                q_rad = z_sol_success[si].tolist()
                q_deg = [np.degrees(v) for v in q_rad]
                rad_str = ", ".join([f"{v:+.4f}" for v in q_rad])
                deg_str = ", ".join([f"{v:+.1f}" for v in q_deg])
                print(f"           q[{si}] rad = [{rad_str}]")
                print(f"           q[{si}] deg = [{deg_str}]")
        else:
            print(f"  {z_val:8.3f} | {n_z_success:6d}/{args.n_rot} | {rate:6.1f}% | "
                  f"{'N/A':>10s}    | {'N/A':>10s} | {'N/A':>10s}")

    # ========== 6. 保存详细结果 ==========
    output_path = "/home/ethanqjiang/workspace/curobo/marvin_arm/ik_reachability_results.npz"
    save_dict = dict(
        z_values=z_values,
        positions=all_positions,
        quaternions=all_quaternions,
        success=success.numpy(),
        position_error=pos_err.numpy(),
        rotation_error=rot_err.numpy(),
        solutions=solutions.numpy(),
        manipulability=manip_all.numpy(),
        n_pos=args.n_pos,
        n_rot=args.n_rot,
        num_seeds=args.num_seeds,
        self_collision=args.self_collision,
    )
    np.savez_compressed(output_path, **save_dict)
    print(f"\n  详细结果已保存: {output_path}")

    # ========== 7. 失败案例分析 ==========
    if n_success < N:
        n_fail = N - n_success
        print(f"\n  失败案例分析 ({n_fail} 个):")
        fail_mask = ~success.bool()
        fail_pos_err = pos_err[fail_mask]
        fail_rot_err = rot_err[fail_mask]
        print(f"    位置误差 mean: {fail_pos_err.mean().item() * 1000:.3f} mm")
        print(f"    位置误差 max:  {fail_pos_err.max().item() * 1000:.3f} mm")
        print(f"    旋转误差 mean: {fail_rot_err.mean().item():.5f} rad")
        print(f"    旋转误差 max:  {fail_rot_err.max().item():.5f} rad")

        # 找出失败最多的 z 值
        fail_z_counts = []
        for z_val in z_values:
            mask = pos_labels_arr == z_val
            mask_t = torch.from_numpy(mask)
            n_fail_z = int((~success[mask_t].bool()).sum().item())
            fail_z_counts.append((z_val, n_fail_z))
        fail_z_counts.sort(key=lambda x: -x[1])
        print(f"\n    失败最多的 z 值 (top 5):")
        for z_val, cnt in fail_z_counts[:5]:
            print(f"      z={z_val:.3f}: {cnt}/{args.n_rot} 失败 ({100.0*cnt/args.n_rot:.1f}%)")

    print("\n" + "=" * 60)
    print("  测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
