#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAKA 7-DOF 机械臂 —— 旋转可达性工作空间分析

原理:
  1. FK 采样确定臂展范围 (bounding box)
  2. 在范围内划分小立方体 (voxel grid)
  3. 对每个立方体中心点, 在内切球面上均匀采样朝向:
     - 球面上均匀撒点 (Fibonacci lattice)
     - 每个采样点的法线方向作为末端 Z 轴
     - 构造完整旋转矩阵 → 四元数
  4. 对每个 (位置, 朝向) 组合做 IK 求解
  5. 统计每个 voxel 的旋转可达比例
  6. 发布 PointCloud2 到 RViz, 颜色表示旋转可达比例

朝向采样原理:
  对于 SO(3) 的完整覆盖, 需要两层采样:
    - 球面方向 (Fibonacci lattice): 末端 Z 轴指向哪 (2 个自由度)
    - 自旋角 (均匀分割): 绕末端 Z 轴旋转多少 (1 个自由度)
  总朝向数 = N_sphere × N_roll

使用方法:
  # 基本用法 (默认 0.05m 网格, 20 个球面方向 × 8 个自旋角 = 160 个朝向)
  python examples/jaka_workspace_reachability.py

  # 精细分析 (0.03m 网格, 50 球面方向 × 12 自旋角 = 600 个朝向)
  python examples/jaka_workspace_reachability.py --resolution 0.03 --num-sphere 50 --num-roll 12

  # 粗略预览 (0.1m 网格, 10 球面方向 × 4 自旋角 = 40 个朝向, 快速)
  python examples/jaka_workspace_reachability.py --resolution 0.1 --num-sphere 10 --num-roll 4

  # 开启自碰撞检测
  python examples/jaka_workspace_reachability.py --self-collision

  # 指定 FK 采样数量来确定范围
  python examples/jaka_workspace_reachability.py --fk-samples 50000

  # 指定 RViz 话题和 frame
  python examples/jaka_workspace_reachability.py --topic /workspace_cloud --frame LINK_BASE

  # 交互式切片查看 (计算完成后进入交互模式)
  python examples/jaka_workspace_reachability.py --publish --slice

  # 从已有结果加载并进入切片模式
  python examples/jaka_workspace_reachability.py --load ./workspace_reachability_results --publish --slice

  # 切片模式下的交互命令:
  #   x=0.3          → 只显示 x≈0.3 的 YZ 切面
  #   y=-0.1         → 只显示 y≈-0.1 的 XZ 切面
  #   z=0.5          → 只显示 z≈0.5 的 XY 切面
  #   x=0.3 z=0.5    → 组合切片, 只显示满足两个条件的点
  #   x=0.1:0.3      → 范围切片, 显示 0.1 ≤ x ≤ 0.3 的点
  #   all / reset     → 恢复显示全部点
  #   info            → 打印当前切片的统计信息
  #   list x / list y / list z → 列出该轴所有可用的切片值
  #   quit / exit     → 退出
"""

import os
import sys
import time
import argparse
import math
import re
import struct
import threading
from typing import Tuple, List, Optional, Dict

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


# ============================================================
#  朝向采样: Fibonacci Lattice 球面采样 × 自旋角采样
# ============================================================

def generate_so3_orientations(
    n_sphere: int,
    n_roll: int,
) -> np.ndarray:
    """完整 SO(3) 朝向采样: 球面方向 × 自旋角.

    原理:
      SO(3) 有 3 个自由度, 分两层采样:
      1. 球面方向 (2 DOF): Fibonacci lattice 均匀采样 n_sphere 个法线方向,
         每个法线方向作为末端 Z 轴
      2. 自旋角 (1 DOF): 对每个法线方向, 绕末端 Z 轴均匀采样 n_roll 个角度
         角度范围 [0, 2π), 间隔 2π/n_roll

      总朝向数 = n_sphere × n_roll

    构造过程:
      对于球面上的每个法线方向 n = (nx, ny, nz):
      1. Z_axis = n
      2. 选参考向量 ref (不与 n 平行)
      3. X0 = normalize(ref × Z), Y0 = Z × X0  → 基准旋转 R0
      4. 对每个自旋角 α:
         X_α = cos(α)*X0 + sin(α)*Y0
         Y_α = -sin(α)*X0 + cos(α)*Y0
         R_α = [X_α, Y_α, Z]  → 四元数

    Args:
        n_sphere: 球面方向采样数
        n_roll: 每个方向的自旋角采样数

    Returns:
        quaternions: (n_sphere * n_roll, 4) 四元数数组, wxyz 格式
    """
    # --- 第一层: Fibonacci lattice 球面均匀采样 ---
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    indices = np.arange(n_sphere, dtype=np.float64)

    theta = np.arccos(1.0 - 2.0 * (indices + 0.5) / n_sphere)
    phi = 2.0 * np.pi * indices / golden_ratio

    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    normals = np.stack([nx, ny, nz], axis=-1)  # (n_sphere, 3)

    # --- 第二层: 自旋角均匀采样 ---
    roll_angles = np.linspace(
        0, 2.0 * np.pi, n_roll, endpoint=False, dtype=np.float64
    )  # (n_roll,)

    total = n_sphere * n_roll
    quaternions = np.zeros((total, 4), dtype=np.float64)
    idx = 0

    for i in range(n_sphere):
        z_axis = normals[i]

        # 选参考向量: 避免与 z_axis 平行
        if abs(z_axis[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([1.0, 0.0, 0.0])

        # 基准 X/Y 轴
        x0 = np.cross(ref, z_axis)
        x0 /= np.linalg.norm(x0)
        y0 = np.cross(z_axis, x0)

        # 对每个自旋角构造旋转矩阵
        for j in range(n_roll):
            alpha = roll_angles[j]
            ca, sa = math.cos(alpha), math.sin(alpha)
            x_axis = ca * x0 + sa * y0
            y_axis = -sa * x0 + ca * y0

            R = np.column_stack([x_axis, y_axis, z_axis])  # 3x3
            quaternions[idx] = _rotation_matrix_to_quaternion_wxyz(R)
            idx += 1

    return quaternions.astype(np.float32)


def _rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 (3x3) → 四元数 (w, x, y, z).

    使用 Shepperd 方法, 数值稳定.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return q


# ============================================================
#  CuRobo 初始化
# ============================================================

def create_ik_solver(
    self_collision: bool = False,
    num_seeds: int = 20,
    use_cuda_graph: bool = True,
) -> IKSolver:
    """创建 CuRobo IK 求解器."""
    tensor_args = TensorDeviceType()
    robot_file = "jaka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=num_seeds,
        self_collision_check=self_collision,
        self_collision_opt=self_collision,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
    )
    return IKSolver(ik_config)


# ============================================================
#  Step 1: FK 采样确定工作空间范围
# ============================================================

def estimate_workspace_bounds(
    ik_solver: IKSolver,
    num_samples: int = 50000,
    padding: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """通过 FK 采样估计工作空间的 AABB 包围盒.

    Returns:
        (min_bounds, max_bounds): 各为 (3,) 的 ndarray, 单位 m
    """
    print(f"\n  [Step 1] FK 采样估计工作空间范围 (N={num_samples})...")
    t0 = time.time()

    batch_size = 10000
    all_positions = []
    collected = 0

    while collected < num_samples:
        n_batch = min(batch_size, num_samples - collected)
        q_samples = ik_solver.sample_configs(n_batch)
        if q_samples.shape[0] == 0:
            q_samples = ik_solver.sample_configs(n_batch, rejection_ratio=50)
            if q_samples.shape[0] == 0:
                break
        state = ik_solver.fk(q_samples)
        ee_pos = state.ee_position.detach().cpu().numpy()
        all_positions.append(ee_pos)
        collected += ee_pos.shape[0]

    positions = np.concatenate(all_positions, axis=0)[:num_samples]
    min_bounds = positions.min(axis=0) - padding
    max_bounds = positions.max(axis=0) + padding

    elapsed = time.time() - t0
    print(f"    采样 {positions.shape[0]} 个点, 耗时 {elapsed:.2f}s")
    print(f"    X: [{min_bounds[0]:.3f}, {max_bounds[0]:.3f}] m")
    print(f"    Y: [{min_bounds[1]:.3f}, {max_bounds[1]:.3f}] m")
    print(f"    Z: [{min_bounds[2]:.3f}, {max_bounds[2]:.3f}] m")

    return min_bounds, max_bounds


# ============================================================
#  Step 2: 生成 Voxel 网格
# ============================================================

def generate_voxel_centers(
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """在 AABB 内生成均匀 voxel 中心点.

    Returns:
        centers: (N, 3) ndarray
    """
    xs = np.arange(min_bounds[0], max_bounds[0] + resolution, resolution)
    ys = np.arange(min_bounds[1], max_bounds[1] + resolution, resolution)
    zs = np.arange(min_bounds[2], max_bounds[2] + resolution, resolution)

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack(
        [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1
    ).astype(np.float32)

    print(f"\n  [Step 2] Voxel 网格生成")
    print(f"    分辨率: {resolution}m")
    print(f"    网格维度: {len(xs)} x {len(ys)} x {len(zs)}")
    print(f"    总 voxel 数: {centers.shape[0]}")

    return centers


# ============================================================
#  Step 3 & 4: 批量 IK 求解 + 统计旋转可达比例
# ============================================================

def fk_prefilter_voxels(
    ik_solver: IKSolver,
    voxel_centers: np.ndarray,
    resolution: float,
    num_fk_samples: int = 200000,
) -> np.ndarray:
    """通过大量 FK 采样预筛选可达 voxel, 跳过明显不可达的区域.

    原理: 随机采样大量关节角 → FK 得到末端位置 → 标记哪些 voxel 被覆盖
    未被任何 FK 采样覆盖的 voxel 大概率不可达, 可以跳过.

    Args:
        ik_solver: IK 求解器 (用其 FK 和采样功能)
        voxel_centers: (M, 3) 全部 voxel 中心
        resolution: voxel 分辨率
        num_fk_samples: FK 采样数量, 越多筛选越准确

    Returns:
        reachable_mask: (M,) bool 数组, True 表示该 voxel 可能可达
    """
    print(f"\n  [FK 预筛选] 采样 {num_fk_samples} 个 FK 点, 筛选可达 voxel...")
    t0 = time.time()

    batch_size = 10000
    all_positions = []
    collected = 0

    while collected < num_fk_samples:
        n_batch = min(batch_size, num_fk_samples - collected)
        q_samples = ik_solver.sample_configs(n_batch)
        if q_samples.shape[0] == 0:
            q_samples = ik_solver.sample_configs(n_batch, rejection_ratio=50)
            if q_samples.shape[0] == 0:
                break
        state = ik_solver.fk(q_samples)
        ee_pos = state.ee_position.detach().cpu().numpy()
        all_positions.append(ee_pos)
        collected += ee_pos.shape[0]

    fk_positions = np.concatenate(all_positions, axis=0)  # (N, 3)

    # 对每个 FK 点, 找到最近的 voxel 并标记
    # 使用向量化距离计算: 将 FK 点量化到 voxel 网格
    M = voxel_centers.shape[0]
    half_res = resolution / 2.0

    # 构建 voxel 中心的 KD-Tree 或使用网格索引
    # 简单方法: 对每个 FK 点, 检查它落在哪个 voxel 内
    reachable_mask = np.zeros(M, dtype=bool)

    # 使用扩大的搜索半径 (1.5 * resolution) 来增加覆盖
    search_radius = resolution * 1.5

    # 分批处理避免内存爆炸
    fk_batch = 50000
    for fi in range(0, fk_positions.shape[0], fk_batch):
        fi_end = min(fi + fk_batch, fk_positions.shape[0])
        fk_pts = fk_positions[fi:fi_end]  # (B, 3)

        # 计算每个 FK 点到所有 voxel 中心的距离 (分块避免 OOM)
        voxel_batch = 100000
        for vi in range(0, M, voxel_batch):
            vi_end = min(vi + voxel_batch, M)
            vc = voxel_centers[vi:vi_end]  # (V, 3)

            # 广播计算距离: (B, 1, 3) - (1, V, 3) → (B, V)
            diff = fk_pts[:, None, :] - vc[None, :, :]  # (B, V, 3)
            dist = np.linalg.norm(diff, axis=-1)  # (B, V)

            # 标记距离小于搜索半径的 voxel
            close_mask = dist.min(axis=0) < search_radius  # (V,)
            reachable_mask[vi:vi_end] |= close_mask

    n_reachable = reachable_mask.sum()
    elapsed = time.time() - t0
    print(f"    FK 采样 {fk_positions.shape[0]} 个点, 耗时 {elapsed:.1f}s")
    print(f"    可能可达的 voxel: {n_reachable}/{M} ({n_reachable/M*100:.1f}%)")
    print(f"    跳过不可达 voxel: {M - n_reachable} ({(M-n_reachable)/M*100:.1f}%)")

    return reachable_mask


def compute_reachability(
    ik_solver: IKSolver,
    voxel_centers: np.ndarray,
    orientations_wxyz: np.ndarray,
    ik_batch_size: int = 2048,
    use_cuda_graph: bool = True,
    early_stop_threshold: int = 0,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = 5000,
    reachable_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """对每个 voxel 中心, 测试所有朝向的 IK 可达性 (GPU 加速版).

    核心加速: 跨 voxel 批量化 —— 将多个 voxel × 多个朝向展平为一个大 batch,
    一次 solve_batch 同时处理多个 voxel 的多个朝向, 大幅减少 Python 循环次数.

    加速策略:
      1. 跨 voxel 批量化: 消除逐 voxel Python 循环瓶颈 (~100x)
      2. 启用 CUDA Graph: 固定 batch size, 减少 kernel launch 开销 (~10x)
      3. 大 batch size: 充分利用 GPU 并行 (2048~4096)
      4. FK 预筛选: 跳过明显不可达的 voxel
      5. 断点续算: 定期保存进度, 支持中断后继续

    Args:
        ik_solver: CuRobo IK 求解器
        voxel_centers: (M, 3) voxel 中心坐标
        orientations_wxyz: (K, 4) 朝向四元数 (wxyz)
        ik_batch_size: 每批 IK 求解的数量 (CUDA Graph 模式下固定)
        use_cuda_graph: 是否启用 CUDA Graph 加速
        early_stop_threshold: 早停阈值 (本版本未使用, 保留接口兼容)
        checkpoint_dir: 断点续算目录, None 则不保存断点
        checkpoint_interval: 每处理多少个 voxel 保存一次断点
        reachable_mask: (M,) FK 预筛选结果, None 则不筛选

    Returns:
        reachability: (M,) 每个 voxel 的旋转可达比例 [0.0, 1.0]
    """
    M = voxel_centers.shape[0]
    K = orientations_wxyz.shape[0]

    # 确定实际需要计算的 voxel
    if reachable_mask is not None:
        compute_indices = np.where(reachable_mask)[0]
    else:
        compute_indices = np.arange(M)
    M_compute = len(compute_indices)
    total_queries = M_compute * K

    print(f"\n  [Step 3] IK 旋转可达性计算 (跨 voxel 批量化 + GPU 加速)")
    print(f"    总 Voxel 数: {M}")
    print(f"    需计算 Voxel 数: {M_compute}" + (f" (FK 预筛选跳过 {M - M_compute})" if reachable_mask is not None else ""))
    print(f"    每个 voxel 朝向数: {K}")
    print(f"    总 IK 查询数: {total_queries:,}")
    print(f"    IK 批大小: {ik_batch_size}")
    print(f"    CUDA Graph: {'开启' if use_cuda_graph else '关闭'}")

    tensor_args = TensorDeviceType()
    device = tensor_args.device
    dtype = tensor_args.dtype

    # 成功计数数组 (在 GPU 上累加)
    success_counts = torch.zeros(M_compute, dtype=torch.int32, device=device)

    # 断点续算: 检查是否有已保存的进度
    start_orient_offset = 0
    if checkpoint_dir is not None:
        ckpt_file = os.path.join(checkpoint_dir, "_checkpoint_reachability.npz")
        if os.path.exists(ckpt_file):
            ckpt = np.load(ckpt_file)
            success_counts_np = ckpt["success_counts"]
            start_orient_offset = int(ckpt["orient_offset"])
            success_counts[:len(success_counts_np)] = torch.from_numpy(
                success_counts_np
            ).to(device)
            print(f"    ✓ 从断点恢复: 已完成朝向 {start_orient_offset}/{K}")

    # 预先将数据放到 GPU
    orient_tensor = torch.tensor(
        orientations_wxyz, device=device, dtype=dtype
    )  # (K, 4)

    # 需要计算的 voxel 中心放到 GPU
    compute_centers = torch.tensor(
        voxel_centers[compute_indices], device=device, dtype=dtype
    )  # (M_compute, 3)

    # 预热: 用固定 batch size 做一次 solve_batch 以触发 CUDA Graph 捕获
    print(f"    预热 CUDA Graph (batch_size={ik_batch_size})...")
    warmup_pos = torch.zeros((ik_batch_size, 3), device=device, dtype=dtype)
    warmup_pos[:, 2] = 0.5
    warmup_quat = orient_tensor[0:1].expand(ik_batch_size, -1).contiguous()
    _ = ik_solver.solve_batch(Pose(warmup_pos, warmup_quat))
    torch.cuda.synchronize()
    print(f"    预热完成!")

    # 预分配固定大小的 GPU buffer
    pos_buffer = torch.zeros((ik_batch_size, 3), device=device, dtype=dtype)
    quat_buffer = torch.zeros((ik_batch_size, 4), device=device, dtype=dtype)

    # ================================================================
    # 核心策略: 跨 voxel 批量化
    #
    # 外层循环: 遍历朝向 (K 个, 每次取 1 个或几个朝向)
    # 内层循环: 对当前朝向, 将所有 M_compute 个 voxel 分批送入 GPU
    #
    # 这样每个 batch 包含 ik_batch_size 个不同 voxel 的同一朝向,
    # 大幅减少 Python 循环次数: 从 M_compute * ceil(K/batch) 降到
    # K * ceil(M_compute/batch)
    #
    # 当 M_compute >> K 时 (本场景: 780万 >> 3万), 两者差不多
    # 但关键优势是: 消除了逐 voxel 的 Python 开销和 tensor 创建
    # ================================================================

    # 计算每个 batch 能处理多少个 voxel
    # 如果 K 很大, 可以一次处理多个朝向 × 多个 voxel
    # 策略: 每次取 n_orient_per_batch 个朝向, n_voxel_per_batch 个 voxel
    # 使得 n_orient_per_batch * n_voxel_per_batch ≈ ik_batch_size

    # 对于本场景 (K=30000, M=780万), 最优策略是:
    # 每次 1 个朝向, ik_batch_size 个 voxel → 每个 batch 处理 2048 个 voxel
    # 总 batch 数 = K * ceil(M_compute / ik_batch_size) = 30000 * 3818 ≈ 1.1亿
    # 这仍然太多!
    #
    # 更好的策略: 每次取多个朝向, 少量 voxel
    # n_orient_per_batch = min(K, ik_batch_size)
    # 如果 K > ik_batch_size: 每次 ik_batch_size 个朝向, 1 个 voxel
    # 如果 K <= ik_batch_size: 每次 K 个朝向, floor(ik_batch_size/K) 个 voxel

    if K >= ik_batch_size:
        # 朝向数 >= batch_size: 每次处理 1 个 voxel 的 ik_batch_size 个朝向
        n_voxels_per_batch = 1
        n_orients_per_step = ik_batch_size
    else:
        # 朝向数 < batch_size: 每次处理多个 voxel 的全部朝向
        n_voxels_per_batch = ik_batch_size // K
        n_orients_per_step = K

    total_batches_est = 0
    if K >= ik_batch_size:
        batches_per_voxel = math.ceil(K / ik_batch_size)
        total_batches_est = M_compute * batches_per_voxel
    else:
        total_batches_est = math.ceil(M_compute / n_voxels_per_batch)

    print(f"    批量策略: 每 batch {n_voxels_per_batch} voxel × {n_orients_per_step} 朝向")
    print(f"    预估总 batch 数: {total_batches_est:,}")

    t0 = time.time()
    solved_total = start_orient_offset * M_compute
    batch_count = 0

    if K >= ik_batch_size:
        # ---- 模式 A: K >= batch_size, 逐 voxel 但朝向分批 ----
        # 这种情况下无法跨 voxel, 但可以优化内部循环
        # 预先构建朝向 batch 索引
        orient_batches = []
        for ki in range(start_orient_offset, K, ik_batch_size):
            ki_end = min(ki + ik_batch_size, K)
            orient_batches.append((ki, ki_end))

        for vi_start in range(0, M_compute, 1):
            vi_end = min(vi_start + 1, M_compute)
            center = compute_centers[vi_start]  # (3,)

            for ki, ki_end in orient_batches:
                n_this = ki_end - ki

                pos_buffer[:n_this] = center.unsqueeze(0).expand(n_this, -1)
                quat_buffer[:n_this] = orient_tensor[ki:ki_end]
                if n_this < ik_batch_size:
                    pos_buffer[n_this:] = center.unsqueeze(0).expand(
                        ik_batch_size - n_this, -1
                    )
                    quat_buffer[n_this:] = orient_tensor[0:1].expand(
                        ik_batch_size - n_this, -1
                    )

                result = ik_solver.solve_batch(Pose(pos_buffer, quat_buffer))
                sub_success = result.success.view(-1)[:n_this]
                success_counts[vi_start] += sub_success.sum().int()
                batch_count += 1
                solved_total += n_this

            # 进度 (每 500 个 voxel 或最后一个)
            if (vi_start + 1) % 500 == 0 or vi_start == M_compute - 1:
                elapsed = time.time() - t0
                speed = solved_total / max(elapsed, 0.001)
                remaining = total_queries - solved_total
                eta = remaining / max(speed, 1.0)
                eta_str = f"{eta:.0f}s" if eta < 3600 else f"{eta/3600:.1f}h"
                print(
                    f"    进度: {vi_start+1}/{M_compute} voxels ({(vi_start+1)/M_compute*100:.2f}%), "
                    f"速度: {speed:.0f} IK/s, ETA: {eta_str}, "
                    f"batch: {batch_count:,}",
                    end="\r",
                )

            # 断点保存
            if checkpoint_dir is not None and (vi_start + 1) % checkpoint_interval == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_file = os.path.join(checkpoint_dir, "_checkpoint_reachability.npz")
                np.savez(
                    ckpt_file,
                    success_counts=success_counts.cpu().numpy(),
                    orient_offset=0,  # 模式 A 按 voxel 保存
                )

    else:
        # ---- 模式 B: K < batch_size, 跨 voxel 批量化 (最优!) ----
        # 每个 batch 处理 n_voxels_per_batch 个 voxel 的全部 K 个朝向
        # 预先构建朝向 tile (K 个朝向重复使用)
        orient_tile = orient_tensor.unsqueeze(0)  # (1, K, 4)

        for vi_start in range(0, M_compute, n_voxels_per_batch):
            vi_end = min(vi_start + n_voxels_per_batch, M_compute)
            n_voxels = vi_end - vi_start
            n_total = n_voxels * K

            # 构造 batch: 每个 voxel 的位置重复 K 次
            batch_pos = compute_centers[vi_start:vi_end].unsqueeze(1).expand(
                n_voxels, K, 3
            ).reshape(-1, 3)  # (n_voxels*K, 3)

            batch_quat = orient_tile.expand(n_voxels, K, 4).reshape(-1, 4)

            # 填充到固定 batch_size
            pos_buffer[:n_total] = batch_pos
            quat_buffer[:n_total] = batch_quat
            if n_total < ik_batch_size:
                pos_buffer[n_total:] = batch_pos[0:1].expand(
                    ik_batch_size - n_total, -1
                )
                quat_buffer[n_total:] = batch_quat[0:1].expand(
                    ik_batch_size - n_total, -1
                )

            result = ik_solver.solve_batch(Pose(pos_buffer, quat_buffer))
            sub_success = result.success.view(-1)[:n_total]

            # 统计每个 voxel 的成功数
            success_per_voxel = sub_success.view(n_voxels, K).sum(dim=1).int()
            success_counts[vi_start:vi_end] += success_per_voxel

            batch_count += 1
            solved_total += n_total

            # 进度
            if batch_count % 200 == 0 or vi_end == M_compute:
                elapsed = time.time() - t0
                speed = solved_total / max(elapsed, 0.001)
                remaining = total_queries - solved_total
                eta = remaining / max(speed, 1.0)
                eta_str = f"{eta:.0f}s" if eta < 3600 else f"{eta/3600:.1f}h"
                print(
                    f"    进度: {vi_end}/{M_compute} voxels ({vi_end/M_compute*100:.2f}%), "
                    f"速度: {speed:.0f} IK/s, ETA: {eta_str}, "
                    f"batch: {batch_count:,}",
                    end="\r",
                )

            # 断点保存
            if checkpoint_dir is not None and batch_count % 1000 == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_file = os.path.join(checkpoint_dir, "_checkpoint_reachability.npz")
                np.savez(
                    ckpt_file,
                    success_counts=success_counts.cpu().numpy(),
                    orient_offset=0,
                )

    # 计算可达比例
    reachability = np.zeros(M, dtype=np.float32)
    counts_np = success_counts.cpu().numpy().astype(np.float32)
    reachability[compute_indices] = counts_np / K

    elapsed = time.time() - t0
    n_reachable = (reachability > 0).sum()
    n_full = (reachability >= 1.0).sum()

    print(f"\n    计算完成! 耗时 {elapsed:.1f}s ({elapsed/3600:.2f}h), 速度 {solved_total/max(elapsed,0.001):.0f} IK/s")
    print(f"    总 batch 数: {batch_count:,}")
    print(f"    有可达朝向的 voxel: {n_reachable}/{M} ({n_reachable/M*100:.1f}%)")
    print(f"    全朝向可达的 voxel: {n_full}/{M} ({n_full/M*100:.1f}%)")
    print(f"    平均旋转可达比例: {reachability[reachability > 0].mean():.3f}" if n_reachable > 0 else "")

    # 清理断点文件
    if checkpoint_dir is not None:
        ckpt_file = os.path.join(checkpoint_dir, "_checkpoint_reachability.npz")
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)

    return reachability


# ============================================================
#  颜色映射: Turbo 色谱 (高区分度)
# ============================================================

# Turbo 色谱关键控制点 (t, R, G, B), t ∈ [0, 1]
# 深蓝 → 青 → 绿 → 黄 → 橙 → 红, 共 9 段插值
_TURBO_CONTROL_POINTS = [
    (0.00, 48, 18, 59),    # 深紫蓝
    (0.10, 40, 80, 160),   # 蓝
    (0.20, 18, 140, 210),  # 天蓝
    (0.30, 0, 185, 195),   # 青
    (0.40, 30, 210, 130),  # 青绿
    (0.50, 80, 220, 60),   # 绿
    (0.60, 160, 220, 20),  # 黄绿
    (0.70, 220, 200, 0),   # 黄
    (0.80, 250, 155, 0),   # 橙
    (0.90, 250, 85, 0),    # 橙红
    (1.00, 220, 30, 10),   # 红
]


def _turbo_colormap(t: float) -> Tuple[int, int, int]:
    """Turbo 风格颜色映射, 输入 t ∈ [0, 1], 输出 (R, G, B) ∈ [0, 255].

    比简单的红→黄→绿有更高的区分度, 在 RViz 暗色背景下清晰可见.
    """
    t = max(0.0, min(1.0, t))
    pts = _TURBO_CONTROL_POINTS

    # 找到 t 所在的区间
    for i in range(len(pts) - 1):
        t0, r0, g0, b0 = pts[i]
        t1, r1, g1, b1 = pts[i + 1]
        if t <= t1:
            # 线性插值
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            r = int(r0 + frac * (r1 - r0))
            g = int(g0 + frac * (g1 - g0))
            b = int(b0 + frac * (b1 - b0))
            return (
                max(0, min(255, r)),
                max(0, min(255, g)),
                max(0, min(255, b)),
            )

    # 兜底: 返回最后一个颜色
    return (pts[-1][1], pts[-1][2], pts[-1][3])


# ============================================================
#  Step 5: 保存结果 & 发布到 RViz
# ============================================================

def save_results(
    output_dir: str,
    voxel_centers: np.ndarray,
    reachability: np.ndarray,
    orientations_wxyz: np.ndarray,
    resolution: float,
):
    """保存分析结果到 npy 文件."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "voxel_centers.npy"), voxel_centers)
    np.save(os.path.join(output_dir, "reachability.npy"), reachability)
    np.save(os.path.join(output_dir, "orientations_wxyz.npy"), orientations_wxyz)
    np.savez(
        os.path.join(output_dir, "metadata.npz"),
        resolution=resolution,
        num_orientations=orientations_wxyz.shape[0],
    )
    print(f"\n  [保存] 结果已保存到: {output_dir}/")


# ============================================================
#  PointCloud2 构造工具
# ============================================================

def _build_pointcloud2_msg(points, values, frame_id, fields, header_cls, msg_cls):
    """从点坐标和可达比例构造 PointCloud2 消息.

    Args:
        points: (N, 3) 坐标
        values: (N,) 可达比例
        frame_id: 坐标系
        fields: PointField 列表
        header_cls: Header 类
        msg_cls: PointCloud2 类

    Returns:
        PointCloud2 消息, 如果 points 为空则返回 None
    """
    if points.shape[0] == 0:
        return None

    point_data = bytearray()
    for i in range(points.shape[0]):
        x, y, z = points[i]
        ratio = values[i]
        r, g, b = _turbo_colormap(ratio)
        rgb_int = (r << 16) | (g << 8) | b
        rgb_float = struct.unpack("f", struct.pack("I", rgb_int))[0]
        point_data.extend(struct.pack("ffff", x, y, z, rgb_float))

    header = header_cls()
    header.frame_id = frame_id

    msg = msg_cls()
    msg.header = header
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * points.shape[0]
    msg.data = bytes(point_data)
    msg.is_dense = True
    return msg


def _apply_slice_filter(
    voxel_centers: np.ndarray,
    reachability: np.ndarray,
    slice_conditions: Dict[str, object],
    resolution: float,
    min_reachability: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """根据切片条件过滤 voxel 数据.

    Args:
        voxel_centers: (M, 3) 全部 voxel 坐标
        reachability: (M,) 全部可达比例
        slice_conditions: 切片条件字典, 例如:
            {"x": 0.3}           → 精确切片, 容差 = resolution/2
            {"x": (0.1, 0.3)}    → 范围切片
            {"x": 0.3, "z": 0.5} → 组合切片
        resolution: voxel 分辨率, 用于计算精确切片的容差
        min_reachability: 最低可达比例阈值

    Returns:
        (filtered_points, filtered_values)
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    tol = resolution / 2.0

    mask = reachability > min_reachability

    for axis_name, value in slice_conditions.items():
        axis_idx = axis_map[axis_name]
        if isinstance(value, tuple):
            # 范围切片
            lo, hi = value
            mask &= (voxel_centers[:, axis_idx] >= lo) & (
                voxel_centers[:, axis_idx] <= hi
            )
        else:
            # 精确切片 (容差 = resolution/2)
            mask &= np.abs(voxel_centers[:, axis_idx] - value) <= tol

    return voxel_centers[mask], reachability[mask]


def _parse_slice_command(
    cmd: str,
) -> Optional[Dict[str, object]]:
    """解析用户输入的切片命令.

    支持格式:
        "x=0.3"          → {"x": 0.3}
        "y=-0.1"         → {"y": -0.1}
        "x=0.3 z=0.5"   → {"x": 0.3, "z": 0.5}
        "x=0.1:0.3"     → {"x": (0.1, 0.3)}
        "x=0.1:0.3 z=0.5" → {"x": (0.1, 0.3), "z": 0.5}

    Returns:
        切片条件字典, 解析失败返回 None
    """
    conditions = {}
    # 匹配 axis=value 或 axis=lo:hi
    pattern = re.compile(
        r"([xyz])\s*=\s*(-?[\d.]+)(?:\s*:\s*(-?[\d.]+))?"
    )
    matches = pattern.findall(cmd)
    if not matches:
        return None

    for axis, val1, val2 in matches:
        if val2:  # 范围切片
            conditions[axis] = (float(val1), float(val2))
        else:
            conditions[axis] = float(val1)

    return conditions if conditions else None


def _get_unique_axis_values(
    voxel_centers: np.ndarray,
    reachability: np.ndarray,
    axis: str,
    min_reachability: float = 0.0,
) -> np.ndarray:
    """获取某个轴上所有可用的唯一值 (已排序, 仅包含有可达点的值)."""
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    mask = reachability > min_reachability
    vals = voxel_centers[mask, axis_idx]
    return np.unique(np.round(vals, decimals=6))


# ============================================================
#  发布到 RViz (普通模式 / 交互式切片模式)
# ============================================================

def publish_to_rviz(
    voxel_centers: np.ndarray,
    reachability: np.ndarray,
    frame_id: str = "LINK_BASE",
    topic: str = "/workspace_reachability",
    rate_hz: float = 1.0,
    min_reachability: float = 0.0,
    interactive_slice: bool = False,
    resolution: float = 0.05,
):
    """将旋转可达性结果发布为 PointCloud2 到 RViz.

    Args:
        voxel_centers: (M, 3) voxel 中心坐标
        reachability: (M,) 旋转可达比例
        frame_id: RViz 坐标系
        topic: 发布话题
        rate_hz: 发布频率
        min_reachability: 最低可达比例阈值
        interactive_slice: 是否进入交互式切片模式
        resolution: voxel 分辨率 (切片容差 = resolution/2)
    """
    try:
        import rospy
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header
    except ImportError:
        print("\n  [错误] 未找到 rospy / sensor_msgs, 请确保已安装 ROS 并 source 了环境。")
        print("    例如: source /opt/ros/noetic/setup.bash")
        return

    rospy.init_node("jaka_workspace_reachability", anonymous=True)
    pub = rospy.Publisher(topic, PointCloud2, queue_size=1, latch=True)

    # 过滤掉可达比例低于阈值的点
    mask = reachability > min_reachability
    if mask.sum() == 0:
        print("  [警告] 没有可达的 voxel, 无法发布点云。")
        return

    print(f"\n  [发布] 发布到 RViz")
    print(f"    话题: {topic}")
    print(f"    坐标系: {frame_id}")
    print(f"    可显示点数: {mask.sum()} (阈值 > {min_reachability:.0%})")
    print(
        "    颜色: 深蓝(0%) → 青(25%) → 绿(50%)"
        " → 黄(75%) → 红(100%)"
    )

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    if interactive_slice:
        # ---- 交互式切片模式 ----
        _interactive_slice_loop(
            voxel_centers, reachability, resolution,
            min_reachability, frame_id, topic, rate_hz,
            pub, fields, Header, PointCloud2, rospy,
        )
    else:
        # ---- 普通模式: 发布全部点, 循环直到 Ctrl+C ----
        points = voxel_centers[mask]
        values = reachability[mask]
        msg = _build_pointcloud2_msg(
            points, values, frame_id, fields, Header, PointCloud2
        )
        if msg is None:
            return

        print(f"    发布 {points.shape[0]} 个点, 按 Ctrl+C 停止")
        rate = rospy.Rate(rate_hz)
        try:
            while not rospy.is_shutdown():
                msg.header.stamp = rospy.Time.now()
                pub.publish(msg)
                rate.sleep()
        except (rospy.ROSInterruptException, KeyboardInterrupt):
            print("\n  发布停止。")


def _interactive_slice_loop(
    voxel_centers, reachability, resolution,
    min_reachability, frame_id, topic, rate_hz,
    pub, fields, Header, PointCloud2, rospy,
):
    """交互式切片查看主循环.

    在后台线程持续发布点云, 主线程接收用户命令并动态更新切片.
    """
    # 共享状态: 当前要发布的消息 (线程安全通过 GIL 保证)
    current_msg = [None]  # 用列表包装以便在闭包中修改
    should_stop = [False]

    # 先发布全部点
    mask_all = reachability > min_reachability
    pts_all, vals_all = voxel_centers[mask_all], reachability[mask_all]
    current_msg[0] = _build_pointcloud2_msg(
        pts_all, vals_all, frame_id, fields, Header, PointCloud2
    )

    # 后台发布线程
    def _publish_loop():
        rate = rospy.Rate(rate_hz)
        while not rospy.is_shutdown() and not should_stop[0]:
            msg = current_msg[0]
            if msg is not None:
                msg.header.stamp = rospy.Time.now()
                pub.publish(msg)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

    pub_thread = threading.Thread(target=_publish_loop, daemon=True)
    pub_thread.start()

    # 打印帮助
    _print_slice_help()
    print(f"    当前显示: 全部 {pts_all.shape[0]} 个点")
    print(f"    切片容差: ±{resolution / 2.0:.4f}m (= resolution/2)")

    current_conditions = {}  # 当前切片条件

    # 主循环: 读取用户命令
    try:
        while not rospy.is_shutdown():
            try:
                user_input = input("\n  slice> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            cmd_lower = user_input.lower()

            # ---- 退出 ----
            if cmd_lower in ("quit", "exit", "q"):
                print("  退出切片模式。")
                break

            # ---- 帮助 ----
            if cmd_lower in ("help", "h", "?"):
                _print_slice_help()
                continue

            # ---- 恢复全部 ----
            if cmd_lower in ("all", "reset", "clear"):
                current_conditions = {}
                current_msg[0] = _build_pointcloud2_msg(
                    pts_all, vals_all, frame_id, fields, Header, PointCloud2
                )
                print(f"  ✓ 恢复显示全部 {pts_all.shape[0]} 个点")
                continue

            # ---- 列出某轴可用值 ----
            if cmd_lower.startswith("list"):
                parts = cmd_lower.split()
                if len(parts) >= 2 and parts[1] in ("x", "y", "z"):
                    axis = parts[1]
                    unique_vals = _get_unique_axis_values(
                        voxel_centers, reachability, axis, min_reachability
                    )
                    print(f"  {axis.upper()} 轴可用值 ({len(unique_vals)} 个):")
                    # 每行显示 10 个
                    for row_start in range(0, len(unique_vals), 10):
                        row = unique_vals[row_start:row_start + 10]
                        print("    " + "  ".join(f"{v:.3f}" for v in row))
                else:
                    print("  用法: list x / list y / list z")
                continue

            # ---- 统计信息 ----
            if cmd_lower == "info":
                _print_slice_info(
                    voxel_centers, reachability, current_conditions,
                    resolution, min_reachability,
                )
                continue

            # ---- 解析切片命令 ----
            conditions = _parse_slice_command(user_input)
            if conditions is None:
                print(f"  [错误] 无法解析命令: '{user_input}'")
                print("  输入 help 查看可用命令")
                continue

            # 验证轴名
            invalid_axes = [k for k in conditions if k not in ("x", "y", "z")]
            if invalid_axes:
                print(f"  [错误] 无效的轴: {invalid_axes}, 只支持 x/y/z")
                continue

            # 应用切片
            current_conditions = conditions
            filtered_pts, filtered_vals = _apply_slice_filter(
                voxel_centers, reachability, conditions,
                resolution, min_reachability,
            )

            if filtered_pts.shape[0] == 0:
                print(f"  [警告] 切片条件 {_format_conditions(conditions)} 没有匹配的点")
                print("  提示: 用 'list x' 查看可用值, 或调整切片值")
                # 发布空点云 (清除 RViz 显示)
                current_msg[0] = _build_pointcloud2_msg(
                    filtered_pts, filtered_vals, frame_id,
                    fields, Header, PointCloud2,
                )
                continue

            # 构造并更新消息
            current_msg[0] = _build_pointcloud2_msg(
                filtered_pts, filtered_vals, frame_id,
                fields, Header, PointCloud2,
            )

            # 打印切片统计
            n_reach = (filtered_vals > 0).sum()
            avg_reach = filtered_vals[filtered_vals > 0].mean() if n_reach > 0 else 0
            print(
                f"  ✓ 切片 {_format_conditions(conditions)}: "
                f"{filtered_pts.shape[0]} 个点, "
                f"平均可达比例 {avg_reach:.1%}"
            )

    except KeyboardInterrupt:
        print("\n  切片模式退出。")
    finally:
        should_stop[0] = True
        pub_thread.join(timeout=2.0)


def _format_conditions(conditions: Dict[str, object]) -> str:
    """格式化切片条件为可读字符串."""
    parts = []
    for axis in ("x", "y", "z"):
        if axis in conditions:
            val = conditions[axis]
            if isinstance(val, tuple):
                parts.append(f"{axis}=[{val[0]:.3f}, {val[1]:.3f}]")
            else:
                parts.append(f"{axis}={val:.3f}")
    return ", ".join(parts)


def _print_slice_help():
    """打印切片模式帮助信息."""
    print("\n  ╔══════════════════════════════════════════════════╗")
    print("  ║         交互式切片查看模式 (Slice Viewer)        ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║  x=0.3          显示 x≈0.3 的 YZ 切面          ║")
    print("  ║  y=-0.1         显示 y≈-0.1 的 XZ 切面         ║")
    print("  ║  z=0.5          显示 z≈0.5 的 XY 切面          ║")
    print("  ║  x=0.3 z=0.5   组合切片 (交集)                 ║")
    print("  ║  x=0.1:0.3     范围切片 (0.1 ≤ x ≤ 0.3)       ║")
    print("  ║  all / reset    恢复显示全部点                  ║")
    print("  ║  list x         列出 X 轴所有可用切片值         ║")
    print("  ║  info           当前切片统计信息                ║")
    print("  ║  help           显示此帮助                     ║")
    print("  ║  quit / exit    退出                           ║")
    print("  ╚══════════════════════════════════════════════════╝")


def _print_slice_info(
    voxel_centers, reachability, conditions, resolution, min_reachability,
):
    """打印当前切片的详细统计信息."""
    if not conditions:
        mask = reachability > min_reachability
        label = "全部数据"
    else:
        pts, vals = _apply_slice_filter(
            voxel_centers, reachability, conditions,
            resolution, min_reachability,
        )
        label = _format_conditions(conditions)

    print(f"\n  ── 切片统计: {label} ──")

    if not conditions:
        pts = voxel_centers[mask]
        vals = reachability[mask]

    if pts.shape[0] == 0:
        print("    无匹配点")
        return

    n_total = pts.shape[0]
    n_reach = (vals > 0).sum()
    n_full = (vals >= 1.0).sum()
    avg = vals.mean()
    avg_reach = vals[vals > 0].mean() if n_reach > 0 else 0

    print(f"    点数: {n_total}")
    print(f"    X 范围: [{pts[:, 0].min():.3f}, {pts[:, 0].max():.3f}] m")
    print(f"    Y 范围: [{pts[:, 1].min():.3f}, {pts[:, 1].max():.3f}] m")
    print(f"    Z 范围: [{pts[:, 2].min():.3f}, {pts[:, 2].max():.3f}] m")
    print(f"    有可达朝向: {n_reach}/{n_total} ({n_reach/n_total*100:.1f}%)")
    print(f"    全朝向可达: {n_full}/{n_total} ({n_full/n_total*100:.1f}%)")
    print(f"    整体平均可达比例: {avg:.1%}")
    if n_reach > 0:
        print(f"    可达点平均比例: {avg_reach:.1%}")

    # 可达比例分布直方图 (文本)
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist, _ = np.histogram(vals[vals > 0], bins=bins)
    if n_reach > 0:
        print("    可达比例分布:")
        max_bar = 30
        max_count = hist.max() if hist.max() > 0 else 1
        for i in range(len(hist)):
            lo = bins[i]
            hi = bins[i + 1]
            bar_len = int(hist[i] / max_count * max_bar)
            bar = "█" * bar_len
            label = f"{lo:.0%}-{min(hi, 1.0):.0%}"
            print(f"      {label:>9s} │{bar} {hist[i]}")


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="JAKA 7-DOF 旋转可达性工作空间分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resolution", type=float, default=0.05,
        help="Voxel 网格分辨率 (m) (默认: 0.05)",
    )
    parser.add_argument(
        "--num-sphere", type=int, default=20,
        help="球面方向采样数 (Fibonacci lattice) (默认: 20)",
    )
    parser.add_argument(
        "--num-roll", type=int, default=8,
        help="每个球面方向的自旋角采样数 (默认: 8)",
    )
    parser.add_argument(
        "--fk-samples", type=int, default=50000,
        help="FK 采样数量, 用于估计工作空间范围 (默认: 50000)",
    )
    parser.add_argument(
        "--self-collision", action="store_true",
        help="开启自碰撞检测",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=20,
        help="IK 求解器 seed 数量 (默认: 20)",
    )
    parser.add_argument(
        "--ik-batch-size", type=int, default=2048,
        help="IK 批量求解大小 (CUDA Graph 模式下固定, 默认: 2048)",
    )
    parser.add_argument(
        "--no-cuda-graph", action="store_true",
        help="禁用 CUDA Graph (允许动态 batch size, 但慢 ~10x)",
    )
    parser.add_argument(
        "--early-stop", type=int, default=0,
        help="早停阈值: 连续失败多少个朝向后跳过该 voxel (0=不早停, 推荐 200~500)",
    )
    parser.add_argument(
        "--fk-prefilter", type=int, default=0,
        help="FK 预筛选采样数 (0=不预筛选, 推荐 200000~500000)",
    )
    parser.add_argument(
        "--checkpoint", action="store_true",
        help="启用断点续算 (定期保存进度, 中断后可继续)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认: ./workspace_reachability_results)",
    )
    parser.add_argument(
        "--publish", action="store_true",
        help="计算完成后发布 PointCloud2 到 RViz",
    )
    parser.add_argument(
        "--topic", type=str, default="/workspace_reachability",
        help="RViz 点云话题 (默认: /workspace_reachability)",
    )
    parser.add_argument(
        "--frame", type=str, default="LINK_BASE",
        help="RViz 坐标系 (默认: LINK_BASE)",
    )
    parser.add_argument(
        "--min-reachability", type=float, default=0.0,
        help="最低可达比例阈值, 低于此值的点不显示 (默认: 0.0)",
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="从已有结果目录加载, 跳过计算直接发布到 RViz",
    )
    parser.add_argument(
        "--slice", action="store_true",
        help="进入交互式切片查看模式 (需配合 --publish 使用)",
    )
    args = parser.parse_args()

    # 输出目录
    if args.output_dir is None:
        suffix = "_selfcol" if args.self_collision else ""
        args.output_dir = os.path.join(
            os.path.dirname(__file__),
            f"workspace_reachability_results{suffix}",
        )

    # 加载 resolution (从 metadata 或命令行)
    load_resolution = args.resolution

    # ---- 模式 1: 从已有结果加载 ----
    if args.load:
        print(f"\n  从已有结果加载: {args.load}")
        voxel_centers = np.load(os.path.join(args.load, "voxel_centers.npy"))
        reachability = np.load(os.path.join(args.load, "reachability.npy"))
        # 尝试从 metadata 读取 resolution
        meta_path = os.path.join(args.load, "metadata.npz")
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            if "resolution" in meta:
                load_resolution = float(meta["resolution"])
        print(f"  加载 {voxel_centers.shape[0]} 个 voxel, 分辨率 {load_resolution}m")

        publish_to_rviz(
            voxel_centers, reachability,
            frame_id=args.frame,
            topic=args.topic,
            min_reachability=args.min_reachability,
            interactive_slice=args.slice,
            resolution=load_resolution,
        )
        return

    # ---- 模式 2: 完整计算 ----
    print(f"\n{'#'*60}")
    print(f"  JAKA 7-DOF 旋转可达性工作空间分析")
    total_orient = args.num_sphere * args.num_roll
    print(f"  网格分辨率: {args.resolution}m")
    print(
        f"  朝向采样: {args.num_sphere} 球面方向"
        f" × {args.num_roll} 自旋角 = {total_orient} 个"
    )
    print(f"  自碰撞检测: {'开启' if args.self_collision else '关闭'}")
    print(f"  输出目录: {args.output_dir}")
    print(f"{'#'*60}")

    use_cuda_graph = not args.no_cuda_graph

    # 初始化 IK 求解器
    print("\n  初始化 CuRobo IK 求解器...")
    ik_solver = create_ik_solver(
        self_collision=args.self_collision,
        num_seeds=args.num_seeds,
        use_cuda_graph=use_cuda_graph,
    )
    print(f"    关节: {ik_solver.joint_names}")
    print(f"    自由度: {ik_solver.dof}")
    print(f"    CUDA Graph: {'开启' if use_cuda_graph else '关闭'}")

    # Step 1: FK 估计范围
    min_bounds, max_bounds = estimate_workspace_bounds(
        ik_solver, num_samples=args.fk_samples
    )

    # Step 2: 生成 voxel 网格
    voxel_centers = generate_voxel_centers(min_bounds, max_bounds, args.resolution)

    # 生成朝向采样 (球面方向 × 自旋角)
    print(
        f"\n  [朝向采样] Fibonacci lattice"
        f" {args.num_sphere} 方向 × {args.num_roll} 自旋角"
    )
    orientations = generate_so3_orientations(args.num_sphere, args.num_roll)
    print(f"    生成 {orientations.shape[0]} 个均匀朝向 (wxyz 四元数)")

    # FK 预筛选 (可选)
    reachable_mask = None
    if args.fk_prefilter > 0:
        reachable_mask = fk_prefilter_voxels(
            ik_solver, voxel_centers, args.resolution,
            num_fk_samples=args.fk_prefilter,
        )

    # Step 3: IK 可达性计算 (GPU 加速)
    reachability = compute_reachability(
        ik_solver, voxel_centers, orientations,
        ik_batch_size=args.ik_batch_size,
        use_cuda_graph=use_cuda_graph,
        early_stop_threshold=args.early_stop,
        checkpoint_dir=args.output_dir if args.checkpoint else None,
        reachable_mask=reachable_mask,
    )

    # 保存结果
    save_results(
        args.output_dir, voxel_centers, reachability,
        orientations, args.resolution,
    )

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"  分析完成!")
    print(f"  总 voxel 数: {voxel_centers.shape[0]}")
    print(f"  有可达朝向: {(reachability > 0).sum()}")
    print(f"  全朝向可达: {(reachability >= 1.0).sum()}")
    if (reachability > 0).sum() > 0:
        print(f"  可达 voxel 平均比例: {reachability[reachability > 0].mean():.1%}")
    print(f"{'='*60}")

    # 发布到 RViz
    if args.publish:
        publish_to_rviz(
            voxel_centers, reachability,
            frame_id=args.frame,
            topic=args.topic,
            min_reachability=args.min_reachability,
            interactive_slice=args.slice,
            resolution=args.resolution,
        )
    else:
        print(f"\n  提示: 添加 --publish 参数可将结果发布到 RViz")
        print(f"  或使用 --load {args.output_dir} --publish 从已有结果加载并发布")
        if not args.slice:
            print(f"  添加 --slice 参数可进入交互式切片查看模式")


if __name__ == "__main__":
    main()
