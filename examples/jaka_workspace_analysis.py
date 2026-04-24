#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAKA 7-DOF 机械臂工作空间分析工具

支持两种分析模式:
  方案A (FK正向映射): 关节空间采样 → FK计算末端位置 → 工作空间点云
  方案B (IK逆向探测): 笛卡尔空间网格采样 → IK求解 → 可达性地图

使用方法:
  # 方案A: FK正向映射 (默认, 推荐先做)
  python examples/jaka_workspace_analysis.py --mode fk --num-samples 100000

  # 方案B: IK逆向探测
  python examples/jaka_workspace_analysis.py --mode ik --grid-resolution 0.05

  # 自定义输出目录
  python examples/jaka_workspace_analysis.py --mode fk --output-dir ./workspace_results

  # 开启自碰撞检测
  python examples/jaka_workspace_analysis.py --mode fk --self-collision
"""

import os
import time
import argparse
from typing import Tuple, Optional

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无头模式，不需要 GUI
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
#  工具函数
# ============================================================

def create_ik_solver(
    self_collision: bool = False,
    num_seeds: int = 20,
) -> IKSolver:
    """创建并返回 CuRobo IK 求解器实例。"""
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
        use_cuda_graph=False,  # 关闭以支持动态 batch size
    )
    return IKSolver(ik_config)


def ensure_output_dir(output_dir: str) -> str:
    """确保输出目录存在。"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ============================================================
#  方案A: FK 正向映射
# ============================================================

def run_fk_analysis(
    ik_solver: IKSolver,
    num_samples: int,
    output_dir: str,
    batch_size: int = 10000,
) -> np.ndarray:
    """
    FK 正向映射分析:
    1. 在关节空间采样合法构型 (过滤碰撞/超限)
    2. 批量 FK 计算末端位置
    3. 返回末端位置点云 (N, 3)
    """
    print(f"\n{'='*60}")
    print(f"  方案A: FK 正向映射")
    print(f"  目标采样数: {num_samples}")
    print(f"{'='*60}\n")

    all_positions = []
    collected = 0
    start_time = time.time()

    while collected < num_samples:
        # 每次采样一批
        n_batch = min(batch_size, num_samples - collected)
        print(f"  采样中... 已收集 {collected}/{num_samples}", end="\r")

        # sample_configs 会自动过滤不合法的构型 (碰撞/超限)
        q_samples = ik_solver.sample_configs(n_batch)
        if q_samples.shape[0] == 0:
            print("  [警告] 本批次未采样到合法构型，增大 rejection_ratio 重试...")
            q_samples = ik_solver.sample_configs(n_batch, rejection_ratio=50)
            if q_samples.shape[0] == 0:
                break

        # FK 计算末端位置
        state = ik_solver.fk(q_samples)
        ee_pos = state.ee_position.detach().cpu().numpy()  # (n, 3)
        all_positions.append(ee_pos)
        collected += ee_pos.shape[0]

    elapsed = time.time() - start_time
    positions = np.concatenate(all_positions, axis=0)[:num_samples]
    print(f"\n  采样完成: {positions.shape[0]} 个点, 耗时 {elapsed:.2f}s")

    # 统计
    print_position_stats(positions)

    # 可视化
    plot_3d_point_cloud(positions, output_dir, "fk")
    plot_2d_projections(positions, output_dir, "fk")
    plot_radial_distribution(positions, output_dir, "fk")

    # 保存原始数据
    np.save(os.path.join(output_dir, "fk_positions.npy"), positions)
    print(f"\n  原始数据已保存到: {output_dir}/fk_positions.npy")

    return positions


# ============================================================
#  方案B: IK 逆向探测
# ============================================================

def run_ik_analysis(
    ik_solver: IKSolver,
    grid_resolution: float,
    output_dir: str,
    x_range: Tuple[float, float] = (-1.2, 1.2),
    y_range: Tuple[float, float] = (-1.2, 1.2),
    z_range: Tuple[float, float] = (-1.2, 1.2),
    orientation_wxyz: Optional[np.ndarray] = None,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    IK 逆向探测分析:
    1. 在笛卡尔空间生成均匀网格点
    2. 对每个点尝试 IK 求解
    3. 返回 (grid_points, success_mask)
    """
    # 默认姿态: 末端朝下 (w=0, x=0, y=1, z=0 → 绕Y轴旋转180°)
    if orientation_wxyz is None:
        orientation_wxyz = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # 生成网格
    xs = np.arange(x_range[0], x_range[1] + grid_resolution, grid_resolution)
    ys = np.arange(y_range[0], y_range[1] + grid_resolution, grid_resolution)
    zs = np.arange(z_range[0], z_range[1] + grid_resolution, grid_resolution)
    total_points = len(xs) * len(ys) * len(zs)

    print(f"\n{'='*60}")
    print(f"  方案B: IK 逆向探测")
    print(f"  网格分辨率: {grid_resolution}m")
    print(f"  网格范围: X{x_range}, Y{y_range}, Z{z_range}")
    print(f"  网格维度: {len(xs)} x {len(ys)} x {len(zs)} = {total_points} 个点")
    print(f"  目标姿态 (wxyz): {orientation_wxyz}")
    print(f"{'='*60}\n")

    # 生成所有网格点
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_points = np.stack(
        [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1
    ).astype(np.float32)

    success_mask = np.zeros(grid_points.shape[0], dtype=bool)
    tensor_args = TensorDeviceType()

    # 预热
    warmup_pos = torch.tensor([[0.0, 0.0, 0.5]], device=tensor_args.device, dtype=tensor_args.dtype)
    warmup_quat = torch.tensor(
        [orientation_wxyz], device=tensor_args.device, dtype=tensor_args.dtype
    )
    warmup_goal = Pose(warmup_pos, warmup_quat)
    _ = ik_solver.solve_single(warmup_goal)
    torch.cuda.synchronize()

    start_time = time.time()
    solved_count = 0

    # 分批求解
    for i in range(0, grid_points.shape[0], batch_size):
        batch_end = min(i + batch_size, grid_points.shape[0])
        batch_points = grid_points[i:batch_end]
        n = batch_points.shape[0]

        pos_tensor = torch.tensor(batch_points, device=tensor_args.device, dtype=tensor_args.dtype)
        quat_tensor = torch.tensor(
            orientation_wxyz, device=tensor_args.device, dtype=tensor_args.dtype
        ).unsqueeze(0).expand(n, -1)

        goal = Pose(pos_tensor, quat_tensor)

        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()

        batch_success = result.success.squeeze().cpu().numpy().astype(bool)
        # solve_batch 返回的 success shape 可能是 (n, 1) 或 (n,)
        if batch_success.ndim > 1:
            batch_success = batch_success.squeeze(-1)
        success_mask[i:batch_end] = batch_success[:n]
        solved_count += batch_success.sum()

        progress = batch_end / grid_points.shape[0] * 100
        print(
            f"  进度: {batch_end}/{grid_points.shape[0]} ({progress:.1f}%), "
            f"可达: {solved_count}/{batch_end} ({solved_count/batch_end*100:.1f}%)",
            end="\r",
        )

    elapsed = time.time() - start_time
    total_success = success_mask.sum()

    print(f"\n\n  IK 探测完成: {total_success}/{grid_points.shape[0]} 可达 "
          f"({total_success/grid_points.shape[0]*100:.2f}%), 耗时 {elapsed:.2f}s")

    # 统计
    reachable_points = grid_points[success_mask]
    if reachable_points.shape[0] > 0:
        print_position_stats(reachable_points)

    # 可视化
    plot_3d_point_cloud(reachable_points, output_dir, "ik")
    plot_2d_projections(reachable_points, output_dir, "ik")
    plot_reachability_slices(grid_points, success_mask, xs, ys, zs, output_dir)
    plot_radial_distribution(reachable_points, output_dir, "ik")

    # 保存原始数据
    np.save(os.path.join(output_dir, "ik_grid_points.npy"), grid_points)
    np.save(os.path.join(output_dir, "ik_success_mask.npy"), success_mask)
    print(f"\n  原始数据已保存到: {output_dir}/ik_*.npy")

    return grid_points, success_mask


# ============================================================
#  统计 & 可视化
# ============================================================

def print_position_stats(positions: np.ndarray):
    """打印末端位置统计信息。"""
    distances = np.linalg.norm(positions, axis=1)

    print(f"\n  【位置统计】(共 {positions.shape[0]} 个点)")
    print(f"    X 范围: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}] m")
    print(f"    Y 范围: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}] m")
    print(f"    Z 范围: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}] m")
    print(f"    距原点距离:")
    print(f"      最小: {distances.min():.4f} m")
    print(f"      最大: {distances.max():.4f} m")
    print(f"      均值: {distances.mean():.4f} m")
    print(f"      中位数: {np.median(distances):.4f} m")


def plot_3d_point_cloud(positions: np.ndarray, output_dir: str, prefix: str):
    """绘制 3D 工作空间点云。"""
    if positions.shape[0] == 0:
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 按距原点距离着色
    distances = np.linalg.norm(positions, axis=1)

    # 降采样以加速绘图
    max_plot_points = 20000
    if positions.shape[0] > max_plot_points:
        idx = np.random.choice(positions.shape[0], max_plot_points, replace=False)
        plot_pos = positions[idx]
        plot_dist = distances[idx]
    else:
        plot_pos = positions
        plot_dist = distances

    scatter = ax.scatter(
        plot_pos[:, 0],
        plot_pos[:, 1],
        plot_pos[:, 2],
        c=plot_dist,
        cmap="viridis",
        s=0.5,
        alpha=0.3,
    )
    fig.colorbar(scatter, ax=ax, label="距原点距离 (m)", shrink=0.6)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"JAKA 工作空间 3D 点云 ({prefix.upper()} 模式, N={positions.shape[0]})")

    # 设置等比例坐标轴
    max_range = max(
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min(),
    ) / 2.0
    mid = positions.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    save_path = os.path.join(output_dir, f"{prefix}_3d_pointcloud.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 3D 点云已保存: {save_path}")


def plot_2d_projections(positions: np.ndarray, output_dir: str, prefix: str):
    """绘制 XY / XZ / YZ 三个平面的 2D 投影密度热力图。"""
    if positions.shape[0] == 0:
        return

    planes = [
        ("X", "Y", 0, 1),
        ("X", "Z", 0, 2),
        ("Y", "Z", 1, 2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (xlabel, ylabel, xi, yi) in zip(axes, planes):
        h = ax.hist2d(
            positions[:, xi],
            positions[:, yi],
            bins=100,
            cmap="hot",
            norm=matplotlib.colors.LogNorm(),
        )
        fig.colorbar(h[3], ax=ax, label="点密度 (log)")
        ax.set_xlabel(f"{xlabel} (m)")
        ax.set_ylabel(f"{ylabel} (m)")
        ax.set_title(f"{xlabel}-{ylabel} 平面投影")
        ax.set_aspect("equal")

    fig.suptitle(
        f"JAKA 工作空间 2D 投影 ({prefix.upper()} 模式, N={positions.shape[0]})",
        fontsize=14,
    )
    fig.tight_layout()

    save_path = os.path.join(output_dir, f"{prefix}_2d_projections.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 2D 投影已保存: {save_path}")


def plot_radial_distribution(positions: np.ndarray, output_dir: str, prefix: str):
    """绘制距原点距离的分布直方图。"""
    if positions.shape[0] == 0:
        return

    distances = np.linalg.norm(positions, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(distances, bins=100, color="#76b900", alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.axvline(distances.mean(), color="red", linestyle="--", label=f"均值: {distances.mean():.3f}m")
    ax.axvline(np.median(distances), color="blue", linestyle="--", label=f"中位数: {np.median(distances):.3f}m")
    ax.set_xlabel("距原点距离 (m)")
    ax.set_ylabel("频次")
    ax.set_title(f"JAKA 末端可达距离分布 ({prefix.upper()} 模式)")
    ax.legend()

    save_path = os.path.join(output_dir, f"{prefix}_radial_distribution.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 距离分布已保存: {save_path}")


def plot_reachability_slices(
    grid_points: np.ndarray,
    success_mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    output_dir: str,
):
    """绘制 IK 可达性切片热力图 (取 Z 方向若干切片)。"""
    # 重塑为 3D 网格
    nx, ny, nz = len(xs), len(ys), len(zs)
    success_3d = success_mask.reshape(nx, ny, nz).astype(float)

    # 选取 Z 方向的几个切片
    n_slices = min(6, nz)
    slice_indices = np.linspace(0, nz - 1, n_slices, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()

    for i, zi in enumerate(slice_indices):
        if i >= len(axes):
            break
        ax = axes[i]
        # success_3d[x, y, z] → 取 z=zi 的切片
        slice_data = success_3d[:, :, zi].T  # 转置使 Y 为纵轴
        ax.imshow(
            slice_data,
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            origin="lower",
            cmap="RdYlGn",
            aspect="equal",
            vmin=0,
            vmax=1,
        )
        reachable_ratio = slice_data.mean() * 100
        ax.set_title(f"Z = {zs[zi]:.3f}m (可达率: {reachable_ratio:.1f}%)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("JAKA IK 可达性切片 (绿=可达, 红=不可达)", fontsize=14)
    fig.tight_layout()

    save_path = os.path.join(output_dir, "ik_reachability_slices.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 可达性切片已保存: {save_path}")


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="JAKA 7-DOF 工作空间分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fk", "ik", "both"],
        default="fk",
        help="分析模式: fk=FK正向映射, ik=IK逆向探测, both=两者都做 (默认: fk)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="FK 模式采样数量 (默认: 100000)",
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=0.05,
        help="IK 模式网格分辨率 (m) (默认: 0.05)",
    )
    parser.add_argument(
        "--self-collision",
        action="store_true",
        help="开启自碰撞检测",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: ./workspace_results_<mode>)",
    )
    parser.add_argument(
        "--ik-range",
        type=float,
        default=1.2,
        help="IK 模式网格范围 [-range, range] (m) (默认: 1.2)",
    )
    args = parser.parse_args()

    # 输出目录
    if args.output_dir is None:
        suffix = "_selfcol" if args.self_collision else ""
        args.output_dir = f"./workspace_results_{args.mode}{suffix}"
    ensure_output_dir(args.output_dir)

    print(f"\n{'#'*60}")
    print(f"  JAKA 7-DOF 工作空间分析")
    print(f"  模式: {args.mode}")
    print(f"  自碰撞检测: {'开启' if args.self_collision else '关闭'}")
    print(f"  输出目录: {args.output_dir}")
    print(f"{'#'*60}")

    # 初始化求解器
    print("\n初始化 CuRobo IK 求解器...")
    ik_solver = create_ik_solver(
        self_collision=args.self_collision,
        num_seeds=20 if args.mode in ("ik", "both") else 20,
    )
    print(f"  关节名称: {ik_solver.joint_names}")
    print(f"  自由度: {ik_solver.dof}")

    # 执行分析
    if args.mode in ("fk", "both"):
        run_fk_analysis(ik_solver, args.num_samples, args.output_dir)

    if args.mode in ("ik", "both"):
        r = args.ik_range
        run_ik_analysis(
            ik_solver,
            args.grid_resolution,
            args.output_dir,
            x_range=(-r, r),
            y_range=(-r, r),
            z_range=(-r, r),
        )

    print(f"\n{'#'*60}")
    print(f"  分析完成! 所有结果保存在: {args.output_dir}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
