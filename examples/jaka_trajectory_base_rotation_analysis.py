#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAKA 机器人轨迹数据IK解算与Base多自由度分析

功能说明:
  1. 读取多组任务的轨迹数据（pose顺序：位置xyz，旋转xyzw）
  2. 对每个pose进行IK解算，计算成功率
  3. 对base进行4个独立自由度的扫描测试：
     - 绕X轴旋转：±60度，步长5度
     - 绕Z轴旋转：±60度，步长5度
     - X轴平移：±15cm，步长3cm
     - Y轴平移：±15cm，步长3cm
  4. 每个自由度独立分析，比较不同参数下的IK成功率

数据格式 (每行至少10个字段):
  字段1:   时间戳
  字段2:   机械臂名称
  字段3:   状态
  字段4-6: 目标位置 (x, y, z)
  字段7-10: 目标四元数 (x, y, z, w)  ← 注意是 xyzw 顺序

使用方法:
  conda activate curobo
  python examples/jaka_trajectory_base_rotation_analysis.py \
    --data-dir ./xiyiye \
    --robot-config jaka.yml \
    --output-dir ./base_rotation_results \
    --arm left
"""

# 标准库
import argparse
from pathlib import Path

# 第三方库
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# 启用 cuDNN benchmark 以加速 CUDA 运算
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_trajectory_data(file_path):
    """加载轨迹数据文件

    数据格式：时间戳 机器人名称 状态 x y z qx qy qz qw ...
    数据文件中四元数顺序为 xyzw，读取后转换为 CuRobo 的 wxyz 格式。

    Returns:
        poses: np.ndarray (N, 7) - [x, y, z, qw, qx, qy, qz] (wxyz格式)
        labels: list[str] - 状态标签
    """
    poses = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            # 提取位置 xyz
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            # 提取四元数：数据文件中为 xyzw 顺序
            qx, qy, qz, qw = (
                float(parts[6]), float(parts[7]),
                float(parts[8]), float(parts[9]),
            )

            # 存储为 CuRobo 的 wxyz 格式: [x, y, z, qw, qx, qy, qz]
            poses.append([x, y, z, qw, qx, qy, qz])
            labels.append(parts[2])

    return np.array(poses, dtype=np.float32), labels


# ── 4个独立自由度的配置 ──────────────────────────────────
# 每个自由度: (名称, 类型, 参数列表, 单位, X轴标签)
DOF_CONFIGS = [
    {
        "name": "rot_x",
        "label": "绕X轴旋转",
        "values": list(range(-60, 65, 5)),   # ±60°, 步长5°
        "unit": "度",
        "xlabel": "Base绕X轴旋转角度 (度)",
    },
    {
        "name": "rot_z",
        "label": "绕Z轴旋转",
        "values": list(range(-60, 65, 5)),   # ±60°, 步长5°
        "unit": "度",
        "xlabel": "Base绕Z轴旋转角度 (度)",
    },
    {
        "name": "trans_x",
        "label": "X轴平移",
        "values": list(range(-15, 18, 3)),    # ±15cm, 步长3cm
        "unit": "cm",
        "xlabel": "Base沿X轴平移距离 (cm)",
    },
    {
        "name": "trans_y",
        "label": "Y轴平移",
        "values": list(range(-15, 18, 3)),    # ±15cm, 步长3cm
        "unit": "cm",
        "xlabel": "Base沿Y轴平移距离 (cm)",
    },
]


def create_base_transform(dof_name, value):
    """根据自由度类型和参数值创建4x4齐次变换矩阵

    Args:
        dof_name: 自由度名称 ("rot_x", "rot_z", "trans_x", "trans_y")
        value: 参数值（旋转为度，平移为cm）

    Returns:
        4x4齐次变换矩阵
    """
    transform = np.eye(4)

    if dof_name == "rot_x":
        rot = R.from_euler('x', np.radians(value)).as_matrix()
        transform[:3, :3] = rot
    elif dof_name == "rot_z":
        rot = R.from_euler('z', np.radians(value)).as_matrix()
        transform[:3, :3] = rot
    elif dof_name == "trans_x":
        transform[0, 3] = value / 100.0  # cm → m
    elif dof_name == "trans_y":
        transform[1, 3] = value / 100.0  # cm → m
    else:
        raise ValueError(f"未知的自由度类型: {dof_name}")

    return transform


def transform_poses_batch(poses, transform):
    """对一批pose进行坐标变换（向量化操作）

    Args:
        poses: np.ndarray (N, 7) - [x, y, z, qw, qx, qy, qz] (wxyz格式)
        transform: 4x4齐次变换矩阵

    Returns:
        np.ndarray (N, 7) - 变换后的pose，仍为 wxyz 格式
    """
    n = len(poses)
    positions = poses[:, :3]       # (N, 3)
    quat_wxyz = poses[:, 3:]       # (N, 4) wxyz

    # --- 变换位置 ---
    ones = np.ones((n, 1), dtype=np.float32)
    positions_homo = np.hstack([positions, ones])  # (N, 4)
    transformed_positions = (transform @ positions_homo.T).T[:, :3]  # (N, 3)

    # --- 变换旋转 ---
    # scipy 使用 xyzw 格式，需要从 wxyz 转换
    quat_xyzw = np.column_stack([
        quat_wxyz[:, 1], quat_wxyz[:, 2],
        quat_wxyz[:, 3], quat_wxyz[:, 0],
    ])  # (N, 4) xyzw

    rotations = R.from_quat(quat_xyzw)
    base_rotation = R.from_matrix(transform[:3, :3])
    transformed_rotations = base_rotation * rotations
    result_xyzw = transformed_rotations.as_quat()  # (N, 4) xyzw

    # 转回 wxyz 格式
    result_wxyz = np.column_stack([
        result_xyzw[:, 3], result_xyzw[:, 0],
        result_xyzw[:, 1], result_xyzw[:, 2],
    ])  # (N, 4) wxyz

    return np.hstack([transformed_positions, result_wxyz]).astype(np.float32)


def batch_ik_solve(ik_solver, poses, batch_size=100):
    """批量进行IK求解

    Args:
        ik_solver: IK求解器
        poses: np.ndarray (N, 7) - [x, y, z, qw, qx, qy, qz] (wxyz格式)
        batch_size: 每批处理的pose数量

    Returns:
        成功率 (float)
    """
    tensor_args = TensorDeviceType()

    total_success = 0
    total_poses = len(poses)

    for i in range(0, total_poses, batch_size):
        batch = poses[i:i + batch_size]

        # 直接从 numpy 数组创建 tensor（避免列表推导式的性能问题）
        positions = torch.as_tensor(
            batch[:, :3].copy(),
            device=tensor_args.device,
            dtype=tensor_args.dtype,
        )
        quaternions = torch.as_tensor(
            batch[:, 3:].copy(),
            device=tensor_args.device,
            dtype=tensor_args.dtype,
        )

        goal = Pose(positions, quaternions)

        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()

        batch_success = torch.count_nonzero(result.success).item()
        total_success += batch_success

        print(f"  批次 {i // batch_size + 1}: "
              f"{batch_success}/{len(batch)} 成功")

    return total_success / total_poses


def analyze_base_dof_effect(data_dir, robot_config_file, output_dir,
                            batch_size=100, arm="left"):
    """分析base多自由度变换对IK成功率的影响

    对4个独立自由度分别扫描，每个自由度独立输出结果。

    Args:
        data_dir: 轨迹数据目录
        robot_config_file: 机器人配置文件
        output_dir: 结果输出目录
        batch_size: IK求解批量大小
        arm: 机械臂选择，"left" 或 "right"
    """
    print("=" * 70)
    print("JAKA 机器人Base多自由度对IK成功率影响分析")
    print("=" * 70)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 初始化IK求解器
    tensor_args = TensorDeviceType()

    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_config_file))
        ["robot_cfg"]
    )

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    )
    ik_solver = IKSolver(ik_config)

    # 查找轨迹数据文件
    data_files = sorted(Path(data_dir).glob(f"{arm}_umi*.txt"))
    if not data_files:
        print(f"在目录 {data_dir} 中未找到 {arm} 臂的轨迹数据文件")
        return

    print(f"找到 {len(data_files)} 个 {arm} 臂轨迹数据文件")
    print(f"将测试 {len(DOF_CONFIGS)} 个独立自由度")

    # ── 逐文件、逐自由度分析 ──
    for data_file in data_files:
        print(f"\n{'━' * 60}")
        print(f"分析文件: {data_file.name}")

        poses, labels = load_trajectory_data(data_file)
        total_count = len(poses)
        success_count = sum(1 for lb in labels if lb == "success")
        recorded_success_rate = (
            success_count / total_count if total_count > 0 else 0.0
        )
        print(f"  加载了 {total_count} 个pose (四元数已转为wxyz格式)")
        print(f"  原始记录: {success_count}/{total_count} success"
              f" ({recorded_success_rate:.2%})")

        # 先计算一次原始IK成功率（所有自由度共享）
        print("\n  [基准] 计算原始pose的IK成功率...")
        original_rate = batch_ik_solve(ik_solver, poses, batch_size)
        print(f"  → 基准成功率: {original_rate:.2%}")

        # 每个自由度独立扫描
        dof_results = {}  # {dof_name: {value: rate}}
        for dof_cfg in DOF_CONFIGS:
            dof_name = dof_cfg["name"]
            dof_label = dof_cfg["label"]
            values = dof_cfg["values"]
            unit = dof_cfg["unit"]

            print(f"\n  {'─' * 40}")
            print(f"  自由度: {dof_label}")

            dof_result = {}
            for val in values:
                if val == 0:
                    # 0值直接复用基准结果
                    dof_result[0] = original_rate
                    print(f"    [{val:+d}{unit}] 基准: "
                          f"{original_rate:.2%}")
                    continue

                print(f"    [{val:+d}{unit}] 计算中...")
                transform = create_base_transform(dof_name, val)
                transformed_poses = transform_poses_batch(
                    poses, transform
                )
                rate = batch_ik_solve(
                    ik_solver, transformed_poses, batch_size
                )
                dof_result[val] = rate
                print(f"    → 成功率: {rate:.2%}")

            dof_results[dof_name] = dof_result

        # ── 保存每个自由度的结果 ──
        stem = Path(data_file.name).stem
        for dof_cfg in DOF_CONFIGS:
            dof_name = dof_cfg["name"]
            save_single_result(
                data_file.name,
                dof_results[dof_name],
                output_dir,
                recorded_success_rate,
                dof_cfg=dof_cfg,
            )
            generate_single_plot(
                data_file.name,
                dof_results[dof_name],
                output_dir,
                recorded_success_rate,
                dof_cfg=dof_cfg,
            )

        # 为当前文件生成4自由度对比图
        generate_file_comparison_plot(
            data_file.name, dof_results, output_dir,
            recorded_success_rate,
        )

    print(f"\n分析完成！结果保存在: {output_dir}")


def save_single_result(filename, file_results, output_dir,
                       recorded_success_rate=None, dof_cfg=None):
    """为单个txt文件的单个自由度保存分析结果

    Args:
        filename: 数据文件名
        file_results: {参数值: 成功率} 字典
        output_dir: 输出目录
        recorded_success_rate: 原始数据文件中记录的success比例
        dof_cfg: 自由度配置字典
    """
    output_path = Path(output_dir)
    stem = Path(filename).stem
    dof_name = dof_cfg["name"] if dof_cfg else "rot_x"
    dof_label = dof_cfg["label"] if dof_cfg else "绕X轴旋转"
    unit = dof_cfg["unit"] if dof_cfg else "度"

    # 文件名带自由度后缀
    out_stem = f"{stem}_{dof_name}"

    # 找到最佳参数
    best_val = max(file_results, key=file_results.get)
    best_rate = file_results[best_val]
    original_rate = file_results[0]

    # 保存详细结果
    with open(output_path / f"{out_stem}_results.txt", "w",
              encoding='utf-8') as f:
        f.write(f"文件: {filename}\n")
        f.write(f"自由度: {dof_label}\n")
        f.write(f"Base {dof_label}对IK成功率影响分析\n")
        f.write("=" * 50 + "\n\n")
        if recorded_success_rate is not None:
            f.write(f"数据采集时记录的success比例: "
                    f"{recorded_success_rate:.2%}\n")
        f.write(f"IK解算成功率 (0{unit}): {original_rate:.2%}\n")
        f.write(f"最佳参数: {best_val:+d}{unit}, "
                f"成功率: {best_rate:.2%}\n")
        f.write(f"提升 (相对0{unit}): "
                f"{best_rate - original_rate:+.2%}\n")
        f.write("\n" + "-" * 30 + "\n")

        for val in sorted(file_results.keys()):
            rate = file_results[val]
            marker = " <-- 最佳" if val == best_val else ""
            marker += " <-- 基准" if val == 0 else ""
            f.write(f"  {val:+4d}{unit}: {rate:.2%}{marker}\n")

    # 保存CSV格式结果
    with open(output_path / f"{out_stem}_results.csv", "w",
              encoding='utf-8') as f:
        if recorded_success_rate is not None:
            f.write(f"# recorded_success_rate="
                    f"{recorded_success_rate:.6f}\n")
        f.write(f"# dof={dof_name} unit={unit}\n")
        f.write("value,success_rate\n")
        for val in sorted(file_results.keys()):
            f.write(f"{val},{file_results[val]:.6f}\n")

    print(f"  结果已保存: {out_stem}_results.txt / .csv")


def generate_single_plot(filename, file_results, output_dir,
                         recorded_success_rate=None, dof_cfg=None):
    """为单个txt文件的单个自由度生成可视化图表

    Args:
        filename: 数据文件名
        file_results: {参数值: 成功率} 字典
        output_dir: 输出目录
        recorded_success_rate: 原始数据文件中记录的success比例
        dof_cfg: 自由度配置字典
    """
    output_path = Path(output_dir)
    stem = Path(filename).stem
    dof_name = dof_cfg["name"] if dof_cfg else "rot_x"
    dof_label = dof_cfg["label"] if dof_cfg else "绕X轴旋转"
    unit = dof_cfg["unit"] if dof_cfg else "度"
    xlabel = dof_cfg["xlabel"] if dof_cfg else "Base绕X轴旋转角度 (度)"

    out_stem = f"{stem}_{dof_name}"

    values = sorted(file_results.keys())
    rates = [file_results[v] for v in values]

    best_val = max(file_results, key=file_results.get)
    best_rate = file_results[best_val]
    original_rate = file_results[0]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(values, rates, marker='o', linewidth=2, markersize=5,
            color='#2196F3', label='IK解算成功率')

    # 标注原始记录的success比例（水平基线）
    if recorded_success_rate is not None:
        ax.axhline(y=recorded_success_rate, color='#E91E63',
                   linestyle=':', linewidth=2, alpha=0.8,
                   label=f'采集记录success比例: '
                         f'{recorded_success_rate:.2%}')

    # 标注基准
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.6)
    ax.plot(0, original_rate, 's', color='#FF9800', markersize=10,
            zorder=5,
            label=f'基准 (0{unit}): {original_rate:.2%}')

    # 标注最佳
    ax.plot(best_val, best_rate, '*', color='#4CAF50', markersize=15,
            zorder=5,
            label=f'最佳 ({best_val:+d}{unit}): {best_rate:.2%}')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('IK成功率', fontsize=12)
    ax.set_title(f'{stem}\n{dof_label} vs IK成功率', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    fig.tight_layout()
    fig.savefig(output_path / f'{out_stem}_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  图表已保存: {out_stem}_analysis.png")


def generate_file_comparison_plot(filename, dof_results, output_dir,
                                   recorded_success_rate=None):
    """为单个文件生成4自由度对比图（2x2子图）

    Args:
        filename: 数据文件名
        dof_results: {dof_name: {value: rate}} 字典
        output_dir: 输出目录
        recorded_success_rate: 原始数据文件中记录的success比例
    """
    output_path = Path(output_dir)
    stem = Path(filename).stem

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{stem}\nBase 4自由度对IK成功率影响对比',
                 fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

    for idx, dof_cfg in enumerate(DOF_CONFIGS):
        ax = axes[idx // 2][idx % 2]
        dof_name = dof_cfg["name"]
        dof_label = dof_cfg["label"]
        unit = dof_cfg["unit"]
        xlabel = dof_cfg["xlabel"]

        result = dof_results[dof_name]
        values = sorted(result.keys())
        rates = [result[v] for v in values]

        best_val = max(result, key=result.get)
        best_rate = result[best_val]
        original_rate = result[0]

        ax.plot(values, rates, marker='o', linewidth=2, markersize=4,
                color=colors[idx], label='IK成功率')

        if recorded_success_rate is not None:
            ax.axhline(y=recorded_success_rate, color='#E91E63',
                       linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'采集success: '
                             f'{recorded_success_rate:.1%}')

        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.plot(0, original_rate, 's', color='#FF9800',
                markersize=8, zorder=5)
        ax.plot(best_val, best_rate, '*', color='#4CAF50',
                markersize=12, zorder=5,
                label=f'最佳: {best_val:+d}{unit} '
                      f'({best_rate:.1%})')

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('IK成功率', fontsize=10)
        ax.set_title(dof_label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path / f'{stem}_4dof_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  4自由度对比图已保存: {stem}_4dof_comparison.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='JAKA机器人轨迹数据Base旋转分析')
    parser.add_argument('--data-dir', type=str, default='./xiyiye',
                        help='轨迹数据目录路径')
    parser.add_argument('--robot-config', type=str, default='jaka.yml',
                        help='机器人配置文件名称')
    parser.add_argument('--output-dir', type=str,
                        default='./base_rotation_results',
                        help='结果输出目录')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='IK求解批量大小 (越大GPU利用率越高，但显存占用也越大)')
    parser.add_argument('--arm', type=str, default='left',
                        choices=['left', 'right'],
                        help='选择机械臂: left 或 right (默认 left)')

    args = parser.parse_args()

    analyze_base_dof_effect(
        data_dir=args.data_dir,
        robot_config_file=args.robot_config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        arm=args.arm,
    )


if __name__ == "__main__":
    main()