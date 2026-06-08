#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
奇异性过滤脚本: 基于 GPU 几何法雅可比 + SVD 条件数筛选

用法:
  python filter_singularity.py --input-dir <workspace_out/run_xxx> \
      --cond-threshold 50 --batch 500000

功能:
  1. 加载 workspace_configs.npz (或 shard 文件) 中的 joint_configs
  2. 配置 CudaRobotModel, 设置 link_names 包含所有关节 child link
  3. 分批 FK 获取各 link 位姿, 用几何法计算 6×7 雅可比矩阵
  4. SVD 求条件数 (σ_max / σ_min), 筛选掉条件数超阈值的配置
  5. 记录被过滤掉的奇异配置对应的末端点云位置 (gripper_link, 即最后一个 fixed link)
  6. 输出过滤后的结果 + 奇异点云位置文件

说明:
  点云采集的是末端最后一个 fixed link (gripper_link) 的 pose, 因此当某组 q
  被判定为奇异时, 需要记录该配置对应的 gripper_link 位置, 以便后续在点云中
  标记/剔除这些奇异区域。

  默认阈值设为 50 (而非常见的 100), 目的是将奇异位置周围"快奇异"的配置也
  一并过滤掉, 保证点云中保留的位置都有足够好的操作性。

原理:
  对于旋转关节 i, 几何雅可比为:
    J_v_i = z_i × (p_ee - p_i)   (线速度部分, 3×1)
    J_w_i = z_i                   (角速度部分, 3×1)
  其中 z_i 是关节 i 的旋转轴在世界坐标系下的方向 (即 child link 坐标系的 z 轴),
  p_i 是关节 i 的位置 (即 child link 的原点位置)。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml


def quat_wxyz_to_z_axis(q: torch.Tensor) -> torch.Tensor:
    """
    从 (w,x,y,z) 四元数提取局部 z 轴在世界坐标系下的方向向量。
    即旋转矩阵的第三列。
    q: [..., 4]  (w, x, y, z)
    返回: [..., 3]
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # 旋转矩阵第三列 (z 轴方向)
    zx = 2.0 * (x * z + w * y)
    zy = 2.0 * (y * z - w * x)
    zz = 1.0 - 2.0 * (x * x + y * y)
    return torch.stack([zx, zy, zz], dim=-1)


def compute_geometric_jacobian(
    link_positions: torch.Tensor,
    link_quaternions: torch.Tensor,
    ee_position: torch.Tensor,
    joint_link_indices: list,
) -> torch.Tensor:
    """
    几何法计算 6×7 雅可比矩阵 (全部在 GPU 上批量计算)。

    Args:
        link_positions: [B, n_links, 3] 各 link 在世界坐标系下的位置
        link_quaternions: [B, n_links, 4] 各 link 在世界坐标系下的四元数 (w,x,y,z)
        ee_position: [B, 3] 末端执行器位置
        joint_link_indices: 长度为 dof 的列表, 每个元素是该关节对应的 link 在
                           links_position 中的索引

    Returns:
        J: [B, 6, dof] 几何雅可比矩阵
    """
    B = ee_position.shape[0]
    dof = len(joint_link_indices)
    device = ee_position.device

    J = torch.zeros(B, 6, dof, device=device, dtype=ee_position.dtype)

    for i, li in enumerate(joint_link_indices):
        # 关节 i 的旋转轴方向 (世界坐标系下的 z 轴)
        z_i = quat_wxyz_to_z_axis(link_quaternions[:, li, :])  # [B, 3]
        # 关节 i 的位置
        p_i = link_positions[:, li, :]  # [B, 3]
        # 线速度部分: z_i × (p_ee - p_i)
        r = ee_position - p_i  # [B, 3]
        J[:, 0, i] = z_i[:, 1] * r[:, 2] - z_i[:, 2] * r[:, 1]
        J[:, 1, i] = z_i[:, 2] * r[:, 0] - z_i[:, 0] * r[:, 2]
        J[:, 2, i] = z_i[:, 0] * r[:, 1] - z_i[:, 1] * r[:, 0]
        # 角速度部分: z_i
        J[:, 3, i] = z_i[:, 0]
        J[:, 4, i] = z_i[:, 1]
        J[:, 5, i] = z_i[:, 2]

    return J


def compute_condition_number(J: torch.Tensor) -> torch.Tensor:
    """
    计算雅可比矩阵的条件数 (σ_max / σ_min)。

    Args:
        J: [B, 6, 7] 雅可比矩阵

    Returns:
        cond: [B] 条件数
    """
    # SVD: J = U @ diag(S) @ Vh, S: [B, min(6,7)] = [B, 6]
    S = torch.linalg.svdvals(J)  # [B, 6]
    s_max = S[:, 0]
    s_min = S[:, -1]
    # 避免除零: 当 s_min ≈ 0 时条件数趋于无穷 (奇异);
    # 当 s_max 也 ≈ 0 时 (零雅可比/完全退化), 同样标记为奇异
    cond = torch.where(
        s_max < 1e-10,
        torch.tensor(float("inf"), device=J.device, dtype=J.dtype),
        s_max / s_min.clamp_min(1e-10),
    )
    return cond


def main():
    ap = argparse.ArgumentParser(description="基于几何雅可比条件数的奇异性过滤")
    ap.add_argument("--input-dir", type=str, required=True,
                    help="workspace 采样输出目录 (包含 workspace_configs.npz 或 shard 文件)")
    ap.add_argument("--cond-threshold", type=float, default=75.0,
                    help="条件数阈值, 超过此值的配置被视为接近奇异 (默认 75, 比常规 100 更严格以覆盖快奇异区域)")
    ap.add_argument("--batch", type=int, default=500_000,
                    help="每批处理的样本数 (默认 500000)")
    ap.add_argument("--robot", type=str, default="marvin_left_arm.yml",
                    help="机器人配置文件名")
    ap.add_argument("--ee-link", type=str, default="gripper_link",
                    help="末端执行器 link 名称 (最后一个 fixed link, 点云采集的位置)")
    ap.add_argument("--output-suffix", type=str, default="_filtered",
                    help="输出文件后缀 (默认 _filtered)")
    ap.add_argument("--save-cond", action="store_true",
                    help="保存每个配置的条件数到输出文件")
    ap.add_argument("--save-singular-positions", action="store_true", default=True,
                    help="保存被过滤掉的奇异配置对应的末端点云位置 (默认开启)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"输入目录不存在: {input_dir}"

    # ========== 1. 加载关节配置 ==========
    print("=" * 60)
    print("奇异性过滤: 几何雅可比 + SVD 条件数")
    print("=" * 60)
    print(f"  输入目录: {input_dir}")
    print(f"  条件数阈值: {args.cond_threshold}")
    print(f"  批大小: {args.batch:,}")

    # 查找配置文件 (单文件或分片)
    single_file = input_dir / "workspace_configs.npz"
    shard_files = sorted(input_dir.glob("workspace_configs_shard*.npz"))

    if single_file.exists():
        config_files = [single_file]
    elif len(shard_files) > 0:
        config_files = shard_files
    else:
        raise FileNotFoundError(
            f"在 {input_dir} 中找不到 workspace_configs.npz 或 workspace_configs_shard*.npz"
        )
    print(f"  配置文件数: {len(config_files)}")

    # ========== 2. 构建 CudaRobotModel (包含所有关节 link) ==========
    # 为了获取每个关节的位姿, 需要在 link_names 中包含所有关节的 child link
    # 运动链: Base_L -> Link1_L -> Link2_L -> ... -> Link7_L -> gripper_link
    # 关节 i 的旋转轴 = Link{i}_L 坐标系的 z 轴
    # 关节 i 的位置 = Link{i}_L 的原点
    joint_child_links = [
        "Link1_L", "Link2_L", "Link3_L", "Link4_L",
        "Link5_L", "Link6_L", "Link7_L",
    ]

    cfg_dict = load_yaml(join_path(get_robot_configs_path(), args.robot))
    cfg_dict["robot_cfg"]["kinematics"]["ee_link"] = args.ee_link
    # 设置 link_names 包含所有关节 child link + ee_link
    cfg_dict["robot_cfg"]["kinematics"]["link_names"] = joint_child_links + [args.ee_link]

    robot_cfg = RobotConfig.from_dict(cfg_dict["robot_cfg"])
    km = CudaRobotModel(robot_cfg.kinematics)

    dev = torch.device("cuda:0")
    dof = km.get_dof()
    print(f"  DOF: {dof}")
    print(f"  link_names (输出): {km.link_names}")
    print(f"  joint_names: {km.joint_names}")

    # 确定每个关节对应的 link 在 links_position 中的索引
    joint_link_indices = []
    for jl in joint_child_links:
        idx = km.link_names.index(jl)
        joint_link_indices.append(idx)
    print(f"  joint_link_indices: {joint_link_indices}")

    # ========== 3. 逐文件/逐批处理 ==========
    total_samples = 0
    total_kept = 0
    total_dropped = 0
    t_start = time.time()

    for fi, config_file in enumerate(config_files):
        print(f"\n--- 处理文件 [{fi+1}/{len(config_files)}]: {config_file.name} ---")
        data = np.load(config_file, allow_pickle=True)
        joint_configs = data["joint_configs"]  # [N, dof]
        voxel_linear_idx = data["voxel_linear_idx"]  # [N]
        global_sample_idx = data.get("global_sample_idx", None)

        N = len(joint_configs)
        total_samples += N
        print(f"  样本数: {N:,}")

        # 存储过滤结果
        keep_mask = np.zeros(N, dtype=bool)
        cond_values = np.zeros(N, dtype=np.float32) if args.save_cond else None
        # 存储奇异配置对应的末端点云位置 (gripper_link 位置)
        singular_positions = [] if args.save_singular_positions else None

        n_batches = (N + args.batch - 1) // args.batch
        for bi in range(n_batches):
            s = bi * args.batch
            e = min(s + args.batch, N)
            B = e - s

            q = torch.from_numpy(joint_configs[s:e]).to(dev)

            # FK: 获取所有 link 的位姿
            state = km.get_state(q)
            link_pos = state.links_position      # [B, n_links, 3]
            link_quat = state.links_quaternion   # [B, n_links, 4]
            ee_pos = state.ee_position           # [B, 3]

            # 计算几何雅可比 [B, 6, 7]
            J = compute_geometric_jacobian(
                link_pos, link_quat, ee_pos, joint_link_indices
            )

            # 计算条件数
            cond = compute_condition_number(J)  # [B]

            # 筛选
            mask = cond < args.cond_threshold
            mask_np = mask.cpu().numpy()
            keep_mask[s:e] = mask_np

            if args.save_cond:
                cond_values[s:e] = cond.cpu().numpy()

            # 记录奇异配置对应的末端点云位置 (gripper_link = ee_position)
            if singular_positions is not None:
                singular_mask = ~mask  # 被过滤掉的
                if singular_mask.any():
                    singular_ee = ee_pos[singular_mask].cpu().numpy()  # [n_singular, 3]
                    singular_positions.append(singular_ee)

            kept = mask_np.sum()
            total_kept += kept
            total_dropped += (B - kept)

            # 释放 GPU 内存
            del q, state, link_pos, link_quat, ee_pos, J, cond, mask

            if (bi + 1) % 20 == 0 or (bi + 1) == n_batches:
                elapsed = time.time() - t_start
                progress = (total_kept + total_dropped) / max(total_samples, 1) * 100
                print(f"    batch [{bi+1}/{n_batches}] "
                      f"kept={kept}/{B} "
                      f"elapsed={elapsed:.1f}s")

        # 保存过滤后的结果
        filtered_configs = joint_configs[keep_mask]
        filtered_voxel = voxel_linear_idx[keep_mask]
        filtered_global_idx = global_sample_idx[keep_mask] if global_sample_idx is not None else None

        # 确定输出文件名
        stem = config_file.stem  # e.g. "workspace_configs" or "workspace_configs_shard000"
        out_name = f"{stem}{args.output_suffix}.npz"
        out_path = input_dir / out_name

        save_dict = {
            "joint_configs": filtered_configs,
            "voxel_linear_idx": filtered_voxel,
            "dof": np.int32(dof),
            "joint_names": data.get("joint_names", np.array(list(km.joint_names))),
            "cond_threshold": np.float32(args.cond_threshold),
        }
        if filtered_global_idx is not None:
            save_dict["global_sample_idx"] = filtered_global_idx
        if args.save_cond:
            save_dict["condition_numbers"] = cond_values[keep_mask]

        np.savez_compressed(out_path, **save_dict)
        file_kept = keep_mask.sum()
        print(f"  [SAVE] {out_path.name}: {file_kept:,}/{N:,} 保留 "
              f"({100.0*file_kept/max(N,1):.2f}%), "
              f"文件大小={out_path.stat().st_size/1e6:.1f}MB")

        # 保存奇异配置对应的末端点云位置
        if singular_positions is not None and len(singular_positions) > 0:
            all_singular_pos = np.concatenate(singular_positions, axis=0)  # [n_total_singular, 3]
            singular_out_name = f"{stem}_singular_positions.npz"
            singular_out_path = input_dir / singular_out_name
            np.savez_compressed(
                singular_out_path,
                ee_positions=all_singular_pos,
                cond_threshold=np.float32(args.cond_threshold),
            )
            print(f"  [SAVE] {singular_out_name}: {len(all_singular_pos):,} 个奇异点云位置, "
                  f"文件大小={singular_out_path.stat().st_size/1e6:.1f}MB")
            del all_singular_pos

        del data, joint_configs, voxel_linear_idx, keep_mask, filtered_configs, filtered_voxel

    # ========== 4. 汇总统计 ==========
    total_t = time.time() - t_start
    throughput = total_samples / max(total_t, 1e-9) / 1e6
    drop_rate = 100.0 * total_dropped / max(total_samples, 1)

    print(f"\n{'=' * 60}")
    print(f"完成! 总耗时: {total_t:.1f}s, 吞吐量: {throughput:.2f} M samples/s")
    print(f"  总样本数: {total_samples:,}")
    print(f"  保留: {total_kept:,} ({100.0*total_kept/max(total_samples,1):.2f}%)")
    print(f"  丢弃 (奇异): {total_dropped:,} ({drop_rate:.2f}%)")
    print(f"  条件数阈值: {args.cond_threshold}")
    print(f"{'=' * 60}")

    # 保存过滤元信息
    meta = {
        "cond_threshold": args.cond_threshold,
        "total_samples": total_samples,
        "kept_samples": total_kept,
        "dropped_samples": total_dropped,
        "drop_rate_percent": drop_rate,
        "elapsed_sec": total_t,
        "throughput_msps": throughput,
        "batch_size": args.batch,
        "robot": args.robot,
        "ee_link": args.ee_link,
        "ee_link_note": "末端最后一个 fixed link, 点云采集的位置",
        "joint_child_links": joint_child_links,
        "save_singular_positions": args.save_singular_positions,
    }
    meta_path = input_dir / f"filter_singularity_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  [SAVE] {meta_path}")


if __name__ == "__main__":
    main()
