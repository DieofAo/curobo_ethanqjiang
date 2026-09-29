#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试IK求解并保存结果到npz文件。
运行: conda run -n curobo python3 test_ik_solve.py
"""
import time
import numpy as np
import torch
import sys

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

NPZ_PATH = "/home/ethanqjiang/workspace/curobo/mcap/left_arm_pose_commands.npz"
OUTPUT_PATH = "/home/ethanqjiang/workspace/curobo/mcap/ik_solutions.npz"
ROBOT_CONFIG_FILE = "marvin_left_arm.yml"
NUM_SEEDS = 32
BATCH_SIZE = 64


def main():
    print("加载数据...", flush=True)
    data = np.load(NPZ_PATH)
    timestamps_ns = data['timestamps_ns']
    positions = data['positions']
    quaternions_xyzw = data['quaternions_xyzw']
    N = len(timestamps_ns)
    print(f"  总帧数: {N}", flush=True)

    # xyzw -> wxyz
    quaternions_wxyz = np.zeros_like(quaternions_xyzw)
    quaternions_wxyz[:, 0] = quaternions_xyzw[:, 3]
    quaternions_wxyz[:, 1] = quaternions_xyzw[:, 0]
    quaternions_wxyz[:, 2] = quaternions_xyzw[:, 1]
    quaternions_wxyz[:, 3] = quaternions_xyzw[:, 2]

    print("初始化IK求解器...", flush=True)
    tensor_args = TensorDeviceType()
    cfg_dict = load_yaml(join_path(get_robot_configs_path(), ROBOT_CONFIG_FILE))
    robot_cfg = RobotConfig.from_dict(cfg_dict["robot_cfg"])
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg, None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=NUM_SEEDS,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    )
    ik_solver = IKSolver(ik_config)
    print(f"  DOF={ik_solver.dof}, joints={ik_solver.joint_names}", flush=True)

    # 转为tensor
    pos_tensor = torch.from_numpy(positions.astype(np.float32)).to(tensor_args.device)
    quat_tensor = torch.from_numpy(quaternions_wxyz.astype(np.float32)).to(tensor_args.device)

    # 顺序批量求解
    print(f"开始IK求解 (batch_size={BATCH_SIZE})...", flush=True)
    all_solutions = np.zeros((N, 7), dtype=np.float32)
    all_success = np.zeros(N, dtype=bool)
    prev_solution = None

    t_start = time.time()
    for i in range(0, N, BATCH_SIZE):
        j = min(i + BATCH_SIZE, N)
        goal = Pose(pos_tensor[i:j], quat_tensor[i:j])
        actual_batch = j - i

        seed_config = None
        if prev_solution is not None:
            # seed_config 形状: (batch, n_seeds, dof)
            # 用前一帧的解作为1个seed，其余由solver随机生成
            seed = prev_solution.unsqueeze(0).unsqueeze(0).expand(actual_batch, 1, -1)
            seed_config = seed  # [batch, 1, dof]

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

        n_s = success.sum().item()
        print(f"  [{i}:{j}] 成功: {n_s}/{actual_batch}", flush=True)

    torch.cuda.synchronize()
    t_elapsed = time.time() - t_start

    n_success = int(all_success.sum())
    print(f"\nIK求解完成:", flush=True)
    print(f"  成功: {n_success}/{N} ({100.0*n_success/N:.1f}%)", flush=True)
    print(f"  耗时: {t_elapsed:.2f}s ({N/t_elapsed:.0f} poses/s)", flush=True)

    # 连续性检查
    diffs = np.abs(np.diff(all_solutions, axis=0))
    max_diffs = diffs.max(axis=1)
    print(f"\n帧间最大关节角变化:", flush=True)
    print(f"  mean={max_diffs.mean():.4f} rad", flush=True)
    print(f"  max={max_diffs.max():.4f} rad", flush=True)
    print(f"  >0.5rad的帧数: {(max_diffs > 0.5).sum()}", flush=True)

    # 平滑处理失败帧
    success_indices = np.where(all_success)[0]
    if len(success_indices) > 0 and n_success < N:
        print(f"\n对 {N - n_success} 个失败帧进行插值...", flush=True)
        for i in range(N):
            if all_success[i]:
                continue
            prev_idx = success_indices[success_indices < i]
            next_idx = success_indices[success_indices > i]
            if len(prev_idx) == 0 and len(next_idx) == 0:
                continue
            elif len(prev_idx) == 0:
                all_solutions[i] = all_solutions[next_idx[0]]
            elif len(next_idx) == 0:
                all_solutions[i] = all_solutions[prev_idx[-1]]
            else:
                p, n = prev_idx[-1], next_idx[0]
                alpha = (i - p) / (n - p)
                all_solutions[i] = (1 - alpha) * all_solutions[p] + alpha * all_solutions[n]

    # 保存
    np.savez_compressed(OUTPUT_PATH,
        solutions=all_solutions,
        success=all_success,
        timestamps_ns=timestamps_ns,
    )
    print(f"\n结果已保存到: {OUTPUT_PATH}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
