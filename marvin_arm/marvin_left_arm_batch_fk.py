"""
Marvin M6 - 左臂并行 FK 脚本 (基于 curobo 标准 yml 配置)

特点:
- 走 curobo 标准 robot yml: src/curobo/content/configs/robot/marvin_left_arm.yml
  自动加载 URDF + 碰撞球
- GPU 一次 forward 处理整批 (N, dof) 关节角, 不用 for 循环
- 输出: EE 位姿、各 link 位姿、所有碰撞球世界坐标 (x,y,z,r)
- 自带耗时统计, 可选 --save 保存为 npz

使用:
    python marvin_left_arm_batch_fk.py                # 默认 batch=1024
    python marvin_left_arm_batch_fk.py --batch 10000
    python marvin_left_arm_batch_fk.py --save out.npz
"""
import argparse
import time

import numpy as np
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.util.logger import setup_curobo_logger

# ---------------- 配置 ----------------
ROBOT_CFG_FILE = "marvin_left_arm.yml"  # 相对 curobo robot configs 根, 自动定位


# ---------------- 加载模型 ----------------
def load_model(device: str = "cuda:0") -> CudaRobotModel:
    """从 curobo 标准位置的 yml 加载左臂运动学模型 (含碰撞球)。"""
    tensor_args = TensorDeviceType(device=torch.device(device))
    cfg = CudaRobotModelConfig.from_robot_yaml_file(
        file_path=ROBOT_CFG_FILE,
        tensor_args=tensor_args,
    )
    return CudaRobotModel(cfg)


# ---------------- 采样关节角 ----------------
def sample_joints(model: CudaRobotModel, n: int, seed: int) -> torch.Tensor:
    """在关节限位内均匀采样 N 组关节角, 返回 GPU tensor (N, dof)。"""
    rng = np.random.default_rng(seed)
    lim = model.get_joint_limits()  # JointLimits.position shape: (n_joints, 2)
    pos = lim.position.detach().cpu().numpy()
    low, high = pos[:, 0], pos[:, 1]
    q_np = rng.uniform(low, high, size=(n, low.shape[0])).astype(np.float32)
    return torch.as_tensor(q_np, device=model.tensor_args.device)


# ---------------- 并行 FK ----------------
def batch_fk(model: CudaRobotModel, q: torch.Tensor) -> dict:
    """一次 GPU forward 算完整批 FK, 返回 numpy 结果。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    state = model.get_state(q)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    out = {
        "q": q.cpu().numpy(),
        "ee_position": state.ee_position.detach().cpu().numpy(),
        "ee_quaternion": state.ee_quaternion.detach().cpu().numpy(),
        "elapsed_sec": elapsed,
    }
    if state.link_spheres_tensor is not None:
        # (N, S, 4) -> [x, y, z, radius] in world frame
        out["link_spheres"] = state.link_spheres_tensor.detach().cpu().numpy()
    if state.link_poses is not None:
        link_pos = {}
        link_quat = {}
        for name, pose in state.link_poses.items():
            link_pos[name] = pose.position.detach().cpu().numpy()
            link_quat[name] = pose.quaternion.detach().cpu().numpy()
        out["link_position"] = link_pos
        out["link_quaternion"] = link_quat
    return out


# ---------------- 打印摘要 ----------------
def summarize(result: dict, model: CudaRobotModel) -> None:
    n = result["q"].shape[0]
    dof = result["q"].shape[1]
    sec = result["elapsed_sec"]
    print(f"[FK] batch={n}, dof={dof}, GPU forward 用时 {sec*1000:.2f} ms "
          f"({n/sec:.0f} 次/秒)")
    print(f"     EE pos shape  : {result['ee_position'].shape}")
    print(f"     EE quat shape : {result['ee_quaternion'].shape}")
    if "link_spheres" in result:
        s = result["link_spheres"]
        print(f"     spheres shape : {s.shape}  (N, S, 4)  共 {s.shape[1]} 个球")
    if "link_position" in result:
        print(f"     link 数量     : {len(result['link_position'])}")

    # 抽样前 3 组打印 EE
    print("\n[抽样] 前 3 组 EE 位姿:")
    for i in range(min(3, n)):
        p = result["ee_position"][i]
        q = result["ee_quaternion"][i]
        print(f"  [{i}] pos=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})  "
              f"quat(wxyz)=({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})")


# ---------------- 保存 ----------------
def save_npz(path: str, result: dict) -> None:
    flat = {
        "q": result["q"],
        "ee_position": result["ee_position"],
        "ee_quaternion": result["ee_quaternion"],
    }
    if "link_spheres" in result:
        flat["link_spheres"] = result["link_spheres"]
    if "link_position" in result:
        for k, v in result["link_position"].items():
            flat[f"linkpos__{k}"] = v
        for k, v in result["link_quaternion"].items():
            flat[f"linkquat__{k}"] = v
    np.savez_compressed(path, **flat)
    print(f"[save] 已写入 {path}")


# ---------------- 入口 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024, help="批量大小 N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save", type=str, default=None, help="保存到 .npz")
    args = parser.parse_args()

    setup_curobo_logger("warn")

    print("=" * 60)
    print(f"Marvin M6 左臂 - 并行 FK (yml = {ROBOT_CFG_FILE})")
    print("=" * 60)

    print("[1/3] 加载模型 ...")
    model = load_model(args.device)
    print(f"      base_link = {model.kinematics_config.base_link}, "
          f"ee_link = {model.kinematics_config.ee_link}, dof = {model.get_dof()}")

    print(f"[2/3] 采样 {args.batch} 组关节角 ...")
    q = sample_joints(model, args.batch, args.seed)

    print("[3/3] GPU 并行 FK ...")
    # 跑 2 次:第 1 次预热, 第 2 次计时
    _ = batch_fk(model, q)
    result = batch_fk(model, q)

    summarize(result, model)
    if args.save:
        save_npz(args.save, result)

    print("\n[done] FK 完成")


if __name__ == "__main__":
    main()
