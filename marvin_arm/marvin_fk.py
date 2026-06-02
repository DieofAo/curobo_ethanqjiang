"""
Marvin M6 双臂机器人 - 前向运动学(FK)计算示例
使用 cuRobo 计算左臂和右臂的末端执行器位姿

使用方法:
    python marvin_fk.py
"""
import os
import torch
import numpy as np

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import load_yaml

# 当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# URDF 绝对路径
URDF_PATH = os.path.join(
    SCRIPT_DIR,
    "Marvin M6-S-CCS-696-V4.0_Robot Stand_Asm",
    "urdf",
    "Marvin M6-S-CCS-696-V4.0_Robot Stand_Asm.urdf",
)


def load_left_arm() -> CudaRobotModel:
    """加载左臂运动学模型（使用from_basic，最简方式）"""
    tensor_args = TensorDeviceType()
    robot_cfg = RobotConfig.from_basic(
        urdf_path=URDF_PATH,
        base_link="robot_stand",
        ee_link="Link7_L",
        tensor_args=tensor_args,
    )
    return CudaRobotModel(robot_cfg.kinematics)


def load_right_arm() -> CudaRobotModel:
    """加载右臂运动学模型（使用from_basic，最简方式）"""
    tensor_args = TensorDeviceType()
    robot_cfg = RobotConfig.from_basic(
        urdf_path=URDF_PATH,
        base_link="robot_stand",
        ee_link="Link7_R",
        tensor_args=tensor_args,
    )
    return CudaRobotModel(robot_cfg.kinematics)


def compute_fk(kin_model: CudaRobotModel, joint_angles: list) -> dict:
    """
    计算单组关节角度的FK
    
    Args:
        kin_model: cuRobo运动学模型
        joint_angles: 关节角度列表(弧度)，长度等于模型自由度
        
    Returns:
        包含位置和姿态的字典
    """
    tensor_args = TensorDeviceType()
    q = torch.tensor([joint_angles], dtype=torch.float32, device=tensor_args.device)
    state = kin_model.get_state(q)
    
    # 提取末端执行器位姿
    ee_pos = state.ee_position.cpu().numpy()[0]  # [x, y, z]
    ee_quat = state.ee_quaternion.cpu().numpy()[0]  # [w, x, y, z]
    
    return {
        "position": ee_pos,
        "quaternion": ee_quat,
    }


def compute_fk_batch(kin_model: CudaRobotModel, joint_angles_batch: np.ndarray) -> dict:
    """
    批量计算FK（利用GPU并行加速）
    
    Args:
        kin_model: cuRobo运动学模型
        joint_angles_batch: (N, dof) 的关节角度数组(弧度)
        
    Returns:
        包含批量位置和姿态的字典
    """
    tensor_args = TensorDeviceType()
    q = torch.tensor(joint_angles_batch, dtype=torch.float32, device=tensor_args.device)
    state = kin_model.get_state(q)
    
    return {
        "positions": state.ee_position.cpu().numpy(),  # (N, 3)
        "quaternions": state.ee_quaternion.cpu().numpy(),  # (N, 4)
    }


def main():
    setup_curobo_logger("info")
    
    print("=" * 60)
    print("Marvin M6 双臂机器人 - 前向运动学(FK)计算")
    print("=" * 60)
    print(f"URDF路径: {URDF_PATH}")
    
    # ========== 左臂 FK ==========
    print("\n[1] 加载左臂运动学模型...")
    left_model = load_left_arm()
    dof_left = left_model.get_dof()
    print(f"    左臂自由度: {dof_left}")
    
    # 零位姿态
    zero_joints = [0.0] * dof_left
    result = compute_fk(left_model, zero_joints)
    print(f"\n[2] 左臂零位姿态 FK 结果:")
    print(f"    关节角度(rad): {zero_joints}")
    print(f"    末端位置(m):   x={result['position'][0]:.4f}, "
          f"y={result['position'][1]:.4f}, z={result['position'][2]:.4f}")
    print(f"    末端姿态(quat): w={result['quaternion'][0]:.4f}, "
          f"x={result['quaternion'][1]:.4f}, y={result['quaternion'][2]:.4f}, "
          f"z={result['quaternion'][3]:.4f}")
    
    # 测试一个非零姿态
    test_joints = [0.5, -0.3, 0.2, -0.8, 0.1, 0.4, -0.2][:dof_left]
    result = compute_fk(left_model, test_joints)
    print(f"\n[3] 左臂测试姿态 FK 结果:")
    print(f"    关节角度(rad): {test_joints}")
    print(f"    末端位置(m):   x={result['position'][0]:.4f}, "
          f"y={result['position'][1]:.4f}, z={result['position'][2]:.4f}")
    print(f"    末端姿态(quat): w={result['quaternion'][0]:.4f}, "
          f"x={result['quaternion'][1]:.4f}, y={result['quaternion'][2]:.4f}, "
          f"z={result['quaternion'][3]:.4f}")
    
    # ========== 右臂 FK ==========
    print("\n[4] 加载右臂运动学模型...")
    right_model = load_right_arm()
    dof_right = right_model.get_dof()
    print(f"    右臂自由度: {dof_right}")
    
    result = compute_fk(right_model, [0.0] * dof_right)
    print(f"\n[5] 右臂零位姿态 FK 结果:")
    print(f"    关节角度(rad): {[0.0] * dof_right}")
    print(f"    末端位置(m):   x={result['position'][0]:.4f}, "
          f"y={result['position'][1]:.4f}, z={result['position'][2]:.4f}")
    print(f"    末端姿态(quat): w={result['quaternion'][0]:.4f}, "
          f"x={result['quaternion'][1]:.4f}, y={result['quaternion'][2]:.4f}, "
          f"z={result['quaternion'][3]:.4f}")
    
    # ========== 批量 FK 演示 ==========
    print("\n[6] 批量FK计算演示（左臂，10组随机关节角度）...")
    np.random.seed(42)
    # 左臂关节限位
    joint_limits_lower = np.array([-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708])[:dof_left]
    joint_limits_upper = np.array([3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 1.5708])[:dof_left]
    
    random_joints = np.random.uniform(
        joint_limits_lower, joint_limits_upper, size=(10, dof_left)
    ).astype(np.float32)
    
    batch_result = compute_fk_batch(left_model, random_joints)
    
    print(f"    输入形状: {random_joints.shape}")
    print(f"    输出位置形状: {batch_result['positions'].shape}")
    print(f"    输出姿态形状: {batch_result['quaternions'].shape}")
    print("\n    各组末端位置:")
    for i in range(10):
        pos = batch_result['positions'][i]
        print(f"      [{i}] x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    
    print("\n" + "=" * 60)
    print("FK 计算完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
