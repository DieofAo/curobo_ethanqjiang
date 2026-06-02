"""
Marvin M6 - 单臂碰撞球可视化

用 cuRobo 加载某只臂的 yml (默认左臂), 零位 FK 后:
  - 把每个 link 的 mesh 按 link_pose 变换到 base 坐标系
  - 把每个 sphere 也按 link_pose 变换 (其实直接拿 state.link_spheres_tensor 即可,
    cuRobo 已经把球算到世界系了)
  - 导出 glb 方便肉眼检查覆盖情况

用法:
    python vis_single_arm.py                  # 默认左臂
    python vis_single_arm.py --arm right      # 右臂
"""
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import trimesh
import yaml

from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel,
    CudaRobotModelConfig,
)
from curobo.types.base import TensorDeviceType


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # curobo_ethanqjiang/

# 左/右臂 yml 是相对 PROJECT_ROOT 的路径前缀引用 mesh, 这里直接用绝对路径加载 mesh
MESH_DIR = os.path.join(
    SCRIPT_DIR, "Marvin M6-S-CCS-696-V4.0_Robot Stand_Asm", "meshes"
)

LINK_TO_MESH = {
    "robot_stand": "robot_stand.STL",
    "Base_L": "Base_L.STL",  "Base_R": "Base_R.STL",
    "Link1_L": "Link1_L.STL", "Link2_L": "Link2_L.STL",
    "Link3_L": "Link3_L.STL", "Link4_L": "Link4_L.STL",
    "Link5_L": "Link5_L.STL", "Link6_L": "Link6_L.STL",
    "Link7_L": "Link7_L.STL",
    "Link1_R": "Link1_R.STL", "Link2_R": "Link2_R.STL",
    "Link3_R": "Link3_R.STL", "Link4_R": "Link4_R.STL",
    "Link5_R": "Link5_R.STL", "Link6_R": "Link6_R.STL",
    "Link7_R": "Link7_R.STL",
}


def quat_wxyz_to_matrix(qw: float, qx: float, qy: float, qz: float,
                        tx: float, ty: float, tz: float) -> np.ndarray:
    """把 (w,x,y,z) 四元数 + 平移 拼成 4x4 齐次矩阵."""
    n = qw * qw + qx * qx + qy * qy + qz * qz
    s = 0.0 if n < 1e-12 else 2.0 / n
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    R = np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy        ],
        [xy + wz,         1.0 - (xx + zz), yz - wx        ],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (tx, ty, tz)
    return T


def load_arm_cfg(yml_path: str) -> dict:
    """读取 yml, 把里面相对路径 urdf_path/asset_root_path 改写成绝对路径,
    并把 collision_link_names 加进 link_names, 让 FK 能输出每个 link 的 pose."""
    with open(yml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    kin = cfg["robot_cfg"]["kinematics"]

    # 相对 -> 绝对
    if not os.path.isabs(kin["urdf_path"]):
        kin["urdf_path"] = os.path.join(PROJECT_ROOT, kin["urdf_path"])
    if not os.path.isabs(kin["asset_root_path"]):
        kin["asset_root_path"] = os.path.join(
            PROJECT_ROOT, kin["asset_root_path"]
        )

    # 让 link_pose 输出包含碰撞 link
    coll_links = kin.get("collision_link_names") or []
    link_names = kin.get("link_names") or []
    merged = list(dict.fromkeys([kin.get("ee_link")] + coll_links + link_names))
    kin["link_names"] = [n for n in merged if n]
    return cfg


def build_robot_model(cfg: dict) -> CudaRobotModel:
    tdev = TensorDeviceType()
    rcfg = CudaRobotModelConfig.from_data_dict(
        cfg["robot_cfg"]["kinematics"], tensor_args=tdev
    )
    return CudaRobotModel(rcfg)


def get_link_transforms(
    robot: CudaRobotModel, link_names: List[str]
) -> Dict[str, np.ndarray]:
    """零位 FK -> 每个 link 的 4x4 (base 坐标系)."""
    dof = len(robot.kinematics_config.joint_names) if hasattr(
        robot.kinematics_config, "joint_names"
    ) else len(robot.joint_names)
    q = torch.zeros(
        (1, dof),
        device=robot.tensor_args.device,
        dtype=robot.tensor_args.dtype,
    )
    state = robot.get_state(q)
    pos = state.links_position[0].cpu().numpy()        # (n_link, 3)
    quat = state.links_quaternion[0].cpu().numpy()     # (n_link, 4) wxyz
    name_to_idx = {n: i for i, n in enumerate(robot.link_names)}

    transforms: Dict[str, np.ndarray] = {}
    for ln in link_names:
        if ln not in name_to_idx:
            print(f"  [warn] {ln} 不在 robot.link_names, 跳过")
            continue
        i = name_to_idx[ln]
        qw, qx, qy, qz = quat[i]
        tx, ty, tz = pos[i]
        transforms[ln] = quat_wxyz_to_matrix(qw, qx, qy, qz, tx, ty, tz)
    return transforms


def build_scene(
    cfg: dict,
    transforms: Dict[str, np.ndarray],
) -> trimesh.Scene:
    """每个 link: mesh 灰半透 + 球红半透, 都已变换到 base 坐标系."""
    kin = cfg["robot_cfg"]["kinematics"]
    spheres_dict: Dict[str, list] = kin.get("collision_spheres") or {}
    coll_links: List[str] = kin.get("collision_link_names") or []

    scene = trimesh.Scene()
    n_links = 0
    n_spheres = 0
    for ln in coll_links:
        if ln not in transforms:
            continue
        T = transforms[ln]

        # mesh
        mesh_file = LINK_TO_MESH.get(ln)
        if mesh_file is not None:
            mesh_path = os.path.join(MESH_DIR, mesh_file)
            if os.path.isfile(mesh_path):
                m = trimesh.load(mesh_path, force="mesh")
                m.apply_transform(T)
                m.visual.face_colors = [200, 200, 200, 110]
                scene.add_geometry(m, node_name=f"mesh_{ln}")

        # spheres
        for i, s in enumerate(spheres_dict.get(ln, [])):
            c_local = np.array(s["center"], dtype=np.float64)
            c_world = (T[:3, :3] @ c_local) + T[:3, 3]
            sph = trimesh.creation.icosphere(radius=s["radius"], subdivisions=2)
            sph.apply_translation(c_world)
            sph.visual.face_colors = [255, 60, 60, 130]
            scene.add_geometry(sph, node_name=f"sph_{ln}_{i}")
            n_spheres += 1
        n_links += 1
    print(f"  [scene] {n_links} links, {n_spheres} spheres")
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--yml", default=None,
                        help="覆盖默认 yml 路径")
    parser.add_argument("--out", default=None,
                        help="输出 glb 路径")
    args = parser.parse_args()

    yml_path = args.yml or os.path.join(
        SCRIPT_DIR,
        f"marvin_{args.arm}_arm.yml",
    )
    out_path = args.out or os.path.join(
        SCRIPT_DIR,
        f"marvin_{args.arm}_arm_vis.glb",
    )

    print("=" * 60)
    print(f"Marvin M6 - 单臂碰撞球可视化 ({args.arm})")
    print("=" * 60)
    print(f"yml : {yml_path}")
    print(f"out : {out_path}\n")

    cfg = load_arm_cfg(yml_path)
    robot = build_robot_model(cfg)

    coll_links = cfg["robot_cfg"]["kinematics"]["collision_link_names"]
    print(f"collision_link_names = {coll_links}")

    transforms = get_link_transforms(robot, coll_links)
    scene = build_scene(cfg, transforms)
    scene.export(out_path)
    print(f"\n[write] glb -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
