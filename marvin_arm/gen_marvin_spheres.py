"""
Marvin M6 双臂机器人 - 碰撞球生成脚本
使用 cuRobo 官方 fit_spheres_to_mesh (VOXEL_VOLUME_SAMPLE_SURFACE 模式)

输出:
    1) marvin_spheres.yml      - cuRobo 加载用的 collision_spheres 配置
    2) marvin_spheres_vis.glb  - 可视化文件（mesh + 拟合球叠加）

使用方法:
    python gen_marvin_spheres.py
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import trimesh
import yaml

from curobo.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(
    SCRIPT_DIR,
    "Marvin M6-S-CCS-696-V4.0_Robot Stand_Asm",
    "meshes",
)
OUT_YML = os.path.join(SCRIPT_DIR, "marvin_spheres.yml")
OUT_GLB = os.path.join(SCRIPT_DIR, "marvin_spheres_vis.glb")

# 左/右臂目标配置文件（脚本会把球数据 + 自碰撞配置合并写回这两个文件）
LEFT_ARM_YML = os.path.join(SCRIPT_DIR, "marvin_left_arm.yml")
RIGHT_ARM_YML = os.path.join(SCRIPT_DIR, "marvin_right_arm.yml")

# 左/右臂参与碰撞检测的 link（按运动学链顺序，方便生成相邻忽略表）
LEFT_ARM_CHAIN: List[str] = [
    "robot_stand", "Base_L",
    "Link1_L", "Link2_L", "Link3_L", "Link4_L", "Link5_L", "Link6_L", "Link7_L",
]
RIGHT_ARM_CHAIN: List[str] = [
    "robot_stand", "Base_R",
    "Link1_R", "Link2_R", "Link3_R", "Link4_R", "Link5_R", "Link6_R", "Link7_R",
]

# 自碰撞配置（保守默认）
SELF_COLLISION_BUFFER_VALUE = 0.005   # 每个 link 0.5 cm
COLLISION_SPHERE_BUFFER = 0.005       # 对外部物体的额外膨胀

# 每个 link 对应的 mesh 文件名 + 球数
# 球数经验值: 大块底座多放, 细长连杆中等, 末端短粗少放
LINK_MESH_CONFIG: List[Tuple[str, str, int]] = [
    # link_name,   mesh_file,        n_spheres
    ("robot_stand", "robot_stand.STL", 20),
    ("Base_L",      "Base_L.STL",      8),
    ("Base_R",      "Base_R.STL",      8),
    ("Link1_L",     "Link1_L.STL",     10),
    ("Link2_L",     "Link2_L.STL",     12),
    ("Link3_L",     "Link3_L.STL",     10),
    ("Link4_L",     "Link4_L.STL",     12),
    ("Link5_L",     "Link5_L.STL",     10),
    ("Link6_L",     "Link6_L.STL",     8),
    ("Link7_L",     "Link7_L.STL",     8),
    ("Link1_R",     "Link1_R.STL",     10),
    ("Link2_R",     "Link2_R.STL",     12),
    ("Link3_R",     "Link3_R.STL",     10),
    ("Link4_R",     "Link4_R.STL",     12),
    ("Link5_R",     "Link5_R.STL",     10),
    ("Link6_R",     "Link6_R.STL",     8),
    ("Link7_R",     "Link7_R.STL",     8),
]

# 拟合参数
# SAMPLE_SURFACE: 表面均匀采样, 球心=表面点, 半径=SURFACE_SPHERE_RADIUS (固定)
# 对包络任务最直接: 球贴着 mesh 表面铺一层, 半径稍大于连杆截面厚度
FIT_TYPE = SphereFitType.SAMPLE_SURFACE
SURFACE_SPHERE_RADIUS = 0.04   # 表面球半径(米), 4cm 约等于连杆截面半径量级
RADIUS_INFLATE = 1.0           # SAMPLE_SURFACE 已用足够大的固定半径, 不再放大
ENABLE_VIS = True              # 是否生成 glb 可视化


def fit_one_link(mesh_path: str, n_spheres: int) -> Tuple[np.ndarray, np.ndarray]:
    """对单个 link 的 mesh 拟合球.

    Returns:
        centers: (M, 3) 球心 (link 局部坐标系)
        radii:   (M,)   球半径
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    pts, rad = fit_spheres_to_mesh(
        mesh,
        n_spheres=n_spheres,
        surface_sphere_radius=SURFACE_SPHERE_RADIUS,
        fit_type=FIT_TYPE,
    )
    centers = np.asarray(pts, dtype=np.float64)
    radii = np.asarray(rad, dtype=np.float64) * RADIUS_INFLATE
    return centers, radii


def build_spheres_dict() -> Dict[str, list]:
    """对所有 link 拟合, 返回 cuRobo collision_spheres 期望的格式."""
    spheres_dict: Dict[str, list] = {}
    for link_name, mesh_file, n in LINK_MESH_CONFIG:
        mesh_path = os.path.join(MESH_DIR, mesh_file)
        if not os.path.isfile(mesh_path):
            print(f"  [skip] {link_name}: mesh not found -> {mesh_path}")
            continue
        centers, radii = fit_one_link(mesh_path, n)
        link_spheres = [
            {"center": [float(c[0]), float(c[1]), float(c[2])],
             "radius": float(r)}
            for c, r in zip(centers, radii)
        ]
        spheres_dict[link_name] = link_spheres
        print(f"  [ok]   {link_name:<14s} mesh={mesh_file:<18s} "
              f"n_req={n:<3d} n_got={len(link_spheres):<3d} "
              f"r_avg={radii.mean():.4f}")
    return spheres_dict


def write_yaml(spheres_dict: Dict[str, list], out_path: str) -> None:
    """写出 cuRobo 兼容的 yml 文件."""
    data = {
        "robot": "Marvin M6",
        "collision_spheres": spheres_dict,
    }
    # 用 default_flow_style=None 让 list of dict 紧凑些
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("##\n## Marvin M6 双臂机器人 - 碰撞球配置 (auto-generated)\n##\n")
        yaml.safe_dump(
            data, f,
            sort_keys=False,
            default_flow_style=False,
            width=float("inf"),  # 防止长字符串(如带空格的urdf路径)被折行
        )
    print(f"\n[write] yml -> {out_path}")


def visualize(spheres_dict: Dict[str, list], out_path: str) -> None:
    """把 mesh 和拟合球叠加渲染, 导出 glb 供肉眼检查.

    注意: 这里只能可视化 link 的局部坐标 (mesh 自己的坐标系),
    没有走 URDF 的 joint transform, 所以各 link 是堆在原点的.
    用法: 用 Blender / 在线 glb viewer 切换显示, 看每个 link 的球是否包络.
    """
    scene = trimesh.Scene()
    for link_name, mesh_file, _ in LINK_MESH_CONFIG:
        if link_name not in spheres_dict:
            continue
        mesh_path = os.path.join(MESH_DIR, mesh_file)
        mesh = trimesh.load(mesh_path, force="mesh")
        mesh.visual.face_colors = [200, 200, 200, 120]
        scene.add_geometry(mesh, node_name=f"mesh_{link_name}")
        for i, s in enumerate(spheres_dict[link_name]):
            sph = trimesh.creation.icosphere(radius=s["radius"], subdivisions=2)
            sph.apply_translation(s["center"])
            sph.visual.face_colors = [255, 60, 60, 120]
            scene.add_geometry(sph, node_name=f"sph_{link_name}_{i}")
    scene.export(out_path)
    print(f"[write] glb -> {out_path}")


def coverage_report(spheres_dict: Dict[str, list]) -> None:
    """对每个 link 算一下顶点被球覆盖的比例, 给出简单报告."""
    print("\n--- coverage report (vertex inside any sphere) ---")
    for link_name, mesh_file, _ in LINK_MESH_CONFIG:
        if link_name not in spheres_dict:
            continue
        mesh = trimesh.load(os.path.join(MESH_DIR, mesh_file), force="mesh")
        v = mesh.vertices
        covered = np.zeros(len(v), dtype=bool)
        for s in spheres_dict[link_name]:
            d = np.linalg.norm(v - np.array(s["center"]), axis=1)
            covered |= (d <= s["radius"])
        ratio = covered.mean() * 100
        flag = "OK" if ratio > 95.0 else "LOW"
        print(f"  [{flag}] {link_name:<14s}: {ratio:6.2f}% covered "
              f"({covered.sum()}/{len(v)})")


def _build_self_collision_ignore(chain: List[str]) -> Dict[str, List[str]]:
    """按链相邻关系生成 self_collision_ignore.

    例: chain = [A, B, C, D]  ->
        A: [B], B: [A, C], C: [B, D], D: [C]
    """
    ignore: Dict[str, List[str]] = {}
    for i, link in enumerate(chain):
        neighbors: List[str] = []
        if i > 0:
            neighbors.append(chain[i - 1])
        if i < len(chain) - 1:
            neighbors.append(chain[i + 1])
        ignore[link] = neighbors
    return ignore


def merge_into_arm_yml(
    spheres_dict: Dict[str, list],
    arm_yml_path: str,
    arm_chain: List[str],
    arm_label: str,
) -> None:
    """把球数据 + 自碰撞默认值合并写回左/右臂的 cuRobo 配置 yml.

    Args:
        spheres_dict: build_spheres_dict 的返回, 含全部 17 个 link 的球
        arm_yml_path: 目标 yml (marvin_left_arm.yml / marvin_right_arm.yml)
        arm_chain:    本臂参与碰撞的 link 列表 (按运动学链顺序)
        arm_label:    'left' / 'right', 仅用于日志和注释
    """
    if not os.path.isfile(arm_yml_path):
        print(f"  [skip merge] {arm_label}: yml not found -> {arm_yml_path}")
        return

    with open(arm_yml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    kin = cfg["robot_cfg"]["kinematics"]

    # 1) 过滤出本臂的球数据
    arm_spheres: Dict[str, list] = {}
    for link in arm_chain:
        if link in spheres_dict:
            arm_spheres[link] = spheres_dict[link]
        else:
            print(f"  [warn] {arm_label}: missing spheres for {link}")

    # 2) 写入 collision_* 字段
    kin["collision_spheres"] = arm_spheres
    kin["collision_link_names"] = list(arm_spheres.keys())
    kin["collision_sphere_buffer"] = COLLISION_SPHERE_BUFFER

    # 3) 自碰撞: 相邻 link 互忽略, 每个 link 给一个统一 buffer
    kin["self_collision_ignore"] = _build_self_collision_ignore(
        list(arm_spheres.keys())
    )
    kin["self_collision_buffer"] = {
        link: SELF_COLLISION_BUFFER_VALUE for link in arm_spheres.keys()
    }

    # 4) mesh_link_names 留空 (cuRobo 用 sphere 做碰撞, 不必声明 mesh)
    kin["mesh_link_names"] = None

    # 5) 写回
    header = (
        "##\n"
        f"## Marvin M6 双臂机器人 - {arm_label} 臂 CuRobo 配置文件\n"
        "## 包含 FK + 自碰撞配置 (auto-merged by gen_marvin_spheres.py)\n"
        "##\n"
    )
    with open(arm_yml_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(
            cfg, f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=float("inf"),  # 防止长字符串(如带空格的urdf路径)被折行
        )

    n_spheres = sum(len(v) for v in arm_spheres.values())
    print(f"[merge] {arm_label:<5s} -> {arm_yml_path}  "
          f"({len(arm_spheres)} links, {n_spheres} spheres)")


def main():
    print("=" * 60)
    print("Marvin M6 - 生成碰撞球配置 (fit_spheres_to_mesh)")
    print("=" * 60)
    print(f"fit_type            : {FIT_TYPE.value}")
    print(f"surface sphere r    : {SURFACE_SPHERE_RADIUS}")
    print(f"radius inflate      : {RADIUS_INFLATE}")
    print(f"mesh dir            : {MESH_DIR}\n")

    spheres_dict = build_spheres_dict()
    write_yaml(spheres_dict, OUT_YML)
    coverage_report(spheres_dict)
    if ENABLE_VIS:
        visualize(spheres_dict, OUT_GLB)

    # 把球数据 + 保守自碰撞默认值合并写回左右臂 cuRobo 配置
    print("\n--- merge into arm yml ---")
    merge_into_arm_yml(spheres_dict, LEFT_ARM_YML, LEFT_ARM_CHAIN, "left")
    merge_into_arm_yml(spheres_dict, RIGHT_ARM_YML, RIGHT_ARM_CHAIN, "right")

    print("\nDone.")


if __name__ == "__main__":
    main()
