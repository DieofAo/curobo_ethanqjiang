#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拼装 rviz 用的「料台场景 + 双 XTrainer」URDF, 输出到 stdout。

设计要点 (与 xtrainer_plan 现有可视化保持兼容):
  * URDF 的根仍然是 LINK_0 —— 即被规划的那条臂的基座。rviz 的 Fixed Frame
    依旧用 LINK_0, play_trajectory_ros.py 发布的 /joint_states、工作空间线框、
    start/goal tf 全部不用改。
  * 料台 (table.stl, 已剔除 STEP 里的两台 DOBOT) 通过「挂载位姿的逆变换」
    挂到 LINK_0 下:  LINK_0 --inv(M_active)--> scene_root --identity--> workbench
  * 另一条臂作为静态展示: scene_root --M_static--> <prefix>LINK_0 ...,
    其所有关节改成 fixed, 因此不需要往 /joint_states 里发它的关节角。

臂的连杆定义直接从规划用的 xtrainer.urdf 复制, 避免与规划模型不一致。

用法 (系统 python3 即可, 无需 curobo):
    python3 build_scene_urdf.py \
        --urdf   .../ur_description/xtrainer.urdf \
        --mounts .../config/cad_mounts.yaml \
        --mesh-dir  file://.../ur_description/meshes \
        --table-mesh file://.../meshes/xtrainer/table.stl \
        --active-arm left

规划模型模式 (--emit-planning-urdf):
    python3 build_scene_urdf.py \
        --urdf   .../ur_description/xtrainer.urdf \
        --mounts .../config/cad_mounts.yaml \
        --active-arm left --emit-planning-urdf --planning-mode mimic \
        --output .../ur_description/xtrainer_dual_mimic.urdf

    python3 build_scene_urdf.py \
        --urdf   .../ur_description/xtrainer.urdf \
        --mounts .../config/cad_mounts_same_side.yaml \
        --active-arm left --emit-planning-urdf --planning-mode independent \
        --output .../ur_description/xtrainer_dual_independent.urdf

  与 rviz 场景模式的差别:
    * 主臂子树与 xtrainer.urdf 完全一致 (visual/collision 原样保留, 行为零差异);
    * 不挂料台 (table 由 curobo 世界模型另行表达);
    * 第二臂只保留 inertial/collision (curobo 用碰撞球, 不读 visual mesh);
    * mimic 模式下 6 个 revolute 关节全部 <mimic> 主臂同名关节
      (multiplier=1, offset=0), curobo 解析后规划维度仍是 6;
    * independent 模式下第二臂活动关节名增加前缀且不含 <mimic>, 两臂合计
      12 DOF, 可由同一个规划问题联合优化;
    * 挂载变换与 rviz 场景完全一致: LINK_0 --inv(M_active)--> scene_root
      --M_other--> second_LINK_0 (同 check_dual_arm_collision.py 的 t_other)。
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

# xtrainer.urdf 里的 mesh 写的是 package://robotics/... , rviz 解析不了
URDF_MESH_PREFIX = "package://robotics/drivers/dobot/description/meshes"


# ------------------------------ 位姿工具 ------------------------------


def rpy_to_mat(rpy) -> np.ndarray:
    """URDF 约定: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    m = np.eye(4)
    m[:3, :3] = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    return m


def pose_to_mat(xyz, rpy) -> np.ndarray:
    m = rpy_to_mat(rpy)
    m[:3, 3] = [float(v) for v in xyz]
    return m


def mat_to_pose(m: np.ndarray):
    r = m[:3, :3]
    sp = max(-1.0, min(1.0, -r[2, 0]))
    pitch = math.asin(sp)
    if abs(sp) < 1.0 - 1e-9:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        yaw = 0.0
    return m[:3, 3].tolist(), [roll, pitch, yaw]


def fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def add_origin(joint: ET.Element, m: np.ndarray) -> None:
    xyz, rpy = mat_to_pose(m)
    ET.SubElement(joint, "origin", {"xyz": fmt(xyz), "rpy": fmt(rpy)})


# ------------------------------ URDF 组装 ------------------------------


def clone_arm_static(robot: ET.Element, src: ET.Element, prefix: str,
                     material: str | None, movable: bool = False) -> str:
    """把 src 里的整条臂以 prefix 复制一份, 只留 visual。

    material=None: 保留原臂自带材质, 显示效果与主臂一致。
    material=名字 : 去掉自带材质, 统一换成指定材质(如半透明灰)。

    movable=False: 所有关节改 fixed, 不需要往 /joint_states 发它的关节角。
    movable=True : 保留 revolute 关节(连 axis/limit 一起复制), 关节名带 prefix,
                   由 /joint_states 里同名的 <prefix>J_* 驱动, 与原臂互不冲突。

    返回复制出来的根 link 名。
    """
    links = {l.get("name"): l for l in src.findall("link")}
    joints = src.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        raise RuntimeError(f"xtrainer.urdf 的根 link 不唯一: {roots}")

    for name, link in links.items():
        new = ET.SubElement(robot, "link", {"name": prefix + name})
        for vis in link.findall("visual"):
            v = copy.deepcopy(vis)
            if material is not None:
                # 复制臂不参与任何计算, 去掉自带材质, 统一换成指定材质
                for old in v.findall("material"):
                    v.remove(old)
                ET.SubElement(v, "material", {"name": material})
            new.append(v)
        # 不复制 inertial / collision: rviz 用不到, 也避免 urdfdom 抱怨重名材质

    for j in joints:
        jtype = j.get("type", "fixed")
        keep = movable and jtype in ("revolute", "prismatic", "continuous")
        nj = ET.SubElement(robot, "joint",
                           {"name": prefix + j.get("name"),
                            "type": jtype if keep else "fixed"})
        o = j.find("origin")
        ET.SubElement(nj, "origin", {
            "xyz": o.get("xyz", "0 0 0") if o is not None else "0 0 0",
            "rpy": o.get("rpy", "0 0 0") if o is not None else "0 0 0",
        })
        ET.SubElement(nj, "parent", {"link": prefix + j.find("parent").get("link")})
        ET.SubElement(nj, "child", {"link": prefix + j.find("child").get("link")})
        if keep:
            # revolute 关节必须带 axis 与 limit, 否则 urdfdom 解析报错
            ax = j.find("axis")
            ET.SubElement(nj, "axis",
                          {"xyz": ax.get("xyz") if ax is not None else "1 0 0"})
            lim = j.find("limit")
            if lim is not None:
                nj.append(copy.deepcopy(lim))
    return prefix + roots[0]


def clone_arm_planning(robot: ET.Element, src: ET.Element, prefix: str,
                       planning_mode: str) -> str:
    """把 src 里的整条臂以 prefix 复制一份, 用于 curobo 双臂规划。

    两种模式均:
      * link 只复制 inertial / collision, 不复制 visual
        (curobo 用碰撞球做碰撞检测, 不读 visual mesh);
      * revolute/prismatic/continuous 关节保持 type/axis/limit 不变;
      * fixed 关节 (gripper_joint / TCP_joint) 原样复制, 保证链路完整。

    mimic 模式额外插入 <mimic joint="主臂关节名" .../>, 规划维度保持 6;
    independent 模式不插入 mimic, 带 prefix 的 6 个关节是独立变量, 总计 12 DOF。

    返回复制出来的根 link 名。
    """
    links = {l.get("name"): l for l in src.findall("link")}
    joints = src.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        raise RuntimeError(f"xtrainer.urdf 的根 link 不唯一: {roots}")

    for name, link in links.items():
        new = ET.SubElement(robot, "link", {"name": prefix + name})
        for tag in ("inertial", "collision"):
            for el in link.findall(tag):
                new.append(copy.deepcopy(el))

    for j in joints:
        jtype = j.get("type", "fixed")
        actuated = jtype in ("revolute", "prismatic", "continuous")
        nj = ET.SubElement(robot, "joint",
                           {"name": prefix + j.get("name"), "type": jtype})
        o = j.find("origin")
        ET.SubElement(nj, "origin", {
            "xyz": o.get("xyz", "0 0 0") if o is not None else "0 0 0",
            "rpy": o.get("rpy", "0 0 0") if o is not None else "0 0 0",
        })
        ET.SubElement(nj, "parent", {"link": prefix + j.find("parent").get("link")})
        ET.SubElement(nj, "child", {"link": prefix + j.find("child").get("link")})
        if actuated:
            ax = j.find("axis")
            ET.SubElement(nj, "axis",
                          {"xyz": ax.get("xyz") if ax is not None else "1 0 0"})
            lim = j.find("limit")
            if lim is not None:
                nj.append(copy.deepcopy(lim))
            if planning_mode == "mimic":
                # multiplier=1/offset=0: 两臂在各自基座系下位形全同
                ET.SubElement(nj, "mimic", {"joint": j.get("name"),
                                            "multiplier": "1", "offset": "0"})
    return prefix + roots[0]


def write_urdf(robot: ET.Element, output: str) -> None:
    """写入文件；output='-' 保持历史行为，输出到 stdout。"""
    xml = '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n"
    if output == "-":
        sys.stdout.write(xml)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", required=True, help="规划用的 xtrainer.urdf")
    ap.add_argument("--mounts", required=True, help="config/cad_mounts.yaml")
    ap.add_argument("--mesh-dir", default="",
                    help="替换 package://robotics/... 的 mesh 根目录 (file://...); "
                         "--emit-planning-urdf 模式下忽略 (规划模型不做替换)")
    ap.add_argument("--table-mesh", default="", help="料台 STL 的 file:// 路径")
    ap.add_argument("--active-arm", default="left", choices=["left", "right"],
                    help="哪条臂被 /joint_states 驱动 (它的基座就是 LINK_0)")
    ap.add_argument("--static-prefix", default="static_", help="静态展示臂的名字前缀")
    ap.add_argument("--active-both", action="store_true",
                    help="第二条臂也保留活动关节 (用 --second-prefix 的名字), "
                         "由 /joint_states 里 <second-prefix>J_* 驱动")
    ap.add_argument("--second-prefix", default="second_",
                    help="--active-both 时第二条臂的名字前缀")
    ap.add_argument("--no-table", action="store_true", help="不加载料台")
    ap.add_argument("--no-static-arm", action="store_true", help="不加载静态展示臂")
    ap.add_argument("--table-rgba", default="0.48 0.52 0.57 0.82")
    ap.add_argument("--static-rgba", default=None,
                    help="第二臂统一替换成的材质 rgba (如 '0.55 0.58 0.62 0.55')。"
                         "留空 = 保留原臂自带材质, 显示效果与主臂一致")
    ap.add_argument("--table-scale", type=float, default=0.001,
                    help="STL 为 STEP 毫米单位, 需缩放到米")
    ap.add_argument("--emit-planning-urdf", action="store_true",
                    help="输出 curobo 双臂规划 URDF: 不挂料台、第二臂去 visual 留 collision")
    ap.add_argument("--planning-mode", choices=["mimic", "independent"],
                    default="mimic",
                    help="--emit-planning-urdf 的关节模式 (默认 mimic，兼容旧调用)")
    ap.add_argument("--planning-prefix", default="second_",
                    help="--emit-planning-urdf 时第二臂的名字前缀")
    ap.add_argument("--output", default="-",
                    help="输出文件；默认 '-' 表示 stdout，兼容原来的 shell 重定向")
    args = ap.parse_args()

    if not args.emit_planning_urdf and not args.mesh_dir:
        ap.error("--mesh-dir 在 rviz 场景模式下必填 (规划模式忽略)")

    with open(args.urdf, "r", encoding="utf-8") as f:
        text = f.read()
    if not args.emit_planning_urdf:
        text = text.replace(URDF_MESH_PREFIX, args.mesh_dir.rstrip("/"))
    robot = ET.fromstring(text)

    with open(args.mounts, "r", encoding="utf-8") as f:
        mounts = yaml.safe_load(f)

    static_arm = "right" if args.active_arm == "left" else "left"
    m_active = pose_to_mat(mounts[f"{args.active_arm}_arm"]["xyz"],
                           mounts[f"{args.active_arm}_arm"]["rpy"])
    m_static = pose_to_mat(mounts[f"{static_arm}_arm"]["xyz"],
                           mounts[f"{static_arm}_arm"]["rpy"])

    active_root = None
    children = {j.find("child").get("link") for j in robot.findall("joint")}
    for l in robot.findall("link"):
        if l.get("name") not in children:
            active_root = l.get("name")
            break

    src = copy.deepcopy(robot)  # 复制静态臂前先留一份干净的原始树

    if args.emit_planning_urdf:
        # 双臂规划模型。主臂子树保持原样, 只追加:
        #   LINK_0 --inv(m_active)--> scene_root --m_static--> <prefix>LINK_0
        robot.set("name", f"xtrainer_dual_{args.planning_mode}")
        ET.SubElement(robot, "link", {"name": "scene_root"})
        jp = ET.SubElement(robot, "joint",
                           {"name": "scene_root_fixed", "type": "fixed"})
        add_origin(jp, np.linalg.inv(m_active))
        ET.SubElement(jp, "parent", {"link": active_root})
        ET.SubElement(jp, "child", {"link": "scene_root"})

        root_name = clone_arm_planning(robot, src, args.planning_prefix,
                                       args.planning_mode)
        jm = ET.SubElement(robot, "joint",
                           {"name": f"{args.planning_prefix}mount", "type": "fixed"})
        add_origin(jm, m_static)
        ET.SubElement(jm, "parent", {"link": "scene_root"})
        ET.SubElement(jm, "child", {"link": root_name})

        write_urdf(robot, args.output)
        return 0

    robot.set("name", "xtrainer_workcell")
    mat_table = ET.SubElement(robot, "material", {"name": "table_gray"})
    ET.SubElement(mat_table, "color", {"rgba": args.table_rgba})
    # --static-rgba 为空时第二臂保留原材质, 不再注入统一半透明灰
    if args.static_rgba is not None:
        mat_static = ET.SubElement(robot, "material", {"name": "xtrainer_static"})
        ET.SubElement(mat_static, "color", {"rgba": args.static_rgba})

    # LINK_0 -> scene_root: 用挂载位姿的逆变换, 把 CAD 装配体根系挂到活动臂基座下
    ET.SubElement(robot, "link", {"name": "scene_root"})
    j = ET.SubElement(robot, "joint", {"name": "scene_root_fixed", "type": "fixed"})
    add_origin(j, np.linalg.inv(m_active))
    ET.SubElement(j, "parent", {"link": active_root})
    ET.SubElement(j, "child", {"link": "scene_root"})

    if not args.no_table and args.table_mesh:
        link = ET.SubElement(robot, "link", {"name": "workbench"})
        vis = ET.SubElement(link, "visual")
        geo = ET.SubElement(vis, "geometry")
        s = args.table_scale
        ET.SubElement(geo, "mesh", {"filename": args.table_mesh,
                                    "scale": f"{s} {s} {s}"})
        ET.SubElement(vis, "material", {"name": "table_gray"})
        jt = ET.SubElement(robot, "joint", {"name": "workbench_fixed", "type": "fixed"})
        ET.SubElement(jt, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(jt, "parent", {"link": "scene_root"})
        ET.SubElement(jt, "child", {"link": "workbench"})

    if not args.no_static_arm:
        prefix = args.second_prefix if args.active_both else args.static_prefix
        static_mat = "xtrainer_static" if args.static_rgba is not None else None
        root_name = clone_arm_static(robot, src, prefix, static_mat,
                                     movable=args.active_both)
        js = ET.SubElement(robot, "joint",
                           {"name": f"{prefix}mount", "type": "fixed"})
        add_origin(js, m_static)
        ET.SubElement(js, "parent", {"link": "scene_root"})
        ET.SubElement(js, "child", {"link": root_name})

    write_urdf(robot, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
