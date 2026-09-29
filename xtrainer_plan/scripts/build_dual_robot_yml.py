#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由单臂配置脚本化生成 mimic 或 independent 双臂的两份 curobo 配置:

  1. spheres/xtrainer_dual.yml
       把 spheres/xtrainer.yml 的碰撞球整体复制一份, link 名加前缀
       (两臂几何相同, 球在各自 link 系下的坐标完全一致)。
  2. xtrainer_dual.yml
       以 xtrainer.yml 为底: collision_link_names/self_collision_ignore/
       self_collision_buffer/mesh_link_names 复制一份带前缀的版本;
       mimic 模式下 cspace 原样照抄 (规划仍是 6 DOF);
       independent 模式下把带前缀的第二臂关节及逐关节参数追加到 cspace
       (规划为 12 DOF);
       urdf_path 指向 build_scene_urdf.py --emit-planning-urdf 对应模式生成的 URDF。

注意: 臂间碰撞对 *不* 加入 self_collision_ignore —— 两臂之间的球对必须保留在
自碰撞检测里, 这正是双臂模型提供规划期硬保证的核心。

用法 (系统 python3 即可, 无需 curobo):
    python3 scripts/build_dual_robot_yml.py            # 默认前缀 second_
    python3 scripts/build_dual_robot_yml.py --prefix right_
    python3 scripts/build_dual_robot_yml.py \
        --planning-mode independent --robot-yml xtrainer_dual_independent.yml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_CFG_DIR = REPO_ROOT / "src/curobo/content/configs/robot"

BASE_SPHERES_YML = ROBOT_CFG_DIR / "spheres/xtrainer.yml"
BASE_ROBOT_YML = ROBOT_CFG_DIR / "xtrainer.yml"

MIMIC_URDF_REL = "robot/ur_description/xtrainer_dual_mimic.urdf"


class NoAliasDumper(yaml.SafeDumper):
    """同一个 list 对象被两臂复用时, 不允许输出锚点/别名, 全部展开。"""

    def ignore_aliases(self, data):  # noqa: D102
        return True


def make_header(planning_mode: str, prefix: str, urdf_path: str) -> str:
    mode_desc = ("方案 A: mimic 耦合双臂, 6 DOF" if planning_mode == "mimic"
                 else "independent 双臂联合模型, 12 DOF")
    return (
        "# !!! 本文件由 xtrainer_plan/scripts/build_dual_robot_yml.py 自动生成, 请勿手改 !!!\n"
        f"# {mode_desc} ({Path(urdf_path).name}), 第二臂前缀 = \"{prefix}\"\n"
    )


def prefixed_sphere_yml(base: Dict[str, Any], prefix: str,
                        robot_name: str) -> Dict[str, Any]:
    src = base["collision_spheres"]
    out: Dict[str, Any] = {}
    for link, spheres in src.items():
        out[link] = spheres
        out[prefix + link] = spheres  # 同一 list 对象, dump 时各自展开
    return {"robot": robot_name, "collision_spheres": out}


def prefixed_map(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """{k: v} -> {k: v, prefix+k: 递归加前缀的 v}。

    self_collision_ignore 的 value 是 link 名列表, 需要加前缀;
    self_collision_buffer 的 value 是数值, 原样复制。
    """
    out = dict(d)
    for k, v in d.items():
        if isinstance(v, list):
            out[prefix + k] = [prefix + x for x in v]
        else:
            out[prefix + k] = v
    return out


def extend_cspace_independent(cspace: Dict[str, Any], prefix: str) -> None:
    """把单臂 cspace 的逐关节向量复制给第二臂，并追加带前缀的关节名。"""
    joint_names = list(cspace["joint_names"])
    n_dof = len(joint_names)
    for key, value in list(cspace.items()):
        if key != "joint_names" and isinstance(value, list) and len(value) == n_dof:
            cspace[key] = list(value) + list(value)
    cspace["joint_names"] = joint_names + [prefix + name for name in joint_names]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="second_",
                    help="第二臂名字前缀, 须与 build_scene_urdf.py --planning-prefix 一致")
    ap.add_argument("--planning-mode", choices=["mimic", "independent"],
                    default="mimic", help="关节模式 (默认 mimic，兼容旧调用)")
    ap.add_argument("--robot-yml", default=None,
                    help="输出的 robot yml 文件名；默认随 planning mode 选择")
    ap.add_argument("--urdf-path", default=None,
                    help="配置中的 URDF 相对路径；默认 mimic 使用历史路径，"
                         "independent 使用 robot-yml 的 stem")
    args = ap.parse_args()

    robot_yml = args.robot_yml or (
        "xtrainer_dual.yml" if args.planning_mode == "mimic"
        else "xtrainer_dual_independent.yml"
    )
    robot_stem = Path(robot_yml).stem
    urdf_path = args.urdf_path or (
        MIMIC_URDF_REL if args.planning_mode == "mimic"
        else f"robot/ur_description/{robot_stem}.urdf"
    )
    robot_name = Path(urdf_path).stem
    header = make_header(args.planning_mode, args.prefix, urdf_path)

    spheres = yaml.safe_load(BASE_SPHERES_YML.read_text())
    n_sphere = sum(len(v) for v in spheres["collision_spheres"].values())
    dual_spheres = prefixed_sphere_yml(spheres, args.prefix, robot_name)
    out_spheres = ROBOT_CFG_DIR / "spheres" / f"{robot_stem}.yml"
    with open(out_spheres, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(dual_spheres, f, Dumper=NoAliasDumper,
                  sort_keys=False, allow_unicode=True)
    print(f"[YML] {out_spheres}  ({n_sphere} 球 -> {2 * n_sphere} 球)")

    robot = yaml.safe_load(BASE_ROBOT_YML.read_text())
    kin = robot["robot_cfg"]["kinematics"]
    kin["urdf_path"] = urdf_path
    # 双臂模型没有对应 usd, 全部置空 (use_usd_kinematics=False 时本就不会被读)
    for k in ("isaac_usd_path", "usd_path", "usd_robot_root"):
        kin[k] = None
    kin["collision_link_names"] = list(kin["collision_link_names"]) + [
        args.prefix + n for n in kin["collision_link_names"]
    ]
    kin["collision_spheres"] = f"spheres/{robot_stem}.yml"
    kin["self_collision_ignore"] = prefixed_map(kin["self_collision_ignore"], args.prefix)
    kin["self_collision_buffer"] = prefixed_map(kin["self_collision_buffer"], args.prefix)
    kin["mesh_link_names"] = list(kin["mesh_link_names"]) + [
        args.prefix + n for n in kin["mesh_link_names"]
    ]
    if args.planning_mode == "independent":
        extend_cspace_independent(kin["cspace"], args.prefix)
    # mimic 模式 cspace 原样照抄: mimic 关节不占 DOF, joint_names 仍是 J_1..J_6

    out_robot = ROBOT_CFG_DIR / robot_yml
    with open(out_robot, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(robot, f, sort_keys=False, allow_unicode=True)
    n_link = len(kin["collision_link_names"])
    print(f"[YML] {out_robot}  (collision_link_names {n_link // 2} -> {n_link}, "
          f"cspace {len(kin['cspace']['joint_names'])} DOF: "
          f"{kin['cspace']['joint_names']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
