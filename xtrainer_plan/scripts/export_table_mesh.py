#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 table.STEP 装配体里的两台 DOBOT Nova 2 剔除后导出为 STL, 供 rviz 作静态场景显示。

保留的是料台/立柱/安装板等所有非机器人零件; 被剔除的机器人本体由 URDF 里可动的
XTrainer 模型替代。导出的 STL 仍是 STEP 的毫米单位, URDF 里用 scale="0.001" 缩放。

需要 gmsh (pip install gmsh), 在 conda 环境里跑:
    conda activate curobo
    python3 export_table_mesh.py <in.STEP> <out.stl>
"""
from __future__ import annotations

import argparse
import os

import gmsh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="输入 STEP 文件")
    ap.add_argument("output", help="输出二进制 STL")
    ap.add_argument("--exclude", default="DOBOT Nova 2", help="按实体名子串剔除")
    ap.add_argument("--mesh-min-mm", type=float, default=8.0)
    ap.add_argument("--mesh-max-mm", type=float, default=25.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if args.verbose else 0)
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.mesh_min_mm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", args.mesh_max_mm)

        gmsh.model.add("xtrainer_workbench")
        gmsh.model.occ.importShapes(os.path.abspath(args.input), highestDimOnly=True)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        excluded = [e for e in volumes if args.exclude in gmsh.model.getEntityName(*e)]
        if not excluded:
            raise RuntimeError(
                f"没有任何 STEP 实体名包含 {args.exclude!r}; 拒绝导出未过滤的装配体"
            )

        gmsh.model.occ.remove(excluded, recursive=True)
        gmsh.model.occ.synchronize()
        retained = gmsh.model.getEntities(3)
        print(f"STEP 过滤: 剔除 {len(excluded)} 个实体, 保留 {len(retained)} 个")

        gmsh.model.mesh.generate(2)
        out = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        gmsh.write(out)
        print(f"已写出 {out}")
    finally:
        gmsh.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
