#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 SolidWorks 导出的 STEP 装配体里提取「某个零件/子装配」的实例位姿。

用途: table.STEP 里嵌了两台 DOBOT Nova 2, 我们要把它们换成自己的可动 XTrainer,
      因此需要知道这两台机器人在装配体坐标系下的安装位姿 (xyz + rpy)。

原理: 解析 AP214 的装配结构
    NEXT_ASSEMBLY_USAGE_OCCURRENCE(NAUO)      -> 父/子 PRODUCT_DEFINITION
    CONTEXT_DEPENDENT_SHAPE_REPRESENTATION    -> 该 NAUO 对应的变换关系
    REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION + ITEM_DEFINED_TRANSFORMATION
                                              -> 两个 AXIS2_PLACEMENT_3D
    子->父变换 M = A_parent * A_child^-1

用法:
    python3 extract_cad_mounts.py table.STEP --match "DOBOT"
    python3 extract_cad_mounts.py table.STEP --match "DOBOT" --depth 2   # 连子零件一起列
"""
from __future__ import annotations

import argparse
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

ENTITY_RE = re.compile(r"#(\d+)\s*=\s*(.*?);", re.S)
TYPE_RE = re.compile(r"^\(?\s*([A-Z_0-9]+)\s*\(")


# ------------------------------- STEP 解析 -------------------------------


def split_args(s: str) -> List[str]:
    """按顶层逗号切分 STEP 参数串。"""
    out, depth, cur, in_str = [], 0, [], False
    for ch in s:
        if in_str:
            cur.append(ch)
            if ch == "'":
                in_str = False
            continue
        if ch == "'":
            in_str = True
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out


class StepFile:
    def __init__(self, path: str):
        with open(path, "r", encoding="latin-1") as f:
            text = f.read()
        data = text[text.index("DATA;") + 5 :]
        self.simple: Dict[int, Tuple[str, List[str]]] = {}     # id -> (type, args)
        self.complex: Dict[int, List[Tuple[str, List[str]]]] = {}  # id -> [(type, args)]
        for m in ENTITY_RE.finditer(data):
            eid = int(m.group(1))
            body = m.group(2).strip()
            if body.startswith("("):
                parts = []
                for sub in re.finditer(r"([A-Z_0-9]+)\s*\(", body):
                    name = sub.group(1)
                    start = sub.end()
                    depth, i = 1, start
                    while depth:
                        if body[i] == "(":
                            depth += 1
                        elif body[i] == ")":
                            depth -= 1
                        i += 1
                    parts.append((name, split_args(body[start : i - 1])))
                self.complex[eid] = parts
            else:
                tm = TYPE_RE.match(body)
                if not tm:
                    continue
                name = tm.group(1)
                inner = body[body.index("(", tm.start(1)) + 1 : body.rindex(")")]
                self.simple[eid] = (name, split_args(inner))

    def args(self, eid: int, etype: str) -> Optional[List[str]]:
        e = self.simple.get(eid)
        if e and e[0] == etype:
            return e[1]
        for t, a in self.complex.get(eid, []):
            if t == etype:
                return a
        return None

    def type_of(self, eid: int) -> str:
        e = self.simple.get(eid)
        if e:
            return e[0]
        parts = self.complex.get(eid)
        return parts[0][0] if parts else ""

    def by_type(self, etype: str) -> List[int]:
        out = [i for i, (t, _) in self.simple.items() if t == etype]
        out += [i for i, ps in self.complex.items() if any(t == etype for t, _ in ps)]
        return out


def ref(tok: str) -> Optional[int]:
    tok = tok.strip()
    return int(tok[1:]) if tok.startswith("#") else None


def unquote(tok: str) -> str:
    tok = tok.strip()
    return tok[1:-1] if tok.startswith("'") else tok


# ------------------------------- 几何 -------------------------------


def placement_matrix(sf: StepFile, eid: int) -> np.ndarray:
    """AXIS2_PLACEMENT_3D -> 4x4。axis = 局部 z, ref_direction = 局部 x 的参考。"""
    a = sf.args(eid, "AXIS2_PLACEMENT_3D")
    if a is None:
        return np.eye(4)
    loc = sf.args(ref(a[1]), "CARTESIAN_POINT")
    p = np.array([float(v) for v in split_args(loc[1].strip()[1:-1])])

    def direction(tok: str, default) -> np.ndarray:
        rid = ref(tok)
        if rid is None:
            return np.array(default, dtype=float)
        d = sf.args(rid, "DIRECTION")
        return np.array([float(v) for v in split_args(d[1].strip()[1:-1])])

    z = direction(a[2] if len(a) > 2 else "$", (0.0, 0.0, 1.0))
    x = direction(a[3] if len(a) > 3 else "$", (1.0, 0.0, 0.0))
    z = z / np.linalg.norm(z)
    x = x - np.dot(x, z) * z
    nx = np.linalg.norm(x)
    x = x / nx if nx > 1e-12 else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, p
    return m


def mat_to_rpy(m: np.ndarray) -> Tuple[float, float, float]:
    """URDF 约定: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    r = m[:3, :3]
    sy = -r[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) < 1.0 - 1e-9:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


# ------------------------------- 装配树 -------------------------------


class Assembly:
    def __init__(self, sf: StepFile):
        self.sf = sf
        self.pd_name: Dict[int, str] = {}
        self.pd_rep: Dict[int, int] = {}
        self.children: Dict[int, List[Tuple[int, int]]] = {}  # parent_pd -> [(child_pd, nauo)]
        self.nauo_tf: Dict[int, np.ndarray] = {}
        self._build()

    def _build(self) -> None:
        sf = self.sf
        # PRODUCT_DEFINITION -> 产品名
        for pd in sf.by_type("PRODUCT_DEFINITION"):
            a = sf.args(pd, "PRODUCT_DEFINITION")
            pdf = sf.args(ref(a[2]), "PRODUCT_DEFINITION_FORMATION")
            if pdf is None:
                pdf = sf.args(ref(a[2]), "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE")
            if pdf is None:
                continue
            prod = sf.args(ref(pdf[2]), "PRODUCT")
            if prod is not None:
                self.pd_name[pd] = unquote(prod[0])

        # PRODUCT_DEFINITION -> SHAPE_REPRESENTATION
        pds_def: Dict[int, int] = {}   # pds -> definition (pd 或 nauo)
        for pds in sf.by_type("PRODUCT_DEFINITION_SHAPE"):
            a = sf.args(pds, "PRODUCT_DEFINITION_SHAPE")
            d = ref(a[2])
            if d is not None:
                pds_def[pds] = d
        for sdr in sf.by_type("SHAPE_DEFINITION_REPRESENTATION"):
            a = sf.args(sdr, "SHAPE_DEFINITION_REPRESENTATION")
            pds, rep = ref(a[0]), ref(a[1])
            d = pds_def.get(pds)
            if d is not None and sf.type_of(d) == "PRODUCT_DEFINITION":
                self.pd_rep[d] = rep

        # NAUO -> 父子关系
        nauo_pair: Dict[int, Tuple[int, int]] = {}
        for n in sf.by_type("NEXT_ASSEMBLY_USAGE_OCCURRENCE"):
            a = sf.args(n, "NEXT_ASSEMBLY_USAGE_OCCURRENCE")
            parent, child = ref(a[3]), ref(a[4])
            nauo_pair[n] = (parent, child)
            self.children.setdefault(parent, []).append((child, n))

        # CDSR -> NAUO 的变换
        for c in sf.by_type("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"):
            a = sf.args(c, "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION")
            rr, pds = ref(a[0]), ref(a[1])
            nauo = pds_def.get(pds)
            if nauo is None or nauo not in nauo_pair:
                continue
            rel = sf.args(rr, "REPRESENTATION_RELATIONSHIP")
            trf = sf.args(rr, "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")
            if rel is None or trf is None:
                continue
            idt = sf.args(ref(trf[0]), "ITEM_DEFINED_TRANSFORMATION")
            if idt is None:
                continue
            rep1, rep2 = ref(rel[2]), ref(rel[3])
            a1 = placement_matrix(sf, ref(idt[2]))
            a2 = placement_matrix(sf, ref(idt[3]))
            child_pd = nauo_pair[nauo][1]
            child_rep = self.pd_rep.get(child_pd)
            # rep1 是子件的 rep 时: 子->父 = A2 @ A1^-1; 反之取逆
            if child_rep is not None and rep2 == child_rep and rep1 != child_rep:
                m = a1 @ np.linalg.inv(a2)
            else:
                m = a2 @ np.linalg.inv(a1)
            self.nauo_tf[nauo] = m

    def roots(self) -> List[int]:
        childs = {c for lst in self.children.values() for c, _ in lst}
        return [pd for pd in self.pd_name if pd not in childs and pd in self.children]

    def walk(self, root: int, max_depth: int):
        """深度优先遍历, 产出 (路径名列表, 累计 4x4)。"""
        stack = [(root, [self.pd_name.get(root, "?")], np.eye(4), 0)]
        while stack:
            pd, path, m, d = stack.pop()
            yield path, m
            if d >= max_depth:
                continue
            for child, nauo in self.children.get(pd, []):
                tf = self.nauo_tf.get(nauo, np.eye(4))
                stack.append(
                    (child, path + [self.pd_name.get(child, "?")], m @ tf, d + 1)
                )


# ------------------------------- 主流程 -------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", help="STEP 文件")
    ap.add_argument("--match", default="DOBOT", help="按产品名子串筛选")
    ap.add_argument("--depth", type=int, default=8, help="遍历深度")
    ap.add_argument("--scale", type=float, default=0.001, help="STEP 单位 -> 米")
    ap.add_argument("--tree", action="store_true", help="打印顶层装配树")
    # 被替换掉的 DOBOT 子装配原点 != XTrainer 的 LINK_0 原点。下面两项是二者之间的
    # 固定差量(在 DOBOT 局部系下右乘), 数值沿用 analysis_workspace 已验证的对齐结果:
    #   平移 (0, -0.002, -0.00541) m, 再绕局部 Z 转 180deg
    ap.add_argument("--extra-xyz", type=float, nargs=3, default=[0.0, -0.002, -0.00541],
                    help="附加局部平移 (m), 用于 DOBOT 原点 -> XTrainer LINK_0")
    ap.add_argument("--extra-rpy", type=float, nargs=3, default=[0.0, 0.0, math.pi],
                    help="附加局部旋转 (rad, URDF rpy)")
    ap.add_argument("--emit-yaml", action="store_true", help="按 cad_mounts.yaml 格式输出")
    args = ap.parse_args()

    def rpy_to_mat(r, p, y) -> np.ndarray:
        cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                                  math.sin(p), math.cos(y), math.sin(y))
        m = np.eye(4)
        m[:3, :3] = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        return m

    extra = rpy_to_mat(*args.extra_rpy)
    extra[:3, 3] = np.array(args.extra_xyz) / args.scale  # 仍在 STEP 单位下叠加

    sf = StepFile(args.step)
    asm = Assembly(sf)
    roots = asm.roots()
    print(f"[step] {len(asm.pd_name)} product_definitions, "
          f"{len(asm.nauo_tf)} placed occurrences, roots={[asm.pd_name[r] for r in roots]}")

    for root in roots:
        if args.tree:
            for path, m in asm.walk(root, 1):
                p = m[:3, 3] * args.scale
                print(f"  {'/'.join(path)}   xyz={np.round(p, 5).tolist()}")
        hits = [(path, m) for path, m in asm.walk(root, args.depth)
                if args.match.lower() in path[-1].lower()]
        # 只保留最浅的那一层匹配 (子装配本身, 而不是它内部的零件)
        if hits:
            dmin = min(len(p) for p, _ in hits)
            hits = [h for h in hits if len(h[0]) == dmin]
        hits = [(p, m @ extra) for p, m in hits]
        # 按 x 从小到大: 第一个记为 left_arm, 第二个记为 right_arm
        hits.sort(key=lambda h: h[1][0, 3])
        names = ["left_arm", "right_arm"]
        print(f"[match] '{args.match}' -> {len(hits)} occurrence(s)")
        if args.emit_yaml:
            print("# 由 extract_cad_mounts.py 从 table.STEP 自动提取, 单位: 米 / 弧度")
        for i, (path, m) in enumerate(hits):
            p = m[:3, 3] * args.scale
            rpy = mat_to_rpy(m)
            if args.emit_yaml:
                key = names[i] if i < len(names) else f"arm_{i}"
                print(f"{key}:")
                print(f"  xyz: [{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}]")
                print(f"  rpy: [{rpy[0]:.9f}, {rpy[1]:.9f}, {rpy[2]:.9f}]")
                continue
            print(f"  #{i}  {'/'.join(path)}")
            print(f"      xyz: [{p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f}]")
            print(f"      rpy: [{rpy[0]:.9f}, {rpy[1]:.9f}, {rpy[2]:.9f}]"
                  f"   (deg {np.round(np.degrees(rpy), 3).tolist()})")
            print(f"      R:\n{np.array2string(m[:3, :3], precision=6, suppress_small=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
