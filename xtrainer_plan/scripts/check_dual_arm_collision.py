#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双臂碰撞检测: 校验镜像同步轨迹或 12-DOF 联合轨迹是否会让两条臂相撞。

原理:
  对旧的 6-DOF 镜像轨迹, 第二条臂(second_) 使用相同关节角并通过两臂
  基座的固定变换得到碰撞球；对新的 12-DOF combined robot 轨迹, 一次 FK
  会直接得到两条独立运动链的碰撞球。两种模式都逐点计算跨臂球对净距离。

  碰撞球模型与规划完全一致(同一个 curobo robot yml / collision_link_names),
  因此这是对规划结果的复核, 不是近似估计。

注意:
  * 只查「臂 vs 臂」。臂 vs 料台/墙体在规划阶段已由 curobo 世界模型保证,
    自碰撞由 self_collision_check 复核, 均不在此重复。
  * 单臂配置 (xtrainer.yml): 第二臂的姿态是镜像假设, 关节角与主臂逐点相同,
    球按基座固定变换搬过去。
  * 双臂 combined 配置 (mimic 或 independent): 一次 FK 直接得到两臂全部
    48 个球 (都在主臂 LINK_0 系下), 只查跨臂球对, 不再做任何镜像假设。

需要在 conda curobo 环境运行 (python 3.11 + curobo):
  conda activate curobo
  python3 scripts/check_dual_arm_collision.py results/<时间戳>
  python3 scripts/check_dual_arm_collision.py results/<时间戳> --margin-mm 5
  python3 scripts/check_dual_arm_collision.py results/<时间戳> --arm right
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xtrainer_common import load_trajectory  # noqa: E402
from build_scene_urdf import pose_to_mat  # noqa: E402


# 跨臂球对对比只沿时间维分块，不改变单个路点的精确计算。
# 2048 在常见 24--48 个碰撞球/臂时可把中间张量限制在可控范围。
PAIR_COLLISION_CHUNK_SIZE = 2048


# ============================== 碰撞球 FK ==============================


def fk_spheres(robot_cfg: Dict[str, Any], positions: np.ndarray,
               ) -> Tuple[np.ndarray, List[str], List[str]]:
    """对整条轨迹做 FK, 返回 (spheres[N,S,4], sph_link_names[S], collision_link_names)。

    spheres[..., :3] 为球心(base_link 系), spheres[..., 3] 为半径。
    sph_link_names[S] 是每个球所属的 link 名。
    """
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig

    from plan_trajectory import load_robot_cfg_dict

    robot_dict = load_robot_cfg_dict(robot_cfg)
    kin = CudaRobotModel(RobotConfig.from_dict(robot_dict["robot_cfg"]).kinematics)
    tensor_args = TensorDeviceType()
    q = tensor_args.to_device(positions.astype(np.float32))
    state = kin.get_state(q)
    sph = state.link_spheres_tensor.detach().cpu().numpy().astype(np.float64)
    idx_map = kin.kinematics_config.link_sphere_idx_map.cpu().numpy()
    coll_links = list(kin.generator_config.collision_link_names)
    # idx_map 取值是运动树全局 link 索引 (link_name_to_idx_map),
    # 与 collision_link_names 的序号只在单臂下同序; 双臂 mimic 模型必须
    # 用逆映射取 link 名。
    name_to_idx = dict(kin.kinematics_config.link_name_to_idx_map)
    idx_to_name = {int(v): k for k, v in name_to_idx.items()}
    sph_link_names = [idx_to_name[int(i)] for i in idx_map]
    return sph, sph_link_names, coll_links


# ============================== 碰撞判定 ==============================


def check_pair_collisions(
    sph_a: np.ndarray,
    sph_b: np.ndarray,
    names_a: List[str],
    names_b: List[str],
    margin_m: float,
) -> Dict[str, Any]:
    """逐点对碰撞判定。sph_a: [N, Sa, 4], sph_b: [N, Sb, 4];
    names_a/names_b 为每个球对应的 link 名 (用于报告)。

    沿 N 维分块，避免一次分配整条轨迹的 [N, Sa, Sb, 3]
    差值张量。各块按时间顺序合并，因此并列最小值仍选全局最早的
    (point, sphere_a, sphere_b)，Counter 的同计数排序也与未分块实现一致。
    """
    sph_a = np.asarray(sph_a)
    sph_b = np.asarray(sph_b)
    if (
        sph_a.ndim != 3
        or sph_b.ndim != 3
        or sph_a.shape[0] != sph_b.shape[0]
        or sph_a.shape[2] != 4
        or sph_b.shape[2] != 4
        or len(names_a) != sph_a.shape[1]
        or len(names_b) != sph_b.shape[1]
    ):
        raise ValueError(
            "碰撞球 shape/name 不匹配: "
            f"a={sph_a.shape}/{len(names_a)}, b={sph_b.shape}/{len(names_b)}"
        )
    if not np.isfinite(float(margin_m)) or float(margin_m) < 0.0:
        raise ValueError("碰撞余量必须是有限非负数")
    if not np.isfinite(sph_a).all() or not np.isfinite(sph_b).all():
        raise ValueError("碰撞球轨迹含 NaN/Inf，拒绝按无碰撞处理")
    if (sph_a[..., 3] < 0.0).any() or (sph_b[..., 3] < 0.0).any():
        raise ValueError("碰撞球半径不能为负数")

    n_pts, n_sph = sph_a.shape[:2]
    ca, ra = sph_a[..., :3], sph_a[..., 3]
    cb, rb = sph_b[..., :3], sph_b[..., 3]

    if n_pts == 0 or n_sph == 0 or sph_b.shape[1] == 0:
        raise ValueError("碰撞球轨迹的时间维和球维必须非空")

    chunk_size = max(1, int(PAIR_COLLISION_CHUNK_SIZE))
    clear_min_per_pt = np.empty(n_pts, dtype=np.float64)
    hit_idx: List[int] = []
    pair_counter: Counter = Counter()
    pair_first_order: Dict[Tuple[str, str], int] = {}
    min_clear = float("inf")
    worst_idx = worst_a = worst_b = 0
    n_sph_b = sph_b.shape[1]

    for start in range(0, n_pts, chunk_size):
        stop = min(start + chunk_size, n_pts)

        # 块内仍与原实现相同：[chunk, A, B] 球心距与净距离
        # (>0 即安全间隙, <0 即穿透)。
        d = np.linalg.norm(
            ca[start:stop, :, None, :] - cb[start:stop, None, :, :],
            axis=-1,
        )
        clear = d - (
            ra[start:stop, :, None] + rb[start:stop, None, :]
        )
        hit = clear < margin_m

        per_point_hit = hit.any(axis=(1, 2))
        local_hit_idx = np.nonzero(per_point_hit)[0]
        hit_idx.extend((local_hit_idx + start).tolist())
        clear_min_per_pt[start:stop] = clear.min(axis=(1, 2))

        # np.argmin 选块内 row-major 的首个最小值。只在严格更小时
        # 更新，使跨块并列最小值也保留时间上最早的那一个。
        local_flat = int(np.argmin(clear))
        local_idx, local_a, local_b = np.unravel_index(local_flat, clear.shape)
        local_min = float(clear[local_idx, local_a, local_b])
        if local_min < min_clear:
            min_clear = local_min
            worst_idx = start + int(local_idx)
            worst_a = int(local_a)
            worst_b = int(local_b)

        # 先沿时间维聚合到球对，避免 np.nonzero(hit) 在大量
        # 碰撞时又产生最多 3 * chunk * A * B 个 int64 索引。
        # 另存每个 link pair 在全局 row-major 中的首次出现位置，
        # 以保持 Counter.most_common 原有的同计数稳定排序。
        if local_hit_idx.size:
            sphere_pair_counts = np.count_nonzero(hit, axis=0)
            first_hit_time = np.argmax(hit, axis=0)
            for a, b in np.argwhere(sphere_pair_counts > 0):
                a, b = int(a), int(b)
                pair = (names_a[a], names_b[b])
                pair_counter[pair] += int(sphere_pair_counts[a, b])
                order = (
                    ((start + int(first_hit_time[a, b])) * n_sph + a)
                    * n_sph_b + b
                )
                if pair not in pair_first_order or order < pair_first_order[pair]:
                    pair_first_order[pair] = order

        # 下一块计算差值张量前释放本块的 [chunk, A, B]
        # 中间结果，避免赋值右侧求值时与上一块短暂重叠。
        del d, clear, hit

    worst = {
        "index": worst_idx,
        "link_pair": [names_a[worst_a], names_b[worst_b]],
        "clearance_mm": min_clear * 1000.0,
    }
    return {
        "n_points": int(n_pts),
        "n_spheres_per_arm": int(n_sph),
        "n_collision_points": len(hit_idx),
        "collision_indices": hit_idx,
        "min_clearance_mm": min_clear * 1000.0,
        "worst_point": worst,
        "top_link_pairs": [
            {"pair": list(pair), "n_sphere_hits": count}
            for pair, count in sorted(
                pair_counter.items(),
                key=lambda item: (-item[1], pair_first_order[item[0]]),
            )[:10]
        ],
        "clearance_min_per_point_mm": (clear_min_per_pt * 1000.0).tolist(),
    }


# ============================== 主流程 ==============================


def main() -> int:
    ap = argparse.ArgumentParser(
        description="双臂碰撞检测: 支持 6-DOF 镜像同步与 12-DOF 联合轨迹",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("traj", type=str, help="轨迹目录或 trajectory.npz 路径")
    ap.add_argument("--arm", default="left", choices=["left", "right"],
                    help="被规划(主)臂, 与规划/播放时的 active_arm 一致")
    ap.add_argument("--mounts", type=str,
                    default=str(TASK_ROOT / "config" / "cad_mounts.yaml"))
    ap.add_argument("--margin-mm", type=float, default=0.0,
                    help="安全余量: 间隙小于该值即视为碰撞")
    ap.add_argument("--config", type=str, default=None,
                    help="pick_place 配置(取 robot 段); 默认用轨迹 meta 里的 config")
    ap.add_argument("--out-json", type=str, default="",
                    help="报告输出路径, 默认 <轨迹目录>/dual_arm_collision.json")
    args = ap.parse_args()

    data, meta = load_trajectory(args.traj)
    positions = np.asarray(data["positions"], dtype=np.float64)
    times = np.asarray(data["times"], dtype=np.float64)
    n = positions.shape[0]
    print(f"[DUAL] traj={args.traj}")
    print(f"[DUAL] {n} 点, {positions.shape[1]} 关节, "
          f"时长 {float(times[-1]):.2f}s")

    robot_cfg = (meta.get("config") or {}).get("robot")
    if robot_cfg is None:
        if args.config is None:
            print("[FAIL] 轨迹 meta 中没有 config.robot, 请用 --config 指定")
            return 2
        with open(args.config, "r") as f:
            robot_cfg = yaml.safe_load(f)["robot"]
    with open(args.mounts, "r", encoding="utf-8") as f:
        mounts = yaml.safe_load(f)

    other = "right" if args.arm == "left" else "left"
    m_active = pose_to_mat(mounts[f"{args.arm}_arm"]["xyz"],
                           mounts[f"{args.arm}_arm"]["rpy"])
    m_other = pose_to_mat(mounts[f"{other}_arm"]["xyz"],
                          mounts[f"{other}_arm"]["rpy"])
    # 第二臂基座在主臂 LINK_0 下的位姿 (与 build_scene_urdf.py 的场景一致)
    t_other = np.linalg.inv(m_active) @ m_other
    r_mat, t_vec = t_other[:3, :3], t_other[:3, 3]
    print(f"[DUAL] 主臂={args.arm}  镜像臂={other}  "
          f"基座平移 {np.round(t_vec, 4).tolist()}")

    sph, sph_link_names, coll_links = fk_spheres(robot_cfg, positions)
    n_coll_links = len(set(sph_link_names))
    print(f"[DUAL] FK 完成: {sph.shape[1]} 个碰撞球, 覆盖 {n_coll_links} 个 link")

    # 方案 A 检测: collision_link_names 里出现带前缀的第二臂 link,
    # 说明 FK 已直接给出两臂球 (都在主臂 LINK_0 系), 直接查跨臂球对即可。
    dual_prefix = robot_cfg.get("dual_arm_prefix") or ""
    if not dual_prefix:
        base_set = {"LINK_0", "LINK_1", "LINK_2", "LINK_3",
                    "LINK_4", "LINK_5", "LINK_6"}
        extra_links = sorted(set(sph_link_names) - base_set)
        if extra_links:
            # 从 extra link 名推出公共前缀 (如 second_LINK_0 -> second_)
            dual_prefix = extra_links[0].split("LINK_")[0]
    direct_mode = bool(dual_prefix) and any(
        n.startswith(dual_prefix) for n in coll_links)

    margin_m = float(args.margin_mm) / 1000.0
    if direct_mode:
        mask_a = np.array([not n.startswith(dual_prefix) for n in sph_link_names])
        mask_b = ~mask_a
        if not mask_a.any() or not mask_b.any():
            print(f"[FAIL] 前缀 {dual_prefix!r} 无法把 {len(sph_link_names)} "
                  f"个球拆成两臂")
            return 2
        rep = check_pair_collisions(
            sph[:, mask_a], sph[:, mask_b],
            [n for n, m in zip(sph_link_names, mask_a) if m],
            [n for n, m in zip(sph_link_names, mask_b) if m],
            margin_m)
        print(f"[DUAL] 直接复核模式 (combined dual robot): "
              f"主臂 {int(mask_a.sum())} 球 vs 第二臂 {int(mask_b.sum())} 球")
        rep.update({
            "checked": True,
            "traj": str(args.traj),
            "active_arm": args.arm,
            "mirror_arm": other,
            "margin_mm": float(args.margin_mm),
            "mode": "direct (combined 双臂, FK 直接输出两臂球, 无镜像假设)",
            "dual_arm_prefix": dual_prefix,
            "collision_sphere_model": "curobo robot yml (与规划一致)",
        })
    else:
        # 镜像臂: 同一批关节角 -> 各自基座系下球心相同, 平移旋转到主臂 LINK_0 系
        sph2 = sph.copy()
        sph2[..., :3] = sph[..., :3] @ r_mat.T + t_vec
        rep = check_pair_collisions(sph, sph2, sph_link_names, sph_link_names,
                                    margin_m)
        rep.update({
            "checked": True,
            "traj": str(args.traj),
            "active_arm": args.arm,
            "mirror_arm": other,
            "margin_mm": float(args.margin_mm),
            "mode": "mirror (第二臂逐点与主臂关节角相同, --both-arms 镜像)",
            "collision_sphere_model": "curobo robot yml (与规划一致)",
        })
    worst = rep["worst_point"]
    worst["time_s"] = float(times[worst["index"]])

    n_bad = rep["n_collision_points"]
    if n_bad == 0:
        print(f"[DUAL] 结果: 无碰撞, 全程最小间隙 "
              f"{rep['min_clearance_mm']:.1f}mm "
              f"(margin={args.margin_mm:g}mm)")
    else:
        print(f"[DUAL] 结果: {n_bad}/{n} 个路点碰撞 "
              f"(margin={args.margin_mm:g}mm)!")
        print(f"[DUAL]   最小净距离 {rep['min_clearance_mm']:.1f}mm "
              f"@ 点 {worst['index']} (t={worst['time_s']:.2f}s) "
              f"{worst['link_pair'][0]} vs {worst['link_pair'][1]}")
        for e in rep["top_link_pairs"][:5]:
            print(f"[DUAL]   {e['pair'][0]:>8} vs {e['pair'][1]:<8} "
                  f"x{e['n_sphere_hits']} 球对")
    out_json = args.out_json or str(
        (Path(args.traj) if Path(args.traj).is_dir()
         else Path(args.traj).parent) / "dual_arm_collision.json")
    with open(out_json, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[DUAL] 报告已保存: {out_json}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
