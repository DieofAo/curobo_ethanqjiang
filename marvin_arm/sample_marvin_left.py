#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Marvin M6 左臂 7-DOF 工作空间采样器（curobo + GPU 并行 FK + 在线奇异性过滤）

策略与 workspace_rviz/sample_workspace.py 完全一致:
  1) 按关节限位以固定步长生成一维网格; 笛卡尔积逐"J1 外层切片"流式喂给 FK。
  2) FK 出 (ee_pos, ee_quat); 位置量化成体素索引;
     四元数取局部 z 轴方向用 HEALPix 量化成姿态桶。
  3) 在线计算几何雅可比 + det(J·Jᵀ) 操纵度, 过滤掉接近奇异的配置 (可选 SVD 条件数)。
  4) 用 torch.scatter_add_ 在 GPU 上累加 reach_count[V] 与 so3_hit[V*NPIX] (bool)。
  5) 跑完后 reach_count.bool() 即"可达体素"; so3_hit.view(V,NPIX).sum(1)/NPIX 即"姿态覆盖率"。

python sample_marvin_left.py --step 0.05 0.05 0.05 0.05 0.15 0.15 0.15 --voxel 0.01 --nside 12 --batch 1097152 --collision self --ckpt-every 30
输出: workspace_data.npz / workspace_meta.json
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import healpy as hp
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


# ============ 几何雅可比 + 条件数 (全 GPU 批量计算) ============

def quat_wxyz_to_z_axis(q: torch.Tensor) -> torch.Tensor:
    """从 (w,x,y,z) 四元数提取局部 z 轴在世界坐标系下的方向向量 (旋转矩阵第三列)。
    q: [..., 4]  返回: [..., 3]"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    zx = 2.0 * (x * z + w * y)
    zy = 2.0 * (y * z - w * x)
    zz = 1.0 - 2.0 * (x * x + y * y)
    return torch.stack([zx, zy, zz], dim=-1)


def compute_geometric_jacobian_batch(
    link_positions: torch.Tensor,
    link_quaternions: torch.Tensor,
    ee_position: torch.Tensor,
    joint_link_indices: list,
) -> torch.Tensor:
    """
    几何法计算 6×DOF 雅可比矩阵 (全部在 GPU 上批量计算)。

    Args:
        link_positions: [B, n_links, 3] 各 link 在世界坐标系下的位置
        link_quaternions: [B, n_links, 4] 各 link 四元数 (w,x,y,z)
        ee_position: [B, 3] 末端执行器位置
        joint_link_indices: 长度为 dof 的列表, 每个元素是该关节对应 link 的索引

    Returns:
        J: [B, 6, dof] 几何雅可比矩阵
    """
    B = ee_position.shape[0]
    dof = len(joint_link_indices)
    device = ee_position.device

    J = torch.zeros(B, 6, dof, device=device, dtype=ee_position.dtype)

    for i, li in enumerate(joint_link_indices):
        z_i = quat_wxyz_to_z_axis(link_quaternions[:, li, :])  # [B, 3]
        p_i = link_positions[:, li, :]  # [B, 3]
        r = ee_position - p_i  # [B, 3]
        # 线速度: z_i × r
        J[:, 0, i] = z_i[:, 1] * r[:, 2] - z_i[:, 2] * r[:, 1]
        J[:, 1, i] = z_i[:, 2] * r[:, 0] - z_i[:, 0] * r[:, 2]
        J[:, 2, i] = z_i[:, 0] * r[:, 1] - z_i[:, 1] * r[:, 0]
        # 角速度: z_i
        J[:, 3, i] = z_i[:, 0]
        J[:, 4, i] = z_i[:, 1]
        J[:, 5, i] = z_i[:, 2]

    return J


def compute_condition_number(J: torch.Tensor) -> torch.Tensor:
    """计算雅可比矩阵的条件数 (σ_max / σ_min)。J: [B, 6, dof], 返回 [B]
    注意: 此函数使用 SVD 分解, 对大 batch 较慢。推荐使用 compute_manipulability 替代。"""
    S = torch.linalg.svdvals(J)  # [B, min(6, dof)]
    s_max = S[:, 0]
    s_min = S[:, -1]
    cond = torch.where(
        s_max < 1e-10,
        torch.tensor(float("inf"), device=J.device, dtype=J.dtype),
        s_max / s_min.clamp_min(1e-10),
    )
    return cond


def compute_manipulability(J: torch.Tensor) -> torch.Tensor:
    """计算 Yoshikawa 操纵度 w = sqrt(det(J·Jᵀ))。
    J: [B, 6, dof], 返回 [B]。
    w 越大表示离奇异越远; w → 0 表示接近奇异。
    使用 det 而非 SVD, 在 GPU 上快 ~100x。"""
    JJT = J @ J.transpose(-1, -2)  # [B, 6, 6]
    det_val = torch.linalg.det(JJT)  # [B]
    # det 可能因数值误差为负, clamp 到 0
    return det_val.clamp_min(0.0).sqrt()


# ============ HEALPix + 工具函数 ============

def quat_wxyz_to_axis_z(q: torch.Tensor) -> torch.Tensor:
    """curobo 输出 (w,x,y,z) 四元数; 返回 ee 局部 +z 在世界系下的方向向量。"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx = 2.0 * (x * z + w * y)
    vy = 2.0 * (y * z - w * x)
    vz = 1.0 - 2.0 * (x * x + y * y)
    return torch.stack([vx, vy, vz], dim=-1)


def healpix_ang2pix_ring_torch(vec: torch.Tensor, nside: int) -> torch.Tensor:
    """vec: (N,3) 单位向量, 返回 RING 编号 (N,) int64。"""
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    z = torch.clamp(z, -1.0, 1.0)
    za = z.abs()
    phi = torch.atan2(y, x)
    phi = torch.where(phi < 0, phi + 2.0 * math.pi, phi)
    tt = phi * (2.0 / math.pi)
    npix = 12 * nside * nside
    pix = torch.empty_like(z, dtype=torch.long)

    eq = za <= (2.0 / 3.0)
    if eq.any():
        z_e, tt_e = z[eq], tt[eq]
        temp1 = nside * (0.5 + tt_e)
        temp2 = nside * z_e * 0.75
        jp = (temp1 - temp2).floor().long()
        jm = (temp1 + temp2).floor().long()
        ir = nside + 1 + jp - jm
        kshift = 1 - (ir & 1)
        ip = ((jp + jm - nside + kshift + 1) // 2).long() % (4 * nside)
        pix[eq] = 2 * nside * (nside - 1) + (ir - 1) * 4 * nside + ip

    pol = ~eq
    if pol.any():
        z_p, za_p, tt_p = z[pol], za[pol], tt[pol]
        tp = tt_p - tt_p.floor()
        tmp = nside * torch.sqrt(3.0 * (1.0 - za_p))
        jp = (tp * tmp).floor().long()
        jm = ((1.0 - tp) * tmp).floor().long()
        ir = jp + jm + 1
        ip = ((tt_p * ir.float()).floor().long()) % (4 * ir)
        pix_n = 2 * ir * (ir - 1) + ip
        pix_s = npix - 2 * ir * (ir + 1) + ip
        pix[pol] = torch.where(z_p > 0, pix_n, pix_s)
    return pix


def _verify_healpix(nside: int, n: int = 5000):
    g = torch.Generator(device="cpu").manual_seed(7)
    v = torch.randn(n, 3, generator=g)
    v = v / v.norm(dim=-1, keepdim=True)
    p_torch = healpix_ang2pix_ring_torch(v.cuda(), nside).cpu().numpy()
    z = v[:, 2].numpy().clip(-1, 1)
    theta = np.arccos(z)
    phi = np.arctan2(v[:, 1].numpy(), v[:, 0].numpy())
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    p_hp = hp.ang2pix(nside, theta, phi, nest=False)
    return int((p_torch != p_hp).sum()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, nargs="+", default=[0.1],
                    help="关节步长(rad). 1 个值=广播到所有关节; N 个值=每关节独立步长.")
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--nside", type=int, default=2)
    ap.add_argument("--bbox", type=float, nargs=6,
                    default=[-1.0, -0.5, -0.2, 1.0, 1.5, 2.0])
    ap.add_argument("--batch", type=int, default=1 << 22)
    ap.add_argument("--ee-link", type=str, default="gripper_link")
    ap.add_argument("--robot", type=str, default="marvin_left_arm.yml")
    ap.add_argument("--out-dir", type=str,
                    default="/data/workspace/curobo_ethanqjiang/marvin_arm/workspace_out")
    ap.add_argument("--collision", type=str, default="none",
                    choices=["none", "self"],
                    help="none: 仅做 FK; self: 在 FK 后过滤掉自碰撞配置")
    ap.add_argument("--ckpt-every", type=int, default=30,
                    help="每 N 个 J1 切片落一次中间 ckpt; 0 关闭中间存盘")
    ap.add_argument("--manip-threshold", type=float, default=0.005,
                    help="操纵度阈值 (Yoshikawa w=√det(JJᵀ)), 低于此值视为奇异 (默认 0.005)")
    ap.add_argument("--cond-threshold", type=float, default=75.0,
                    help="条件数阈值 (仅 --use-svd 时生效), 超过此值视为奇异 (默认 75)")
    ap.add_argument("--use-svd", action="store_true",
                    help="使用 SVD 条件数过滤 (精确但慢 ~100x); 默认用 det 操纵度 (快)")
    ap.add_argument("--no-singularity", action="store_true",
                    help="跳过奇异性过滤 (兼容旧行为, 不计算雅可比)")
    ap.add_argument("--no-timestamp", action="store_true",
                    help="关闭 out-dir 自动追加时间戳子目录 (默认追加)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.step = [0.5]
        print("[SMOKE] step -> 0.5 rad")

    out_dir = Path(args.out_dir)
    if not args.no_timestamp:
        ts = time.strftime("run_%Y%m%d_%H%M%S")
        out_dir = out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] out_dir = {out_dir}")

    diff, ntot = _verify_healpix(args.nside)
    assert diff == 0, f"healpix self-check failed: {diff}/{ntot}"
    print(f"[OK] HEALPix nside={args.nside}: torch impl matches healpy on {ntot} samples")

    # ========== 构建运动学模型 ==========
    # 关节 child link 列表 (用于计算几何雅可比)
    joint_child_links = [
        "Link1_L", "Link2_L", "Link3_L", "Link4_L",
        "Link5_L", "Link6_L", "Link7_L",
    ]

    cfg_dict = load_yaml(join_path(get_robot_configs_path(), args.robot))
    cfg_dict["robot_cfg"]["kinematics"]["ee_link"] = args.ee_link
    # link_names 包含所有关节 child link + ee_link, 用于 FK 输出所有关节位姿
    cfg_dict["robot_cfg"]["kinematics"]["link_names"] = joint_child_links + [args.ee_link]

    rw = None
    if args.collision == "self":
        rwc = RobotWorldConfig.load_from_config(robot_config=cfg_dict, world_model=None)
        rw = RobotWorld(rwc)
        km = rw.kinematics
        n_sph = km.kinematics_config.total_spheres
        print(f"[OK] collision=self enabled, n_spheres={n_sph}")
    else:
        robot_cfg = RobotConfig.from_dict(cfg_dict["robot_cfg"])
        km = CudaRobotModel(robot_cfg.kinematics)
        print("[OK] collision=none (FK only)")

    dev = torch.device("cuda:0")
    dof = km.get_dof()
    jl = km.get_joint_limits().position
    lower = jl[0].cpu().numpy()
    upper = jl[1].cpu().numpy()
    base_link = km.generator_config.base_link if km.generator_config else "unknown"
    ee_link = km.generator_config.ee_link if km.generator_config else args.ee_link
    print(f"[OK] FK chain: {base_link} --> {ee_link}")
    print(f"[OK] dof={dof}, joints={km.joint_names}")
    print(f"[OK] lower={lower}")
    print(f"[OK] upper={upper}")

    # 确定每个关节对应的 link 在 links_position 中的索引 (用于雅可比计算)
    joint_link_indices = []
    for jl_name in joint_child_links:
        idx = km.link_names.index(jl_name)
        joint_link_indices.append(idx)
    ee_link_idx = km.link_names.index(args.ee_link)
    print(f"[OK] link_names: {km.link_names}")
    print(f"[OK] joint_link_indices: {joint_link_indices}, ee_link_idx: {ee_link_idx}")

    # 奇异性过滤配置
    do_singularity = not args.no_singularity
    use_svd = args.use_svd
    if do_singularity:
        if use_svd:
            print(f"[OK] singularity filter: ON (SVD), cond_threshold={args.cond_threshold}")
        else:
            print(f"[OK] singularity filter: ON (det), manip_threshold={args.manip_threshold}")
    else:
        print(f"[OK] singularity filter: OFF (--no-singularity)")

    # 解析每关节步长
    if len(args.step) == 1:
        steps = [float(args.step[0])] * dof
    elif len(args.step) == dof:
        steps = [float(s) for s in args.step]
    else:
        raise ValueError(f"--step 需要 1 个或 {dof} 个值, 实际 {len(args.step)} 个")
    args.step = steps
    print(f"[OK] per-joint steps (rad) = {[round(s,4) for s in steps]}")

    grids = []
    for i in range(dof):
        si = steps[i]
        n_i = int(math.floor((upper[i] - lower[i]) / si)) + 1
        g = lower[i] + np.arange(n_i, dtype=np.float32) * si
        if g[-1] < upper[i] - 1e-9:
            g = np.concatenate([g, np.array([upper[i]], dtype=np.float32)])
        grids.append(g)
        print(f"  J{i+1}: step={si:.4f}, n={len(g):4d}, range=[{g[0]:+.4f}, {g[-1]:+.4f}]")
    sizes = [len(g) for g in grids]
    total = int(np.prod(sizes, dtype=np.int64))
    print(f"[OK] total samples N = {total:,}")

    bbox = np.asarray(args.bbox, dtype=np.float32)
    origin = bbox[:3]
    extent = bbox[3:] - bbox[:3]
    dims = np.ceil(extent / args.voxel).astype(np.int64)
    V = int(dims.prod())
    NPIX = 12 * args.nside * args.nside
    print(f"[OK] voxel grid dims={dims.tolist()}, V={V:,}, voxel={args.voxel} m")
    print(f"[OK] HEALPix npix={NPIX}")
    so3_mem_gb = V * NPIX / 1e9
    reach_mem_gb = V * 4 / 1e9
    print(f"[OK] est mem: reach_count~{reach_mem_gb:.2f}GB, so3_hit(bool)~{so3_mem_gb:.2f}GB")

    # GPU 显存预检查: 判断是否有足够显存分配 sing_so3_hit
    gpu_free = torch.cuda.mem_get_info(dev)[0] / 1e9
    base_needed = so3_mem_gb + reach_mem_gb + 2.0  # so3_hit + reach_count + 余量
    full_sing_needed = base_needed + so3_mem_gb + reach_mem_gb  # + sing_so3_hit + sing_reach_count
    lite_sing_needed = base_needed + reach_mem_gb  # + sing_reach_count only

    # 决定奇异性记录模式: full (体素+HEALPix) 或 lite (仅体素计数)
    sing_record_healpix = False
    if do_singularity:
        if full_sing_needed <= gpu_free:
            sing_record_healpix = True
            print(f"[OK] sing mode: FULL (体素位置 + HEALPix), need~{full_sing_needed:.1f}GB, free={gpu_free:.1f}GB")
        else:
            sing_record_healpix = False
            print(f"[WARN] sing mode: LITE (仅体素位置, 无 HEALPix), "
                  f"full需要~{full_sing_needed:.1f}GB > free={gpu_free:.1f}GB")
            print(f"[WARN] 如需奇异部分 HEALPix, 请增大 --voxel 或减小 --nside 或缩小 --bbox")

    gpu_needed = full_sing_needed if sing_record_healpix else lite_sing_needed
    if gpu_needed > gpu_free:
        print(f"[WARN] GPU free={gpu_free:.1f}GB, needed~{gpu_needed:.1f}GB")
        print(f"[WARN] 建议: 增大 --voxel (如 0.02) 或减小 --nside (如 8) 或缩小 --bbox")

    reach_count = torch.zeros(V, dtype=torch.int32, device=dev)
    so3_hit = torch.zeros(V * NPIX, dtype=torch.bool, device=dev)
    if do_singularity:
        sing_reach_count = torch.zeros(V, dtype=torch.int32, device=dev)
        if sing_record_healpix:
            sing_so3_hit = torch.zeros(V * NPIX, dtype=torch.bool, device=dev)

    origin_t = torch.from_numpy(origin).to(dev)
    voxel_t = torch.tensor(args.voxel, device=dev, dtype=torch.float32)
    dims_t = torch.from_numpy(dims).to(dev).long()
    in_bbox_count = torch.zeros((), dtype=torch.int64, device=dev)
    out_bbox_count = torch.zeros((), dtype=torch.int64, device=dev)
    coll_pass_count = torch.zeros((), dtype=torch.int64, device=dev)
    coll_drop_count = torch.zeros((), dtype=torch.int64, device=dev)
    sing_pass_count = torch.zeros((), dtype=torch.int64, device=dev)
    sing_drop_count = torch.zeros((), dtype=torch.int64, device=dev)

    inner_total = int(np.prod(sizes[1:], dtype=np.int64))
    g_gpu = [torch.from_numpy(g).to(dev) for g in grids]

    print(f"\n=== sampling: {sizes[0]} J1 slices, {inner_total:,} samples each ===")
    t_all = time.time()

    for i_j1, q1 in enumerate(g_gpu[0]):
        for s in range(0, inner_total, args.batch):
            e = min(s + args.batch, inner_total)
            B = e - s
            idx = torch.arange(s, e, device=dev, dtype=torch.long)
            cur = idx
            qs = [q1.expand(B)]
            local_dims = sizes[1:]
            sub_idx = []
            for d in local_dims:
                sub_idx.append(cur % d)
                cur = cur // d
            for k, sub in enumerate(sub_idx):
                qs.append(g_gpu[k + 1][sub])
            q = torch.stack(qs, dim=1).contiguous()

            st = km.get_state(q)

            pos = st.ee_position.clone()
            quat = st.ee_quaternion.clone()
            # 保留 link 位姿用于雅可比计算 (仅在需要奇异性过滤时)
            if do_singularity:
                link_pos = st.links_position.clone()
                link_quat = st.links_quaternion.clone()
            del q

            if rw is not None:
                # 自碰撞过滤
                sph = st.link_spheres_tensor.unsqueeze(1)  # [B, H=1, n_sph, 4]
                del st
                d_self = rw.get_self_collision_distance(sph)
                del sph
                if d_self.dim() > 1:
                    d_self = d_self.squeeze(-1)
                keep = d_self <= 0
                del d_self
                coll_pass_count += keep.sum()
                coll_drop_count += (~keep).sum()
                pos = pos[keep]
                quat = quat[keep]
                if do_singularity:
                    link_pos = link_pos[keep]
                    link_quat = link_quat[keep]
                del keep
            else:
                del st

            # bbox 过滤
            ix = ((pos - origin_t) / voxel_t).floor().long()
            in_bbox = ((ix >= 0) & (ix < dims_t)).all(dim=-1)
            in_bbox_count += in_bbox.sum()
            out_bbox_count += (~in_bbox).sum()
            ix = ix[in_bbox]
            quat = quat[in_bbox]
            if do_singularity:
                # 用 pos[in_bbox] 作为 ee_position (而非从 link_pos 取, 因为 ee 可能是 fixed link)
                ee_pos_filtered = pos[in_bbox]
                link_pos = link_pos[in_bbox]
                link_quat = link_quat[in_bbox]
            del pos, in_bbox

            if ix.numel() == 0:
                del ix, quat
                if do_singularity:
                    del link_pos, link_quat, ee_pos_filtered
                continue

            # ========== 体素线性索引 + HEALPix (对所有通过 bbox 的样本统一计算) ==========
            vox_lin = ix[:, 0] + dims_t[0] * (ix[:, 1] + dims_t[1] * ix[:, 2])
            axis_z = quat_wxyz_to_axis_z(quat)
            axis_z = axis_z / axis_z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            pix = healpix_ang2pix_ring_torch(axis_z, args.nside)
            del axis_z, ix

            # ========== 奇异性过滤 (在 bbox 过滤后, 减少计算量) ==========
            if do_singularity:
                J = compute_geometric_jacobian_batch(
                    link_pos, link_quat, ee_pos_filtered, joint_link_indices
                )
                if use_svd:
                    # 精确但慢: SVD 条件数
                    cond = compute_condition_number(J)
                    sing_keep = cond < args.cond_threshold
                    del cond
                else:
                    # 快速: det 操纵度 (默认)
                    manip = compute_manipulability(J)
                    sing_keep = manip > args.manip_threshold
                    del manip
                del J
                sing_pass_count += sing_keep.sum()
                sing_drop_count += (~sing_keep).sum()

                # ---- 被奇异性丢弃的样本: 记录体素位置 (+ HEALPix 如果显存够) ----
                sing_drop_mask = ~sing_keep
                if sing_drop_mask.any():
                    vox_lin_drop = vox_lin[sing_drop_mask]
                    sing_reach_count.scatter_add_(0, vox_lin_drop, torch.ones_like(vox_lin_drop, dtype=torch.int32))
                    if sing_record_healpix:
                        pix_drop = pix[sing_drop_mask]
                        lin_drop = vox_lin_drop * NPIX + pix_drop
                        sing_so3_hit[lin_drop] = True
                        del pix_drop, lin_drop
                    del vox_lin_drop
                del sing_drop_mask

                # 保留通过奇异性过滤的样本
                vox_lin = vox_lin[sing_keep]
                pix = pix[sing_keep]
                del link_pos, link_quat, ee_pos_filtered, sing_keep

            del quat

            if vox_lin.numel() == 0:
                del vox_lin, pix
                continue

            # ========== 体素统计 (只统计通过所有过滤的样本) ==========
            reach_count.scatter_add_(0, vox_lin, torch.ones_like(vox_lin, dtype=torch.int32))
            lin = vox_lin * NPIX + pix
            del pix, vox_lin
            so3_hit[lin] = True
            del lin

        # 每个 J1 切片结束后同步一次, 用于打印进度
        torch.cuda.synchronize()
        elapsed = time.time() - t_all
        eta = elapsed / (i_j1 + 1) * (sizes[0] - i_j1 - 1)
        print(f"  [J1 {i_j1+1}/{sizes[0]}] q1={q1.item():+.3f}  "
              f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

        # ---- 中间检查点 ----
        if args.ckpt_every and (i_j1 + 1) % args.ckpt_every == 0 and (i_j1 + 1) < sizes[0]:
            t_ck = time.time()
            so3_hit_3d_ck = so3_hit.view(V, NPIX)
            so3_cov_ck = torch.zeros(V, dtype=torch.int32, device=dev)
            chunk_ck = 1 << 16
            for cs in range(0, V, chunk_ck):
                ce = min(cs + chunk_ck, V)
                so3_cov_ck[cs:ce] = so3_hit_3d_ck[cs:ce].sum(dim=1, dtype=torch.int32)
            reach_mask_ck = reach_count > 0
            nz_idx_ck = torch.nonzero(reach_mask_ck, as_tuple=False).squeeze(1).cpu().numpy()
            nz_cnt_ck = reach_count[reach_mask_ck].cpu().numpy()
            nz_so3_ck = so3_cov_ck[reach_mask_ck].cpu().numpy()
            del so3_cov_ck
            ix_c = (nz_idx_ck % dims[0])
            iy_c = ((nz_idx_ck // dims[0]) % dims[1])
            iz_c = (nz_idx_ck // (dims[0] * dims[1]))
            ckpt_path = out_dir / "workspace_ckpt.npz"
            np.savez_compressed(
                ckpt_path,
                voxel_index=np.stack([ix_c, iy_c, iz_c], axis=1).astype(np.int32),
                reach_count=nz_cnt_ck.astype(np.int64),
                so3_hit=nz_so3_ck.astype(np.int32),
                npix=np.int32(NPIX),
                voxel=np.float32(args.voxel),
                bbox=bbox, dims=dims.astype(np.int32),
                j1_done=np.int32(i_j1 + 1),
                j1_total=np.int32(sizes[0]),
            )
            print(f"  [CKPT] {ckpt_path.name} reach={len(nz_idx_ck):,} "
                  f"({ckpt_path.stat().st_size/1e6:.1f}MB, +{time.time()-t_ck:.1f}s)")

    total_t = time.time() - t_all
    throughput = total / max(total_t, 1e-9) / 1e6
    print(f"\n=== done: total={total_t:.1f}s, throughput={throughput:.2f} M samples/s ===")
    if rw is not None:
        cp = int(coll_pass_count.item()); cd = int(coll_drop_count.item())
        rate = 100.0 * cd / max(cp + cd, 1)
        print(f"  coll-pass: {cp:,} / {cp+cd:,}")
        print(f"  coll-drop: {cd:,}  ({rate:.2f}% self-collision)")
    if do_singularity:
        sp = int(sing_pass_count.item()); sd = int(sing_drop_count.item())
        rate = 100.0 * sd / max(sp + sd, 1)
        if use_svd:
            thr_str = f"cond_threshold={args.cond_threshold}"
        else:
            thr_str = f"manip_threshold={args.manip_threshold}"
        print(f"  sing-pass: {sp:,} / {sp+sd:,}")
        print(f"  sing-drop: {sd:,}  ({rate:.2f}% near-singular, {thr_str})")
    print(f"  in-bbox : {in_bbox_count.item():,}")
    print(f"  out-bbox: {out_bbox_count.item():,} (dropped, check --bbox)")

    # 分块求和
    so3_hit_3d = so3_hit.view(V, NPIX)
    so3_cov = torch.zeros(V, dtype=torch.int32, device=dev)
    chunk = 1 << 16
    for cs in range(0, V, chunk):
        ce = min(cs + chunk, V)
        so3_cov[cs:ce] = so3_hit_3d[cs:ce].sum(dim=1, dtype=torch.int32)
    reach_mask = reach_count > 0
    n_reach = int(reach_mask.sum().item())
    print(f"  reachable voxels: {n_reach:,} / {V:,}  ({100.0*n_reach/V:.2f}%)")
    if n_reach:
        print(f"  cov-bins mean = {so3_cov[reach_mask].float().mean().item():.2f} / {NPIX}")
        print(f"  cov-bins max  = {so3_cov[reach_mask].max().item()} / {NPIX}")

    nz_idx = torch.nonzero(reach_mask, as_tuple=False).squeeze(1).cpu().numpy()
    nz_count = reach_count[reach_mask].cpu().numpy()
    nz_so3 = so3_cov[reach_mask].cpu().numpy()
    ix_arr = (nz_idx % dims[0])
    iy_arr = ((nz_idx // dims[0]) % dims[1])
    iz_arr = (nz_idx // (dims[0] * dims[1]))
    centers = np.stack([
        origin[0] + (ix_arr + 0.5) * args.voxel,
        origin[1] + (iy_arr + 0.5) * args.voxel,
        origin[2] + (iz_arr + 0.5) * args.voxel,
    ], axis=1).astype(np.float32)

    npz_path = out_dir / "workspace_data.npz"
    np.savez_compressed(
        npz_path,
        voxel_centers=centers,
        voxel_index=np.stack([ix_arr, iy_arr, iz_arr], axis=1).astype(np.int32),
        reach_count=nz_count.astype(np.int64),
        so3_hit=nz_so3.astype(np.int32),
        npix=np.int32(NPIX),
        voxel=np.float32(args.voxel),
        bbox=bbox,
        dims=dims.astype(np.int32),
    )

    # ========== 保存奇异性丢弃样本的点云 ==========
    if do_singularity:
        sing_reach_mask = sing_reach_count > 0
        n_sing_reach = int(sing_reach_mask.sum().item())
        print(f"  singular voxels: {n_sing_reach:,} / {V:,}  ({100.0*n_sing_reach/V:.2f}%)")

        # HEALPix 覆盖统计 (仅 full 模式)
        if sing_record_healpix:
            sing_so3_hit_3d = sing_so3_hit.view(V, NPIX)
            sing_so3_cov = torch.zeros(V, dtype=torch.int32, device=dev)
            for cs in range(0, V, chunk):
                ce = min(cs + chunk, V)
                sing_so3_cov[cs:ce] = sing_so3_hit_3d[cs:ce].sum(dim=1, dtype=torch.int32)
            if n_sing_reach:
                print(f"  sing cov-bins mean = {sing_so3_cov[sing_reach_mask].float().mean().item():.2f} / {NPIX}")
                print(f"  sing cov-bins max  = {sing_so3_cov[sing_reach_mask].max().item()} / {NPIX}")

        sing_nz_idx = torch.nonzero(sing_reach_mask, as_tuple=False).squeeze(1).cpu().numpy()
        sing_nz_count = sing_reach_count[sing_reach_mask].cpu().numpy()
        sing_ix_arr = (sing_nz_idx % dims[0])
        sing_iy_arr = ((sing_nz_idx // dims[0]) % dims[1])
        sing_iz_arr = (sing_nz_idx // (dims[0] * dims[1]))
        sing_centers = np.stack([
            origin[0] + (sing_ix_arr + 0.5) * args.voxel,
            origin[1] + (sing_iy_arr + 0.5) * args.voxel,
            origin[2] + (sing_iz_arr + 0.5) * args.voxel,
        ], axis=1).astype(np.float32)

        sing_npz_path = out_dir / "workspace_singular.npz"
        save_dict = dict(
            voxel_centers=sing_centers,
            voxel_index=np.stack([sing_ix_arr, sing_iy_arr, sing_iz_arr], axis=1).astype(np.int32),
            reach_count=sing_nz_count.astype(np.int64),
            npix=np.int32(NPIX),
            voxel=np.float32(args.voxel),
            bbox=bbox,
            dims=dims.astype(np.int32),
        )
        if sing_record_healpix:
            save_dict["so3_hit"] = sing_so3_cov[sing_reach_mask].cpu().numpy().astype(np.int32)
        np.savez_compressed(sing_npz_path, **save_dict)
        print(f"[SAVE] {sing_npz_path}  ({sing_npz_path.stat().st_size/1e6:.1f} MB)"
              f"  [{'with' if sing_record_healpix else 'without'} HEALPix]")
    else:
        n_sing_reach = 0

    meta = {
        "step": args.step, "voxel": args.voxel, "nside": args.nside, "npix": NPIX,
        "bbox": bbox.tolist(), "dims": dims.tolist(),
        "ee_link": args.ee_link, "robot": args.robot,
        "collision": args.collision,
        "singularity_filter": do_singularity,
        "singularity_method": ("svd" if use_svd else "det") if do_singularity else None,
        "cond_threshold": args.cond_threshold if (do_singularity and use_svd) else None,
        "manip_threshold": args.manip_threshold if (do_singularity and not use_svd) else None,
        "joint_names": list(km.joint_names),
        "joint_child_links": joint_child_links,
        "joint_lower": lower.tolist(), "joint_upper": upper.tolist(),
        "joint_grid_sizes": [int(s) for s in sizes],
        "total_samples": total,
        "coll_pass_samples": int(coll_pass_count.item()),
        "coll_drop_samples": int(coll_drop_count.item()),
        "sing_pass_samples": int(sing_pass_count.item()) if do_singularity else None,
        "sing_drop_samples": int(sing_drop_count.item()) if do_singularity else None,
        "in_bbox_samples": int(in_bbox_count.item()),
        "out_bbox_samples": int(out_bbox_count.item()),
        "reachable_voxels": n_reach, "total_voxels": V,
        "singular_voxels": n_sing_reach if do_singularity else None,
        "singular_healpix": sing_record_healpix if do_singularity else None,
        "elapsed_sec": total_t,
        "throughput_msps": throughput,
    }
    with open(out_dir / "workspace_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)")
    if do_singularity:
        print(f"[SAVE] {out_dir/'workspace_singular.npz'}")
    print(f"[SAVE] {out_dir/'workspace_meta.json'}")


if __name__ == "__main__":
    main()
