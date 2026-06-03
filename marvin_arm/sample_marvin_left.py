#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Marvin M6 左臂 7-DOF 工作空间采样器（curobo + GPU 并行 FK）

策略与 workspace_rviz/sample_workspace.py 完全一致:
  1) 按关节限位以固定步长生成一维网格; 笛卡尔积逐"J1 外层切片"流式喂给 FK。
  2) FK 出 (ee_pos, ee_quat); 位置量化成体素索引;
     四元数取局部 z 轴方向用 HEALPix 量化成姿态桶。
  3) 用 torch.scatter_add_ 在 GPU 上累加 reach_count[V] 与 so3_hit[V*NPIX] (bool)。
  4) 跑完后 reach_count.bool() 即"可达体素"; so3_hit.view(V,NPIX).sum(1)/NPIX 即"姿态覆盖率"。

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


def quat_wxyz_to_axis_z(q: torch.Tensor) -> torch.Tensor:
    """curobo 输出 (w,x,y,z) 四元数; 返回 ee 局部 +z 在世界系下的方向向量。"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx = 2.0 * (x * z + w * y)
    vy = 2.0 * (y * z - w * x)
    vz = 1.0 - 2.0 * (x * x + y * y)
    return torch.stack([vx, vy, vz], dim=-1)


def healpix_ang2pix_ring_torch(vec: torch.Tensor, nside: int) -> torch.Tensor:
    """vec: (N,3) 单位向量, 返回 RING 编号 (N,) int64。等价 healpy.ang2pix(nside, theta, phi, nest=False)。"""
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
                    help="关节步长(rad). 1 个值=广播到所有关节; N 个值=每关节独立步长. 例: --step 0.02 0.02 0.02 0.02 0.15 0.15 0.15")
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--nside", type=int, default=2)
    # 默认 bbox 覆盖 marvin 左臂工作空间: x∈[-1,1], y∈[-0.5,1.5], z∈[-0.2,2.0]
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

    cfg_dict = load_yaml(join_path(get_robot_configs_path(), args.robot))
    cfg_dict["robot_cfg"]["kinematics"]["ee_link"] = args.ee_link
    cfg_dict["robot_cfg"]["kinematics"]["link_names"] = [args.ee_link]

    rw = None
    if args.collision == "self":
        # RobotWorld 内部会构建自己的 CudaRobotModel; 复用它的 km, 不再单独建一个
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

    # 解析每关节步长: 单值广播, 否则必须等于 dof
    if len(args.step) == 1:
        steps = [float(args.step[0])] * dof
    elif len(args.step) == dof:
        steps = [float(s) for s in args.step]
    else:
        raise ValueError(f"--step 需要 1 个或 {dof} 个值, 实际 {len(args.step)} 个")
    args.step = steps  # 后续 meta 输出使用列表
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
    print(f"[OK] est mem: reach_count~{V*4/1e9:.2f}GB, so3_hit(bool)~{V*NPIX/1e9:.2f}GB")

    reach_count = torch.zeros(V, dtype=torch.int32, device=dev)
    so3_hit = torch.zeros(V * NPIX, dtype=torch.bool, device=dev)

    origin_t = torch.from_numpy(origin).to(dev)
    voxel_t = torch.tensor(args.voxel, device=dev, dtype=torch.float32)
    dims_t = torch.from_numpy(dims).to(dev).long()
    in_bbox_count = torch.zeros((), dtype=torch.int64, device=dev)
    out_bbox_count = torch.zeros((), dtype=torch.int64, device=dev)
    coll_pass_count = torch.zeros((), dtype=torch.int64, device=dev)
    coll_drop_count = torch.zeros((), dtype=torch.int64, device=dev)

    inner_total = int(np.prod(sizes[1:], dtype=np.int64))
    g_gpu = [torch.from_numpy(g).to(dev) for g in grids]

    print(f"\n=== sampling: {sizes[0]} J1 slices, {inner_total:,} samples each ===")
    t_all = time.time(); fk_time = 0.0; agg_time = 0.0

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

            torch.cuda.synchronize(); t0 = time.time()
            st = km.get_state(q)
            torch.cuda.synchronize(); fk_time += time.time() - t0

            t0 = time.time()
            pos = st.ee_position
            quat = st.ee_quaternion

            if rw is not None:
                # 自碰撞过滤: d_self > 0 表示有穿透, ==0 表示无碰撞
                sph = st.link_spheres_tensor.unsqueeze(1)  # [B, H=1, n_sph, 4]
                d_self = rw.get_self_collision_distance(sph)
                if d_self.dim() > 1:
                    d_self = d_self.squeeze(-1)
                keep = d_self <= 0
                coll_pass_count += keep.sum()
                coll_drop_count += (~keep).sum()
                pos = pos[keep]
                quat = quat[keep]

            ix = ((pos - origin_t) / voxel_t).floor().long()
            in_bbox = ((ix >= 0) & (ix < dims_t)).all(dim=-1)
            in_bbox_count += in_bbox.sum()
            out_bbox_count += (~in_bbox).sum()
            ix = ix[in_bbox]
            quat = quat[in_bbox]
            if ix.numel() > 0:
                vox_lin = ix[:, 0] + dims_t[0] * (ix[:, 1] + dims_t[1] * ix[:, 2])
                reach_count.scatter_add_(0, vox_lin, torch.ones_like(vox_lin, dtype=torch.int32))
                axis_z = quat_wxyz_to_axis_z(quat)
                axis_z = axis_z / axis_z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                pix = healpix_ang2pix_ring_torch(axis_z, args.nside)
                lin = vox_lin * NPIX + pix
                so3_hit[lin] = True
            agg_time += time.time() - t0

        elapsed = time.time() - t_all
        eta = elapsed / (i_j1 + 1) * (sizes[0] - i_j1 - 1)
        print(f"  [J1 {i_j1+1}/{sizes[0]}] q1={q1.item():+.3f}  "
              f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s  fk={fk_time:.1f}s  agg={agg_time:.1f}s")

        # ---- 中间检查点: 仅落稀疏可达体素和朝向覆盖, 覆盖式写入 ----
        if args.ckpt_every and (i_j1 + 1) % args.ckpt_every == 0 and (i_j1 + 1) < sizes[0]:
            t_ck = time.time()
            so3_hit_3d_ck = so3_hit.view(V, NPIX)
            so3_cov_ck = torch.zeros(V, dtype=torch.int32, device=dev)
            chunk_ck = 1 << 16  # 64K 体素一块, 避免 nside>=6 时 sum 临时张量 OOM
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
    print(f"\n=== done: total={total_t:.1f}s, fk={fk_time:.1f}s, agg={agg_time:.1f}s ===")
    if rw is not None:
        cp = int(coll_pass_count.item()); cd = int(coll_drop_count.item())
        rate = 100.0 * cd / max(cp + cd, 1)
        print(f"  coll-pass: {cp:,} / {cp+cd:,}")
        print(f"  coll-drop: {cd:,}  ({rate:.2f}% self-collision)")
    print(f"  in-bbox : {in_bbox_count.item():,}")
    print(f"  out-bbox: {out_bbox_count.item():,} (dropped, check --bbox)")

    # 分块求和: 避免 (V,NPIX) bool->int64 一次性临时分配撑爆显存
    so3_hit_3d = so3_hit.view(V, NPIX)
    so3_cov = torch.zeros(V, dtype=torch.int32, device=dev)
    chunk = 1 << 16  # 64K 体素一块, 避免 nside>=6 时 sum 临时张量 OOM
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
    meta = {
        "step": args.step, "voxel": args.voxel, "nside": args.nside, "npix": NPIX,
        "bbox": bbox.tolist(), "dims": dims.tolist(),
        "ee_link": args.ee_link, "robot": args.robot,
        "collision": args.collision,
        "joint_names": list(km.joint_names),
        "joint_lower": lower.tolist(), "joint_upper": upper.tolist(),
        "joint_grid_sizes": [int(s) for s in sizes],
        "total_samples": total,
        "coll_pass_samples": int(coll_pass_count.item()),
        "coll_drop_samples": int(coll_drop_count.item()),
        "in_bbox_samples": int(in_bbox_count.item()),
        "out_bbox_samples": int(out_bbox_count.item()),
        "reachable_voxels": n_reach, "total_voxels": V,
        "elapsed_sec": total_t, "fk_sec": fk_time, "agg_sec": agg_time,
        "throughput_msps": total / max(fk_time, 1e-9) / 1e6,
    }
    with open(out_dir / "workspace_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)")
    print(f"[SAVE] {out_dir/'workspace_meta.json'}")


if __name__ == "__main__":
    main()
