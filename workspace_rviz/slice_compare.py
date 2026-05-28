#!/usr/bin/env python3
"""
切片对比图：把 baseline / target / diff 沿 z 轴投影或切片，绘制三联图。

用法：
  python3 slice_compare.py \
      --baseline runs/no_collision_step0p2/workspace_data.npz \
      --target   runs/self_collision_step0p2/workspace_data.npz \
      --out_dir  runs/diff_no_vs_self_step0p2/figs
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_grid(npz_path: Path):
    d = np.load(npz_path)
    idx = d["voxel_index"].astype(np.int64)
    dims = tuple(int(x) for x in d["dims"])
    grid = np.zeros(dims, dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid, dims, d["bbox"].astype(np.float32), float(d["voxel"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] baseline ...")
    G_a, dims, bbox, voxel = load_grid(Path(args.baseline))
    print("[LOAD] target ...")
    G_b, _, _, _ = load_grid(Path(args.target))
    G_diff = G_a & (~G_b)

    print(f"  dims={dims}, voxel={voxel:.4f}, bbox={bbox.tolist()}")
    print(f"  baseline cells = {int(G_a.sum())}")
    print(f"  target   cells = {int(G_b.sum())}")
    print(f"  diff     cells = {int(G_diff.sum())}")

    extent = [bbox[0], bbox[3], bbox[1], bbox[4]]

    # Figure 1: z-axis max projection
    proj_a = G_a.any(axis=2).T[::-1]
    proj_b = G_b.any(axis=2).T[::-1]
    proj_d = G_diff.any(axis=2).T[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [
        f"baseline (no collision)\n{int(G_a.sum())} voxels",
        f"target (self collision)\n{int(G_b.sum())} voxels",
        f"diff (pruned by self-coll)\n{int(G_diff.sum())} voxels",
    ]
    for ax, img, t, cm in zip(axes, [proj_a, proj_b, proj_d], titles, ["Blues", "Greens", "Reds"]):
        ax.imshow(img, extent=extent, origin="upper", cmap=cm, vmin=0, vmax=1)
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Z-axis Max Projection (top-down view)", fontsize=14)
    fig.tight_layout()
    fout = out_dir / "z_max_projection_compare.png"
    fig.savefig(fout, dpi=130)
    plt.close(fig)
    print(f"[SAVE] {fout}")

    # Figure 2: 6 z-slices x 3 rows
    z_levels = np.linspace(int(dims[2] * 0.15), int(dims[2] * 0.85), 6).astype(int)
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    for col, zi in enumerate(z_levels):
        z_world = bbox[2] + (zi + 0.5) * voxel
        for row, (G, cm, lab) in enumerate(
            [(G_a, "Blues", "baseline"), (G_b, "Greens", "target"), (G_diff, "Reds", "diff")]
        ):
            slc = G[:, :, zi].T[::-1]
            ax = axes[row, col]
            ax.imshow(slc, extent=extent, origin="upper", cmap=cm, vmin=0, vmax=1)
            ax.set_title(f"{lab} z={z_world:+.2f}m  ({int(slc.sum())})", fontsize=9)
            ax.set_aspect("equal")
            if row == 2:
                ax.set_xlabel("x [m]")
            if col == 0:
                ax.set_ylabel("y [m]")
    fig.suptitle("Z-slice Comparison: baseline / target / diff", fontsize=14)
    fig.tight_layout()
    fout = out_dir / "z_slices_compare.png"
    fig.savefig(fout, dpi=110)
    plt.close(fig)
    print(f"[SAVE] {fout}")

    # Figure 3: per-z-layer voxel count curve
    counts_a = G_a.sum(axis=(0, 1))
    counts_b = G_b.sum(axis=(0, 1))
    counts_d = G_diff.sum(axis=(0, 1))
    z_axis = bbox[2] + (np.arange(dims[2]) + 0.5) * voxel
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(z_axis, counts_a, "b-", label="baseline", linewidth=1.6)
    ax1.plot(z_axis, counts_b, "g-", label="target (self-coll)", linewidth=1.6)
    ax1.set_xlabel("z [m]")
    ax1.set_ylabel("voxel count per z-layer", color="b")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(z_axis, counts_d, "r-", alpha=0.7, label="diff (pruned)")
    ax2.set_ylabel("pruned voxels per z-layer", color="r")
    ax2.legend(loc="upper right")
    fig.suptitle("Voxel count vs Z (per layer)")
    fig.tight_layout()
    fout = out_dir / "z_layer_count.png"
    fig.savefig(fout, dpi=130)
    plt.close(fig)
    print(f"[SAVE] {fout}")
    print("\nDone.")


if __name__ == "__main__":
    main()
