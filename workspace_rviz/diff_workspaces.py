#!/usr/bin/env python3
"""diff two workspace_data.npz files (vectorized via numpy.isin)."""
import argparse, json
from pathlib import Path
import numpy as np


def load_voxels(npz_path: Path):
    d = np.load(npz_path)
    idx = d["voxel_index"].astype(np.int64)
    centers = d["voxel_centers"].astype(np.float32)
    dims = tuple(int(x) for x in d["dims"])
    keys = idx[:, 0] * (dims[1] * dims[2]) + idx[:, 1] * dims[2] + idx[:, 2]
    return {"keys": keys, "idx": idx.astype(np.int32), "centers": centers,
            "dims": dims, "voxel": float(d["voxel"]), "bbox": d["bbox"].astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] baseline = {args.baseline}")
    A = load_voxels(Path(args.baseline))
    print(f"[LOAD] target   = {args.target}")
    B = load_voxels(Path(args.target))
    assert A["dims"] == B["dims"]

    only_a = ~np.isin(A["keys"], B["keys"], assume_unique=True)
    only_b = ~np.isin(B["keys"], A["keys"], assume_unique=True)
    n_only_a = int(only_a.sum()); n_only_b = int(only_b.sum())
    n_common = len(A["keys"]) - n_only_a

    print("\n=== voxel set stats ===")
    print(f"  baseline      : {len(A['keys']):>10d}")
    print(f"  target        : {len(B['keys']):>10d}")
    print(f"  common        : {n_common:>10d}")
    print(f"  only baseline : {n_only_a:>10d}  (pruned)")
    print(f"  only target   : {n_only_b:>10d}  (should be 0)")

    np.savez_compressed(
        out_dir / "diff_data.npz",
        only_baseline_centers=A["centers"][only_a],
        only_baseline_idx=A["idx"][only_a],
        only_target_centers=B["centers"][only_b],
        only_target_idx=B["idx"][only_b],
        dims=np.array(A["dims"], dtype=np.int32),
        voxel=np.float32(A["voxel"]),
        bbox=A["bbox"],
    )
    meta = {
        "baseline": args.baseline, "target": args.target,
        "dims": list(A["dims"]), "voxel": A["voxel"], "bbox": A["bbox"].tolist(),
        "n_baseline": len(A["keys"]), "n_target": len(B["keys"]),
        "n_common": n_common, "n_only_baseline": n_only_a, "n_only_target": n_only_b,
        "shrink_ratio": (len(A["keys"]) - len(B["keys"])) / max(len(A["keys"]), 1),
    }
    with open(out_dir / "diff_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVE] {out_dir/'diff_data.npz'}")
    print(f"[SAVE] {out_dir/'diff_meta.json'}")


if __name__ == "__main__":
    main()
