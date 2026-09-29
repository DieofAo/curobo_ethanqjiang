#!/usr/bin/env python3
"""Record a non-overwriting base XY x place X Cartesian smoke experiment."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--base-x", type=float, nargs="+", required=True)
    ap.add_argument("--base-y", type=float, nargs="+", required=True)
    ap.add_argument("--base-z", type=float, default=.65)
    ap.add_argument("--place-x", type=float, nargs="+", required=True)
    ap.add_argument("--place-y", type=float, default=-.12)
    ap.add_argument("--clip", type=float, default=.14)
    ap.add_argument("--size", type=int, default=3)
    args = ap.parse_args()
    for values in (args.base_x, args.base_y, args.place_x):
        if len(set(values)) != len(values):
            ap.error("Sweep coordinates must be unique")
    if not all(math.isfinite(v) for v in args.base_x + args.base_y + args.place_x
               + [args.base_z, args.place_y, args.clip]) or args.clip < 0 or args.size < 2:
        ap.error("Finite coordinates, nonnegative clip and size >= 2 required")
    if not args.prefix or Path(args.prefix).name != args.prefix:
        ap.error("Prefix must be a nonempty filename component")
    source = Path(args.source).resolve()
    root = Path(args.out_root).resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {root}")
    root.mkdir(parents=True)
    script = Path(__file__).with_name("derive_overhead_experiment.py")
    rows = []
    for index, (bx, by, px) in enumerate(itertools.product(args.base_x, args.base_y, args.place_x)):
        name = f"{args.prefix}_{index:02d}"
        config = root / "configs" / f"{name}.json"
        result = root / "runs" / name
        command = [sys.executable, str(script), "--source", str(source), "--output", str(config),
                   "--mount-position", str(bx), str(by), str(args.base_z),
                   "--place-x", str(px), "--place-y", str(args.place_y),
                   "--joint-limit-clip", str(args.clip), "--rows", str(args.size),
                   "--cols", str(args.size), "--no-prescreen", "--run-output-dir", str(result)]
        subprocess.run(command, check=True)
        cfg = json.loads(config.read_text())
        rows.append({"index": index, "name": name, "config": str(config), "result": str(result),
                     "base_xyz_m": [bx, by, args.base_z],
                     "place_xyz_m": cfg["pick_place"]["place"]["position"],
                     "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                     "derive_command": command})
    manifest = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "parameters": vars(args), "n_configs": len(rows), "candidates": rows}
    with (root / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Generated {len(rows)} candidates: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
