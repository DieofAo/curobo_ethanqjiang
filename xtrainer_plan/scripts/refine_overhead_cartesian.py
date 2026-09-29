#!/usr/bin/env python3
"""Record denser grid configs for explicitly selected Cartesian candidates.

Selection is supplied by the caller, not inferred from IK or preliminary logs.
This CPU-only generator invokes the existing config derivation CLI. It never
plans, changes defaults, or overwrites a prior experiment. The optional matched
reference retains the baseline source's mount, place X/Z, Home and constraints,
except place Y=-0.12, joint clip=0.14 and disabling the independent IK prescreen.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    mount = cfg["robot"]["mount_transform"]
    place = cfg["pick_place"]["place"]["position"]
    if (len(mount) != 4 or any(len(row) != 4 for row in mount) or len(place) != 3 or
            not all(math.isfinite(v) for row in mount for v in row) or
            not all(math.isfinite(v) for v in place)):
        raise ValueError(f"Invalid mount/place in {path}")
    if "overhead" not in cfg:
        raise ValueError(f"Expected mounted experiment config: {path}")
    grid = cfg["pick_place"]["grasp_grid"]
    if grid.get("perimeter_only", False):
        raise ValueError("Refinement requires a complete grid, not perimeter-only sampling")
    return cfg


def coordinates(cfg):
    return ([cfg["robot"]["mount_transform"][i][3] for i in range(3)],
            cfg["pick_place"]["place"]["position"])


def preflight(manifest_path, names, baseline_source):
    """Validate all selections, sources and hashes before creating any output."""
    manifest = json.loads(manifest_path.read_text())
    candidates = manifest["candidates"]
    if len(candidates) != manifest["n_configs"]:
        raise ValueError("Coarse manifest candidate count differs")
    by_name = {row["name"]: row for row in candidates}
    if len(by_name) != len(candidates):
        raise ValueError("Coarse manifest contains duplicate names")
    if not names or len(set(names)) != len(names):
        raise ValueError("Selected names must be nonempty and unique")
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Selected names are absent from the coarse manifest: {missing}")
    if manifest.get("source") and manifest.get("source_sha256"):
        if sha256(manifest["source"]) != manifest["source_sha256"]:
            raise ValueError("Original coarse source changed after its manifest was recorded")
    sources = []
    for name in names:
        coarse = by_name[name]
        source = Path(coarse["config"]).resolve()
        if sha256(source) != coarse["config_sha256"]:
            raise ValueError(f"Coarse config changed after recording: {source}")
        cfg = read_config(source)
        base, place = coordinates(cfg)
        if base != coarse["base_xyz_m"] or place != coarse["place_xyz_m"]:
            raise ValueError(f"Coarse config coordinates differ from its manifest: {name}")
        sources.append({"source": source, "config": cfg, "sha256": coarse["config_sha256"],
                        "role": "selected_coarse_candidate", "coarse_name": name,
                        "coarse_config": str(source), "coarse_result": coarse["result"]})
    if baseline_source is not None:
        source = Path(baseline_source).resolve()
        cfg = read_config(source)
        sources.append({"source": source, "config": cfg, "sha256": sha256(source),
                        "role": "matched_baseline", "baseline_source": str(source)})
    return sources


def validate_derived(source, derived, size, result, is_baseline):
    expected = copy.deepcopy(source)
    expected["pick_place"]["grasp_grid"].update(rows=size, cols=size)
    expected["output"]["dir"] = str(result)
    if is_baseline:
        expected["robot"]["joint_limit_clip"] = .14
        expected["pick_place"]["place"]["position"][1] = -.12
        expected["pick_place"].setdefault("criterion", {})["prescreen_by_ik"] = False
    derived = copy.deepcopy(derived)
    # These are the only provenance fields the derivation CLI replaces.
    for cfg in (expected, derived):
        for key in ("derived_from", "variant_options"):
            cfg["overhead"].pop(key, None)
    if expected != derived:
        raise ValueError("Derived config changed fields outside the explicitly allowed refinement")


def generate(manifest_path, names, root, prefix, size=5, baseline_source=None):
    manifest_path, root = Path(manifest_path).resolve(), Path(root).resolve()
    if size < 2:
        raise ValueError("Grid size must be at least 2")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", prefix):
        raise ValueError("Prefix must be an alphanumeric filename component, with optional '-' or '_'")
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {root}")
    sources = preflight(manifest_path, names, baseline_source)
    manifest_hash = sha256(manifest_path)
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("derive_overhead_experiment.py")
    rows = []
    for index, source in enumerate(sources):
        if sha256(source["source"]) != source["sha256"]:
            raise ValueError(f"Source changed during refinement generation: {source['source']}")
        name = f"{prefix}_{index:02d}"
        config, result = root / "configs" / f"{name}.json", root / "runs" / name
        command = [sys.executable, str(script), "--source", str(source["source"]),
                   "--output", str(config), "--rows", str(size), "--cols", str(size),
                   "--run-output-dir", str(result)]
        if source["role"] == "matched_baseline":
            command += ["--place-y", "-0.12", "--joint-limit-clip", "0.14", "--no-prescreen"]
        subprocess.run(command, check=True)
        cfg = read_config(config)
        validate_derived(source["config"], cfg, size, result, source["role"] == "matched_baseline")
        base, place = coordinates(cfg)
        row = {"index": index, "name": name, "config": str(config), "result": str(result),
               "base_xyz_m": base, "place_xyz_m": place, "config_sha256": sha256(config),
               "derive_command": command, "role": source["role"],
               "source_config": str(source["source"]), "source_config_sha256": source["sha256"]}
        for key in ("coarse_name", "coarse_config", "coarse_result", "baseline_source"):
            if key in source:
                row[key] = source[key]
        rows.append(row)
    if sha256(manifest_path) != manifest_hash:
        raise ValueError("Coarse manifest changed during refinement generation")
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "source": str(manifest_path), "source_sha256": manifest_hash,
              "source_type": "coarse_experiment_manifest", "coarse_manifest": str(manifest_path),
              "parameters": {"names": list(names), "out_root": str(root), "prefix": prefix,
                             "size": size, "baseline_source": str(Path(baseline_source).resolve())
                             if baseline_source is not None else None},
              "scope": "Explicitly selected candidates, denser grid; no planning or default changes.",
              "n_configs": len(rows), "candidates": rows}
    with (root / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--baseline-source", type=Path)
    args = parser.parse_args()
    report = generate(args.manifest, args.names, args.out_root, args.prefix, args.size, args.baseline_source)
    print(f"Generated {report['n_configs']} candidates: {args.out_root.resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
