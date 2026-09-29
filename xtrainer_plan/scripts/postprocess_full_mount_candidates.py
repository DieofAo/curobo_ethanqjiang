#!/usr/bin/env python3
"""Build audited full-grid reports for selected mounts and compare five candidates.

Run after planning and both independent audits have completed.  One candidate
can be processed while other full runs are still in progress; --compare requires
all five candidates from the experiment manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from compare_full_mount_candidates import load_candidate, matched_data
from plot_full_mount_joint_distributions import sha256


SCRIPTS = Path(__file__).resolve().parent


def manifest_names(root: Path) -> list[str]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["candidates"]
    if manifest["n_configs"] != len(rows) or len(rows) != 5:
        raise ValueError("Expected exactly five full-run candidates in manifest")
    names = []
    for row in rows:
        name = row["name"]
        if (not isinstance(name, str) or name in names or
                Path(name).name != name or name in ("", ".", "..")):
            raise ValueError(f"Invalid or duplicate candidate name: {name!r}")
        if Path(row["result"]).resolve() != (root / "runs" / name).resolve():
            raise ValueError(f"Manifest result path differs for {name}")
        names.append(name)
    return names


def call(script: str, *arguments: object) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / script),
                    *(str(argument) for argument in arguments)], check=True)


def report_one(root: Path, name: str) -> None:
    run = root / "runs" / name
    report = root / "reports" / name
    joint = report / "joint_distributions"
    scene = report / "scene_distribution"
    angle = report / "grasp_search_angle.png"
    angle_audit = report / "grasp_search_angle.audit.json"
    report.mkdir(parents=True, exist_ok=True)
    joint_files = ("joint_distribution_summary.json", "joint_distribution_cases.csv",
                   "joint_distribution_comparison.png", "min_raw_limit_distance.png",
                   "min_effective_limit_margin.png", "max_full_cycle_single_joint_span.png",
                   "max_stage_j1_j5_span.png", "nearest_limit_joint.png")
    for folder, expected, script in (
        (joint, joint_files, "plot_full_mount_joint_distributions.py"),
        (scene, ("task_scene_distribution.json", "task_scene_distribution.png"),
         "plot_full_mount_task_scene.py"),
    ):
        if folder.exists():
            if any(not (folder / filename).is_file() or
                   (folder / filename).stat().st_size == 0 for filename in expected):
                raise RuntimeError(f"Incomplete report directory; inspect before retrying: {folder}")
        else:
            call(script, run, "--out-dir", folder)
    # This reload checks both independent audits and all source hashes before
    # the grasp-angle map is accepted or generated.
    load_candidate(root, name)
    if angle.exists() != angle_audit.exists():
        raise RuntimeError(f"Partial grasp-angle report; inspect before retrying: {report}")
    if not angle.exists():
        call("plot_overhead_results.py", run, "--out", angle,
             "--annotate", "none")
        angle_audit.write_text(json.dumps({
            "source_run": str(run.resolve()),
            "trajectory_meta_sha256": sha256(run / "trajectory_meta.json"),
            "plotter_sha256": sha256(SCRIPTS / "plot_overhead_results.py"),
            "plot_sha256": sha256(angle),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if angle.stat().st_size == 0 or angle_audit.stat().st_size == 0:
        raise RuntimeError(f"Empty grasp-angle report: {report}")
    angle_source = json.loads(angle_audit.read_text(encoding="utf-8"))
    if (Path(angle_source.get("source_run", "")).resolve() != run.resolve() or
            angle_source.get("trajectory_meta_sha256") != sha256(run / "trajectory_meta.json") or
            angle_source.get("plotter_sha256") != sha256(SCRIPTS / "plot_overhead_results.py") or
            angle_source.get("plot_sha256") != sha256(angle)):
        raise RuntimeError(f"Grasp-angle report provenance changed: {report}")
    cell_maps = report / "cell_maps"
    cell_files = ("success_cell_map.png", "raw_limit_distance_cell_map.png",
                  "effective_limit_margin_cell_map.png", "max_joint_span_cell_map.png",
                  "success_task_style.png", "plot_data_summary.json", "README.md")
    if cell_maps.exists():
        if any(not (cell_maps / filename).is_file() or
               (cell_maps / filename).stat().st_size == 0 for filename in cell_files):
            raise RuntimeError(f"Incomplete cell-map directory; inspect before retrying: {cell_maps}")
        summary = json.loads((cell_maps / "plot_data_summary.json").read_text(encoding="utf-8"))
        sources = (
            ("source_joint_csv_sha256", joint / "joint_distribution_cases.csv"),
            ("source_joint_json_sha256", joint / "joint_distribution_summary.json"),
            ("source_scene_json_sha256", scene / "task_scene_distribution.json"),
        )
        images = cell_files[:5]
        if (summary.get("candidate") != name or
                any(summary.get(key) != sha256(path) for key, path in sources) or
                summary.get("figure_sha256") !=
                {filename: sha256(cell_maps / filename) for filename in images}):
            raise RuntimeError(f"Existing cell maps refer to different source data: {cell_maps}")
    else:
        call("plot_full_mount_cell_maps.py", root, name, "--task-style")
    print(f"[ok] {name}: joint, scene, grasp-angle, cell-map reports", flush=True)


def compare(root: Path, names: list[str]) -> None:
    out = root / "reports" / "comparison"
    if out.exists():
        summary_path = out / "comparison_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"Incomplete comparison directory; inspect before retrying: {out}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        recorded = summary.get("candidates", [])
        if [row["name"] for row in recorded] != names:
            raise RuntimeError(f"Existing comparison uses different candidates: {out}")
        current_candidates = [load_candidate(root, name) for name in names]
        current_records, current_summary = matched_data(current_candidates)
        if summary != current_summary:
            raise RuntimeError(f"Existing comparison differs from audited source data: {out}")
        expected = [out / "full_candidate_comparison.png",
                    out / "same_point_metrics.csv", out / "README.md"]
        expected += [out / f"{name}_shared_scale_distributions.png" for name in names]
        expected += [out / "artifact_sha256.json"]
        if any(not path.is_file() or path.stat().st_size == 0 for path in expected):
            raise RuntimeError(f"Existing comparison is incomplete: {out}")
        with (out / "same_point_metrics.csv").open(newline="", encoding="utf-8") as stream:
            saved_records = list(csv.DictReader(stream))
        if saved_records != [{key: str(value) for key, value in record.items()}
                             for record in current_records]:
            raise RuntimeError(f"Existing same-point CSV differs from audited source data: {out}")
        artifact_hashes = json.loads((out / "artifact_sha256.json").read_text(encoding="utf-8"))
        if artifact_hashes != {path.name: sha256(path) for path in expected
                               if path.name != "artifact_sha256.json"}:
            raise RuntimeError(f"Existing comparison images or tables changed: {out}")
        print(f"[reuse] {out}", flush=True)
        return
    call("compare_full_mount_candidates.py", root, "--names", *names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Five-candidate full-run result root")
    parser.add_argument("--names", nargs="+", help="Subset to postprocess; default all five")
    parser.add_argument("--compare", action="store_true",
                        help="Also create the five-candidate shared-scale comparison")
    args = parser.parse_args()
    root = args.root.resolve()
    all_names = manifest_names(root)
    names = args.names or all_names
    if len(set(names)) != len(names) or set(names) - set(all_names):
        parser.error("--names must be distinct names in the five-candidate manifest")
    if args.compare and names != all_names:
        parser.error("--compare requires all five candidates in manifest order")
    for name in names:
        report_one(root, name)
    if args.compare:
        compare(root, names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
