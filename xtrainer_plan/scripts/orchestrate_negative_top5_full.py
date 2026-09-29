#!/usr/bin/env python3
"""Resume audited V67 negative smoke, Top 5 full planning and offline outputs.

The script never deletes or overwrites an existing result. Completed stages are
checked against current source hashes before reuse; an interrupted stage with
only some published files stops for inspection instead of guessing which files
are safe to replace. Launch via run_negative_top5_offline_pipeline.sh so CUDA
extension compilation uses the recorded cuRobo environment.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from postprocess_v67_negative_smoke import OUTPUT_NAMES, validate_complete_smoke
from render_v67_negative_best_angle_walkthrough import LEGACY_RENDERER, VIDEO_NAME
from prepare_negative_local_y_top5_full import DEFAULT_REFERENCE, select_and_build
from run_local_y_mount_sweep import verify_inputs
from summarize_local_y_mount_sweep import aggregate
from summarize_overhead_cartesian import inspect_candidate


SCRIPTS = Path(__file__).resolve().parent
SMOKE_TERMINAL = {"completed_verified", "completed_zero_success", "home_failed"}
BEST_3D_FILES = ("best_angle_by_position.json", "best_angle_by_position.csv",
                 "best_angle_3d.html", "plotly.min.js")
BEST_VIDEO_FILES = (VIDEO_NAME, "frame_manifest.json", "video_audit.json",
                    "all_18_detail_frames_contact_sheet.jpg", "README.md")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def call(*args: object, env: dict[str, str] | None = None) -> None:
    command = [str(arg) for arg in args]
    print(f"[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, env=env)


def check_environment() -> None:
    repo = SCRIPTS.parents[1]
    expected_python = Path("/home/ethanqjiang/miniconda3/envs/curobo/bin/python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError("Launch through run_negative_top5_offline_pipeline.sh with cuRobo Python")
    targets = Path("/home/ethanqjiang/miniconda3/envs/curobo/targets/x86_64-linux")
    fields = {
        "PATH": str(expected_python.parent),
        "PYTHONPATH": str(repo / "src"),
        "LD_LIBRARY_PATH": (str(expected_python.parents[1] / "lib"),
                            str(targets / "lib"), "/opt/ros/noetic/lib"),
        "CPATH": str(targets / "include"),
        "LIBRARY_PATH": str(targets / "lib"),
        "MAX_JOBS": "1",
    }
    for key, expected in fields.items():
        value = os.environ.get(key, "")
        if isinstance(expected, tuple):
            if not all(part in value.split(":") for part in expected):
                raise RuntimeError(f"Missing {key} entry: {expected}")
        elif key in ("PATH", "CPATH", "LIBRARY_PATH"):
            if expected not in value.split(":"):
                raise RuntimeError(f"Missing {key} entry: {expected}")
        elif value != expected:
            raise RuntimeError(f"Unexpected {key}: {value}")


def wait_for_smoke(smoke_root: Path, poll_seconds: int, timeout_seconds: int) -> Path:
    manifest_path = smoke_root / "manifest.json"
    manifest = read_json(manifest_path)
    verify_inputs(manifest, manifest_path)
    if manifest["n_configs"] != 54:
        raise RuntimeError("Smoke manifest must contain 54 configurations")
    expected_names = {row["name"] for row in manifest["candidates"]}
    started = time.monotonic()
    last_count = -1
    while True:
        status_path = smoke_root / "sweep_status.json"
        if status_path.is_file():
            status = read_json(status_path)
            if status.get("manifest_sha256") != digest(manifest_path):
                raise RuntimeError("Smoke checkpoint belongs to another manifest")
            outcomes = status.get("outcomes", {})
            if set(outcomes) - expected_names:
                raise RuntimeError("Smoke checkpoint contains unknown candidates")
            abnormal = {name: row["state"] for name, row in outcomes.items()
                        if row["state"] not in SMOKE_TERMINAL}
            if abnormal:
                raise RuntimeError(f"Smoke has abnormal completed outcomes: {abnormal}")
            if len(outcomes) != last_count:
                print(f"[WAIT] audited smoke outcomes {len(outcomes)}/54", flush=True)
                last_count = len(outcomes)
            if set(outcomes) == expected_names:
                validate_complete_smoke(smoke_root)
                print("[OK] all 54 smoke outcomes complete and manifest verified", flush=True)
                return manifest_path
        if timeout_seconds and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"Smoke did not finish within {timeout_seconds} s: {smoke_root}")
        time.sleep(poll_seconds)


def verify_smoke_reports(smoke_root: Path, manifest_path: Path) -> None:
    saved = read_json(smoke_root / "final_comparison.json")
    current = aggregate(manifest_path)
    if (saved["manifest_sha256"] != digest(manifest_path) or
            saved["source_sha256"] != current["source_sha256"] or
            saved["n_candidates"] != current["n_candidates"] or
            current["n_candidates"] != 54):
        raise RuntimeError("Published smoke comparison source or counts differ")
    fields = ("name", "status", "n_success", "n_total", "config_sha256", "ranking_eligible")
    for old, new in zip(saved["candidates"], current["candidates"]):
        if any(old.get(key) != new.get(key) for key in fields):
            raise RuntimeError(f"Published smoke candidate changed: {new['name']}")
    for filename in ("final_comparison.csv", "final_joint_margin.csv",
                     "final_observed_posture_cases.csv"):
        with (smoke_root / filename).open(newline="", encoding="utf-8") as stream:
            if len(list(csv.DictReader(stream))) != 54:
                raise RuntimeError(f"Expected 54 report rows: {filename}")
    best = read_json(smoke_root / "best_angle_3d/best_angle_by_position.json")
    expected_sources = {
        path.name: digest(path) for path in (
            manifest_path, smoke_root / "final_comparison.csv",
            smoke_root / "final_joint_margin.csv",
            smoke_root / "final_observed_posture_cases.csv")
    }
    if (best["source_sha256"] != expected_sources or best["n_positions"] != 18 or
            best["plotly_js_sha256"] != digest(smoke_root / "best_angle_3d/plotly.min.js")):
        raise RuntimeError("3D smoke report no longer matches its sources")
    verify_smoke_walkthrough(smoke_root)
    print("[REUSE] V67 smoke reports and 18-position video verified", flush=True)


def verify_smoke_walkthrough(smoke_root: Path) -> None:
    best = smoke_root / "best_angle_3d"
    video_dir = best / "video"
    for filename in BEST_VIDEO_FILES:
        path = video_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty V67 walkthrough artifact: {path}")
    for index in range(1, 19):
        still = video_dir / "stills" / f"point_{index:02d}.png"
        if not still.is_file() or still.stat().st_size == 0:
            raise RuntimeError(f"Missing V67 walkthrough detail still: {still}")
    source = best / "best_angle_by_position.json"
    manifest = read_json(video_dir / "frame_manifest.json")
    audit = read_json(video_dir / "video_audit.json")
    video = video_dir / VIDEO_NAME
    segments = manifest.get("segments", [])
    details = [part for part in segments if part.get("kind") == "detail"]
    expected_frames = manifest.get("expected_frames")
    if (Path(manifest.get("data_file", "")).resolve() != source or
            manifest.get("data_sha256") != digest(source) or
            manifest.get("v67_renderer_sha256") != digest(SCRIPTS / "render_v67_negative_best_angle_walkthrough.py") or
            manifest.get("v64_renderer_base_sha256") != digest(LEGACY_RENDERER) or
            manifest.get("video_file") != VIDEO_NAME or
            manifest.get("n_individual_detail_segments") != 18 or
            not isinstance(expected_frames, int) or expected_frames <= 0 or
            len(details) != 18 or
            [part.get("position_number") for part in details] != list(range(1, 19)) or
            any(part.get("end_frame_exclusive", 0) <= part.get("start_frame", -1)
                for part in segments) or
            any(right.get("start_frame") != left.get("end_frame_exclusive")
                for left, right in zip(segments, segments[1:])) or
            not segments or segments[0].get("start_frame") != 0 or
            segments[-1].get("end_frame_exclusive") != expected_frames):
        raise RuntimeError("V67 walkthrough frame manifest or source hashes differ")
    checks = audit.get("point_midframe_checks", [])
    probe = audit.get("probe", {})
    if (audit.get("source_data_sha256") != manifest["data_sha256"] or
            audit.get("video_sha256") != digest(video) or
            audit.get("full_decode") != "passed" or
            audit.get("n_individual_positions_verified") != 18 or
            len(checks) != 18 or
            any(check.get("position_number") != part["position_number"] or
                check.get("case") != part.get("case") or
                check.get("success") is not True
                for check, part in zip(checks, details)) or
            int(probe.get("nb_frames", -1)) != expected_frames or
            probe.get("codec_name") != "h264" or
            int(probe.get("width", -1)) != manifest.get("width") or
            int(probe.get("height", -1)) != manifest.get("height") or
            audit.get("no_grid_result_positions") !=
            read_json(source)["n_no_grid_result_positions"]):
        raise RuntimeError("V67 walkthrough video audit or video SHA256 differs")


def ensure_smoke_reports(smoke_root: Path, manifest_path: Path) -> None:
    stages = (
        (("final_comparison.json", "final_comparison.csv", "final_comparison.png"),
         (sys.executable, SCRIPTS / "summarize_local_y_mount_sweep.py",
          "--manifest", manifest_path, "--out", smoke_root / "final_comparison.json",
          "--csv", smoke_root / "final_comparison.csv",
          "--plot", smoke_root / "final_comparison.png")),
        (("final_joint_margin.csv", "final_joint_margin.png"),
         (sys.executable, SCRIPTS / "plot_local_y_sweep_joint_margin.py",
          "--comparison", smoke_root / "final_comparison.json",
          "--out", smoke_root / "final_joint_margin.png",
          "--csv", smoke_root / "final_joint_margin.csv")),
        (("final_observed_posture.json", "final_observed_posture_cases.csv",
          "final_observed_posture_grasps.csv", "final_observed_posture.png"),
         (sys.executable, SCRIPTS / "analyze_grasp_posture.py",
          "--manifest", manifest_path,
          "--output-prefix", smoke_root / "final_observed_posture")),
    )
    all_paths = [smoke_root / name for name in OUTPUT_NAMES]
    all_paths += [smoke_root / "best_angle_3d" / name for name in BEST_3D_FILES]
    all_paths += [smoke_root / "best_angle_3d/video" / name for name in BEST_VIDEO_FILES]
    if not any(path.exists() for path in all_paths) and not (smoke_root / "best_angle_3d").exists():
        call(sys.executable, SCRIPTS / "postprocess_v67_negative_smoke.py",
             "--smoke-root", smoke_root)
    else:
        for filenames, command in stages:
            paths = [smoke_root / filename for filename in filenames]
            if not any(path.exists() for path in paths):
                call(*command)
            elif not all(path.is_file() and path.stat().st_size > 0 for path in paths):
                raise RuntimeError(f"Partial smoke report stage; inspect before retrying: {paths}")
        best = smoke_root / "best_angle_3d"
        if not best.exists():
            call(sys.executable, SCRIPTS / "plot_v67_negative_best_angle_3d.py",
                 "--smoke-root", smoke_root)
        elif not all((best / name).is_file() and (best / name).stat().st_size > 0
                     for name in BEST_3D_FILES):
            raise RuntimeError(f"Partial 3D smoke report; inspect before retrying: {best}")
        walkthrough = best / "video"
        if not walkthrough.exists():
            call(sys.executable, SCRIPTS / "render_v67_negative_best_angle_walkthrough.py",
                 "--smoke-root", smoke_root)
        elif not all((walkthrough / name).is_file() and
                     (walkthrough / name).stat().st_size > 0
                     for name in BEST_VIDEO_FILES):
            raise RuntimeError(f"Partial V67 walkthrough; inspect before retrying: {walkthrough}")
    for path in all_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty smoke report: {path}")
    verify_smoke_reports(smoke_root, manifest_path)


def ensure_full_manifest(smoke_root: Path, full_root: Path) -> tuple[Path, list[str]]:
    manifest_path = full_root / "manifest.json"
    if not full_root.exists():
        call(sys.executable, SCRIPTS / "prepare_negative_local_y_top5_full.py",
             "--smoke-root", smoke_root, "--out-root", full_root)
    elif not manifest_path.is_file():
        raise RuntimeError(f"Existing full root has no manifest; inspect before retrying: {full_root}")
    recorded = read_json(manifest_path)
    verify_inputs(recorded, manifest_path)
    smoke_source = recorded["smoke_source"]
    for key, path in (("manifest_sha256", smoke_root / "manifest.json"),
                      ("sweep_status_sha256", smoke_root / "sweep_status.json")):
        if smoke_source[key] != digest(path):
            raise RuntimeError(f"Top 5 selection source changed: {key}")
    if Path(smoke_source["root"]).resolve() != smoke_root:
        raise RuntimeError("Top 5 manifest references another smoke experiment")
    planned, expected = select_and_build(smoke_root, full_root, DEFAULT_REFERENCE.resolve())
    if (recorded["n_configs"] != 5 or recorded["parameters"] != expected["parameters"] or
            len(recorded["candidates"]) != len(planned)):
        raise RuntimeError("Top 5 manifest differs from current audited selection")
    names = []
    for (configuration, expected_row), actual in zip(planned, recorded["candidates"]):
        name = expected_row["name"]
        if (any(actual.get(key) != value for key, value in expected_row.items()) or
                read_json(Path(actual["config"])) != configuration or
                digest(Path(actual["config"])) != actual["config_sha256"]):
            raise RuntimeError(f"Top 5 full configuration or origin changed: {name}")
        names.append(name)
    print(f"[OK] Top 5 manifest frozen: {names}", flush=True)
    return manifest_path, names


def verify_full_results(full_root: Path, manifest_path: Path, names: list[str]) -> bool:
    status_path = full_root / "sweep_status.json"
    if not status_path.is_file():
        return False
    status = read_json(status_path)
    if status["manifest_sha256"] != digest(manifest_path):
        raise RuntimeError("Full-run checkpoint belongs to another manifest")
    outcomes = status["outcomes"]
    if set(outcomes) - set(names):
        raise RuntimeError("Full-run checkpoint has unknown candidate names")
    if set(outcomes) != set(names) or any(row["state"] != "completed_verified" for row in outcomes.values()):
        return False
    for name in names:
        row, _ = inspect_candidate(full_root / "runs" / name)
        if (row["status"] != "completed_verified" or row["n_success"] != outcomes[name]["n_success"] or
                row["n_total"] != outcomes[name]["n_total"] or
                outcomes[name]["n_total"] != 400):
            raise RuntimeError(f"Full-run checkpoint and independent audit differ: {name}")
    print("[REUSE] five independently audited 20x20 planning runs", flush=True)
    return True


def ensure_full_results(full_root: Path, manifest_path: Path, names: list[str]) -> None:
    if not verify_full_results(full_root, manifest_path, names):
        call(sys.executable, SCRIPTS / "run_local_y_mount_sweep.py",
             "--manifest", manifest_path)
        if not verify_full_results(full_root, manifest_path, names):
            raise RuntimeError("Full planning finished without five completed_verified 20x20 runs")


def ensure_reports(full_root: Path, names: list[str]) -> None:
    for name in names:
        call(sys.executable, SCRIPTS / "postprocess_full_mount_candidates.py",
             full_root, "--names", name)
    call(sys.executable, SCRIPTS / "postprocess_full_mount_candidates.py",
         full_root, "--compare")


def ensure_rviz_commands(full_root: Path) -> None:
    path = full_root / "RVIZ_COMMANDS.md"
    script = SCRIPTS / "write_full_mount_rviz_commands.py"
    if not path.exists():
        call(sys.executable, script, full_root)
    else:
        with tempfile.TemporaryDirectory(prefix=".rviz_commands_check_", dir=full_root) as scratch:
            expected = Path(scratch) / "RVIZ_COMMANDS.md"
            call(sys.executable, script, full_root, "--out", expected)
            if path.read_bytes() != expected.read_bytes():
                raise RuntimeError("Existing RVIZ_COMMANDS.md differs from the frozen manifest")
        print("[REUSE] five RViz commands verified", flush=True)


def clean_video_scan(audit: dict) -> bool:
    return all(not side.get("black_scene_frame_indices") and
               not side.get("occluded_scene_frame_indices")
               for side in audit["scans"].values())


def verify_video_bundle(full_root: Path, name: str) -> None:
    run = full_root / "runs" / name
    folder = full_root / "videos" / name
    full = folder / "full_40x"
    extreme = folder / "joint_span_extremes"
    meta = read_json(run / "trajectory_meta.json")
    npz_sha, meta_sha = digest(run / "trajectory.npz"), digest(run / "trajectory_meta.json")
    first_success = next(int(item["index"]) + 1 for item in meta["items"] if item["success"])
    video = full / f"{name}_complete_40x.mp4"
    render = read_json(full / "video_render_audit.json")
    decode = read_json(full / "video_decode_audit.json")
    boundary = read_json(full / "boundary_review.json")
    if (Path(render["source_run"]).resolve() != run or
            render["source_npz_sha256"] != npz_sha or render["source_meta_sha256"] != meta_sha or
            render["video_sha256"] != digest(video) or decode["sha256"] != render["video_sha256"] or
            decode["mode"] != "full" or decode["source_coverage"]["successful_cases"] != meta["n_items_success"] or
            decode["source_coverage"]["source_sha256_match"] is not True or
            not clean_video_scan(decode) or boundary["video_sha256_match"] is not True or
            not (full / "boundary_review.png").is_file()):
        raise RuntimeError(f"Full video audit or saved source differs: {name}")
    extraction = read_json(extreme / "manifest.json")
    source = extraction["source"]
    if (Path(source["result"]).resolve() != run or
            source["sha256"]["trajectory.npz"] != npz_sha or
            source["sha256"]["trajectory_meta.json"] != meta_sha or
            extraction["excluded_original_case_numbers_1based"] != [first_success]):
        raise RuntimeError(f"Extreme selection source or excluded first case differs: {name}")
    selected = {item["label"]: item for item in extraction["selected"]}
    if set(selected) != {"max", "min"}:
        raise RuntimeError(f"Extreme selection is incomplete: {name}")
    side = read_json(extreme / "side_by_side_manifest.json")
    side_video = Path(side["output"])
    side_audit = read_json(extreme / "side_by_side_video_audit.json")
    if (side_video.parent.resolve() != extreme or digest(side_video) != side["sha256"] or
            side_audit["sha256"] != side["sha256"] or side_audit["mode"] != "comparison" or
            not clean_video_scan(side_audit) or
            side["left"]["case"] != selected["max"]["original_case_number_1based"] or
            side["right"]["case"] != selected["min"]["original_case_number_1based"]):
        raise RuntimeError(f"Side-by-side video audit differs: {name}")
    for tier in ("max", "min"):
        case = selected[tier]["original_case_number_1based"]
        clip = extreme / f"{tier}_case_{case:03d}.mp4"
        clip_render = read_json(extreme / tier / "video_render_audit.json")
        clip_decode = read_json(extreme / f"{tier}_video_decode_audit.json")
        if (clip_render["video_sha256"] != digest(clip) or
                clip_render["source_npz_sha256"] != selected[tier]["output_sha256"]["trajectory.npz"] or
                clip_decode["sha256"] != clip_render["video_sha256"] or
                clip_decode["mode"] != "clip" or not clean_video_scan(clip_decode)):
            raise RuntimeError(f"{tier} clip audit differs: {name}")
    if not (folder / "README.md").is_file():
        raise RuntimeError(f"Video README missing: {name}")
    print(f"[REUSE] audited full and extreme videos: {name}", flush=True)


def system_video_environment() -> dict[str, str]:
    """Keep conda CUDA libraries out of the system ROS/RViz recorder process."""
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD",
                "LD_AUDIT", "CPATH", "LIBRARY_PATH", "CUDA_HOME",
                "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
                "QML2_IMPORT_PATH", "QT_QPA_FONTDIR"):
        env.pop(key, None)
    env["PATH"] = (str(SCRIPTS / "system_media_bin") +
                   ":/opt/ros/noetic/bin:/usr/local/sbin:/usr/local/bin:"
                   "/usr/sbin:/usr/bin:/sbin:/bin")
    env["PYTHONNOUSERSITE"] = "1"
    return env


def ensure_videos(full_root: Path, names: list[str]) -> None:
    for name in names:
        folder = full_root / "videos" / name
        if not folder.exists():
            call("/bin/bash", SCRIPTS / "run_full_mount_video_pipeline.sh",
                 full_root, name, env=system_video_environment())
        verify_video_bundle(full_root, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke_root", type=Path)
    parser.add_argument("full_root", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait-timeout-seconds", type=int, default=43200,
                        help="0 means wait without a deadline")
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.wait_timeout_seconds < 0:
        parser.error("Poll seconds must be positive and timeout nonnegative")
    smoke_root, full_root = args.smoke_root.resolve(), args.full_root.resolve()
    if smoke_root == full_root or smoke_root in full_root.parents:
        parser.error("FULL_ROOT must be separate from SMOKE_ROOT")
    check_environment()
    full_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = full_root.parent / f".{full_root.name}.pipeline.lock"
    with lock_path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = wait_for_smoke(smoke_root, args.poll_seconds, args.wait_timeout_seconds)
        ensure_smoke_reports(smoke_root, manifest)
        full_manifest, names = ensure_full_manifest(smoke_root, full_root)
        ensure_full_results(full_root, full_manifest, names)
        ensure_reports(full_root, names)
        ensure_rviz_commands(full_root)
        ensure_videos(full_root, names)
    print(f"[COMPLETE] V67 negative smoke Top 5 full pipeline: {full_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
