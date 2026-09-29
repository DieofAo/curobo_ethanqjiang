#!/usr/bin/env python3
"""Sequential, resumable execution of a recorded overhead smoke manifest.

The status JSON is atomically checkpointed after each candidate.  Completed
planning runs are reused only when their run_status points to the exact hashed
config in the manifest.  Partial run directories are retained and reported;
they are never overwritten.  Use --names for a small pilot, then omit it to
continue the remaining candidates.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from summarize_overhead_cartesian import inspect_candidate


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".sweep_status_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_inputs(manifest, manifest_path):
    root = manifest_path.parent
    if len(manifest["candidates"]) != manifest["n_configs"]:
        raise ValueError("Manifest candidate count differs")
    if sha256(Path(manifest["source"])) != manifest["source_sha256"]:
        raise ValueError("Source config hash differs from manifest")
    for label, entry in manifest["runtime_inputs"].items():
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Runtime input changed since recording: {label}: {path}")
    names = set()
    if manifest.get("parameters", {}).get("tilt_axis") != "original_base_local_y":
        raise ValueError("Expected original_base_local_y sweep manifest")
    for index, row in enumerate(manifest["candidates"]):
        if row.get("tilt_axis") != "original_base_local_y":
            raise ValueError("Expected original_base_local_y candidate")
        name = row["name"]
        config, result = Path(row["config"]), Path(row["result"])
        if (name in names or row["index"] != index or
                config != root / "configs" / f"{name}.json" or result != root / "runs" / name):
            raise ValueError(f"Bad manifest row {index}")
        if sha256(config) != row["config_sha256"]:
            raise ValueError(f"Candidate config changed: {config}")
        names.add(name)


def read_completed_status(result, config):
    path = result / "run_status.json"
    if not path.exists():
        return None
    status = json.loads(path.read_text(encoding="utf-8"))
    if Path(status["config"]).resolve() != config:
        raise ValueError(f"Completed run config does not match manifest: {result}")
    command = status.get("command", [])
    if "--config" not in command or Path(command[command.index("--config") + 1]).resolve() != config:
        raise ValueError(f"Completed run command does not match manifest: {result}")
    return status


def run_audits(result, scripts, log):
    tasks = [
        ("summarize_overhead_plan.py", result / "analysis_summary.json",
         ["--result", str(result), "--out", str(result / "analysis_summary.json")]),
        ("analyze_link3_grasp_clearance.py", result / "link3_grasp_clearance.json",
         ["--result", str(result), "--out", str(result / "link3_grasp_clearance.json")]),
        ("verify_overhead_trajectory.py", result / "independent_verification.json", [str(result)]),
        ("verify_joint_limit_clip.py", result / "joint_limit_clip_audit.json",
         [str(result), "--out", str(result / "joint_limit_clip_audit.json")]),
    ]
    outcomes = []
    for script, output, args in tasks:
        if output.exists():
            outcomes.append({"script": script, "status": "existing", "returncode": 0})
            continue
        command = [sys.executable, str(scripts / script), *args]
        log.write(f"[AUDIT] {command}\n")
        log.flush()
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        log.flush()
        outcomes.append({"script": script, "status": "ran", "returncode": code})
    return outcomes


def execute_candidate(row, root, scripts, orchestration_log):
    config, result = Path(row["config"]), Path(row["result"])
    status = read_completed_status(result, config)
    if status is None:
        if result.exists():
            return {"name": row["name"], "state": "interrupted_partial", "result": str(result),
                    "note": "Result directory exists without run_status.json; preserved for inspection"}
        command = [sys.executable, str(scripts / "run_overhead_batch.py"),
                   "--mode", "plan", "--configs", str(config), "--out-root", str(root / "runs")]
        print(f"[PLAN] {row['name']} base={row['base_xyz_m']} local base +Y={row['local_y_tilt_deg']}°", flush=True)
        orchestration_log.write(f"[PLAN] {command}\n")
        orchestration_log.flush()
        outer_code = subprocess.run(command, stdout=orchestration_log,
                                    stderr=subprocess.STDOUT).returncode
        orchestration_log.flush()
        status = read_completed_status(result, config)
        if status is None:
            return {"name": row["name"], "state": "interrupted_partial", "result": str(result),
                    "batch_returncode": outer_code,
                    "note": "Batch exited without a completed run_status.json; result preserved"}
    outcome = {"name": row["name"], "result": str(result), "state": "plan_failed",
               "config_sha256": row["config_sha256"],
               "plan_returncode": int(status["returncode"]),
               "plan_elapsed_s": status.get("elapsed_s"), "postprocessing": []}
    if status["returncode"] == 0 and (result / "trajectory_meta.json").exists():
        meta = json.loads((result / "trajectory_meta.json").read_text(encoding="utf-8"))
        outcome.update(n_success=meta["n_items_success"], n_total=meta["n_items_total"])
        with (result / "postprocessing.log").open("a", encoding="utf-8") as log:
            outcome["postprocessing"] = run_audits(result, scripts, log)
        outcome["state"] = ("audit_outputs_present" if all(
            record["returncode"] == 0 for record in outcome["postprocessing"]) else "audit_script_failed")
    else:
        failure = result / "plan_failed.json"
        outcome["failure_stage"] = (json.loads(failure.read_text(encoding="utf-8")).get("stage")
                                    if failure.exists() else None)
    inspected, _ = inspect_candidate(result)
    outcome["state"] = inspected["status"]
    outcome["ranking_eligible"] = inspected["ranking_eligible"]
    outcome["n_success"] = inspected["n_success"]
    outcome["n_total"] = inspected["n_total"]
    outcome["issues"] = inspected["issues"]
    print(f"[DONE] {row['name']}: {outcome['state']} "
          f"{outcome.get('n_success')}/{outcome.get('n_total')}", flush=True)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--names", nargs="+", help="Only run named candidates; omit to resume all")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_inputs(manifest, manifest_path)
    all_rows = manifest["candidates"]
    by_name = {row["name"]: row for row in all_rows}
    if args.names:
        if len(set(args.names)) != len(args.names) or set(args.names) - set(by_name):
            parser.error("--names must be unique names present in the manifest")
        selected = [by_name[name] for name in args.names]
    else:
        selected = all_rows
    root = manifest_path.parent
    status_path = root / "sweep_status.json"
    scripts = Path(__file__).resolve().parent
    with (root / "sweep_runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            state = json.loads(status_path.read_text(encoding="utf-8"))
            if state["manifest_sha256"] != sha256(manifest_path):
                raise ValueError("Existing sweep status belongs to a different manifest")
        else:
            state = {"schema_version": 1, "manifest": str(manifest_path),
                     "manifest_sha256": sha256(manifest_path), "started_at": now(),
                     "outcomes": {}}
        with (root / "orchestration.log").open("a", encoding="utf-8") as log:
            for row in selected:
                verify_inputs(manifest, manifest_path)
                outcome = execute_candidate(row, root, scripts, log)
                verify_inputs(manifest, manifest_path)
                state["outcomes"][row["name"]] = outcome
                state["updated_at"] = now()
                atomic_json(status_path, state)
        print(f"[STATUS] {status_path}: {len(state['outcomes'])}/{len(all_rows)} recorded", flush=True)
        return int(any(state["outcomes"][row["name"]]["state"] in
                       ("interrupted_partial", "audit_script_failed", "audit_failed", "audit_missing", "config_mismatch") for row in selected))


if __name__ == "__main__":
    raise SystemExit(main())
