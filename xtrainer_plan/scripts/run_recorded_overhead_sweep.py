#!/usr/bin/env python3
"""Plan and independently audit every config in a recorded Cartesian manifest.

Sequential GPU work avoids concurrent solver memory pressure. Child logs and
artifacts are retained. This runner does not change defaults or launch ROS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    scripts = Path(__file__).resolve().parent
    candidates = manifest["candidates"]
    if len(candidates) != manifest["n_configs"]:
        raise ValueError("Manifest candidate count differs")
    for row in candidates:
        config, result = Path(row["config"]), Path(row["result"])
        if hashlib.sha256(config.read_bytes()).hexdigest() != row["config_sha256"]:
            raise ValueError(f"Config changed after recording: {config}")
        if result.parent != root / "runs" or result.name != config.stem:
            raise ValueError("Result paths must be manifest-local runs/<config stem>")
        if result.exists():
            raise FileExistsError(f"Refusing to overwrite result: {result}")
    status_path = root / "sweep_status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    outcomes = []
    started = time.time()
    with (root / "orchestration.log").open("x") as log:
        for number, row in enumerate(candidates, 1):
            result = Path(row["result"])
            label = f"[{number}/{len(candidates)}] {row['name']}"
            print(f"{label} base={row['base_xyz_m']} place={row['place_xyz_m']}", flush=True)
            command = [sys.executable, str(scripts / "run_overhead_batch.py"),
                       "--mode", "plan", "--configs", row["config"],
                       "--out-root", str(root / "runs")]
            outer = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            log.flush()
            outcome = {"name": row["name"], "result": str(result),
                       "batch_returncode": outer.returncode, "postprocessing": []}
            status_file = result / "run_status.json"
            if not status_file.exists():
                outcome["error"] = "Batch did not record child completion"
            else:
                run_status = json.loads(status_file.read_text())
                outcome["plan_returncode"] = run_status["returncode"]
                outcome["plan_elapsed_s"] = run_status["elapsed_s"]
                meta_path = result / "trajectory_meta.json"
                if run_status["returncode"] == 0 and meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    outcome["success"] = meta["n_items_success"]
                    outcome["total"] = meta["n_items_total"]
                    tasks = [
                        ("summarize_overhead_plan.py", ["--result", str(result), "--out",
                                                       str(result / "analysis_summary.json")]),
                        ("analyze_link3_grasp_clearance.py", ["--result", str(result), "--out",
                                                            str(result / "link3_grasp_clearance.json")]),
                        ("verify_overhead_trajectory.py", [str(result)]),
                        ("verify_joint_limit_clip.py", [str(result), "--out",
                                                       str(result / "joint_limit_clip_audit.json")]),
                    ]
                    with (result / "postprocessing.log").open("x") as audit_log:
                        for name, extra in tasks:
                            code = subprocess.run([sys.executable, str(scripts / name), *extra],
                                                  stdout=audit_log, stderr=subprocess.STDOUT).returncode
                            audit_log.flush()
                            outcome["postprocessing"].append({"script": name, "returncode": code})
                    passed = all(r["returncode"] == 0 for r in outcome["postprocessing"])
                    print(f"{label} complete {outcome['success']}/{outcome['total']}; "
                          f"postprocessing_ok={passed}; plan_s={run_status['elapsed_s']:.1f}", flush=True)
                else:
                    failure = result / "plan_failed.json"
                    outcome["failure_stage"] = json.loads(failure.read_text()).get("stage") if failure.exists() else None
                    print(f"{label} plan code={run_status['returncode']} "
                          f"stage={outcome['failure_stage']}", flush=True)
            outcomes.append(outcome)
    with status_path.open("x") as stream:
        json.dump({"manifest": str(manifest_path), "elapsed_s": time.time() - started,
                   "outcomes": outcomes}, stream, indent=2)
        stream.write("\n")
    print(f"Sweep finished: {status_path}", flush=True)
    return int(any("error" in r or any(t["returncode"] != 0 for t in r["postprocessing"])
                   for r in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
