#!/usr/bin/env python3
"""Run a bounded list of isolated IK screens or full pick/place experiments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["ik", "plan"], required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--perimeter", action="store_true")
    args = ap.parse_args()
    scripts = Path(__file__).resolve().parent
    root = Path(args.out_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        config = Path(config).resolve()
        out = root / config.stem
        out.mkdir(exist_ok=True)
        if (out / "run_status.json").exists() or (out / "plan.log").exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {out}")
        script = "scan_overhead_ik.py" if args.mode == "ik" else "plan_pick_place.py"
        command = [sys.executable, str(scripts / script), "--config", str(config), "--out-dir", str(out)]
        if args.mode == "plan":
            command += ["--no-timestamp", "--no-incremental-save"]
            if args.perimeter:
                command += ["--perimeter", "--order", "ring"]
        for flag in ("rows", "cols"):
            if getattr(args, flag) is not None:
                command += ["--" + flag, str(getattr(args, flag))]
        started = time.time()
        print(f"[RUN] {args.mode} {config.stem} -> {out}", flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with (out / "plan.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=scripts.parent, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = proc.wait()
        status = {"command": command, "returncode": code,
                  "elapsed_s": time.time() - started, "config": str(config)}
        (out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(f"[DONE] {config.stem} code={code} elapsed={status['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
