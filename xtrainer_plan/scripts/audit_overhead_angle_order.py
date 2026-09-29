#!/usr/bin/env python3
"""Independently audit an explicitly selected ascending/descending angle policy.

CPU-only: reads completed planning logs, recorded configs and item metadata;
exclusively writes a separate report. This is NOT a trajectory safety audit.
Zero-success runs can pass the search-policy audit, but Home failures, pending
runs, missing metadata and incomplete grids cannot pass. No CuRobo/ROS imports.
Default: +30,+28,...,-30 (desc); asc requires -30,-28,...,+30 instead.
"""
import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


ANGLES = list(range(30, -31, -2))
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
CASE = re.compile(r"^--- 物料 (\d+)/(\d+)\s+抓取点\s+(\[[^\n]+\])\s+---\s*$", re.MULTILINE)
ATTEMPT = re.compile(rf"^[ \t]*g=({NUMBER})[ \t]+p=({NUMBER})deg", re.MULTILINE)
FAILURE = re.compile(r"->\s*规划失败:\s*([^\n]+?)\s+@\s+i(\d+)_\S+")
SUCCESS = re.compile(r"->\s*OK\s+\d+\s+点")
COMPLETION = re.compile(r"^\[PLAN\] 完成 (\d+)/(\d+) 个物料", re.MULTILINE)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_equal(actual, expected):
    return isinstance(actual, (int, float)) and math.isfinite(actual) and abs(actual - expected) < 1e-9


def expected_angles(expected_order="desc"):
    require(expected_order in ("asc", "desc"), "expected_order must be asc or desc")
    return ANGLES[:] if expected_order == "desc" else ANGLES[::-1]


def validate_policy(cfg, expected_order="desc"):
    angles = expected_angles(expected_order)
    pp = cfg["pick_place"]
    policy = pp["angle_search"]
    grasp = policy["grasp"]
    for key, expected in (("min_deg", -30), ("max_deg", 30), ("step_deg", 2)):
        require(finite_equal(grasp[key], expected), f"Unexpected grasp.{key}: {grasp[key]}")
    require(grasp["axis"] == "x" and policy["place"]["axis"] == "x", "Both search axes must be tool x")
    require(policy.get("order") == expected_order, f"angle_search.order must be {expected_order}")
    require(policy.get("couple_place_to_grasp") is True, "place must remain coupled to grasp")
    require(policy.get("reuse_last_success", True) is False, "reuse_last_success must be false")
    require((policy.get("stage2") or {}).get("enable", False) is False, "stage2 must remain disabled")
    cap = policy.get("max_trials")
    require(cap is None or cap == 0 or
            (isinstance(cap, int) and not isinstance(cap, bool) and cap >= len(angles)),
            "max_trials truncates or invalidates the 31 candidates")
    require(pp["criterion"].get("prescreen_by_ik", True) is False, "Independent IK prescreen must remain disabled")
    require(pp["on_fail"]["mode"] == "skip", "Full-grid audit requires on_fail=skip")
    grid = pp["grasp_grid"]
    require(grid.get("perimeter_only", False) is False, "This audit requires a full, non-perimeter grid")
    require(all(isinstance(grid[key], int) and not isinstance(grid[key], bool) and grid[key] > 0
                for key in ("rows", "cols")), "Invalid grid dimensions")
    return policy, grid["rows"] * grid["cols"]


def parse_cases(log, n_total, expected_order="desc"):
    """Parse candidate blocks, tolerating solver chatter adjacent to g= lines."""
    angles = expected_angles(expected_order)
    grasp = re.findall(rf"^\[ANGLE\] 抓取侧: 绕工具 (\S+) 轴 \[({NUMBER}),\s*({NUMBER})\] "
                       rf"deg, 步长 ({NUMBER}) -> (\d+) 个候选", log, re.MULTILINE)
    require(len(grasp) == 1 and grasp[0][0] == "X" and
            [float(v) for v in grasp[0][1:4]] == [-30, 30, 2] and int(grasp[0][4]) == len(angles),
            "Log grasp-range header does not confirm 31 angles from -30 to +30")
    coupled = re.findall(r"^\[ANGLE\] 两侧耦合, 候选 = (\d+) 组 \(g, g\), "
                         r"实际尝试上限 (\d+) 组 \(order=([^\)]+)\)", log, re.MULTILINE)
    require(coupled == [("31", "31", expected_order)],
            f"Log does not confirm untruncated {expected_order} p=g coupling")
    require("[STAGE2] 未开启" in log and "[S2]" not in log, "Log stage2 policy differs or stage2 attempts occurred")
    require("复用上次成功组合" not in log, "Log contains reuse of a previous successful angle")
    headers = list(CASE.finditer(log))
    require(len(headers) == n_total, f"Log contains {len(headers)}/{n_total} case headers")
    require(sorted(int(h[1]) - 1 for h in headers) == list(range(n_total)), "Case indices are duplicated or missing")
    require(all(int(h[2]) == n_total for h in headers), "Log case denominators differ from configured grid")
    require(not ATTEMPT.search(log[:headers[0].start()]), "Candidate attempt outside a case block")
    newline_positions = [match.start() for match in re.finditer("\n", log)]
    rows = []
    for offset, header in enumerate(headers):
        index = int(header[1]) - 1
        end = headers[offset + 1].start() if offset + 1 < len(headers) else len(log)
        block = log[header.end():end]
        attempts = list(ATTEMPT.finditer(block))
        require(1 <= len(attempts) <= len(angles), f"Case {index}: expected 1..31 candidate attempts")
        pairs = [[float(attempt[1]), float(attempt[2])] for attempt in attempts]
        require(pairs == [[angle, angle] for angle in angles[:len(pairs)]],
                f"Case {index}: candidates are not the exact {expected_order} p=g prefix "
                f"starting at {angles[0]:+d}")
        outcomes, line_numbers = [], []
        for j, attempt in enumerate(attempts):
            stop = attempts[j + 1].start() if j + 1 < len(attempts) else len(block)
            chunk = block[attempt.end():stop]
            failed, succeeded = list(FAILURE.finditer(chunk)), list(SUCCESS.finditer(chunk))
            require(len(failed) + len(succeeded) == 1, f"Case {index}, angle {pairs[j][0]}: missing or ambiguous outcome")
            if failed:
                require(int(failed[0][2]) == index, f"Case {index}: failure refers to a different item")
            outcomes.append("success" if succeeded else "failure")
            line_numbers.append(bisect_right(newline_positions, header.end() + attempt.start()) + 1)
        success = outcomes[-1] == "success"
        require(all(outcome == "failure" for outcome in outcomes[:-1]),
                f"Case {index}: search continued after a successful attempt")
        declared_failure = re.findall(r"\[FAIL\] 物料 (\d+) 在 (\d+) 组候选", block)
        if success:
            require(not declared_failure, f"Case {index}: successful case also declared failed")
        else:
            require(len(attempts) == len(angles), f"Case {index}: failure before all 31 candidates")
            require(declared_failure == [(str(index + 1), "31")],
                    f"Case {index}: missing/inconsistent final failure declaration")
        point = json.loads(header[3])
        require(isinstance(point, list) and len(point) == 3 and
                all(isinstance(value, (int, float)) and math.isfinite(value) for value in point),
                f"Case {index}: invalid logged grasp coordinates")
        rows.append({"index": index, "position_raw_rounded_in_log": point, "success": success,
                     "n_angles_tried": len(attempts), "candidate_angles_grasp_place_deg": pairs,
                     "candidate_log_line_numbers": line_numbers,
                     "selected_angle_grasp_deg": pairs[-1][0] if success else None,
                     "selected_angle_place_deg": pairs[-1][1] if success else None})
    completion = COMPLETION.findall(log)
    require(completion == [(str(sum(row["success"] for row in rows)), str(n_total))],
            "Missing/inconsistent final full-grid completion summary")
    return rows


def check_items(meta, cases, n_total, policy, expected_order="desc"):
    require(not meta.get("partial"), "Saved trajectory metadata is partial")
    require(meta["config"]["pick_place"]["angle_search"] == policy and meta["angle_search"] == policy,
            "Config, meta.config.angle_search and meta.angle_search differ")
    _, meta_total = validate_policy(meta["config"], expected_order)
    require(meta_total == n_total, "Metadata and source config grid counts differ")
    items = meta["items"]
    require(len(items) == n_total and meta["n_items_total"] == n_total, "Incomplete metadata item list")
    require([item["index"] for item in items] == [row["index"] for row in cases],
            "Metadata item indices/order differ from log")
    for item, row in zip(items, cases):
        index = row["index"]
        require(item["success"] == row["success"] and item["n_angles_tried"] == row["n_angles_tried"],
                f"Case {index}: metadata outcome/attempt count differs from log")
        raw = item["position_raw"]
        require(len(raw) == 3 and all(math.isfinite(v) and abs(v - logged) <= .0000500001
                                     for v, logged in zip(raw, row["position_raw_rounded_in_log"])),
                f"Case {index}: metadata position differs from four-decimal log coordinates")
        if row["success"]:
            require(all(finite_equal(item[key], row[key.replace("angle_", "selected_angle_", 1)])
                        for key in ("angle_grasp_deg", "angle_place_deg")),
                    f"Case {index}: saved successful angles differ from the final log attempt")
    require(meta["n_items_success"] == sum(row["success"] for row in cases), "Metadata success count differs from log")


def audit_result(result, config_path=None, config_sha256=None, expected_order="desc"):
    result = Path(result).resolve()
    row = {"result": str(result), "passed": False, "verification_completed": False,
           "status": "pending", "issues": [], "source": {"files": {}, "sha256": {}},
           "n_total": None, "n_success": None, "n_angle_attempts": None, "cases": []}
    snapshots = {}

    def read(path, role, json_data=True):
        path = Path(path).resolve()
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        row["source"]["files"][role] = str(path)
        row["source"]["sha256"][str(path)] = digest
        snapshots[path] = digest
        return json.loads(blob) if json_data else blob.decode("utf-8", errors="replace")

    try:
        angles = expected_angles(expected_order)
        status_path = result / "run_status.json"
        status = read(status_path, "run_status") if status_path.exists() else None
        if status is not None:
            recorded_config = Path(status["config"]).resolve()
            require(config_path is None or recorded_config == Path(config_path).resolve(),
                    "run_status config path differs from manifest")
            config_path = recorded_config
        if config_path is not None:
            cfg = read(config_path, "config")
            if config_sha256 is not None:
                require(snapshots[Path(config_path).resolve()] == config_sha256,
                        "Config hash differs from the recorded manifest")
            policy, n_total = validate_policy(cfg, expected_order)
            row["angle_search"] = policy
            row["configured_n_total"] = n_total
        if status is None:
            row["issues"].append("No run_status.json: planning completion not established")
            return row
        require(config_path is not None, "Missing recorded config source")
        row["run_returncode"] = status["returncode"]
        log = read(result / "plan.log", "plan_log", json_data=False)
        failed_path = result / "plan_failed.json"
        failed = read(failed_path, "plan_failed") if failed_path.exists() else None
        if failed and failed.get("stage") == "home_ik":
            row.update(status="home_failed", verification_completed=True)
            row["issues"].append("Home IK failed; the per-case search policy was not exercised")
            return row
        cases = parse_cases(log, n_total, expected_order)
        n_success = sum(case["success"] for case in cases)
        meta_path = result / "trajectory_meta.json"
        if n_success:
            require(status["returncode"] == 0, "Successful trajectory run did not exit normally")
            meta = read(meta_path, "trajectory_meta")
            check_items(meta, cases, n_total, policy, expected_order)
            row["status"] = "passed"
        else:
            require(status["returncode"] == 4 and failed is not None and
                    failed.get("stage") == "no_item_success", "Zero-success result lacks matching completion evidence")
            require(not meta_path.exists(), "Zero-success result contains unexpected/stale trajectory metadata")
            require(failed["config"]["pick_place"]["angle_search"] == policy,
                    "plan_failed config angle policy differs from source")
            _, failed_total = validate_policy(failed["config"], expected_order)
            require(failed_total == n_total, "plan_failed config grid count differs")
            detail = failed["failed"]
            require(detail["failed_item_index"] == cases[-1]["index"] and detail["n_combos_tried"] == len(angles),
                    "Zero-success final failure item/count differs from log")
            pairs = [[attempt["angle_grasp_deg"], attempt["angle_place_deg"]] for attempt in detail["candidates"]]
            require(pairs == [[angle, angle] for angle in angles], "plan_failed candidate sequence differs from log")
            row["status"] = "zero_success_policy_passed"
        for path, digest in snapshots.items():
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Source changed while auditing: {path}")
        row.update(passed=True, verification_completed=True, n_total=n_total, n_success=n_success,
                   n_angle_attempts=sum(case["n_angles_tried"] for case in cases), cases=cases)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        row.update(status="failed", verification_completed=True)
        row["issues"].append(str(exc))
    return row


def audit(results=None, manifest=None, expected_order="desc"):
    angles = expected_angles(expected_order)
    provenance = None
    if manifest is not None:
        manifest = Path(manifest).resolve()
        blob = manifest.read_bytes()
        recorded = json.loads(blob)
        candidates = recorded["candidates"]
        require(len(candidates) == recorded["n_configs"] and len(candidates) > 0, "Invalid manifest candidate count")
        require(len({row["name"] for row in candidates}) == len(candidates), "Duplicate manifest names")
        require(len({str(Path(row["result"]).resolve()) for row in candidates}) == len(candidates), "Duplicate manifest results")
        provenance = {"path": str(manifest), "sha256": hashlib.sha256(blob).hexdigest()}
        rows = []
        for candidate in candidates:
            row = audit_result(candidate["result"], candidate["config"], candidate["config_sha256"], expected_order)
            row["name"] = candidate["name"]
            rows.append(row)
        require(hashlib.sha256(manifest.read_bytes()).hexdigest() == provenance["sha256"], "Manifest changed while auditing")
    else:
        require(results is not None and len(results) > 0, "At least one result is required")
        require(len({str(Path(path).resolve()) for path in results}) == len(results), "Duplicate results")
        rows = [audit_result(result, expected_order=expected_order) for result in results]
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "CPU-only search-policy audit. Zero-success policy pass is NOT planning success. "
                     "No FK, collision, joint-limit, or trajectory-safety validation is performed.",
            "expected_angles_grasp_deg": angles, "expected_place_relation": "place=grasp",
            "expected_order": expected_order, "expected_reuse_last_success": False,
            "expected_stage2_enabled": False, "manifest_source": provenance,
            "passed": all(row["passed"] for row in rows),
            "verification_completed": all(row["verification_completed"] for row in rows),
            "n_results": len(rows), "n_passed": sum(row["passed"] for row in rows),
            "status_counts": dict(Counter(row["status"] for row in rows)), "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result", "--results", dest="results", type=Path, nargs="+")
    group.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-order", choices=("asc", "desc"), default="desc",
                        help="Required search order, selected independently of the recorded config (default: desc)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to replace existing report: {args.out}")
    report = audit(args.results, args.manifest, args.expected_order)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps({key: report[key] for key in
                     ("passed", "verification_completed", "n_results", "n_passed", "status_counts")}, indent=2))
    print(f"[OUT] {args.out.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
