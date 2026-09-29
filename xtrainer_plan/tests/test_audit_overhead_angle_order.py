#!/usr/bin/env python3
"""Synthetic log/config tests; stdlib only and no GPU/ROS initialization."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from audit_overhead_angle_order import ANGLES, audit, audit_result  # noqa: E402


class AngleAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {"pick_place": {"angle_search": {
            "grasp": {"axis": "x", "min_deg": -30., "max_deg": 30., "step_deg": 2.},
            "place": {"axis": "x", "min_deg": -30., "max_deg": 30., "step_deg": 5.},
            "order": "desc", "couple_place_to_grasp": True, "reuse_last_success": False,
            "max_trials": 1200, "stage2": {"enable": False}},
            "criterion": {"prescreen_by_ik": False}, "on_fail": {"mode": "skip"},
            "grasp_grid": {"rows": 2, "cols": 2, "perimeter_only": False}}}

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    def make(self, name="run", successes=(2, 0, 1, 0), order="desc"):
        result = self.root / name
        result.mkdir()
        source = self.root / f"{name}.json"
        cfg = copy.deepcopy(self.cfg)
        cfg["pick_place"]["angle_search"]["order"] = order
        angles = list(range(-30, 31, 2)) if order == "asc" else ANGLES
        self.write(source, cfg)
        log = ["[ANGLE] 抓取侧: 绕工具 X 轴 [-30, 30] deg, 步长 2 -> 31 个候选",
               f"[ANGLE] 两侧耦合, 候选 = 31 组 (g, g), 实际尝试上限 31 组 (order={order})",
               "[STAGE2] 未开启 (仅做第一阶段搜索)"]
        items = []
        for index, successful_attempt in enumerate(successes):
            log.append(f"\n--- 物料 {index + 1}/4  抓取点 [-0.41, 0.15, 0.03] ---")
            count = successful_attempt or 31
            for n, angle in enumerate(angles[:count], 1):
                # The production logger uses end="", so solver messages may
                # interrupt a candidate line or put its outcome on a new line.
                log.append(f"    g={angle:+d} p={angle:+d}degCouldn't find solution, resetting seeds")
                if successful_attempt == n:
                    log.append("  -> OK  300 点  6.00s  最大变化 150deg")
                else:
                    log.append(f"  -> 规划失败: MotionGenStatus.IK_FAIL @ i{index}_place")
            if not successful_attempt:
                log.append(f"    [FAIL] 物料 {index + 1} 在 31 组候选(抓取角 x 放置角)中均无可行解")
                log.append("    [SKIP] on_fail=skip, 跳过该物料")
            items.append({"index": index, "position_raw": [-.41, .15, .03],
                          "success": bool(successful_attempt), "n_angles_tried": count,
                          "angle_grasp_deg": angles[count - 1], "angle_place_deg": angles[count - 1]})
        n_success = sum(bool(value) for value in successes)
        log.append(f"\n[PLAN] 完成 {n_success}/4 个物料, 累计 30.0s")
        (result / "plan.log").write_text("\n".join(log), encoding="utf-8")
        self.write(result / "run_status.json", {"returncode": 0 if n_success else 4, "config": str(source)})
        if n_success:
            self.write(result / "trajectory_meta.json", {"config": cfg,
                       "angle_search": cfg["pick_place"]["angle_search"],
                       "n_items_total": 4, "n_items_success": n_success, "items": items})
        else:
            self.write(result / "plan_failed.json", {"stage": "no_item_success", "config": cfg,
                       "failed": {"failed_item_index": 3, "n_combos_tried": 31,
                                  "candidates": [{"angle_grasp_deg": angle, "angle_place_deg": angle}
                                                 for angle in angles]}})
        return result, source

    def test_positive_success_mixed_failures_and_solver_chatter(self):
        result, source = self.make()
        row = audit_result(result)
        self.assertTrue(row["passed"], row["issues"])
        self.assertEqual(row["status"], "passed")
        self.assertEqual(row["n_success"], 2)
        self.assertEqual(row["n_angle_attempts"], 65)
        self.assertEqual(row["cases"][0]["selected_angle_grasp_deg"], 28)
        self.assertEqual(row["cases"][1]["n_angles_tried"], 31)
        for case in row["cases"]:
            self.assertEqual(case["candidate_angles_grasp_place_deg"][0], [30, 30])
            self.assertEqual(len(case["candidate_log_line_numbers"]), case["n_angles_tried"])
        self.assertEqual(row["source"]["sha256"][str(source)], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_zero_success_has_policy_pass_without_planning_success(self):
        result, _ = self.make(successes=(0, 0, 0, 0))
        row = audit_result(result)
        self.assertTrue(row["passed"], row["issues"])
        self.assertEqual(row["status"], "zero_success_policy_passed")
        self.assertEqual(row["n_success"], 0)
        self.assertEqual(row["n_angle_attempts"], 124)
        failed = json.loads((result / "plan_failed.json").read_text())
        failed["failed"]["candidates"].reverse()
        self.write(result / "plan_failed.json", failed)
        self.assertFalse(audit_result(result)["passed"])

    def test_pending_and_home_failure_never_pass(self):
        result, source = self.make()
        (result / "run_status.json").unlink()
        row = audit_result(result, source)
        self.assertEqual(row["status"], "pending")
        self.assertFalse(row["passed"])
        self.assertFalse(row["verification_completed"])
        self.write(result / "run_status.json", {"returncode": 3, "config": str(source)})
        self.write(result / "plan_failed.json", {"stage": "home_ik", "config": self.cfg})
        row = audit_result(result)
        self.assertEqual(row["status"], "home_failed")
        self.assertFalse(row["passed"])

    def test_wrong_order_place_sign_duplicate_or_missing_outcome_rejected(self):
        mutations = [
            lambda log: log.replace("g=+30 p=+30deg", "g=+28 p=+28deg", 1),
            lambda log: log.replace("g=+30 p=+30deg", "g=+30 p=-30deg", 1),
            lambda log: log.replace("g=+28 p=+28deg", "g=+30 p=+30deg", 1),
            lambda log: log.replace("  -> OK  300 点  6.00s  最大变化 150deg", "", 1),
            lambda log: log.replace("  -> 规划失败: MotionGenStatus.IK_FAIL @ i0_place", "  -> OK  300 点", 1),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                result, _ = self.make(str(index))
                path = result / "plan.log"
                path.write_text(mutate(path.read_text()), encoding="utf-8")
                self.assertFalse(audit_result(result)["passed"])

    def test_metadata_policy_attempt_count_and_adopted_angle_are_checked(self):
        for index, field in enumerate(("policy", "count", "angle")):
            with self.subTest(field=field):
                result, _ = self.make(str(index))
                path = result / "trajectory_meta.json"
                meta = json.loads(path.read_text())
                if field == "policy":
                    meta["angle_search"]["reuse_last_success"] = True
                elif field == "count":
                    meta["items"][0]["n_angles_tried"] += 1
                else:
                    meta["items"][0]["angle_grasp_deg"] = 30
                self.write(path, meta)
                self.assertFalse(audit_result(result)["passed"])

    def test_truncated_failed_case_and_missing_case_rejected(self):
        result, _ = self.make(successes=(0, 0, 0, 0))
        log_path = result / "plan.log"
        original = log_path.read_text()
        log_path.write_text(original.replace("    g=-30 p=-30degCouldn't find solution, resetting seeds\n"
                                            "  -> 规划失败: MotionGenStatus.IK_FAIL @ i0_place\n", "", 1))
        self.assertFalse(audit_result(result)["passed"])
        log_path.write_text(original[:original.index("--- 物料 4/4")] + "[PLAN] 完成 0/4 个物料")
        self.assertFalse(audit_result(result)["passed"])

    def test_manifest_hashes_all_results_and_pending(self):
        a, a_source = self.make("a")
        b, b_source = self.make("b", successes=(0, 0, 0, 0))
        manifest = self.root / "manifest.json"
        candidates = [{"name": result.name, "result": str(result), "config": str(source),
                       "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                      for result, source in ((a, a_source), (b, b_source))]
        self.write(manifest, {"n_configs": 2, "candidates": candidates})
        report = audit(manifest=manifest)
        self.assertTrue(report["passed"])
        self.assertEqual(report["n_passed"], 2)
        self.assertEqual(report["manifest_source"]["sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
        b_source.write_text(b_source.read_text() + "\n")
        report = audit(manifest=manifest)
        self.assertFalse(report["passed"])
        self.assertEqual(report["n_passed"], 1)

    def test_config_range_reuse_stage2_and_truncation_rejected(self):
        for index, (key, value) in enumerate((("reuse_last_success", True), ("order", "asc"),
                                            ("stage2", {"enable": True}), ("max_trials", 16))):
            result, source = self.make(str(index))
            cfg = copy.deepcopy(self.cfg)
            cfg["pick_place"]["angle_search"][key] = value
            self.write(source, cfg)
            self.assertFalse(audit_result(result)["passed"])

    def test_cli_output_is_exclusive(self):
        output = self.root / "existing.json"
        output.write_text("user data")
        proc = subprocess.run([sys.executable, str(SCRIPTS / "audit_overhead_angle_order.py"),
                               "--result", "missing", "--out", str(output)], text=True, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Refusing to replace", proc.stderr)
        self.assertEqual(output.read_text(), "user data")

    def test_ascending_success_negative_zero_positive_and_complete_failure(self):
        result, _ = self.make(successes=(1, 16, 31, 0), order="asc")
        report = audit(results=[result], expected_order="asc")
        self.assertTrue(report["passed"], report["results"][0]["issues"])
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["expected_order"], "asc")
        self.assertEqual(report["expected_angles_grasp_deg"], list(range(-30, 31, 2)))
        row = report["results"][0]
        self.assertEqual(row["n_success"], 3)
        self.assertEqual(row["n_angle_attempts"], 79)
        self.assertEqual([c["selected_angle_grasp_deg"] for c in row["cases"]], [-30, 0, 30, None])
        for case in row["cases"]:
            self.assertEqual(case["candidate_angles_grasp_place_deg"][0], [-30, -30])
        self.assertEqual(row["cases"][3]["candidate_angles_grasp_place_deg"],
                         [[a, a] for a in range(-30, 31, 2)])

    def test_expected_order_is_explicit_not_inferred_from_config(self):
        ascending, _ = self.make("ascending", order="asc")
        descending, _ = self.make("descending")
        self.assertFalse(audit_result(ascending)["passed"])
        self.assertFalse(audit_result(descending, expected_order="asc")["passed"])
        self.assertFalse(audit(results=[ascending])["passed"])
        report = audit(results=[descending])
        self.assertTrue(report["passed"])
        self.assertEqual(report["expected_order"], "desc")
        self.assertEqual(report["expected_angles_grasp_deg"], ANGLES)
        with self.assertRaisesRegex(ValueError, "expected_order"):
            audit(results=[ascending], expected_order="abs")

    def test_ascending_wrong_header_first_order_sign_and_truncation_rejected(self):
        mutations = [
            lambda log: log.replace("order=asc", "order=desc"),
            lambda log: log.replace("g=-30 p=-30deg", "g=+30 p=+30deg", 1),
            lambda log: log.replace("g=-28 p=-28deg", "g=-30 p=-30deg", 1),
            lambda log: log.replace("g=-30 p=-30deg", "g=-30 p=+30deg", 1),
            lambda log: log.replace("    g=+30 p=+30degCouldn't find solution, resetting seeds\n"
                                    "  -> 规划失败: MotionGenStatus.IK_FAIL @ i0_place\n", "", 1),
            lambda log: log.replace("  -> 规划失败: MotionGenStatus.IK_FAIL @ i0_place", "", 1),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                result, _ = self.make(str(index), successes=(0, 0, 0, 0), order="asc")
                path = result / "plan.log"
                path.write_text(mutate(path.read_text()), encoding="utf-8")
                self.assertFalse(audit_result(result, expected_order="asc")["passed"])

    def test_ascending_metadata_policy_count_and_adopted_angle_are_checked(self):
        for field in ("policy", "config_policy", "count", "angle"):
            with self.subTest(field=field):
                result, _ = self.make(field, order="asc")
                path = result / "trajectory_meta.json"
                meta = json.loads(path.read_text())
                if field == "policy":
                    meta["angle_search"]["order"] = "desc"
                elif field == "config_policy":
                    meta["config"]["pick_place"]["angle_search"]["order"] = "desc"
                elif field == "count":
                    meta["items"][0]["n_angles_tried"] += 1
                else:
                    meta["items"][0]["angle_grasp_deg"] = 28
                self.write(path, meta)
                self.assertFalse(audit_result(result, expected_order="asc")["passed"])

    def test_ascending_zero_success_failed_config_and_sequence_are_checked(self):
        for field in ("config", "sequence"):
            with self.subTest(field=field):
                result, _ = self.make(field, successes=(0, 0, 0, 0), order="asc")
                row = audit_result(result, expected_order="asc")
                self.assertTrue(row["passed"], row["issues"])
                self.assertEqual(row["status"], "zero_success_policy_passed")
                self.assertEqual(row["n_angle_attempts"], 124)
                path = result / "plan_failed.json"
                failed = json.loads(path.read_text())
                if field == "config":
                    failed["config"]["pick_place"]["angle_search"]["order"] = "desc"
                else:
                    failed["failed"]["candidates"].reverse()
                self.write(path, failed)
                self.assertFalse(audit_result(result, expected_order="asc")["passed"])

    def test_ascending_manifest_pending_and_home_failure_never_pass(self):
        result, source = self.make(order="asc")
        manifest = self.root / "manifest.json"
        self.write(manifest, {"n_configs": 1, "candidates": [{"name": "run", "result": str(result),
                   "config": str(source), "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]})
        self.assertTrue(audit(manifest=manifest, expected_order="asc")["passed"])
        self.assertFalse(audit(manifest=manifest)["passed"])
        (result / "run_status.json").unlink()
        report = audit(manifest=manifest, expected_order="asc")
        self.assertFalse(report["passed"])
        self.assertFalse(report["verification_completed"])
        self.assertEqual(report["results"][0]["status"], "pending")
        self.write(result / "run_status.json", {"returncode": 3, "config": str(source)})
        self.write(result / "plan_failed.json", {"stage": "home_ik"})
        row = audit_result(result, expected_order="asc")
        self.assertFalse(row["passed"])
        self.assertEqual(row["status"], "home_failed")

    def test_cli_expected_order_ascending_and_default_descending(self):
        result, _ = self.make(order="asc")
        for order in (None, "asc"):
            output = self.root / f"report_{order}.json"
            command = [sys.executable, str(SCRIPTS / "audit_overhead_angle_order.py"),
                       "--result", str(result), "--out", str(output)]
            if order is not None:
                command += ["--expected-order", order]
            proc = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0 if order == "asc" else 1, proc.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["passed"], order == "asc")
            self.assertEqual(report["expected_order"], order or "desc")
        invalid = subprocess.run(command + ["--expected-order", "abs"], text=True, capture_output=True)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("invalid choice", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
