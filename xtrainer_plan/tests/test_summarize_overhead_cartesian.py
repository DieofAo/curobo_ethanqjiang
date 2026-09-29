#!/usr/bin/env python3
"""CPU-only failure classification and selection tests; no ROS/GPU use."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import summarize_overhead_cartesian as sweep  # noqa: E402


class CartesianSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {"robot": {"joint_limit_clip": .14, "mount_transform": [
            [0, 1, 0, -.41], [1, 0, 0, .35], [0, 0, -1, .65], [0, 0, 0, 1]]},
            "overhead": {"mount_rpy_deg": [180, 0, 90]}, "pick_place": {
                "place": {"position": [-.26, -.12, .1]}, "grasp_grid": {
                    "rows": 3, "cols": 3, "x_range": [-.62, -.2], "y_range": [-.1, .4],
                    "z": .03, "order": "ring", "perimeter_only": False}}}

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def candidate(self, name="candidate", cfg=None, code=0, completed=True, success=None):
        result = self.root / name
        result.mkdir()
        cfg = copy.deepcopy(self.cfg if cfg is None else cfg)
        cfg_path = self.root / f"{name}.json"
        self.write(cfg_path, cfg)
        if completed:
            self.write(result / "run_status.json", {"returncode": code, "elapsed_s": 30,
                                                   "config": str(cfg_path)})
        if success is not None:
            items = [{"index": i, "position_raw": [-.62 + (i % 3) * .21, -.1 + (i // 3) * .25, .03],
                      "success": i < success, "n_points": 2 if i < success else 0} for i in range(9)]
            self.write(result / "trajectory_meta.json", {"config": cfg, "n_items_total": 9,
                       "n_items_success": success, "n_points": 2 * success, "items": items})
        return result

    def add_reports(self, result):
        for name in sweep.REPORT_NAMES:
            self.write(result / name, {})

    def validated(self, result, *, passed=True, link3_cases=0, j6_margin=.2):
        meta = sweep.read_json(result / "trajectory_meta.json")
        return {"result": str(result.resolve()), "n_total": 9,
                "n_success": meta["n_items_success"], "success_fraction": meta["n_items_success"] / 9,
                "independent_verification_passed": passed, "joint_limit_clip_audit_passed": True,
                "J6": {"minimum_effective_limit_margin_rad": j6_margin, "max_segment_span_deg": 170.},
                "LINK3": {"place_related": {"n_plane_intersection_cases": link3_cases}}}, sweep.case_map(meta)

    def test_home_failure_does_not_claim_grid_failures(self):
        result = self.candidate(code=2)
        self.write(result / "plan_failed.json", {"stage": "home_ik", "config": self.cfg})
        row, cases = sweep.inspect_candidate(result)
        self.assertEqual(row["status"], "home_failed")
        self.assertIsNone(row["n_success"])
        self.assertIsNone(row["n_total"])
        self.assertEqual(row["expected_n_total"], 9)
        self.assertFalse(row["ranking_eligible"])
        self.assertIsNone(cases)

    def test_zero_requires_all_cases_completed_log_not_just_exit_code(self):
        result = self.candidate(code=4)
        self.write(result / "plan_failed.json", {"stage": "no_item_success", "config": self.cfg})
        headers = "\n".join(f"--- 物料 {i}/9  抓取点 [-0.4, 0.1, 0.03] ---" for i in range(1, 10))
        (result / "plan.log").write_text(headers + "\n[PLAN] 完成 0/9 个物料, 累计 20.0s\n", encoding="utf-8")
        row, _ = sweep.inspect_candidate(result)
        self.assertEqual(row["status"], "completed_zero_success")
        self.assertEqual((row["n_success"], row["n_total"]), (0, 9))
        self.assertFalse(row["ranking_eligible"])
        (result / "plan.log").write_text("[PLAN] 完成 0/3 个物料\n", encoding="utf-8")
        row, _ = sweep.inspect_candidate(result)
        self.assertEqual(row["status"], "planning_failed")
        self.assertIsNone(row["n_success"])
        (result / "plan.log").write_text("--- 物料 1/9  抓取点 [-0.4, 0.1, 0.03] ---\n"
                                         "[PLAN] 完成 0/9 个物料, 累计 20.0s\n", encoding="utf-8")
        row, _ = sweep.inspect_candidate(result)
        self.assertEqual(row["status"], "planning_failed")
        self.assertIsNone(row["n_success"])

    def test_unfinished_and_partial_runs_are_not_zero(self):
        result = self.candidate(completed=False, success=5)
        row, _ = sweep.inspect_candidate(result)
        self.assertEqual(row["status"], "pending")
        self.assertIsNone(row["n_success"])
        self.assertIsNone(row["n_total"])
        missing = self.root / "not_yet_created"
        row, _ = sweep.inspect_candidate(missing)
        self.assertEqual(row["status"], "pending")
        self.assertIsNone(row["n_success"])
        self.write(result / "run_status.json", {"returncode": 0, "config": str(self.root / "candidate.json")})
        meta = sweep.read_json(result / "trajectory_meta.json")
        meta["partial"] = True
        self.write(result / "trajectory_meta.json", meta)
        row, _ = sweep.inspect_candidate(result)
        self.assertEqual(row["status"], "planning_failed")
        self.assertIsNone(row["n_success"])

    def test_audit_missing_has_unverified_counts_but_no_rank(self):
        result = self.candidate(success=9)
        row, _ = sweep.inspect_candidate(result)
        self.assertEqual(row["status"], "audit_missing")
        self.assertEqual(row["n_success"], 9)
        self.assertFalse(row["ranking_eligible"])
        self.assertIn("run_status.json", row["provenance"]["sha256"])

    def test_verified_requires_helper_consistency_and_passed_flags(self):
        result = self.candidate(success=8)
        self.add_reports(result)
        with patch.object(sweep, "summarize", return_value=self.validated(result)) as helper:
            row, cases = sweep.inspect_candidate(result)
            helper.assert_called_once_with(result.resolve())
        self.assertEqual(row["status"], "completed_verified")
        self.assertTrue(row["ranking_eligible"])
        self.assertEqual(len(cases), 9)
        self.assertEqual(row["place_xyz_original_LINK0_m"], [-.26, -.12, .1])
        with patch.object(sweep, "summarize", side_effect=ValueError("stale source hash")):
            row, _ = sweep.inspect_candidate(result)
        self.assertEqual(row["status"], "audit_failed")
        self.assertFalse(row["ranking_eligible"])
        with patch.object(sweep, "summarize", return_value=self.validated(result, passed=False)):
            row, _ = sweep.inspect_candidate(result)
        self.assertEqual(row["status"], "audit_failed")
        self.assertFalse(row["ranking_eligible"])

    def test_rank_coverage_first_diagnostic_ties_and_unverified_excluded(self):
        configs = [copy.deepcopy(self.cfg) for _ in range(4)]
        for cfg, px in zip(configs, [-.46, -.36, -.26, -.16]):
            cfg["pick_place"]["place"]["position"][0] = px
        a, b, c, d = [self.candidate(str(i), cfg=cfg, success=s)
                       for i, (cfg, s) in enumerate(zip(configs, [7, 8, 8, 9]))]
        for result in (a, b, c):
            self.add_reports(result)
        values = {a: self.validated(a, j6_margin=.3),
                  b: self.validated(b, link3_cases=1), c: self.validated(c, link3_cases=0)}
        with patch.object(sweep, "summarize", side_effect=lambda path: values[path]):
            report = sweep.aggregate([a, b, c, d])
        self.assertEqual(report["ranked_result_paths"], [str(c), str(b), str(a)])
        self.assertEqual(report["best_coverage_result_paths"], [str(c), str(b)])
        self.assertEqual(report["selected_result"], str(c))
        self.assertEqual(report["n_ranked"], 3)
        self.assertEqual(report["status_counts"]["audit_missing"], 1)
        self.assertEqual(report["missing_cartesian_coordinates"], [])
        json.dumps(report, allow_nan=False)

    def test_fixed_parameters_duplicate_coordinates_and_input_paths_rejected(self):
        a = self.candidate("a", success=1)
        b = self.candidate("b", success=1)
        with self.assertRaisesRegex(ValueError, "Duplicate base XY"):
            sweep.aggregate([a, b])
        with self.assertRaisesRegex(ValueError, "Duplicate result"):
            sweep.aggregate([a, a])
        cfg = copy.deepcopy(self.cfg)
        cfg["robot"]["joint_limit_clip"] = .15
        c = self.candidate("c", cfg=cfg, success=1)
        with self.assertRaisesRegex(ValueError, "different fixed"):
            sweep.aggregate([a, c])

    def test_no_verified_candidate_has_no_winner_and_plot_is_exclusive(self):
        result = self.candidate(success=9)
        report = sweep.aggregate([result])
        self.assertIsNone(report["selected_result"])
        self.assertEqual(report["best_coverage_result_paths"], [])
        output = self.root / "map.png"
        sweep.plot_report(report, output)
        self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))
        with self.assertRaisesRegex(FileExistsError, "Refusing to replace"):
            sweep.plot_report(report, output)

    def test_saved_config_parameter_mismatch_blocks_rank(self):
        result = self.candidate(success=8)
        meta = sweep.read_json(result / "trajectory_meta.json")
        meta["config"]["pick_place"]["place"]["position"][1] = -.23
        self.write(result / "trajectory_meta.json", meta)
        row, _ = sweep.inspect_candidate(result)
        self.assertEqual(row["status"], "planning_failed")
        self.assertFalse(row["ranking_eligible"])

    def test_cli_never_replaces_outputs(self):
        output = self.root / "user.json"
        output.write_text("user data", encoding="utf-8")
        completed = subprocess.run([sys.executable, str(SCRIPTS / "summarize_overhead_cartesian.py"),
                                    "--results", "missing", "--out", str(output)],
                                   text=True, capture_output=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Refusing to replace", completed.stderr)
        self.assertEqual(output.read_text(), "user data")


if __name__ == "__main__":
    unittest.main()
