#!/usr/bin/env python3
"""CPU-only checks for sample slicing, paired denominators and audit gating."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_original_base_threeway as comparison


def item(index, success=True):
    result = {"index": index, "row": index // 2, "col": index % 2,
              "position_raw": [-.6 + .1 * (index // 2), -.1 + .1 * (index % 2), .03],
              "success": success, "n_angles_tried": 2}
    if success:
        result.update({"n_points": 7, "duration_s": .14,
                       "angle_grasp_deg": -2., "angle_place_deg": -2.,
                       "segments": [{"index": i, "to": f"i{index}_{phase}", "n_points": 2}
                                    for i, phase in enumerate(comparison.PHASES)]})
    return result


def example_metrics():
    q = np.zeros((14, 6))
    q[:7, 0] = np.deg2rad([10, 20, 0, 40, 20, 30, 10])
    q[:7, 5] = np.deg2rad([170, 0, -170, -100, -50, 50, 170])
    q[7:, 0] = np.deg2rad(10.)
    q[7:, 5] = np.deg2rad([170, 160, 150, 160, 170, 165, 160])
    meta = {"items": [item(0), item(1, False), item(2)], "n_points": 14}
    raw = np.array([[-np.pi] * 6, [np.pi] * 6])
    effective = raw + np.array([[.14], [-.14]])
    return meta, q, np.arange(14) * .02, raw, effective


class ComparisonMetricsTest(unittest.TestCase):
    def test_six_phases_and_failed_case_cursor(self):
        cases = comparison.case_metrics(*example_metrics())
        self.assertEqual(cases[0]["sample_range_half_open"], [0, 7])
        self.assertEqual([p["sample_range_half_open"] for p in cases[0]["phases"]],
                         [[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]])
        self.assertNotIn("sample_range_half_open", cases[1])
        self.assertEqual(cases[2]["sample_range_half_open"], [7, 14])
        self.assertEqual(cases[2]["previous_success_index"], 0)
        self.assertEqual(cases[1]["previous_success_index"], 0)
        self.assertAlmostEqual(cases[0]["full_case_J6_ptp_deg"], 340)
        self.assertAlmostEqual(cases[0]["travel_per_joint_deg"][5], 680)
        self.assertAlmostEqual(cases[0]["max_phase_J6_ptp_deg"], 170)
        self.assertAlmostEqual(cases[0]["J6_raw_margin_deg"], 10)
        self.assertAlmostEqual(cases[0]["J6_effective_margin_deg"], 10 - np.degrees(.14))

    def test_malformed_phase_and_sample_counts_fail_closed(self):
        for mutation in ("count", "order", "missing", "final"):
            args = list(example_metrics())
            meta = args[0]
            if mutation == "count":
                meta["items"][0]["n_points"] = 6
            elif mutation == "order":
                meta["items"][0]["segments"][1]["to"] = "i0_place"
            elif mutation == "missing":
                meta["items"][0]["segments"].pop()
            else:
                meta["n_points"] = 13
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                comparison.case_metrics(*args)

    def test_invalid_time_or_positions_rejected(self):
        args = list(example_metrics())
        args[1][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            comparison.case_metrics(*args)
        args = list(example_metrics())
        args[2][2] = args[2][1]
        with self.assertRaisesRegex(ValueError, "time"):
            comparison.case_metrics(*args)

    def test_motion_quality_denominator_excludes_failed_cases(self):
        cases = comparison.case_metrics(*example_metrics())
        stats = comparison.population_stats([cases[0], cases[2]])
        for threshold in (150, 180, 210):
            stat = stats["ptp_greater_or_equal_thresholds_deg"]["full_case_any_joint"][str(threshold)]
            self.assertEqual(stat["count"], 1)
            self.assertEqual(stat["denominator"], 2)
            self.assertEqual(stat["fraction"], .5)
        with self.assertRaisesRegex(ValueError, "successful cases"):
            comparison.population_stats(cases)

    def test_empty_matched_population_uses_null_not_zero(self):
        stats = comparison.population_stats([])
        self.assertIsNone(stats["metrics"]["J6_raw_margin_deg"]["min"])
        self.assertIsNone(stats["ptp_greater_or_equal_thresholds_deg"]["full_case_J6"]["180"]["fraction"])
        self.assertIsNone(stats["effective_margin_below_1deg"]["any_joint"]["fraction"])
        json.dumps(stats, allow_nan=False)

    def test_near_limit_thresholds_are_strict_and_use_success_denominator(self):
        template = comparison.case_metrics(*example_metrics())[0]
        cases = []
        for index, margin in enumerate((.999, 1., 4.999, 5.)):
            case = copy.deepcopy(template)
            case.update(index=index, min_effective_margin_deg=margin, J6_effective_margin_deg=margin)
            cases.append(case)
        stats = comparison.population_stats(cases)
        for joint in ("any_joint", "J6"):
            one = stats["effective_margin_below_1deg"][joint]
            five = stats["effective_margin_below_5deg"][joint]
            self.assertEqual(one, {"count": 1, "denominator": 4, "fraction": .25, "indices": [0]})
            self.assertEqual(five, {"count": 3, "denominator": 4, "fraction": .75, "indices": [0, 1, 2]})

    def test_stale_hash_rejected(self):
        comparison.check_hash("abc", "abc", "test")
        for value in ("different", None):
            with self.assertRaisesRegex(ValueError, "Stale/mismatched"):
                comparison.check_hash("abc", value, "test")

    def test_missing_or_nonboolean_audit_checks_rejected(self):
        comparison.require_true_checks({"one": True, "two": True}, {"one", "two"}, "test")
        for checks in ({}, {"one": True}, {"one": True, "two": False}, {"one": 1, "two": True}):
            with self.subTest(checks=checks), self.assertRaisesRegex(ValueError, "missing or failed"):
                comparison.require_true_checks(checks, {"one", "two"}, "test")

    def test_intercase_joint_gap_is_not_silently_excluded_from_travel(self):
        args = list(example_metrics())
        args[1][7, 0] += np.deg2rad(2.)
        with self.assertRaisesRegex(ValueError, "uncounted joint gap"):
            comparison.case_metrics(*args)

    def test_link3_reference_plane_diagnostics_keep_warning_and_null_height(self):
        meta = {"items": [item(0), item(1, False), item(2)]}
        cpu = {"cases": [], "groups": {
            "all_segments": {"n_unique_samples": 14, "n_samples_intersecting_finite_grasp_plane": 3,
                             "minimum_height_above_grasp_plane_over_rectangle_m": {"value_m": -.027}},
            "place_related": {"n_unique_samples": 8, "n_samples_intersecting_finite_grasp_plane": 0,
                              "minimum_height_above_grasp_plane_over_rectangle_m": None}}}
        for source in meta["items"]:
            case = {key: source[key] for key in ("index", "success", "position_raw")}
            case["groups"] = {}
            if case["success"]:
                case["groups"] = {
                    "all_segments": {"n_unique_samples": 7,
                                     "n_samples_intersecting_finite_grasp_plane": 3 if case["index"] == 2 else 0},
                    "place_related": {"n_unique_samples": 4, "n_samples_intersecting_finite_grasp_plane": 0}}
            cpu["cases"].append(case)
        report = comparison.link3_reference_plane_diagnostics(cpu, meta)
        all_segments = report["groups"]["all_segments"]
        self.assertEqual(all_segments["case_indices"], [2])
        self.assertEqual(all_segments["n_samples_intersecting_finite_grasp_plane"], 3)
        self.assertEqual(all_segments["success_case_denominator"], 2)
        self.assertEqual(all_segments["success_case_fraction"], .5)
        self.assertEqual(all_segments["minimum_height_above_grasp_plane_over_rectangle_mm"], -27.)
        self.assertIsNone(report["groups"]["place_related"]["minimum_height_above_grasp_plane_over_rectangle_mm"])
        self.assertIn("NOT a CAD collision result", report["warning"])
        cpu["groups"]["all_segments"]["n_samples_intersecting_finite_grasp_plane"] = 99
        with self.assertRaisesRegex(ValueError, "intersection counts disagree"):
            comparison.link3_reference_plane_diagnostics(cpu, meta)


class ComparisonReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.names = ["place180_asc", "place0_desc", "place0_asc"]
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"n_configs": 3, "candidates": [
            {"name": name} for name in self.names]}), encoding="utf-8")
        self.sweep = {"manifest": str(self.manifest),
                      "outcomes": [{"name": name, "batch_returncode": 0, "plan_returncode": 0,
                                   "postprocessing": [{"script": s, "returncode": 0}
                                                      for s in comparison.AUDIT_SCRIPTS]}
                                  for name in self.names]}
        self.write_status()
        template = comparison.case_metrics(*example_metrics())[0]
        self.runs = []
        for group, successes in enumerate(({0, 2}, {0, 1}, {0, 2, 3})):
            cases = []
            for index in range(4):
                case = item(index, index in successes)
                if index in successes:
                    extra = copy.deepcopy(template)
                    extra.update({key: case[key] for key in ("index", "row", "col", "position_raw")})
                    extra["incoming_joint_deg"][0] += group * 20.
                    case = extra
                cases.append(case)
            self.runs.append({
                "name": self.names[group], "result": str(self.root / self.names[group]),
                "config": {"robot": {"joint_limit_clip": .14},
                           "pick_place": {"place": {"position": [-.36, -.12, .1],
                                                      "tool_z_rotation_deg": 180 if group == 0 else 0},
                                          "angle_search": {"order": "desc" if group == 1 else "asc"}}},
                "cases": cases, "failure_classification": {},
                "LINK3_reference_plane_diagnostics": {"groups": {
                    key: {"n_success_cases_intersecting_finite_grasp_plane": 0,
                          "success_case_denominator": len(successes),
                          "n_samples_intersecting_finite_grasp_plane": 0,
                          "minimum_height_above_grasp_plane_over_rectangle_mm": None}
                    for key in ("all_segments", "place_related")}},
            })

    def tearDown(self):
        self.temporary.cleanup()

    def write_status(self):
        (self.root / "sweep_status.json").write_text(json.dumps(self.sweep), encoding="utf-8")

    def report(self):
        with patch.object(comparison, "load_candidate", side_effect=self.runs):
            return comparison.build_comparison(self.manifest, expected_cases=4)

    def test_common_target_denominators_and_incoming_difference(self):
        report = self.report()
        self.assertEqual(report["overlap"]["all_three_success_indices"], [0])
        self.assertEqual(report["overlap"]["any_success_count"], 4)
        self.assertEqual([g["own_success_population"]["n_successful_cases_in_population"]
                          for g in report["groups"]], [2, 2, 3])
        self.assertEqual([g["matched_all_three_success_population"]["n_successful_cases_in_population"]
                          for g in report["groups"]], [1, 1, 1])
        for pair in report["matched_pairwise_quality"]:
            self.assertEqual(pair["denominator_all_three_common_success"], 1)
            self.assertEqual(pair["n_incoming_states_equal_within_0p001_deg"], 0)
        self.assertEqual(sum(p["count"] for p in report["overlap"]["success_patterns"]), 4)

    def test_failed_audit_or_missing_outcome_rejected_before_loading(self):
        self.sweep["outcomes"][0]["postprocessing"][0]["returncode"] = 1
        self.write_status()
        with patch.object(comparison, "load_candidate") as loader:
            with self.assertRaisesRegex(ValueError, "postprocessing"):
                comparison.build_comparison(self.manifest, expected_cases=4)
            loader.assert_not_called()
        self.sweep["outcomes"].pop()
        self.write_status()
        with self.assertRaisesRegex(ValueError, "outcomes incomplete"):
            self.report()

    def test_noncontrolled_changes_or_mapping_changes_rejected(self):
        self.runs[1]["config"]["robot"]["joint_limit_clip"] = .10
        with self.assertRaisesRegex(ValueError, "Non-controlled"):
            self.report()
        self.runs[1]["config"]["robot"]["joint_limit_clip"] = .14
        self.runs[1]["cases"][0]["row"] = 1
        with self.assertRaisesRegex(ValueError, "index/row/column"):
            self.report()

    def test_csv_aligns_all_targets_blanks_failure_and_refuses_overwrite(self):
        report = self.report()
        comparison.write_outputs(report, self.root)
        with (self.root / "case_comparison.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1][f"{self.names[0]}.success"], "False")
        self.assertEqual(rows[1][f"{self.names[0]}.J6_raw_margin_deg"], "")
        self.assertNotEqual(rows[1][f"{self.names[1]}.J6_raw_margin_deg"], "")
        self.assertIn("同点配对不等于相同入场关节构型", (self.root / "summary.md").read_text())
        with self.assertRaises(FileExistsError):
            comparison.write_outputs(report, self.root)


if __name__ == "__main__":
    unittest.main()
