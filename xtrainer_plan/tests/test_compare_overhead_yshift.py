#!/usr/bin/env python3
"""CPU-only comparison tests; synthetic data never initializes CuRobo or ROS."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from compare_overhead_yshift import case_map, compare_cases, summarize  # noqa: E402


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.result = Path(self.temp.name) / "run"
        self.result.mkdir()
        self.items = [{"index": i, "position_raw": [-.4, i * .1, .03],
                       "success": i != 2, "n_points": 2 if i != 2 else 0} for i in range(3)]
        self.meta = {"n_items_total": 3, "n_items_success": 2, "n_points": 4,
                     "total_duration_s": .06, "items": self.items, "config": {
                         "robot": {"joint_limit_clip": .15, "mount_transform": np.eye(4).tolist()},
                         "overhead": {"mount_rpy_deg": [180, 0, 90]}}}
        self.names = [f"J_{i}" for i in range(1, 7)]

    def write_json(self, name, data):
        (self.result / name).write_text(json.dumps(data), encoding="utf-8")

    def fixture(self):
        self.write_json("trajectory_meta.json", self.meta)
        q = np.zeros((4, 6))
        q[0, 0] = -2.99  # This case is near J1, but NOT J6.
        q[2, 5] = 2.99
        np.savez(self.result / "trajectory.npz", positions=q, joint_names=self.names)
        hashes = {name: hashlib.sha256((self.result / name).read_bytes()).hexdigest()
                  for name in ("trajectory_meta.json", "trajectory.npz")}
        self.write_json("analysis_summary.json", {
            "source": {"sha256": hashes}, "summary": {"n_items_total": 3, "n_items_success": 2},
            "joint_metrics": {"max_J6_segment": {"J6_span_deg": float(np.degrees(2.99))}},
            "failure_classification": {"last_attempt_categories_per_skipped_case": {"IK_FAIL": 1}}})
        group = {"n_unique_samples": 4, "n_samples_intersecting_finite_grasp_plane": 1,
                 "minimum_height_above_grasp_plane_over_rectangle_m": {"value_m": -.01},
                 "minimum_lowest_surface_over_rectangle_z_m": {"value_m": .02}}
        groups = ("all_segments", "place_related")
        self.write_json("link3_grasp_clearance.json", {
            "source": {"sha256": hashes}, "groups": {name: group for name in groups},
            "cases": [dict(item, groups={name: {"n_samples_intersecting_finite_grasp_plane":
                                               int(item["index"] == 1)} for name in groups})
                      for item in self.items]})
        self.write_json("independent_verification.json", {
            "source": {"metadata_sha256": hashes["trajectory_meta.json"],
                       "npz_sha256": hashes["trajectory.npz"]},
            "verification_completed": True, "passed": True,
            "n_items_total": 3, "n_items_success": 2, "joint_names": self.names,
            "joint_position_limits": {"lower_rad": [-2.99] * 6, "upper_rad": [2.99] * 6}})
        self.write_json("joint_limit_clip_audit.json", {
            "source": {"config_source_sha256": hashes["trajectory_meta.json"],
                       "trajectory_sha256": hashes["trajectory.npz"]},
            "verification_completed": True, "passed": True, "joint_names": self.names,
            "clip_rad": .15, "raw_limits": {"lower_rad": [-3.14] * 6, "upper_rad": [3.14] * 6}})

    def test_gained_and_lost_are_matched_by_same_original_point(self):
        candidate = copy.deepcopy(self.meta)
        candidate["items"][0]["success"] = False
        candidate["items"][2]["success"] = True
        result = compare_cases(case_map(self.meta), case_map(candidate))
        self.assertEqual(result["new_success_indices"], [2])
        self.assertEqual(result["lost_success_indices"], [0])
        self.assertEqual(result["common_success_indices"], [1])

    def test_changed_point_or_order_or_duplicate_index_rejected(self):
        candidate = copy.deepcopy(self.meta)
        candidate["items"][0]["position_raw"][0] += .01
        with self.assertRaisesRegex(ValueError, "different original grasp"):
            compare_cases(case_map(self.meta), case_map(candidate))
        candidate = copy.deepcopy(self.meta)
        candidate["items"].reverse()
        with self.assertRaisesRegex(ValueError, "execution orders"):
            compare_cases(case_map(self.meta), case_map(candidate))
        candidate["items"][0]["index"] = candidate["items"][1]["index"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            case_map(candidate)

    def test_j6_only_margin_and_case_membership_and_link3(self):
        self.fixture()
        row, _ = summarize(self.result)
        self.assertAlmostEqual(row["J6"]["minimum_raw_limit_margin_rad"], .15)
        self.assertAlmostEqual(row["J6"]["minimum_effective_limit_margin_rad"], 0.)
        self.assertEqual(row["J6"]["n_samples_effective_margin_below_1deg"], 1)
        self.assertEqual(row["J6"]["cases_effective_margin_below_1deg"], [1])
        self.assertEqual(row["LINK3"]["place_related"]["plane_intersection_case_indices"], [1])
        json.dumps(row, allow_nan=False)

    def test_stale_hash_and_missing_audit_rejected(self):
        self.fixture()
        self.meta["total_duration_s"] = 5
        self.write_json("trajectory_meta.json", self.meta)
        with self.assertRaisesRegex(ValueError, "stale source hash"):
            summarize(self.result)
        (self.result / "joint_limit_clip_audit.json").unlink()
        with self.assertRaises(FileNotFoundError):
            summarize(self.result)

    def test_historical_parent_report_layout(self):
        self.fixture()
        for name in ("analysis_summary.json", "link3_grasp_clearance.json"):
            (self.result / name).rename(self.result.parent / name)
        row, _ = summarize(self.result)
        self.assertEqual(row["n_success"], 2)

    def test_cli_never_overwrites_existing_output(self):
        output = Path(self.temp.name) / "existing.json"
        output.write_text("user data")
        result = subprocess.run([sys.executable, str(SCRIPTS / "compare_overhead_yshift.py"),
                                 "--results", "missing", "--out", str(output)],
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to replace", result.stderr)
        self.assertEqual(output.read_text(), "user data")


if __name__ == "__main__":
    unittest.main()
