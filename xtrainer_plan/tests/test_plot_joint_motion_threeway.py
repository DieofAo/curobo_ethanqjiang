"""Focused checks for per-joint argmax selection and spatial max-over-joints."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_joint_motion_threeway import map_rows, worst_cases


class JointSpanTest(unittest.TestCase):
    def example(self):
        degrees = np.array([[0, 0, 0, 0, 0, 170], [30, 0, 0, 0, 0, -170],
                            [0, 0, 0, 0, 0, 10], [80, 0, 0, 0, 0, 20]], dtype=float)
        q = np.radians(degrees)
        cases = []
        for index, lo, hi in ((0, 0, 2), (2, 2, 4)):
            cases.append({"index": index, "row": 0, "col": index, "position_raw": [-.5, index * .02, .03],
                          "success": True, "sample_range_half_open": [lo, hi],
                          "full_case_ptp_per_joint_deg": np.ptp(degrees[lo:hi], axis=0).tolist(),
                          "phases": [{"phase": "test", "sample_range_half_open": [lo, hi]}]})
        cases.insert(1, {"index": 1, "row": 0, "col": 1, "position_raw": [-.5, .02, .03], "success": False})
        return cases, q

    def test_per_joint_maximum_may_come_from_different_cases_no_wrap(self):
        rows = worst_cases(*self.example())
        self.assertEqual(rows[0]["case_index"], 2)
        self.assertAlmostEqual(rows[0]["max_span_deg"], 80)
        self.assertEqual(rows[5]["case_index"], 0)
        self.assertAlmostEqual(rows[5]["max_span_deg"], 340)
        self.assertEqual(rows[5]["min_sample_index"], 1)
        self.assertEqual(rows[5]["max_sample_index"], 0)
        self.assertEqual(rows[5]["n_success_in_population"], 2)

    def test_common_population_filter(self):
        cases, q = self.example()
        rows = worst_cases(cases, q, {2})
        self.assertAlmostEqual(rows[5]["max_span_deg"], 10)
        self.assertEqual(rows[5]["n_success_in_population"], 1)
        with self.assertRaises(ValueError):
            worst_cases(cases, q, {1})

    def test_map_maximum_joint_and_failed_missing_value(self):
        rows = map_rows(self.example()[0])
        self.assertEqual(rows[0]["max_joint"], "J_6")
        self.assertEqual(rows[0]["max_joint_span_deg"], 340)
        self.assertNotIn("max_joint_span_deg", rows[1])
        self.assertEqual(rows[2]["max_joint"], "J_1")


if __name__ == "__main__":
    unittest.main()
