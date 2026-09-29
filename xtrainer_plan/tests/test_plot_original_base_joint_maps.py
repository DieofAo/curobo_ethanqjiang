"""Focused tests for saved grasp/place endpoint extraction and failed cases."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_original_base_threeway import case_metrics, PHASES
from plot_original_base_joint_maps import extract_rows


class JointMapTest(unittest.TestCase):
    def fixture(self):
        items = []
        for index, success in enumerate([True, False, True]):
            item = {"index": index, "row": 0, "col": index,
                    "position_raw": [-.62, -.1 + .02 * index, .03],
                    "success": success, "n_angles_tried": 1}
            if success:
                item.update(n_points=7, duration_s=.12, angle_grasp_deg=-30., angle_place_deg=-30.,
                            segments=[{"index": i, "to": f"i{index}_{p}", "n_points": 2}
                                      for i, p in enumerate(PHASES)])
            items.append(item)
        q = np.zeros((14, 6))
        q[:, 5] = np.deg2rad([10, 20, 30, 40, -160, -60, -50,
                             -50, -10, -20, -30, 160, 60, 50])
        raw = np.array([[-3.14] * 6, [3.14] * 6])
        effective = np.array([[-3.] * 6, [3.] * 6])
        cases = case_metrics({"items": items, "n_points": 14}, q, np.arange(14) * .02, raw, effective)
        return extract_rows(cases, q, 5, raw, effective)

    def test_phase_endpoint_not_start_or_tool_search_angle(self):
        first = self.fixture()[0]
        self.assertEqual(first["grasp_sample_index"], 2)
        self.assertEqual(first["place_sample_index"], 5)
        self.assertAlmostEqual(first["grasp_deg"], 30.)
        self.assertAlmostEqual(first["place_deg"], -60.)
        self.assertNotEqual(first["grasp_deg"], first["grasp_search_angle_deg"])

    def test_failed_case_blank_and_does_not_advance_sample_cursor(self):
        first, failed, third = self.fixture()
        self.assertFalse(failed["success"])
        self.assertNotIn("grasp_deg", failed)
        self.assertNotIn("cycle_min_deg", failed)
        self.assertEqual(third["sample_start"], 7)
        self.assertEqual(third["grasp_sample_index"], 9)
        self.assertEqual(third["place_sample_index"], 12)
        self.assertAlmostEqual(third["grasp_deg"], -20.)
        self.assertAlmostEqual(third["place_deg"], 60.)

    def test_full_cycle_extrema_and_both_sided_margin(self):
        first, _, third = self.fixture()
        self.assertAlmostEqual(first["cycle_min_deg"], -160.)
        self.assertAlmostEqual(first["cycle_max_deg"], 40.)
        self.assertAlmostEqual(first["cycle_span_deg"], 200.)
        self.assertAlmostEqual(first["effective_margin_deg"], np.degrees(3.) - 160.)
        self.assertAlmostEqual(first["raw_margin_deg"] - first["effective_margin_deg"], np.degrees(.14))
        self.assertEqual(first["nearest_limit_sample_index"], 4)
        self.assertAlmostEqual(third["effective_margin_deg"], np.degrees(3.) - 160.)
        self.assertEqual(third["nearest_limit_sample_index"], 11)


if __name__ == "__main__":
    unittest.main()
