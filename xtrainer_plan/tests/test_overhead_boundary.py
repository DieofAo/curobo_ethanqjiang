#!/usr/bin/env python3
"""CPU-only tests for sampled boundary summaries; no curobo/torch imports."""
import copy
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_overhead_boundary import (  # noqa: E402
    compatible_configs, largest_success_rectangle, rectangle_bounds, summarize_grid,
)


class BoundaryTest(unittest.TestCase):
    def test_no_padding_and_outer_edge(self):
        result = largest_success_rectangle([[True] * 3 for _ in range(2)], [-.4, -.2], [-.2, 0., .2])
        self.assertAlmostEqual(result["area_m2"], .08)
        self.assertEqual(result["sample_count"], 6)
        self.assertTrue(result["touches_scan_outer_edge"])

    def test_empty_and_degenerate(self):
        self.assertIsNone(largest_success_rectangle([[False]], [0.], [0.]))
        result = largest_success_rectangle([[True, True, True]], [0.], [0., 1., 2.])
        self.assertEqual(result["area_m2"], 0.)
        self.assertEqual(result["sample_count"], 3)

    def test_interior_rectangle(self):
        result = largest_success_rectangle([
            [False] * 4, [False, True, True, False],
            [False, True, True, False], [False] * 4,
        ], [0., 1., 2., 3.], [0., 1., 2., 3.])
        self.assertEqual(result["area_m2"], 1.)
        self.assertFalse(result["touches_scan_outer_edge"])

    def test_matches_bruteforce_for_random_masks(self):
        rng = random.Random(41)
        for rows in range(1, 6):
            for cols in range(1, 6):
                xs, ys = [r * .02 for r in range(rows)], [c * .03 for c in range(cols)]
                for _ in range(25):
                    mask = [[rng.random() < .7 for _ in ys] for _ in xs]
                    best, key_best = None, None
                    for r0 in range(rows):
                        for r1 in range(r0, rows):
                            for c0 in range(cols):
                                for c1 in range(c0, cols):
                                    if not all(mask[r][c] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
                                        continue
                                    box = rectangle_bounds(xs, ys, r0, r1, c0, c1)
                                    key = (round(box["area_m2"], 14), box["sample_count"],
                                           round(box["x_span_m"] + box["y_span_m"], 14), -r0, -c0)
                                    if key_best is None or key > key_best:
                                        best, key_best = box, key
                    self.assertEqual(largest_success_rectangle(mask, xs, ys), best)

    def test_missing_perimeter_interior_is_unverified(self):
        points = [{"position": [x, y, .03], "grasp_place_endpoint_intersection_angle_indices": [0]}
                  for x in range(3) for y in range(3) if (x, y) != (1, 1)]
        grid = {"x_range": [0, 2], "y_range": [0, 2], "z": .03, "rows": 3, "cols": 3}
        result = summarize_grid(points, grid, ik_only=True)
        self.assertEqual(result["counts"]["unverified"], 1)
        self.assertEqual(result["largest_all_success_sampled_rectangle"]["area_m2"], 0.)
        self.assertFalse(result["success_outer_bounding_box"]["all_bounding_box_samples_successful"])

    def test_ik_grasp_alone_is_not_round_success(self):
        result = summarize_grid([
            {"position": [0, 0, 0], "grasp_and_lift_feasible_angle_indices": [0],
             "grasp_place_endpoint_intersection_angle_indices": []}
        ], {"x_range": [0, 0], "y_range": [0, 0], "z": 0, "rows": 1, "cols": 1}, ik_only=True)
        self.assertEqual(result["counts"]["success"], 0)

    def test_trajectory_requires_raw_positions(self):
        grid = {"x_range": [0, 0], "y_range": [0, 0], "z": 0, "rows": 1, "cols": 1}
        with self.assertRaisesRegex(ValueError, "position_raw"):
            summarize_grid([{"position": [0, 0, 0], "success": True}], grid, ik_only=False)
        result = summarize_grid([{"position_raw": [0, 0, 0], "success": True}], grid, ik_only=False)
        self.assertEqual(result["evidence"], "planner_trajectory_item_success")

    def test_duplicate_and_off_grid_rejected(self):
        grid = {"x_range": [0, 1], "y_range": [0, 1], "z": 0, "rows": 2, "cols": 2}
        point = {"position_raw": [0, 0, 0], "success": True}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_grid([point, point], grid, ik_only=False)
        with self.assertRaisesRegex(ValueError, "outside"):
            summarize_grid([dict(point, position_raw=[.5, 0, 0])], grid, ik_only=False)

    def test_mismatched_mount_rejected_but_grid_density_allowed(self):
        config = {"robot": {"mount_transform": [[1, 0], [0, 1]]},
                  "pick_place": {"grasp_grid": {"rows": 2}}}
        second = copy.deepcopy(config)
        second["pick_place"]["grasp_grid"]["rows"] = 20
        compatible_configs(config, second)
        second["robot"]["mount_transform"][0][0] = -1
        with self.assertRaisesRegex(ValueError, "mount_transform"):
            compatible_configs(config, second)


if __name__ == "__main__":
    unittest.main()
