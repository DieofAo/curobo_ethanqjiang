#!/usr/bin/env python3
"""CPU-only boundary checks for finite grasp-plane sphere measurements."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_link3_grasp_clearance import sphere_rectangle_metrics  # noqa: E402


class Link3GraspClearanceTest(unittest.TestCase):
    def metrics(self, center, radius=0.1, plane_z=0.03):
        return {key: value[0, 0] for key, value in sphere_rectangle_metrics(
            np.asarray([[center]], dtype=float), np.asarray([radius]),
            [-0.5, 0.5], [-0.5, 0.5], plane_z).items()}

    def test_xy_overlap_is_not_a_three_dimensional_intersection(self):
        result = self.metrics([0, 0, 0.5])
        self.assertTrue(result["xy_projection_overlap"])
        self.assertAlmostEqual(result["xy_signed_clearance_m"], -0.6)
        self.assertAlmostEqual(result["lowest_surface_z_m"], 0.4)
        self.assertAlmostEqual(result["height_above_grasp_plane_over_rectangle_m"], 0.37)
        self.assertAlmostEqual(result["sphere_to_finite_grasp_plane_clearance_m"], 0.37)

    def test_corner_distance_and_restricted_spherical_cap(self):
        # The center is 3 cm beyond one edge and 4 cm beyond the other:
        # the closest XY point is the rectangle corner, 5 cm away.
        result = self.metrics([0.53, 0.54, 0.2])
        self.assertTrue(result["xy_projection_overlap"])
        self.assertAlmostEqual(result["xy_signed_clearance_m"], -0.05)
        self.assertAlmostEqual(result["lowest_surface_over_rectangle_z_m"],
                               0.2 - np.sqrt(0.1 ** 2 - 0.05 ** 2))
        self.assertAlmostEqual(result["sphere_to_finite_grasp_plane_clearance_m"],
                               np.hypot(0.05, 0.17) - 0.1)
        self.assertGreater(result["lowest_surface_over_rectangle_z_m"],
                           result["lowest_surface_z_m"])

    def test_xy_disjoint_has_no_restricted_surface_height(self):
        result = self.metrics([0.8, 0, 0.03])
        self.assertFalse(result["xy_projection_overlap"])
        self.assertTrue(np.isnan(result["lowest_surface_over_rectangle_z_m"]))
        self.assertTrue(np.isnan(result["height_above_grasp_plane_over_rectangle_m"]))
        self.assertAlmostEqual(result["sphere_to_finite_grasp_plane_clearance_m"], 0.2)

    def test_plane_contact_includes_tangency(self):
        # Binary-exact coordinates avoid classifying float roundoff as a gap.
        result = self.metrics([0, 0, 0.125], radius=0.125, plane_z=0)
        self.assertEqual(result["sphere_to_finite_grasp_plane_clearance_m"], 0)
        result = self.metrics([0.625, 0, 0], radius=0.125, plane_z=0)
        self.assertTrue(result["xy_projection_overlap"])
        self.assertEqual(result["lowest_surface_over_rectangle_z_m"], 0)
        self.assertEqual(result["sphere_to_finite_grasp_plane_clearance_m"], 0)

    def test_sphere_entirely_below_plane_is_not_plane_intersection(self):
        result = self.metrics([0, 0, -0.3])
        self.assertLess(result["height_above_grasp_plane_over_rectangle_m"], 0)
        self.assertGreater(result["sphere_to_finite_grasp_plane_clearance_m"], 0)

    def test_multiple_frames_and_per_sphere_radii_broadcast(self):
        centers = np.asarray([[[0, 0, 0.3], [0, 0, 0.3]],
                              [[0.8, 0, 0.3], [0.8, 0, 0.3]]])
        result = sphere_rectangle_metrics(centers, np.asarray([0.1, 0.2]),
                                          [-0.5, 0.5], [-0.5, 0.5], 0.03)
        for value in result.values():
            self.assertEqual(value.shape, (2, 2))
        np.testing.assert_allclose(result["lowest_surface_z_m"], [[0.2, 0.1], [0.2, 0.1]])
        np.testing.assert_array_equal(result["xy_projection_overlap"],
                                      [[True, True], [False, False]])


if __name__ == "__main__":
    unittest.main()
