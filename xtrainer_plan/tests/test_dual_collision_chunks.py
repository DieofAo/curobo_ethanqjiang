#!/usr/bin/env python3
"""CPU-only tests for chunked dual-arm collision reporting."""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np


TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "scripts"))

import check_dual_arm_collision as collision  # noqa: E402


def reference_report(sph_a, sph_b, names_a, names_b, margin_m):
    """Original all-at-once implementation used as a numerical oracle."""
    n_pts, n_sph = sph_a.shape[:2]
    ca, ra = sph_a[..., :3], sph_a[..., 3]
    cb, rb = sph_b[..., :3], sph_b[..., 3]
    distance = np.linalg.norm(
        ca[:, :, None, :] - cb[:, None, :, :], axis=-1
    )
    clearance = distance - (ra[:, :, None] + rb[:, None, :])
    hit = clearance < margin_m

    flat = int(np.argmin(clearance))
    worst_idx, worst_a, worst_b = np.unravel_index(flat, clearance.shape)
    pair_counter = Counter()
    _, hit_a, hit_b = np.nonzero(hit)
    for a, b in zip(hit_a.tolist(), hit_b.tolist()):
        pair_counter[(names_a[a], names_b[b])] += 1

    hit_idx = np.nonzero(hit.any(axis=(1, 2)))[0]
    return {
        "n_points": int(n_pts),
        "n_spheres_per_arm": int(n_sph),
        "n_collision_points": int(hit_idx.size),
        "collision_indices": hit_idx.tolist(),
        "min_clearance_mm": float(clearance.min()) * 1000.0,
        "worst_point": {
            "index": int(worst_idx),
            "link_pair": [names_a[worst_a], names_b[worst_b]],
            "clearance_mm": float(
                clearance[worst_idx, worst_a, worst_b]
            ) * 1000.0,
        },
        "top_link_pairs": [
            {"pair": list(pair), "n_sphere_hits": count}
            for pair, count in pair_counter.most_common(10)
        ],
        "clearance_min_per_point_mm": (
            clearance.min(axis=(1, 2)) * 1000.0
        ).tolist(),
    }


class ChunkedPairCollisionTest(unittest.TestCase):
    def test_invalid_sphere_inputs_raise_instead_of_reporting_safe(self) -> None:
        sph_a = np.zeros((2, 1, 4), dtype=np.float64)
        sph_b = np.zeros((2, 1, 4), dtype=np.float64)
        sph_a[..., 3] = 0.1
        sph_b[..., 3] = 0.1

        nan_spheres = sph_a.copy()
        nan_spheres[0, 0, 1] = np.nan
        negative_radius = sph_a.copy()
        negative_radius[0, 0, 3] = -0.1
        cases = (
            ("nan", nan_spheres, sph_b, ["a"], ["b"], 0.0),
            ("negative_radius", negative_radius, sph_b, ["a"], ["b"], 0.0),
            ("name_mismatch", sph_a, sph_b, [], ["b"], 0.0),
            ("shape_mismatch", sph_a[:1], sph_b, ["a"], ["b"], 0.0),
            ("nonfinite_margin", sph_a, sph_b, ["a"], ["b"], np.inf),
        )

        for label, arm_a, arm_b, names_a, names_b, margin in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    collision.check_pair_collisions(
                        arm_a, arm_b, names_a, names_b, margin
                    )

    def test_multiple_chunks_match_original_report(self) -> None:
        rng = np.random.default_rng(20260907)
        n_points, n_a, n_b = 9, 3, 4
        sph_a = np.empty((n_points, n_a, 4), dtype=np.float64)
        sph_b = np.empty((n_points, n_b, 4), dtype=np.float64)
        sph_a[..., :3] = rng.normal(0.0, 0.35, (n_points, n_a, 3))
        sph_b[..., :3] = rng.normal(0.0, 0.35, (n_points, n_b, 3))
        sph_a[..., 3] = rng.uniform(0.08, 0.22, (n_points, n_a))
        sph_b[..., 3] = rng.uniform(0.08, 0.22, (n_points, n_b))
        names_a = ["arm_a", "arm_a", "wrist_a"]
        names_b = ["arm_b", "wrist_b", "wrist_b", "tool_b"]
        margin_m = 0.025

        expected = reference_report(
            sph_a, sph_b, names_a, names_b, margin_m
        )
        with mock.patch.object(collision, "PAIR_COLLISION_CHUNK_SIZE", 2):
            actual = collision.check_pair_collisions(
                sph_a, sph_b, names_a, names_b, margin_m
            )

        self.assertEqual(actual["n_points"], expected["n_points"])
        self.assertEqual(
            actual["n_spheres_per_arm"], expected["n_spheres_per_arm"]
        )
        self.assertEqual(
            actual["n_collision_points"], expected["n_collision_points"]
        )
        self.assertEqual(
            actual["collision_indices"], expected["collision_indices"]
        )
        self.assertEqual(actual["top_link_pairs"], expected["top_link_pairs"])
        self.assertEqual(
            actual["worst_point"]["index"], expected["worst_point"]["index"]
        )
        self.assertEqual(
            actual["worst_point"]["link_pair"],
            expected["worst_point"]["link_pair"],
        )
        self.assertAlmostEqual(
            actual["min_clearance_mm"], expected["min_clearance_mm"], places=12
        )
        self.assertAlmostEqual(
            actual["worst_point"]["clearance_mm"],
            expected["worst_point"]["clearance_mm"],
            places=12,
        )
        np.testing.assert_allclose(
            actual["clearance_min_per_point_mm"],
            expected["clearance_min_per_point_mm"],
            rtol=0.0,
            atol=1e-12,
        )

    def test_known_collisions_and_equal_minimum_across_chunk_boundary(self) -> None:
        # Two 0.5 m-radius spheres per arm.  Points 3 and 4 have the same
        # global minimum; the report must retain point 3, matching np.argmin.
        n_points = 5
        sph_a = np.zeros((n_points, 2, 4), dtype=np.float64)
        sph_b = np.zeros((n_points, 2, 4), dtype=np.float64)
        sph_a[:, 0, 0] = 0.0
        sph_a[:, 1, 0] = 10.0
        sph_b[:, :, 0] = np.array([
            [3.0, 13.0],
            [0.75, 13.0],
            [3.0, 10.5],
            [0.6, 10.25],
            [3.0, 10.25],
        ])
        sph_a[..., 3] = 0.5
        sph_b[..., 3] = 0.5

        with mock.patch.object(collision, "PAIR_COLLISION_CHUNK_SIZE", 2):
            report = collision.check_pair_collisions(
                sph_a, sph_b, ["a0", "a1"], ["b0", "b1"], 0.0
            )

        self.assertEqual(report["collision_indices"], [1, 2, 3, 4])
        self.assertEqual(report["n_collision_points"], 4)
        self.assertEqual(report["min_clearance_mm"], -750.0)
        self.assertEqual(
            report["worst_point"],
            {
                "index": 3,
                "link_pair": ["a1", "b1"],
                "clearance_mm": -750.0,
            },
        )
        self.assertEqual(
            report["top_link_pairs"],
            [
                {"pair": ["a1", "b1"], "n_sphere_hits": 3},
                {"pair": ["a0", "b0"], "n_sphere_hits": 2},
            ],
        )
        np.testing.assert_allclose(
            report["clearance_min_per_point_mm"],
            [2000.0, -250.0, -500.0, -750.0, -750.0],
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
