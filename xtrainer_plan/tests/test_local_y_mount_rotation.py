#!/usr/bin/env python3
"""Focused geometry checks for original-base-local-Y mount rotations."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from derive_local_y_overhead_experiment import remount_from_task_local_y  # noqa: E402
from summarize_local_y_mount_sweep import ry, validate_config  # noqa: E402


class LocalYMountRotationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / (
            "results_overhead/20260928/v61_joint_home_source.json")
        cls.source = json.loads(source.read_text(encoding="utf-8"))

    def test_original_base_local_y_is_world_x_and_differs_from_world_y(self):
        source = copy.deepcopy(self.source)
        baseline = np.asarray(source["robot"]["mount_transform"], dtype=float)
        r0 = baseline[:3, :3]
        np.testing.assert_allclose(r0 @ [0., 1., 0.], [1., 0., 0.], atol=1e-12)
        position = [-.20, .15, .65]
        original_target = np.asarray(source["overhead"]["original_target_transform"], dtype=float)
        for degrees in (0., 30., 45., 60.):
            with self.subTest(degrees=degrees):
                cfg = remount_from_task_local_y(source, position, degrees)
                mount = np.asarray(cfg["robot"]["mount_transform"], dtype=float)
                target_in_base = np.asarray(cfg["pick_place"]["link0_target_transform"], dtype=float)
                np.testing.assert_allclose(mount[:3, :3], r0 @ ry(degrees), atol=1e-12)
                np.testing.assert_allclose(mount[:3, 3], position, atol=1e-12)
                np.testing.assert_allclose(mount @ target_in_base, original_target, atol=1e-9)
                np.testing.assert_allclose(mount[:3, :3] @ [0., 1., 0.], [1., 0., 0.], atol=1e-12)
                self.assertEqual(cfg["pick_place"]["home"], source["pick_place"]["home"])
                self.assertEqual(cfg["overhead"]["tilt_axis"], "original_base_local_y")
                self.assertEqual(cfg["overhead"]["local_y_tilt_deg"], degrees)
                if degrees:
                    self.assertGreater(np.linalg.norm(r0 @ ry(degrees) - ry(degrees) @ r0), .1)
        self.assertEqual(source, self.source)

    def test_summary_accepts_local_rotation_and_rejects_world_rotation(self):
        source = self.source
        cfg = remount_from_task_local_y(source, [-.20, .15, .65], 60.)
        cfg["pick_place"]["grasp_grid"].update(rows=3, cols=3)
        cfg["output"]["dir"] = "/tmp/local_y_test_result"
        row = {"base_xyz_m": [-.20, .15, .65], "local_y_tilt_deg": 60.,
               "tilt_axis": "original_base_local_y"}
        validate_config(source, cfg, row, Path(cfg["output"]["dir"]), 3)
        baseline = np.asarray(source["robot"]["mount_transform"], dtype=float)
        bad_mount = np.eye(4)
        bad_mount[:3, :3] = ry(60.) @ baseline[:3, :3]
        bad_mount[:3, 3] = [-.20, .15, .65]
        cfg["robot"]["mount_transform"] = bad_mount.tolist()
        with self.assertRaisesRegex(ValueError, "original base local"):
            validate_config(source, cfg, row, Path(cfg["output"]["dir"]), 3)


if __name__ == "__main__":
    unittest.main()
