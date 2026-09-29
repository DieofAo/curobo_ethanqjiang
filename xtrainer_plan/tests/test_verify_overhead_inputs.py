#!/usr/bin/env python3
"""CPU-only regression checks for resolved overhead mounting orientation."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_overhead_trajectory import inspect_inputs  # noqa: E402
from xtrainer_common import rpy_deg_to_matrix  # noqa: E402


class VerifyOverheadInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.result = Path(self.temp.name)
        self.urdf = self.result / "robot.urdf"
        self.urdf.write_text(
            '<robot name="test"><link name="LINK_6"/><link name="TCP_LINK"/>'
            '<joint name="TCP_joint" type="fixed"><parent link="LINK_6"/>'
            '<child link="TCP_LINK"/><origin xyz="0 0 0.19" rpy="0 0 0"/>'
            '</joint></robot>', encoding="utf-8")
        self.names = [f"J_{i}" for i in range(1, 7)]
        np.savez(self.result / "trajectory.npz", joint_names=self.names,
                 positions=np.zeros((2, 6)), times=np.array([0., .02]),
                 ee_positions=np.zeros((2, 3)),
                 ee_quats_wxyz=np.tile([1., 0., 0., 0.], (2, 1)))

    def metadata(self, actual_rpy, expected_rpy):
        mount = np.eye(4)
        mount[:3, :3] = rpy_deg_to_matrix(actual_rpy)
        mount[:3, 3] = [-.41, .15, .50]
        correction = np.linalg.inv(mount).tolist()
        return {
            "n_points": 2,
            "robot": {"joint_names": self.names, "link0_target_transform": correction},
            "config": {
                "robot": {"mount_transform": mount.tolist(), "urdf": str(self.urdf),
                          "ee_link": "TCP_LINK", "base_link": "LINK_0"},
                "pick_place": {"link0_target_transform": correction},
                "overhead": {"mount_rpy_deg": expected_rpy,
                             "urdf_sha256": hashlib.sha256(self.urdf.read_bytes()).hexdigest()},
                "planner": {"self_collision_check": True},
                "workspace": {"wall": {"enable": True,
                                      "collision_link_names": ["LINK_6", "LINK_3"]}},
            },
        }

    def inspect(self, meta):
        (self.result / "trajectory_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return inspect_inputs(self.result)[0]

    def test_side_and_downward_mounts_match_complete_rotation(self):
        for rpy, expected_z in [([90, 0, 90], [1, 0, 0]),
                                ([180, 0, 90], [0, 0, -1])]:
            with self.subTest(rpy=rpy):
                report = self.inspect(self.metadata(rpy, rpy))
                self.assertTrue(all(report["input_checks"].values()), report["input_checks"])
                np.testing.assert_allclose(report["frames"]["base_positive_z_in_old_frame"],
                                           expected_z, atol=1e-12)
                np.testing.assert_allclose(
                    report["frames"]["expected_base_positive_z_in_old_frame"],
                    expected_z, atol=1e-12)

    def test_different_z_mount_is_rejected(self):
        report = self.inspect(self.metadata([180, 0, 90], [90, 0, 90]))
        self.assertFalse(report["input_checks"]["mount_rotation_matches_config"])
        self.assertFalse(report["input_checks"]["new_base_z_matches_config"])

    def test_same_z_wrong_yaw_is_rejected(self):
        report = self.inspect(self.metadata([180, 0, 0], [180, 0, 90]))
        self.assertTrue(report["input_checks"]["new_base_z_matches_config"])
        self.assertFalse(report["input_checks"]["mount_rotation_matches_config"])

    def test_missing_or_invalid_expected_rpy_is_rejected(self):
        for invalid in [None, [], [180, 0], [180, 0, 90, 0], "180 0 90",
                        [180, "0", 90], [180, True, 90], [180, float("nan"), 90],
                        [180, 0, float("inf")]]:
            with self.subTest(expected=invalid):
                with self.assertRaisesRegex(ValueError, "overhead.mount_rpy_deg"):
                    self.inspect(self.metadata([180, 0, 90], invalid))
        for missing in ["overhead", "mount_rpy_deg"]:
            with self.subTest(missing=missing):
                meta = self.metadata([180, 0, 90], [180, 0, 90])
                if missing == "overhead":
                    del meta["config"]["overhead"]
                else:
                    del meta["config"]["overhead"]["mount_rpy_deg"]
                with self.assertRaisesRegex(ValueError, "overhead.mount_rpy_deg"):
                    self.inspect(meta)

    def test_other_safety_checks_remain_active(self):
        meta = self.metadata([180, 0, 90], [180, 0, 90])
        meta["config"]["planner"]["self_collision_check"] = False
        meta["config"]["workspace"]["wall"]["collision_link_names"] = ["LINK_6"]
        report = self.inspect(meta)
        self.assertTrue(report["input_checks"]["mount_rotation_matches_config"])
        self.assertFalse(report["input_checks"]["self_collision_enabled"])
        self.assertFalse(report["input_checks"]["world_collision_enabled_exact_links"])


if __name__ == "__main__":
    unittest.main()
