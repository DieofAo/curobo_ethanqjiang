#!/usr/bin/env python3
"""CPU tests for tilted-mount Cartesian insertion planning."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plan_pick_place as single  # noqa: E402
from xtrainer_common import PoseSpec  # noqa: E402


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeResult:
    def __init__(self, q):
        self.success = types.SimpleNamespace(item=lambda: True)
        self.status = "SUCCESS"
        self.attempts = 1
        self.interpolation_dt = 0.02
        zeros = np.zeros_like(q)
        self.trajectory = types.SimpleNamespace(
            position=FakeTensor(q),
            velocity=FakeTensor(zeros),
            acceleration=FakeTensor(zeros),
        )

    def get_interpolated_plan(self):
        return self.trajectory


class TiltedLinearWaypointsTest(unittest.TestCase):
    def run_sequence(self, lateral_error_m=0.0, angular_error_deg=0.0,
                     direction=None, full_round=False):
        direction = np.asarray(
            [np.sqrt(0.5), 0.0, np.sqrt(0.5)] if direction is None else direction,
            dtype=np.float64,
        )
        lift = PoseSpec("lift", [0.0, 0.0, 0.0], [1, 0, 0, 0], "lift")
        grasp = PoseSpec("grasp", 0.03 * direction, [1, 0, 0, 0], "grasp")
        seq = [lift, grasp]
        if full_round:
            place_lift = np.asarray([0.0, 0.05, 0.0])
            seq.extend([
                PoseSpec("g_lift_out", lift.position, [1, 0, 0, 0], "lift"),
                PoseSpec("p_lift_in", place_lift, [1, 0, 0, 0], "lift"),
                PoseSpec("place", place_lift + 0.03 * direction,
                         [1, 0, 0, 0], "place"),
                PoseSpec("p_lift_out", place_lift, [1, 0, 0, 0], "lift"),
            ])
        targets = []

        def fake_plan(_mg, q_start, target, _pl, pose_metric=None):
            self.assertIsNone(pose_metric)
            targets.append(target)
            q_end = np.asarray(q_start, dtype=np.float64).copy()
            q_end[:3] = target.position
            if target.name == "grasp_linear_step1":
                q_end[1] += lateral_error_m
            return FakeResult(np.stack([q_start, q_end]))

        def fake_fk(_mg, q, _links):
            quat = np.tile([1.0, 0.0, 0.0, 0.0], (len(q), 1))
            if angular_error_deg:
                quat[2] = [np.cos(np.deg2rad(angular_error_deg) / 2),
                           np.sin(np.deg2rad(angular_error_deg) / 2), 0.0, 0.0]
            return {"ee/pos": np.asarray(q)[:, :3], "ee/quat": quat}

        lin = {
            "enable": True, "kinds": ["grasp", "place"], "method": "waypoints_fk",
            "free_axis": "vector", "free_direction_root": direction.tolist(),
            "waypoint_step_m": 0.0075, "max_deviation_mm": 3.0,
            "max_rotation_deg": 5.0, "hold_rotation": True,
        }
        with mock.patch.object(single, "plan_segment", side_effect=fake_plan), \
             mock.patch.object(single, "compute_fk", side_effect=fake_fk):
            ok, result = single.plan_sequence(
                None, np.zeros(6), seq,
                {"interpolation_dt": 0.02},
                {"joints": [1, 2, 3, 4, 5], "max_joint_delta_deg": 170.0},
                np.full(6, -3.0), np.full(6, 3.0), lin, "TCP_LINK",
            )
        return ok, result, targets, direction

    def test_oblique_path_keeps_four_waypoints_and_one_external_stage(self):
        ok, result, targets, direction = self.run_sequence()
        self.assertTrue(ok)
        self.assertEqual(len(targets), 5)  # lift plus four grasp subsegments
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["segments"][1]["to"], "grasp")
        self.assertEqual(result["segments"][1]["linear"]["subsegments"], 4)
        self.assertEqual(result["segments"][1]["attempts"], 4)
        self.assertEqual(result["position"].shape[0], 6)
        for index, target in enumerate(targets[1:], start=1):
            np.testing.assert_allclose(target.position, 0.03 * index / 4 * direction)
            np.testing.assert_allclose(target.quat_wxyz, [1, 0, 0, 0])

    def test_full_pick_place_keeps_six_external_stages(self):
        ok, result, targets, _ = self.run_sequence(full_round=True)
        self.assertTrue(ok)
        self.assertEqual(len(targets), 12)
        self.assertEqual(
            [stage["to"] for stage in result["segments"]],
            ["lift", "grasp", "g_lift_out", "p_lift_in", "place", "p_lift_out"],
        )
        self.assertEqual(
            [stage["linear"]["subsegments"] for stage in result["segments"]
             if "linear" in stage],
            [4, 4],
        )
        self.assertEqual(
            sum(stage["n_points"] - 1 for stage in result["segments"]) + 1,
            result["position"].shape[0],
        )

    def test_same_waypoint_method_accepts_vertical_control(self):
        ok, result, targets, direction = self.run_sequence(direction=[0, 0, -1])
        self.assertTrue(ok)
        self.assertEqual(len(targets), 5)
        self.assertEqual(result["segments"][1]["linear"]["method"], "waypoints_fk")
        np.testing.assert_allclose(targets[-1].position, 0.03 * direction)

    def test_whole_stage_rejects_lateral_or_angular_drift(self):
        ok, result, _, _ = self.run_sequence(lateral_error_m=0.004)
        self.assertFalse(ok)
        self.assertTrue(result["status"].startswith("NOT_STRAIGHT"))
        ok, result, _, _ = self.run_sequence(angular_error_deg=6.0)
        self.assertFalse(ok)
        self.assertTrue(result["status"].startswith("ROT_DRIFT"))


if __name__ == "__main__":
    unittest.main()
