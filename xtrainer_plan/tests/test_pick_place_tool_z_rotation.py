#!/usr/bin/env python3
"""CPU-only regressions for fixed local-TCP-Z rotation of place candidates."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plan_pick_place as planner  # noqa: E402


def config():
    return {"robot": {"joint_limit_clip": .14}, "workspace": {}, "planner": {}, "output": {},
            "pick_place": {
                "home": {"position": [-.31, 0., .03], "rpy_deg": [-180., 0., 0.]},
                "place": {"position": [-.36, -.12, .1]},
                "grasp_grid": {}, "criterion": {},
                "base_rpy": {"grasp": [-180., 0., -180.], "place": [-90., 0., -150.]},
                "lift": {"grasp_z": .03, "place_z": .03, "axis": "base_z"},
                "angle_search": {
                    "grasp": {"axis": "x", "min_deg": -30., "max_deg": 30., "step_deg": 2.},
                    "place": {"axis": "x", "min_deg": -30., "max_deg": 30., "step_deg": 5.},
                    "order": "asc", "couple_place_to_grasp": True, "reuse_last_success": False,
                    "max_trials": 1200, "stage2": {"enable": False, "frame": "base", "axis": "z"}}}}


def poses(pp, g=-30., p=-30., g2=0., p2=0.):
    return planner.make_round_poses([-.41, .15, .03], pp["place"]["position"], g, p, pp, 7, g2, p2)


def rotation(pose):
    return planner.quat_wxyz_to_matrix(pose.quat_wxyz)


class PlaceToolZRotationTest(unittest.TestCase):
    def assert_postrotation(self, old, new, angle=180.):
        delta = planner.quat_wxyz_to_matrix(planner.rpy_deg_to_quat_wxyz([0., 0., angle]))
        self.assertEqual([p.name for p in old], [p.name for p in new])
        for index, (a, b) in enumerate(zip(old, new)):
            np.testing.assert_allclose(a.position, b.position, atol=1e-14, rtol=0)
            if index < 3:
                np.testing.assert_array_equal(a.quat_wxyz, b.quat_wxyz)
            else:
                np.testing.assert_allclose(rotation(b), rotation(a) @ delta, atol=2e-14, rtol=0)
                np.testing.assert_allclose(rotation(b)[:, 2], rotation(a)[:, 2], atol=2e-14, rtol=0)
                if angle == 180.:
                    np.testing.assert_allclose(rotation(b)[:, :2], -rotation(a)[:, :2], atol=2e-14, rtol=0)

    def test_all_31_candidates_keep_pairing_order_grasp_home_positions_and_clip(self):
        cfg = config()
        before = copy.deepcopy(cfg)
        pp = cfg["pick_place"]
        changed = copy.deepcopy(pp)
        changed["place"]["tool_z_rotation_deg"] = 180.
        expected = [(float(a), float(a)) for a in range(-30, 31, 2)]
        self.assertEqual(planner.angle_combos(pp["angle_search"]), expected)
        self.assertEqual(planner.angle_combos(changed["angle_search"]), expected)
        for g, p in expected:
            with self.subTest(g=g):
                self.assert_postrotation(poses(pp, g, p), poses(changed, g, p))
        self.assertEqual(cfg, before)
        self.assertEqual(changed["home"], pp["home"])
        self.assertEqual(cfg["robot"]["joint_limit_clip"], .14)
        expected_pp = copy.deepcopy(pp)
        expected_pp["place"]["tool_z_rotation_deg"] = 180.
        self.assertEqual(changed, expected_pp)

    def test_missing_and_explicit_zero_preserve_legacy_quaternions_exactly(self):
        pp = config()["pick_place"]
        for explicit in (False, True):
            for frame in ("base", "tool"):
                changed = copy.deepcopy(pp)
                changed["angle_search"]["stage2"]["frame"] = frame
                if explicit:
                    changed["place"]["tool_z_rotation_deg"] = 0.
                for g, p in planner.angle_combos(changed["angle_search"]):
                    result = poses(changed, g, p)
                    compose2 = planner.compose_left if frame == "base" else planner.compose_right
                    for side, angle, index in (("grasp", g, 1), ("place", p, 4)):
                        # Original two-stage expression, without a final extra compose.
                        original = compose2(planner.compose_right(
                            planner.rpy_deg_to_quat_wxyz(changed["base_rpy"][side]),
                            planner.axis_delta("x", angle)), planner.axis_delta("z", 0.))
                        original = planner.PoseSpec("legacy", result[index].position, original,
                                                    result[index].kind).quat_wxyz
                        np.testing.assert_array_equal(result[index].quat_wxyz, original)

    def test_local_rotation_remains_right_multiplication_after_task_transform(self):
        pp = config()["pick_place"]
        changed = copy.deepcopy(pp)
        changed["place"]["tool_z_rotation_deg"] = 180.
        correction = np.array([[0., 1., 0., -.4], [1., 0., 0., .31],
                               [0., 0., -1., .65], [0., 0., 0., 1.]])
        for g, p in planner.angle_combos(pp["angle_search"]):
            old = [planner.transform_pose(pose, correction) for pose in poses(pp, g, p)]
            new = [planner.transform_pose(pose, correction) for pose in poses(changed, g, p)]
            self.assert_postrotation(old, new)

    def test_offset_follows_both_stage2_frames_and_every_stage2_axis(self):
        for frame in ("base", "tool"):
            for axis in ("x", "y", "z"):
                with self.subTest(frame=frame, axis=axis):
                    pp = config()["pick_place"]
                    pp["angle_search"]["stage2"].update(enable=True, frame=frame, axis=axis)
                    changed = copy.deepcopy(pp)
                    changed["place"]["tool_z_rotation_deg"] = 180.
                    self.assert_postrotation(poses(pp, 18., -12., 16., -22.),
                                             poses(changed, 18., -12., 16., -22.))

    def test_general_finite_angles_and_tool_axis_lifts_preserve_positions(self):
        for angle in (-180., 90., 360., 17.5):
            pp = config()["pick_place"]
            pp["lift"]["axis"] = "tool_z_neg"
            changed = copy.deepcopy(pp)
            changed["place"]["tool_z_rotation_deg"] = angle
            self.assert_postrotation(poses(pp), poses(changed), angle)

    def test_nonfinite_and_invalid_angles_rejected_by_pose_builder(self):
        for angle in (float("nan"), float("inf"), -float("inf"), None, True, [], "not-an-angle"):
            with self.subTest(angle=angle):
                pp = config()["pick_place"]
                pp["place"]["tool_z_rotation_deg"] = angle
                with self.assertRaisesRegex(ValueError, "tool_z_rotation_deg must be a finite number"):
                    poses(pp)

    def test_main_rejects_nonfinite_before_output_world_robot_or_gpu(self):
        for angle in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(angle=angle):
                cfg = config()
                cfg["pick_place"]["place"]["tool_z_rotation_deg"] = angle
                with mock.patch.object(sys, "argv", ["plan_pick_place.py"]), \
                        mock.patch.object(planner, "load_pick_place_config", return_value=cfg), \
                        mock.patch.object(planner, "make_world_config") as world, \
                        mock.patch.object(planner, "load_robot_cfg_dict") as robot, \
                        mock.patch.object(planner, "make_motion_gen") as gpu, \
                        mock.patch.object(planner.Path, "mkdir") as mkdir, \
                        redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(planner.main(), 6)
                    self.assertIn("tool_z_rotation_deg", output.getvalue())
                    world.assert_not_called()
                    robot.assert_not_called()
                    gpu.assert_not_called()
                    mkdir.assert_not_called()

    def test_cli_only_changes_optional_place_offset_and_old_namespace_works(self):
        cfg = config()
        expected = copy.deepcopy(cfg)
        args = planner.build_argparser().parse_args(["--place-tool-z-rotation-deg", "180"])
        expected["pick_place"]["place"]["tool_z_rotation_deg"] = 180.
        self.assertEqual(planner.apply_cli(cfg, args), expected)
        default_args = planner.build_argparser().parse_args([])
        self.assertIsNone(default_args.place_tool_z_rotation_deg)
        del default_args.place_tool_z_rotation_deg
        original = config()
        self.assertEqual(planner.apply_cli(copy.deepcopy(original), default_args), original)


if __name__ == "__main__":
    unittest.main()
