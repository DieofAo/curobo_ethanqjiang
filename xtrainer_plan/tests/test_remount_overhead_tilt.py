#!/usr/bin/env python3
"""CPU regressions for world-frame Y tilt and exact workspace geometry."""
import argparse
import copy
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from derive_overhead_experiment import remount_from_task, translate_mount
from plan_pick_place import load_pick_place_config
from prepare_overhead_config import prepare
from xtrainer_common import (
    build_workspace_wall_cuboids, check_in_bounds, quat_wxyz_to_matrix,
    rpy_deg_to_matrix,
)

try:
    from play_trajectory_ros import build_static_markers
except ImportError:
    build_static_markers = None


class RemountOverheadTiltTest(unittest.TestCase):
    def setUp(self):
        self.source = prepare(load_pick_place_config(), [-.31, .40, .65],
                              mount_rpy=[180., 0., 90.])
        self.old_mount = np.asarray(self.source["robot"]["mount_transform"])
        self.original_c = self.old_mount @ np.asarray(
            self.source["pick_place"]["link0_target_transform"])
        self.task_walls = build_workspace_wall_cuboids(
            self.source["overhead"]["task_workspace"])

    def assert_physical_geometry(self, cfg):
        mount = np.asarray(cfg["robot"]["mount_transform"])
        boxes = build_workspace_wall_cuboids(cfg["workspace"])
        self.assertEqual(len(boxes), len(self.task_walls))

        def corners(box, frame):
            pose = np.asarray(box["pose"])
            local = np.array(list(itertools.product((-1., 1.), repeat=3)))
            points = (local * np.asarray(box["dims"]) / 2) @ (
                frame[:3, :3] @ quat_wxyz_to_matrix(pose[3:])).T
            points += frame[:3, :3] @ pose[:3] + frame[:3, 3]
            return tuple(sorted(map(tuple, np.round(points, 10))))

        actual = {corners(box, mount) for box in boxes}
        expected = {corners(box, np.eye(4)) for box in self.task_walls}
        self.assertEqual(actual, expected)

    def test_full_cartesian_product_has_fixed_mount_and_exact_world(self):
        saved = copy.deepcopy(self.source)
        task_bounds = self.source["overhead"]["task_workspace"]
        task_points = np.array([
            [-.5, .0, .03], [-.1, .5, .1], [-1.95, -1.2, 1.65],
            [.05, .0, .03], [-.5, 1.5, .03], [-.5, .0, 1.8],
        ])
        expected_inside, expected_violation = check_in_bounds(task_points, task_bounds)
        for x, y, z, angle in itertools.product(
                (-.20, -.10), (.15, .35, .45), (.45, .55, .65),
                (0., 30., 45., 60.)):
            with self.subTest(x=x, y=y, z=z, angle=angle):
                cfg = remount_from_task(self.source, [x, y, z], angle)
                mount = np.asarray(cfg["robot"]["mount_transform"])
                correction = np.asarray(cfg["pick_place"]["link0_target_transform"])
                expected_rotation = rpy_deg_to_matrix([0., angle, 0.]) @ self.old_mount[:3, :3]
                np.testing.assert_allclose(mount[:3, :3], expected_rotation, atol=1e-12)
                np.testing.assert_allclose(mount[:3, 3], [x, y, z], atol=1e-12)
                np.testing.assert_allclose(mount @ correction, self.original_c, atol=1e-12)
                self.assert_physical_geometry(cfg)
                base_points = (task_points - mount[:3, 3]) @ mount[:3, :3]
                inside, violation = check_in_bounds(base_points, cfg["workspace"])
                np.testing.assert_array_equal(inside, expected_inside)
                np.testing.assert_allclose(violation, expected_violation, atol=1e-12)
                self.assertEqual(cfg["pick_place"]["linear_move"]["method"], "waypoints_fk")
                self.assertEqual(cfg["pick_place"]["linear_move"]["waypoint_step_m"], .0075)
                if angle:
                    self.assertIn("oriented_bounds", cfg["workspace"])
                    self.assertEqual(len(cfg["workspace"]["wall"]["cuboids_override"]),
                                     len(self.task_walls))
        self.assertEqual(self.source, saved)

    def test_shared_joint_home_is_preserved_when_remounting(self):
        source = copy.deepcopy(self.source)
        source["pick_place"]["home"]["joint_deg"] = [-55., -74., -95., 80., -90., 55.]
        saved = copy.deepcopy(source)
        cfg = remount_from_task(source, [-.10, .45, .55], 60.)
        self.assertEqual(cfg["pick_place"]["home"], saved["pick_place"]["home"])
        self.assertEqual(source, saved)
        self.assert_physical_geometry(cfg)

    def test_oriented_bounds_reject_point_inside_display_aabb(self):
        cfg = remount_from_task(self.source, [-.20, .35, .55], 45.)
        frame = np.asarray(cfg["workspace"]["oriented_bounds"]["frame_transform"])
        # The exterior point is inside the broad display AABB for this tilt.
        task_point = np.array([.02, 0., .80])
        base_point = frame[:3, :3] @ task_point + frame[:3, 3]
        b = cfg["workspace"]["bounds"]
        self.assertTrue(all(b[a][0] < base_point[i] < b[a][1]
                            for i, a in enumerate("xyz")))
        inside, violation = check_in_bounds(base_point[None, :], cfg["workspace"])
        self.assertFalse(inside[0])
        self.assertGreater(violation[0], 0.)

    def test_translating_tilted_config_preserves_oriented_geometry(self):
        tilted = remount_from_task(self.source, [-.20, .35, .55], 45.)
        translated = translate_mount(tilted, [-.10, .45, .65])
        self.assert_physical_geometry(translated)
        task_point = np.array([-.25, .2, .03])
        for cfg in (tilted, translated):
            mount = np.asarray(cfg["robot"]["mount_transform"])
            base_point = (task_point - mount[:3, 3]) @ mount[:3, :3]
            inside, violation = check_in_bounds(base_point[None, :], cfg["workspace"])
            self.assertTrue(inside[0])
            self.assertEqual(violation[0], 0.)

    @unittest.skipIf(build_static_markers is None, "ROS marker modules unavailable")
    def test_rviz_markers_show_rotated_walls_and_real_workspace(self):
        cfg = remount_from_task(self.source, [-.20, .35, .55], 45.)
        mount = np.asarray(cfg["robot"]["mount_transform"])
        meta = {
            "workspace": cfg["workspace"],
            "wall_cuboids": build_workspace_wall_cuboids(cfg["workspace"]),
        }
        markers = build_static_markers(
            meta, "LINK_0", None,
            argparse.Namespace(show_walls=True, wall_alpha=.2)).markers
        outline = next(marker for marker in markers if marker.ns == "workspace"
                       and len(marker.points) == 24)
        task_points = {
            tuple(np.round(mount[:3, :3] @ [p.x, p.y, p.z] + mount[:3, 3], 10))
            for p in outline.points
        }
        b = self.source["overhead"]["task_workspace"]["bounds"]
        expected = {
            tuple(np.round(p, 10))
            for p in itertools.product(*(b[a] for a in "xyz"))
        }
        self.assertEqual(task_points, expected)
        walls = [marker for marker in markers if marker.ns == "walls"]
        self.assertEqual(len(walls), len(self.task_walls))
        for old, marker in zip(self.task_walls, walls):
            center = np.array([marker.pose.position.x, marker.pose.position.y,
                               marker.pose.position.z])
            quat = [marker.pose.orientation.w, marker.pose.orientation.x,
                    marker.pose.orientation.y, marker.pose.orientation.z]
            np.testing.assert_allclose(
                mount[:3, :3] @ center + mount[:3, 3], old["pose"][:3],
                atol=1e-12)
            np.testing.assert_allclose(
                mount[:3, :3] @ quat_wxyz_to_matrix(quat), np.eye(3),
                atol=1e-12)
            np.testing.assert_allclose(
                [marker.scale.x, marker.scale.y, marker.scale.z],
                old["dims"], atol=1e-12)

    def test_invalid_input_rejected_without_source_mutation(self):
        saved = copy.deepcopy(self.source)
        for position, angle in (([0., 0.], 30.), ([0., 0., math.nan], 30.),
                                ([0., 0., 0.], math.inf)):
            with self.assertRaises(ValueError):
                remount_from_task(self.source, position, angle)
        tilted = remount_from_task(self.source, [-.20, .35, .55], 45.)
        with self.assertRaisesRegex(ValueError, "compound"):
            remount_from_task(tilted, [-.10, .45, .65], 30.)
        self.assertEqual(self.source, saved)


if __name__ == "__main__":
    unittest.main()
