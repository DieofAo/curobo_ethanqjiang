#!/usr/bin/env python3
"""CPU-only regression tests for the accepted overhead mounting frame.

These tests construct poses and collision boxes, but do not import torch or run IK.
"""
import copy
import itertools
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plan_pick_place import (  # noqa: E402
    load_pick_place_config, make_round_poses, transform_pose,
)
from prepare_overhead_config import prepare, transform_workspace  # noqa: E402
from xtrainer_common import (  # noqa: E402
    REPO_ROOT, PoseSpec, build_workspace_wall_cuboids,
    parse_rigid_transform_matrix, quat_wxyz_to_matrix, rpy_deg_to_matrix,
    rpy_deg_to_quat_wxyz,
)


class OverheadConfigTest(unittest.TestCase):
    # Accepted by the user: new +Z points along old +X, new +Y along old +Z.
    MOUNT_RPY = [90.0, 0.0, 90.0]
    MOUNTS = [(-.41, .10, .45), (-.41, .10, .55), (-.41, .10, .75),
              (-.51, -.05, .65), (-.31, .20, .85)]

    def setUp(self):
        self.source = load_pick_place_config()

    def mounted(self, source=None, position=None):
        return prepare(self.source if source is None else source,
                       self.MOUNTS[1] if position is None else position,
                       mount_rpy=self.MOUNT_RPY)

    def assert_same_pose(self, first, second):
        np.testing.assert_allclose(first.position, second.position, atol=1e-12)
        np.testing.assert_allclose(quat_wxyz_to_matrix(first.quat_wxyz),
                                   quat_wxyz_to_matrix(second.quat_wxyz), atol=1e-12)
        self.assertEqual(first.name, second.name)
        self.assertEqual(first.kind, second.kind)

    def test_accepted_axis_directions_and_frame_names(self):
        cfg = self.mounted()
        mount = np.asarray(cfg["robot"]["mount_transform"])
        np.testing.assert_allclose(mount[:3, 0], [0., 1., 0.], atol=1e-12)
        np.testing.assert_allclose(mount[:3, 1], [0., 0., 1.], atol=1e-12)
        np.testing.assert_allclose(mount[:3, 2], [1., 0., 0.], atol=1e-12)
        np.testing.assert_allclose(mount[:3, 3], self.MOUNTS[1], atol=1e-12)
        self.assertEqual(cfg["robot"]["task_frame"], "task_world")
        self.assertEqual(cfg["robot"]["legacy_task_frame"], "original_LINK_0")
        self.assertEqual(cfg["robot"]["base_link"], "LINK_0")

    def test_round_poses_keep_physical_targets_at_all_mounts(self):
        pp = self.source["pick_place"]
        grid = pp["grasp_grid"]
        # Cover the rectangle's corners and center, all allowed first-stage
        # angles, and both supported second-stage conventions.
        points = [[x, y, grid["z"]] for x, y in
                  itertools.product(grid["x_range"], grid["y_range"])]
        points.append([np.mean(grid["x_range"]), np.mean(grid["y_range"]), grid["z"]])
        original_c = parse_rigid_transform_matrix(pp["link0_target_transform"])
        for position, stage2_frame in itertools.product(self.MOUNTS, ["base", "tool"]):
            source = copy.deepcopy(self.source)
            source["pick_place"]["angle_search"]["stage2"]["frame"] = stage2_frame
            cfg = self.mounted(source, position)
            mount = np.asarray(cfg["robot"]["mount_transform"])
            correction = np.asarray(cfg["pick_place"]["link0_target_transform"])
            np.testing.assert_allclose(mount @ correction, original_c, atol=1e-12)
            for point, angle in itertools.product(points, range(-30, 1, 2)):
                with self.subTest(mount=position, frame=stage2_frame,
                                  point=point, angle=angle):
                    before = make_round_poses(point, pp["place"]["position"],
                                              angle, angle, source["pick_place"], 1,
                                              angle2_grasp_deg=10., angle2_place_deg=-14.)
                    after = make_round_poses(point, cfg["pick_place"]["place"]["position"],
                                             angle, angle, cfg["pick_place"], 1,
                                             angle2_grasp_deg=10., angle2_place_deg=-14.)
                    self.assertEqual(len(before), 6)
                    for old, raw in zip(before, after):
                        self.assert_same_pose(old, raw)
                        expected = transform_pose(old, original_c)
                        actual = transform_pose(transform_pose(raw, correction), mount)
                        self.assert_same_pose(expected, actual)

    def test_old_vertical_lift_is_new_base_y(self):
        cfg = self.mounted()
        pp = cfg["pick_place"]
        raw = make_round_poses([-.41, .10, .03], pp["place"]["position"],
                               -12., -12., pp, 1)
        correction = np.asarray(pp["link0_target_transform"])
        effective = [transform_pose(pose, correction) for pose in raw]
        for lifted, contact, height in [(0, 1, pp["lift"]["grasp_z"]),
                                        (3, 4, pp["lift"]["place_z"])]:
            np.testing.assert_allclose(raw[lifted].position - raw[contact].position,
                                       [0., 0., height], atol=1e-12)
            np.testing.assert_allclose(effective[lifted].position - effective[contact].position,
                                       [0., height, 0.], atol=1e-12)

    def test_original_nonidentity_correction_and_home_are_preserved(self):
        source = copy.deepcopy(self.source)
        original_c = np.eye(4)
        original_c[:3, :3] = rpy_deg_to_matrix([0., 0., -17.])
        original_c[:3, 3] = [.02, -.04, .01]
        source["pick_place"]["link0_target_transform"] = original_c.tolist()
        cfg = self.mounted(source)
        mount = np.asarray(cfg["robot"]["mount_transform"])
        correction = np.asarray(cfg["pick_place"]["link0_target_transform"])
        np.testing.assert_allclose(mount @ correction, original_c, atol=1e-12)
        home = source["pick_place"]["home"]
        pose = PoseSpec("home", np.asarray(home["position"]),
                        rpy_deg_to_quat_wxyz(home["rpy_deg"]), "home")
        self.assert_same_pose(transform_pose(pose, original_c),
                              transform_pose(transform_pose(pose, correction), mount))
        self.assertIsNone(cfg["pick_place"]["home"]["ik_seed_joint_deg"])
        for key in ("position", "rpy_deg"):
            self.assertEqual(cfg["pick_place"]["home"][key], home[key])

    def test_all_walls_keep_physical_geometry_and_collision_scope(self):
        for position, partial_faces in itertools.product(self.MOUNTS, [False, True]):
            source = copy.deepcopy(self.source)
            if partial_faces:
                # Asymmetry makes an incorrectly permuted face mask detectable.
                source["workspace"]["wall"]["faces"] = {
                    "x_min": True, "x_max": False, "y_min": False,
                    "y_max": True, "z_min": True, "z_max": False,
                }
            cfg = self.mounted(source, position)
            mount = np.asarray(cfg["robot"]["mount_transform"])
            old_boxes = build_workspace_wall_cuboids(source["workspace"])
            new_boxes = build_workspace_wall_cuboids(cfg["workspace"])
            self.assertEqual(len(old_boxes), 3 if partial_faces else 6)
            self.assertEqual(len(new_boxes), len(old_boxes))
            remaining = [np.r_[box["pose"][:3], box["dims"]] for box in old_boxes]
            for box in new_boxes:
                center = mount[:3, :3] @ np.asarray(box["pose"][:3]) + mount[:3, 3]
                dims = np.abs(mount[:3, :3]) @ np.asarray(box["dims"])
                matches = [i for i, old in enumerate(remaining)
                           if np.allclose(np.r_[center, dims], old, atol=1e-12)]
                self.assertEqual(len(matches), 1)
                remaining.pop(matches[0])
            self.assertFalse(remaining)
            for key in ("enable", "thickness", "collision_link_names"):
                self.assertEqual(cfg["workspace"]["wall"][key],
                                 source["workspace"]["wall"][key])
            self.assertEqual(cfg["overhead"]["task_workspace"], source["workspace"])

    def test_default_tcp_is_reduced_to_019_m(self):
        cfg = self.mounted()
        self.assertEqual(cfg["overhead"]["tcp_offset_m"], .19)
        self.assertEqual(cfg["robot"]["ee_link"], "TCP_LINK")
        urdf = ET.parse(REPO_ROOT / cfg["robot"]["urdf"]).getroot()
        joint = urdf.find("joint[@name='TCP_joint']")
        self.assertIsNotNone(joint)
        self.assertEqual(joint.find("parent").get("link"), "LINK_6")
        self.assertEqual(joint.find("child").get("link"), "TCP_LINK")
        origin = joint.find("origin")
        np.testing.assert_allclose([float(v) for v in origin.get("xyz").split()], [0., 0., .19])
        np.testing.assert_allclose([float(v) for v in origin.get("rpy", "0 0 0").split()], [0., 0., 0.])

    def test_non_axis_aligned_mount_keeps_exact_task_geometry(self):
        source_walls = build_workspace_wall_cuboids(self.source["workspace"])
        for rpy in ([90., 0., 45.], [80., 0., 90.], [90., 10., 90.]):
            with self.subTest(rpy=rpy):
                cfg = prepare(self.source, self.MOUNTS[1], mount_rpy=rpy)
                mount = np.asarray(cfg["robot"]["mount_transform"])
                correction = np.asarray(cfg["pick_place"]["link0_target_transform"])
                np.testing.assert_allclose(mount @ correction, np.eye(4), atol=1e-12)
                walls = build_workspace_wall_cuboids(cfg["workspace"])
                self.assertEqual([b["name"] for b in walls],
                                 [b["name"] for b in source_walls])
                for old, new in zip(source_walls, walls):
                    world_center = mount[:3, :3] @ new["pose"][:3] + mount[:3, 3]
                    world_rotation = mount[:3, :3] @ quat_wxyz_to_matrix(new["pose"][3:])
                    np.testing.assert_allclose(world_center, old["pose"][:3], atol=1e-12)
                    np.testing.assert_allclose(world_rotation, np.eye(3), atol=1e-12)
                    np.testing.assert_allclose(new["dims"], old["dims"], atol=1e-12)

    def test_source_is_not_mutated_and_targets_are_unchanged(self):
        original = copy.deepcopy(self.source)
        cfg = self.mounted()
        self.assertEqual(self.source, original)
        for key in ("grasp_grid", "place", "base_rpy", "lift", "angle_search", "linear_move"):
            self.assertEqual(cfg["pick_place"][key], original["pick_place"][key])
        self.assertEqual(cfg["planner"], original["planner"])

    def test_unsupported_base_cutout_is_rejected(self):
        workspace = copy.deepcopy(self.source["workspace"])
        workspace["wall"]["base_clearance"]["enable"] = True
        with self.assertRaisesRegex(ValueError, "base-clearance"):
            transform_workspace(workspace, np.eye(4))


if __name__ == "__main__":
    unittest.main()
