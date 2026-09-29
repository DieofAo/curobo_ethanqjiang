#!/usr/bin/env python3
"""CPU-only checks for the original CAD assembly in the overhead RViz model."""
import copy
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "scripts"))
from build_overhead_scene_urdf import (  # noqa: E402
    DEFAULT_ASSEMBLY_MOUNTS, URDF_MESH_PREFIX, assembly_settings,
    build_argparser, build_scene, load_resolved_config, scene_settings,
)
from build_scene_urdf import pose_to_mat  # noqa: E402


def joint_transform(robot, name):
    origin = robot.find(f"joint[@name='{name}']/origin")
    return pose_to_mat([float(x) for x in origin.get("xyz").split()],
                       [float(x) for x in origin.get("rpy").split()])


class OverheadSceneTest(unittest.TestCase):
    def setUp(self):
        self.cfg, self.source = load_resolved_config(
            str(TASK_ROOT / "config/pick_place_overhead_default.json"), None
        )

    def scene(self, **kwargs):
        settings = scene_settings(self.cfg, self.source, **kwargs)
        robot, summary = build_scene(settings)
        return settings, robot, summary

    def test_original_assembly_is_attached_to_unchanged_task_frame(self):
        settings, robot, summary = self.scene()
        assembly = summary["assembly"]
        self.assertTrue(assembly["enabled"])
        self.assertTrue(assembly["visual_only"])
        self.assertEqual(assembly["mounts"], str(DEFAULT_ASSEMBLY_MOUNTS))
        self.assertEqual(robot.find("joint[@name='scene_root_fixed']/parent").get("link"),
                         settings["task_frame"])
        self.assertEqual(robot.find("joint[@name='workbench_fixed']/parent").get("link"),
                         "scene_root")
        transform = joint_transform(robot, "scene_root_fixed")
        np.testing.assert_allclose(transform @ assembly["original_base_in_cad"],
                                   np.eye(4), atol=5e-9)
        np.testing.assert_allclose(joint_transform(robot, "workbench_fixed"), np.eye(4))
        np.testing.assert_allclose(joint_transform(robot, "overhead_original_link0_fixed"),
                                   np.eye(4))
        mesh = robot.find("link[@name='workbench']/visual/geometry/mesh")
        self.assertEqual(mesh.get("filename"), assembly["mesh_uri"])
        self.assertEqual(mesh.get("scale"), "0.001 0.001 0.001")
        self.assertEqual(robot.find("material[@name='table_gray']/color").get("rgba"),
                         "0.48 0.52 0.57 0.82")
        json.dumps(summary)  # --print-settings payload must remain serializable.

    def test_assembly_does_not_move_with_new_robot_mount(self):
        _, before, _ = self.scene()
        self.cfg = copy.deepcopy(self.cfg)
        mount = np.asarray(self.cfg["robot"]["mount_transform"])
        mount[:3, 3] += [.18, -.11, .23]
        self.cfg["robot"]["mount_transform"] = mount.tolist()
        self.cfg["pick_place"]["link0_target_transform"] = np.linalg.inv(mount).tolist()
        _, after, _ = self.scene()
        np.testing.assert_allclose(joint_transform(before, "scene_root_fixed"),
                                   joint_transform(after, "scene_root_fixed"))
        self.assertFalse(np.allclose(joint_transform(before, "overhead_robot_mount"),
                                    joint_transform(after, "overhead_robot_mount")))

    def test_default_assembly_restores_original_left_mount(self):
        old_path = TASK_ROOT / "config/cad_mounts_right_y_plus_0p4.yaml"
        old = yaml.safe_load(old_path.read_text(encoding="utf-8"))
        current = yaml.safe_load(DEFAULT_ASSEMBLY_MOUNTS.read_text(encoding="utf-8"))
        original = yaml.safe_load((TASK_ROOT / "config/cad_mounts.yaml").read_text(encoding="utf-8"))
        self.assertEqual(current["left_arm"], original["left_arm"])
        self.assertEqual(current["left_arm"]["xyz"][::2], old["left_arm"]["xyz"][::2])
        self.assertAlmostEqual(old["left_arm"]["xyz"][1] - current["left_arm"]["xyz"][1], .257146)
        self.assertEqual(current["left_arm"]["rpy"][:2], old["left_arm"]["rpy"][:2])
        self.assertEqual(current["right_arm"], old["right_arm"])
        self.assertEqual(current["step_filter"], old["step_filter"])
        self.assertEqual(current["left_arm"]["rpy"][2], np.pi)
        _, robot, summary = self.scene()
        transform = joint_transform(robot, "scene_root_fixed")
        np.testing.assert_allclose(transform[:3, :3], np.diag([-1., 1., -1.]), atol=5e-9)
        np.testing.assert_allclose(summary["assembly"]["xyz"],
                                   [-.478775, -.250794, .456503], atol=1e-12)
        np.testing.assert_allclose(joint_transform(robot, "overhead_robot_mount"),
                                   self.cfg["robot"]["mount_transform"], atol=5e-9)

    def test_explicit_old_mounts_cli_preserves_extra_yaw(self):
        old_path = TASK_ROOT / "config/cad_mounts_right_y_plus_0p4.yaml"
        args = build_argparser().parse_args([
            "--config", str(TASK_ROOT / "config/pick_place_overhead_default.json"),
            "--assembly-mounts", str(old_path),
        ])
        _, robot, summary = self.scene(assembly_mounts=args.assembly_mounts)
        self.assertEqual(summary["assembly"]["mounts"], str(old_path))
        self.assertAlmostEqual(summary["assembly"]["rpy"][2], 2.9687592653589793)
        self.assertGreater(abs(joint_transform(robot, "scene_root_fixed")[0, 1]), .17)

    def test_robot_joints_and_collision_are_unchanged_and_no_second_arm(self):
        settings, robot, _ = self.scene()
        source_text = settings["urdf"].read_text(encoding="utf-8")
        source_text = source_text.replace(
            URDF_MESH_PREFIX, f"file://{(settings['urdf'].parent / 'meshes').resolve()}"
        )
        original = ET.fromstring(source_text)
        self.assertEqual(len([j for j in robot.findall("joint") if j.get("type") == "revolute"]), 6)
        for joint in original.findall("joint"):
            actual = robot.find(f"joint[@name='{joint.get('name')}']")
            self.assertEqual(ET.tostring(joint), ET.tostring(actual))
        for link in original.findall("link"):
            actual = robot.find(f"link[@name='{link.get('name')}']")
            self.assertEqual([ET.tostring(c) for c in link.findall("collision")],
                             [ET.tostring(c) for c in actual.findall("collision")])
        self.assertEqual(len(robot.findall("link/collision")), len(original.findall("link/collision")))
        self.assertFalse(any(x.get("name", "").startswith(("static_", "second_")) for x in robot))
        self.assertFalse(robot.findall("joint/mimic"))

    def test_no_assembly_restores_simplified_view_without_needing_assets(self):
        _, robot, summary = self.scene(include_assembly=False,
                                      assembly_mounts="missing.yaml", assembly_mesh="missing.stl")
        self.assertFalse(summary["assembly"]["enabled"])
        self.assertIsNone(robot.find("link[@name='workbench']"))
        self.assertIsNone(robot.find("link[@name='scene_root']"))
        self.assertIsNotNone(robot.find("link[@name='overhead_grasp_plane']"))

    def test_explicit_original_mounts_and_right_arm(self):
        mounts = TASK_ROOT / "config/cad_mounts.yaml"
        left = assembly_settings(mounts_path=mounts)
        np.testing.assert_allclose(left["xyz"], [-.478775, -.250794, .456503], atol=1e-12)
        right = assembly_settings(mounts_path=mounts, arm="right", mesh_path=left["mesh_uri"])
        np.testing.assert_allclose(right["xyz"], [-.321225, -.230206, .456503], atol=1e-12)
        self.assertEqual(right["arm"], "right")

    def test_missing_files_and_invalid_arm_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "assembly mounts"):
            assembly_settings(mounts_path="missing.yaml")
        with self.assertRaisesRegex(FileNotFoundError, "assembly mesh"):
            assembly_settings(mesh_path="missing.stl")
        with self.assertRaisesRegex(ValueError, "arm must"):
            assembly_settings(arm="middle")

    def test_invalid_mount_content_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mounts.yaml"
            for payload in ("[]", "right_arm: {}", "left_arm: {}",
                            "left_arm: {xyz: [1, 2], rpy: [0, 0, 0]}",
                            "left_arm: {xyz: [1, 2, .nan], rpy: [0, 0, 0]}"):
                path.write_text(payload, encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    assembly_settings(mounts_path=path)

    def test_generated_name_collision_rejected(self):
        settings = scene_settings(self.cfg, self.source)
        with tempfile.TemporaryDirectory() as temporary:
            urdf = Path(temporary) / "robot.urdf"
            source = ET.parse(settings["urdf"]).getroot()
            ET.SubElement(source, "material", {"name": "table_gray"})
            urdf.write_text(ET.tostring(source, encoding="unicode"), encoding="utf-8")
            settings["urdf"] = urdf
            with self.assertRaisesRegex(ValueError, "generated URDF names already exist.*table_gray"):
                build_scene(settings)

    def test_cli_accepts_assembly_options(self):
        args = build_argparser().parse_args([
            "--config", "test.json", "--no-assembly", "--assembly-mounts", "mounts.yaml",
            "--assembly-arm", "right", "--assembly-mesh", "table.stl",
        ])
        self.assertTrue(args.no_assembly)
        self.assertEqual(args.assembly_arm, "right")


if __name__ == "__main__":
    unittest.main()
