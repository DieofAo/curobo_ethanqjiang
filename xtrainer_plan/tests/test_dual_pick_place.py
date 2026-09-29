#!/usr/bin/env python3
"""CPU-only contract tests for the independent dual-arm pick/place setup.

These tests intentionally stop at configuration, transforms, scheduling, and
trajectory serialization.  Importing this module must not require CuRobo,
CUDA, ROS, or a running robot.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPO_ROOT / "xtrainer_plan"
SCRIPTS_DIR = TASK_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import plan_dual_pick_place as dual  # noqa: E402
import publish_dual_place_tf_ros as dual_preview  # noqa: E402
from xtrainer_common import (  # noqa: E402
    PoseSpec,
    build_place_position_tf_specs,
    matrix_to_quat_wxyz,
    normalize_link0_target_transform_config_layer,
    parse_rigid_transform_matrix,
    save_trajectory,
)


def _pose(name: str, x: float = 0.0) -> PoseSpec:
    return PoseSpec(name, [x, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])


class RigidTransformParsingTest(unittest.TestCase):
    def test_accepts_valid_rigid_transform_as_float64_copy(self) -> None:
        raw = [
            [0, -1, 0, 0.12],
            [1, 0, 0, -0.34],
            [0, 0, 1, 0.56],
            [0, 0, 0, 1],
        ]

        parsed = parse_rigid_transform_matrix(raw, "test transform")

        self.assertEqual(parsed.dtype, np.float64)
        np.testing.assert_allclose(parsed, raw)

    def test_accepts_axis_angle_and_position_mapping(self) -> None:
        raw = {
            "position": [0.12, -0.34, 0.56],
            "rotation": {"axis": [0.0, 0.0, 2.0], "angle_deg": 90.0},
        }

        parsed = parse_rigid_transform_matrix(raw, "test transform")

        self.assertEqual(parsed.dtype, np.float64)
        np.testing.assert_allclose(parsed, [
            [0.0, -1.0, 0.0, 0.12],
            [1.0, 0.0, 0.0, -0.34],
            [0.0, 0.0, 1.0, 0.56],
            [0.0, 0.0, 0.0, 1.0],
        ], atol=1e-12)

    def test_accepts_named_axes_and_translation_alias(self) -> None:
        expected_rotations = {
            "x": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            "y": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            "z": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        }

        for axis, expected_rotation in expected_rotations.items():
            with self.subTest(axis=axis):
                parsed = parse_rigid_transform_matrix({
                    "translation": [1, 2, 3],
                    "rotation": {"axis": axis.upper(), "angle_deg": 90},
                })
                np.testing.assert_allclose(
                    parsed[:3, :3], expected_rotation, atol=1e-12
                )
                np.testing.assert_allclose(parsed[:3, 3], [1, 2, 3])

    def test_accepts_matching_position_and_translation_alias(self) -> None:
        parsed = parse_rigid_transform_matrix({
            "position": [1, 2, 3],
            "translation": [1.0, 2.0, 3.0],
            "rotation": {"axis": "z", "angle_deg": 0},
        })

        np.testing.assert_allclose(parsed, [
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ], atol=1e-12)

    def test_rejects_invalid_rigid_transform_components(self) -> None:
        invalid_cases = {
            "shape": np.eye(3),
            "finite": [[1, 0, 0, np.inf], [0, 1, 0, 0],
                       [0, 0, 1, 0], [0, 0, 0, 1]],
            "last row": [[1, 0, 0, 0], [0, 1, 0, 0],
                         [0, 0, 1, 0], [0, 0, 1, 1]],
            "orthogonal": [[1, 0.1, 0, 0], [0, 1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]],
            "reflection": [[-1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]],
        }

        for label, value in invalid_cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_rigid_transform_matrix(value, label)

    def test_rejects_invalid_axis_angle_mapping(self) -> None:
        valid_rotation = {"axis": "z", "angle_deg": 0.0}
        invalid_cases = {
            "missing rotation": {"position": [0, 0, 0]},
            "missing position": {"rotation": valid_rotation},
            "unknown transform field": {
                "position": [0, 0, 0],
                "rotation": valid_rotation,
                "scale": 1,
            },
            "rotation type": {
                "position": [0, 0, 0],
                "rotation": [0, 0, 1, 0],
            },
            "unknown rotation field": {
                "position": [0, 0, 0],
                "rotation": {**valid_rotation, "angle_rad": 0},
            },
            "missing axis": {
                "position": [0, 0, 0],
                "rotation": {"angle_deg": 0},
            },
            "missing angle": {
                "position": [0, 0, 0],
                "rotation": {"axis": "z"},
            },
            "position shape": {
                "position": [0, 0],
                "rotation": valid_rotation,
            },
            "position finite": {
                "position": [0, np.nan, 0],
                "rotation": valid_rotation,
            },
            "position bool": {
                "position": [True, 0, 0],
                "rotation": valid_rotation,
            },
            "position string": {
                "position": ["0", 0, 0],
                "rotation": valid_rotation,
            },
            "axis name": {
                "position": [0, 0, 0],
                "rotation": {"axis": "xy", "angle_deg": 0},
            },
            "axis shape": {
                "position": [0, 0, 0],
                "rotation": {"axis": [0, 1], "angle_deg": 0},
            },
            "zero axis at zero angle": {
                "position": [0, 0, 0],
                "rotation": {"axis": [0, 0, 0], "angle_deg": 0},
            },
            "axis finite": {
                "position": [0, 0, 0],
                "rotation": {"axis": [0, np.inf, 0], "angle_deg": 1},
            },
            "angle scalar": {
                "position": [0, 0, 0],
                "rotation": {"axis": "z", "angle_deg": [1]},
            },
            "angle finite": {
                "position": [0, 0, 0],
                "rotation": {"axis": "z", "angle_deg": np.inf},
            },
            "angle bool": {
                "position": [0, 0, 0],
                "rotation": {"axis": "z", "angle_deg": True},
            },
            "angle string": {
                "position": [0, 0, 0],
                "rotation": {"axis": "z", "angle_deg": "30"},
            },
            "conflicting position aliases": {
                "position": [0, 0, 0],
                "translation": [1, 0, 0],
                "rotation": valid_rotation,
            },
        }

        for label, value in invalid_cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_rigid_transform_matrix(value, label)


class Link0TargetTransformConfigTest(unittest.TestCase):
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    correction = [
        [0.0, -1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    axis_angle_correction = {
        "position": [1.0, 2.0, 3.0],
        "rotation": {"axis": "z", "angle_deg": 90.0},
    }

    def test_default_yaml_uses_only_canonical_pick_place_path(self) -> None:
        with open(
            TASK_ROOT / "config" / "pick_place_default.yaml",
            "r",
            encoding="utf-8",
        ) as stream:
            pick_default = yaml.safe_load(stream)
        with open(
            TASK_ROOT / "config" / "dual_pick_place_default.yaml",
            "r",
            encoding="utf-8",
        ) as stream:
            dual_default = yaml.safe_load(stream)

        raw_default = pick_default["pick_place"]["link0_target_transform"]
        self.assertIsInstance(raw_default, dict)
        self.assertEqual(set(raw_default), {"position", "rotation"})
        default_transform = parse_rigid_transform_matrix(raw_default)
        self.assertNotIn(
            "link0_target_transform", dual_default.get("dual_arm", {})
        )
        merged = dual.load_dual_config(None)
        np.testing.assert_allclose(parse_rigid_transform_matrix(
            merged["pick_place"]["link0_target_transform"]
        ), default_transform)
        self.assertNotIn("link0_target_transform", merged["dual_arm"])

    def test_layer_normalizer_migrates_legacy_without_mutating_input(self) -> None:
        layer = {
            "dual_arm": {
                "link0_target_transform": self.correction,
                "task_layout": "same_local_grid",
            }
        }

        normalized = normalize_link0_target_transform_config_layer(
            layer, "legacy.yaml"
        )

        self.assertIn("link0_target_transform", layer["dual_arm"])
        np.testing.assert_allclose(
            normalized["pick_place"]["link0_target_transform"], self.correction
        )
        self.assertNotIn(
            "link0_target_transform", normalized["dual_arm"]
        )
        self.assertEqual(
            normalized["dual_arm"]["task_layout"], "same_local_grid"
        )

    def test_layer_normalizer_accepts_equal_new_and_legacy_values(self) -> None:
        normalized = normalize_link0_target_transform_config_layer({
            "pick_place": {"link0_target_transform": self.correction},
            "dual_arm": {"link0_target_transform": self.correction},
        })

        np.testing.assert_allclose(
            normalized["pick_place"]["link0_target_transform"], self.correction
        )
        self.assertNotIn(
            "link0_target_transform", normalized["dual_arm"]
        )

    def test_layer_normalizer_compares_axis_angle_and_legacy_as_matrices(self) -> None:
        normalized = normalize_link0_target_transform_config_layer({
            "pick_place": {
                "link0_target_transform": self.axis_angle_correction,
            },
            "dual_arm": {"link0_target_transform": self.correction},
        })

        np.testing.assert_allclose(
            parse_rigid_transform_matrix(
                normalized["pick_place"]["link0_target_transform"]
            ),
            self.correction,
            atol=1e-12,
        )
        self.assertNotIn(
            "link0_target_transform", normalized["dual_arm"]
        )

    def test_layer_normalizer_rejects_conflicting_new_and_legacy_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "\u4e24\u8005\u4e0d\u540c"):
            normalize_link0_target_transform_config_layer({
                "pick_place": {"link0_target_transform": self.identity},
                "dual_arm": {"link0_target_transform": self.correction},
            }, "conflict.yaml")

    def test_dual_loader_migrates_legacy_user_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "legacy.yaml"
            custom.write_text(
                yaml.safe_dump({
                    "dual_arm": {
                        "link0_target_transform": self.correction,
                    }
                }),
                encoding="utf-8",
            )
            cfg = dual.load_dual_config(str(custom))

        np.testing.assert_allclose(
            cfg["pick_place"]["link0_target_transform"], self.correction
        )
        self.assertNotIn("link0_target_transform", cfg["dual_arm"])

    def test_dual_loader_normalizes_translation_alias_before_layer_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "translation_alias.yaml"
            custom.write_text(
                yaml.safe_dump({
                    "pick_place": {
                        "link0_target_transform": {
                            "translation": [0.4, -0.5, 0.6],
                        },
                    },
                }),
                encoding="utf-8",
            )
            cfg = dual.load_dual_config(str(custom))

        transform_cfg = cfg["pick_place"]["link0_target_transform"]
        self.assertNotIn("translation", transform_cfg)
        self.assertEqual(transform_cfg["position"], [0.4, -0.5, 0.6])
        np.testing.assert_allclose(
            parse_rigid_transform_matrix(transform_cfg)[:3, 3],
            [0.4, -0.5, 0.6],
        )

    def test_dual_loader_rejects_conflicting_paths_in_one_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "conflict.yaml"
            custom.write_text(
                yaml.safe_dump({
                    "pick_place": {
                        "link0_target_transform": self.identity,
                    },
                    "dual_arm": {
                        "link0_target_transform": self.correction,
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "\u4e24\u8005\u4e0d\u540c"):
                dual.load_dual_config(str(custom))


class RigidTransformCompositionTest(unittest.TestCase):

    def test_dual_target_transform_preserves_left_multiply_order(self) -> None:
        correction = np.array([
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ])
        arm2_in_arm1 = np.array([
            [-1.0, 0.0, 0.0, -0.8],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        arm1, arm2 = dual.arm_target_transforms(
            correction, arm2_in_arm1
        )

        np.testing.assert_allclose(arm1, correction)
        np.testing.assert_allclose(arm2, correction @ arm2_in_arm1)
        self.assertFalse(np.allclose(arm2, arm2_in_arm1 @ correction))

    def test_transformed_cardinal_axis_maps_or_rejects_direction(self) -> None:
        rotate_y_90 = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.assertEqual(
            dual.transformed_cardinal_axis(rotate_y_90, "z"), "x"
        )

        angle = np.radians(45.0)
        rotate_y_45 = np.array([
            [np.cos(angle), 0.0, np.sin(angle), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        with self.assertRaises(ValueError):
            dual.transformed_cardinal_axis(rotate_y_45, "z")

    def test_every_generated_pose_is_the_old_pose_left_multiplied_by_c(self) -> None:
        cfg = dual.load_dual_config(None)
        correction = np.array([
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ])
        arm2_in_arm1 = np.array([
            [-1.0, 0.0, 0.0, -0.8],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        effective_transforms = dual.arm_target_transforms(
            correction, arm2_in_arm1
        )
        choice = dual.AngleChoice(-4.0, -6.0)
        item_local = [-0.22, 0.13, 0.04]
        place_local = [-0.31, -0.17, 0.11]

        for arm, old_transform, effective_transform in (
            (1, np.eye(4), effective_transforms[0]),
            (2, arm2_in_arm1, effective_transforms[1]),
        ):
            old_sequence = dual.make_arm_sequence(
                item_local,
                place_local,
                choice,
                cfg["pick_place"],
                3,
                old_transform,
                f"arm{arm}_",
            )
            effective_sequence = dual.make_arm_sequence(
                item_local,
                place_local,
                choice,
                cfg["pick_place"],
                3,
                effective_transform,
                f"arm{arm}_",
            )

            self.assertEqual(len(effective_sequence), len(old_sequence))
            self.assertGreater(len(effective_sequence), 0)
            for old_pose, effective_pose in zip(
                old_sequence, effective_sequence
            ):
                expected = dual.transform_pose(old_pose, correction)
                np.testing.assert_allclose(
                    effective_pose.position, expected.position, atol=1e-12
                )
                np.testing.assert_allclose(
                    dual.quat_wxyz_to_matrix(effective_pose.quat_wxyz),
                    dual.quat_wxyz_to_matrix(expected.quat_wxyz),
                    atol=1e-12,
                )
                self.assertEqual(effective_pose.kind, old_pose.kind)


class DualMountAndPlaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = dual.load_dual_config(None)
        cls.mounts_path = REPO_ROOT / cls.cfg["robot"]["mounts"]
        cls.t12, cls.mounts = dual.mounts_relative_transform(
            str(cls.mounts_path), "left"
        )

    def test_same_side_mount_is_x_offset_with_opposite_yaw(self) -> None:
        # In arm1/LINK_0, arm2 is 0.8 m along -X, with no lateral offset.
        np.testing.assert_allclose(
            self.t12[:3, 3], [-0.8, 0.0, 0.0], atol=1e-9
        )
        np.testing.assert_allclose(
            self.t12[:3, :3], np.diag([-1.0, -1.0, 1.0]), atol=1e-9
        )
        self.assertAlmostEqual(
            self.mounts["left_arm"]["xyz"][1],
            self.mounts["right_arm"]["xyz"][1],
        )

    def test_arm2_place_is_declared_in_arm1_base_then_converted_local(self) -> None:
        arm1_place = np.asarray(
            self.cfg["pick_place"]["place"]["position"], dtype=np.float64
        )
        expected_x = float(self.cfg["dual_arm"]["second_place_x"])
        self.assertIsNone(self.cfg["dual_arm"]["second_place_position"])
        arm2_place_root = dual.derive_second_place_position(
            arm1_place, self.cfg["dual_arm"]
        )

        self.assertAlmostEqual(arm2_place_root[0], expected_x)
        np.testing.assert_allclose(arm2_place_root[1:], arm1_place[1:])

        arm2_place_local = dual.transform_points(
            arm2_place_root[None, :], np.linalg.inv(self.t12)
        )[0]
        np.testing.assert_allclose(
            arm2_place_local,
            [-(expected_x + 0.8), -arm1_place[1], arm1_place[2]],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            dual.transform_points(arm2_place_local[None, :], self.t12)[0],
            arm2_place_root,
            atol=1e-9,
        )

    def test_custom_arm1_place_only_changes_x_for_arm2(self) -> None:
        arm1_place = np.array([-0.12, 0.137, 0.428], dtype=np.float64)
        original = arm1_place.copy()

        arm2_place = dual.derive_second_place_position(
            arm1_place, self.cfg["dual_arm"]
        )

        np.testing.assert_allclose(
            arm2_place,
            [self.cfg["dual_arm"]["second_place_x"], 0.137, 0.428],
        )
        np.testing.assert_array_equal(arm2_place[1:], arm1_place[1:])
        np.testing.assert_array_equal(arm1_place, original)

    def test_rviz_place_tf_positions_use_joint_root_coordinates(self) -> None:
        specs = build_place_position_tf_specs({
            "place_positions_arm1_base": {
                "arm1": [-0.26, -0.26, 0.10],
                "arm2": [-0.47, -0.26, 0.10],
            },
            # 该值只用于证明 helper 没有误取二号臂本地坐标。
            "place_position_arm2_local": [-0.33, 0.26, 0.10],
        })

        self.assertEqual(
            specs,
            [
                ("xtrainer_arm1_place", [-0.26, -0.26, 0.10],
                 [1.0, 0.0, 0.0, 0.0]),
                ("xtrainer_arm2_place", [-0.47, -0.26, 0.10],
                 [1.0, 0.0, 0.0, 0.0]),
            ],
        )

    def test_rviz_place_tf_supports_legacy_single_arm_metadata(self) -> None:
        self.assertEqual(
            build_place_position_tf_specs({"place_position": [-0.1, 0.2, 0.3]}),
            [("xtrainer_arm1_place", [-0.1, 0.2, 0.3],
              [1.0, 0.0, 0.0, 0.0])],
        )

    def test_rviz_place_tf_shows_link0_target_frame_rotation(self) -> None:
        specs = build_place_position_tf_specs({
            "robot": {
                "link0_target_transform": [
                    [0.0, -1.0, 0.0, 0.3],
                    [1.0, 0.0, 0.0, -0.2],
                    [0.0, 0.0, 1.0, 0.1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "place_positions_arm1_base": {
                "arm1": [0.1, 0.2, 0.3],
                "arm2": [-0.4, 0.2, 0.3],
            },
        })

        expected_quat = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
        np.testing.assert_allclose(specs[0][2], expected_quat)
        np.testing.assert_allclose(specs[1][2], expected_quat)

    def test_static_preview_reads_layered_dual_default_yaml(self) -> None:
        settings = dual_preview.load_preview_settings()
        correction = parse_rigid_transform_matrix(
            self.cfg["pick_place"]["link0_target_transform"]
        )
        raw_place1 = np.asarray(
            self.cfg["pick_place"]["place"]["position"], dtype=np.float64
        )
        raw_place2 = dual.derive_second_place_position(
            raw_place1, self.cfg["dual_arm"]
        )

        self.assertEqual(settings["base_frame"], "LINK_0")
        self.assertEqual(
            Path(settings["config"]),
            TASK_ROOT / "config" / "dual_pick_place_default.yaml",
        )
        self.assertEqual(
            Path(settings["mounts"]),
            TASK_ROOT / "config" / "cad_mounts_same_side.yaml",
        )
        np.testing.assert_allclose(
            settings["link0_target_transform"], correction
        )
        np.testing.assert_allclose(settings["arm1_place_raw"], raw_place1)
        np.testing.assert_allclose(settings["arm2_place_raw"], raw_place2)
        np.testing.assert_allclose(
            settings["arm1_place"],
            correction[:3, :3] @ raw_place1 + correction[:3, 3],
        )
        np.testing.assert_allclose(
            settings["arm2_place"],
            correction[:3, :3] @ raw_place2 + correction[:3, 3],
        )
        np.testing.assert_allclose(
            settings["arm1_quat_wxyz"],
            matrix_to_quat_wxyz(correction[:3, :3]),
        )

    def test_static_preview_applies_canonical_pick_place_transform(self) -> None:
        transform_cfg = Link0TargetTransformConfigTest.axis_angle_correction
        correction = parse_rigid_transform_matrix(
            transform_cfg, "test preview transform"
        )
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "preview.yaml"
            custom.write_text(
                yaml.safe_dump({
                    "pick_place": {
                        "link0_target_transform": (
                            transform_cfg
                        ),
                    }
                }),
                encoding="utf-8",
            )
            settings = dual_preview.load_preview_settings(str(custom))

        raw_place1 = np.asarray(settings["arm1_place_raw"], dtype=np.float64)
        raw_place2 = np.asarray(settings["arm2_place_raw"], dtype=np.float64)
        np.testing.assert_allclose(settings["link0_target_transform"], correction)
        np.testing.assert_allclose(
            settings["arm1_place"],
            correction[:3, :3] @ raw_place1 + correction[:3, 3],
        )
        np.testing.assert_allclose(
            settings["arm2_place"],
            correction[:3, :3] @ raw_place2 + correction[:3, 3],
        )

    def test_static_preview_migrates_legacy_dual_yaml_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "preview.yaml"
            custom.write_text(
                "pick_place:\n"
                "  place:\n"
                "    position: [-0.12, 0.13, 0.22]\n"
                "dual_arm:\n"
                "  second_place_x: -0.55\n"
                "  link0_target_transform:\n"
                "    - [0.0, -1.0, 0.0, 1.0]\n"
                "    - [1.0, 0.0, 0.0, 2.0]\n"
                "    - [0.0, 0.0, 1.0, 3.0]\n"
                "    - [0.0, 0.0, 0.0, 1.0]\n",
                encoding="utf-8",
            )
            settings = dual_preview.load_preview_settings(str(custom))

        np.testing.assert_allclose(
            settings["arm1_place_raw"], [-0.12, 0.13, 0.22]
        )
        np.testing.assert_allclose(
            settings["arm2_place_raw"], [-0.55, 0.13, 0.22]
        )
        np.testing.assert_allclose(
            settings["arm1_place"], [0.87, 1.88, 3.22]
        )
        np.testing.assert_allclose(
            settings["arm2_place"], [0.87, 1.45, 3.22]
        )
        np.testing.assert_allclose(
            settings["arm1_quat_wxyz"],
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        )


class SharedWorldGridAllocationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = dual.load_dual_config(None)
        cls.t12, _ = dual.mounts_relative_transform(
            str(REPO_ROOT / cfg["robot"]["mounts"]), "left"
        )
        cls.t21 = np.linalg.inv(cls.t12)

    @staticmethod
    def _shared_items():
        # Deliberately shuffled: allocation order must be geometric, not input
        # order.  The shared/root X range is [-0.6, -0.2], center=-0.4.
        coordinates = [
            (-0.4, 0.04),
            (-0.2, 0.02),
            (-0.6, -0.06),
            (-0.3, 0.03),
            (-0.5, -0.05),
        ]
        return [
            {
                "index": index,
                "row": index,
                "col": 0,
                "position": [x, y, 0.1],
            }
            for index, (x, y) in enumerate(coordinates)
        ]

    def test_total_quota_two_assigns_one_outer_source_to_each_arm(self) -> None:
        arm1, arm2 = dual.allocate_shared_grid_outside_in(
            self._shared_items(), self.t12, 2
        )

        self.assertEqual((len(arm1), len(arm2)), (1, 1))
        self.assertAlmostEqual(arm1[0]["root_position"][0], -0.2)
        self.assertAlmostEqual(arm2[0]["root_position"][0], -0.6)
        self.assertNotEqual(arm1[0]["index"], arm2[0]["index"])

    def test_full_allocation_is_disjoint_ordered_and_converts_arm2_local(self) -> None:
        items = self._shared_items()
        arm1, arm2 = dual.allocate_shared_grid_outside_in(
            items, self.t12, None
        )

        self.assertEqual(
            [item["root_position"][0] for item in arm1],
            [-0.2, -0.3, -0.4],
        )
        self.assertEqual(
            [item["root_position"][0] for item in arm2],
            [-0.6, -0.5],
        )
        arm1_ids = [item["index"] for item in arm1]
        arm2_ids = [item["index"] for item in arm2]
        self.assertEqual(len(set(arm1_ids + arm2_ids)), len(items))
        self.assertEqual(set(arm1_ids).intersection(arm2_ids), set())
        self.assertEqual(set(arm1_ids + arm2_ids), {
            item["index"] for item in items
        })
        # The exact center line belongs to arm1.
        self.assertIn(0, arm1_ids)

        for item in arm1:
            np.testing.assert_allclose(item["position"], item["root_position"])
        for item in arm2:
            root = np.asarray(item["root_position"], dtype=np.float64)
            expected_local = dual.transform_points(root[None, :], self.t21)[0]
            np.testing.assert_allclose(item["position"], expected_local)
            np.testing.assert_allclose(
                dual.transform_points(
                    np.asarray(item["position"])[None, :], self.t12
                )[0],
                root,
            )

    def test_partition_uses_robot_base_midpoint_not_grid_bbox_midpoint(self) -> None:
        t12 = np.eye(4, dtype=np.float64)
        t12[0, 3] = -1.0
        # Robot bases split at x=-0.5.  This deliberately asymmetric grid has
        # bbox midpoint x=-0.6, so the x=-0.6 item distinguishes the rules:
        # bbox allocation would give it to arm1, base geometry gives it arm2.
        roots = [-0.9, -0.6, -0.5, -0.49, -0.3]
        items = [
            {"index": index, "position": [x, 0.0, 0.1]}
            for index, x in enumerate(roots)
        ]

        arm1, arm2 = dual.allocate_shared_grid_outside_in(items, t12, None)

        self.assertEqual(
            [item["root_position"][0] for item in arm1],
            [-0.3, -0.49, -0.5],
        )
        self.assertEqual(
            [item["root_position"][0] for item in arm2],
            [-0.9, -0.6],
        )
        self.assertIn(1, [item["index"] for item in arm2])
        # The exact robot-base midpoint belongs to arm1.
        self.assertIn(2, [item["index"] for item in arm1])


class SharedArmIKFilterTest(unittest.TestCase):
    @staticmethod
    def _items(indices):
        return [
            {
                "index": index,
                "position": [float(index), 0.0, 0.1],
                "root_position": [float(index), 0.0, 0.1],
            }
            for index in indices
        ]

    def test_each_arm_skips_and_refills_only_its_own_quota(self) -> None:
        arm1_source = self._items([0, 1, 2, 3, 4])
        arm2_source = self._items([10, 11, 12, 13])
        calls = {1: [], 2: []}

        def probe1(_offset, item):
            calls[1].append(item["index"])
            ok = item["index"] in (1, 3, 4)
            return ok, {
                "status": "IK_REACHABLE" if ok else "NO_PRIMARY_IK"
            }

        def probe2(_offset, item):
            calls[2].append(item["index"])
            ok = item["index"] in (10, 12, 13)
            return ok, {
                "status": "IK_REACHABLE" if ok else "NO_PRIMARY_IK"
            }

        selected1, skipped1 = dual.filter_arm_items_by_primary_ik(
            arm1_source, 2, "skip", 1, probe1
        )
        selected2, skipped2 = dual.filter_arm_items_by_primary_ik(
            arm2_source, 1, "skip", 2, probe2
        )

        self.assertEqual([item["index"] for item in selected1], [1, 3])
        self.assertEqual([item["source_item_index"] for item in skipped1], [0, 2])
        self.assertEqual(calls[1], [0, 1, 2, 3])
        self.assertEqual([item["index"] for item in selected2], [10])
        self.assertEqual(skipped2, [])
        self.assertEqual(calls[2], [10])
        self.assertTrue(all(item["arm"] == 1 for item in skipped1))

    def test_non_ik_result_and_probe_exception_are_fail_closed(self) -> None:
        items = self._items([20, 21, 22])
        cases = (
            (
                "non_ik_status",
                mock.Mock(return_value=(False, {
                    "status": "TRAJOPT_FAIL",
                    "reason": "not an IK certificate",
                })),
                "TRAJOPT_FAIL",
            ),
            (
                "probe_exception",
                mock.Mock(side_effect=RuntimeError("IK backend unavailable")),
                "IK backend unavailable",
            ),
        )
        for name, probe, message in cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(RuntimeError, message):
                    dual.filter_arm_items_by_primary_ik(
                        items, 2, "skip", 2, probe
                    )
                probe.assert_called_once()


class FirstPregraspHomeTest(unittest.TestCase):
    @staticmethod
    def _filtered_item(arm: int, source: int, grasp: float):
        item = {
            "index": source,
            "position": [float(source), 0.1, 0.2],
            "root_position": [float(source), 0.1, 0.2],
        }
        selected, skipped = dual.filter_arm_items_by_primary_ik(
            [item],
            1,
            "skip",
            arm,
            lambda _offset, _item: (
                True,
                {
                    "status": "IK_REACHABLE",
                    "probe": {
                        "selected_choice": dual.AngleChoice(
                            grasp, 90.0
                        ).to_dict()
                    },
                },
            ),
        )
        if skipped or len(selected) != 1:
            raise AssertionError("test fixture failed to select one item")
        return selected[0]

    def test_preferred_pose_pair_retains_multiple_home_ik_branches(self) -> None:
        item1 = self._filtered_item(1, 7, 30.0)
        item2 = self._filtered_item(2, 11, 60.0)
        choices = [
            dual.AngleChoice(0.0, 0.0),
            dual.AngleChoice(30.0, 0.0),
            dual.AngleChoice(60.0, 0.0),
        ]
        q_seed = np.linspace(-0.2, 0.2, 12)
        q0 = q_seed + 0.1
        q1 = q_seed + 0.2
        q2 = q_seed + 0.3
        home_calls = []

        def make_home(
            _item, _place, choice, _pp, item_index, _transform, arm
        ):
            home_calls.append((arm, item_index, choice))
            return _pose(f"arm{arm}_pregrasp_g{choice.grasp:g}", choice.grasp)

        solver = mock.Mock(side_effect=[
            (
                np.stack([q0, q1]),
                types.SimpleNamespace(status="PREFERRED_PAIR_OK"),
                {
                    "n_success": 2,
                    "n_unique": 2,
                    "n_returned": 2,
                    "max_solutions": 4,
                    "truncated": False,
                },
            ),
            (
                np.stack([q2]),
                types.SimpleNamespace(status="SECOND_PAIR_OK"),
                {
                    "n_success": 1,
                    "n_unique": 1,
                    "n_returned": 1,
                    "max_solutions": 4,
                    "truncated": False,
                },
            ),
        ])
        mg = types.SimpleNamespace(ik_solver=object())
        idle2 = _pose("arm2_idle_home")
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                dual, "unique_primary_grasp_choices", return_value=choices
            ))
            stack.enter_context(mock.patch.object(
                dual, "first_pregrasp_home_pose", side_effect=make_home
            ))
            stack.enter_context(mock.patch.object(
                dual, "solve_dual_ik_solutions", solver
            ))
            roots, report = dual.enumerate_first_pregrasp_home_roots(
                mg=mg,
                q_seed=q_seed,
                arm1_item=item1,
                arm2_item=item2,
                place1_local=[-0.2, -0.2, 0.1],
                place2_local=[-0.3, 0.2, 0.1],
                pp={},
                asr={},
                transforms=(np.eye(4), np.eye(4)),
                ee_links=("TCP_LINK", "second_TCP_LINK"),
                idle_arm2_home=idle2,
                q_lo=np.full(12, -10.0),
                q_hi=np.full(12, 10.0),
                max_angle_pair_trials=2,
                max_branches_per_pair=4,
                max_roots=0,
            )

        self.assertEqual(
            [(arm, index, choice.grasp) for arm, index, choice in home_calls],
            [
                (1, 7, 30.0), (2, 11, 60.0),
                (1, 7, 0.0), (2, 11, 0.0),
            ],
        )
        self.assertEqual(len(roots), 3)
        np.testing.assert_array_equal(roots[0].q_home, q0)
        np.testing.assert_array_equal(roots[1].q_home, q2)
        np.testing.assert_array_equal(roots[2].q_home, q1)
        self.assertIs(roots[0].arm1_home, roots[2].arm1_home)
        self.assertIs(roots[0].arm2_home, roots[2].arm2_home)
        self.assertEqual(
            [root.angle_pair_trial for root in roots], [1, 2, 1]
        )
        self.assertEqual(
            [root.ik_branch_index for root in roots], [0, 0, 1]
        )
        self.assertEqual(
            [root.arm1_choice.grasp for root in roots], [30.0, 0.0, 30.0]
        )
        self.assertEqual(
            [root.arm2_choice.grasp for root in roots], [60.0, 0.0, 60.0]
        )
        self.assertEqual(report["status"], "FIRST_PREGRASP_HOME_ROOTS_OK")
        self.assertEqual(report["n_angle_pair_trials"], 2)
        self.assertEqual(report["n_raw_success_ik_branches"], 3)
        self.assertEqual(report["n_unique_ik_branches"], 3)
        self.assertEqual(
            report["n_ik_branches_after_dedup_and_pair_cap"], 3
        )
        self.assertEqual(report["n_roots_available"], 3)
        self.assertEqual(report["n_roots_returned"], 3)
        self.assertTrue(report["angle_pair_budget_exhausted"])
        self.assertFalse(report["root_budget_exhausted"])
        self.assertEqual(solver.call_count, 2)
        for call in solver.call_args_list:
            self.assertIs(call.args[0], mg.ik_solver)
            self.assertEqual(call.args[3], "second_TCP_LINK")
            np.testing.assert_array_equal(call.kwargs["seed_q"], q_seed)
            self.assertEqual(call.kwargs["max_solutions"], 4)


class IndependentDualIKTest(unittest.TestCase):
    @staticmethod
    def _limits():
        return np.full(12, -100.0), np.full(12, 100.0)

    @staticmethod
    def _arm_report(n: int = 1):
        return {
            "n_success": n,
            "n_unique": n,
            "n_returned": n,
            "max_solutions": 8,
            "truncated": False,
        }

    @classmethod
    def _context(cls, collision_filter, root_to_arm2=None):
        q_lo, q_hi = cls._limits()
        return dual.SeparateArmIkContext(
            arm_solvers=(
                types.SimpleNamespace(dof=6),
                types.SimpleNamespace(dof=6),
            ),
            root_to_arm=(
                np.eye(4),
                np.eye(4) if root_to_arm2 is None else root_to_arm2,
            ),
            q_lo=q_lo,
            q_hi=q_hi,
            collision_filter=collision_filter,
            max_candidates_per_arm=8,
            max_pair_trials=64,
            max_goal_solutions=4,
        )

    def test_cartesian_cross_pair_survives_collision_filter(self) -> None:
        arm1 = np.zeros((2, 6), dtype=np.float64)
        arm2 = np.zeros((2, 6), dtype=np.float64)
        arm1[:, 0] = [0.1, 0.2]
        arm2[:, 0] = [0.1, 0.2]
        seen = []

        def only_cross_pair_is_safe(candidates):
            candidates = np.asarray(candidates, dtype=np.float64)
            seen.append(candidates.copy())
            # Both same-index pairs collide. A zip-based implementation would
            # therefore miss the only safe cross product (arm1[0], arm2[1]).
            safe = np.isclose(candidates[:, 0], 0.1) & np.isclose(
                candidates[:, 6], 0.2
            )
            return safe, {"checked": True}

        q_lo, q_hi = self._limits()
        solutions, report = dual.combine_independent_ik_candidates(
            arm1,
            arm2,
            seed_q=np.zeros(12),
            q_lo=q_lo,
            q_hi=q_hi,
            collision_filter=only_cross_pair_is_safe,
        )

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].shape, (4, 12))
        self.assertEqual(report["n_pair_candidates"], 4)
        self.assertEqual(report["n_pair_trials"], 4)
        self.assertEqual(report["n_collision_rejected"], 3)
        self.assertEqual(report["n_safe_pairs"], 1)
        self.assertEqual(report["status"], "SEPARATE_IK_OK")
        np.testing.assert_array_equal(
            solutions,
            np.concatenate((arm1[0], arm2[1])).reshape(1, 12),
        )

    def test_collision_filter_runs_before_max_solutions_truncation(self) -> None:
        arm1 = np.zeros((1, 6), dtype=np.float64)
        arm2 = np.zeros((2, 6), dtype=np.float64)
        arm1[0, 0] = 0.1
        arm2[:, 0] = [0.1, 0.2]
        checked = []

        def reject_nearest(candidates):
            checked.append(np.asarray(candidates).copy())
            return np.array([False, True]), {"checked": True}

        q_lo, q_hi = self._limits()
        solutions, report = dual.combine_independent_ik_candidates(
            arm1,
            arm2,
            seed_q=np.zeros(12),
            q_lo=q_lo,
            q_hi=q_hi,
            collision_filter=reject_nearest,
            max_solutions=1,
        )

        # max_solutions limits safe outputs, not the candidates presented to
        # the collision checker; otherwise the farther safe pair is lost.
        self.assertEqual(checked[0].shape, (2, 12))
        np.testing.assert_array_equal(
            solutions,
            np.concatenate((arm1[0], arm2[1])).reshape(1, 12),
        )
        self.assertEqual(report["n_pair_trials"], 2)
        self.assertEqual(report["n_collision_rejected"], 1)
        self.assertEqual(report["n_safe_pairs"], 1)
        self.assertEqual(report["n_returned"], 1)
        self.assertFalse(report["output_truncated"])

    def test_collision_filter_exception_fails_closed(self) -> None:
        def unavailable(_candidates):
            raise RuntimeError("collision backend unavailable")

        q_lo, q_hi = self._limits()
        solutions, report = dual.combine_independent_ik_candidates(
            np.zeros((1, 6)),
            np.zeros((1, 6)),
            seed_q=np.zeros(12),
            q_lo=q_lo,
            q_hi=q_hi,
            collision_filter=unavailable,
        )

        self.assertEqual(solutions.shape, (0, 12))
        self.assertEqual(report["status"], "IK_PAIR_COLLISION_CHECK_ERROR")
        self.assertIn("collision backend unavailable", report["reason"])
        self.assertEqual(report["n_safe_pairs"], 0)

    def test_inactive_arm_is_fixed_to_fixed_q_without_calling_its_solver(self) -> None:
        collision_filter = mock.Mock(
            side_effect=lambda candidates: (
                np.ones(len(candidates), dtype=bool),
                {"checked": True},
            )
        )
        context = self._context(collision_filter)
        seed_q = np.linspace(-0.6, 0.5, 12)
        fixed_q = seed_q.copy()
        fixed_q[6:] = np.linspace(10.0, 10.5, 6)
        arm1_solution = np.linspace(1.0, 1.5, 6).reshape(1, 6)
        arm_result = types.SimpleNamespace(status="ARM1_OK")

        with mock.patch.object(
            dual,
            "solve_arm_ik_solutions",
            return_value=(arm1_solution, arm_result, self._arm_report()),
        ) as solve_arm:
            solutions, result, report = dual.solve_dual_ik_solutions(
                context,
                _pose("arm1_target"),
                _pose("arm2_hold"),
                "second_TCP_LINK",
                seed_q=seed_q,
                max_solutions=1,
                active_arms=(True, False),
                fixed_q=fixed_q,
            )

        solve_arm.assert_called_once()
        self.assertIs(solve_arm.call_args.args[0], context.arm_solvers[0])
        np.testing.assert_array_equal(
            solve_arm.call_args.kwargs["seed_q"], seed_q[:6]
        )
        np.testing.assert_array_equal(solutions[0, :6], arm1_solution[0])
        np.testing.assert_array_equal(solutions[0, 6:], fixed_q[6:])
        np.testing.assert_array_equal(
            collision_filter.call_args.args[0][0, 6:], fixed_q[6:]
        )
        self.assertEqual(result.status, "SEPARATE_IK_OK")
        self.assertEqual(report["active_arms"], [True, False])
        self.assertEqual(
            report["arm_reports"][1]["status"], "INACTIVE_ARM_FIXED"
        )

    def test_arm2_target_is_transformed_to_its_local_base(self) -> None:
        angle = np.radians(90.0)
        root_to_arm2 = np.array([
            [np.cos(angle), -np.sin(angle), 0.0, 0.4],
            [np.sin(angle), np.cos(angle), 0.0, -0.3],
            [0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ])
        context = self._context(
            lambda candidates: (
                np.ones(len(candidates), dtype=bool),
                {"checked": True},
            ),
            root_to_arm2=root_to_arm2,
        )
        seed_q = np.linspace(-0.3, 0.8, 12)
        arm2_solution = np.linspace(2.0, 2.5, 6).reshape(1, 6)
        target2 = PoseSpec.from_rpy_deg(
            "arm2_root_target", [0.2, -0.1, 0.6], [10.0, 20.0, 30.0]
        )

        with mock.patch.object(
            dual,
            "solve_arm_ik_solutions",
            return_value=(
                arm2_solution,
                types.SimpleNamespace(status="ARM2_OK"),
                self._arm_report(),
            ),
        ) as solve_arm:
            solutions, result, _report = dual.solve_dual_ik_solutions(
                context,
                _pose("arm1_hold"),
                target2,
                "second_TCP_LINK",
                seed_q=seed_q,
                active_arms=(False, True),
                fixed_q=seed_q,
            )

        solve_arm.assert_called_once()
        self.assertIs(solve_arm.call_args.args[0], context.arm_solvers[1])
        local_target = solve_arm.call_args.args[1]
        np.testing.assert_allclose(
            local_target.position,
            root_to_arm2[:3, :3] @ target2.position + root_to_arm2[:3, 3],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            dual.quat_wxyz_to_matrix(local_target.quat_wxyz),
            root_to_arm2[:3, :3]
            @ dual.quat_wxyz_to_matrix(target2.quat_wxyz),
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            solve_arm.call_args.kwargs["seed_q"], seed_q[6:]
        )
        np.testing.assert_array_equal(solutions[0, :6], seed_q[:6])
        np.testing.assert_array_equal(solutions[0, 6:], arm2_solution[0])
        self.assertEqual(result.status, "SEPARATE_IK_OK")


class IndependentRobotConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = dual.load_dual_config(None)
        rb = cfg["robot"]
        cls.prefix = rb["dual_arm_prefix"]
        cls.robot_override = rb
        cls.urdf_path = REPO_ROOT / rb["urdf"]
        cls.robot_yml_path = (
            REPO_ROOT
            / "src/curobo/content/configs/robot"
            / rb["robot_yml"]
        )

    def test_independent_urdf_has_twelve_actuated_joints_and_no_mimic(self) -> None:
        root = ET.parse(self.urdf_path).getroot()
        actuated = [
            joint
            for joint in root.findall("joint")
            if joint.get("type") in {"revolute", "continuous", "prismatic"}
        ]
        names = [joint.get("name") for joint in actuated]

        expected_arm1 = [f"J_{i}" for i in range(1, 7)]
        expected_arm2 = [self.prefix + name for name in expected_arm1]
        self.assertEqual(names, expected_arm1 + expected_arm2)
        self.assertEqual(len(actuated), 12)
        self.assertEqual(root.findall(".//mimic"), [])

    def test_robot_yml_has_12d_vectors_and_keeps_cross_arm_collisions(self) -> None:
        with self.robot_yml_path.open("r", encoding="utf-8") as stream:
            kin = yaml.safe_load(stream)["robot_cfg"]["kinematics"]
        cspace = kin["cspace"]

        expected_arm1 = [f"J_{i}" for i in range(1, 7)]
        expected_arm2 = [self.prefix + name for name in expected_arm1]
        self.assertEqual(cspace["joint_names"], expected_arm1 + expected_arm2)
        for key in ("retract_config", "null_space_weight", "cspace_distance_weight"):
            with self.subTest(cspace_field=key):
                self.assertEqual(len(cspace[key]), 12)
                self.assertEqual(cspace[key][:6], cspace[key][6:])

        # The task-level override passed to MotionGen must also match 12 DOF.
        self.assertEqual(len(self.robot_override["cspace_distance_weight"]), 12)

        ignore = kin["self_collision_ignore"]
        self.assertTrue(any(not name.startswith(self.prefix) for name in ignore))
        self.assertTrue(any(name.startswith(self.prefix) for name in ignore))
        for link, ignored_links in ignore.items():
            for ignored in ignored_links:
                with self.subTest(link=link, ignored=ignored):
                    self.assertEqual(
                        link.startswith(self.prefix),
                        ignored.startswith(self.prefix),
                        "cross-arm pairs must not be put in self_collision_ignore",
                    )


class PreMotionIKFilterTest(unittest.TestCase):
    @staticmethod
    def _items(count: int):
        arm1 = []
        arm2 = []
        for index in range(count):
            local = [float(index), 0.1, 0.2]
            arm1.append({
                "index": index,
                "row": index,
                "col": 0,
                "position": local,
                "root_position": list(local),
            })
            arm2.append({
                "index": index,
                "row": index,
                "col": 0,
                "position": list(local),
                "root_position": [-float(index), -0.1, 0.2],
            })
        return arm1, arm2

    def test_skip_scans_later_source_items_until_two_pairs_are_reachable(self) -> None:
        arm1, arm2 = self._items(5)
        outcomes = {
            0: (False, True),
            1: (True, True),
            2: (True, False),
            3: (True, True),
            4: (True, True),
        }
        calls = []

        def probe(source_index, item1, item2):
            self.assertEqual(item1["index"], item2["index"])
            calls.append(source_index)
            arm1_ok, arm2_ok = outcomes[source_index]
            ok = arm1_ok and arm2_ok
            return ok, {
                "status": "IK_REACHABLE" if ok else "NO_PRIMARY_IK",
                "arm1_reachable": arm1_ok,
                "arm2_reachable": arm2_ok,
            }

        selected1, selected2, skipped = dual.filter_dual_items_by_primary_ik(
            arm1, arm2, 2, "skip", probe
        )

        self.assertEqual([item["index"] for item in selected1], [1, 3])
        self.assertEqual([item["index"] for item in selected2], [1, 3])
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(
            [item["source_item_index"] for item in skipped], [0, 2]
        )
        self.assertFalse(skipped[0]["arm1_reachable"])
        self.assertFalse(skipped[1]["arm2_reachable"])

    def test_stop_preserves_first_n_and_never_runs_ik_prefilter(self) -> None:
        arm1, arm2 = self._items(4)
        probe = mock.Mock(side_effect=AssertionError("stop must not prefilter"))

        selected1, selected2, skipped = dual.filter_dual_items_by_primary_ik(
            arm1, arm2, 2, "stop", probe
        )

        self.assertEqual([item["index"] for item in selected1], [0, 1])
        self.assertEqual([item["index"] for item in selected2], [0, 1])
        self.assertEqual(skipped, [])
        probe.assert_not_called()

    def test_all_explicit_ik_failures_produce_no_selected_pair(self) -> None:
        arm1, arm2 = self._items(3)
        probe = mock.Mock(return_value=(False, {
            "status": "NO_PRIMARY_IK",
            "arm1_reachable": False,
            "arm2_reachable": False,
        }))

        selected1, selected2, skipped = dual.filter_dual_items_by_primary_ik(
            arm1, arm2, 2, "skip", probe
        )

        self.assertEqual(selected1, [])
        self.assertEqual(selected2, [])
        self.assertEqual(len(skipped), 3)
        self.assertEqual(probe.call_count, 3)

    def test_non_ik_failure_is_fail_closed_instead_of_skipped(self) -> None:
        arm1, arm2 = self._items(3)
        probe = mock.Mock(return_value=(False, {
            "status": "PLANNING_FAILED",
            "reason": "trajectory optimizer failed after IK",
        }))

        with self.assertRaisesRegex(RuntimeError, "PLANNING_FAILED"):
            dual.filter_dual_items_by_primary_ik(
                arm1, arm2, 2, "skip", probe
            )

        probe.assert_called_once()

    def test_probe_exception_propagates_instead_of_becoming_skip(self) -> None:
        arm1, arm2 = self._items(3)
        probe = mock.Mock(side_effect=RuntimeError("IK backend unavailable"))

        with self.assertRaisesRegex(RuntimeError, "IK backend unavailable"):
            dual.filter_dual_items_by_primary_ik(
                arm1, arm2, 2, "skip", probe
            )

        probe.assert_called_once()

    def test_grasp_prefix_follows_dynamic_place_boundary(self) -> None:
        for extra_approaches in (0, 1, 4):
            with self.subTest(extra_approaches=extra_approaches):
                grasp_side = [
                    _pose(f"arm1_item0_extra_{index}")
                    for index in range(extra_approaches)
                ] + [
                    _pose("arm1_item0_g_lift_in"),
                    _pose("arm1_item0_grasp"),
                    _pose("arm1_item0_lift"),
                ]
                sequence = grasp_side + [
                    _pose("arm1_item0_p_lift_in"),
                    _pose("arm1_item0_place"),
                ]

                prefix = dual.grasp_prefix(sequence)

                self.assertEqual(
                    [pose.name for pose in prefix],
                    [pose.name for pose in grasp_side],
                )

    def test_probe_resets_q_home_for_every_prefix_waypoint_and_candidate(self) -> None:
        choices = [
            dual.AngleChoice(0.0, 0.0),
            dual.AngleChoice(0.0, 90.0),
            dual.AngleChoice(30.0, 0.0),
        ]
        q_home = np.linspace(-0.3, 0.3, 12)
        seeds = []
        target_calls = []

        def make_sequence(
            _item, _place, choice, _pp, item_index, _transform, prefix
        ):
            stem = f"{prefix}item{item_index}_g{choice.grasp:g}"
            # The successful candidate intentionally has five grasp-side
            # waypoints, proving the probe does not assume a fixed prefix size.
            count = 4 if choice.grasp == 0.0 else 5
            return [
                _pose(stem + f"_wp{index}") for index in range(count)
            ] + [
                _pose(stem + "_p_lift_in"),
                _pose(stem + "_place"),
            ]

        def solve(
            _ik, target1, target2, _second_ee, seed_q=None,
            active_arms=(True, True), fixed_q=None,
        ):
            del active_arms, fixed_q
            seeds.append(np.asarray(seed_q).copy())
            active = target1 if "item7" in target1.name else target2
            target_calls.append(active.name)
            # Reject the second waypoint of the first unique grasp candidate;
            # the later candidate must restart from q_home and may succeed.
            if "_g0_" in active.name and active.name.endswith("_wp1"):
                return None, types.SimpleNamespace(status="NO_IK")
            return q_home + 0.01, types.SimpleNamespace(status="OK")

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                dual, "arm_primary_choices", return_value=choices
            ))
            stack.enter_context(mock.patch.object(
                dual, "make_arm_sequence", side_effect=make_sequence
            ))
            stack.enter_context(mock.patch.object(
                dual, "pose_from_fk",
                side_effect=lambda _mg, _q, _link, name: _pose(name),
            ))
            stack.enter_context(mock.patch.object(
                dual, "solve_dual_ik", side_effect=solve
            ))
            ok, detail = dual.probe_dual_item_primary_ik(
                types.SimpleNamespace(ik_solver=object()),
                q_home,
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.4, 0.5, 0.6],
                7,
                {},
                {},
                (np.eye(4), np.eye(4)),
                np.full(12, -10.0),
                np.full(12, 10.0),
                ("TCP_LINK", "second_TCP_LINK"),
            )

        self.assertTrue(ok)
        self.assertEqual(detail["status"], "IK_REACHABLE")
        self.assertTrue(detail["arm1_reachable"])
        self.assertTrue(detail["arm2_reachable"])
        self.assertEqual(detail["probe"]["arm1"]["n_ik_calls"], 7)
        self.assertEqual(detail["probe"]["arm2"]["n_ik_calls"], 7)
        self.assertEqual(len(target_calls), 14)
        self.assertEqual(sum("_g0_" in name for name in target_calls), 4)
        self.assertEqual(sum("_g30_" in name for name in target_calls), 10)
        # Returning q_home + 0.01 must never become the seed of the next call.
        for seed in seeds:
            np.testing.assert_array_equal(seed, q_home)


class PipelineIKRetryClassifierTest(unittest.TestCase):
    selected = [
        {"pipeline_item_index": 0, "source_item_index": 8},
        {"pipeline_item_index": 1, "source_item_index": 9},
        {"pipeline_item_index": 2, "source_item_index": 10},
    ]

    @staticmethod
    def _ik_failure(start: int, event: int):
        return {
            "pipeline_block": "test_block",
            "angle_stage": "primary",
            "stage": "joint_ik_prescreen",
            "status": "IK_PRESCREEN_FAIL",
            "global_event_start": start,
            "event": event,
            # Deliberately misleading names: attribution must only use numeric
            # timeline metadata, never parse these strings.
            "targets": ["arm1_i999_fake", "arm2_i777_fake"],
        }

    def test_all_terminal_conflicts_uniquely_map_dense_item_to_source(self) -> None:
        failures = [
            self._ik_failure(3, 3),       # global 6 -> dense {0, 1}
            {
                "status": "DOWNSTREAM_BACKTRACK",
                "global_event_start": 3,
            },
            self._ik_failure(9, 0),       # global 9 -> dense {1}
            self._ik_failure(8, 1),       # same deepest frontier via 8 + 1
        ]

        decision = dual.classify_pipeline_ik_retry(
            failures,
            {"node_budget_exhausted": False},
            self.selected,
            phase=3,
            sequence_length=6,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(
            decision["status"], "PIPELINE_JOINT_IK_SEARCH_FAILED"
        )
        self.assertEqual(decision["pipeline_item_index"], 1)
        self.assertEqual(decision["source_item_index"], 9)
        self.assertEqual(decision["deepest_global_event"], 9)
        self.assertEqual(decision["terminal_failure_count"], 3)
        self.assertEqual(
            decision["terminal_conflict_item_indices"], [[0, 1], [1], [1]]
        )

    def test_cross_source_frontier_is_ambiguous_and_not_retryable(self) -> None:
        # With L=6 and phase=3, global event 6 combines A item1 with B item0.
        failures = [self._ik_failure(3, 3), self._ik_failure(5, 1)]

        decision = dual.classify_pipeline_ik_retry(
            failures,
            {"node_budget_exhausted": False},
            self.selected,
            phase=3,
            sequence_length=6,
        )

        self.assertIsNone(decision)

    def test_conflicting_singletons_do_not_fall_back_to_deepest_only(self) -> None:
        failures = [
            self._ik_failure(0, 0),  # global 0 -> singleton {0}
            self._ik_failure(9, 0),  # global 9 -> singleton {1}
        ]

        decision = dual.classify_pipeline_ik_retry(
            failures,
            {"node_budget_exhausted": False},
            self.selected,
            phase=3,
            sequence_length=6,
        )

        self.assertIsNone(decision)

    def test_singleton_and_cross_item_conflict_intersect_at_one_culprit(self) -> None:
        failures = [
            self._ik_failure(0, 0),  # singleton {0}
            self._ik_failure(3, 3),  # global 6 -> cross-source {0, 1}
        ]

        decision = dual.classify_pipeline_ik_retry(
            failures,
            {"node_budget_exhausted": False},
            self.selected,
            phase=3,
            sequence_length=6,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision["pipeline_item_index"], 0)
        self.assertEqual(decision["source_item_index"], 8)
        self.assertEqual(decision["deepest_global_event"], 6)
        self.assertEqual(
            decision["terminal_conflict_item_indices"], [[0], [0, 1]]
        )

    def test_non_ik_node_cap_and_bad_metadata_all_fail_closed(self) -> None:
        valid = self._ik_failure(3, 0)
        cases = {
            "trajopt": [{
                "stage": "plan_schedule",
                "status": "TRAJOPT_FAIL",
                "global_event_start": 3,
                "event": 0,
            }],
            "ik_exception": [{
                **valid, "status": "IK_PRESCREEN_EXCEPTION"
            }],
            "criterion": [{
                **valid, "status": "IK_PRESCREEN_CRITERION"
            }],
            "stage2_is_not_a_primary_terminal": [{
                **valid, "angle_stage": "stage2"
            }],
            "mixed_terminal": [valid, {
                "stage": "plan_schedule",
                "status": "COLLISION",
                "global_event_start": 3,
                "event": 0,
            }],
            "missing_event": [{
                key: value for key, value in valid.items() if key != "event"
            }],
            "string_event": [{**valid, "event": "0"}],
            "boolean_start": [{**valid, "global_event_start": True}],
            "negative_event": [{**valid, "event": -1}],
            "out_of_timeline": [{**valid, "global_event_start": 999}],
            "backtrack_only": [{"status": "DOWNSTREAM_BACKTRACK"}],
        }
        for name, failures in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(dual.classify_pipeline_ik_retry(
                    failures,
                    {"node_budget_exhausted": False},
                    self.selected,
                    phase=3,
                    sequence_length=6,
                ))

        self.assertIsNone(dual.classify_pipeline_ik_retry(
            [valid],
            {"node_budget_exhausted": True},
            self.selected,
            phase=3,
            sequence_length=6,
        ))


class PreMotionIKMainFlowTest(unittest.TestCase):
    @staticmethod
    def _source_items(count: int):
        return [
            {
                "index": index,
                "row": index,
                "col": 0,
                "position": [float(index), 0.1, 0.2],
            }
            for index in range(count)
        ]

    @staticmethod
    def _successful_pipeline(n_items: int):
        choice = dual.AngleChoice(0.0, 0.0)
        sequence_len = 6
        phase = 3
        arm1_sequences = [
            [_pose(f"arm1_i{item}_s{stage}") for stage in range(sequence_len)]
            for item in range(n_items)
        ]
        arm2_sequences = [
            [_pose(f"arm2_i{item}_s{stage}") for stage in range(sequence_len)]
            for item in range(n_items)
        ]
        segments = []
        for event in range(phase + n_items * sequence_len):
            shifted2 = event - phase
            segments.append({
                "global_event_index": event,
                "arm1_item_index": (
                    event // sequence_len
                    if event < n_items * sequence_len else None
                ),
                "arm2_item_index": (
                    shifted2 // sequence_len
                    if 0 <= shifted2 < n_items * sequence_len else None
                ),
            })
        position = np.zeros((2, 12), dtype=np.float64)
        return dual.PlannedPipeline(
            position=position,
            velocity=np.zeros_like(position),
            acceleration=np.zeros_like(position),
            dt=0.02,
            q_end=position[-1].copy(),
            segments=segments,
            chunks=[],
            arm1_choices=[choice] * n_items,
            arm2_choices=[choice] * n_items,
            arm1_sequences=arm1_sequences,
            arm2_sequences=arm2_sequences,
            angle_stage="primary",
            search_nodes=1,
            block_attempts={"mock": 1},
        )

    @staticmethod
    def _shared_source_fixture():
        return [
            {
                "index": 101,
                "row": 0,
                "col": 0,
                "position": [-0.2, 0.1, 0.2],
            },
            {
                "index": 202,
                "row": 0,
                "col": 1,
                "position": [-0.8, 0.1, 0.2],
            },
        ]

    @staticmethod
    def _two_branch_home_roots():
        q0 = np.concatenate([np.full(6, 0.1), np.full(6, 0.2)])
        q1 = np.concatenate([np.full(6, 0.3), np.full(6, 0.4)])
        choice1 = dual.AngleChoice(30.0, 10.0)
        choice2 = dual.AngleChoice(60.0, 20.0)
        # Separate PoseSpec instances with identical values prove that each
        # pipeline attempt receives the terminal owned by that root record.
        home10 = _pose("arm1_home_first_pregrasp", -0.2)
        home20 = _pose("arm2_home_first_pregrasp", -0.8)
        home11 = _pose("arm1_home_first_pregrasp", -0.2)
        home21 = _pose("arm2_home_first_pregrasp", -0.8)
        roots = [
            dual.FirstPregraspHomeRoot(
                q_home=q0,
                arm1_home=home10,
                arm2_home=home20,
                arm1_choice=choice1,
                arm2_choice=choice2,
                angle_pair_trial=1,
                ik_branch_index=0,
                distance_to_seed=float(np.linalg.norm(q0)),
            ),
            dual.FirstPregraspHomeRoot(
                q_home=q1,
                arm1_home=home11,
                arm2_home=home21,
                arm1_choice=choice1,
                arm2_choice=choice2,
                angle_pair_trial=1,
                ik_branch_index=1,
                distance_to_seed=float(np.linalg.norm(q1)),
            ),
        ]
        return roots, q0, q1

    @staticmethod
    def _home_enumeration_report(n_returned=2, budget_exhausted=False):
        return {
            "status": "FIRST_PREGRASP_HOME_ROOTS_OK",
            "arm1_source_item_index": 101,
            "arm2_source_item_index": 202,
            "n_angle_pair_trials": 1,
            "n_raw_ik_branches": 2,
            "n_outside_limit_branches": 0,
            "n_roots_available": 2,
            "n_roots": n_returned,
            "n_roots_returned": n_returned,
            "max_angle_pair_trials": 24,
            "max_branches_per_pair": 4,
            "max_roots": 1 if budget_exhausted else 12,
            "angle_pair_budget_exhausted": False,
            "root_budget_exhausted": budget_exhausted,
            "search_incomplete": budget_exhausted,
            "failure_status_counts": {},
        }

    @staticmethod
    def _reachable_shared_probe(
        _mg,
        _q_home,
        _item,
        _place,
        _item_index,
        arm,
        *_args,
    ):
        choice = (
            dual.AngleChoice(30.0, 10.0)
            if int(arm) == 1 else dual.AngleChoice(60.0, 20.0)
        )
        return True, {
            "status": "IK_REACHABLE",
            "probe": {"selected_choice": choice.to_dict()},
        }

    def _run_main(
        self,
        probe_side_effect,
        pipeline_result=None,
        *,
        pipeline_side_effect=None,
        source_count=5,
        task_layout="same_local_grid",
        single_probe_side_effect=None,
        home_roots_result=None,
        source_items_override=None,
        mount_t12=None,
        link0_target_transform=None,
        linear_free_axis=None,
        home_root_cap=None,
        self_collision_side_effect=None,
        inter_arm_side_effect=None,
    ):
        cfg = dual.load_dual_config(None)
        cfg["pick_place"]["on_fail"]["mode"] = "skip"
        cfg["dual_arm"]["task_layout"] = task_layout
        if link0_target_transform is not None:
            if isinstance(link0_target_transform, dict):
                cfg["pick_place"]["link0_target_transform"] = (
                    link0_target_transform
                )
            else:
                cfg["pick_place"]["link0_target_transform"] = np.asarray(
                    link0_target_transform, dtype=np.float64
                ).tolist()
        if linear_free_axis is not None:
            cfg["pick_place"]["linear_move"]["free_axis"] = str(
                linear_free_axis
            )
        if home_root_cap is not None:
            cfg["dual_arm"]["max_home_root_trials"] = int(home_root_cap)
        cfg["pick_place"]["home"]["joint_deg"] = [0.0] * 6
        cfg["pick_place"]["angle_search"]["stage2"]["enable"] = False
        cfg["workspace"]["check_after_plan"] = False
        cfg["workspace"]["report_gripper_extent"] = False
        cfg["workspace"].setdefault("wall", {})["enable"] = False
        cfg["output"]["add_timestamp"] = False
        source_items = (
            self._source_items(source_count)
            if source_items_override is None
            else list(source_items_override)
        )
        fixture_t12 = (
            np.eye(4, dtype=np.float64)
            if mount_t12 is None else np.asarray(mount_t12, dtype=np.float64)
        )
        choice = dual.AngleChoice(0.0, 0.0)
        arm_sequence = [_pose(f"stage_{index}") for index in range(6)]

        class FakeTensor:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float64)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        expected_joints = [f"J_{index}" for index in range(1, 7)] + [
            f"second_J_{index}" for index in range(1, 7)
        ]
        fake_limits = types.SimpleNamespace(position=[
            FakeTensor(np.full(12, -10.0)),
            FakeTensor(np.full(12, 10.0)),
        ])
        fake_mg = types.SimpleNamespace(
            ik_solver=object(),
            joint_names=expected_joints,
            kinematics=types.SimpleNamespace(
                get_joint_limits=lambda: fake_limits
            ),
        )
        fake_curobo = types.ModuleType("curobo")
        fake_curobo.__path__ = []
        fake_util_file = types.ModuleType("curobo.util_file")
        fake_util_file.get_assets_path = lambda: str(
            REPO_ROOT / "src/curobo/content/assets"
        )

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            out_dir = Path(tmp) / "ik_filter_result"
            cfg["output"]["dir"] = str(out_dir)
            parser = mock.Mock()
            parser.parse_args.return_value = types.SimpleNamespace(
                config=None, max_items=2
            )
            rb = cfg["robot"]
            robot_dict = {
                "robot_cfg": {
                    "kinematics": {
                        "urdf_path": str((REPO_ROOT / rb["urdf"]).resolve())
                    }
                }
            }

            stack.enter_context(mock.patch.dict(
                sys.modules,
                {"curobo": fake_curobo, "curobo.util_file": fake_util_file},
            ))
            stack.enter_context(mock.patch.object(
                dual, "dual_argparser", return_value=parser
            ))
            stack.enter_context(mock.patch.object(
                dual, "load_dual_config", return_value=cfg
            ))
            stack.enter_context(mock.patch.object(
                dual, "apply_dual_cli", side_effect=lambda value, _args: value
            ))
            stack.enter_context(mock.patch.object(
                dual,
                "mounts_relative_transform",
                return_value=(fixture_t12, {}),
            ))
            stack.enter_context(mock.patch.object(
                dual, "urdf_link_transform", return_value=fixture_t12
            ))
            build = stack.enter_context(mock.patch.object(
                dual, "build_grasp_points", return_value=source_items
            ))
            stack.enter_context(mock.patch.object(
                dual, "arm_primary_choices", return_value=[choice]
            ))
            stack.enter_context(mock.patch.object(
                dual, "make_arm_sequence", return_value=arm_sequence
            ))
            stack.enter_context(mock.patch.object(
                dual, "make_world_config", return_value={}
            ))
            stack.enter_context(mock.patch.object(
                dual, "load_robot_cfg_dict", return_value=robot_dict
            ))
            stack.enter_context(mock.patch.object(
                dual, "make_motion_gen", return_value=fake_mg
            ))
            stack.enter_context(mock.patch.object(
                dual,
                "make_separate_arm_ik_context",
                return_value=types.SimpleNamespace(
                    max_candidates_per_arm=8,
                    q_lo=np.full(12, -10.0),
                    q_hi=np.full(12, 10.0),
                ),
            ))
            stack.enter_context(mock.patch.object(
                dual, "solve_dual_ik",
                return_value=(np.zeros(12), types.SimpleNamespace(status="OK")),
            ))
            probe = stack.enter_context(mock.patch.object(
                dual, "probe_dual_item_primary_ik",
                side_effect=probe_side_effect,
            ))
            single_probe = stack.enter_context(mock.patch.object(
                dual,
                "probe_arm_item_primary_ik",
                side_effect=(
                    single_probe_side_effect
                    if single_probe_side_effect is not None
                    else AssertionError("unexpected single-arm IK probe")
                ),
            ))
            home_enumerator = (
                stack.enter_context(mock.patch.object(
                    dual,
                    "enumerate_first_pregrasp_home_roots",
                    return_value=home_roots_result,
                ))
                if home_roots_result is not None else None
            )
            pipeline_patch = (
                mock.patch.object(
                    dual,
                    "try_continuous_pipeline",
                    side_effect=pipeline_side_effect,
                )
                if pipeline_side_effect is not None
                else mock.patch.object(
                    dual,
                    "try_continuous_pipeline",
                    return_value=pipeline_result,
                )
            )
            pipeline = stack.enter_context(pipeline_patch)
            stack.enter_context(mock.patch.object(
                dual,
                "compute_fk_batched",
                side_effect=lambda _mg, positions, _links: {
                    "TCP_LINK/pos": np.zeros((len(positions), 3)),
                    "TCP_LINK/quat": np.tile(
                        [1.0, 0.0, 0.0, 0.0], (len(positions), 1)
                    ),
                    "second_TCP_LINK/pos": np.ones((len(positions), 3)),
                    "second_TCP_LINK/quat": np.tile(
                        [1.0, 0.0, 0.0, 0.0], (len(positions), 1)
                    ),
                },
            ))
            self_check = stack.enter_context(
                mock.patch.object(
                    dual,
                    "self_collision_report",
                    side_effect=self_collision_side_effect,
                )
                if self_collision_side_effect is not None
                else mock.patch.object(
                    dual,
                    "self_collision_report",
                    return_value={"checked": True, "n_collision": 0},
                )
            )
            pair_check = stack.enter_context(
                mock.patch.object(
                    dual,
                    "inter_arm_report",
                    side_effect=inter_arm_side_effect,
                )
                if inter_arm_side_effect is not None
                else mock.patch.object(
                    dual,
                    "inter_arm_report",
                    return_value={
                        "checked": True,
                        "n_collision_points": 0,
                        "margin_mm": 0.0,
                    },
                )
            )
            stack.enter_context(mock.patch.object(
                dual, "compute_joint_motion_report", return_value={}
            ))
            stack.enter_context(mock.patch.object(
                dual, "compute_joint_limit_margin_report", return_value={}
            ))
            saver = stack.enter_context(mock.patch.object(
                dual,
                "save_trajectory",
                return_value=(
                    out_dir / "trajectory.npz",
                    out_dir / "trajectory_meta.json",
                ),
            ))
            stack.enter_context(mock.patch("builtins.print"))

            return_code = dual.main()
            marker_path = out_dir / "plan_failed.json"
            marker = (
                json.loads(marker_path.read_text(encoding="utf-8"))
                if marker_path.is_file()
                else None
            )
            skipped_path = out_dir / "plan_skipped.json"
            skipped_payload = (
                json.loads(skipped_path.read_text(encoding="utf-8"))
                if skipped_path.is_file()
                else None
            )
            self._last_single_probe = single_probe
            self._last_home_enumerator = home_enumerator
            self._last_save = saver
            self._last_self_check = self_check
            self._last_pair_check = pair_check

        return return_code, marker, skipped_payload, build, probe, pipeline

    def test_main_left_multiplies_all_task_targets_without_changing_mount(self) -> None:
        transform_cfg = {
            "position": [0.3, -0.2, 0.1],
            "rotation": {"axis": "z", "angle_deg": 90.0},
        }
        correction = parse_rigid_transform_matrix(
            transform_cfg, "test main transform"
        )
        t12 = np.array([
            [-1.0, 0.0, 0.0, -0.8],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        def reachable(*_args):
            return True, {"status": "IK_REACHABLE"}

        code, marker, skipped_payload, _build, probe, pipeline = self._run_main(
            reachable,
            (self._successful_pipeline(2), [], {"search_nodes": 1}),
            mount_t12=t12,
            link0_target_transform=transform_cfg,
        )

        self.assertEqual(code, 0)
        self.assertIsNone(marker)
        self.assertIsNone(skipped_payload)
        expected_transforms = (correction, correction @ t12)
        self.assertEqual(probe.call_count, 2)
        for call in probe.call_args_list:
            np.testing.assert_allclose(call.args[9][0], expected_transforms[0])
            np.testing.assert_allclose(call.args[9][1], expected_transforms[1])
        np.testing.assert_allclose(pipeline.call_args.args[9][0], correction)
        np.testing.assert_allclose(
            pipeline.call_args.args[9][1], correction @ t12
        )

        meta = self._last_save.call_args.args[8]
        np.testing.assert_allclose(
            meta["robot"]["arm2_in_arm1_transform"], t12
        )
        np.testing.assert_allclose(
            meta["robot"]["link0_target_transform"], correction
        )
        np.testing.assert_allclose(
            meta["robot"]["arm_target_transforms"]["arm2"], correction @ t12
        )

        raw_home = np.asarray(
            dual.load_dual_config(None)["pick_place"]["home"]["position"],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            meta["home_poses"]["arm1"]["position"],
            dual.transform_points(raw_home[None, :], correction)[0],
        )
        np.testing.assert_allclose(
            meta["home_poses"]["arm2"]["position"],
            dual.transform_points(raw_home[None, :], correction @ t12)[0],
        )

        raw_places = meta["place_positions_raw_arm1_base"]
        np.testing.assert_allclose(
            meta["place_positions_arm1_base"]["arm1"],
            dual.transform_points(
                np.asarray(raw_places["arm1"])[None, :], correction
            )[0],
        )
        np.testing.assert_allclose(
            meta["place_positions_arm1_base"]["arm2"],
            dual.transform_points(
                np.asarray(raw_places["arm2"])[None, :], correction
            )[0],
        )
        first_item = meta["grid"]["selected_item_map"][0]
        np.testing.assert_allclose(
            first_item["arm1_root_position"],
            dual.transform_points(
                np.asarray(first_item["arm1_raw_root_position"])[None, :],
                correction,
            )[0],
        )

    def test_main_rejects_linear_axis_that_disagrees_with_base_z_lift(self) -> None:
        code, marker, _skipped, _build, probe, pipeline = self._run_main(
            AssertionError("target probes must not run for an invalid axis"),
            (self._successful_pipeline(2), [], {}),
            linear_free_axis="x",
        )

        self.assertEqual(code, 2)
        self.assertEqual(marker["stage"], "planning_in_progress")
        probe.assert_not_called()
        pipeline.assert_not_called()

    def test_main_maps_rotated_local_z_to_effective_root_axis(self) -> None:
        rotate_y_90 = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        code, marker, skipped_payload, _build, _probe, pipeline = self._run_main(
            lambda *_args: (True, {"status": "IK_REACHABLE"}),
            (self._successful_pipeline(2), [], {}),
            link0_target_transform=rotate_y_90,
        )

        self.assertEqual(code, 0)
        self.assertIsNone(marker)
        self.assertIsNone(skipped_payload)
        self.assertEqual(pipeline.call_args.args[12]["configured_free_axis"], "z")
        self.assertEqual(pipeline.call_args.args[12]["free_axis"], "x")

    def test_main_rejects_non_cardinal_rotated_linear_axis(self) -> None:
        angle = np.radians(45.0)
        rotate_y_45 = np.array([
            [np.cos(angle), 0.0, np.sin(angle), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        code, marker, _skipped, _build, probe, pipeline = self._run_main(
            AssertionError("target probes must not run for a diagonal axis"),
            (self._successful_pipeline(2), [], {}),
            link0_target_transform=rotate_y_45,
        )

        self.assertEqual(code, 2)
        self.assertEqual(marker["stage"], "planning_in_progress")
        probe.assert_not_called()
        pipeline.assert_not_called()

    def test_all_unreachable_fails_before_pipeline_is_called(self) -> None:
        def no_ik(_mg, _q_home, _item1, _item2, _place1, _place2,
                  item_index, *_args):
            return False, {
                "status": "NO_PRIMARY_IK",
                "arm1_reachable": item_index % 2 == 0,
                "arm2_reachable": item_index % 2 == 1,
            }

        code, marker, skipped_payload, build, probe, pipeline = self._run_main(
            no_ik, (None, [], {})
        )

        self.assertEqual(code, 4)
        self.assertIsNone(skipped_payload)
        self.assertEqual(marker["stage"], "no_primary_grasp_ik_items")
        self.assertEqual(marker["n_candidates_scanned"], 5)
        self.assertEqual(probe.call_count, 5)
        pipeline.assert_not_called()
        # skip interprets max-items as a reachable quota, so grid construction
        # itself must remain uncapped.
        self.assertIsNone(build.call_args.args[1])

    def test_shared_empty_home_roots_do_not_refilter_or_plan(self) -> None:
        selected = dual.AngleChoice(30.0, 10.0)

        def single_probe(
            _mg,
            _q_home,
            _item,
            _place,
            _item_index,
            _arm,
            *_args,
        ):
            return True, {
                "status": "IK_REACHABLE",
                "probe": {"selected_choice": selected.to_dict()},
            }

        home_report = {
            "status": "NO_FIRST_PREGRASP_HOME_IK",
            "n_angle_pair_trials": 3,
            "n_roots": 0,
        }
        correction = np.array([
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ])
        code, marker, skipped_payload, build, paired_probe, pipeline = (
            self._run_main(
                probe_side_effect=AssertionError(
                    "shared layout must not use paired source filtering"
                ),
                pipeline_result=(None, [], {}),
                task_layout="shared_global_outside_in",
                single_probe_side_effect=single_probe,
                home_roots_result=([], home_report),
                link0_target_transform=correction,
            )
        )

        self.assertEqual(code, 3)
        self.assertIsNone(skipped_payload)
        self.assertEqual(marker["stage"], "first_pregrasp_home_ik")
        self.assertEqual(marker["home_search"], home_report)
        self.assertEqual(len(marker["iterations"]), 1)
        self.assertFalse(marker["iterations"][0]["filter_rerun"])
        self.assertEqual(
            marker["iterations"][0]["input_first_source_indices"], [4, None]
        )
        first_item = marker["selected_item_map"]["arm1"]
        np.testing.assert_allclose(
            first_item["root_position"],
            dual.transform_points(
                np.asarray(first_item["raw_root_position"])[None, :],
                correction,
            )[0],
        )
        self.assertEqual(self._last_single_probe.call_count, 1)
        self._last_home_enumerator.assert_called_once()
        paired_probe.assert_not_called()
        pipeline.assert_not_called()
        self._last_save.assert_not_called()
        self.assertIsNone(build.call_args.args[1])

    def test_shared_root0_failure_restarts_entire_pipeline_from_root1(self) -> None:
        roots, q0, q1 = self._two_branch_home_roots()
        enumeration = self._home_enumeration_report()
        pipeline_calls = []

        def plan_pipeline(*args, **kwargs):
            pipeline_calls.append({
                "q_start": np.asarray(args[1], dtype=np.float64).copy(),
                "arm1_items": np.asarray(args[2], dtype=np.float64).copy(),
                "arm2_items": np.asarray(args[3], dtype=np.float64).copy(),
                "transforms": tuple(
                    np.asarray(value, dtype=np.float64).copy()
                    for value in args[9]
                ),
                "terminal": kwargs["arm2_terminal"],
                "home_grasps": kwargs["first_home_grasp_angles"],
            })
            if len(pipeline_calls) == 1:
                return None, [{"status": "MOCK_ROOT0_PIPELINE_FAIL"}], {
                    "search_nodes": 4,
                }
            successful = self._successful_pipeline(1)
            successful.position = np.stack([q1, q1])
            successful.velocity = np.zeros_like(successful.position)
            successful.acceleration = np.zeros_like(successful.position)
            successful.q_end = q1.copy()
            successful.arm1_choices = [roots[1].arm1_choice]
            successful.arm2_choices = [roots[1].arm2_choice]
            successful.arm2_terminal = kwargs["arm2_terminal"]
            successful.parked_joint_refs = (None, q1.copy())
            return successful, [], {"search_nodes": 2}

        t12 = np.eye(4, dtype=np.float64)
        t12[0, 3] = -1.0
        correction = np.array([
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ])
        code, marker, skipped_payload, build, paired_probe, pipeline = (
            self._run_main(
                probe_side_effect=AssertionError(
                    "shared layout must not use paired source filtering"
                ),
                pipeline_side_effect=plan_pipeline,
                task_layout="shared_global_outside_in",
                single_probe_side_effect=self._reachable_shared_probe,
                home_roots_result=(roots, enumeration),
                source_items_override=self._shared_source_fixture(),
                mount_t12=t12,
                link0_target_transform=correction,
            )
        )

        self.assertEqual(code, 0)
        self.assertIsNone(marker)
        self.assertIsNone(skipped_payload)
        self.assertEqual(pipeline.call_count, 2)
        np.testing.assert_array_equal(pipeline_calls[0]["q_start"], q0)
        np.testing.assert_array_equal(pipeline_calls[1]["q_start"], q1)
        np.testing.assert_array_equal(
            pipeline_calls[0]["arm1_items"], pipeline_calls[1]["arm1_items"]
        )
        np.testing.assert_array_equal(
            pipeline_calls[0]["arm2_items"], pipeline_calls[1]["arm2_items"]
        )
        for call in pipeline_calls:
            np.testing.assert_allclose(call["transforms"][0], correction)
            np.testing.assert_allclose(
                call["transforms"][1], correction @ t12
            )
        np.testing.assert_allclose(
            self._last_single_probe.call_args_list[0].args[8], correction
        )
        np.testing.assert_allclose(
            self._last_single_probe.call_args_list[1].args[8],
            correction @ t12,
        )
        home_transforms = self._last_home_enumerator.call_args.args[8]
        np.testing.assert_allclose(home_transforms[0], correction)
        np.testing.assert_allclose(home_transforms[1], correction @ t12)
        self.assertIs(pipeline_calls[0]["terminal"], roots[0].arm2_home)
        self.assertIs(pipeline_calls[1]["terminal"], roots[1].arm2_home)
        self.assertEqual(
            pipeline_calls[0]["home_grasps"],
            (roots[0].arm1_choice.grasp, roots[0].arm2_choice.grasp),
        )
        self.assertEqual(
            pipeline_calls[1]["home_grasps"],
            (roots[1].arm1_choice.grasp, roots[1].arm2_choice.grasp),
        )
        self.assertEqual(self._last_single_probe.call_count, 2)
        self._last_home_enumerator.assert_called_once()
        paired_probe.assert_not_called()
        self._last_save.assert_called_once()
        meta = self._last_save.call_args.args[8]
        np.testing.assert_allclose(
            meta["home_joint_deg"], np.degrees(q1), atol=5e-5
        )
        np.testing.assert_allclose(
            meta["configured_home_seed_joint_deg"], np.zeros(12)
        )
        self.assertEqual(meta["home_poses"]["arm1"], roots[1].arm1_home.to_dict())
        self.assertEqual(meta["home_poses"]["arm2"], roots[1].arm2_home.to_dict())
        home_search = meta["coordination"]["home_search"]
        self.assertEqual(home_search["selected_root_index"], 1)
        self.assertEqual(
            home_search["selected_root"]["ik_branch_index"], 1
        )
        self.assertEqual(home_search["n_root_attempts"], 2)
        self.assertEqual(home_search["total_pipeline_search_nodes"], 6)
        self.assertEqual(meta["pipeline_search"]["selected_home_root_index"], 1)
        self.assertEqual(
            meta["pipeline_search"]["selected_home_ik_branch_index"], 1
        )
        self.assertEqual(
            meta["coordination"]["arm2_terminal_home_pose"],
            roots[1].arm2_home.to_dict(),
        )
        self.assertEqual(
            [attempt["home_root_index"] for attempt in meta["pipeline_attempts"]],
            [0, 1],
        )
        self.assertEqual(
            [attempt["selected_source_item_indices"]
             for attempt in meta["pipeline_attempts"]],
            [[101, 202], [101, 202]],
        )
        self.assertEqual(meta["pipeline_attempts"][0]["retry"], "next_home_root")
        self.assertTrue(meta["pipeline_attempts"][1]["success"])
        self.assertIsNone(build.call_args.args[1])

    def test_shared_root0_post_collision_is_discarded_before_root1_success(self) -> None:
        roots, q0, q1 = self._two_branch_home_roots()
        enumeration = self._home_enumeration_report()
        pipeline_calls = []

        def plan_pipeline(*args, **kwargs):
            q_start = np.asarray(args[1], dtype=np.float64).copy()
            pipeline_calls.append({
                "q_start": q_start,
                "arm1_items": np.asarray(args[2], dtype=np.float64).copy(),
                "arm2_items": np.asarray(args[3], dtype=np.float64).copy(),
                "terminal": kwargs["arm2_terminal"],
                "home_grasps": kwargs["first_home_grasp_angles"],
                "home_joint_reference": np.asarray(
                    kwargs["first_home_joint_reference"], dtype=np.float64
                ).copy(),
            })
            successful = self._successful_pipeline(1)
            successful.position = np.stack([q_start, q_start])
            successful.velocity = np.zeros_like(successful.position)
            successful.acceleration = np.zeros_like(successful.position)
            successful.q_end = q_start.copy()
            root = roots[len(pipeline_calls) - 1]
            successful.arm1_choices = [root.arm1_choice]
            successful.arm2_choices = [root.arm2_choice]
            successful.arm2_terminal = kwargs["arm2_terminal"]
            successful.parked_joint_refs = (None, q_start.copy())
            return successful, [], {"search_nodes": 2}

        t12 = np.eye(4, dtype=np.float64)
        t12[0, 3] = -1.0
        collision = {
            "checked": True,
            "n_collision_points": 1,
            "margin_mm": 0.0,
        }
        safe = {
            "checked": True,
            "n_collision_points": 0,
            "margin_mm": 0.0,
        }
        code, marker, skipped_payload, _build, paired_probe, pipeline = (
            self._run_main(
                probe_side_effect=AssertionError(
                    "shared layout must not use paired source filtering"
                ),
                pipeline_side_effect=plan_pipeline,
                task_layout="shared_global_outside_in",
                single_probe_side_effect=self._reachable_shared_probe,
                home_roots_result=(roots, enumeration),
                source_items_override=self._shared_source_fixture(),
                mount_t12=t12,
                inter_arm_side_effect=[collision, safe],
            )
        )

        self.assertEqual(code, 0)
        self.assertIsNone(marker)
        self.assertIsNone(skipped_payload)
        self.assertEqual(pipeline.call_count, 2)
        np.testing.assert_array_equal(pipeline_calls[0]["q_start"], q0)
        np.testing.assert_array_equal(pipeline_calls[1]["q_start"], q1)
        np.testing.assert_array_equal(
            pipeline_calls[0]["home_joint_reference"], q0
        )
        np.testing.assert_array_equal(
            pipeline_calls[1]["home_joint_reference"], q1
        )
        np.testing.assert_array_equal(
            pipeline_calls[0]["arm1_items"], pipeline_calls[1]["arm1_items"]
        )
        np.testing.assert_array_equal(
            pipeline_calls[0]["arm2_items"], pipeline_calls[1]["arm2_items"]
        )
        self.assertIs(pipeline_calls[0]["terminal"], roots[0].arm2_home)
        self.assertIs(pipeline_calls[1]["terminal"], roots[1].arm2_home)
        self.assertEqual(
            pipeline_calls[0]["home_grasps"], (30.0, 60.0)
        )
        self.assertEqual(
            pipeline_calls[1]["home_grasps"], (30.0, 60.0)
        )
        self.assertEqual(self._last_pair_check.call_count, 2)
        self.assertEqual(self._last_self_check.call_count, 2)
        self._last_save.assert_called_once()
        meta = self._last_save.call_args.args[8]
        attempts = meta["pipeline_attempts"]
        self.assertEqual([attempt["home_root_index"] for attempt in attempts], [0, 1])
        self.assertFalse(attempts[0]["success"])
        self.assertEqual(attempts[0]["retry"], "next_home_root")
        self.assertEqual(
            attempts[0]["post_validation_errors"],
            [{
                "check": "inter_arm_collision",
                "n_collision_points": 1,
                "margin_mm": 0.0,
            }],
        )
        self.assertEqual(
            attempts[0]["failure_status_counts"],
            {"POST_VALIDATION_REJECTED": 1},
        )
        self.assertTrue(attempts[1]["success"])
        self.assertEqual(
            attempts[0]["selected_source_item_indices"],
            attempts[1]["selected_source_item_indices"],
        )
        self.assertEqual(
            meta["coordination"]["home_search"]["selected_root_index"], 1
        )
        self.assertEqual(meta["pipeline_search"]["selected_home_root_index"], 1)
        np.testing.assert_allclose(
            meta["home_joint_deg"], np.degrees(q1), atol=5e-5
        )
        paired_probe.assert_not_called()

    def test_shared_home_root_budget_exhaustion_publishes_failure_marker(self) -> None:
        roots, q0, _q1 = self._two_branch_home_roots()
        enumeration = self._home_enumeration_report(
            n_returned=1, budget_exhausted=True
        )
        t12 = np.eye(4, dtype=np.float64)
        t12[0, 3] = -1.0
        code, marker, skipped_payload, _build, paired_probe, pipeline = (
            self._run_main(
                probe_side_effect=AssertionError(
                    "shared layout must not use paired source filtering"
                ),
                pipeline_result=(
                    None,
                    [{"status": "MOCK_ONLY_ALLOWED_ROOT_FAILED"}],
                    {"search_nodes": 3},
                ),
                task_layout="shared_global_outside_in",
                single_probe_side_effect=self._reachable_shared_probe,
                home_roots_result=([roots[0]], enumeration),
                source_items_override=self._shared_source_fixture(),
                mount_t12=t12,
                home_root_cap=1,
            )
        )

        self.assertEqual(code, 5)
        self.assertIsNone(skipped_payload)
        self.assertEqual(marker["stage"], "continuous_pipeline_root_search")
        self.assertTrue(marker["home_search"]["all_attempted_roots_failed"])
        self.assertIsNone(marker["home_search"]["selected_root_index"])
        self.assertTrue(marker["home_search"]["search_incomplete"])
        self.assertEqual(
            marker["home_search"]["result_status"],
            "HOME_ROOT_SEARCH_BUDGET_EXHAUSTED",
        )
        self.assertTrue(
            marker["home_search"]["enumeration"]["root_budget_exhausted"]
        )
        self.assertEqual(marker["home_search"]["n_root_attempts"], 1)
        self.assertEqual(len(marker["pipeline_attempts"]), 1)
        self.assertEqual(marker["pipeline_attempts"][0]["home_root_index"], 0)
        self.assertIsNone(marker["pipeline_attempts"][0]["retry"])
        self.assertEqual(
            marker["pipeline_attempts"][0]["selected_source_item_indices"],
            [101, 202],
        )
        pipeline.assert_called_once()
        np.testing.assert_array_equal(pipeline.call_args.args[1], q0)
        self.assertIs(
            pipeline.call_args.kwargs["arm2_terminal"], roots[0].arm2_home
        )
        self.assertEqual(
            pipeline.call_args.kwargs["first_home_grasp_angles"], (30.0, 60.0)
        )
        self.assertEqual(
            self._last_home_enumerator.call_args.kwargs["max_roots"], 1
        )
        self.assertEqual(self._last_single_probe.call_count, 2)
        paired_probe.assert_not_called()
        self._last_save.assert_not_called()

    def test_unique_pipeline_ik_failure_refills_quota_and_restarts_from_home(self) -> None:
        probed_items = []
        q_starts = []
        planned_source_sets = []

        def prefix_probe(
            _mg, _q_home, _item1, _item2, _place1, _place2,
            item_index, *_args,
        ):
            probed_items.append(item_index)
            ok = item_index >= 8
            return ok, {
                "status": "IK_REACHABLE" if ok else "NO_PRIMARY_IK",
                "arm1_reachable": ok,
                "arm2_reachable": ok,
            }

        def plan_pipeline(*args):
            q_starts.append(np.asarray(args[1]).copy())
            source_set = [int(point[0]) for point in args[2]]
            planned_source_sets.append(source_set)
            if len(planned_source_sets) == 1:
                self.assertEqual(source_set, [8, 9])
                return None, [
                    {
                        "pipeline_block": "block_0",
                        "angle_stage": "primary",
                        "stage": "joint_ik_prescreen",
                        "status": "IK_PRESCREEN_FAIL",
                        "global_event_start": 3,
                        "event": 0,
                        "targets": ["do_not_parse_A", "do_not_parse_B"],
                    },
                    {
                        "pipeline_block": "prime",
                        "status": "DOWNSTREAM_BACKTRACK",
                        "global_event_start": 0,
                    },
                ], {
                    "node_budget_exhausted": False,
                    "search_nodes": 12,
                }
            self.assertEqual(source_set, [9, 10])
            return self._successful_pipeline(2), [], {
                "node_budget_exhausted": False,
                "search_nodes": 3,
            }

        correction = np.array([
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ])
        code, marker, skipped_payload, _build, probe, pipeline = self._run_main(
            prefix_probe,
            pipeline_side_effect=plan_pipeline,
            source_count=11,
            link0_target_transform=correction,
        )

        self.assertEqual(code, 0)
        self.assertIsNone(marker)
        self.assertIsNotNone(skipped_payload)
        self.assertEqual(
            [
                item["source_item_index"]
                for item in skipped_payload["selected_item_map"]
            ],
            [9, 10],
        )
        self.assertEqual(
            [
                item["source_item_index"]
                for item in skipped_payload["pipeline_ik_skipped_items"]
            ],
            [8],
        )
        self.assertEqual(len(skipped_payload["pipeline_attempts"]), 2)
        self.assertEqual(
            [
                attempt["selected_source_item_indices"]
                for attempt in skipped_payload["pipeline_attempts"]
            ],
            [[8, 9], [9, 10]],
        )
        self.assertEqual(
            [attempt["success"] for attempt in skipped_payload["pipeline_attempts"]],
            [False, True],
        )
        self.assertEqual(planned_source_sets, [[8, 9], [9, 10]])
        self.assertEqual(pipeline.call_count, 2)
        self.assertEqual(probed_items, list(range(11)))
        self.assertEqual(probe.call_count, 11)
        for skipped in (
            skipped_payload["skipped_ik_items"]
            + skipped_payload["pipeline_ik_skipped_items"]
        ):
            for arm in (1, 2):
                raw_root = np.asarray(
                    skipped[f"arm{arm}_raw_root_position"], dtype=np.float64
                )
                np.testing.assert_allclose(
                    skipped[f"arm{arm}_root_position"],
                    dual.transform_points(raw_root[None, :], correction)[0],
                )
        self.assertEqual(len(q_starts), 2)
        for q_start in q_starts:
            np.testing.assert_array_equal(q_start, np.zeros(12))

    def test_pipeline_failure_after_filter_is_not_treated_as_skippable_ik(self) -> None:
        probe_indices = []

        def selective_ik(_mg, _q_home, _item1, _item2, _place1, _place2,
                         item_index, *_args):
            probe_indices.append(item_index)
            ok = item_index in (1, 3)
            return ok, {
                "status": "IK_REACHABLE" if ok else "NO_PRIMARY_IK",
                "arm1_reachable": ok,
                "arm2_reachable": ok,
            }

        code, marker, skipped_payload, _build, probe, pipeline = self._run_main(
            selective_ik,
            (None, [{"status": "TRAJOPT_FAIL"}], {"search_nodes": 7}),
        )

        self.assertEqual(code, 5)
        self.assertIsNone(skipped_payload)
        self.assertEqual(marker["stage"], "continuous_pipeline_search")
        self.assertEqual(marker["failures"][0]["status"], "TRAJOPT_FAIL")
        self.assertEqual(probe_indices, [0, 1, 2, 3])
        self.assertEqual(probe.call_count, 4)
        pipeline.assert_called_once()
        call = pipeline.call_args.args
        self.assertEqual([point[0] for point in call[2]], [1.0, 3.0])
        self.assertEqual([point[0] for point in call[3]], [1.0, 3.0])


class InterleavedScheduleTest(unittest.TestCase):
    @staticmethod
    def _path_collision_context(collision_filter, margin_mm, max_goals=2):
        return dual.SeparateArmIkContext(
            arm_solvers=(
                types.SimpleNamespace(dof=6),
                types.SimpleNamespace(dof=6),
            ),
            root_to_arm=(np.eye(4), np.eye(4)),
            q_lo=np.full(12, -10.0),
            q_hi=np.full(12, 10.0),
            collision_filter=collision_filter,
            max_candidates_per_arm=8,
            max_pair_trials=64,
            max_goal_solutions=max_goals,
            collision_margin_mm=margin_mm,
        )

    @staticmethod
    def _successful_segment_result(q_start, q_goal):
        class Array:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float64)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        position = np.stack([q_start, q_goal])
        zeros = np.zeros_like(position)
        trajectory = types.SimpleNamespace(
            position=Array(position),
            velocity=Array(zeros),
            acceleration=Array(zeros),
        )
        return types.SimpleNamespace(
            success=types.SimpleNamespace(item=lambda: True),
            status="SUCCESS",
            interpolation_dt=0.02,
            attempts=1,
            get_interpolated_plan=lambda: trajectory,
        )

    def _run_path_collision_schedule(
        self, context, goals, endpoint_result=(True, {"status": "TCP_OK"})
    ):
        q_start = np.zeros(12, dtype=np.float64)
        mg = types.SimpleNamespace(
            ik_solver=object(),
            _xtrainer_separate_ik=context,
        )
        ik_report = {"n_returned": len(goals), "solver_mode": "test"}

        def plan_segment(_mg, current, *_args, goal_q=None, **_kwargs):
            return self._successful_segment_result(current, goal_q)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                dual,
                "pose_from_fk",
                side_effect=lambda _mg, _q, _link, name: _pose(name),
            ))
            solve_ik = stack.enter_context(mock.patch.object(
                dual,
                "solve_dual_ik_solutions",
                return_value=(
                    np.asarray(goals, dtype=np.float64).copy(),
                    types.SimpleNamespace(
                        status="SEPARATE_IK_OK", report=ik_report
                    ),
                    ik_report,
                ),
            ))
            planner = stack.enter_context(mock.patch.object(
                dual, "plan_dual_segment", side_effect=plan_segment
            ))
            endpoint = stack.enter_context(mock.patch.object(
                dual,
                "validate_dual_tcp_endpoint",
                return_value=endpoint_result,
            ))
            ok, result = dual.plan_schedule(
                mg=mg,
                q_start=q_start,
                seq1=[_pose("arm1_target")],
                seq2=[_pose("arm2_target")],
                delay=0,
                pl={},
                cr={
                    "joints": [1, 2, 3, 4, 5, 6],
                    "max_joint_delta_deg": 180.0,
                    "min_limit_margin_deg": 0.0,
                },
                linear_cfg={"enable": False},
                q_lo=np.full(12, -10.0),
                q_hi=np.full(12, 10.0),
                ee_links=("arm1_ee", "arm2_ee"),
                inactive_tolerance_deg=1.0,
            )
        return ok, result, solve_ik, planner, endpoint

    def test_no_ik_jump_check_cli_disables_only_prescreen_gate(self) -> None:
        args = dual.dual_argparser().parse_args(["--no-ik-jump-check"])
        self.assertIs(args.prescreen_joint_delta_check, False)

        cfg = dual.load_dual_config(None)
        self.assertIs(
            cfg["pick_place"]["criterion"]["prescreen_joint_delta_check"],
            True,
        )
        updated = dual.apply_dual_cli(cfg, args)
        self.assertIs(
            updated["pick_place"]["criterion"][
                "prescreen_joint_delta_check"
            ],
            False,
        )

    def test_path_collision_rejects_first_goal_then_uses_second(self) -> None:
        nearest = np.full(12, 0.1, dtype=np.float64)
        second = np.full(12, 0.2, dtype=np.float64)
        first_report = {"checked": True, "candidate": "first"}
        second_report = {"checked": True, "candidate": "second"}
        collision_filter = mock.Mock(side_effect=[
            (np.array([True, False]), first_report),
            (np.array([True, True]), second_report),
        ])
        context = self._path_collision_context(collision_filter, margin_mm=5.0)

        ok, result, _solve, planner, endpoint = (
            self._run_path_collision_schedule(
                context, np.stack([nearest, second])
            )
        )

        self.assertTrue(ok, result)
        self.assertEqual(planner.call_count, 2)
        self.assertEqual(endpoint.call_count, 2)
        self.assertEqual(collision_filter.call_count, 2)
        np.testing.assert_array_equal(
            collision_filter.call_args_list[0].args[0],
            np.stack([np.zeros(12), nearest]),
        )
        np.testing.assert_array_equal(
            collision_filter.call_args_list[1].args[0],
            np.stack([np.zeros(12), second]),
        )
        np.testing.assert_array_equal(result["q_end"], second)
        segment = result["segments"][0]
        self.assertEqual(segment["ik_goal_candidate_rank"], 2)
        self.assertEqual(
            [attempt["status"] for attempt in segment["ik_goal_attempts"]],
            ["TRAJECTORY_COLLISION_MARGIN_POSTCHECK", "SUCCESS"],
        )
        rejected = segment["ik_goal_attempts"][0]
        self.assertEqual(rejected["collision_margin_mm"], 5.0)
        self.assertEqual(rejected["n_rejected_points"], 1)
        self.assertEqual(rejected["first_rejected_point"], 1)
        self.assertEqual(rejected["path_collision"], first_report)
        self.assertEqual(
            segment["trajectory_collision_margin"], second_report
        )

    def test_zero_collision_margin_does_not_recheck_trajectory(self) -> None:
        goal = np.full(12, 0.1, dtype=np.float64)
        collision_filter = mock.Mock(
            side_effect=AssertionError("margin=0 must not check path")
        )
        context = self._path_collision_context(collision_filter, margin_mm=0.0)

        ok, result, _solve, planner, _endpoint = (
            self._run_path_collision_schedule(context, goal.reshape(1, 12))
        )

        self.assertTrue(ok, result)
        planner.assert_called_once()
        collision_filter.assert_not_called()
        self.assertIsNone(
            result["segments"][0]["trajectory_collision_margin"]
        )

    def test_path_collision_checker_exception_fails_closed_immediately(self) -> None:
        nearest = np.full(12, 0.1, dtype=np.float64)
        second = np.full(12, 0.2, dtype=np.float64)
        collision_filter = mock.Mock(
            side_effect=RuntimeError("path collision checker unavailable")
        )
        context = self._path_collision_context(collision_filter, margin_mm=5.0)

        ok, result, _solve, planner, endpoint = (
            self._run_path_collision_schedule(
                context, np.stack([nearest, second])
            )
        )

        self.assertFalse(ok)
        planner.assert_called_once()
        endpoint.assert_called_once()
        collision_filter.assert_called_once()
        self.assertEqual(
            result["status"], "TRAJECTORY_COLLISION_MARGIN_CHECK_ERROR"
        )
        self.assertIn("path collision checker unavailable", result["reason"])
        self.assertEqual(result["collision_margin_mm"], 5.0)
        self.assertEqual(result["segments"], [])
        self.assertEqual(len(result["ik_goal_attempts"]), 1)
        self.assertEqual(result["ik_goal_attempts"][0]["rank"], 1)
        self.assertEqual(
            result["ik_goal_attempts"][0]["status"],
            "TRAJECTORY_COLLISION_MARGIN_CHECK_ERROR",
        )

    def test_plan_schedule_retries_sorted_joint_goal_candidates(self) -> None:
        class Array:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float64)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class SuccessfulResult:
            def __init__(self, q_start, q_goal):
                position = np.stack([q_start, q_goal])
                zeros = np.zeros_like(position)
                self.success = types.SimpleNamespace(item=lambda: True)
                self.status = "SUCCESS"
                self.interpolation_dt = 0.02
                self.attempts = 1
                self.trajectory = types.SimpleNamespace(
                    position=Array(position),
                    velocity=Array(zeros),
                    acceleration=Array(zeros),
                )

            def get_interpolated_plan(self):
                return self.trajectory

        q_start = np.zeros(12, dtype=np.float64)
        nearest = np.full(12, 0.1, dtype=np.float64)
        middle = np.full(12, 0.2, dtype=np.float64)
        farthest = np.full(12, 0.3, dtype=np.float64)
        # The IK stub is intentionally unsorted; plan_schedule owns the
        # deterministic closest-to-current ordering contract.
        unsorted_goals = np.stack([farthest, nearest, middle])
        ik_report = {"solver_mode": "test_independent_ik"}
        ik_context = types.SimpleNamespace(max_goal_solutions=7)
        mg = types.SimpleNamespace(
            ik_solver=object(),
            _xtrainer_separate_ik=ik_context,
        )
        planned_goals = []

        def plan_segment(_mg, current, *_args, goal_q=None, **_kwargs):
            goal = np.asarray(goal_q, dtype=np.float64).copy()
            planned_goals.append(goal)
            if len(planned_goals) == 1:
                return types.SimpleNamespace(
                    success=types.SimpleNamespace(item=lambda: False),
                    status="TRAJOPT_FAIL",
                )
            return SuccessfulResult(np.asarray(current), goal)

        tcp_mismatch = {
            "status": "IK_GOAL_TCP_ENDPOINT_MISMATCH",
            "arms": {"arm2": {"valid": False}},
        }
        tcp_ok = {
            "status": "IK_GOAL_TCP_ENDPOINT_OK",
            "arms": {
                "arm1": {"valid": True},
                "arm2": {"valid": True},
            },
        }
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                dual,
                "pose_from_fk",
                side_effect=lambda _mg, _q, _link, name: _pose(name),
            ))
            solve_ik = stack.enter_context(mock.patch.object(
                dual,
                "solve_dual_ik_solutions",
                return_value=(
                    unsorted_goals.copy(),
                    types.SimpleNamespace(
                        status="SEPARATE_IK_OK", report=ik_report
                    ),
                    ik_report,
                ),
            ))
            planner = stack.enter_context(mock.patch.object(
                dual, "plan_dual_segment", side_effect=plan_segment
            ))
            endpoint = stack.enter_context(mock.patch.object(
                dual,
                "validate_dual_tcp_endpoint",
                side_effect=[(False, tcp_mismatch), (True, tcp_ok)],
            ))
            ok, result = dual.plan_schedule(
                mg=mg,
                q_start=q_start,
                seq1=[_pose("arm1_target")],
                seq2=[_pose("arm2_target")],
                delay=0,
                pl={},
                cr={
                    "joints": [1, 2, 3, 4, 5, 6],
                    "max_joint_delta_deg": 180.0,
                    "min_limit_margin_deg": 0.0,
                },
                linear_cfg={"enable": False},
                q_lo=np.full(12, -10.0),
                q_hi=np.full(12, 10.0),
                ee_links=("arm1_ee", "arm2_ee"),
                inactive_tolerance_deg=1.0,
            )

        self.assertTrue(ok, result)
        solve_ik.assert_called_once()
        solve_call = solve_ik.call_args
        self.assertIs(solve_call.args[0], ik_context)
        np.testing.assert_array_equal(solve_call.kwargs["seed_q"], q_start)
        np.testing.assert_array_equal(solve_call.kwargs["fixed_q"], q_start)
        self.assertEqual(solve_call.kwargs["active_arms"], (True, True))
        self.assertEqual(
            solve_call.kwargs["max_solutions"],
            ik_context.max_goal_solutions,
        )
        self.assertEqual(planner.call_count, 3)
        np.testing.assert_array_equal(
            np.stack(planned_goals), np.stack([nearest, middle, farthest])
        )
        self.assertEqual(endpoint.call_count, 2)
        np.testing.assert_array_equal(endpoint.call_args_list[0].args[1], middle)
        np.testing.assert_array_equal(endpoint.call_args_list[1].args[1], farthest)
        np.testing.assert_array_equal(result["q_end"], farthest)

        segment = result["segments"][0]
        self.assertEqual(segment["planner_goal_mode"], "separate_ik_joint_goal")
        self.assertEqual(segment["ik_goal_candidate_rank"], 3)
        self.assertEqual(segment["ik_goal_candidates_available"], 3)
        self.assertEqual(
            [attempt["status"] for attempt in segment["ik_goal_attempts"]],
            [
                "TRAJOPT_FAIL",
                "IK_GOAL_TCP_ENDPOINT_MISMATCH",
                "SUCCESS",
            ],
        )
        self.assertFalse(
            segment["ik_goal_attempts"][1]["arms"]["arm2"]["valid"]
        )
        self.assertEqual(segment["ik_goal_endpoint_error_deg"], 0.0)
        self.assertEqual(segment["tcp_goal_endpoint"], tcp_ok)
        self.assertEqual(segment["ik_goal_report"], ik_report)

    def test_disabled_prescreen_jump_gate_does_not_disable_trajectory_check(self) -> None:
        q_start = np.zeros(12, dtype=np.float64)
        q_jump = q_start.copy()
        q_jump[[0, 6]] = np.radians(20.0)
        q_lo = np.full(12, -10.0, dtype=np.float64)
        q_hi = np.full(12, 10.0, dtype=np.float64)
        seq1 = [_pose("arm1_target")]
        seq2 = [_pose("arm2_target")]
        criterion = {
            "joints": [1, 2, 3, 4, 5, 6],
            "max_joint_delta_deg": 5.0,
            "min_limit_margin_deg": 0.0,
            "prescreen_joint_delta_check": False,
        }
        mg = types.SimpleNamespace(ik_solver=object())

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                dual,
                "pose_from_fk",
                side_effect=lambda _mg, _q, _link, name: _pose(name),
            ))
            stack.enter_context(mock.patch.object(
                dual,
                "solve_dual_ik_solutions",
                return_value=(
                    q_jump.reshape(1, 12).copy(),
                    types.SimpleNamespace(
                        status="SEPARATE_IK_OK", report={"n_returned": 1}
                    ),
                    {"n_returned": 1},
                ),
            ))
            ok, report = dual.prescreen_dual_schedule(
                mg,
                q_start,
                seq1,
                seq2,
                delay=0,
                cr=criterion,
                q_lo=q_lo,
                q_hi=q_hi,
                ee_links=("arm1_ee", "arm2_ee"),
                inactive_tolerance_deg=1.0,
            )

        self.assertTrue(ok, report)
        self.assertEqual(report["status"], "IK_PRESCREEN_OK")
        self.assertIs(report["joint_delta_check"], False)
        self.assertAlmostEqual(report["max_joint_delta_deg"], 20.0)

        # The CLI/config switch applies only to the inexpensive IK prescreen.
        # Actual MotionGen trajectories keep validate_segment's fail-closed
        # default, even when the same criterion dictionary carries false.
        valid, trajectory_report = dual.validate_segment(
            None,
            q_start,
            np.stack([q_start, q_jump]),
            (seq1[0], seq2[0]),
            active=(True, True),
            linear=(False, False),
            cr=criterion,
            linear_cfg={},
            q_lo=q_lo,
            q_hi=q_hi,
            ee_links=("arm1_ee", "arm2_ee"),
            inactive_tolerance_deg=1.0,
        )
        self.assertFalse(valid)
        self.assertEqual(trajectory_report["status"], "JOINT_DELTA_EXCEED")
        self.assertAlmostEqual(trajectory_report["limit_deg"], 5.0)

    def test_prescreen_trace_keeps_accepted_and_rejected_ik_states(self) -> None:
        q_start = np.zeros(12, dtype=np.float64)
        q_first = np.radians(np.full(12, 1.0, dtype=np.float64))
        q_rejected = q_first.copy()
        q_rejected[[0, 6]] = np.radians(20.0)
        seq1 = [_pose("arm1_a"), _pose("arm1_b")]
        seq2 = [_pose("arm2_a"), _pose("arm2_b")]
        criterion = {
            "joints": [1, 2, 3, 4, 5, 6],
            "max_joint_delta_deg": 5.0,
            "min_limit_margin_deg": 0.0,
            "prescreen_joint_delta_check": True,
        }
        mg = types.SimpleNamespace(ik_solver=object())

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                dual,
                "pose_from_fk",
                side_effect=lambda _mg, _q, _link, name: _pose(name),
            ))
            stack.enter_context(mock.patch.object(
                dual,
                "solve_dual_ik_solutions",
                side_effect=[
                    (
                        q_first.reshape(1, 12).copy(),
                        types.SimpleNamespace(
                            status="SEPARATE_IK_OK", report={"n_returned": 1}
                        ),
                        {"n_returned": 1},
                    ),
                    (
                        q_rejected.reshape(1, 12).copy(),
                        types.SimpleNamespace(
                            status="SEPARATE_IK_OK", report={"n_returned": 1}
                        ),
                        {"n_returned": 1},
                    ),
                ],
            ))
            ok, report = dual.prescreen_dual_schedule(
                mg,
                q_start,
                seq1,
                seq2,
                delay=0,
                cr=criterion,
                q_lo=np.full(12, -10.0),
                q_hi=np.full(12, 10.0),
                ee_links=("arm1_ee", "arm2_ee"),
                inactive_tolerance_deg=1.0,
            )

        self.assertFalse(ok)
        self.assertEqual(report["status"], "IK_PRESCREEN_CRITERION")
        self.assertEqual(report["criterion_status"], "JOINT_DELTA_EXCEED")
        trace = report["ik_trace"]
        self.assertEqual(trace["n_solved_events"], 2)
        self.assertEqual(trace["n_accepted_events"], 1)
        self.assertEqual(trace["deepest_solved_event"], 1)
        self.assertFalse(trace["complete"])
        np.testing.assert_allclose(
            np.asarray(trace["joint_positions_rad"]),
            np.stack([q_start, q_first, q_rejected]),
        )
        self.assertTrue(trace["events"][0]["accepted"])
        self.assertFalse(trace["events"][1]["accepted"])
        self.assertEqual(
            trace["events"][1]["criterion_status"], "JOINT_DELTA_EXCEED"
        )

    def test_prescreen_trace_does_not_invent_state_for_unsolved_event(self) -> None:
        q_start = np.zeros(12, dtype=np.float64)
        q_first = np.radians(np.full(12, 1.0, dtype=np.float64))
        poses1 = [_pose("arm1_a"), _pose("arm1_b")]
        poses2 = [_pose("arm2_a"), _pose("arm2_b")]
        criterion = {
            "joints": [1, 2, 3, 4, 5, 6],
            "max_joint_delta_deg": 5.0,
            "min_limit_margin_deg": 0.0,
        }
        mg = types.SimpleNamespace(ik_solver=object())

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                dual,
                "pose_from_fk",
                side_effect=lambda _mg, _q, _link, name: _pose(name),
            ))
            stack.enter_context(mock.patch.object(
                dual,
                "solve_dual_ik_solutions",
                side_effect=[
                    (
                        q_first.reshape(1, 12).copy(),
                        types.SimpleNamespace(
                            status="SEPARATE_IK_OK", report={"n_returned": 1}
                        ),
                        {"n_returned": 1},
                    ),
                    (
                        np.empty((0, 12), dtype=np.float64),
                        types.SimpleNamespace(
                            status="NO_COLLISION_FREE_IK_PAIR",
                            report={"n_returned": 0},
                        ),
                        {"n_returned": 0},
                    ),
                ],
            ))
            ok, report = dual.prescreen_dual_schedule(
                mg,
                q_start,
                poses1,
                poses2,
                delay=0,
                cr=criterion,
                q_lo=np.full(12, -10.0),
                q_hi=np.full(12, 10.0),
                ee_links=("arm1_ee", "arm2_ee"),
                inactive_tolerance_deg=1.0,
            )

        self.assertFalse(ok)
        self.assertEqual(report["status"], "IK_PRESCREEN_FAIL")
        trace = report["ik_trace"]
        self.assertEqual(trace["n_solved_events"], 1)
        self.assertEqual(len(trace["joint_positions_rad"]), 2)
        self.assertTrue(trace["events"][0]["accepted"])
        self.assertFalse(trace["events"][1]["solve_success"])
        self.assertIsNone(trace["events"][1]["state_index"])

    def test_prescreen_tries_next_candidate_after_joint_delta_rejection(self) -> None:
        q_start = np.zeros(12, dtype=np.float64)
        nearest_but_jumps = q_start.copy()
        nearest_but_jumps[0] = np.radians(6.0)
        farther_but_valid = np.radians(
            np.full(12, 2.0, dtype=np.float64)
        )
        self.assertLess(
            np.linalg.norm(nearest_but_jumps - q_start),
            np.linalg.norm(farther_but_valid - q_start),
        )
        candidates = np.stack([nearest_but_jumps, farther_but_valid])
        ik_report = {"n_returned": 2, "solver_mode": "test"}
        ik_context = types.SimpleNamespace(max_goal_solutions=4)
        mg = types.SimpleNamespace(
            ik_solver=object(),
            _xtrainer_separate_ik=ik_context,
        )
        criterion = {
            "joints": [1, 2, 3, 4, 5, 6],
            "max_joint_delta_deg": 5.0,
            "min_limit_margin_deg": 0.0,
            "prescreen_joint_delta_check": True,
        }

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                dual,
                "pose_from_fk",
                side_effect=lambda _mg, _q, _link, name: _pose(name),
            ))
            solve_ik = stack.enter_context(mock.patch.object(
                dual,
                "solve_dual_ik_solutions",
                return_value=(
                    candidates.copy(),
                    types.SimpleNamespace(
                        status="SEPARATE_IK_OK", report=ik_report
                    ),
                    ik_report,
                ),
            ))
            validator = stack.enter_context(mock.patch.object(
                dual, "validate_segment", wraps=dual.validate_segment
            ))
            ok, report = dual.prescreen_dual_schedule(
                mg,
                q_start,
                [_pose("arm1_target")],
                [_pose("arm2_target")],
                delay=0,
                cr=criterion,
                q_lo=np.full(12, -10.0),
                q_hi=np.full(12, 10.0),
                ee_links=("arm1_ee", "arm2_ee"),
                inactive_tolerance_deg=1.0,
            )

        self.assertTrue(ok, report)
        solve_ik.assert_called_once()
        self.assertIs(solve_ik.call_args.args[0], ik_context)
        self.assertEqual(
            solve_ik.call_args.kwargs["max_solutions"],
            ik_context.max_goal_solutions,
        )
        self.assertEqual(validator.call_count, 2)
        np.testing.assert_array_equal(
            validator.call_args_list[0].args[2][-1], nearest_but_jumps
        )
        np.testing.assert_array_equal(
            validator.call_args_list[1].args[2][-1], farther_but_valid
        )
        self.assertEqual(report["status"], "IK_PRESCREEN_OK")
        self.assertAlmostEqual(report["max_joint_delta_deg"], 2.0)
        trace = report["ik_trace"]
        self.assertTrue(trace["complete"])
        self.assertEqual(trace["n_accepted_events"], 1)
        np.testing.assert_array_equal(
            np.asarray(trace["joint_positions_rad"])[-1], farther_but_valid
        )

    def test_default_disables_sequential_fallback_and_delays_stay_overlapped(self) -> None:
        dual_cfg = dual.load_dual_config(None)["dual_arm"]
        self.assertIs(dual_cfg["sequential_fallback"], False)

        # Empty/one-step sequences cannot overlap, and a non-positive maximum
        # explicitly disables delay search.  Every usable delay otherwise
        # keeps arm1 ahead without waiting for its full sequence.
        for sequence_len in range(0, 9):
            for base in (-10, 0, 2, sequence_len, sequence_len + 5):
                for maximum in (-5, 0, 1, sequence_len, sequence_len + 10):
                    with self.subTest(
                        sequence_len=sequence_len, base=base, maximum=maximum
                    ):
                        delays = dual.delay_candidates(base, maximum, sequence_len)
                        if sequence_len <= 1 or maximum <= 0:
                            self.assertEqual(delays, [])
                        else:
                            self.assertTrue(delays)
                            self.assertTrue(
                                all(1 <= delay < sequence_len for delay in delays)
                            )

    def test_delay_starts_arm1_first_but_still_overlaps(self) -> None:
        seq1 = [_pose(f"arm1_{i}", i) for i in range(6)]
        seq2 = [_pose(f"arm2_{i}", i) for i in range(6)]
        events = dual.scheduled_targets(
            seq1, seq2, _pose("hold1"), _pose("hold2"), delay=2
        )
        activity = [(advance1, advance2) for _, _, advance1, advance2 in events]

        self.assertEqual(activity[:2], [(True, False), (True, False)])
        self.assertEqual(activity[2:6], [(True, True)] * 4)
        self.assertEqual(activity[6:], [(False, True), (False, True)])
        first_arm2 = next(i for i, (_, active2) in enumerate(activity) if active2)
        last_arm1 = max(i for i, (active1, _) in enumerate(activity) if active1)
        self.assertEqual(first_arm2, 2)
        self.assertLess(first_arm2, last_arm1)

    def test_paired_angle_choices_obey_trial_cap(self) -> None:
        arm1 = [dual.AngleChoice(float(i), -float(i)) for i in range(4)]
        arm2 = [dual.AngleChoice(float(10 + i), -float(10 + i)) for i in range(4)]
        pairs = list(dual.paired_angle_choices(arm1, arm2, limit=5))

        self.assertEqual(len(pairs), 5)
        self.assertEqual(
            pairs,
            [
                (arm1[0], arm2[0]),
                (arm1[1], arm2[1]),
                (arm1[2], arm2[2]),
                (arm1[3], arm2[3]),
                (arm1[0], arm2[1]),
            ],
        )

    def test_default_first_twelve_pairs_cover_every_grasp_angle(self) -> None:
        cfg = dual.load_dual_config(None)
        choices = dual.arm_angle_choices(cfg["pick_place"]["angle_search"])
        first_pairs = list(dual.paired_angle_choices(choices, choices, limit=12))
        expected_grasps = {choice.grasp for choice in choices}

        self.assertEqual(len(expected_grasps), 12)
        self.assertEqual(len(first_pairs), 12)
        self.assertTrue(all(left.grasp == right.grasp for left, right in first_pairs))
        self.assertEqual({left.grasp for left, _ in first_pairs}, expected_grasps)
        self.assertEqual({right.grasp for _, right in first_pairs}, expected_grasps)

    def test_stage2_candidates_follow_all_unmodified_primary_candidates(self) -> None:
        angle_search = {
            "couple_place_to_grasp": False,
            "grasp": {"axis": "x", "min_deg": -20.0, "max_deg": 0.0, "step_deg": 10.0},
            "place": {"axis": "x", "min_deg": 0.0, "max_deg": 10.0, "step_deg": 10.0},
            "order": "abs",
            "strategy": "abs_sum",
            "max_trials": 0,
            "stage2": {
                "enable": True,
                "axis": "z",
                "min_deg": -30.0,
                "max_deg": 30.0,
                "step_deg": 30.0,
                "order": "abs",
                "max_trials": 0,
            },
        }
        primary = dual.angle_combos(angle_search)
        choices = dual.arm_angle_choices(angle_search)
        n_primary = len(primary)

        self.assertGreater(len(choices), n_primary)
        self.assertEqual(
            [(choice.grasp, choice.place) for choice in choices[:n_primary]],
            primary,
        )
        self.assertTrue(
            all(
                choice.stage2_grasp == 0.0 and choice.stage2_place == 0.0
                for choice in choices[:n_primary]
            )
        )
        first_nonzero_stage2 = next(
            i
            for i, choice in enumerate(choices)
            if choice.stage2_grasp != 0.0 or choice.stage2_place != 0.0
        )
        self.assertEqual(first_nonzero_stage2, n_primary)

        # A cap equal to the number of raw primary choices therefore cannot
        # spend any trial on stage2 before every primary has been represented.
        capped = choices[:n_primary]
        self.assertEqual(
            {(choice.grasp, choice.place) for choice in capped}, set(primary)
        )

    def test_validate_segment_rejects_only_excess_inactive_arm_drift(self) -> None:
        q_start = np.zeros(12, dtype=np.float64)
        q_lo = np.full(12, -10.0, dtype=np.float64)
        q_hi = np.full(12, 10.0, dtype=np.float64)
        target = _pose("unused_without_linear_validation")
        criterion = {
            "joints": [1, 2, 3, 4, 5, 6],
            "max_joint_delta_deg": 720.0,
            "min_limit_margin_deg": 0.0,
        }

        q_excess = np.zeros((3, 12), dtype=np.float64)
        q_excess[:, 0] = np.radians([0.0, 8.0, 16.0])
        q_excess[:, 6] = np.radians([0.0, 0.5, 1.25])
        valid, report = dual.validate_segment(
            None,
            q_start,
            q_excess,
            (target, target),
            active=(True, False),
            linear=(False, False),
            cr=criterion,
            linear_cfg={},
            q_lo=q_lo,
            q_hi=q_hi,
            ee_links=("unused_arm1", "unused_arm2"),
            inactive_tolerance_deg=1.0,
        )
        self.assertFalse(valid)
        self.assertEqual(report["status"], "INACTIVE_ARM_MOVED")
        self.assertEqual(report["arm"], 2)
        self.assertIsNone(report["inactive_joint_excursion_deg"][0])
        self.assertAlmostEqual(report["inactive_joint_excursion_deg"][1], 1.25)

        q_allowed = q_excess.copy()
        q_allowed[:, 6] = np.radians([0.0, 0.25, 0.75])
        valid, report = dual.validate_segment(
            None,
            q_start,
            q_allowed,
            (target, target),
            active=(True, False),
            linear=(False, False),
            cr=criterion,
            linear_cfg={},
            q_lo=q_lo,
            q_hi=q_hi,
            ee_links=("unused_arm1", "unused_arm2"),
            inactive_tolerance_deg=1.0,
        )
        self.assertTrue(valid)
        self.assertIsNone(report["inactive_joint_excursion_deg"][0])
        self.assertAlmostEqual(report["inactive_joint_excursion_deg"][1], 0.75)

    def test_validate_segment_fails_closed_on_bad_shape_and_nonfinite_data(self) -> None:
        target = _pose("unused_without_linear_validation")
        common = {
            "mg": None,
            "q_start": np.zeros(12, dtype=np.float64),
            "targets": (target, target),
            "active": (True, True),
            "linear": (False, False),
            "cr": {
                "joints": [1, 2, 3, 4, 5, 6],
                "max_joint_delta_deg": 720.0,
                "min_limit_margin_deg": 0.0,
            },
            "linear_cfg": {},
            "q_lo": np.full(12, -10.0),
            "q_hi": np.full(12, 10.0),
            "ee_links": ("unused_arm1", "unused_arm2"),
            "inactive_tolerance_deg": 1.0,
        }
        q_nan = np.zeros((2, 12), dtype=np.float64)
        q_nan[1, 7] = np.nan
        cases = (
            ("nan", q_nan, "NONFINITE_TRAJECTORY"),
            ("empty", np.empty((0, 12)), "INVALID_TRAJECTORY_SHAPE"),
            ("odd_width", np.zeros((2, 11)), "INVALID_TRAJECTORY_SHAPE"),
        )

        for label, trajectory, expected_status in cases:
            with self.subTest(case=label):
                valid, report = dual.validate_segment(q=trajectory, **common)
                self.assertFalse(valid)
                self.assertEqual(report["status"], expected_status)


class ContinuousPipelineScheduleTest(unittest.TestCase):
    @staticmethod
    def _items(arm: int):
        return [
            [_pose(f"arm{arm}_i{item}_s{stage}") for stage in range(6)]
            for item in range(2)
        ]

    def test_default_pipeline_phase_is_three_stages(self) -> None:
        dual_cfg = dual.load_dual_config(None)["dual_arm"]
        self.assertEqual(dual_cfg["start_delay_stages"], 3)

    def test_two_items_share_one_global_timeline_without_round_barrier(self) -> None:
        events = dual.scheduled_pipeline_targets(
            self._items(1),
            self._items(2),
            _pose("arm1_initial_hold"),
            _pose("arm2_initial_hold"),
            phase=3,
        )

        # A occupies ticks 0..11, B ticks 3..14, so the three B tail stages
        # must be flushed after A's final stage rather than being dropped.
        self.assertEqual(len(events), 15)
        expected_activity = (
            [(True, False)] * 3
            + [(True, True)] * 9
            + [(False, True)] * 3
        )
        self.assertEqual(
            [(advance1, advance2) for _, _, advance1, advance2 in events],
            expected_activity,
        )

        expected_arm1 = (
            [f"arm1_i0_s{stage}" for stage in range(6)]
            + [f"arm1_i1_s{stage}" for stage in range(6)]
            + ["arm1_i1_s5"] * 3
        )
        expected_arm2 = (
            ["arm2_initial_hold"] * 3
            + [f"arm2_i0_s{stage}" for stage in range(6)]
            + [f"arm2_i1_s{stage}" for stage in range(6)]
        )
        self.assertEqual(
            [target1.name for target1, _, _, _ in events], expected_arm1
        )
        self.assertEqual(
            [target2.name for _, target2, _, _ in events], expected_arm2
        )

        # No round barrier: A starts item 1 while B still has three stages of
        # item 0 left.  B likewise starts item 1 on the tick immediately after
        # finishing item 0.
        self.assertEqual(
            (events[6][0].name, events[6][1].name),
            ("arm1_i1_s0", "arm2_i0_s3"),
        )
        self.assertEqual(events[8][1].name, "arm2_i0_s5")
        self.assertEqual(events[9][1].name, "arm2_i1_s0")
        self.assertEqual(
            [(event[0].name, event[2], event[3]) for event in events[12:]],
            [("arm1_i1_s5", False, True)] * 3,
        )

    def test_three_vs_two_items_returns_arm2_home_then_parks(self) -> None:
        arm1_items = [
            [_pose(f"arm1_i{item}_s{stage}") for stage in range(6)]
            for item in range(3)
        ]
        arm2_items = [
            [_pose(f"arm2_i{item}_s{stage}") for stage in range(6)]
            for item in range(2)
        ]
        arm2_home = _pose("arm2_terminal_home")
        # Home is a terminal event, not a synthetic third work item.
        arm2_timeline_items = arm2_items + [[arm2_home]]

        events = dual.scheduled_pipeline_targets(
            arm1_items,
            arm2_timeline_items,
            _pose("arm1_initial_hold"),
            _pose("arm2_initial_hold"),
            phase=3,
        )

        self.assertEqual(len(events), 18)
        chunks = (events[:3], events[3:9], events[9:15], events[15:18])
        self.assertEqual(
            [
                (
                    sum(int(event[2]) for event in chunk),
                    sum(int(event[3]) for event in chunk),
                )
                for chunk in chunks
            ],
            [(3, 0), (6, 6), (6, 6), (3, 1)],
        )
        self.assertEqual(
            (events[15][0].name, events[15][1].name, events[15][2:]),
            ("arm1_i2_s3", "arm2_terminal_home", (True, True)),
        )
        self.assertEqual(
            [
                (target1.name, target2.name, active1, active2)
                for target1, target2, active1, active2 in events[16:]
            ],
            [
                ("arm1_i2_s4", "arm2_terminal_home", True, False),
                ("arm1_i2_s5", "arm2_terminal_home", True, False),
            ],
        )


class ParkedArmScheduleTest(unittest.TestCase):
    class _Array:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float64)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class _Result:
        def __init__(self, position):
            position = np.asarray(position, dtype=np.float64)
            zeros = np.zeros_like(position)
            self.success = types.SimpleNamespace(item=lambda: True)
            self.status = "SUCCESS"
            self.interpolation_dt = 0.02
            self.attempts = 1
            self.trajectory = types.SimpleNamespace(
                position=ParkedArmScheduleTest._Array(position),
                velocity=ParkedArmScheduleTest._Array(zeros),
                acceleration=ParkedArmScheduleTest._Array(zeros),
            )

        def get_interpolated_plan(self):
            return self.trajectory

    def test_cumulative_parked_drift_uses_fixed_reference(self) -> None:
        plan_starts = []

        def plan_segment(_mg, q_start, *_args, **_kwargs):
            q_start = np.asarray(q_start, dtype=np.float64).copy()
            plan_starts.append(q_start)
            q_end = q_start.copy()
            # Each segment moves the inactive arm by only 0.6 degrees, below
            # the per-segment tolerance.  The second endpoint is nevertheless
            # 1.2 degrees away from the fixed Home reference.
            q_end[6:] += np.radians(0.6)
            return self._Result(np.stack([q_start, q_end]))

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                dual,
                "pose_from_fk",
                side_effect=lambda _mg, _q, _link, name: _pose(name),
            ))
            stack.enter_context(mock.patch.object(
                dual, "make_hold_pose_metric", return_value=object()
            ))
            planner = stack.enter_context(mock.patch.object(
                dual, "plan_dual_segment", side_effect=plan_segment
            ))
            ok, result = dual.plan_schedule(
                mg=object(),
                q_start=np.zeros(12, dtype=np.float64),
                seq1=[_pose("arm1_stage0"), _pose("arm1_stage1")],
                seq2=[],
                delay=0,
                pl={},
                cr={
                    "joints": [1, 2, 3, 4, 5, 6],
                    "max_joint_delta_deg": 10.0,
                    "min_limit_margin_deg": 0.0,
                },
                linear_cfg={"enable": False},
                q_lo=np.full(12, -10.0),
                q_hi=np.full(12, 10.0),
                ee_links=("arm1_ee", "arm2_ee"),
                inactive_tolerance_deg=1.0,
                parked_joint_refs=(None, np.zeros(12, dtype=np.float64)),
            )

        self.assertFalse(ok)
        self.assertEqual(planner.call_count, 2)
        self.assertEqual(result["status"], "PARKED_ARM_MOVED")
        self.assertEqual(result["arm"], 2)
        self.assertEqual(result["failed_event"], 1)
        self.assertAlmostEqual(
            result["parked_joint_excursion_deg"][1], 1.2
        )
        self.assertEqual(len(result["segments"]), 1)
        self.assertAlmostEqual(
            result["segments"][0]["parked_joint_excursion_deg"][1], 0.6
        )
        np.testing.assert_allclose(
            plan_starts[1][6:], np.full(6, np.radians(0.6))
        )


class ContinuousPipelineSearchIntegrationTest(unittest.TestCase):
    @staticmethod
    def _make_sequence(
        _item, _place, _choice, _pp, item_index, _transform, prefix
    ):
        arm = prefix.rstrip("_")
        return [_pose(f"{arm}_i{item_index}_s{stage}") for stage in range(6)]

    @staticmethod
    def _segment_reports(seq1, seq2, marker):
        current1 = seq1[0] if seq1 else _pose("mock_arm1_hold")
        current2 = seq2[0] if seq2 else _pose("mock_arm2_hold")
        reports = []
        for index in range(max(len(seq1), len(seq2))):
            active1 = index < len(seq1)
            active2 = index < len(seq2)
            if active1:
                current1 = seq1[index]
            if active2:
                current2 = seq2[index]
            reports.append({
                "targets": [current1.name, current2.name],
                "active_arms": [
                    arm
                    for arm, active in ((1, active1), (2, active2))
                    if active
                ],
                "mock_chunk": marker,
            })
        return reports

    @classmethod
    def _success(cls, q_begin, seq1, seq2, q_end_value, marker):
        q_begin = np.asarray(q_begin, dtype=np.float64)
        q_end = np.full(12, q_end_value, dtype=np.float64)
        position = np.stack([q_begin, q_end])
        return True, {
            "position": position,
            "velocity": np.zeros_like(position),
            "acceleration": np.zeros_like(position),
            "dt": 0.02,
            "q_end": q_end,
            "segments": cls._segment_reports(seq1, seq2, marker),
        }

    @staticmethod
    def _failure(status):
        return False, {
            "status": status,
            "targets": ["mock_arm1_target", "mock_arm2_target"],
            "segments": [],
        }

    def _run_pipeline(
        self,
        choice_side_effect,
        plan_side_effect,
        max_pairs=10,
        max_nodes=0,
        stage2_enabled=False,
        stage2_increments=None,
        first_home_grasp_angles=(None, None),
    ):
        with ExitStack() as stack:
            choices = stack.enter_context(
                mock.patch.object(
                    dual, "arm_primary_choices", side_effect=choice_side_effect
                )
            )
            stack.enter_context(
                mock.patch.object(
                    dual, "make_arm_sequence", side_effect=self._make_sequence
                )
            )
            if stage2_increments is not None:
                stack.enter_context(
                    mock.patch.object(
                        dual, "stage2_combos", return_value=stage2_increments
                    )
                )
            planner = stack.enter_context(
                mock.patch.object(
                    dual, "plan_schedule", side_effect=plan_side_effect
                )
            )
            stack.enter_context(mock.patch("builtins.print"))
            pipeline, failures, diagnostics = dual.try_continuous_pipeline(
                mg=None,
                q_start=np.zeros(12, dtype=np.float64),
                arm1_items_local=[[-0.6, -0.05, 0.03], [-0.6, -0.02, 0.03]],
                arm2_items_local=[[-0.6, -0.05, 0.03], [-0.6, -0.02, 0.03]],
                place1_local=[-0.26, -0.26, 0.1],
                place2_local=[-0.33, 0.26, 0.1],
                pp={},
                asr={"stage2": {"enable": stage2_enabled}},
                dc={
                    "start_delay_stages": 3,
                    "max_pair_angle_trials": max_pairs,
                    "max_pipeline_search_nodes": max_nodes,
                    "inactive_joint_tolerance_deg": 1.0,
                },
                transforms=(np.eye(4), np.eye(4)),
                pl={},
                cr={"prescreen_by_ik": False},
                linear_cfg={},
                q_lo=np.full(12, -10.0),
                q_hi=np.full(12, 10.0),
                ee_links=("arm1_ee", "arm2_ee"),
                first_home_grasp_angles=first_home_grasp_angles,
            )
        return pipeline, failures, diagnostics, choices, planner

    def test_production_search_plans_prime_and_two_rolling_blocks(self) -> None:
        choice = dual.AngleChoice(10.0, 20.0)
        calls = []

        def plan_success(_mg, q_begin, seq1, seq2, *_args):
            calls.append({
                "q_begin": np.asarray(q_begin).copy(),
                "seq1": [pose.name for pose in seq1],
                "seq2": [pose.name for pose in seq2],
            })
            index = len(calls)
            return self._success(q_begin, seq1, seq2, float(index), f"chunk_{index}")

        pipeline, failures, diagnostics, _, planner = self._run_pipeline(
            choice_side_effect=lambda *_args, **_kwargs: [choice],
            plan_side_effect=plan_success,
        )

        self.assertIsNotNone(pipeline)
        self.assertEqual(planner.call_count, 3)
        self.assertEqual(
            [max(len(call["seq1"]), len(call["seq2"])) for call in calls],
            [3, 6, 6],
        )
        self.assertEqual([len(call["seq1"]) for call in calls], [3, 6, 3])
        self.assertEqual([len(call["seq2"]) for call in calls], [0, 6, 6])
        self.assertEqual(
            calls[1]["seq1"],
            [
                "arm1_i0_s3", "arm1_i0_s4", "arm1_i0_s5",
                "arm1_i1_s0", "arm1_i1_s1", "arm1_i1_s2",
            ],
        )
        self.assertEqual(
            calls[1]["seq2"], [f"arm2_i0_s{stage}" for stage in range(6)]
        )
        self.assertEqual(len(pipeline.chunks), 3)
        self.assertEqual(len(pipeline.segments), 15)
        self.assertEqual(
            [segment["global_event_index"] for segment in pipeline.segments],
            list(range(15)),
        )
        self.assertEqual(
            (
                pipeline.segments[6]["arm1_item_index"],
                pipeline.segments[6]["arm1_stage_index"],
                pipeline.segments[6]["arm2_item_index"],
                pipeline.segments[6]["arm2_stage_index"],
            ),
            (1, 0, 0, 3),
        )
        self.assertEqual(
            diagnostics["block_attempts"],
            {"prime": 1, "block_0": 1, "block_1": 1},
        )
        self.assertEqual(failures, [])

    def test_downstream_failure_backtracks_without_committing_prefix(self) -> None:
        a0 = dual.AngleChoice(1.0, 1.0)
        b0_first, b0_second = dual.AngleChoice(4.0, 4.0), dual.AngleChoice(5.0, 5.0)
        a1_first, a1_second = dual.AngleChoice(2.0, 2.0), dual.AngleChoice(3.0, 3.0)
        b1_first, b1_second = dual.AngleChoice(6.0, 6.0), dual.AngleChoice(7.0, 7.0)
        candidate_calls = [
            [a0],
            [a0],
            [b0_first, b0_second],
            [a1_first, a1_second],
            [b1_first],
            [b1_second],
        ]
        calls = []

        def plan_with_backtrack(_mg, q_begin, seq1, seq2, *_args):
            calls.append({
                "q_begin": np.asarray(q_begin).copy(),
                "seq1": [pose.name for pose in seq1],
                "seq2": [pose.name for pose in seq2],
            })
            index = len(calls)
            if index == 3:
                return self._failure("MOCK_DOWNSTREAM_FAIL")
            end_value = {1: 1.0, 2: 2.0, 4: 3.0, 5: 4.0}[index]
            marker = {
                1: "prime",
                2: "discarded_block0_prefix",
                4: "committed_block0",
                5: "committed_block1",
            }[index]
            return self._success(q_begin, seq1, seq2, end_value, marker)

        pipeline, failures, diagnostics, choices, planner = self._run_pipeline(
            choice_side_effect=candidate_calls,
            plan_side_effect=plan_with_backtrack,
        )

        self.assertIsNotNone(pipeline)
        self.assertEqual(choices.call_count, 6)
        self.assertEqual(planner.call_count, 5)
        # The alternative block_0 must restart from prime.q_end=1, not from
        # the abandoned speculative prefix whose q_end was 2.
        np.testing.assert_array_equal(calls[1]["q_begin"], np.full(12, 1.0))
        np.testing.assert_array_equal(calls[3]["q_begin"], np.full(12, 1.0))
        self.assertEqual(
            [float(chunk.q_end[0]) for chunk in pipeline.chunks], [1.0, 3.0, 4.0]
        )
        self.assertEqual(pipeline.arm1_choices, [a0, a1_second])
        self.assertEqual(pipeline.arm2_choices, [b0_second, b1_second])
        committed_markers = {segment["mock_chunk"] for segment in pipeline.segments}
        self.assertNotIn("discarded_block0_prefix", committed_markers)
        self.assertEqual(
            pipeline.position[:, 0].tolist(), [0.0, 1.0, 3.0, 4.0]
        )
        self.assertEqual(
            diagnostics["block_attempts"],
            {"prime": 1, "block_0": 2, "block_1": 2},
        )
        self.assertIn("MOCK_DOWNSTREAM_FAIL", [x.get("status") for x in failures])
        self.assertIn("DOWNSTREAM_BACKTRACK", [x.get("status") for x in failures])

    def test_pair_cap_resets_when_deep_block_is_reentered_from_new_q(self) -> None:
        a0 = dual.AngleChoice(1.0, 1.0)
        b0_first, b0_second = dual.AngleChoice(4.0, 4.0), dual.AngleChoice(5.0, 5.0)
        a1_first, a1_second = dual.AngleChoice(2.0, 2.0), dual.AngleChoice(3.0, 3.0)
        b1_first = dual.AngleChoice(6.0, 6.0)
        b1_second = dual.AngleChoice(7.0, 7.0)
        b1_after_backtrack = dual.AngleChoice(8.0, 8.0)
        candidate_calls = [
            [a0],
            [a0],
            [b0_first, b0_second],
            [a1_first, a1_second],
            [b1_first, b1_second],
            [b1_after_backtrack],
        ]
        calls = []

        def plan_with_exhausted_deep_invocation(
            _mg, q_begin, seq1, seq2, *_args
        ):
            calls.append({
                "q_begin": np.asarray(q_begin).copy(),
                "seq1": [pose.name for pose in seq1],
                "seq2": [pose.name for pose in seq2],
            })
            index = len(calls)
            if index in (3, 4):
                return self._failure(f"FIRST_DEEP_INVOCATION_FAIL_{index}")
            end_value = {1: 1.0, 2: 2.0, 5: 3.0, 6: 4.0}[index]
            return self._success(
                q_begin, seq1, seq2, end_value, f"committed_call_{index}"
            )

        pipeline, failures, diagnostics, _, planner = self._run_pipeline(
            choice_side_effect=candidate_calls,
            plan_side_effect=plan_with_exhausted_deep_invocation,
            max_pairs=2,
        )

        self.assertIsNotNone(pipeline)
        self.assertEqual(planner.call_count, 6)
        # The first block_1 invocation consumes both local attempts at q=2.
        np.testing.assert_array_equal(calls[2]["q_begin"], np.full(12, 2.0))
        np.testing.assert_array_equal(calls[3]["q_begin"], np.full(12, 2.0))
        # After block_0 backtracks, block_1 is a fresh invocation at q=3 and
        # must receive a new local cap instead of being blocked by telemetry=2.
        np.testing.assert_array_equal(calls[5]["q_begin"], np.full(12, 3.0))
        self.assertEqual(
            diagnostics["block_attempts"],
            {"prime": 1, "block_0": 2, "block_1": 3},
        )
        self.assertGreater(diagnostics["block_attempts"]["block_1"], 2)
        self.assertEqual(pipeline.arm1_choices, [a0, a1_second])
        self.assertEqual(
            pipeline.arm2_choices, [b0_second, b1_after_backtrack]
        )
        self.assertEqual(
            [float(chunk.q_end[0]) for chunk in pipeline.chunks], [1.0, 3.0, 4.0]
        )
        statuses = [failure.get("status") for failure in failures]
        self.assertIn("FIRST_DEEP_INVOCATION_FAIL_3", statuses)
        self.assertIn("FIRST_DEEP_INVOCATION_FAIL_4", statuses)
        self.assertIn("DOWNSTREAM_BACKTRACK", statuses)

    def test_global_stage2_pass_keeps_primary_and_prefers_nonzero_refinement(self) -> None:
        primary = dual.AngleChoice(10.0, 20.0)
        calls = []

        def plan_success(_mg, q_begin, seq1, seq2, *_args):
            calls.append((np.asarray(q_begin).copy(), list(seq1), list(seq2)))
            index = len(calls)
            return self._success(
                q_begin, seq1, seq2, float(index), f"stage_pass_call_{index}"
            )

        pipeline, failures, diagnostics, _, planner = self._run_pipeline(
            choice_side_effect=lambda *_args, **_kwargs: [primary],
            plan_side_effect=plan_success,
            stage2_enabled=True,
            stage2_increments=[(0.0, 0.0), (15.0, -15.0)],
        )

        self.assertIsNotNone(pipeline)
        self.assertEqual(planner.call_count, 6)
        self.assertEqual(pipeline.angle_stage, "stage2")
        self.assertEqual(diagnostics["angle_stage"], "stage2")
        all_choices = pipeline.arm1_choices + pipeline.arm2_choices
        self.assertTrue(all(choice.primary == primary.primary for choice in all_choices))
        self.assertTrue(all(
            (choice.stage2_grasp, choice.stage2_place) == (15.0, -15.0)
            for choice in all_choices
        ))
        self.assertEqual(failures, [])

    def test_first_home_grasps_lock_primary_and_first_item_stage2_grasp(self) -> None:
        candidates = [
            dual.AngleChoice(0.0, 5.0),
            dual.AngleChoice(30.0, 10.0),
            dual.AngleChoice(60.0, 15.0),
        ]
        calls = []

        def plan_success(_mg, q_begin, seq1, seq2, *_args):
            calls.append((np.asarray(q_begin).copy(), list(seq1), list(seq2)))
            index = len(calls)
            return self._success(
                q_begin, seq1, seq2, float(index), f"locked_call_{index}"
            )

        pipeline, failures, diagnostics, _, planner = self._run_pipeline(
            choice_side_effect=lambda *_args, **_kwargs: list(candidates),
            plan_side_effect=plan_success,
            stage2_enabled=True,
            # A nonzero grasp refinement is intentionally offered first.  It
            # must be removed for each first item because q_start was solved
            # for the recorded pregrasp orientation.
            stage2_increments=[(15.0, -5.0), (0.0, 20.0), (0.0, 0.0)],
            first_home_grasp_angles=(30.0, 60.0),
        )

        self.assertIsNotNone(pipeline)
        self.assertEqual(failures, [])
        self.assertEqual(planner.call_count, 6)
        self.assertEqual(pipeline.angle_stage, "stage2")
        first1 = pipeline.arm1_choices[0]
        first2 = pipeline.arm2_choices[0]
        self.assertEqual(first1.grasp, 30.0)
        self.assertEqual(first2.grasp, 60.0)
        self.assertEqual(first1.stage2_grasp, 0.0)
        self.assertEqual(first2.stage2_grasp, 0.0)
        self.assertEqual(first1.stage2_place, 20.0)
        self.assertEqual(first2.stage2_place, 20.0)
        self.assertEqual(
            diagnostics["first_home_grasp_angles"], [30.0, 60.0]
        )
        self.assertEqual(
            diagnostics["stage2_status"], "selected_nonzero_refinement"
        )

    def test_home_joint_reference_rejects_prime_anchor_and_terminal_hold_drift(self) -> None:
        choice = dual.AngleChoice(30.0, 20.0)
        root = np.concatenate([np.full(6, 0.1), np.full(6, 0.2)])
        terminal = _pose("arm2_terminal_home")

        for scenario, expected_status, expected_context, expected_calls in (
            (
                "prime_anchor",
                "HOME_BRANCH_MISMATCH",
                "arm1_first_pregrasp_anchor",
                1,
            ),
            (
                "terminal_hold",
                "HOME_BRANCH_HOLD_MISMATCH",
                "arm2_terminal_home",
                3,
            ),
        ):
            with self.subTest(scenario=scenario):
                returned_results = []

                def plan_with_branch_drift(
                    _mg,
                    q_begin,
                    seq1,
                    seq2,
                    _delay,
                    _pl,
                    _cr,
                    _linear_cfg,
                    _q_lo,
                    _q_hi,
                    _ee_links,
                    _inactive_tolerance_deg,
                    parked_joint_refs=(None, None),
                    park_after_targets=(None, None),
                ):
                    n_events = max(len(seq1), len(seq2))
                    position = np.repeat(
                        np.asarray(q_begin, dtype=np.float64)[None, :],
                        n_events + 1,
                        axis=0,
                    )
                    is_prime = len(returned_results) == 0
                    is_terminal = park_after_targets[1] == terminal.name
                    if scenario == "prime_anchor" and is_prime:
                        position[1:, :6] += np.radians(2.0)
                    if scenario == "terminal_hold" and is_terminal:
                        # Event 0 reaches the selected B branch exactly, then
                        # the nominally parked suffix drifts beyond tolerance.
                        position[2:, 6:] += np.radians(2.0)
                    reports = self._segment_reports(
                        seq1, seq2, f"{scenario}_{len(returned_results)}"
                    )
                    for event_index, report in enumerate(reports):
                        report["trajectory_point_start"] = event_index
                        report["trajectory_point_end"] = event_index + 1
                    result = {
                        "position": position,
                        "velocity": np.zeros_like(position),
                        "acceleration": np.zeros_like(position),
                        "dt": 0.02,
                        "q_end": position[-1].copy(),
                        "segments": reports,
                        "parked_joint_refs": parked_joint_refs,
                    }
                    returned_results.append(result)
                    return True, result

                arm1_items = (
                    [[-0.2, 0.0, 0.1]]
                    if scenario == "prime_anchor"
                    else [[-0.2, 0.0, 0.1], [-0.3, 0.0, 0.1]]
                )
                arm2_items = (
                    [] if scenario == "prime_anchor"
                    else [[0.2, 0.0, 0.1]]
                )
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        dual, "arm_primary_choices", return_value=[choice]
                    ))
                    stack.enter_context(mock.patch.object(
                        dual, "make_arm_sequence", side_effect=self._make_sequence
                    ))
                    planner = stack.enter_context(mock.patch.object(
                        dual, "plan_schedule", side_effect=plan_with_branch_drift
                    ))
                    stack.enter_context(mock.patch("builtins.print"))
                    pipeline, failures, diagnostics = dual.try_continuous_pipeline(
                        mg=None,
                        q_start=root.copy(),
                        arm1_items_local=arm1_items,
                        arm2_items_local=arm2_items,
                        place1_local=[-0.26, -0.26, 0.1],
                        place2_local=[-0.33, 0.26, 0.1],
                        pp={},
                        asr={"stage2": {"enable": False}},
                        dc={
                            "start_delay_stages": 3,
                            "max_pair_angle_trials": 5,
                            "max_pipeline_search_nodes": 0,
                            "inactive_joint_tolerance_deg": 1.0,
                            "home_branch_tolerance_deg": 1.0,
                        },
                        transforms=(np.eye(4), np.eye(4)),
                        pl={},
                        cr={"prescreen_by_ik": False},
                        linear_cfg={},
                        q_lo=np.full(12, -10.0),
                        q_hi=np.full(12, 10.0),
                        ee_links=("arm1_ee", "arm2_ee"),
                        arm2_terminal=(
                            None if scenario == "prime_anchor" else terminal
                        ),
                        first_home_grasp_angles=(
                            choice.grasp,
                            None if scenario == "prime_anchor" else choice.grasp,
                        ),
                        first_home_joint_reference=root.copy(),
                    )

                self.assertIsNone(pipeline)
                self.assertEqual(planner.call_count, expected_calls)
                matched = [
                    failure for failure in failures
                    if failure.get("status") == expected_status
                ]
                self.assertEqual(len(matched), 1)
                failure = matched[0]
                self.assertEqual(failure["stage"], "home_branch_check")
                self.assertEqual(failure["home_branch_context"], expected_context)
                self.assertEqual(
                    failure["arm"], 1 if scenario == "prime_anchor" else 2
                )
                self.assertAlmostEqual(
                    failure["home_branch_tolerance_deg"], 1.0
                )
                if scenario == "prime_anchor":
                    self.assertAlmostEqual(failure["home_branch_error_deg"], 2.0)
                else:
                    self.assertAlmostEqual(failure["home_branch_error_deg"], 0.0)
                    self.assertAlmostEqual(
                        failure["home_branch_post_terminal_hold_error_deg"], 2.0
                    )
                self.assertTrue(all(
                    "trajectory_point_end" in segment
                    for result in returned_results
                    for segment in result["segments"]
                ))
                self.assertEqual(
                    diagnostics["failure_status_counts"][expected_status], 1
                )

    def test_failed_global_stage2_pass_returns_primary_baseline(self) -> None:
        primary = dual.AngleChoice(10.0, 20.0)
        calls = []

        def fail_stage2_prime(_mg, q_begin, seq1, seq2, *_args):
            calls.append((np.asarray(q_begin).copy(), list(seq1), list(seq2)))
            if len(calls) == 4:
                return self._failure("GLOBAL_STAGE2_FAIL")
            index = len(calls)
            return self._success(
                q_begin, seq1, seq2, float(index), f"primary_call_{index}"
            )

        pipeline, failures, diagnostics, _, planner = self._run_pipeline(
            choice_side_effect=lambda *_args, **_kwargs: [primary],
            plan_side_effect=fail_stage2_prime,
            stage2_enabled=True,
            stage2_increments=[(15.0, -15.0)],
        )

        self.assertIsNotNone(pipeline)
        self.assertEqual(planner.call_count, 4)
        self.assertEqual(pipeline.angle_stage, "primary")
        self.assertEqual(diagnostics["angle_stage"], "primary")
        self.assertEqual(
            pipeline.position[:, 0].tolist(), [0.0, 1.0, 2.0, 3.0]
        )
        self.assertTrue(all(
            choice.stage2_grasp == 0.0 and choice.stage2_place == 0.0
            for choice in pipeline.arm1_choices + pipeline.arm2_choices
        ))
        self.assertIn("GLOBAL_STAGE2_FAIL", [x.get("status") for x in failures])

    def test_global_node_cap_exhaustion_returns_primary_baseline(self) -> None:
        primary = dual.AngleChoice(10.0, 20.0)
        calls = []

        def plan_primary(_mg, q_begin, seq1, seq2, *_args):
            calls.append((np.asarray(q_begin).copy(), list(seq1), list(seq2)))
            index = len(calls)
            return self._success(
                q_begin, seq1, seq2, float(index), f"primary_call_{index}"
            )

        pipeline, failures, diagnostics, _, planner = self._run_pipeline(
            choice_side_effect=lambda *_args, **_kwargs: [primary],
            plan_side_effect=plan_primary,
            max_nodes=3,
            stage2_enabled=True,
            stage2_increments=[(15.0, -15.0)],
        )

        self.assertIsNotNone(pipeline)
        self.assertEqual(planner.call_count, 3)
        self.assertEqual(pipeline.angle_stage, "primary")
        self.assertEqual(diagnostics["angle_stage"], "primary")
        self.assertEqual(diagnostics["search_nodes"], 3)
        self.assertTrue(diagnostics["node_budget_exhausted"])
        self.assertEqual(
            pipeline.position[:, 0].tolist(), [0.0, 1.0, 2.0, 3.0]
        )
        self.assertEqual(failures, [])

    def test_three_vs_two_items_plans_terminal_home_and_propagates_park(self) -> None:
        choice = dual.AngleChoice(10.0, 20.0)
        arm2_home = _pose("arm2_terminal_home")
        calls = []

        def plan_success(
            _mg,
            q_begin,
            seq1,
            seq2,
            _delay,
            _pl,
            _cr,
            _linear_cfg,
            _q_lo,
            _q_hi,
            _ee_links,
            _inactive_tolerance_deg,
            parked_joint_refs=(None, None),
            park_after_targets=(None, None),
        ):
            calls.append({
                "q_begin": np.asarray(q_begin).copy(),
                "seq1": [pose.name for pose in seq1],
                "seq2": [pose.name for pose in seq2],
                "parked_in": tuple(
                    None if ref is None else np.asarray(ref).copy()
                    for ref in parked_joint_refs
                ),
                "park_after": tuple(park_after_targets),
            })
            index = len(calls)
            ok, result = self._success(
                q_begin, seq1, seq2, float(index), f"chunk_{index}"
            )
            if park_after_targets[1] == arm2_home.name:
                result["parked_joint_refs"] = (
                    parked_joint_refs[0], result["q_end"].copy()
                )
            else:
                result["parked_joint_refs"] = parked_joint_refs
            return ok, result

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                dual, "arm_primary_choices", return_value=[choice]
            ))
            stack.enter_context(mock.patch.object(
                dual, "make_arm_sequence", side_effect=self._make_sequence
            ))
            planner = stack.enter_context(mock.patch.object(
                dual, "plan_schedule", side_effect=plan_success
            ))
            stack.enter_context(mock.patch("builtins.print"))
            pipeline, failures, diagnostics = dual.try_continuous_pipeline(
                mg=None,
                q_start=np.zeros(12, dtype=np.float64),
                arm1_items_local=[
                    [-0.2, -0.05, 0.03],
                    [-0.3, -0.05, 0.03],
                    [-0.4, -0.05, 0.03],
                ],
                arm2_items_local=[
                    [-0.2, -0.05, 0.03],
                    [-0.3, -0.05, 0.03],
                ],
                place1_local=[-0.26, -0.26, 0.1],
                place2_local=[-0.33, 0.26, 0.1],
                pp={},
                asr={"stage2": {"enable": False}},
                dc={
                    "start_delay_stages": 3,
                    "max_pair_angle_trials": 10,
                    "max_pipeline_search_nodes": 0,
                    "inactive_joint_tolerance_deg": 1.0,
                },
                transforms=(np.eye(4), np.eye(4)),
                pl={},
                cr={"prescreen_by_ik": False},
                linear_cfg={},
                q_lo=np.full(12, -10.0),
                q_hi=np.full(12, 10.0),
                ee_links=("arm1_ee", "arm2_ee"),
                arm2_terminal=arm2_home,
            )

        self.assertIsNotNone(pipeline)
        self.assertEqual(failures, [])
        self.assertEqual(planner.call_count, 4)
        self.assertEqual(
            [(len(call["seq1"]), len(call["seq2"])) for call in calls],
            [(3, 0), (6, 6), (6, 6), (3, 1)],
        )
        self.assertEqual(calls[-1]["seq2"], [arm2_home.name])
        self.assertEqual(calls[-1]["park_after"], (None, arm2_home.name))
        self.assertTrue(all(
            call["park_after"] == (None, None) for call in calls[:-1]
        ))
        self.assertEqual(len(pipeline.chunks), 4)
        self.assertEqual(len(pipeline.segments), 18)
        self.assertEqual(len(pipeline.arm1_choices), 3)
        self.assertEqual(len(pipeline.arm2_choices), 2)
        self.assertIs(pipeline.arm2_terminal, arm2_home)
        self.assertEqual(
            [segment["global_event_index"] for segment in pipeline.segments],
            list(range(18)),
        )
        self.assertEqual(
            [segment["arm2_terminal_home"] for segment in pipeline.segments],
            [False] * 15 + [True, False, False],
        )
        self.assertEqual(
            pipeline.segments[15]["targets"][1], arm2_home.name
        )
        self.assertEqual(
            pipeline.segments[15]["active_arms"], [1, 2]
        )
        self.assertEqual(
            pipeline.segments[16]["active_arms"], [1]
        )
        np.testing.assert_array_equal(
            pipeline.chunks[-1].parked_joint_refs[1], np.full(12, 4.0)
        )
        np.testing.assert_array_equal(
            pipeline.parked_joint_refs[1], np.full(12, 4.0)
        )
        self.assertEqual(diagnostics["n_arm1_items"], 3)
        self.assertEqual(diagnostics["n_arm2_items"], 2)
        self.assertTrue(diagnostics["arm2_terminal_home"])
        self.assertEqual(diagnostics["expected_global_events"], 18)


class PipelineAssemblyValidationTest(unittest.TestCase):
    @staticmethod
    def _round(q_start, q_end, seq1, seq2, marker, choice):
        position = np.stack([
            np.full(12, q_start, dtype=np.float64),
            np.full(12, q_end, dtype=np.float64),
        ])
        return dual.PlannedRound(
            position=position,
            velocity=np.zeros_like(position),
            acceleration=np.zeros_like(position),
            dt=0.02,
            q_end=position[-1].copy(),
            segments=ContinuousPipelineSearchIntegrationTest._segment_reports(
                seq1, seq2, marker
            ),
            arm1_choice=choice,
            arm2_choice=choice if seq2 else None,
            delay_stages=0,
            arm1_sequence=list(seq1),
            arm2_sequence=list(seq2),
        )

    @classmethod
    def _fixture(cls):
        choice = dual.AngleChoice(10.0, 20.0)
        seq1 = [_pose(f"arm1_i0_s{stage}") for stage in range(6)]
        seq2 = [_pose(f"arm2_i0_s{stage}") for stage in range(6)]
        chunks = [
            cls._round(0.0, 1.0, seq1[:3], [], "prime", choice),
            cls._round(1.0, 2.0, seq1[3:], seq2, "block_0", choice),
        ]
        return chunks, choice, seq1, seq2

    @classmethod
    def _assemble(cls, chunks, choice, seq1, seq2):
        return dual._assemble_pipeline(
            chunks=chunks,
            arm1_choices=[choice],
            arm2_choices=[choice],
            arm1_sequences=[seq1],
            arm2_sequences=[seq2],
            phase=3,
            angle_stage="primary",
            search_nodes=2,
            block_attempts={"prime": 1, "block_0": 1},
        )

    def test_valid_fixture_assembles_all_nine_events(self) -> None:
        chunks, choice, seq1, seq2 = self._fixture()
        pipeline = self._assemble(chunks, choice, seq1, seq2)
        self.assertEqual(len(pipeline.segments), 9)
        self.assertEqual(pipeline.position[:, 0].tolist(), [0.0, 1.0, 2.0])

    def test_rejects_boundary_dt_nonfinite_and_shape_violations(self) -> None:
        for violation in ("boundary", "dt", "nan", "shape"):
            with self.subTest(violation=violation):
                chunks, choice, seq1, seq2 = self._fixture()
                if violation == "boundary":
                    chunks[1].position[0, 0] = 99.0
                elif violation == "dt":
                    chunks[1].dt = 0.03
                elif violation == "nan":
                    chunks[1].position[1, 0] = np.nan
                else:
                    chunks[1].velocity = np.zeros((1, 12), dtype=np.float64)

                with self.assertRaises(ValueError):
                    self._assemble(chunks, choice, seq1, seq2)


class TwoStageRoundSearchTest(unittest.TestCase):
    @staticmethod
    def _success(marker: float):
        position = np.full((2, 12), marker, dtype=np.float64)
        return True, {
            "position": position,
            "velocity": np.zeros_like(position),
            "acceleration": np.zeros_like(position),
            "dt": 0.02,
            "q_end": position[-1].copy(),
            "segments": [{"marker": marker}],
        }

    @staticmethod
    def _failure(label: str):
        return False, {
            "status": label,
            "targets": ["arm1_target", "arm2_target"],
            "segments": [],
        }

    def _run_round(
        self,
        primary,
        stage2,
        max_pairs,
        schedule_results,
    ):
        def fake_sequence(
            _item, _place, choice, _pp, _item_idx, _transform, prefix
        ):
            return [_pose(f"{prefix}{choice.stage2_grasp}:{choice.stage2_place}")]

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(dual, "angle_combos", return_value=primary)
            )
            stack.enter_context(
                mock.patch.object(dual, "stage2_combos", return_value=stage2)
            )
            make_seq = stack.enter_context(
                mock.patch.object(
                    dual, "make_arm_sequence", side_effect=fake_sequence
                )
            )
            stack.enter_context(
                mock.patch.object(dual, "delay_candidates", return_value=[1])
            )
            plan = stack.enter_context(
                mock.patch.object(
                    dual, "plan_schedule", side_effect=schedule_results
                )
            )
            stack.enter_context(mock.patch("builtins.print"))
            planned, failures = dual.try_joint_round(
                mg=None,
                q_start=np.zeros(12, dtype=np.float64),
                item1_local=[-0.3, 0.1, 0.1],
                item2_local=[-0.3, 0.1, 0.1],
                place1_local=[-0.2, -0.2, 0.1],
                place2_local=[-0.2, -0.2, 0.1],
                item_idx=0,
                pp={},
                asr={"stage2": {"enable": True}},
                dc={
                    "max_pair_angle_trials": max_pairs,
                    "start_delay_stages": 1,
                    "max_start_delay_stages": 1,
                    "inactive_joint_tolerance_deg": 1.0,
                },
                transforms=(np.eye(4), np.eye(4)),
                last_choices=(None, None),
                pl={},
                cr={"prescreen_by_ik": False},
                linear_cfg={},
                q_lo=np.full(12, -10.0),
                q_hi=np.full(12, 10.0),
                ee_links=("arm1_ee", "arm2_ee"),
            )
        arm1_attempts = [
            call.args[2]
            for call in make_seq.call_args_list
            if call.args[6] == "arm1_"
        ]
        return planned, failures, plan, arm1_attempts

    def test_primary_success_is_followed_by_nonzero_stage2_and_refined_wins(self) -> None:
        planned, failures, plan, attempts = self._run_round(
            primary=[(10.0, 20.0)],
            stage2=[(0.0, 0.0), (15.0, -15.0)],
            max_pairs=4,
            schedule_results=[self._success(1.0), self._success(2.0)],
        )

        self.assertEqual(plan.call_count, 2)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(
            (attempts[0].stage2_grasp, attempts[0].stage2_place), (0.0, 0.0)
        )
        self.assertEqual(
            (attempts[1].stage2_grasp, attempts[1].stage2_place), (15.0, -15.0)
        )
        self.assertEqual(
            (planned.arm1_choice.stage2_grasp, planned.arm1_choice.stage2_place),
            (15.0, -15.0),
        )
        np.testing.assert_array_equal(planned.position, np.full((2, 12), 2.0))
        self.assertEqual(failures, [])

    def test_all_stage2_failures_return_cached_primary_round(self) -> None:
        planned, failures, plan, attempts = self._run_round(
            primary=[(10.0, 20.0)],
            stage2=[(0.0, 0.0), (15.0, -15.0)],
            max_pairs=0,
            schedule_results=[
                self._success(1.0),
                self._failure("S2_SAME_FAIL"),
                self._failure("S2_RIGHT_ONLY_FAIL"),
                self._failure("S2_LEFT_ONLY_FAIL"),
            ],
        )

        self.assertEqual(plan.call_count, 4)
        self.assertEqual(len(attempts), 4)
        self.assertEqual(
            (planned.arm1_choice.stage2_grasp, planned.arm1_choice.stage2_place),
            (0.0, 0.0),
        )
        np.testing.assert_array_equal(planned.position, np.full((2, 12), 1.0))
        self.assertEqual(len(failures), 3)
        self.assertTrue(all(failure["angle_stage"] == "stage2" for failure in failures))

    def test_primary_and_stage2_share_one_pair_trial_budget(self) -> None:
        planned, failures, plan, attempts = self._run_round(
            primary=[(10.0, 20.0), (30.0, 40.0)],
            stage2=[(0.0, 0.0), (5.0, -5.0), (10.0, -10.0)],
            max_pairs=3,
            schedule_results=[
                self._failure("PRIMARY_FIRST_FAIL"),
                self._success(2.0),
                self._failure("S2_FIRST_FAIL"),
            ],
        )

        # Two primary trials leave exactly one attempt for stage2.  Further
        # nonzero stage2 candidates must not receive a fresh phase budget.
        self.assertEqual(plan.call_count, 3)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(
            [(choice.grasp, choice.place) for choice in attempts[:2]],
            [(10.0, 20.0), (30.0, 40.0)],
        )
        self.assertNotEqual(
            (attempts[2].stage2_grasp, attempts[2].stage2_place), (0.0, 0.0)
        )
        self.assertEqual(
            (planned.arm1_choice.grasp, planned.arm1_choice.place), (30.0, 40.0)
        )
        self.assertEqual(
            (planned.arm1_choice.stage2_grasp, planned.arm1_choice.stage2_place),
            (0.0, 0.0),
        )
        self.assertEqual(len(failures), 2)


class PostValidationPublishingTest(unittest.TestCase):
    def test_failure_marker_precedes_diagnostic_trajectory_write(self) -> None:
        cfg = dual.load_dual_config(None)
        cfg["pick_place"]["on_fail"]["mode"] = "stop"
        cfg["dual_arm"]["task_layout"] = "same_local_grid"
        cfg["output"]["add_timestamp"] = False
        cfg["pick_place"]["home"]["joint_deg"] = [0.0] * 6
        cfg["pick_place"]["angle_search"]["stage2"]["enable"] = False
        cfg["workspace"]["check_after_plan"] = False
        cfg["workspace"]["report_gripper_extent"] = False
        cfg["workspace"].setdefault("wall", {})["enable"] = False

        choice = dual.AngleChoice(0.0, 0.0)
        arm1_sequence = [_pose(f"arm1_i0_s{stage}") for stage in range(6)]
        arm2_sequence = [_pose(f"arm2_i0_s{stage}") for stage in range(6)]
        segments = [
            {
                "global_event_index": event,
                "arm1_item_index": 0 if event < 6 else None,
                "arm2_item_index": 0 if 3 <= event < 9 else None,
            }
            for event in range(9)
        ]
        trajectory = np.zeros((2, 12), dtype=np.float64)
        pipeline = dual.PlannedPipeline(
            position=trajectory,
            velocity=np.zeros_like(trajectory),
            acceleration=np.zeros_like(trajectory),
            dt=0.02,
            q_end=trajectory[-1].copy(),
            segments=segments,
            chunks=[],
            arm1_choices=[choice],
            arm2_choices=[choice],
            arm1_sequences=[arm1_sequence],
            arm2_sequences=[arm2_sequence],
            angle_stage="primary",
            search_nodes=3,
            block_attempts={"prime": 1, "block_0": 1},
        )

        class FakeTensor:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float64)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        expected_joints = [f"J_{index}" for index in range(1, 7)] + [
            f"second_J_{index}" for index in range(1, 7)
        ]
        fake_limits = types.SimpleNamespace(
            position=[FakeTensor(np.full(12, -10.0)), FakeTensor(np.full(12, 10.0))]
        )
        fake_mg = types.SimpleNamespace(
            joint_names=expected_joints,
            kinematics=types.SimpleNamespace(
                get_joint_limits=lambda: fake_limits
            ),
        )
        fake_curobo = types.ModuleType("curobo")
        fake_curobo.__path__ = []
        fake_util_file = types.ModuleType("curobo.util_file")
        fake_util_file.get_assets_path = lambda: str(REPO_ROOT / "src/curobo/content/assets")

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            out_dir = Path(tmp) / "unsafe_result"
            cfg["output"]["dir"] = str(out_dir)
            args = types.SimpleNamespace(config=None, max_items=1)
            parser = mock.Mock()
            parser.parse_args.return_value = args
            rb = cfg["robot"]
            robot_dict = {
                "robot_cfg": {
                    "kinematics": {
                        "urdf_path": str((REPO_ROOT / rb["urdf"]).resolve())
                    }
                }
            }
            ee1_pos = np.zeros((2, 3), dtype=np.float64)
            ee2_pos = np.ones((2, 3), dtype=np.float64)
            ee_quat = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))

            stack.enter_context(
                mock.patch.dict(
                    sys.modules,
                    {"curobo": fake_curobo, "curobo.util_file": fake_util_file},
                )
            )
            stack.enter_context(mock.patch.object(dual, "dual_argparser", return_value=parser))
            stack.enter_context(mock.patch.object(dual, "load_dual_config", return_value=cfg))
            stack.enter_context(mock.patch.object(dual, "apply_dual_cli", side_effect=lambda value, _args: value))
            stack.enter_context(mock.patch.object(dual, "mounts_relative_transform", return_value=(np.eye(4), {})))
            stack.enter_context(mock.patch.object(dual, "urdf_link_transform", return_value=np.eye(4)))
            stack.enter_context(mock.patch.object(dual, "build_grasp_points", return_value=[{
                "index": 0, "row": 0, "col": 0,
                "position": [-0.6, -0.05, 0.03],
            }]))
            stack.enter_context(mock.patch.object(dual, "arm_primary_choices", return_value=[choice]))
            stack.enter_context(mock.patch.object(dual, "make_arm_sequence", return_value=arm1_sequence))
            stack.enter_context(mock.patch.object(dual, "make_world_config", return_value={}))
            stack.enter_context(mock.patch.object(dual, "load_robot_cfg_dict", return_value=robot_dict))
            stack.enter_context(mock.patch.object(dual, "make_motion_gen", return_value=fake_mg))
            stack.enter_context(mock.patch.object(
                dual,
                "make_separate_arm_ik_context",
                return_value=types.SimpleNamespace(
                    max_candidates_per_arm=8,
                    q_lo=np.full(12, -10.0),
                    q_hi=np.full(12, 10.0),
                ),
            ))
            stack.enter_context(mock.patch.object(
                dual, "try_continuous_pipeline",
                return_value=(pipeline, [], {"search_nodes": 3}),
            ))
            stack.enter_context(mock.patch.object(
                dual, "compute_fk_batched",
                return_value={
                    "TCP_LINK/pos": ee1_pos,
                    "TCP_LINK/quat": ee_quat,
                    "second_TCP_LINK/pos": ee2_pos,
                    "second_TCP_LINK/quat": ee_quat,
                },
            ))
            stack.enter_context(mock.patch.object(
                dual, "self_collision_report",
                return_value={"checked": True, "n_collision": 1},
            ))
            stack.enter_context(mock.patch.object(
                dual, "inter_arm_report",
                return_value={
                    "checked": True,
                    "n_collision_points": 0,
                    "margin_mm": 0.0,
                },
            ))
            stack.enter_context(mock.patch.object(dual, "compute_joint_motion_report", return_value={}))
            stack.enter_context(mock.patch.object(dual, "compute_joint_limit_margin_report", return_value={}))

            def fail_diagnostic_save(*_args, **_kwargs):
                marker = out_dir / "plan_failed.json"
                self.assertTrue(marker.is_file())
                payload = json.loads(marker.read_text(encoding="utf-8"))
                self.assertEqual(payload["stage"], "post_validation")
                self.assertEqual(payload["errors"][0]["check"], "self_collision")
                raise RuntimeError("mock trajectory write failed")

            save = stack.enter_context(mock.patch.object(
                dual, "save_trajectory", side_effect=fail_diagnostic_save
            ))
            stack.enter_context(mock.patch("builtins.print"))

            with self.assertRaisesRegex(RuntimeError, "mock trajectory write failed"):
                dual.main()

            self.assertEqual(save.call_count, 1)
            self.assertTrue((out_dir / "plan_failed.json").is_file())


class RvizSafetyGateTest(unittest.TestCase):
    def test_explicit_failed_trajectory_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unsafe_dir = Path(tmp) / "unsafe_result"
            unsafe_dir.mkdir()
            (unsafe_dir / "trajectory.npz").write_bytes(b"diagnostic only")
            (unsafe_dir / "plan_failed.json").write_text(
                '{"stage": "post_validation"}', encoding="utf-8"
            )

            result = subprocess.run(
                [
                    "bash",
                    str(TASK_ROOT / "run_rviz.sh"),
                    "--traj",
                    str(unsafe_dir),
                ],
                cwd=str(TASK_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("拒绝播放失败/不安全轨迹", result.stderr)
        self.assertIn(str(unsafe_dir / "plan_failed.json"), result.stderr)


class TrajectorySerializationTest(unittest.TestCase):
    def test_save_trajectory_writes_secondary_ee_arrays(self) -> None:
        n_points = 3
        joint_names = [f"J_{i}" for i in range(1, 7)] + [
            f"second_J_{i}" for i in range(1, 7)
        ]
        positions = np.arange(n_points * 12, dtype=np.float64).reshape(n_points, 12)
        velocities = positions + 0.1
        accelerations = positions + 0.2
        times = np.arange(n_points, dtype=np.float64) * 0.02
        ee1_pos = np.arange(n_points * 3, dtype=np.float64).reshape(n_points, 3)
        ee1_quat = np.tile([1.0, 0.0, 0.0, 0.0], (n_points, 1))
        ee2_pos = ee1_pos + 10.0
        ee2_quat = np.tile([0.0, 0.0, 0.0, 1.0], (n_points, 1))

        with tempfile.TemporaryDirectory() as tmp:
            npz_path, meta_path = save_trajectory(
                Path(tmp),
                joint_names,
                positions,
                velocities,
                accelerations,
                times,
                ee1_pos,
                ee1_quat,
                {"task_type": "test_dual"},
                extra_arrays={
                    "second_ee_positions": ee2_pos,
                    "second_ee_quats_wxyz": ee2_quat,
                },
            )

            self.assertTrue(meta_path.is_file())
            with np.load(npz_path, allow_pickle=False) as saved:
                self.assertEqual(saved["joint_names"].dtype.kind, "S")
                np.testing.assert_array_equal(saved["positions"], positions)
                np.testing.assert_array_equal(saved["second_ee_positions"], ee2_pos)
                np.testing.assert_array_equal(
                    saved["second_ee_quats_wxyz"], ee2_quat
                )


if __name__ == "__main__":
    unittest.main()
