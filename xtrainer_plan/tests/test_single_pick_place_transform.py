#!/usr/bin/env python3
"""CPU-only regression tests for single-arm LINK_0 target correction."""
from __future__ import annotations

import copy
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import yaml


TASK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = TASK_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import plan_pick_place as single  # noqa: E402
from xtrainer_common import (  # noqa: E402
    PoseSpec,
    build_place_position_tf_specs,
    parse_rigid_transform_matrix,
    quat_wxyz_to_matrix,
)


class SingleTransformConfigTest(unittest.TestCase):
    def test_ee_only_motiongen_config_keeps_collision_geometry_and_source(self):
        source = {"robot_cfg": {"kinematics": {
            "link_names": ["TCP_LINK", "gripper_link", "LINK_6"],
            "collision_link_names": ["LINK_0", "LINK_3", "LINK_6"],
            "collision_spheres": {"LINK_6": [{"center": [0, 0, 0], "radius": 0.1}]},
            "self_collision_ignore": {"LINK_0": ["LINK_1"]},
        }}}
        original = copy.deepcopy(source)
        actual = single.single_ee_motion_gen_config(source, "TCP_LINK")
        self.assertEqual(source, original)
        self.assertEqual(actual["robot_cfg"]["kinematics"].pop("link_names"), ["TCP_LINK"])
        original["robot_cfg"]["kinematics"].pop("link_names")
        self.assertEqual(actual, original)

    def test_default_and_partial_axis_angle_overlay_are_parseable(self) -> None:
        default_cfg = single.load_pick_place_config(None)
        default_raw = copy.deepcopy(
            default_cfg["pick_place"]["link0_target_transform"]
        )
        default_matrix = parse_rigid_transform_matrix(
            default_raw, "default link0 transform"
        )
        self.assertEqual(default_matrix.shape, (4, 4))

        requested_angle = 73.0
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "partial.yaml"
            custom.write_text(
                yaml.safe_dump({
                    "pick_place": {
                        "link0_target_transform": {
                            "rotation": {"angle_deg": requested_angle},
                        },
                    },
                }),
                encoding="utf-8",
            )
            merged = single.load_pick_place_config(str(custom))

        actual_raw = merged["pick_place"]["link0_target_transform"]
        self.assertEqual(actual_raw["rotation"]["angle_deg"], requested_angle)
        self.assertEqual(
            actual_raw["rotation"]["axis"], default_raw["rotation"]["axis"]
        )
        self.assertEqual(actual_raw["position"], default_raw["position"])
        expected_raw = copy.deepcopy(default_raw)
        expected_raw["rotation"]["angle_deg"] = requested_angle
        np.testing.assert_allclose(
            parse_rigid_transform_matrix(actual_raw),
            parse_rigid_transform_matrix(expected_raw),
        )

    def test_single_loader_accepts_legacy_path_with_old_4x4_value(self) -> None:
        legacy_matrix = [
            [0.0, -1.0, 0.0, 0.12],
            [1.0, 0.0, 0.0, -0.34],
            [0.0, 0.0, 1.0, 0.56],
            [0.0, 0.0, 0.0, 1.0],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "legacy.yaml"
            custom.write_text(
                yaml.safe_dump({
                    "dual_arm": {
                        "link0_target_transform": legacy_matrix,
                    },
                }),
                encoding="utf-8",
            )
            merged = single.load_pick_place_config(str(custom))

        np.testing.assert_allclose(
            parse_rigid_transform_matrix(
                merged["pick_place"]["link0_target_transform"]
            ),
            legacy_matrix,
        )
        self.assertNotIn(
            "link0_target_transform", merged.get("dual_arm", {})
        )


class _FakeTensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value, dtype=np.float64)

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class SingleTransformMainTest(unittest.TestCase):
    transform_cfg = {
        "position": [0.31, -0.22, 0.13],
        "rotation": {"axis": [0.0, 0.0, 2.0], "angle_deg": 90.0},
    }

    @staticmethod
    def _configured_task(transform_cfg, joint_deg=None):
        cfg = single.load_pick_place_config(None)
        pp = cfg["pick_place"]
        pp["link0_target_transform"] = copy.deepcopy(transform_cfg)
        pp["home"] = {
            "joint_deg": joint_deg,
            "position": [-0.12, 0.23, 0.34],
            "rpy_deg": [17.0, -23.0, 31.0],
        }
        pp["grasp_grid"].update({
            "x_range": [0.21, 0.21],
            "y_range": [-0.16, -0.16],
            "z": 0.27,
            "rows": 1,
            "cols": 1,
            "order": "row_major",
            "perimeter_only": False,
        })
        pp["place"]["position"] = [0.42, 0.18, 0.29]
        pp["base_rpy"] = {
            "grasp": [11.0, -19.0, 37.0],
            "place": [-29.0, 13.0, 41.0],
        }
        pp["angle_search"]["couple_place_to_grasp"] = False
        pp["angle_search"]["reuse_last_success"] = False
        pp["angle_search"]["max_trials"] = 1
        for side in ("grasp", "place"):
            pp["angle_search"][side].update({
                "min_deg": 0.0,
                "max_deg": 0.0,
                "step_deg": 1.0,
            })
        pp["angle_search"]["stage2"]["enable"] = False
        pp["criterion"]["prescreen_by_ik"] = False
        pp["lift"].update({
            "grasp_z": 0.05,
            "place_z": 0.07,
            "axis": "base_z",
        })
        pp["linear_move"]["enable"] = False
        cfg["workspace"]["check_after_plan"] = False
        cfg["workspace"]["report_gripper_extent"] = False
        cfg["workspace"].setdefault("wall", {})["enable"] = False
        cfg["planner"]["self_collision_check"] = False
        cfg["output"]["add_timestamp"] = False
        return cfg

    def _run_main(
        self,
        transform_cfg,
        *,
        joint_deg=None,
        linear=False,
        linear_method=None,
        prescreen=False,
        wall_filter=False,
    ):
        cfg = self._configured_task(transform_cfg, joint_deg=joint_deg)
        cfg["workspace"]["wall"]["enable"] = wall_filter
        cfg["workspace"]["wall"]["collision_link_names"] = ["LINK_6", "LINK_3"]
        cfg["pick_place"]["criterion"]["prescreen_by_ik"] = bool(prescreen)
        if linear:
            cfg["pick_place"]["linear_move"].update({
                "enable": True,
                "kinds": ["grasp", "place"],
                "free_axis": "z",
                "hold_rotation": True,
            })
            if linear_method is not None:
                cfg["pick_place"]["linear_move"].update({
                    "method": linear_method,
                    "waypoint_step_m": 0.0075,
                })

        fake_limits = types.SimpleNamespace(position=[
            _FakeTensor(np.full(6, -10.0)),
            _FakeTensor(np.full(6, 10.0)),
        ])
        fake_mg = types.SimpleNamespace(
            joint_names=[f"J_{index}" for index in range(1, 7)],
            kinematics=types.SimpleNamespace(
                get_joint_limits=lambda: fake_limits
            ),
        )
        q_from_pose_home = np.linspace(-0.15, 0.15, 6)
        initialization_events = []

        def make_fake_motion_gen(*_args, pre_warmup=None):
            if pre_warmup is not None:
                pre_warmup(fake_mg)
            initialization_events.append("warmup")
            return fake_mg

        def apply_fake_wall_filter(_solver, links):
            initialization_events.append("wall_filter")
            self.assertEqual(links, ["LINK_6", "LINK_3"])
            return {"wall_link_names": links}

        def successful_plan(_mg, q_start, _seq, pl, *_args):
            q_start = np.asarray(q_start, dtype=np.float64)
            positions = np.stack([q_start, q_start + 0.01], axis=0)
            zeros = np.zeros_like(positions)
            return True, {
                "position": positions,
                "velocity": zeros,
                "acceleration": zeros,
                "dt": float(pl["interpolation_dt"]),
                "q_end": positions[-1],
                "segments": [{
                    "max_joint_delta_deg": 0.0,
                    "min_limit_margin_deg": 90.0,
                }],
            }

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            out_dir = Path(tmp) / "single_transform"
            cfg["output"]["dir"] = str(out_dir)
            parser = mock.Mock()
            parser.parse_args.return_value = types.SimpleNamespace(
                config=None, max_items=1, no_incremental_save=True
            )
            stack.enter_context(mock.patch.object(
                single, "build_argparser", return_value=parser
            ))
            stack.enter_context(mock.patch.object(
                single, "load_pick_place_config", return_value=cfg
            ))
            stack.enter_context(mock.patch.object(
                single, "apply_cli", side_effect=lambda value, _args: value
            ))
            stack.enter_context(mock.patch.object(
                single, "make_world_config", return_value={}
            ))
            stack.enter_context(mock.patch.object(
                single, "load_robot_cfg_dict", return_value={"robot_cfg": {"kinematics": {
                    "link_names": ["TCP_LINK", "LINK_6"],
                    "collision_link_names": ["LINK_0", "LINK_3", "LINK_6"],
                }}}
            ))
            stack.enter_context(mock.patch.object(
                single, "make_motion_gen", side_effect=make_fake_motion_gen
            ))
            stack.enter_context(mock.patch.object(
                single, "restrict_world_collision_to_links", side_effect=apply_fake_wall_filter
            ))
            home_solver = stack.enter_context(mock.patch.object(
                single,
                "solve_ik_for_pose",
                return_value=(
                    q_from_pose_home.copy(),
                    types.SimpleNamespace(position_error=0.0, status="OK"),
                ),
            ))
            prescreener = stack.enter_context(mock.patch.object(
                single,
                "prescreen_angle",
                return_value=(True, {
                    "max_delta_deg": 0.0,
                    "min_limit_margin_deg": 90.0,
                    "at": "",
                    "joint": "J_1",
                }),
            ))
            planner = stack.enter_context(mock.patch.object(
                single, "plan_sequence", side_effect=successful_plan
            ))
            stack.enter_context(mock.patch.object(
                single,
                "compute_fk",
                side_effect=lambda _mg, positions, _links: {
                    "ee/pos": np.zeros((len(positions), 3), dtype=np.float64),
                    "ee/quat": np.tile(
                        [1.0, 0.0, 0.0, 0.0], (len(positions), 1)
                    ),
                },
            ))
            stack.enter_context(mock.patch.object(
                single, "compute_joint_motion_report", return_value={}
            ))
            stack.enter_context(mock.patch.object(
                single, "compute_joint_limit_margin_report", return_value={}
            ))
            saver = stack.enter_context(mock.patch.object(
                single,
                "save_trajectory",
                return_value=(
                    out_dir / "trajectory.npz",
                    out_dir / "trajectory_meta.json",
                ),
            ))
            stack.enter_context(mock.patch("builtins.print"))

            return_code = single.main()
            meta = saver.call_args.args[8]
            sequence = planner.call_args.args[2]
            q_start = np.asarray(planner.call_args.args[1], dtype=np.float64)
            linear_cfg = planner.call_args.args[7]

        return {
            "code": return_code,
            "cfg": cfg,
            "meta": meta,
            "sequence": sequence,
            "q_start": q_start,
            "linear": linear_cfg,
            "home_solver": home_solver,
            "prescreener": prescreener,
            "q_from_pose_home": q_from_pose_home,
            "initialization_events": initialization_events,
        }

    def test_world_link_filter_is_installed_before_motiongen_warmup(self):
        result = self._run_main(self.transform_cfg, wall_filter=True)
        self.assertEqual(result["code"], 0)
        self.assertEqual(result["initialization_events"], ["wall_filter", "warmup"])

    def assert_pose_is_left_transform(
        self, actual: PoseSpec, raw: PoseSpec, correction: np.ndarray
    ) -> None:
        np.testing.assert_allclose(
            actual.position,
            correction[:3, :3] @ raw.position + correction[:3, 3],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            quat_wxyz_to_matrix(actual.quat_wxyz),
            correction[:3, :3] @ quat_wxyz_to_matrix(raw.quat_wxyz),
            atol=1e-12,
        )

    def test_axis_angle_left_multiplies_home_grasp_place_and_metadata(self) -> None:
        result = self._run_main(self.transform_cfg, prescreen=True)
        self.assertEqual(result["code"], 0)
        correction = parse_rigid_transform_matrix(self.transform_cfg)
        cfg = result["cfg"]
        pp = cfg["pick_place"]

        raw_home = PoseSpec.from_rpy_deg(
            "home",
            pp["home"]["position"],
            pp["home"]["rpy_deg"],
            "start",
        )
        planned_home = result["home_solver"].call_args.args[3]
        self.assert_pose_is_left_transform(planned_home, raw_home, correction)
        np.testing.assert_allclose(
            result["q_start"], result["q_from_pose_home"]
        )

        raw_grasp = [
            pp["grasp_grid"]["x_range"][0],
            pp["grasp_grid"]["y_range"][0],
            pp["grasp_grid"]["z"],
        ]
        raw_place = pp["place"]["position"]
        raw_sequence = single.make_round_poses(
            raw_grasp, raw_place, 0.0, 0.0, pp, 0
        )
        self.assertEqual(len(result["sequence"]), len(raw_sequence))
        result["prescreener"].assert_called_once()
        self.assertIs(
            result["prescreener"].call_args.args[5], result["sequence"]
        )
        for actual, raw in zip(result["sequence"], raw_sequence):
            self.assert_pose_is_left_transform(actual, raw, correction)

        grasp = next(p for p in result["sequence"] if p.kind == "grasp")
        place = next(p for p in result["sequence"] if p.kind == "place")
        self.assert_pose_is_left_transform(
            grasp, next(p for p in raw_sequence if p.kind == "grasp"), correction
        )
        self.assert_pose_is_left_transform(
            place, next(p for p in raw_sequence if p.kind == "place"), correction
        )

        meta = result["meta"]
        expected_place = (
            correction[:3, :3] @ np.asarray(raw_place, dtype=np.float64)
            + correction[:3, 3]
        )
        np.testing.assert_allclose(
            meta["robot"]["link0_target_transform"], correction
        )
        np.testing.assert_allclose(meta["place_position_raw"], raw_place)
        np.testing.assert_allclose(meta["place_position"], expected_place)
        np.testing.assert_allclose(meta["items"][0]["position_raw"], raw_grasp)
        np.testing.assert_allclose(
            meta["items"][0]["position"],
            correction[:3, :3] @ np.asarray(raw_grasp) + correction[:3, 3],
        )
        np.testing.assert_allclose(
            meta["items"][0]["effective_position"],
            correction[:3, :3] @ np.asarray(raw_grasp) + correction[:3, 3],
        )
        self.assertEqual(len(meta["pose_sequence"]), len(result["sequence"]))
        for record, pose in zip(meta["pose_sequence"], result["sequence"]):
            np.testing.assert_allclose(record["position"], pose.position)
            np.testing.assert_allclose(
                quat_wxyz_to_matrix(record["quat_wxyz"]),
                quat_wxyz_to_matrix(pose.quat_wxyz),
            )

        marker_specs = build_place_position_tf_specs(meta)
        self.assertEqual(len(marker_specs), 1)
        self.assertEqual(marker_specs[0][0], "xtrainer_arm1_place")
        np.testing.assert_allclose(marker_specs[0][1], expected_place)
        np.testing.assert_allclose(
            quat_wxyz_to_matrix(marker_specs[0][2]),
            correction[:3, :3],
            atol=1e-12,
        )

    def test_explicit_joint_home_is_not_transformed_or_resolved_by_ik(self) -> None:
        joint_deg = [7.0, -11.0, 19.0, -23.0, 31.0, -37.0]
        result = self._run_main(self.transform_cfg, joint_deg=joint_deg)

        self.assertEqual(result["code"], 0)
        result["home_solver"].assert_not_called()
        np.testing.assert_allclose(result["q_start"], np.radians(joint_deg))
        np.testing.assert_allclose(result["meta"]["home_joint_deg"], joint_deg)
        correction = parse_rigid_transform_matrix(self.transform_cfg)
        pp = result["cfg"]["pick_place"]
        raw_sequence = single.make_round_poses(
            [
                pp["grasp_grid"]["x_range"][0],
                pp["grasp_grid"]["y_range"][0],
                pp["grasp_grid"]["z"],
            ],
            pp["place"]["position"],
            0.0,
            0.0,
            pp,
            0,
        )
        for actual, raw in zip(result["sequence"], raw_sequence):
            self.assert_pose_is_left_transform(actual, raw, correction)

    def test_linear_free_axis_is_mapped_after_link0_rotation(self) -> None:
        rotate_local_z_to_root_x = {
            "position": [0.0, 0.0, 0.0],
            "rotation": {"axis": "y", "angle_deg": 90.0},
        }
        result = self._run_main(
            rotate_local_z_to_root_x,
            joint_deg=[0.0] * 6,
            linear=True,
        )

        self.assertEqual(result["code"], 0)
        self.assertEqual(result["linear"]["configured_free_axis"], "z")
        self.assertEqual(result["linear"]["free_axis"], "x")

    def test_tilted_linear_waypoints_preserve_task_direction(self) -> None:
        tilt = {
            "position": [0.0, 0.0, 0.0],
            "rotation": {"axis": "y", "angle_deg": 45.0},
        }
        result = self._run_main(
            tilt, joint_deg=[0.0] * 6, linear=True,
            linear_method="waypoints_fk",
        )
        self.assertEqual(result["code"], 0)
        self.assertEqual(result["linear"]["method"], "waypoints_fk")
        self.assertEqual(result["linear"]["free_axis"], "vector")
        np.testing.assert_allclose(
            result["linear"]["free_direction_root"],
            [np.sqrt(0.5), 0.0, np.sqrt(0.5)], atol=1e-12,
        )
        self.assertEqual(result["linear"]["waypoint_step_m"], 0.0075)

    def test_linear_free_axis_rejects_non_cardinal_result(self) -> None:
        rotate_local_z_to_diagonal = parse_rigid_transform_matrix({
            "position": [0.0, 0.0, 0.0],
            "rotation": {"axis": "y", "angle_deg": 45.0},
        })

        with self.assertRaisesRegex(ValueError, "不是规划 root"):
            single.transformed_cardinal_axis(
                rotate_local_z_to_diagonal, "z"
            )

    def test_invalid_transform_fails_before_world_or_motion_gen(self) -> None:
        invalid = {
            "position": [0.0, 0.0, 0.0],
            "rotation": {"axis": [0.0, 0.0, 0.0], "angle_deg": 0.0},
        }
        cfg = self._configured_task(invalid, joint_deg=[0.0] * 6)
        parser = mock.Mock()
        parser.parse_args.return_value = types.SimpleNamespace(
            config=None, max_items=1
        )

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                single, "build_argparser", return_value=parser
            ))
            stack.enter_context(mock.patch.object(
                single, "load_pick_place_config", return_value=cfg
            ))
            stack.enter_context(mock.patch.object(
                single, "apply_cli", side_effect=lambda value, _args: value
            ))
            world_builder = stack.enter_context(mock.patch.object(
                single, "make_world_config"
            ))
            motion_gen_builder = stack.enter_context(mock.patch.object(
                single, "make_motion_gen"
            ))
            stack.enter_context(mock.patch("builtins.print"))

            return_code = single.main()

        self.assertEqual(return_code, 6)
        world_builder.assert_not_called()
        motion_gen_builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
