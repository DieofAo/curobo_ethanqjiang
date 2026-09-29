#!/usr/bin/env python3
"""CPU-only contracts for exporting failed dual-arm Home IK candidates."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TASK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = TASK_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import export_dual_ik_results as exporter  # noqa: E402
from xtrainer_common import load_trajectory  # noqa: E402


def _home_pose(x: float, quaternion=None):
    return {
        "name": f"home_{x:g}",
        "kind": "start",
        "position": [x, x + 0.1, x + 0.2],
        "quat_wxyz": (
            [1.0, 0.0, 0.0, 0.0]
            if quaternion is None else list(quaternion)
        ),
    }


def _home_root(root_index: int):
    offset = 10.0 * root_index
    return {
        "angle_pair_trial": root_index + 1,
        "ik_branch_index": root_index % 2,
        "distance_to_seed": root_index + 0.25,
        "arm1_home_grasp_deg": -8.0 - root_index,
        "arm2_home_grasp_deg": -24.0 + root_index,
        "arm1_choice": {"angle_grasp_deg": -8.0 - root_index},
        "arm2_choice": {"angle_grasp_deg": -24.0 + root_index},
        "arm1_home": _home_pose(-0.2 - 0.01 * root_index),
        "arm2_home": _home_pose(-0.6 - 0.01 * root_index),
        "home_joint_deg": [offset + value for value in range(12)],
    }


def _failure_record():
    # Deliberately reverse the attempts.  Playback order is the stable
    # home_root_index order used by the planner, not JSON list order.
    return {
        "stage": "continuous_pipeline_root_search",
        "pipeline_attempts": [
            {
                "attempt": 2,
                "home_root_index": 1,
                "home_root": _home_root(1),
            },
            {
                "attempt": 1,
                "home_root_index": 0,
                "home_root": _home_root(0),
            },
        ],
        "config": {
            "robot": {
                "dual_arm_prefix": "second_",
                "base_link": "LINK_0",
                "ee_link": "TCP_LINK",
                "second_ee_link": "second_TCP_LINK",
                "mounts": "xtrainer_plan/config/cad_mounts_same_side.yaml",
            },
            "workspace": {
                "bounds": {
                    "x": [-0.7, 0.0],
                    "y": [-0.6, 0.6],
                    "z": [-0.1, 0.7],
                }
            },
        },
    }


class DualIkExportTest(unittest.TestCase):
    def test_export_is_accepted_by_the_existing_player_loader(self) -> None:
        record = _failure_record()
        expected_names = [f"J_{index}" for index in range(1, 7)] + [
            f"second_J_{index}" for index in range(1, 7)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "plan_failed.json"
            source.write_text(json.dumps(record), encoding="utf-8")

            npz_path, meta_path, count = exporter.export_ik_results(
                source, frame_seconds=0.25
            )

            self.assertEqual(count, 2)
            self.assertEqual(npz_path, source.parent / "ik_playback/ik_results.npz")
            self.assertEqual(meta_path, source.parent / "ik_playback/trajectory_meta.json")
            self.assertTrue(npz_path.is_file())
            self.assertTrue(meta_path.is_file())
            self.assertFalse((npz_path.parent / "plan_failed.json").exists())

            # This is the same allow_pickle=False loader called by
            # play_trajectory_ros.py before it imports any values into ROS.
            data, meta = load_trajectory(str(npz_path))
            self.assertEqual(data["joint_names"], expected_names)
            np.testing.assert_allclose(
                data["positions"],
                np.deg2rad([
                    _home_root(0)["home_joint_deg"],
                    _home_root(1)["home_joint_deg"],
                ]),
            )
            np.testing.assert_array_equal(data["times"], [0.0, 0.25])
            np.testing.assert_allclose(
                data["ee_positions"],
                [_home_root(0)["arm1_home"]["position"],
                 _home_root(1)["arm1_home"]["position"]],
            )
            np.testing.assert_allclose(
                data["second_ee_positions"],
                [_home_root(0)["arm2_home"]["position"],
                 _home_root(1)["arm2_home"]["position"]],
            )

            expected_shapes = {
                "positions": (2, 12),
                "times": (2,),
                "ee_positions": (2, 3),
                "ee_quats_wxyz": (2, 4),
                "second_ee_positions": (2, 3),
                "second_ee_quats_wxyz": (2, 4),
                "home_root_index": (2,),
                "attempt_index": (2,),
                "angle_pair_trial": (2,),
                "ik_branch_index": (2,),
                "distance_to_seed": (2,),
            }
            with np.load(npz_path, allow_pickle=False) as archive:
                self.assertEqual(archive["joint_names"].dtype.kind, "S")
                self.assertEqual(archive["joint_names"].shape, (12,))
                for key, shape in expected_shapes.items():
                    self.assertEqual(archive[key].shape, shape, key)
                for key in archive.files:
                    self.assertNotEqual(archive[key].dtype.kind, "O", key)
                for key in (
                    "positions", "times", "ee_positions", "ee_quats_wxyz",
                    "second_ee_positions", "second_ee_quats_wxyz",
                    "distance_to_seed",
                ):
                    self.assertEqual(archive[key].dtype, np.dtype(np.float64), key)
                np.testing.assert_array_equal(archive["home_root_index"], [0, 1])
                np.testing.assert_array_equal(archive["attempt_index"], [1, 2])

            self.assertEqual(meta["task_type"], "dual_arm_discrete_ik_candidates")
            self.assertTrue(meta["visualization_only"])
            self.assertFalse(meta["execution_safe"])
            self.assertFalse(meta["safe_to_execute"])
            self.assertFalse(meta["transition_checked"])
            self.assertEqual(meta["n_points"], 2)
            self.assertEqual(meta["frame_seconds"], 0.25)
            self.assertEqual(meta["robot"]["joint_names"], expected_names)
            self.assertEqual(
                meta["robot"]["mounts"],
                "xtrainer_plan/config/cad_mounts_same_side.yaml",
            )
            self.assertEqual(
                [candidate["home_root_index"] for candidate in meta["candidates"]],
                [0, 1],
            )
            self.assertEqual(Path(meta["source_plan_failed"]), source.resolve())

    def test_bad_records_and_bad_frame_period_fail_closed(self) -> None:
        cases = {}

        empty = _failure_record()
        empty["pipeline_attempts"] = []
        cases["empty_attempts"] = empty

        duplicate = _failure_record()
        duplicate["pipeline_attempts"][0]["home_root_index"] = 0
        cases["duplicate_root_index"] = duplicate

        gap = _failure_record()
        gap["pipeline_attempts"][0]["home_root_index"] = 2
        cases["non_contiguous_root_index"] = gap

        wrong_dof = _failure_record()
        wrong_dof["pipeline_attempts"][0]["home_root"]["home_joint_deg"] = [0.0] * 11
        cases["wrong_dof"] = wrong_dof

        nonfinite = _failure_record()
        nonfinite["pipeline_attempts"][0]["home_root"]["home_joint_deg"][3] = math.nan
        cases["nonfinite_joint"] = nonfinite

        bad_quaternion = _failure_record()
        bad_quaternion["pipeline_attempts"][0]["home_root"]["arm2_home"][
            "quat_wxyz"
        ] = [2.0, 0.0, 0.0, 0.0]
        cases["non_unit_quaternion"] = bad_quaternion

        missing_mounts = _failure_record()
        del missing_mounts["config"]["robot"]["mounts"]
        cases["missing_mounts"] = missing_mounts

        for label, record in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(exporter.IkExportError):
                    exporter.build_ik_playback_data(
                        record, Path("/tmp/plan_failed.json"), 0.5
                    )

        for frame_seconds in (0.0, -0.1, math.nan, math.inf):
            with self.subTest(frame_seconds=frame_seconds):
                with self.assertRaises(exporter.IkExportError):
                    exporter.build_ik_playback_data(
                        _failure_record(),
                        Path("/tmp/plan_failed.json"),
                        frame_seconds,
                    )

    def test_incomplete_or_malformed_input_never_publishes_an_npz(self) -> None:
        records = (
            {"stage": "planning_in_progress", "safe_to_play": False},
            {
                **_failure_record(),
                "pipeline_attempts": [{
                    "attempt": 1,
                    "home_root_index": 0,
                    "home_root": {
                        **_home_root(0),
                        "home_joint_deg": [0.0] * 11,
                    },
                }],
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, record in enumerate(records):
                with self.subTest(index=index):
                    source = root / f"case_{index}/plan_failed.json"
                    source.parent.mkdir()
                    source.write_text(json.dumps(record), encoding="utf-8")
                    output = root / f"output_{index}"
                    with self.assertRaises(exporter.IkExportError):
                        exporter.export_ik_results(source, output, 0.5)
                    self.assertFalse((output / exporter.IK_RESULTS_NPZ).exists())
                    self.assertFalse((output / exporter.PLAYBACK_META_JSON).exists())

    def test_output_containing_failure_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source/plan_failed.json"
            source.parent.mkdir()
            source.write_text(json.dumps(_failure_record()), encoding="utf-8")
            output = root / "unsafe_output"
            output.mkdir()
            marker = output / "plan_failed.json"
            marker.write_text('{"stage":"post_validation"}', encoding="utf-8")

            with self.assertRaisesRegex(
                exporter.IkExportError, "不能包含 plan_failed.json"
            ):
                exporter.export_ik_results(source, output, 0.5)

            self.assertEqual(
                marker.read_text(encoding="utf-8"),
                '{"stage":"post_validation"}',
            )
            self.assertFalse((output / exporter.IK_RESULTS_NPZ).exists())
            self.assertFalse((output / exporter.PLAYBACK_META_JSON).exists())


class DualIkRvizWrapperTest(unittest.TestCase):
    def test_shell_is_valid_and_help_does_not_start_rviz(self) -> None:
        wrapper = TASK_ROOT / "run_dual_ik_rviz.sh"
        syntax = subprocess.run(
            ["bash", "-n", str(wrapper)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        help_result = subprocess.run(
            ["bash", str(wrapper), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("只能可视化", help_result.stdout)

    def test_wrapper_exports_then_forwards_the_explicit_npz_safely(self) -> None:
        # Copy the wrapper under a temporary TASK_ROOT and replace run_rviz.sh
        # with an argument-printing stand-in.  This tests shell argument
        # handling without ROS, RViz, a display, or a GPU.
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp) / "task root with spaces"
            scripts = sandbox / "scripts"
            scripts.mkdir(parents=True)
            wrapper = sandbox / "run_dual_ik_rviz.sh"
            shutil.copy2(TASK_ROOT / "run_dual_ik_rviz.sh", wrapper)
            shutil.copy2(
                SCRIPTS_DIR / "export_dual_ik_results.py",
                scripts / "export_dual_ik_results.py",
            )
            fake_rviz = sandbox / "run_rviz.sh"
            fake_rviz.write_text(
                "#!/usr/bin/env bash\n"
                "for value in \"$@\"; do printf 'ARG<%s>\\n' \"$value\"; done\n",
                encoding="utf-8",
            )
            fake_rviz.chmod(0o755)

            source = sandbox / "failed result/plan_failed.json"
            source.parent.mkdir()
            source.write_text(json.dumps(_failure_record()), encoding="utf-8")
            output = sandbox / "IK output"
            result = subprocess.run(
                [
                    "bash", str(wrapper),
                    "--plan-failed", str(source),
                    "--out", str(output),
                    "--frame-seconds", "0.75",
                    "--once",
                    "--speed", "0.3",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            npz_path = (output / exporter.IK_RESULTS_NPZ).resolve()
            self.assertTrue(npz_path.is_file())
            self.assertTrue((output / exporter.PLAYBACK_META_JSON).is_file())
            forwarded = [
                line[4:-1]
                for line in result.stdout.splitlines()
                if line.startswith("ARG<") and line.endswith(">")
            ]
            self.assertEqual(forwarded[:2], ["--traj", str(npz_path)])
            self.assertIn("--rate-hz", forwarded)
            rate_index = forwarded.index("--rate-hz")
            self.assertAlmostEqual(float(forwarded[rate_index + 1]), 1.0 / 0.75)
            self.assertIn("--no-scan-cloud", forwarded)
            self.assertIn("--speed", forwarded)
            self.assertIn("0.3", forwarded)
            self.assertNotIn("--loop", forwarded)


if __name__ == "__main__":
    unittest.main()
