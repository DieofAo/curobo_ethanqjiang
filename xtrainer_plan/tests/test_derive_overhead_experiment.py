#!/usr/bin/env python3
"""CPU regressions for disabling only the independent IK prescreen."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "derive_overhead_experiment.py"


class DeriveOverheadExperimentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.json"
        self.output = self.root / "variant.json"
        self.cfg = {
            "overhead": {"mount_rpy_deg": [180, 0, 90]},
            "robot": {"joint_limit_clip": .05, "dual_arm_prefix": None},
            "planner": {"self_collision_check": True, "self_collision_opt": True,
                        "position_threshold": .003, "rotation_threshold": .03,
                        "use_cuda_graph": True},
            "workspace": {"wall": {"enable": True,
                                    "collision_link_names": ["LINK_6", "LINK_3"]}},
            "pick_place": {
                "criterion": {"prescreen_by_ik": True, "max_joint_delta_deg": 170.,
                              "joints": [1, 2, 3, 4, 5], "min_limit_margin_deg": 0},
                "linear_move": {"enable": True, "max_deviation_mm": 3.,
                                "max_rotation_deg": 5., "hold_rotation": True},
                "home": {"position": [-.31, 0, .03], "rpy_deg": [-180, 0, 0]},
                "place": {"position": [-.16, -.23, .10]},
                "grasp_grid": {"rows": 3, "cols": 3},
                "angle_search": {"order": "abs", "reuse_last_success": True,
                                 "couple_place_to_grasp": True,
                                 "grasp": {"axis": "x", "min_deg": -30.,
                                           "max_deg": 0., "step_deg": 2.},
                                 "place": {"axis": "x", "min_deg": -30.,
                                           "max_deg": 30., "step_deg": 5.}},
            },
            "output": {"dir": "unchanged", "add_timestamp": False},
        }
        self.source.write_text(json.dumps(self.cfg), encoding="utf-8")
        self.source_bytes = self.source.read_bytes()

    def run_derive(self, *options, output=None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--source", str(self.source),
             "--output", str(output or self.output), *options],
            capture_output=True, text=True, check=False)

    def test_no_prescreen_changes_only_requested_criterion_and_provenance(self):
        result = self.run_derive("--no-prescreen")
        self.assertEqual(result.returncode, 0, result.stderr)
        variant = json.loads(self.output.read_text())
        expected = copy.deepcopy(self.cfg)
        expected["pick_place"]["criterion"]["prescreen_by_ik"] = False
        expected["overhead"]["derived_from"] = str(self.source.resolve())
        expected["overhead"]["variant_options"] = {
            "source": str(self.source), "output": str(self.output), "no_prescreen": True}
        self.assertEqual(variant, expected)
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_omitting_option_keeps_prescreen_enabled(self):
        result = self.run_derive()
        self.assertEqual(result.returncode, 0, result.stderr)
        variant = json.loads(self.output.read_text())
        self.assertEqual(variant["pick_place"], self.cfg["pick_place"])
        self.assertNotIn("no_prescreen", variant["overhead"]["variant_options"])

    def test_refuses_overwrite_of_source_or_existing_variant(self):
        result = self.run_derive("--no-prescreen", output=self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileExistsError", result.stderr)
        self.assertEqual(self.source.read_bytes(), self.source_bytes)
        self.output.write_text("existing user data\n", encoding="utf-8")
        result = self.run_derive("--no-prescreen")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileExistsError", result.stderr)
        self.assertEqual(self.output.read_text(), "existing user data\n")

    def test_joint_limit_clip_replaces_without_other_constraint_changes(self):
        result = self.run_derive("--joint-limit-clip", "0.15")
        self.assertEqual(result.returncode, 0, result.stderr)
        variant = json.loads(self.output.read_text())
        expected = copy.deepcopy(self.cfg)
        expected["robot"]["joint_limit_clip"] = .15
        expected["overhead"]["derived_from"] = str(self.source.resolve())
        expected["overhead"]["variant_options"] = {
            "source": str(self.source), "output": str(self.output), "joint_limit_clip": .15}
        self.assertEqual(variant, expected)
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_joint_limit_clip_rejects_invalid_values(self):
        for value in ("-0.1", "nan", "inf"):
            with self.subTest(value=value):
                result = self.run_derive("--joint-limit-clip", value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be finite and nonnegative", result.stderr)
                self.assertFalse(self.output.exists())

    def test_zero_joint_limit_clip_is_recorded(self):
        result = self.run_derive("--joint-limit-clip", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        variant = json.loads(self.output.read_text())
        self.assertEqual(variant["robot"]["joint_limit_clip"], 0.)
        self.assertEqual(variant["overhead"]["variant_options"]["joint_limit_clip"], 0.)

    def test_place_xy_changes_only_requested_axes(self):
        result = self.run_derive("--place-x", "-0.36", "--place-y", "-0.12")
        self.assertEqual(result.returncode, 0, result.stderr)
        variant = json.loads(self.output.read_text())
        expected = copy.deepcopy(self.cfg["pick_place"])
        expected["place"]["position"] = [-.36, -.12, .10]
        self.assertEqual(variant["pick_place"], expected)
        self.assertEqual(variant["robot"], self.cfg["robot"])
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_place_y_alone_preserves_x_and_zero_is_recorded(self):
        result = self.run_derive("--place-y", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        variant = json.loads(self.output.read_text())
        self.assertEqual(variant["pick_place"]["place"]["position"], [-.16, 0., .10])
        self.assertEqual(variant["overhead"]["variant_options"]["place_y"], 0.)

    def test_place_axes_reject_nonfinite(self):
        for flag in ("--place-x", "--place-y"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(flag=flag, value=value):
                    result = self.run_derive(f"{flag}={value}")
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("must be finite", result.stderr)
                    self.assertFalse(self.output.exists())

    def test_descending_grasp_changes_only_requested_angle_policy(self):
        result = self.run_derive("--grasp-angle-range", "-30", "30",
                                 "--grasp-angle-step", "2", "--search-order", "desc",
                                 "--no-angle-reuse")
        self.assertEqual(result.returncode, 0, result.stderr)
        variant = json.loads(self.output.read_text())
        expected = copy.deepcopy(self.cfg)
        expected["pick_place"]["angle_search"]["grasp"]["max_deg"] = 30.
        expected["pick_place"]["angle_search"]["order"] = "desc"
        expected["pick_place"]["angle_search"]["reuse_last_success"] = False
        variant.pop("overhead")
        expected.pop("overhead")
        self.assertEqual(variant, expected)
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_angle_options_reject_invalid_values(self):
        invalid = [("--grasp-angle-range", "30", "-30"),
                   ("--grasp-angle-range", "nan", "30"),
                   ("--grasp-angle-range", "-30", "inf"),
                   ("--grasp-angle-step", "0"), ("--grasp-angle-step", "-2"),
                   ("--grasp-angle-step", "nan"), ("--grasp-angle-step", "inf")]
        for options in invalid:
            with self.subTest(options=options):
                result = self.run_derive(*options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must", result.stderr)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
