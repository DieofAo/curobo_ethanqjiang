#!/usr/bin/env python3
"""CPU-only regression tests; never import CuRobo or construct GPU solvers."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from verify_joint_limit_clip import (  # noqa: E402
    audit_samples, audit_solver, compare_limits, expected_limits, read_urdf_limits,
)


class JointLimitClipTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.urdf = Path(self.tmp.name) / "model.urdf"
        self.urdf.write_text('''<robot name="test">
          <joint name="B" type="revolute"><limit lower="-2" upper="1"/></joint>
          <joint name="A" type="revolute"><limit lower="-3.14" upper="3.14"/></joint>
        </robot>''', encoding="utf-8")
        self.names = ["A", "B"]
        self.raw = read_urdf_limits(self.urdf, self.names)
        self.expected = expected_limits(self.raw, .15)

    def test_reads_ordered_raw_limits_and_clips_both_sides_once(self):
        np.testing.assert_allclose(self.raw, [[-3.14, -2], [3.14, 1]])
        np.testing.assert_allclose(self.expected, [[-2.99, -1.85], [2.99, .85]])
        np.testing.assert_allclose(self.raw, [[-3.14, -2], [3.14, 1]])

    def test_invalid_names_clip_and_empty_interval_fail(self):
        for names in (["missing"], ["A", "A"], []):
            with self.assertRaises(ValueError):
                read_urdf_limits(self.urdf, names)
        for clip in (-.1, float("nan"), 1.5):
            with self.assertRaises(ValueError):
                expected_limits(self.raw, clip)

    def test_missing_or_nonrevolute_limits_fail(self):
        for body in ('<joint name="A" type="fixed"/>',
                     '<joint name="A" type="revolute"/>',
                     '<joint name="A" type="continuous"><limit lower="-3" upper="3"/></joint>',
                     '<joint name="A" type="revolute"><mimic joint="B"/><limit lower="-3" upper="3"/></joint>'):
            self.urdf.write_text(f'<robot name="test">{body}</robot>', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_urdf_limits(self.urdf, ["A"])

    def test_extra_or_missing_clip_is_detected(self):
        self.assertTrue(compare_limits(self.expected.astype(np.float32), self.expected, 1e-6)["matches_expected"])
        for actual in (self.raw, expected_limits(self.raw, .30)):
            self.assertFalse(compare_limits(actual, self.expected, 1e-6)["matches_expected"])

    def test_saved_samples_require_margin_on_every_joint_and_both_ends(self):
        valid = audit_samples([[-2.99, .85], [2.99, -1.85]], self.names, self.raw, .15, 1e-6)
        self.assertTrue(valid["passed"])
        self.assertAlmostEqual(valid["minimum_raw_margin_rad"], .15)
        invalid = audit_samples([[-2.991, 0], [0, .851]], self.names, self.raw, .15, 1e-6)
        self.assertFalse(invalid["passed"])
        self.assertEqual(invalid["n_violation_samples"], 2)
        self.assertEqual(invalid["n_violations_per_joint"], [1, 1])
        self.assertEqual([v["joint"] for v in invalid["first_violations"]], ["A", "B"])
        json.dumps(invalid, allow_nan=False)
        self.assertTrue(audit_samples([[2.9900005, 0]], self.names, self.raw, .15, 1e-6)["passed"])

    def test_nonfinite_empty_wrong_shape_samples_fail(self):
        for q in ([[float("nan"), 0]], [], [[0]], [[0, float("inf")]]):
            with self.assertRaises(ValueError):
                audit_samples(q, self.names, self.raw, .15, 1e-6)

    def fake_solver(self, constraint_limits=None):
        kin = SimpleNamespace(generator_config=SimpleNamespace(urdf_path=str(self.urdf)),
                              get_joint_limits=lambda: SimpleNamespace(position=self.expected))
        constraint = SimpleNamespace(enabled=True, joint_limits=SimpleNamespace(
            position=self.expected if constraint_limits is None else constraint_limits))
        rollout = SimpleNamespace(joint_names=self.names, kinematics=kin, bound_constraint=constraint)
        return SimpleNamespace(joint_names=self.names, kinematics=kin,
                               get_all_rollout_instances=lambda: [rollout])

    def test_rollout_constraint_must_match_model(self):
        report = audit_solver(self.fake_solver(), self.names, self.expected, self.urdf, 1e-6)
        self.assertTrue(report["passed"])
        self.assertEqual(report["n_rollouts"], 1)
        report = audit_solver(self.fake_solver(self.raw), self.names, self.expected, self.urdf, 1e-6)
        self.assertFalse(report["passed"])
        self.assertFalse(report["rollouts"][0]["bound_constraint"]["matches_expected"])
        json.dumps(report, allow_nan=False)

    def test_empty_or_disabled_rollout_cannot_pass(self):
        solver = self.fake_solver()
        solver.get_all_rollout_instances()[0].bound_constraint.enabled = False
        self.assertFalse(audit_solver(solver, self.names, self.expected, self.urdf, 1e-6)["passed"])
        solver.get_all_rollout_instances = lambda: []
        self.assertFalse(audit_solver(solver, self.names, self.expected, self.urdf, 1e-6)["passed"])

    def test_cli_refuses_existing_report_before_any_solver_load(self):
        output = Path(self.tmp.name) / "report.json"
        output.write_text("existing user data\n", encoding="utf-8")
        result = subprocess.run([sys.executable, str(SCRIPTS / "verify_joint_limit_clip.py"),
                                 "missing.json", "--out", str(output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to replace", result.stderr)
        self.assertEqual(output.read_text(), "existing user data\n")


if __name__ == "__main__":
    unittest.main()
