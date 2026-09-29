#!/usr/bin/env python3
"""Focused checks that posture reports preserve local versus world tilt axes."""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import analyze_grasp_posture as posture  # noqa: E402


class PostureAxisTest(unittest.TestCase):
    def test_axis_metadata_accepts_legacy_world_and_explicit_local(self):
        world_item = {"name": "world", "world_y_tilt_deg": 60.}
        world_cfg = {"overhead": {"world_y_tilt_deg": 60.}}
        self.assertEqual(posture.candidate_tilt(world_item, world_cfg), ("task_world_y", 60.))
        local_item = {"name": "local", "local_y_tilt_deg": 60.,
                      "tilt_axis": "original_base_local_y"}
        local_cfg = {"overhead": {"local_y_tilt_deg": 60.,
                                  "tilt_axis": "original_base_local_y"}}
        self.assertEqual(posture.candidate_tilt(local_item, local_cfg),
                         ("original_base_local_y", 60.))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            posture.candidate_tilt({**local_item, "local_y_tilt_deg": 45.}, local_cfg)
        with self.assertRaisesRegex(ValueError, "explicit"):
            posture.candidate_tilt({**local_item, "tilt_axis": None}, local_cfg)
        with self.assertRaisesRegex(ValueError, "mixes"):
            posture.candidate_tilt({**local_item, "world_y_tilt_deg": 60.}, local_cfg)

    def test_local_angles_at_same_position_remain_distinct_in_plot_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for angle in (0., 60.):
                rows.append({"name": f"local{int(angle)}", "run": str(root / str(angle)),
                             "mount_xyz_m": [-.20, .15, .65],
                             "tilt_axis": "original_base_local_y", "tilt_deg": angle,
                             "local_y_tilt_deg": angle, "world_y_tilt_deg": None,
                             "status": "pending", "audit": "missing", "angle": None,
                             "n_success": None, "n_total": 9, "n_observed_grasps": 0,
                             "grasps": []})
            report = {"baseline": None, "candidates": rows}
            plot = root / "local_posture.png"
            posture.plot_report(report, plot)
            self.assertTrue(plot.read_bytes().startswith(b"\x89PNG"))
            cases, _ = posture.write_csv(report, root / "local_posture")
            content = cases.read_text(encoding="utf-8")
            self.assertIn("original_base_local_y", content)
            self.assertIn("local_y_tilt_deg", content)
            self.assertIn("local60", content)


if __name__ == "__main__":
    unittest.main()
