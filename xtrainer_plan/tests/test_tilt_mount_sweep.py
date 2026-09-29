#!/usr/bin/env python3
"""CPU-only checks for the mounted tilt sweep manifest and status semantics."""

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_tilt_mount_sweep as runner  # noqa: E402
import summarize_tilt_mount_sweep as sweep  # noqa: E402


class TiltSweepTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original = Path(__file__).resolve().parents[1] / (
            "results_overhead/20260917/v57_grasp30_asc_full/configs/v57_00.json")
        self.source = self.root / "source.json"
        self.source.write_bytes(original.read_bytes())
        self.base = json.loads(self.source.read_text())

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def candidate(self, name, xyz, tilt):
        cfg = copy.deepcopy(self.base)
        mount = np.asarray(cfg["robot"]["mount_transform"], dtype=float)
        mount[:3, :3] = sweep.ry(tilt) @ mount[:3, :3]
        mount[:3, 3] = xyz
        cfg["robot"]["mount_transform"] = mount.tolist()
        cfg["pick_place"]["grasp_grid"].update(rows=3, cols=3)
        cfg["pick_place"]["linear_move"].update(method="waypoints_fk", waypoint_step_m=.0075)
        result = self.root / "runs" / name
        cfg["output"].update(dir=str(result), add_timestamp=False)
        config = self.root / "configs" / f"{name}.json"
        self.write(config, cfg)
        return {"index": int(name[-1]), "name": name, "config": str(config),
                "result": str(result), "base_xyz_m": xyz, "world_y_tilt_deg": tilt,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}

    def manifest(self, candidates):
        value = {"source": str(self.source),
                 "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
                 "runtime_inputs": {}, "parameters": {"size": 3},
                 "n_configs": len(candidates), "candidates": candidates}
        path = self.root / "manifest.json"
        self.write(path, value)
        return path

    def test_geometry_angle_is_line_not_arm_angle(self):
        grid = {"x_range": [-.62, -.2], "y_range": [-.1, .4], "z": .03,
                "rows": 3, "cols": 3}
        angle = sweep.line_angles([-.41, .15, .45], grid)
        self.assertAlmostEqual(angle["center_deg"], 0.)
        self.assertGreater(angle["median_grid_deg"], 0.)
        self.assertLess(angle["median_grid_deg"], angle["max_grid_deg"])

    def test_pending_and_home_failure_are_not_success_rates(self):
        first = self.candidate("case0", [-.2, .15, .45], 0.)
        second = self.candidate("case1", [-.2, .35, .45], 30.)
        result = Path(second["result"])
        self.write(result / "run_status.json", {"config": second["config"],
                    "command": ["python", "plan_pick_place.py", "--config", second["config"]],
                    "returncode": 2, "elapsed_s": 2.})
        self.write(result / "plan_failed.json", {"stage": "home_ik"})
        manifest = self.manifest([first, second])
        report = sweep.aggregate(manifest)
        self.assertEqual(report["status_counts"], {"pending": 1, "home_failed": 1})
        self.assertEqual(report["n_ranked"], 0)
        self.assertIsNone(report["best_verified_success_fraction"])
        self.assertIsNone(report["candidates"][1]["n_success"])
        self.assertEqual(len(report["missing_cartesian_coordinates"]), 2)
        plot = self.root / "comparison.png"
        sweep.plot_report(report, plot)
        self.assertTrue(plot.read_bytes().startswith(b"\x89PNG"))
        sweep.write_csv(report, self.root / "comparison.csv")
        self.assertIn("home_failed", (self.root / "comparison.csv").read_text())
        runner.verify_inputs(json.loads(manifest.read_text()), manifest)
        self.assertEqual(runner.read_completed_status(result, Path(second["config"])),
                         json.loads((result / "run_status.json").read_text()))

    def test_existing_bad_audits_are_not_marked_passed_on_resume(self):
        item = self.candidate("case0", [-.2, .15, .45], 0.)
        result = Path(item["result"])
        cfg = json.loads(Path(item["config"]).read_text())
        self.write(result / "run_status.json", {"config": item["config"],
                    "command": ["python", "plan_pick_place.py", "--config", item["config"]],
                    "returncode": 0, "elapsed_s": 2.})
        items = [{"index": i, "position_raw": [-.62 + (i % 3) * .21,
                  -.1 + (i // 3) * .25, .03], "success": i == 0,
                  "n_points": 2 if i == 0 else 0} for i in range(9)]
        self.write(result / "trajectory_meta.json", {"config": cfg, "n_items_total": 9,
                   "n_items_success": 1, "n_points": 2, "items": items})
        for name in ("analysis_summary.json", "link3_grasp_clearance.json",
                     "independent_verification.json", "joint_limit_clip_audit.json"):
            self.write(result / name, {})
        outcome = runner.execute_candidate(item, self.root, SCRIPTS, io.StringIO())
        self.assertEqual(outcome["state"], "audit_failed")
        self.assertFalse(outcome["ranking_eligible"])

    def test_rotation_mismatch_and_changed_config_rejected(self):
        item = self.candidate("case0", [-.2, .15, .45], 30.)
        manifest = self.manifest([item])
        report = sweep.aggregate(manifest)
        self.assertEqual(report["status_counts"], {"pending": 1})
        cfg_path = Path(item["config"])
        cfg = json.loads(cfg_path.read_text())
        cfg["robot"]["mount_transform"][0][0] += .1
        self.write(cfg_path, cfg)
        with self.assertRaisesRegex(ValueError, "Config changed"):
            sweep.aggregate(manifest)


if __name__ == "__main__":
    unittest.main()
