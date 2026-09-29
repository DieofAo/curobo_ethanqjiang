"""CPU-only checks for full-run XTrainer joint distribution reconstruction."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_full_mount_joint_distributions import (  # noqa: E402
    grid_layout, metric_rows, mount_tilt, verified_inputs, write_outputs,
)


class FullMountJointDistributionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.result = self.root / "result"
        self.result.mkdir()
        self.urdf = self.root / "robot.urdf"
        self.urdf.write_text("<robot name='fixture'>" + "".join(
            f"<joint name='J_{i}' type='revolute'><limit lower='-1' upper='1'/></joint>"
            for i in range(1, 7)) + "</robot>")
        self.urdf_hash = hashlib.sha256(self.urdf.read_bytes()).hexdigest()
        self.q = np.zeros((7, 6), dtype=float)
        self.q[:, 0] = [0, .4, .8, .5, .2, -.3, -.1]
        self.q[:, 5] = [0, -.6, .6, 0, 0, 0, 0]
        self.times = np.arange(7, dtype=float) * .02
        self.meta = self.make_meta()

    def make_meta(self):
        items = []
        for index in range(4):
            row, col = divmod(index, 2)
            item = {"index": index, "row": row, "col": col,
                    "position_raw": [float(row), float(col), .03],
                    "success": index == 0, "n_angles_tried": 1}
            if index == 0:
                item.update({
                    "n_points": 7, "duration_s": .12,
                    "angle_grasp_deg": 0., "angle_place_deg": 0.,
                    "min_limit_margin_deg": math.degrees(.1),
                    "max_joint_delta_deg": math.degrees(.5),
                    "segments": [{"index": phase, "to": f"i0_{name}", "n_points": 2}
                                 for phase, name in enumerate((
                                     "g_lift_in", "grasp", "g_lift_out",
                                     "p_lift_in", "place", "p_lift_out"))],
                })
            else:
                item["n_points"] = 0
            items.append(item)
        return {
            "task_type": "pick_place_cycle", "n_points": 7,
            "n_items_total": 4, "n_items_success": 1, "items": items,
            "grid": {"rows": 2, "cols": 2, "x_range": [0., 1.],
                     "y_range": [0., 1.], "z": .03},
            "criterion": {"joints": [1, 2, 3, 4, 5]},
            "robot": {"joint_names": [f"J_{i}" for i in range(1, 7)]},
            "config": {"robot": {"urdf": str(self.urdf), "joint_limit_clip": .1,
                                  "mount_transform": np.eye(4).tolist()},
                       "overhead": {"urdf_sha256": self.urdf_hash,
                                    "world_y_tilt_deg": 60.}},
        }

    def test_full_cycle_and_stage_metrics_differ_and_failures_stay_missing(self):
        raw = np.array([[-1.] * 6, [1.] * 6])
        rows = metric_rows(self.meta, self.q, self.times, raw,
                           raw + np.array([[.1] * 6, [-.1] * 6]), .1)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["max_cycle_joint"], "J_6")
        self.assertAlmostEqual(rows[0]["max_cycle_joint_span_deg"], math.degrees(1.2))
        self.assertEqual(rows[0]["max_stage_checked_joint"], "J_1")
        self.assertEqual(rows[0]["max_stage_checked_stage"], "place")
        self.assertAlmostEqual(rows[0]["max_stage_checked_span_deg"], math.degrees(.5))
        self.assertEqual(rows[0]["min_raw_margin_joint"], "J_1")
        self.assertEqual(rows[0]["min_effective_margin_sample"], 2)
        self.assertAlmostEqual(rows[0]["min_raw_margin_deg"], math.degrees(.2))
        self.assertAlmostEqual(rows[0]["min_effective_margin_deg"], math.degrees(.1))
        self.assertEqual(rows[0]["sample_end_exclusive"], len(self.q))
        for failed in rows[1:]:
            self.assertFalse(failed["success"])
            self.assertNotIn("min_raw_margin_deg", failed)
            self.assertNotIn("max_cycle_joint_span_deg", failed)
        files = write_outputs(self.root / "plots", self.meta, rows, .1, {"fixture": True})
        self.assertEqual(len(files), 8)
        self.assertTrue(all(file.is_file() for file in files))
        report = json.loads((self.root / "plots/joint_distribution_summary.json").read_text())
        self.assertEqual(report["n_failed_or_skipped"], 3)
        self.assertNotIn("min_raw_margin_deg", report["cases"][1])
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            write_outputs(self.root / "plots", self.meta, rows, .1, {})

    def test_plot_tilt_metadata_distinguishes_local_and_world_axes(self):
        self.assertEqual(mount_tilt(self.meta["config"]), ("task_world_y", 60.))
        local = {"overhead": {"tilt_axis": "original_base_local_y",
                              "local_y_tilt_deg": 60.}}
        self.assertEqual(mount_tilt(local), ("original_base_local_y", 60.))
        with self.assertRaisesRegex(ValueError, "both"):
            mount_tilt({"overhead": {**local["overhead"], "world_y_tilt_deg": 60.}})
        with self.assertRaisesRegex(ValueError, "identify"):
            mount_tilt({"overhead": {"local_y_tilt_deg": 60.}})

    def test_grid_rejects_position_mismatch(self):
        self.meta["items"][1]["position_raw"][0] = .3
        with self.assertRaisesRegex(ValueError, "recorded grasp target differs"):
            grid_layout(self.meta)

    def test_independent_audit_hashes_and_limits_are_required(self):
        meta_path = self.result / "trajectory_meta.json"
        meta_path.write_text(json.dumps(self.meta))
        npz_path = self.result / "trajectory.npz"
        np.savez(npz_path, positions=self.q, times=self.times,
                 joint_names=np.asarray([f"J_{i}" for i in range(1, 7)]))
        meta_hash = hashlib.sha256(meta_path.read_bytes()).hexdigest()
        npz_hash = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        lower, upper = [-1.] * 6, [1.] * 6
        effective_lower, effective_upper = [-.9] * 6, [.9] * 6
        verification = {
            "passed": True, "verification_completed": True,
            "gpu_checks": {"stored_fk_matches_independent_fk": True},
            "source": {"metadata_sha256": meta_hash, "npz_sha256": npz_hash,
                       "urdf_sha256": self.urdf_hash},
            "joint_names": self.meta["robot"]["joint_names"],
            "joint_position_limits": {"lower_rad": effective_lower,
                                      "upper_rad": effective_upper},
            "n_samples": 7, "n_items_total": 4, "n_items_success": 1,
        }
        clip_audit = {
            "passed": True, "verification_completed": True,
            "trajectory": {"passed": True, "checked": True, "n_samples": 7},
            "source": {"config_source_sha256": meta_hash,
                       "trajectory_sha256": npz_hash},
            "urdf": str(self.urdf), "urdf_sha256": self.urdf_hash,
            "joint_names": self.meta["robot"]["joint_names"], "clip_rad": .1,
            "raw_limits": {"lower_rad": lower, "upper_rad": upper},
            "expected_limits": {"lower_rad": effective_lower,
                                "upper_rad": effective_upper},
        }
        (self.result / "independent_verification.json").write_text(json.dumps(verification))
        (self.result / "joint_limit_clip_audit.json").write_text(json.dumps(clip_audit))
        _, q, _, _, _, clip, _ = verified_inputs(self.result)
        self.assertEqual(len(q), 7)
        self.assertEqual(clip, .1)
        verification["gpu_checks"]["stored_fk_matches_independent_fk"] = False
        (self.result / "independent_verification.json").write_text(json.dumps(verification))
        with self.assertRaisesRegex(ValueError, "FK comparison"):
            verified_inputs(self.result)
        verification["gpu_checks"]["stored_fk_matches_independent_fk"] = True
        (self.result / "independent_verification.json").write_text(json.dumps(verification))
        np.savez(npz_path, positions=self.q + .01, times=self.times,
                 joint_names=np.asarray([f"J_{i}" for i in range(1, 7)]))
        with self.assertRaisesRegex(ValueError, "Trajectory hash differs"):
            verified_inputs(self.result)


if __name__ == "__main__":
    unittest.main()
