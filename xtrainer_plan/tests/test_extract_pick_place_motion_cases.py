#!/usr/bin/env python3
"""CPU-only extraction/ranking tests, including existing playback consumers."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from extract_pick_place_motion_cases import PHASES, extract, rank_cases  # noqa: E402
from playback_start import resolve_playback_start  # noqa: E402
from xtrainer_common import load_trajectory  # noqa: E402
from build_overhead_scene_urdf import load_resolved_config  # noqa: E402


class ExtractCasesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "clips"
        items, poses, qs = [], [], []
        for index, span in enumerate((10., None, 20., 30., 40.)):
            if span is None:
                items.append({"index": index, "success": False})
                continue
            q = np.zeros((7, 6))
            # Full-cycle J6 range is span, while each individual segment spans
            # only span/2: neither J1--5 nor segment-max is the selection metric.
            q[:, 5] = np.radians([0, -span / 2, 0, span / 2, 0, 0, 0])
            q[:, 0] = q[:, 5] / 2
            qs.append(q)
            segments = [{"index": j, "to": f"i{index}_{phase}", "n_points": 2,
                         "kind": phase if phase in ("grasp", "place") else "lift"}
                        for j, phase in enumerate(PHASES)]
            items.append({"index": index, "success": True, "n_points": 7, "duration_s": .14,
                          "row": index, "col": 0, "position_raw": [-.5 + index * .01, .1, .03],
                          "angle_grasp_deg": -30., "angle_place_deg": -30., "segments": segments})
            poses.extend({"name": s["to"], "item_index": index, "kind": s["kind"],
                          "position": [0., 0., 0.], "quat_wxyz": [1., 0., 0., 0.], "rpy_deg": [0., 0., 0.]}
                         for s in segments)
        q = np.concatenate(qs)
        n = len(q)
        self.arrays = {"joint_names": np.asarray([f"J_{i}" for i in range(1, 7)], dtype="S64"),
                       "positions": q, "times": 5. + np.arange(n) * .02,
                       "ee_positions": np.arange(n * 3, dtype=np.float64).reshape(n, 3),
                       "ee_quats_wxyz": np.tile([1., 0., 0., 0.], (n, 1)),
                       "velocities": np.ones((n, 6), dtype=np.float32),
                       "accelerations": np.zeros((n, 6), dtype=np.float32),
                       "extra_sample_values": np.arange(n, dtype=np.int32),
                       "static_values": np.asarray([42., 24.])}
        self.meta = {"task_type": "pick_place_cycle", "n_items_total": 5, "n_items_success": 4,
                     "n_points": n, "items": items, "pose_sequence": poses, "interpolation_dt": .02,
                     "config": {"robot": {"mount_transform": np.eye(4).tolist()},
                                "pick_place": {"grasp_grid": {"x_range": [-.62, -.2], "y_range": [-.1, .4]}}},
                     "robot": {"base_link": "LINK_0", "ee_link": "TCP_LINK"},
                     "grid": {"n_items": 5}, "workspace": {"bounds": {"x": [-2, 0]}},
                     "place_position": [-.36, -.12, .1], "home_joint_deg": [5.] * 6,
                     "workspace_check": {"checked": True, "n_points": n},
                     "self_collision_check": {"checked": True, "n_points": n},
                     "joint_limit_margin": {"n_near_limit": [123] * 6},
                     "failed": {"failed_item_index": 1}, "skipped": [{"index": 1}]}
        self.save_source()

    def save_source(self):
        (self.source / "trajectory_meta.json").write_text(json.dumps(self.meta), encoding="utf-8")
        np.savez_compressed(self.source / "trajectory.npz", **self.arrays)

    def test_ranking_uses_all_six_joints_and_complete_cycle_span(self):
        ranked, selected, median = rank_cases(self.meta, self.arrays)
        self.assertEqual([r["index"] for r in ranked], [0, 2, 3, 4])
        self.assertEqual([r["rank_1based"] for r in ranked], [1, 2, 3, 4])
        self.assertAlmostEqual(median, 25.)
        self.assertEqual({k: v["index"] for k, v in selected.items()}, {"min": 0, "median": 2, "max": 4})
        for row, span in zip(ranked, (10, 20, 30, 40)):
            self.assertAlmostEqual(row["max_joint_span_deg"], span)
            self.assertAlmostEqual(row["max_segment_joint_span_deg"], span / 2)
            self.assertAlmostEqual(row["total_variation_per_joint_deg"][5], span * 2)
            self.assertEqual(row["max_span_joint"], "J_6")
        self.assertEqual(selected["median"]["source_sample_range_half_open"], [7, 14])
        self.assertEqual(selected["median"]["previous_successful_case_index"], 0)

    def test_equal_extrema_and_median_ties_use_lowest_original_index(self):
        arrays = copy.deepcopy(self.arrays)
        for k in range(4):
            arrays["positions"][k * 7:(k + 1) * 7] = self.arrays["positions"][:7]
        _, selected, _ = rank_cases(self.meta, arrays)
        self.assertEqual([r["index"] for r in selected.values()], [0, 0, 0])

    def test_exact_slices_endpoints_arrays_metadata_and_consumers(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        manifest = extract(self.source, self.output)
        self.assertEqual(manifest["n_ranked_cases"], 4)
        for row in manifest["selected"]:
            target = Path(row["result"])
            data, meta = load_trajectory(str(target))
            lo, hi = row["source_sample_range_half_open"]
            for key, original in self.arrays.items():
                if key == "joint_names":
                    self.assertEqual(data[key], [f"J_{i}" for i in range(1, 7)])
                else:
                    sliced = original[lo:hi] if original.shape[0] == len(self.arrays["times"]) else original
                    if key == "times":
                        sliced = sliced - self.arrays["times"][lo]
                    np.testing.assert_array_equal(data[key], sliced)
                    self.assertEqual(data[key].dtype, original.dtype)
            self.assertEqual(meta["n_items_total"], 5)  # original index space, not clip count
            self.assertEqual(meta["n_items_success"], 1)
            self.assertEqual(meta["n_items_done"], 1)
            self.assertEqual(meta["n_items_skipped"], 0)
            self.assertEqual(meta["n_points"], 7)
            self.assertAlmostEqual(meta["total_duration_s"], .12)
            self.assertEqual(meta["items"][0]["index"], row["index"])
            self.assertEqual(meta["items"][0]["original_index"], row["index"])
            self.assertEqual(meta["items"][0]["segments"], self.meta["items"][row["index"]]["segments"])
            self.assertEqual(len(meta["pose_sequence"]), 6)
            self.assertTrue(all(p["item_index"] == row["index"] for p in meta["pose_sequence"]))
            self.assertEqual(meta["config"], self.meta["config"])
            self.assertEqual(meta["extraction"]["source_home_joint_deg"], [5.] * 6)
            self.assertEqual(meta["extraction"]["source_item_duration_s"], .14)
            self.assertNotIn("segments", meta)  # top-level segments has another player schema
            for stale in ("workspace_check", "self_collision_check", "joint_limit_margin", "home_joint_deg"):
                self.assertNotIn(stale, meta)
            self.assertEqual(resolve_playback_start(data["times"], meta, start_item=row["index"] + 1), 0)
            cfg, _ = load_resolved_config(None, str(target))
            self.assertEqual(cfg, self.meta["config"])
            for name, digest in row["output_sha256"].items():
                self.assertEqual(hashlib.sha256((target / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), manifest)

    def test_existing_output_is_never_reused(self):
        self.output.mkdir()
        with self.assertRaisesRegex(FileExistsError, "Refusing to replace"):
            extract(self.source, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_invalid_metadata_or_arrays_fail_before_creating_output(self):
        original_meta, original_arrays = copy.deepcopy(self.meta), copy.deepcopy(self.arrays)
        mutations = [
            lambda: self.meta.update(partial=True),
            lambda: self.meta.update(n_items_success=3),
            lambda: self.meta["items"][2].update(n_points=6),
            lambda: self.meta["items"][2]["segments"].pop(),
            lambda: self.meta["items"][2].update(index=0),
            lambda: self.meta["pose_sequence"].pop(),
            lambda: self.arrays["positions"].__setitem__((7, 0), .5),
            lambda: self.arrays["positions"].__setitem__((7, 0), np.nan),
            lambda: self.arrays["times"].__setitem__(7, 0.),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.meta, self.arrays = copy.deepcopy(original_meta), copy.deepcopy(original_arrays)
                mutate()
                self.save_source()
                with self.assertRaises(ValueError):
                    extract(self.source, self.output)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
