"""CPU-only start selection tests, including read-only real trajectory checks."""

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

PLAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAN_ROOT / "scripts"))
from playback_start import resolve_playback_start  # noqa: E402


def example_meta():
    return {
        "task_type": "pick_place_cycle",
        "n_points": 9,
        "n_items_total": 6,
        "n_items_success": 3,
        "n_items_skipped": 1,
        "items": [
            {"index": 0, "success": True, "n_points": 3,
             "duration_s": 99999, "segments": [{"n_points": 2}, {"n_points": 2}]},
            {"index": 1, "success": False},
            {"index": 3, "success": True, "n_points": 4,
             "duration_s": 0, "segments": [{"n_points": 3}, {"n_points": 2}]},
            {"index": 5, "success": True, "n_points": 2},
        ],
    }


class PlaybackStartTest(unittest.TestCase):
    def test_default_and_no_metadata_needed_for_non_item_selectors(self):
        self.assertEqual(resolve_playback_start([12, 13], None), 0)
        self.assertEqual(resolve_playback_start([12, 13], {}, start_percent=100), 1)

    def test_percent_is_elapsed_time_not_sample_percentage(self):
        times = [50, 50.1, 50.2, 50.3, 60]
        self.assertEqual(resolve_playback_start(times, None, start_percent=50), 4)
        self.assertEqual(resolve_playback_start(times, None, start_percent=0), 0)
        self.assertEqual(resolve_playback_start(times, None, start_percent=100), 4)

    def test_time_relative_to_origin_chooses_first_at_or_after(self):
        times = [30, 31, 32, 35, 40]
        for requested, expected in [(0, 0), (1, 1), (1.5, 2), (2, 2), (9.99, 4), (10, 4)]:
            with self.subTest(requested=requested):
                self.assertEqual(resolve_playback_start(times, None, start_time=requested), expected)

    def test_duplicate_timestamps_choose_first_matching_sample(self):
        self.assertEqual(resolve_playback_start([0, 1, 1, 2, 2], None, start_time=1), 1)
        self.assertEqual(resolve_playback_start([0, 1, 1, 2, 2], None, start_percent=50), 1)
        self.assertEqual(resolve_playback_start([0, 1, 1, 2, 2], None, start_percent=100), 4)
        self.assertEqual(resolve_playback_start([0, 1, 1, 2, 2], None, start_time=2), 4)
        self.assertEqual(resolve_playback_start([0, 0, 1], None, start_percent=0), 0)
        self.assertEqual(resolve_playback_start([0, 0, 1], None, start_time=0), 0)

    def test_single_frame_and_zero_duration(self):
        for times in ([17], [17, 17, 17]):
            for percent in (0, 50):
                self.assertEqual(resolve_playback_start(times, None, start_percent=percent), 0)
            self.assertEqual(resolve_playback_start(times, None, start_percent=100), len(times) - 1)
            self.assertEqual(resolve_playback_start(times, None, start_time=0), 0)
            with self.assertRaises(ValueError):
                resolve_playback_start(times, None, start_time=0.01)

    def test_frame_numbers_are_zero_based_including_last(self):
        for requested in (0, 1, np.int64(2)):
            self.assertEqual(resolve_playback_start([20, 21, 22], None, start_frame=requested), requested)

    def test_mutually_exclusive_even_with_zero(self):
        names = ["start_percent", "start_time", "start_frame", "start_item"]
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                with self.subTest(left=names[left], right=names[right]):
                    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                        resolve_playback_start([0, 1], example_meta(), **{names[left]: 0, names[right]: 0})

    def test_invalid_numeric_selectors(self):
        for name, invalid in (
            ("start_percent", [-1, 100.01, float("inf"), float("nan"), True, "50"]),
            ("start_time", [-1, 2.01, float("inf"), float("nan"), False, "1"]),
            ("start_frame", [-1, 3, 1.0, True, "1", float("nan")]),
            ("start_item", [0, -1, 1.0, True, "1", float("nan")]),
        ):
            for value in invalid:
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        resolve_playback_start([0, 1, 2], example_meta(), **{name: value})

    def test_timing_validation_runs_even_without_selector(self):
        for times in ([], [[0, 1]], [1, 0], [0, float("nan")], [0, float("inf")], [-1e308, 1e308]):
            with self.subTest(times=times):
                with self.assertRaises(ValueError):
                    resolve_playback_start(times, None)

    def test_input_arrays_and_metadata_are_not_mutated(self):
        times = np.arange(9, dtype=float) + 100
        meta = example_meta()
        original_times = times.copy()
        original_meta = copy.deepcopy(meta)
        self.assertEqual(resolve_playback_start(times, meta, start_item=4), 3)
        np.testing.assert_array_equal(times, original_times)
        self.assertEqual(meta, original_meta)

    def test_item_prefix_counts_include_stored_start_state_without_minus_one(self):
        meta = example_meta()
        for requested, expected in ((1, 0), (4, 3), (6, 7)):
            self.assertEqual(resolve_playback_start(np.arange(9), meta, start_item=requested), expected)

    def test_failed_and_missing_items_do_not_select_following_case(self):
        with self.assertRaisesRegex(ValueError, "case #2 failed"):
            resolve_playback_start(np.arange(9), example_meta(), start_item=2)
        for number in (3, 5, 7):
            with self.subTest(number=number):
                with self.assertRaisesRegex(ValueError, f"case #{number} is not present"):
                    resolve_playback_start(np.arange(9), example_meta(), start_item=number)

    def test_incremental_metadata_missing_task_type_is_supported(self):
        meta = example_meta()
        del meta["task_type"]
        meta["partial"] = True
        meta["n_items_done"] = 4
        self.assertEqual(resolve_playback_start(np.arange(9), meta, start_item=6), 7)

    def test_wrong_task_type_and_missing_structure_rejected(self):
        for meta in (None, {}, {"items": []}, {"task_type": "dual_pick_place_interleaved"},
                     {"task_type": "dual_arm_ik_prescreen_trace"}):
            with self.subTest(meta=meta):
                with self.assertRaises(ValueError):
                    resolve_playback_start(np.arange(9), meta, start_item=1)

    def test_frame_count_mismatch_rejected_even_when_requested_item_is_first(self):
        meta = example_meta()
        meta["items"][-1]["n_points"] = 1
        with self.assertRaisesRegex(ValueError, "sum of successful item n_points=8"):
            resolve_playback_start(np.arange(9), meta, start_item=1)
        meta = example_meta()
        meta["n_points"] = 8
        with self.assertRaisesRegex(ValueError, "metadata n_points=8"):
            resolve_playback_start(np.arange(9), meta, start_item=1)

    def test_duplicate_and_invalid_item_indices(self):
        for value in (0, -1, 0.0, True, 6, None):
            meta = example_meta()
            meta["items"][-1]["index"] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_playback_start(np.arange(9), meta, start_item=1)

    def test_invalid_item_counts_success_flags_and_segments(self):
        mutations = (
            (0, "n_points", 0), (0, "n_points", -1), (0, "n_points", 3.0),
            (0, "n_points", True), (0, "success", 1), (0, "success", None),
            (1, "n_points", 1), (0, "segments", []),
            (0, "segments", [{"n_points": 3}, {"n_points": 3}]),
            (0, "segments", [{"n_points": -2}]),
        )
        for index, field, value in mutations:
            meta = example_meta()
            meta["items"][index][field] = value
            with self.subTest(index=index, field=field, value=value):
                with self.assertRaises(ValueError):
                    resolve_playback_start(np.arange(9), meta, start_item=1)

    def test_metadata_summary_counts_validated(self):
        for field, value in (("n_items_success", 4), ("n_items_skipped", 2),
                             ("n_items_done", 3), ("n_items_total", 5)):
            meta = example_meta()
            meta[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    resolve_playback_start(np.arange(9), meta, start_item=1)

    def test_percentage_multiplication_does_not_overflow(self):
        self.assertEqual(resolve_playback_start([0, 1e308], None, start_percent=100), 1)


class RealTrajectoryStartTest(unittest.TestCase):
    def check_saved_run(self, relative, expected_frames, expected_success):
        folder = PLAN_ROOT / "results_overhead" / "20260915" / relative
        if not (folder / "trajectory.npz").is_file():
            self.skipTest("optional local planning result is unavailable")
        meta = json.loads((folder / "trajectory_meta.json").read_text())
        with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
            times = data["times"]
            positions = data["positions"]
        self.assertEqual(len(times), expected_frames)
        self.assertEqual(len(positions), expected_frames)
        offset = 0
        successes = 0
        for item in meta["items"]:
            number = item["index"] + 1
            if not item["success"]:
                with self.assertRaisesRegex(ValueError, "failed planning"):
                    resolve_playback_start(times, meta, start_item=number)
                continue
            self.assertEqual(resolve_playback_start(times, meta, start_item=number), offset)
            if offset:
                # Complete per-case arrays retain q_cur: the first frame is
                # the preceding successful case's final robot state.
                np.testing.assert_allclose(positions[offset], positions[offset - 1], atol=1e-4, rtol=0)
            offset += item["n_points"]
            successes += 1
        self.assertEqual(offset, expected_frames)
        self.assertEqual(successes, expected_success)
        self.assertEqual(resolve_playback_start(times, meta, start_percent=100), expected_frames - 1)

    def test_v17_original_400_items(self):
        self.check_saved_run("v17_zx_original_20x20/zx_x-0.51_h0.50", 115306, 400)

    def test_v19_expanded_77_items_with_14_failed(self):
        self.check_saved_run("v19_zx_expanded_trajectory/zx_expanded_trajectory_10cm", 19191, 63)


if __name__ == "__main__":
    unittest.main()
