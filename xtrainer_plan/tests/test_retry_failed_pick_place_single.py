#!/usr/bin/env python3
"""CPU-only tests for failed-item retry selection and narrowly scoped overrides."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import retry_failed_pick_place_single as retry


def source_fixture():
    config = {
        "robot": {
            "robot_yml": "xtrainer.yml", "dual_arm_prefix": None,
            "mount_transform": [[0, 0, 1, -.51], [1, 0, 0, .1], [0, 1, 0, .5], [0, 0, 0, 1]],
        },
        "world": {"walls": {"enabled": True}},
        "pick_place": {
            "home": {"joint_deg": None, "ik_seed_joint_deg": None,
                     "position": [-.31, .11, .03], "rpy_deg": [-180, 0, 0]},
            "grasp_grid": {"x_range": [-.61, -.01], "y_range": [-.36, .64]},
            "place": {"position": [-.16, -.23, .1]},
            "link0_target_transform": {"translation": [-.1, -.5, .51]},
            "angle_search": {"grasp": {"axis": "x", "min_deg": -30., "max_deg": 0., "step_deg": 2.},
                             "couple_place_to_grasp": True, "order": "abs", "reuse_last_success": True},
            "criterion": {"prescreen_by_ik": True, "max_joint_delta_deg": 170.,
                          "joints": [1, 2, 3, 4, 5], "min_limit_margin_deg": 0.},
            "linear_move": {"enable": True, "max_deviation_m": .003, "max_angle_deg": 5.},
            "on_fail": {"mode": "abort", "print_reasons": True},
        },
        "output": {"dir": "source-output", "add_timestamp": True},
    }
    items = [{"index": n, "success": ok, "position": [float(n), 0., .03]}
             for n, ok in [(0, True), (2, False), (7, False), (9, True)]]
    meta = {"config": config, "home_joint_deg": [1., 2., 3., 4., 5., 6.],
            "items": items, "n_items_total": 4, "n_items_success": 2}
    skipped = {"skipped": [copy.deepcopy(item) for item in items if not item["success"]]}
    return meta, skipped


class RetrySelectionTest(unittest.TestCase):
    def test_default_all_and_explicit_order(self):
        meta, skipped = source_fixture()
        self.assertEqual(retry.select_failed_indices(meta, skipped), ([2, 7], [2, 7]))
        self.assertEqual(retry.select_failed_indices(meta, skipped, [7, 2]), ([2, 7], [7, 2]))
        self.assertEqual(retry.select_failed_indices(meta, skipped, [7]), ([2, 7], [7]))

    def test_invalid_selections_rejected(self):
        meta, skipped = source_fixture()
        for indices, message in [([2, 2], "duplicate"), ([0], "successful"),
                                 ([3], "nonexistent"), ([-1], "nonexistent"), ([], "at least one")]:
            with self.subTest(indices=indices), self.assertRaisesRegex(RuntimeError, message):
                retry.select_failed_indices(meta, skipped, indices)

    def test_inconsistent_or_duplicate_source_rejected(self):
        meta, skipped = source_fixture()
        skipped["skipped"].reverse()
        with self.assertRaisesRegex(RuntimeError, "exactly match"):
            retry.select_failed_indices(meta, skipped)
        meta, skipped = source_fixture()
        meta["items"].append(copy.deepcopy(meta["items"][0]))
        with self.assertRaisesRegex(RuntimeError, "duplicate item"):
            retry.select_failed_indices(meta, skipped)

    def test_no_source_failures_rejected(self):
        meta, _ = source_fixture()
        for item in meta["items"]:
            item["success"] = True
        with self.assertRaisesRegex(RuntimeError, "no failed"):
            retry.select_failed_indices(meta, {"skipped": []})


class RetryConfigTest(unittest.TestCase):
    def test_defaults_preserve_previous_behavior_without_source_mutation(self):
        meta, _ = source_fixture()
        before = copy.deepcopy(meta)
        cfg, audit = retry.prepare_retry_config(meta, Path("new-output"))
        self.assertEqual(meta, before)
        self.assertEqual(cfg["pick_place"]["home"]["ik_seed_joint_deg"], meta["home_joint_deg"])
        self.assertEqual(cfg["pick_place"]["criterion"], meta["config"]["pick_place"]["criterion"])
        self.assertEqual(cfg["output"], {"dir": "new-output", "add_timestamp": False})
        self.assertEqual(cfg["pick_place"]["on_fail"]["mode"], "skip")
        self.assertIsNone(audit["angle_change"])
        self.assertIn("(default)", audit["home_seed"]["mode"])
        cfg["pick_place"]["home"]["ik_seed_joint_deg"][0] = 1000
        self.assertEqual(meta, before)

    def test_explicit_seed_prescreen_do_not_relax_any_other_config(self):
        meta, _ = source_fixture()
        seed = [10., -20., 30., -40., 50., -60.]
        cfg, audit = retry.prepare_retry_config(meta, "new", home_seed_deg=seed, skip_ik_prescreen=True)
        expected = copy.deepcopy(meta["config"])
        expected["pick_place"]["home"]["ik_seed_joint_deg"] = seed
        expected["pick_place"]["criterion"]["prescreen_by_ik"] = False
        expected["pick_place"]["on_fail"]["mode"] = "skip"
        expected["output"] = {"dir": "new", "add_timestamp": False}
        self.assertEqual(cfg, expected)
        self.assertTrue(audit["home_seed"]["target_pose_unchanged"])
        self.assertTrue(audit["criterion"]["source"]["prescreen_by_ik"])
        self.assertFalse(audit["criterion"]["effective"]["prescreen_by_ik"])
        self.assertFalse(audit["criterion"]["trajectory_acceptance_checks_relaxed"])

    def test_nonfinite_or_wrong_length_seed_rejected(self):
        meta, _ = source_fixture()
        for seed in [[0.] * 5, [0.] * 7, [float("nan")] + [0.] * 5,
                     [float("inf")] + [0.] * 5, [float("-inf")] + [0.] * 5]:
            with self.subTest(seed=seed), self.assertRaisesRegex(RuntimeError, "six finite"):
                retry.prepare_retry_config(meta, "new", home_seed_deg=seed)

    def test_joint_home_preserved_and_seed_override_rejected(self):
        meta, _ = source_fixture()
        meta["config"]["pick_place"]["home"]["joint_deg"] = [0.] * 6
        cfg, _ = retry.prepare_retry_config(meta, "new")
        self.assertEqual(cfg["pick_place"]["home"], meta["config"]["pick_place"]["home"])
        with self.assertRaisesRegex(RuntimeError, "home.joint_deg is explicit"):
            retry.prepare_retry_config(meta, "new", home_seed_deg=[0.] * 6)

    def test_grasp_override_and_invalid_bound(self):
        meta, _ = source_fixture()
        cfg, audit = retry.prepare_retry_config(meta, "new", grasp_max_deg=4.)
        self.assertEqual(cfg["pick_place"]["angle_search"]["grasp"]["max_deg"], 4.)
        self.assertEqual(audit["grasp_search"]["source"]["max_deg"], 0.)
        for maximum in [-31., float("nan"), float("inf")]:
            with self.subTest(maximum=maximum), self.assertRaises(RuntimeError):
                retry.prepare_retry_config(meta, "new", grasp_max_deg=maximum)


class RetryCliTest(unittest.TestCase):
    def test_parse_new_options_and_planner_argv(self):
        args = retry.parse_args(["source", "output", "--indices", "7", "2",
                                 "--home-seed-deg", "1", "2", "3", "4", "5", "6",
                                 "--skip-ik-prescreen", "--no-incremental-save"])
        self.assertEqual(args.indices, [7, 2])
        self.assertEqual(args.home_seed_deg, [1., 2., 3., 4., 5., 6.])
        self.assertTrue(args.skip_ik_prescreen)
        self.assertEqual(retry.build_planner_argv(args.no_incremental_save),
                         [str(retry.PLANNER_PATH), "--no-incremental-save"])
        self.assertEqual(retry.build_planner_argv(), [str(retry.PLANNER_PATH)])

    def test_validate_only_creates_nothing_and_never_imports_planner(self):
        meta, skipped = source_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "trajectory_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (source / "plan_skipped.json").write_text(json.dumps(skipped), encoding="utf-8")
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            output = root / "not-created" / "retry"
            with patch.object(retry, "REPO", root), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(retry.importlib.util, "spec_from_file_location",
                                 side_effect=AssertionError("planner import is forbidden")):
                status = retry.main([str(source), str(output), "--validate-only", "--indices", "7",
                                     "--skip-ik-prescreen", "--no-incremental-save"])
            self.assertEqual(status, 0)
            self.assertFalse(output.parent.exists())
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_retry_scope_records_all_failures_selection_and_strategy(self):
        meta, skipped = source_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "trajectory_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (source / "plan_skipped.json").write_text(json.dumps(skipped), encoding="utf-8")
            output = root / "retry"
            # Stop immediately before planner import: the persisted config is
            # inspectable without GPU dependencies or an actual planning run.
            with patch.object(retry, "REPO", root), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(retry.importlib.util, "spec_from_file_location", return_value=None):
                try:
                    with self.assertRaisesRegex(RuntimeError, "Cannot import planner"):
                        retry.main([str(source), str(output), "--indices", "7", "2",
                                    "--home-seed-deg", "10", "20", "30", "40", "50", "60",
                                    "--skip-ik-prescreen", "--no-incremental-save"])
                finally:
                    # main's Tee is intended to survive until the CLI exits.
                    # Close the test's log handle while preserving real stdout.
                    for stream in getattr(sys.stdout, "streams", []):
                        if stream is not sys.__stdout__:
                            stream.close()
            config = json.loads((output / "retry_config.json").read_text(encoding="utf-8"))
            scope = config["retry_scope"]
            self.assertEqual(scope["source_failed_item_indices"], [2, 7])
            self.assertEqual(scope["selected_item_indices"], [7, 2])
            self.assertEqual(scope["source_n_items_skipped"], 2)
            self.assertEqual(scope["selected_n_items"], 2)
            self.assertEqual(scope["source_trajectory_meta_sha256"], retry.sha256(source / "trajectory_meta.json"))
            self.assertEqual(scope["source_plan_skipped_sha256"], retry.sha256(source / "plan_skipped.json"))
            self.assertTrue(scope["no_incremental_save"])
            self.assertIn("--no-incremental-save", scope["planner_argv"])
            self.assertEqual(scope["search_strategy_changes"]["home_seed"]["effective_seed_deg"],
                             [10., 20., 30., 40., 50., 60.])


if __name__ == "__main__":
    unittest.main()
