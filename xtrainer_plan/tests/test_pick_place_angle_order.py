#!/usr/bin/env python3
"""CPU regressions for descending pick/place angle search and legacy orders."""
import copy
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plan_pick_place as planner  # noqa: E402


def angle_config(order="desc", coupled=True):
    return {
        "order": order,
        "couple_place_to_grasp": coupled,
        "grasp": {"axis": "x", "min_deg": -30., "max_deg": 30., "step_deg": 2.},
        # Coupled search must ignore the independently configured place range.
        "place": {"axis": "x", "min_deg": -4., "max_deg": 4., "step_deg": 2.},
        "strategy": "abs_sum", "coarse_place": 2, "max_trials": 0,
    }


class PickPlaceAngleOrderTest(unittest.TestCase):
    def test_desc_has_all_31_angles_from_positive_to_negative(self):
        cfg = angle_config()
        before = copy.deepcopy(cfg)
        expected = list(range(30, -31, -2))
        self.assertEqual(planner.side_candidates(cfg["grasp"], "desc"), expected)
        self.assertEqual(planner.angle_combos(cfg), [(g, g) for g in expected])
        self.assertEqual(len(set(planner.angle_combos(cfg))), 31)
        self.assertEqual(cfg, before)

    def test_existing_abs_and_asc_orders_are_unchanged(self):
        for order, expected in (
            ("abs", [0., 2., -2., 4., -4.]),
            ("asc", [-4., -2., 0., 2., 4.]),
        ):
            with self.subTest(order=order):
                cfg = angle_config(order)
                cfg["grasp"].update(min_deg=-4., max_deg=4.)
                self.assertEqual(planner.side_candidates(cfg["grasp"], order), expected)
                self.assertEqual(planner.angle_combos(cfg), [(g, g) for g in expected])

    def test_nonnegative_grasp_min_retains_opposite_place_sign(self):
        cfg = angle_config()
        cfg["grasp"].update(min_deg=0., max_deg=4.)
        self.assertEqual(planner.angle_combos(cfg), [(4., -4.), (2., -2.), (0., 0.)])

    def test_first_reuse_still_precedes_order_and_deduplicates_before_truncation(self):
        first = (-12., -12.)
        for order in ("abs", "asc", "desc"):
            with self.subTest(order=order):
                cfg = angle_config(order)
                usual = planner.angle_combos(cfg)
                expected = [first] + [c for c in usual if c != first]
                self.assertEqual(planner.angle_combos(cfg, first=first), expected)
                self.assertEqual(len(planner.angle_combos(cfg, first=first)), 31)
                cfg["max_trials"] = 3
                self.assertEqual(planner.angle_combos(cfg, first=first), expected[:3])

    def test_each_unseeded_desc_search_starts_at_positive_30(self):
        cfg = angle_config()
        # The caller controls reuse; no candidate ordering state is retained.
        planner.angle_combos(cfg, first=(-12., -12.))
        for _ in range(3):
            self.assertEqual(planner.angle_combos(cfg, first=None)[0], (30., 30.))

    def test_desc_preserves_endpoint_inclusion_reversed_range_and_deduplication(self):
        side = {"min_deg": 5., "max_deg": -4., "step_deg": -4.}
        with redirect_stdout(io.StringIO()) as output:
            actual = planner.side_candidates(side, "desc", "test")
        self.assertIn("[WARN]", output.getvalue())
        self.assertEqual(actual, [5., 4., 0., -4.])
        self.assertEqual(planner.side_candidates(
            {"min_deg": 2., "max_deg": 2., "step_deg": 2.}, "desc"), [2.])

    def test_independent_desc_preserves_cartesian_and_chunk_semantics(self):
        cfg = angle_config(coupled=False)
        cfg["grasp"].update(min_deg=-2., max_deg=2., step_deg=4.)
        gs, ps = [2., -2.], [4., 2., 0., -2., -4.]
        self.assertEqual(planner.angle_combos(cfg), [(g, p) for g in gs for p in ps])
        cfg["strategy"] = "coarse_to_fine"
        expected = [(g, p) for start in (0, 2, 4) for g in gs for p in ps[start:start + 2]]
        self.assertEqual(planner.angle_combos(cfg), expected)

    def test_stage2_keeps_zero_first_then_each_configured_order(self):
        for order, sides in (("abs", [0., 2., -2.]),
                             ("asc", [-2., 0., 2.]),
                             ("desc", [2., 0., -2.])):
            with self.subTest(order=order):
                cfg = angle_config(order)
                cfg["stage2"] = {"min_deg": -2., "max_deg": 2., "step_deg": 2.}
                expected = [(g, p) for g in sides for p in sides]
                if order == "abs":
                    expected.sort(key=lambda pair: (abs(pair[0]) + abs(pair[1]),
                                                     abs(pair[0]), abs(pair[1])))
                expected = [(0., 0.)] + [pair for pair in expected if pair != (0., 0.)]
                self.assertEqual(planner.stage2_combos(cfg), expected)

    def test_stage2_explicit_order_overrides_primary_and_keeps_zero_fallback(self):
        cfg = angle_config("desc")
        cfg["stage2"] = {"min_deg": -4., "max_deg": -2., "step_deg": 2.,
                         "order": "asc", "max_trials": 3}
        self.assertEqual(planner.stage2_combos(cfg), [(0., 0.), (-4., -4.), (-4., -2.)])

    def test_cli_desc_applies_only_angle_order_and_requested_range(self):
        cfg = {"pick_place": {"grasp_grid": {}, "angle_search": angle_config("abs"),
                              "criterion": {"prescreen_by_ik": False}},
               "robot": {"joint_limit_clip": .14}, "output": {}}
        cfg["pick_place"]["angle_search"]["grasp"]["max_deg"] = 0.
        expected = copy.deepcopy(cfg)
        expected["pick_place"]["angle_search"]["order"] = "desc"
        expected["pick_place"]["angle_search"]["grasp"]["max_deg"] = 30.
        args = planner.build_argparser().parse_args(
            ["--search-order", "desc", "--grasp-range", "-30", "30", "--grasp-step", "2"])
        self.assertEqual(planner.apply_cli(cfg, args), expected)
        self.assertEqual(planner.angle_combos(cfg["pick_place"]["angle_search"])[0], (30., 30.))

    def test_cli_keeps_existing_orders_default_and_invalid_order_rejection(self):
        parser = planner.build_argparser()
        self.assertIsNone(parser.parse_args([]).search_order)
        for order in ("abs", "asc", "desc"):
            self.assertEqual(parser.parse_args(["--search-order", order]).search_order, order)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parser.parse_args(["--search-order", "descending"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
