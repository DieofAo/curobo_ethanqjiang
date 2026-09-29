"""Small alignment/statistics tests; rendering is checked on the actual saved runs."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_original_base_joint_maps import population, validate_alignment


def fixture():
    rows = [{"index": i, "row": 0, "col": i, "x_m": -.5, "y_m": i * .02, "z_m": .03,
             "success": True} for i in range(3)]
    meta = {"grid": {"rows": 1, "cols": 3}, "place_position_raw": [-.36, -.12, .1]}
    report = {"joint": "J_6", "raw_limits_deg": [-179.91, 179.91], "effective_limits_deg": [-171.89, 171.89]}
    data = [copy.deepcopy((meta, rows, report)) for _ in range(3)]
    data[1][1][0]["success"] = False
    data[2][1][1]["success"] = False
    return data


class ComparisonMapTest(unittest.TestCase):
    def test_common_success_is_intersection_by_id_not_list_order(self):
        data = fixture()
        data[2][1].reverse()
        self.assertEqual(validate_alignment(data), [2])

    def test_reject_changed_positions_limits_and_duplicates(self):
        for mutation in ("position", "limit", "duplicate"):
            data = fixture()
            if mutation == "position":
                data[1][1][0]["x_m"] += .01
            elif mutation == "limit":
                data[1][2]["effective_limits_deg"][0] += .1
            else:
                data[1][1][1]["index"] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_alignment(data)

    def test_near_limit_denominator_excludes_failures_threshold_strict(self):
        values = [dict(success=True, index=i, grasp_deg=-90, place_deg=-170,
                       cycle_min_deg=-171, cycle_max_deg=5, effective_margin_deg=margin,
                       raw_margin_deg=margin + 8.02) for i, margin in enumerate([0, .999, 1.0])]
        values.append(dict(success=False, index=3))
        result = population(values)
        self.assertEqual(result["n_success"], 3)
        self.assertEqual(result["effective_margin_below_1deg"], {"count": 2, "denominator": 3, "indices": [0, 1]})


if __name__ == "__main__":
    unittest.main()
