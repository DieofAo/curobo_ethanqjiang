"""CPU-only checks for translating mounted configs without moving the task."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from derive_overhead_experiment import translate_mount
from plan_pick_place import load_pick_place_config
from prepare_overhead_config import prepare
from xtrainer_common import build_workspace_wall_cuboids, rpy_deg_to_matrix


class TranslateMountTest(unittest.TestCase):
    def setUp(self):
        self.source = prepare(load_pick_place_config(), [-.36, .15, .65],
                              mount_rpy=[180, 0, 90])
        self.source["pick_place"]["home"]["ik_seed_joint_deg"] = [1, 2, 3, 4, 5, 6]

    def test_task_transform_and_home_seed_preserved_at_all_candidates(self):
        saved = copy.deepcopy(self.source)
        old = np.asarray(self.source["robot"]["mount_transform"])
        original = old @ self.source["pick_place"]["link0_target_transform"]
        for y in (.15, .20, .25, .30):
            cfg = translate_mount(self.source, [-.36, y, .65])
            mount = np.asarray(cfg["robot"]["mount_transform"])
            np.testing.assert_allclose(mount[:3, :3], old[:3, :3], atol=1e-12)
            np.testing.assert_allclose(mount[:3, 3], [-.36, y, .65], atol=1e-12)
            np.testing.assert_allclose(mount @ cfg["pick_place"]["link0_target_transform"],
                                       original, atol=1e-12)
            for key in ("home", "grasp_grid", "place", "base_rpy", "lift", "angle_search",
                        "linear_move", "criterion"):
                self.assertEqual(cfg["pick_place"][key], self.source["pick_place"][key])
            self.assertEqual(cfg["planner"], self.source["planner"])
            self.assertEqual(cfg["robot"]["joint_limit_clip"], .15)
        self.assertEqual(self.source, saved)

    def test_world_boxes_and_collision_scope_remain_fixed(self):
        def physical_boxes(cfg):
            m = np.asarray(cfg["robot"]["mount_transform"])
            return [np.r_[m[:3, :3] @ b["pose"][:3] + m[:3, 3],
                          np.abs(m[:3, :3]) @ b["dims"]]
                    for b in build_workspace_wall_cuboids(cfg["workspace"])]
        before = physical_boxes(self.source)
        for y in (.20, .25, .30):
            cfg = translate_mount(self.source, [-.36, y, .65])
            after = physical_boxes(cfg)
            self.assertEqual(len(after), len(before))
            for box in before:
                self.assertTrue(any(np.allclose(box, a, atol=1e-12) for a in after))
            for key in ("enable", "thickness", "collision_link_names", "faces"):
                self.assertEqual(cfg["workspace"]["wall"][key],
                                 self.source["workspace"]["wall"][key])

    def test_nonidentity_original_task_correction_survives(self):
        c = np.eye(4)
        c[:3, :3] = rpy_deg_to_matrix([0, 0, -17])
        c[:3, 3] = [.01, -.03, .02]
        m = np.asarray(self.source["robot"]["mount_transform"])
        self.source["pick_place"]["link0_target_transform"] = (np.linalg.inv(m) @ c).tolist()
        self.source["overhead"]["original_target_transform"] = c.tolist()
        cfg = translate_mount(self.source, [-.36, .25, .65])
        np.testing.assert_allclose(np.asarray(cfg["robot"]["mount_transform"]) @
                                   cfg["pick_place"]["link0_target_transform"], c, atol=1e-12)

    def test_invalid_or_inconsistent_inputs_rejected(self):
        for position in ([0, 0], [0, 0, float("nan")], [0, 0, float("inf")]):
            with self.assertRaises(ValueError):
                translate_mount(self.source, position)
        for field, value in (("dual", "second_"), ("home", [0]*6), ("rpy", [0, 0, 0]),
                             ("correction", np.eye(4).tolist())):
            cfg = copy.deepcopy(self.source)
            if field == "dual": cfg["robot"]["dual_arm_prefix"] = value
            elif field == "home": cfg["pick_place"]["home"]["joint_deg"] = value
            elif field == "rpy": cfg["overhead"]["mount_rpy_deg"] = value
            else: cfg["pick_place"]["link0_target_transform"] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                translate_mount(cfg, [-.36, .25, .65])


if __name__ == "__main__":
    unittest.main()
