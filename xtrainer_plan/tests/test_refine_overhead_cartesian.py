#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from refine_overhead_cartesian import generate, sha256  # noqa: E402


class RefineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {"robot": {"mount_transform": [[0, 1, 0, -.41], [1, 0, 0, .35],
                                                  [0, 0, -1, .65], [0, 0, 0, 1]],
                              "joint_limit_clip": .14},
                    "pick_place": {"grasp_grid": {"rows": 3, "cols": 3, "perimeter_only": False},
                                   "place": {"position": [-.26, -.12, .1]},
                                   "home": {"position": [-.31, 0, .03], "ik_seed_joint_deg": [1] * 6},
                                   "criterion": {"prescreen_by_ik": False}},
                    "output": {"dir": "original", "add_timestamp": False},
                    "overhead": {"mount_rpy_deg": [180, 0, 90], "derived_from": "old"}}
        self.rows = []
        for index, name in enumerate(("coarse_a", "coarse_b")):
            cfg = copy.deepcopy(self.cfg)
            cfg["robot"]["mount_transform"][0][3] += index * .1
            path = self.root / f"{name}.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            self.rows.append({"index": index, "name": name, "config": str(path),
                              "result": str(self.root / "runs" / name),
                              "config_sha256": sha256(path),
                              "base_xyz_m": [cfg["robot"]["mount_transform"][i][3] for i in range(3)],
                              "place_xyz_m": cfg["pick_place"]["place"]["position"]})
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"n_configs": 2, "candidates": self.rows}), encoding="utf-8")
        self.out = self.root / "refined"

    def test_explicit_order_unchanged_geometry_and_runner_compatible_manifest(self):
        report = generate(self.manifest, ["coarse_b", "coarse_a"], self.out, "v53")
        self.assertEqual(report["n_configs"], 2)
        self.assertEqual([row["coarse_name"] for row in report["candidates"]], ["coarse_b", "coarse_a"])
        for index, row in enumerate(report["candidates"]):
            self.assertEqual(row["name"], f"v53_{index:02d}")
            cfg = json.loads(Path(row["config"]).read_text())
            old = json.loads(Path(row["coarse_config"]).read_text())
            self.assertEqual(cfg["pick_place"]["grasp_grid"]["rows"], 5)
            self.assertEqual(cfg["pick_place"]["grasp_grid"]["cols"], 5)
            self.assertEqual(cfg["robot"], old["robot"])
            self.assertEqual(cfg["pick_place"]["home"], old["pick_place"]["home"])
            self.assertEqual(cfg["pick_place"]["place"], old["pick_place"]["place"])
            self.assertEqual(row["config_sha256"], sha256(row["config"]))
            self.assertEqual(Path(row["result"]).parent, self.out / "runs")
            self.assertEqual(Path(row["result"]).name, Path(row["config"]).stem)
            self.assertFalse(Path(row["result"]).exists())

    def test_optional_matched_baseline_only_changes_requested_fields(self):
        baseline = copy.deepcopy(self.cfg)
        baseline["robot"]["mount_transform"][0][3] = -.36
        baseline["robot"]["mount_transform"][1][3] = .15
        baseline["robot"]["joint_limit_clip"] = .15
        baseline["pick_place"]["place"]["position"] = [-.16, -.23, .1]
        baseline["pick_place"]["criterion"]["prescreen_by_ik"] = True
        path = self.root / "baseline.json"
        path.write_text(json.dumps(baseline), encoding="utf-8")
        report = generate(self.manifest, ["coarse_b"], self.out, "v53", baseline_source=path)
        row = report["candidates"][-1]
        cfg = json.loads(Path(row["config"]).read_text())
        self.assertEqual(row["role"], "matched_baseline")
        self.assertEqual(row["base_xyz_m"], [-.36, .15, .65])
        self.assertEqual(row["place_xyz_m"], [-.16, -.12, .1])
        self.assertEqual(cfg["robot"]["joint_limit_clip"], .14)
        self.assertFalse(cfg["pick_place"]["criterion"]["prescreen_by_ik"])
        self.assertEqual(cfg["pick_place"]["home"], baseline["pick_place"]["home"])
        self.assertNotIn("--place-x", row["derive_command"])

    def test_invalid_selection_validated_before_output_creation(self):
        for names in (["missing"], ["coarse_a", "coarse_a"], []):
            with self.assertRaises(ValueError):
                generate(self.manifest, names, self.out, "v53")
            self.assertFalse(self.out.exists())

    def test_stale_source_rejected_before_output_creation(self):
        path = Path(self.rows[1]["config"])
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed after recording"):
            generate(self.manifest, ["coarse_a", "coarse_b"], self.out, "v53")
        self.assertFalse(self.out.exists())

    def test_existing_root_and_invalid_prefix_rejected(self):
        with self.assertRaisesRegex(ValueError, "Prefix"):
            generate(self.manifest, ["coarse_a"], self.out, "../bad")
        self.assertFalse(self.out.exists())
        self.out.mkdir()
        existing = self.out / "user.txt"
        existing.write_text("user data", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            generate(self.manifest, ["coarse_a"], self.out, "v53")
        self.assertEqual(existing.read_text(), "user data")


if __name__ == "__main__":
    unittest.main()
