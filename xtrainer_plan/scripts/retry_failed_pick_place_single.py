#!/usr/bin/env python3
"""Retry exactly the failed items of one pick-and-place result with one arm.

The source result's task geometry and planning options are preserved.  The robot
model is changed to ``xtrainer.yml`` with no mimic-arm prefix.  Optionally,
``--grasp-max-deg`` changes only the upper bound of the grasp-angle search for a
second retry pass.  Subsets, a different Cartesian Home IK seed, and bypassing
the IK prescreen are optional; none relaxes trajectory acceptance checks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLANNER_PATH = REPO / "xtrainer_plan/scripts/plan_pick_place.py"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def select_failed_indices(source_meta, skipped_doc, requested_indices=None):
    """Validate the source failure list and select an ordered subset, without IO."""
    source_items = source_meta["items"]
    item_by_index = {int(item["index"]): item for item in source_items}
    if len(item_by_index) != len(source_items):
        raise RuntimeError("Source metadata contains duplicate item indices")
    skipped_indices = [int(item["index"]) for item in skipped_doc.get("skipped", [])]
    failed_indices = [int(item["index"]) for item in source_items if not item.get("success")]
    if skipped_indices != failed_indices or len(set(skipped_indices)) != len(skipped_indices):
        raise RuntimeError(
            "Source plan_skipped.json does not exactly match unique metadata failures"
        )
    if not failed_indices:
        raise RuntimeError("Source has no failed items to retry")
    selected = list(failed_indices if requested_indices is None else requested_indices)
    if not selected:
        raise RuntimeError("--indices must select at least one failed item")
    if len(set(selected)) != len(selected):
        raise RuntimeError("--indices contains duplicate indices")
    for index in selected:
        if index not in item_by_index:
            raise RuntimeError(f"--indices contains nonexistent source index: {index}")
        if index not in failed_indices:
            raise RuntimeError(f"--indices cannot retry a successful source item: {index}")
    return failed_indices, selected


def prepare_retry_config(
    source_meta, output_dir, *, grasp_max_deg=None,
    home_seed_deg=None, skip_ik_prescreen=False,
):
    """Build a fresh config and a complete search-change audit, without GPU or IO."""
    retry_config = copy.deepcopy(source_meta["config"])
    source_config = source_meta["config"]
    robot = retry_config["robot"]
    source_robot_yml = robot.get("robot_yml")
    source_dual_prefix = robot.get("dual_arm_prefix")
    robot.update({"robot_yml": "xtrainer.yml", "dual_arm_prefix": None})
    pick_place = retry_config["pick_place"]
    angle_search = pick_place["angle_search"]
    grasp_search = angle_search["grasp"]
    source_grasp = copy.deepcopy(grasp_search)
    if grasp_search.get("axis") != "x" or not angle_search.get("couple_place_to_grasp"):
        raise RuntimeError("Expected coupled grasp/place search about the grasp x axis")
    if grasp_max_deg is not None:
        if not math.isfinite(grasp_max_deg):
            raise RuntimeError("--grasp-max-deg must be finite")
        if grasp_max_deg < float(grasp_search["min_deg"]):
            raise RuntimeError("--grasp-max-deg is below grasp.min_deg")
        grasp_search["max_deg"] = float(grasp_max_deg)

    home = pick_place["home"]
    if home_seed_deg is not None:
        if home.get("joint_deg") is not None:
            raise RuntimeError("--home-seed-deg requires Cartesian Home; home.joint_deg is explicit")
        if len(home_seed_deg) != 6 or not all(math.isfinite(v) for v in home_seed_deg):
            raise RuntimeError("--home-seed-deg requires exactly six finite values")
        home["ik_seed_joint_deg"] = [float(v) for v in home_seed_deg]
        seed_mode = "explicit Cartesian Home IK seed; target pose unchanged"
    elif home.get("joint_deg") is None:
        home["ik_seed_joint_deg"] = copy.deepcopy(source_meta["home_joint_deg"])
        seed_mode = "source resolved Home joints used as Cartesian Home IK seed (default)"
    else:
        seed_mode = "explicit Home joints preserved; no IK seed override"

    source_criterion = copy.deepcopy(pick_place.get("criterion", {}))
    if skip_ik_prescreen:
        pick_place.setdefault("criterion", {})["prescreen_by_ik"] = False
    source_on_fail = copy.deepcopy(pick_place["on_fail"])
    pick_place["on_fail"]["mode"] = "skip"
    retry_config["output"].update({"dir": str(output_dir), "add_timestamp": False})
    audit = {
        "model_change": (
            f"{source_robot_yml} + prefix={source_dual_prefix!r} -> "
            "xtrainer.yml + prefix=None"
        ),
        "angle_change": (
            None if grasp_max_deg is None
            else f"grasp.max_deg {source_grasp['max_deg']} -> {grasp_max_deg:g}"
        ),
        "grasp_search": {"source": source_grasp, "effective": copy.deepcopy(grasp_search)},
        "home_seed": {
            "mode": seed_mode,
            "source_config_seed_deg": copy.deepcopy(
                source_config["pick_place"]["home"].get("ik_seed_joint_deg")
            ),
            "source_resolved_home_joint_deg": copy.deepcopy(source_meta.get("home_joint_deg")),
            "effective_seed_deg": copy.deepcopy(home.get("ik_seed_joint_deg")),
            "target_pose_unchanged": True,
        },
        "criterion": {
            "source": source_criterion,
            "effective": copy.deepcopy(pick_place.get("criterion", {})),
            "skip_ik_prescreen_requested": bool(skip_ik_prescreen),
            "trajectory_acceptance_checks_relaxed": False,
        },
        "on_fail": {"source": source_on_fail, "effective": copy.deepcopy(pick_place["on_fail"])},
    }
    return retry_config, audit


def build_planner_argv(no_incremental_save=False):
    return [str(PLANNER_PATH)] + (["--no-incremental-save"] if no_incremental_save else [])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--grasp-max-deg",
        type=float,
        default=None,
        help="Optional replacement for pick_place.angle_search.grasp.max_deg",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--indices", type=int, nargs="+",
        help="Ordered subset of failed source item indices (original zero-based indices)",
    )
    parser.add_argument(
        "--home-seed-deg", type=float, nargs=6,
        help="Six finite Cartesian Home IK seed angles in degrees; does not change its target pose",
    )
    parser.add_argument(
        "--skip-ik-prescreen", action="store_true",
        help="Skip only IK prescreen; preserve joint, linear-motion and collision acceptance checks",
    )
    parser.add_argument("--no-incremental-save", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not is_relative_to(source_dir, REPO):
        raise SystemExit(f"Source must be under repository: {source_dir}")
    if not is_relative_to(output_dir, REPO):
        raise SystemExit(f"Output must be under repository: {output_dir}")
    if source_dir == output_dir:
        raise SystemExit("Output directory must differ from source directory")
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output_dir}")

    source_meta_path = source_dir / "trajectory_meta.json"
    source_skipped_path = source_dir / "plan_skipped.json"
    if not source_meta_path.is_file() or not source_skipped_path.is_file():
        raise SystemExit(
            f"Source must contain trajectory_meta.json and plan_skipped.json: {source_dir}"
        )
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    skipped_doc = json.loads(source_skipped_path.read_text(encoding="utf-8"))
    skipped_items = skipped_doc.get("skipped", [])
    failed_indices, selected_indices = select_failed_indices(source_meta, skipped_doc, args.indices)
    retry_config, search_changes = prepare_retry_config(
        source_meta, output_dir, grasp_max_deg=args.grasp_max_deg,
        home_seed_deg=args.home_seed_deg, skip_ik_prescreen=args.skip_ik_prescreen,
    )
    item_by_index = {int(item["index"]): item for item in source_meta["items"]}
    model_change = search_changes["model_change"]
    angle_change = search_changes["angle_change"]
    planner_argv = build_planner_argv(args.no_incremental_save)
    retry_config["retry_scope"] = {
        "source_result_dir": str(source_dir.relative_to(REPO)),
        "source_trajectory_meta_sha256": sha256(source_meta_path),
        "source_plan_skipped_sha256": sha256(source_skipped_path),
        "selection": (
            "all entries in source plan_skipped.json" if args.indices is None
            else "explicit ordered subset of failed source indices (--indices)"
        ),
        "model_change": model_change,
        "angle_change": angle_change,
        "source_n_items_total": int(source_meta["n_items_total"]),
        "source_n_items_success": int(source_meta["n_items_success"]),
        "source_n_items_skipped": len(failed_indices),
        "source_failed_item_indices": failed_indices,
        "selected_item_indices": selected_indices,
        "selected_n_items": len(selected_indices),
        "search_strategy_changes": search_changes,
        "planner_argv": planner_argv,
        "no_incremental_save": bool(args.no_incremental_save),
    }

    pick_place = retry_config["pick_place"]
    angle_search = pick_place["angle_search"]
    grasp_search = angle_search["grasp"]
    transform = pick_place.get("link0_target_transform")
    print(
        f"[VALIDATE] source_total={source_meta['n_items_total']}, "
        f"source_success={source_meta['n_items_success']}, "
        f"source_failures={len(failed_indices)}, selected_failures={len(selected_indices)}"
    )
    print(f"[VALIDATE] model_change={model_change}")
    print(f"[VALIDATE] link0_target_transform={transform}")
    print(f"[VALIDATE] selected_indices={selected_indices}")
    print(f"[VALIDATE] home_seed={search_changes['home_seed']}")
    print(f"[VALIDATE] criterion={search_changes['criterion']}")
    print(f"[VALIDATE] planner_argv={planner_argv}")
    print(
        "[VALIDATE] grasp="
        f"{grasp_search['min_deg']}..{grasp_search['max_deg']} "
        f"step{grasp_search['step_deg']}, coupled, order={angle_search.get('order')}"
    )
    if args.validate_only:
        return 0

    output_dir.mkdir(parents=True)
    (output_dir / "retry_config.json").write_text(
        json.dumps(retry_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "source_failed_items.json").write_text(
        json.dumps(
            {
                "source_plan_skipped": str(source_skipped_path),
                "source_plan_skipped_sha256": sha256(source_skipped_path),
                "source_failed_item_indices": failed_indices,
                "selected_item_indices": selected_indices,
                "items": skipped_items,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log_file = (output_dir / "plan.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    spec = importlib.util.spec_from_file_location(
        "retry_failed_pick_place_single_impl", PLANNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import planner: {PLANNER_PATH}")
    planner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = planner
    spec.loader.exec_module(planner)
    original_build = planner.build_grasp_points

    def build_selected_points(grid_config, max_items=None):
        all_items = original_build(grid_config, None)
        by_index = {int(item["index"]): item for item in all_items}
        missing = [index for index in selected_indices if index not in by_index]
        if missing:
            raise RuntimeError(f"Failed indices absent from reconstructed grid: {missing}")
        selected = [by_index[index] for index in selected_indices]
        for selected_item in selected:
            index = int(selected_item["index"])
            source_item = item_by_index[index]
            expected = source_item.get("position_raw", source_item["position"])
            actual = selected_item["position"]
            if len(expected) != len(actual) or any(
                abs(float(a) - float(b)) > 1.0e-12
                for a, b in zip(actual, expected)
            ):
                raise RuntimeError(f"Source raw-position mismatch at index {index}")
        return selected if max_items is None else selected[: int(max_items)]

    planner.load_pick_place_config = lambda _path=None: copy.deepcopy(retry_config)
    planner.build_grasp_points = build_selected_points
    sys.argv = planner_argv

    print(f"[RETRY] source={source_dir}")
    print(f"[RETRY] output={output_dir}")
    print(f"[RETRY] selected_count={len(selected_indices)}")
    print(f"[RETRY] model_change={model_change}")
    if angle_change:
        print(f"[RETRY] angle_change={angle_change}")
    return int(planner.main())


if __name__ == "__main__":
    raise SystemExit(main())
