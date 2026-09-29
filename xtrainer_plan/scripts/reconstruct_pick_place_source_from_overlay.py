#!/usr/bin/env python3
"""Reconstruct a planning/plotting source after the original metadata was lost.

This is intentionally a derived, non-executable artifact.  It combines the
original item outcomes retained by a visualization overlay with the exact
original-failure scope and full task configuration retained by the old retry.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_vector(left, right, tolerance: float = 1.0e-9) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overlay_dir", type=Path)
    parser.add_argument("old_retry_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    overlay_dir = args.overlay_dir.resolve()
    retry_dir = args.old_retry_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output_dir}")

    overlay_path = overlay_dir / "trajectory_meta.json"
    retry_path = retry_dir / "trajectory_meta.json"
    overlay = load(overlay_path)
    retry = load(retry_path)
    if overlay.get("artifact_kind") != "visualization_only":
        raise RuntimeError("Expected a visualization-only overlay")
    if overlay.get("executable_trajectory") is not False:
        raise RuntimeError("Overlay is not explicitly marked non-executable")

    original_run = next(
        entry for entry in overlay["source_runs"] if entry.get("id") == "original"
    )
    original_pass = next(
        entry for entry in overlay["search_passes"] if entry.get("id") == "original"
    )
    scope = retry["config"]["retry_scope"]
    failed_indices = [int(index) for index in scope["selected_item_indices"]]
    retry_indices = [int(item["index"]) for item in retry["items"]]
    if retry_indices != failed_indices or len(set(failed_indices)) != len(failed_indices):
        raise RuntimeError("Old retry items do not exactly match its recorded source scope")
    if len(failed_indices) != int(original_run["n_items_skipped"]):
        raise RuntimeError("Original failure count disagrees with the retry scope")

    overlay_items = overlay["items"]
    overlay_by_index = {int(item["index"]): item for item in overlay_items}
    retry_by_index = {int(item["index"]): item for item in retry["items"]}
    if len(overlay_by_index) != int(original_run["n_items_total"]):
        raise RuntimeError("Overlay does not contain one unique item per original grid point")
    if set(failed_indices) != {
        int(item["index"]) for item in overlay_items if item.get("retry_attempted")
    }:
        raise RuntimeError("Overlay retry-attempted items disagree with the old retry scope")

    reconstructed_items = []
    for overlay_item in overlay_items:
        index = int(overlay_item["index"])
        was_original_failure = index in retry_by_index
        if was_original_failure:
            retry_item = retry_by_index[index]
            for key in ("position", "position_raw", "effective_position"):
                if not same_vector(overlay_item[key], retry_item[key]):
                    raise RuntimeError(f"Geometry mismatch for index={index}, field={key}")
        elif overlay_item.get("solution_source") != "original":
            raise RuntimeError(f"Non-retried index={index} is not marked original success")

        attempts = overlay_item.get("attempts_by_pass") or {}
        item = {
            "index": index,
            "success": not was_original_failure,
            "row": int(overlay_item["row"]),
            "col": int(overlay_item["col"]),
            "position": copy.deepcopy(overlay_item["position"]),
            "position_raw": copy.deepcopy(overlay_item.get("position_raw")),
            "effective_position": copy.deepcopy(overlay_item.get("effective_position")),
            "n_angles_tried": int(attempts.get("original", 0)),
        }
        if item["success"]:
            item["angle_grasp_deg"] = float(overlay_item["angle_grasp_deg"])
            item["angle_place_deg"] = float(overlay_item["angle_place_deg"])
        reconstructed_items.append(item)

    n_success = sum(bool(item["success"]) for item in reconstructed_items)
    if n_success != int(original_run["n_items_success"]):
        raise RuntimeError("Reconstructed original-success count is inconsistent")

    config = copy.deepcopy(retry["config"])
    config.pop("retry_scope", None)
    pick_place = config["pick_place"]
    pick_place["home"]["joint_deg"] = None
    pick_place["home"].pop("ik_seed_joint_deg", None)
    pick_place["angle_search"]["grasp"] = copy.deepcopy(original_pass["grasp"])
    pick_place["angle_search"]["reuse_last_success"] = bool(
        original_pass["reuse_last_success"]
    )
    config["output"].update({"dir": str(output_dir), "add_timestamp": False})

    angle_search = copy.deepcopy(retry["angle_search"])
    angle_search["grasp"] = copy.deepcopy(original_pass["grasp"])
    angle_search["reuse_last_success"] = bool(original_pass["reuse_last_success"])
    grid = copy.deepcopy(retry["grid"])
    grid["n_items"] = int(original_run["n_items_total"])
    reconstructed = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task_type": "pick_place_cycle",
        "artifact_kind": "reconstructed_planning_and_visualization_source",
        "executable_trajectory": False,
        "trajectory_available": False,
        "warning": (
            "Derived metadata only: original trajectory files were missing. "
            "Use for failed-item selection and plotting, not execution."
        ),
        "reconstruction_provenance": {
            "original_result_dir": original_run["result_dir"],
            "original_trajectory_meta_sha256": original_run[
                "trajectory_meta_sha256"
            ],
            "original_plan_skipped_sha256": scope[
                "source_plan_skipped_sha256"
            ],
            "overlay_meta": str(overlay_path),
            "overlay_meta_sha256": sha256(overlay_path),
            "old_retry_meta": str(retry_path),
            "old_retry_meta_sha256": sha256(retry_path),
            "original_success_source": "overlay items with solution_source=original",
            "original_failure_source": "old retry selected_item_indices",
            "old_retry_outcomes_included": False,
        },
        "robot": copy.deepcopy(retry["robot"]),
        "rotation_convention": retry.get("rotation_convention"),
        "quaternion_order": copy.deepcopy(retry.get("quaternion_order")),
        "home_joint_deg": copy.deepcopy(retry["home_joint_deg"]),
        "grid": grid,
        "place_position": copy.deepcopy(retry["place_position"]),
        "place_position_raw": copy.deepcopy(retry.get("place_position_raw")),
        "angle_search": angle_search,
        "linear_move": copy.deepcopy(retry.get("linear_move")),
        "criterion": copy.deepcopy(retry.get("criterion")),
        "n_items_total": int(original_run["n_items_total"]),
        "n_items_success": n_success,
        "n_items_skipped": len(failed_indices),
        "items": reconstructed_items,
        "config": config,
    }

    reconstructed_by_index = {
        int(item["index"]): item for item in reconstructed_items
    }
    skipped = []
    for index in failed_indices:
        item = reconstructed_by_index[index]
        skipped.append(
            {
                "index": index,
                "row": item["row"],
                "col": item["col"],
                "position": copy.deepcopy(item["position"]),
                "position_raw": copy.deepcopy(item["position_raw"]),
                "effective_position": copy.deepcopy(item["effective_position"]),
                "n_combos_tried": item["n_angles_tried"],
            }
        )

    output_dir.mkdir(parents=True)
    (output_dir / "trajectory_meta.json").write_text(
        json.dumps(reconstructed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "plan_skipped.json").write_text(
        json.dumps({"skipped": skipped}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] reconstructed source: {output_dir}")
    print(
        f"[summary] original outcomes restored for planning/plotting: "
        f"{n_success} success + {len(failed_indices)} failure = "
        f"{len(reconstructed_items)}; old retry outcomes excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
