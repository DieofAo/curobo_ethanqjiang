#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the single-arm overhead RViz model from a planning config/result.

The planning model deliberately keeps ``LINK_0`` as the CuRobo root.  This
RViz-only model adds the physical mounting transform above it::

    task_world --M--> LINK_0 ... TCP_LINK
         |
         +--identity--> original_LINK_0

``M`` is ``robot.mount_transform`` from the resolved pick/place config and is
the pose of the real robot base in the unchanged task coordinate system.  The
planner must use ``pick_place.link0_target_transform == inv(M)``.  This script
checks that invariant rather than drawing a model which disagrees with the
planned trajectory.

Besides the arm, the generated URDF contains a translucent grasp plane,
a place pad, a simple ceiling plate/beam and the original CAD workbench.
The workbench stays in the unchanged task frame; no second arm or assembly
collision geometry is added.  ``--no-assembly`` restores the simplified view.
The display-only mounts remove the old CAD base's extra Z yaw and Y shift; explicitly pass
the former mounts file with ``--assembly-mounts`` to reproduce its tilted view.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml


TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from build_scene_urdf import mat_to_pose, pose_to_mat  # noqa: E402
from plan_pick_place import load_pick_place_config  # noqa: E402
from xtrainer_common import parse_rigid_transform_matrix, resolve_repo_path  # noqa: E402


URDF_MESH_PREFIX = "package://robotics/drivers/dobot/description/meshes"
DEFAULT_ASSEMBLY_MOUNTS = TASK_ROOT / "config/cad_mounts_overhead_display.yaml"
DEFAULT_ASSEMBLY_MESH = (REPO_ROOT / "src/curobo/content/assets/robot/"
                         "ur_description/meshes/xtrainer/table.stl")


def _assembly_path(value: str | Path, description: str) -> Path:
    raw = str(value)
    if raw.startswith("file://"):
        raw = raw[len("file://"):]
    path = Path(raw).expanduser()
    if not path.is_absolute():
        for candidate in (Path.cwd() / path, TASK_ROOT / path, REPO_ROOT / path):
            if candidate.is_file():
                path = candidate
                break
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"assembly {description} not found: {path}")
    return path


def assembly_settings(
    *, enabled: bool = True, mounts_path: str | Path | None = None,
    arm: str = "left", mesh_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Resolve the old CAD scene in the old LINK_0/task frame, not the new base.

    The display-only default restores the original left CAD mount, removing its
    extra Z yaw and Y shift.  Pass the old mounts file explicitly to reproduce
    those offsets.  These settings do not alter the planned robot mount;
    they are JSON serializable and can also drive a live marker.
    """
    if arm not in ("left", "right"):
        raise ValueError("assembly arm must be left or right")
    if not enabled:
        return {"enabled": False, "visual_only": True}
    mounts = _assembly_path(
        DEFAULT_ASSEMBLY_MOUNTS if mounts_path is None else mounts_path, "mounts"
    )
    mesh = _assembly_path(
        DEFAULT_ASSEMBLY_MESH if mesh_path is None else mesh_path, "mesh"
    )
    with mounts.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    mount = document.get(f"{arm}_arm") if isinstance(document, dict) else None
    if not isinstance(mount, dict):
        raise ValueError(f"assembly mounts {mounts} must contain {arm}_arm")
    xyz = np.asarray(mount.get("xyz"), dtype=np.float64)
    rpy = np.asarray(mount.get("rpy"), dtype=np.float64)
    if (xyz.shape != (3,) or rpy.shape != (3,) or
            not np.all(np.isfinite(np.r_[xyz, rpy]))):
        raise ValueError(f"assembly {arm}_arm xyz/rpy must each contain 3 finite values")
    cad_mount = pose_to_mat(xyz, rpy)
    transform = np.linalg.inv(cad_mount)
    inverse_xyz, inverse_rpy = mat_to_pose(transform)
    return {
        "enabled": True, "visual_only": True,
        "mounts": str(mounts), "arm": arm,
        "mesh": str(mesh), "mesh_uri": mesh.as_uri(),
        "scale": 0.001, "rgba": [0.48, 0.52, 0.57, 0.82],
        "original_base_in_cad": cad_mount.tolist(),
        "task_world_to_scene_root": transform.tolist(),
        "xyz": inverse_xyz, "rpy": inverse_rpy,
    }


def _result_record_path(value: str) -> Path:
    """Resolve a result dir/NPZ/JSON to the JSON carrying resolved config."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        for candidate in (Path.cwd() / path, TASK_ROOT / path, REPO_ROOT / path):
            if candidate.exists():
                path = candidate
                break
    path = path.resolve()
    if path.is_dir():
        for name in ("trajectory_meta.json", "plan_failed.json"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"result directory has neither trajectory_meta.json nor plan_failed.json: {path}"
        )
    if path.suffix == ".npz":
        candidate = path.parent / "trajectory_meta.json"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"trajectory metadata not found next to {path}")
    if not path.is_file():
        raise FileNotFoundError(f"result/config source not found: {path}")
    return path


def load_resolved_config(
    config_path: str | None,
    traj_path: str | None,
) -> Tuple[Dict[str, Any], str]:
    if (config_path is None) == (traj_path is None):
        raise ValueError("exactly one of --config and --traj is required")
    if config_path is not None:
        cfg = load_pick_place_config(config_path)
        return cfg, str(Path(config_path))

    record_path = _result_record_path(str(traj_path))
    with record_path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    cfg = record.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"{record_path} does not contain resolved config")
    return copy.deepcopy(cfg), str(record_path)


def _require_string(mapping: Dict[str, Any], key: str, scope: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{scope}.{key} must be a non-empty string")
    return value.strip()


def scene_settings(
    cfg: Dict[str, Any], source: str, *, include_assembly: bool = True,
    assembly_mounts: str | Path | None = None, assembly_arm: str = "left",
    assembly_mesh: str | Path | None = None,
) -> Dict[str, Any]:
    robot_cfg = cfg.get("robot")
    pp_cfg = cfg.get("pick_place")
    if not isinstance(robot_cfg, dict) or not isinstance(pp_cfg, dict):
        raise ValueError(f"{source}: config must contain robot and pick_place mappings")

    base_link = _require_string(robot_cfg, "base_link", "robot")
    task_frame = _require_string(robot_cfg, "task_frame", "robot")
    legacy_frame = _require_string(robot_cfg, "legacy_task_frame", "robot")
    if len({base_link, task_frame, legacy_frame}) != 3:
        raise ValueError("robot base/task/legacy frame names must be distinct")

    mount = parse_rigid_transform_matrix(
        robot_cfg.get("mount_transform"), "robot.mount_transform"
    )
    correction = parse_rigid_transform_matrix(
        pp_cfg.get("link0_target_transform"),
        "pick_place.link0_target_transform",
    )
    expected = np.linalg.inv(mount)
    inverse_error = float(np.max(np.abs(correction - expected)))
    if inverse_error > 1.0e-8:
        raise ValueError(
            "planning/RViz frame mismatch: pick_place.link0_target_transform "
            f"must equal inv(robot.mount_transform), max error={inverse_error:.9g}"
        )

    urdf_raw = _require_string(robot_cfg, "urdf", "robot")
    urdf_path = resolve_repo_path(urdf_raw).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"robot URDF not found: {urdf_path}")

    grid = pp_cfg.get("grasp_grid")
    place = pp_cfg.get("place")
    if not isinstance(grid, dict) or not isinstance(place, dict):
        raise ValueError("pick_place must contain grasp_grid and place mappings")
    x_range = np.asarray(grid.get("x_range"), dtype=np.float64)
    y_range = np.asarray(grid.get("y_range"), dtype=np.float64)
    place_position = np.asarray(place.get("position"), dtype=np.float64)
    if x_range.shape != (2,) or y_range.shape != (2,):
        raise ValueError("grasp_grid x_range/y_range must each contain two values")
    if place_position.shape != (3,):
        raise ValueError("pick_place.place.position must contain three values")
    numeric = np.concatenate((x_range, y_range, place_position,
                              np.asarray([grid.get("z")], dtype=np.float64)))
    if not np.all(np.isfinite(numeric)):
        raise ValueError("grasp grid/place values must be finite")

    return {
        "source": source,
        "base_link": base_link,
        "task_frame": task_frame,
        "legacy_task_frame": legacy_frame,
        "mount_transform": mount,
        "link0_target_transform": correction,
        "inverse_error": inverse_error,
        "urdf": urdf_path,
        "x_range": np.sort(x_range),
        "y_range": np.sort(y_range),
        "grasp_z": float(grid["z"]),
        "place_position": place_position,
        "task_workspace": copy.deepcopy((cfg.get("overhead") or {}).get(
            "task_workspace"
        )),
        "assembly": assembly_settings(
            enabled=include_assembly, mounts_path=assembly_mounts,
            arm=assembly_arm, mesh_path=assembly_mesh,
        ),
    }


def _fmt(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _add_fixed_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    transform: np.ndarray,
) -> None:
    xyz, rpy = mat_to_pose(transform)
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": _fmt(xyz), "rpy": _fmt(rpy)})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})


def _add_material(robot: ET.Element, name: str, rgba) -> None:
    material = ET.SubElement(robot, "material", {"name": name})
    ET.SubElement(material, "color", {"rgba": _fmt(rgba)})


def _add_visual_link(
    robot: ET.Element,
    *,
    name: str,
    parent: str,
    xyz,
    geometry: str,
    dimensions,
    material: str,
) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    shape = ET.SubElement(visual, "geometry")
    if geometry == "box":
        ET.SubElement(shape, "box", {"size": _fmt(dimensions)})
    elif geometry == "cylinder":
        radius, length = dimensions
        ET.SubElement(shape, "cylinder", {
            "radius": f"{float(radius):.9g}",
            "length": f"{float(length):.9g}",
        })
    else:  # pragma: no cover - internal programming error
        raise ValueError(f"unsupported visual geometry: {geometry}")
    ET.SubElement(visual, "material", {"name": material})

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    _add_fixed_joint(robot, f"{name}_fixed", parent, name, transform)


def build_scene(settings: Dict[str, Any]) -> Tuple[ET.Element, Dict[str, Any]]:
    urdf_path = settings["urdf"]
    text = urdf_path.read_text(encoding="utf-8")
    mesh_dir = (urdf_path.parent / "meshes").resolve()
    text = text.replace(URDF_MESH_PREFIX, f"file://{mesh_dir}")
    robot = ET.fromstring(text)

    children = {
        joint.find("child").get("link")
        for joint in robot.findall("joint")
        if joint.find("child") is not None
    }
    roots = [
        link.get("name") for link in robot.findall("link")
        if link.get("name") not in children
    ]
    if roots != [settings["base_link"]]:
        raise ValueError(
            f"URDF root must be {settings['base_link']!r}, found {roots}"
        )
    all_names = {
        element.get("name") for tag in ("link", "joint", "material")
        for element in robot.findall(tag)
    }
    generated_names = {
        settings["task_frame"], settings["legacy_task_frame"],
        "overhead_robot_mount", "overhead_original_link0_fixed",
        "overhead_grasp_plane", "overhead_grasp_plane_fixed",
        "overhead_place_pad", "overhead_place_pad_fixed",
        "overhead_mount_plate", "overhead_mount_plate_fixed",
        "overhead_ceiling_beam", "overhead_ceiling_beam_fixed",
        "overhead_hanger", "overhead_hanger_fixed",
    }
    assembly = settings.get("assembly", {"enabled": False, "visual_only": True})
    if assembly["enabled"]:
        generated_names.update({"scene_root", "scene_root_fixed", "workbench",
                                "workbench_fixed", "table_gray"})
    collision = sorted(str(name) for name in generated_names.intersection(all_names))
    if collision:
        raise ValueError(f"generated URDF names already exist: {collision}")

    robot.set("name", "xtrainer_single_overhead_rviz")
    ET.SubElement(robot, "link", {"name": settings["task_frame"]})
    _add_fixed_joint(
        robot, "overhead_robot_mount", settings["task_frame"],
        settings["base_link"], settings["mount_transform"],
    )
    ET.SubElement(robot, "link", {"name": settings["legacy_task_frame"]})
    _add_fixed_joint(
        robot, "overhead_original_link0_fixed", settings["task_frame"],
        settings["legacy_task_frame"], np.eye(4, dtype=np.float64),
    )

    if assembly["enabled"]:
        _add_material(robot, "table_gray", assembly["rgba"])
        ET.SubElement(robot, "link", {"name": "scene_root"})
        _add_fixed_joint(
            robot, "scene_root_fixed", settings["task_frame"], "scene_root",
            np.asarray(assembly["task_world_to_scene_root"], dtype=np.float64),
        )
        link = ET.SubElement(robot, "link", {"name": "workbench"})
        visual = ET.SubElement(link, "visual")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "mesh", {
            "filename": assembly["mesh_uri"],
            "scale": _fmt([assembly["scale"]] * 3),
        })
        ET.SubElement(visual, "material", {"name": "table_gray"})
        _add_fixed_joint(robot, "workbench_fixed", "scene_root", "workbench",
                         np.eye(4, dtype=np.float64))

    _add_material(robot, "overhead_task_surface", (0.12, 0.55, 0.88, 0.28))
    _add_material(robot, "overhead_place_surface", (0.95, 0.30, 0.55, 0.72))
    _add_material(robot, "overhead_support_dark", (0.16, 0.18, 0.22, 0.92))

    x_range = settings["x_range"]
    y_range = settings["y_range"]
    grid_center = np.array([
        float(np.mean(x_range)), float(np.mean(y_range)), settings["grasp_z"] - 0.006,
    ])
    grid_size = np.array([
        max(float(np.ptp(x_range)) + 0.04, 0.10),
        max(float(np.ptp(y_range)) + 0.04, 0.10),
        0.010,
    ])
    _add_visual_link(
        robot, name="overhead_grasp_plane", parent=settings["task_frame"],
        xyz=grid_center, geometry="box", dimensions=grid_size,
        material="overhead_task_surface",
    )

    place = settings["place_position"]
    _add_visual_link(
        robot, name="overhead_place_pad", parent=settings["task_frame"],
        xyz=(place[0], place[1], place[2] - 0.006), geometry="cylinder",
        dimensions=(0.035, 0.010), material="overhead_place_surface",
    )

    mount_xyz = settings["mount_transform"][:3, 3]
    _add_visual_link(
        robot, name="overhead_mount_plate", parent=settings["task_frame"],
        xyz=(mount_xyz[0], mount_xyz[1], mount_xyz[2] + 0.025),
        geometry="box", dimensions=(0.24, 0.18, 0.05),
        material="overhead_support_dark",
    )
    _add_visual_link(
        robot, name="overhead_hanger", parent=settings["task_frame"],
        xyz=(mount_xyz[0], mount_xyz[1], mount_xyz[2] + 0.09),
        geometry="box", dimensions=(0.08, 0.08, 0.10),
        material="overhead_support_dark",
    )
    beam_x = max(float(np.ptp(x_range)) + 0.30, 0.80)
    _add_visual_link(
        robot, name="overhead_ceiling_beam", parent=settings["task_frame"],
        xyz=(mount_xyz[0], mount_xyz[1], mount_xyz[2] + 0.16),
        geometry="box", dimensions=(beam_x, 0.10, 0.10),
        material="overhead_support_dark",
    )

    tcp_offset = None
    for joint in robot.findall("joint"):
        if joint.get("name") == "TCP_joint":
            origin = joint.find("origin")
            if origin is not None:
                tcp_offset = [float(value) for value in origin.get("xyz", "0 0 0").split()]
            break
    summary = {
        "source": settings["source"],
        "urdf": str(urdf_path),
        "base_link": settings["base_link"],
        "task_frame": settings["task_frame"],
        "legacy_task_frame": settings["legacy_task_frame"],
        "mount_transform": settings["mount_transform"].tolist(),
        "link0_target_transform": settings["link0_target_transform"].tolist(),
        "inverse_max_abs_error": settings["inverse_error"],
        "grasp_grid_center_task_world": grid_center.tolist(),
        "grasp_grid_size": grid_size.tolist(),
        "place_position_task_world": place.tolist(),
        "tcp_joint_xyz": tcp_offset,
        "assembly": copy.deepcopy(assembly),
    }
    return robot, summary


def _serialize(robot: ET.Element) -> str:
    return '<?xml version="1.0"?>\n' + ET.tostring(
        robot, encoding="unicode"
    ) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="pick/place config (partial overlay is accepted)")
    source.add_argument(
        "--traj",
        help="result directory, trajectory.npz, trajectory_meta.json or plan_failed.json",
    )
    parser.add_argument("--output", default="-", help="URDF output; '-' writes stdout")
    parser.add_argument("--no-assembly", action="store_true",
                        help="omit the original CAD workbench (simplified view)")
    parser.add_argument("--assembly-mounts",
                        help="CAD display mounts YAML (default: extra old-base Z yaw and Y shift removed)")
    parser.add_argument("--assembly-arm", choices=("left", "right"), default="left",
                        help="which original CAD base defines the old task frame")
    parser.add_argument("--assembly-mesh", help="filtered workbench STL path or file URI")
    parser.add_argument(
        "--print-settings", action="store_true",
        help="validate and print resolved scene settings as JSON instead of URDF",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    try:
        cfg, source = load_resolved_config(args.config, args.traj)
        settings = scene_settings(
            cfg, source, include_assembly=not args.no_assembly,
            assembly_mounts=args.assembly_mounts, assembly_arm=args.assembly_arm,
            assembly_mesh=args.assembly_mesh,
        )
        robot, summary = build_scene(settings)
        if args.print_settings:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        payload = _serialize(robot)
        if args.output == "-":
            sys.stdout.write(payload)
        else:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            print(
                f"[overhead scene] {output}  "
                f"{summary['task_frame']} -> {summary['base_link']}",
                file=sys.stderr,
            )
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError, ET.ParseError,
            yaml.YAMLError) as exc:
        print(f"[FAIL] overhead RViz scene: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
