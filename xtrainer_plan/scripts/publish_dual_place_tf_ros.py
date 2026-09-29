#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无轨迹双臂 RViz 预览的 place TF 发布器。

配置加载顺序与双臂规划入口一致：
``pick_place_default.yaml`` -> ``dual_pick_place_default.yaml`` -> 自定义差异配置。
不导入 CuRobo，也不执行 IK 或轨迹规划。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtrainer_common import (  # noqa: E402
    deep_update,
    matrix_to_quat_wxyz,
    normalize_link0_target_transform_config_layer,
    parse_rigid_transform_matrix,
    resolve_repo_path,
)


TASK_ROOT = Path(__file__).resolve().parents[1]
PICK_PLACE_DEFAULT = TASK_ROOT / "config" / "pick_place_default.yaml"
DUAL_DEFAULT = TASK_ROOT / "config" / "dual_pick_place_default.yaml"


def _resolve_config_path(path: Optional[str]) -> Path:
    if path is None:
        return DUAL_DEFAULT
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for resolved in (
        Path.cwd() / candidate,
        TASK_ROOT / candidate,
        TASK_ROOT / "config" / candidate,
    ):
        if resolved.exists():
            return resolved
    return candidate


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 3 个数字") from exc
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} 必须是 3 个有限数字，得到 {value!r}")
    return out


def load_preview_settings(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载与双臂规划相同的层叠配置并解析静态预览目标及轴角修正。"""
    with open(PICK_PLACE_DEFAULT, "r", encoding="utf-8") as stream:
        cfg = normalize_link0_target_transform_config_layer(
            yaml.safe_load(stream) or {}, str(PICK_PLACE_DEFAULT)
        )
    with open(DUAL_DEFAULT, "r", encoding="utf-8") as stream:
        dual_layer = normalize_link0_target_transform_config_layer(
            yaml.safe_load(stream) or {}, str(DUAL_DEFAULT)
        )
        cfg = deep_update(cfg, dual_layer)

    selected = _resolve_config_path(config_path)
    if not selected.exists():
        raise FileNotFoundError(f"config not found: {selected}")
    if selected.resolve() != DUAL_DEFAULT.resolve():
        with open(selected, "r", encoding="utf-8") as stream:
            selected_layer = normalize_link0_target_transform_config_layer(
                yaml.safe_load(stream) or {}, str(selected)
            )
            cfg = deep_update(cfg, selected_layer)

    pick_place_cfg = cfg["pick_place"]
    place1 = _vector3(
        pick_place_cfg["place"]["position"],
        "pick_place.place.position",
    )
    dual_cfg = cfg.get("dual_arm") or {}
    explicit_place2 = dual_cfg.get("second_place_position")
    if explicit_place2 is None:
        place2 = place1.copy()
        place2[0] = float(dual_cfg.get("second_place_x", -0.47))
    else:
        place2 = _vector3(
            explicit_place2, "dual_arm.second_place_position"
        )
    place2 = _vector3(place2, "arm2 place")

    link0_target_config = pick_place_cfg.get(
        "link0_target_transform",
        {
            "position": [0.0, 0.0, 0.0],
            "rotation": {"axis": [0.0, 0.0, 1.0], "angle_deg": 0.0},
        },
    )
    link0_target_transform = parse_rigid_transform_matrix(
        link0_target_config,
        "pick_place.link0_target_transform",
    )
    rotation = link0_target_transform[:3, :3]
    translation = link0_target_transform[:3, 3]
    place1_effective = rotation @ place1 + translation
    place2_effective = rotation @ place2 + translation
    effective_quat_wxyz = matrix_to_quat_wxyz(rotation).tolist()

    robot_cfg = cfg.get("robot") or {}
    mounts_raw = str(robot_cfg.get("mounts") or "")
    if not mounts_raw:
        raise ValueError("robot.mounts 不能为空")
    mounts = resolve_repo_path(mounts_raw).resolve()
    if not mounts.is_file():
        raise FileNotFoundError(f"mounts not found: {mounts}")

    return {
        "config": str(selected.resolve()),
        "base_frame": str(robot_cfg.get("base_link") or "LINK_0"),
        "mounts": str(mounts),
        "link0_target_transform_config": link0_target_config,
        "link0_target_transform": link0_target_transform.tolist(),
        "arm1_place_raw": place1.tolist(),
        "arm2_place_raw": place2.tolist(),
        "arm1_place": place1_effective.tolist(),
        "arm2_place": place2_effective.tolist(),
        "arm1_quat_wxyz": effective_quat_wxyz,
        "arm2_quat_wxyz": effective_quat_wxyz.copy(),
    }


def _make_tf(parent: str, child: str, position, quat_wxyz, stamp):
    from geometry_msgs.msg import TransformStamped  # noqa: PLC0415

    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = parent
    transform.child_frame_id = child
    transform.transform.translation.x = float(position[0])
    transform.transform.translation.y = float(position[1])
    transform.transform.translation.z = float(position[2])
    transform.transform.rotation.w = float(quat_wxyz[0])
    transform.transform.rotation.x = float(quat_wxyz[1])
    transform.transform.rotation.y = float(quat_wxyz[2])
    transform.transform.rotation.z = float(quat_wxyz[3])
    return transform


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 dual pick/place YAML 发布两臂 place TF"
    )
    parser.add_argument("--config", default=str(DUAL_DEFAULT))
    parser.add_argument(
        "--print-settings", action="store_true",
        help="只输出解析后的 JSON，不启动 ROS",
    )
    args, unknown = parser.parse_known_args()
    ros_injected = [
        value for value in unknown
        if value.startswith("__") or "robot_state_publisher" in value
    ]
    bad = [value for value in unknown if value not in ros_injected]
    if bad:
        parser.error(f"unrecognized arguments: {' '.join(bad)}")

    settings = load_preview_settings(args.config)
    if args.print_settings:
        print(json.dumps(settings, ensure_ascii=False))
        return 0

    import rospy  # noqa: PLC0415
    import tf2_ros  # noqa: PLC0415

    rospy.init_node("xtrainer_dual_place_tf", anonymous=False)
    broadcaster = tf2_ros.TransformBroadcaster()
    rospy.loginfo("[dual preview] config=%s", settings["config"])
    rospy.loginfo("[dual preview] mounts=%s", settings["mounts"])
    rospy.loginfo(
        "[dual preview] %s raw places: arm1=%s, arm2=%s",
        settings["base_frame"],
        settings["arm1_place_raw"], settings["arm2_place_raw"],
    )
    rospy.loginfo(
        "[dual preview] LINK_0 transform config=%s; C=%s; "
        "effective places: arm1=%s, arm2=%s",
        settings["link0_target_transform_config"],
        settings["link0_target_transform"],
        settings["arm1_place"], settings["arm2_place"],
    )
    rate = rospy.Rate(30.0)
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        broadcaster.sendTransform([
            _make_tf(
                settings["base_frame"], "xtrainer_arm1_place",
                settings["arm1_place"], settings["arm1_quat_wxyz"], now,
            ),
            _make_tf(
                settings["base_frame"], "xtrainer_arm2_place",
                settings["arm2_place"], settings["arm2_quat_wxyz"], now,
            ),
        ])
        rate.sleep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
