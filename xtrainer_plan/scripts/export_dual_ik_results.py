#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把双臂规划失败记录中的离散 Home IK 根导出为 RViz 可播放数据。

这个脚本只依赖 Python 和 NumPy，不导入 CuRobo。输出沿用现有轨迹播放器
所需的数值字段，但每一帧都是彼此独立的 IK 候选，不是连续轨迹，也绝不
能作为机器人执行输入。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


IK_RESULTS_NPZ = "ik_results.npz"
# play_trajectory_ros.load_trajectory() 当前固定读取这个 companion 文件名。
PLAYBACK_META_JSON = "trajectory_meta.json"
DEFAULT_FRAME_SECONDS = 2.0
EXPECTED_ARM_DOF = 6
EXPECTED_DOF = 2 * EXPECTED_ARM_DOF
IK_ARTIFACT_TYPES = {
    "dual_arm_discrete_ik_candidates",
    "dual_arm_ik_prescreen_trace",
}


class IkExportError(ValueError):
    """输入不是可安全解释的 12-DOF Home IK 失败记录。"""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IkExportError(f"{field} 必须是 JSON object")
    return value


def _finite_vector(value: Any, size: int, field: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise IkExportError(f"{field} 必须是 {size} 个有限数字")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise IkExportError(f"{field} 必须是 {size} 个有限数字") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise IkExportError(
            f"{field} 必须是 shape=({size},) 的有限数字，实际 {array.shape}"
        )
    return array


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise IkExportError(f"{field} 必须是整数")
    result = int(value)
    if result < minimum:
        raise IkExportError(f"{field} 必须 >= {minimum}")
    return result


def _finite_number(value: Any, field: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool):
        raise IkExportError(f"{field} 必须是有限数字")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IkExportError(f"{field} 必须是有限数字") from exc
    if not math.isfinite(result):
        raise IkExportError(f"{field} 必须是有限数字")
    if minimum is not None and result < minimum:
        raise IkExportError(f"{field} 必须 >= {minimum}")
    return result


def _pose(root: Mapping[str, Any], arm_key: str, context: str) -> Tuple[np.ndarray, np.ndarray]:
    pose = _mapping(root.get(arm_key), f"{context}.{arm_key}")
    position = _finite_vector(
        pose.get("position"), 3, f"{context}.{arm_key}.position"
    )
    quaternion = _finite_vector(
        pose.get("quat_wxyz"), 4, f"{context}.{arm_key}.quat_wxyz"
    )
    quat_norm = float(np.linalg.norm(quaternion))
    if abs(quat_norm - 1.0) > 1.0e-3:
        raise IkExportError(
            f"{context}.{arm_key}.quat_wxyz 不是单位四元数，norm={quat_norm:.8g}"
        )
    return position, quaternion


def load_failure_record(plan_failed: Path) -> Dict[str, Any]:
    """读取并初步验证 plan_failed.json；不接受未完成 marker。"""
    source = Path(plan_failed).expanduser()
    if not source.is_file():
        raise IkExportError(f"找不到 plan_failed.json: {source}")
    try:
        with source.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise IkExportError(f"无法读取有效 JSON: {source}: {exc}") from exc
    record = dict(_mapping(record, "plan_failed"))
    stage = str(record.get("stage") or "")
    if not stage:
        raise IkExportError("plan_failed.stage 不能为空")
    if stage == "planning_in_progress":
        raise IkExportError("该文件只是 planning_in_progress marker，不是完整失败记录")
    return record


def extract_home_root_frames(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """严格提取、校验并按 home_root_index 排序所有联合 Home IK 根。"""
    attempts = record.get("pipeline_attempts")
    if not isinstance(attempts, list) or not attempts:
        raise IkExportError("plan_failed.pipeline_attempts 必须是非空数组")

    frames: List[Dict[str, Any]] = []
    seen_root_indices = set()
    for list_index, raw_attempt in enumerate(attempts):
        attempt = _mapping(raw_attempt, f"pipeline_attempts[{list_index}]")
        raw_root = attempt.get("home_root")
        if raw_root is None:
            continue
        root = _mapping(raw_root, f"pipeline_attempts[{list_index}].home_root")
        context = f"pipeline_attempts[{list_index}].home_root"
        root_index = _integer(
            attempt.get("home_root_index"),
            f"pipeline_attempts[{list_index}].home_root_index",
        )
        if root_index in seen_root_indices:
            raise IkExportError(f"重复的 home_root_index: {root_index}")
        seen_root_indices.add(root_index)

        attempt_number = _integer(
            attempt.get("attempt"), f"pipeline_attempts[{list_index}].attempt", 1
        )
        angle_pair_trial = _integer(
            root.get("angle_pair_trial"), f"{context}.angle_pair_trial", 1
        )
        branch_index = _integer(
            root.get("ik_branch_index"), f"{context}.ik_branch_index"
        )
        distance = _finite_number(
            root.get("distance_to_seed"), f"{context}.distance_to_seed", 0.0
        )
        joint_deg = _finite_vector(
            root.get("home_joint_deg"), EXPECTED_DOF, f"{context}.home_joint_deg"
        )
        arm1_position, arm1_quaternion = _pose(root, "arm1_home", context)
        arm2_position, arm2_quaternion = _pose(root, "arm2_home", context)

        # 保留原始候选信息供人工追溯；数值数组使用上面已经验证过的副本。
        frames.append({
            "home_root_index": root_index,
            "attempt_index": attempt_number,
            "angle_pair_trial": angle_pair_trial,
            "ik_branch_index": branch_index,
            "distance_to_seed": distance,
            "joint_deg": joint_deg,
            "arm1_position": arm1_position,
            "arm1_quat_wxyz": arm1_quaternion,
            "arm2_position": arm2_position,
            "arm2_quat_wxyz": arm2_quaternion,
            "root": dict(root),
        })

    if not frames:
        raise IkExportError("pipeline_attempts 中没有带 home_joint_deg 的 home_root")
    frames.sort(key=lambda frame: frame["home_root_index"])
    root_indices = [int(frame["home_root_index"]) for frame in frames]
    expected_indices = list(range(len(frames)))
    if root_indices != expected_indices:
        raise IkExportError(
            "home_root_index 必须从 0 连续递增；"
            f"实际 {root_indices}，期望 {expected_indices}"
        )
    return frames


def _joint_names(record: Mapping[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    config = _mapping(record.get("config"), "plan_failed.config")
    robot = dict(_mapping(config.get("robot"), "plan_failed.config.robot"))
    prefix = robot.get("dual_arm_prefix")
    if not isinstance(prefix, str) or not prefix:
        raise IkExportError("config.robot.dual_arm_prefix 必须是非空字符串")
    for field in ("base_link", "ee_link", "second_ee_link", "mounts"):
        if not isinstance(robot.get(field), str) or not robot[field]:
            raise IkExportError(f"config.robot.{field} 必须是非空字符串")
    names = [f"J_{index}" for index in range(1, EXPECTED_ARM_DOF + 1)]
    names += [f"{prefix}J_{index}" for index in range(1, EXPECTED_ARM_DOF + 1)]
    if len(set(names)) != EXPECTED_DOF:
        raise IkExportError(f"生成的 12-DOF joint_names 不唯一: {names}")
    robot["joint_names"] = names
    return names, robot


def build_ik_playback_data(
    record: Mapping[str, Any], source: Path, frame_seconds: float
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """由已加载的失败记录构造播放器兼容数组和安全标记 metadata。"""
    frame_seconds = _finite_number(frame_seconds, "frame_seconds")
    if frame_seconds <= 0.0:
        raise IkExportError("frame_seconds 必须 > 0")
    frames = extract_home_root_frames(record)
    joint_names, robot = _joint_names(record)
    config = dict(_mapping(record.get("config"), "plan_failed.config"))

    positions = np.deg2rad(np.stack([frame["joint_deg"] for frame in frames]))
    arm1_positions = np.stack([frame["arm1_position"] for frame in frames])
    arm1_quaternions = np.stack([frame["arm1_quat_wxyz"] for frame in frames])
    arm2_positions = np.stack([frame["arm2_position"] for frame in frames])
    arm2_quaternions = np.stack([frame["arm2_quat_wxyz"] for frame in frames])
    times = np.arange(len(frames), dtype=np.float64) * frame_seconds
    if positions.shape != (len(frames), EXPECTED_DOF):
        raise IkExportError(f"内部错误: positions shape={positions.shape}")
    if not all(np.isfinite(array).all() for array in (
        positions, arm1_positions, arm1_quaternions,
        arm2_positions, arm2_quaternions, times,
    )):
        raise IkExportError("导出数组含 NaN/Inf")
    if len(times) > 1 and not np.all(np.diff(times) > 0.0):
        raise IkExportError("times 必须严格递增")

    payload = {
        "joint_names": np.asarray(joint_names, dtype="S64"),
        "positions": positions.astype(np.float64, copy=False),
        "times": times,
        "ee_positions": arm1_positions.astype(np.float64, copy=False),
        "ee_quats_wxyz": arm1_quaternions.astype(np.float64, copy=False),
        "second_ee_positions": arm2_positions.astype(np.float64, copy=False),
        "second_ee_quats_wxyz": arm2_quaternions.astype(np.float64, copy=False),
        "home_root_index": np.asarray(
            [frame["home_root_index"] for frame in frames], dtype=np.int64
        ),
        "attempt_index": np.asarray(
            [frame["attempt_index"] for frame in frames], dtype=np.int64
        ),
        "angle_pair_trial": np.asarray(
            [frame["angle_pair_trial"] for frame in frames], dtype=np.int64
        ),
        "ik_branch_index": np.asarray(
            [frame["ik_branch_index"] for frame in frames], dtype=np.int64
        ),
        "distance_to_seed": np.asarray(
            [frame["distance_to_seed"] for frame in frames], dtype=np.float64
        ),
    }

    candidate_meta: List[Dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        root = frame["root"]
        candidate_meta.append({
            "frame_index": frame_index,
            "label": (
                f"home_root_{frame['home_root_index']:02d}_"
                f"angle_{frame['angle_pair_trial']}_"
                f"branch_{frame['ik_branch_index']}"
            ),
            "attempt_index": frame["attempt_index"],
            "home_root_index": frame["home_root_index"],
            "angle_pair_trial": frame["angle_pair_trial"],
            "ik_branch_index": frame["ik_branch_index"],
            "distance_to_seed": frame["distance_to_seed"],
            "home_joint_deg": frame["joint_deg"].tolist(),
            "arm1_home_grasp_deg": root.get("arm1_home_grasp_deg"),
            "arm2_home_grasp_deg": root.get("arm2_home_grasp_deg"),
            "arm1_choice": root.get("arm1_choice"),
            "arm2_choice": root.get("arm2_choice"),
            "arm1_home": root.get("arm1_home"),
            "arm2_home": root.get("arm2_home"),
        })

    metadata: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_type": "dual_arm_discrete_ik_candidates",
        "artifact_type": "dual_arm_discrete_ik_candidates",
        "schema_version": 1,
        "visualization_only": True,
        "execution_safe": False,
        "safe_to_execute": False,
        "unsafe_for_execution": True,
        "transition_checked": False,
        "warning": (
            "每帧是独立的 12-DOF IK 候选；帧间没有轨迹规划或碰撞检查，"
            "只能用于 RViz 可视化，禁止下发机器人。"
        ),
        "ee_pose_source": "ik_target_from_home_root_not_recomputed_fk",
        "source_plan_failed": str(Path(source).expanduser().resolve()),
        "source_stage": str(record.get("stage")),
        "robot": robot,
        "workspace": config.get("workspace") or {},
        "config": config,
        "n_points": len(frames),
        "frame_seconds": frame_seconds,
        "interpolation_dt": frame_seconds,
        "total_duration_s": float(times[-1]) if len(times) else 0.0,
        "checks": {
            "source_ik_solver_accepted": True,
            "finite_values": True,
            "joint_dimension": EXPECTED_DOF,
            "joint_limits_checked_by_planner_before_recording": True,
            "independent_static_collision_recheck": False,
            "transition_planned": False,
            "transition_collision_checked": False,
        },
        "candidates": candidate_meta,
    }
    return payload, metadata


def _stage_npz(path: Path, payload: Mapping[str, np.ndarray]) -> Path:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return Path(temporary)


def _stage_json(path: Path, value: Mapping[str, Any]) -> Path:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return Path(temporary)


def write_ik_playback_artifact(
    out_dir: Path,
    payload: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> Tuple[Path, Path, int]:
    """原子写入一份播放器兼容的离散 IK artifact。

    规划器运行时 checkpoint 和事后 CPU 导出共用这个入口，
    确保两者都不会暴露半写入的 NPZ。
    """
    output = Path(out_dir).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise IkExportError(f"输出路径必须是目录: {output}")
    if (output / "plan_failed.json").exists():
        raise IkExportError(
            "输出必须是独立播放子目录，不能包含 plan_failed.json"
        )
    if (output / "trajectory.npz").exists():
        raise IkExportError(
            "输出目录已有正式 trajectory.npz，拒绝覆盖其 metadata"
        )
    existing_meta = output / PLAYBACK_META_JSON
    if existing_meta.exists():
        try:
            with existing_meta.open("r", encoding="utf-8") as stream:
                existing_type = json.load(stream).get("artifact_type")
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise IkExportError(
                f"输出目录已有不可识别的 {PLAYBACK_META_JSON}"
            ) from exc
        if existing_type not in IK_ARTIFACT_TYPES:
            raise IkExportError(
                f"输出目录的 metadata 不是离散 IK artifact: "
                f"{existing_type!r}"
            )
    positions = np.asarray(payload.get("positions"))
    if positions.ndim != 2 or positions.shape[0] == 0:
        raise IkExportError(
            f"payload.positions 必须是非空二维数组，实际 {positions.shape}"
        )
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / IK_RESULTS_NPZ
    meta_path = output / PLAYBACK_META_JSON

    staged_npz: Optional[Path] = None
    staged_meta: Optional[Path] = None
    try:
        staged_npz = _stage_npz(npz_path, payload)
        staged_meta = _stage_json(meta_path, metadata)
        # 先提交 metadata，最后原子替换 NPZ；这样新目录绝不会短暂暴露一个
        # 没有安全说明和 mounts 信息的可播放文件。
        os.replace(str(staged_meta), str(meta_path))
        staged_meta = None
        os.replace(str(staged_npz), str(npz_path))
        staged_npz = None
    finally:
        if staged_npz is not None:
            staged_npz.unlink(missing_ok=True)
        if staged_meta is not None:
            staged_meta.unlink(missing_ok=True)
    return npz_path, meta_path, int(positions.shape[0])


def export_ik_results(
    plan_failed: Path,
    out_dir: Optional[Path] = None,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
) -> Tuple[Path, Path, int]:
    """导出一个失败记录，返回 ``(npz_path, meta_path, frame_count)``。"""
    source = Path(plan_failed).expanduser().resolve()
    record = load_failure_record(source)
    payload, metadata = build_ik_playback_data(record, source, frame_seconds)

    output = (
        source.parent / "ik_playback"
        if out_dir is None else Path(out_dir).expanduser().resolve()
    )
    return write_ik_playback_artifact(output, payload, metadata)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 plan_failed.json 导出独立双臂 Home IK 的 RViz 播放数据"
    )
    parser.add_argument(
        "--plan-failed", required=True, type=Path,
        help="含 pipeline_attempts[*].home_root 的完整 plan_failed.json",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="输出目录；默认 <plan_failed 所在目录>/ik_playback",
    )
    parser.add_argument(
        "--frame-seconds", type=float, default=DEFAULT_FRAME_SECONDS,
        help="离散候选的展示间隔（秒）",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    try:
        npz_path, meta_path, count = export_ik_results(
            args.plan_failed, args.out, args.frame_seconds
        )
    except (IkExportError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(f"[IK EXPORT] candidates={count}")
    print(f"[IK EXPORT] npz={npz_path}")
    print(f"[IK EXPORT] meta={meta_path}")
    print("[IK EXPORT] visualization only; transitions are not execution-safe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
