#!/usr/bin/env python3
"""Check IK reachability for right Marvin action poses where observation pose is zero."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MCAP = REPO_ROOT / "mcap" / "data" / "20260702-191242_optimized.mcap"
DEFAULT_URDF = (
    REPO_ROOT
    / "src"
    / "curobo"
    / "content"
    / "assets"
    / "robot"
    / "marvin_description"
    / "urdf"
    / "right_marvin.urdf"
)
DEFAULT_OBS_TOPIC = "/robot/data/marvin_arm_right/observation"
DEFAULT_ACTION_TOPIC = "/robot/data/marvin_arm_right/action"
DEFAULT_JOINT_NAMES = "Joint1 Joint2 Joint3 Joint4 Joint5 Joint6 Joint7"
DEFAULT_LEFT_TRANSLATION = "0 -0.061 0"
DEFAULT_LEFT_ROTATION_XYZW = "0.68304129 0.18291100 0.18291033 0.68303878"
DEFAULT_RIGHT_TRANSLATION = "0 -0.3202976292452259 0"
DEFAULT_RIGHT_ROTATION_XYZW = "0.5 -0.5 0.5 0.5"


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: Optional[float]
    upper: Optional[float]


@dataclass
class ActionSample:
    source_index: int
    obs_log_time_ns: int
    action_log_time_ns: int
    action_position: np.ndarray
    action_quaternion_xyzw: np.ndarray
    target_transform: np.ndarray


@dataclass
class IkResult:
    success: bool
    q: np.ndarray
    position_error_m: float
    rotation_error_rad: float
    iterations: int
    seed_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read right Marvin observation/action messages, select frames where "
            "observation.multibody_pose.pose is all zeros, transform the paired "
            "action pose as inv(T_left) @ T_action @ inv(T_right), and solve "
            "Base->Link7 IK from right_marvin.urdf without collision checks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mcap", type=Path, default=DEFAULT_MCAP)
    parser.add_argument("--observation-topic", default=DEFAULT_OBS_TOPIC)
    parser.add_argument("--action-topic", default=DEFAULT_ACTION_TOPIC)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--base-link", default="Base")
    parser.add_argument("--ee-link", default="Link7")
    parser.add_argument("--joint-names", default=DEFAULT_JOINT_NAMES)
    parser.add_argument("--left-translation", default=DEFAULT_LEFT_TRANSLATION)
    parser.add_argument("--left-rotation-xyzw", default=DEFAULT_LEFT_ROTATION_XYZW)
    parser.add_argument("--right-translation", default=DEFAULT_RIGHT_TRANSLATION)
    parser.add_argument("--right-rotation-xyzw", default=DEFAULT_RIGHT_ROTATION_XYZW)
    parser.add_argument("--zero-atol", type=float, default=1e-12)
    parser.add_argument("--position-threshold", type=float, default=0.005)
    parser.add_argument("--rotation-threshold", type=float, default=0.05)
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--max-step-rad", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=0, help="Use <=0 for all selected poses.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def parse_float_list(value: str, expected: int, name: str) -> np.ndarray:
    parts = value.replace(",", " ").split()
    if len(parts) != expected:
        raise ValueError(f"{name} expects {expected} numbers, got: {value!r}")
    return np.asarray([float(part) for part in parts], dtype=np.float64)


def parse_name_list(value: str) -> List[str]:
    names = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if not names:
        raise ValueError("Expected at least one joint name.")
    return names


def normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Encountered zero-length quaternion.")
    return quat / norm


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rotation)))
        if idx == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    if quat[3] < 0.0:
        quat *= -1.0
    return normalize_quaternion_xyzw(quat)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0 or abs(angle) <= 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def pose_to_transform(position_xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    return make_transform(quat_xyzw_to_matrix(quat_xyzw), position_xyz)


def xyz_rpy_to_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    return make_transform(rpy_to_matrix(rpy), xyz)


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ translation)
    return result


def rotation_angle_rad(rotation: np.ndarray) -> float:
    cos_angle = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.acos(max(-1.0, min(1.0, cos_angle)))


def rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    angle = rotation_angle_rad(rotation)
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if math.pi - angle < 1e-5:
        quat = matrix_to_quat_xyzw(rotation)
        axis = quat[:3]
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            return np.zeros(3, dtype=np.float64)
        return axis / norm * angle
    scale = angle / (2.0 * math.sin(angle))
    return scale * np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )


def parse_origin(joint_elem: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
    origin = joint_elem.find("origin")
    if origin is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = parse_float_list(origin.attrib.get("xyz", "0 0 0"), 3, "origin xyz")
    rpy = parse_float_list(origin.attrib.get("rpy", "0 0 0"), 3, "origin rpy")
    return xyz, rpy


def parse_axis(joint_elem: ET.Element) -> np.ndarray:
    axis = joint_elem.find("axis")
    if axis is None:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return parse_float_list(axis.attrib.get("xyz", "1 0 0"), 3, "axis xyz")


def parse_limit(joint_elem: ET.Element) -> Tuple[Optional[float], Optional[float]]:
    limit = joint_elem.find("limit")
    if limit is None:
        return None, None
    lower = float(limit.attrib["lower"]) if "lower" in limit.attrib else None
    upper = float(limit.attrib["upper"]) if "upper" in limit.attrib else None
    return lower, upper


def load_urdf_joints(urdf_path: Path) -> Dict[str, JointSpec]:
    root = ET.parse(urdf_path).getroot()
    joints: Dict[str, JointSpec] = {}
    for joint_elem in root.findall("joint"):
        parent = joint_elem.find("parent")
        child = joint_elem.find("child")
        if parent is None or child is None:
            continue
        xyz, rpy = parse_origin(joint_elem)
        lower, upper = parse_limit(joint_elem)
        joint = JointSpec(
            name=joint_elem.attrib["name"],
            joint_type=joint_elem.attrib.get("type", "fixed"),
            parent=parent.attrib["link"],
            child=child.attrib["link"],
            xyz=xyz,
            rpy=rpy,
            axis=parse_axis(joint_elem),
            lower=lower,
            upper=upper,
        )
        joints[joint.name] = joint
    return joints


def find_joint_chain(joints: Dict[str, JointSpec], base_link: str, ee_link: str) -> List[JointSpec]:
    by_parent: Dict[str, List[JointSpec]] = {}
    for joint in joints.values():
        by_parent.setdefault(joint.parent, []).append(joint)

    queue: List[Tuple[str, List[JointSpec]]] = [(base_link, [])]
    visited = set()
    while queue:
        link, chain = queue.pop(0)
        if link == ee_link:
            return chain
        if link in visited:
            continue
        visited.add(link)
        for joint in by_parent.get(link, []):
            queue.append((joint.child, chain + [joint]))
    raise ValueError(f"Could not find joint chain from {base_link!r} to {ee_link!r}.")


def active_chain_joints(chain: Sequence[JointSpec]) -> List[JointSpec]:
    return [joint for joint in chain if joint.joint_type in {"revolute", "continuous"}]


def joint_limits(active_joints: Sequence[JointSpec]) -> Tuple[np.ndarray, np.ndarray]:
    lower: List[float] = []
    upper: List[float] = []
    for joint in active_joints:
        if joint.lower is None or joint.upper is None:
            raise RuntimeError(f"Joint {joint.name} is missing lower/upper limits in URDF.")
        lower.append(joint.lower)
        upper.append(joint.upper)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def compute_fk_and_jacobian(
    chain: Sequence[JointSpec],
    joint_name_to_index: Dict[str, int],
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    transform = np.eye(4, dtype=np.float64)
    joint_origins: List[np.ndarray] = []
    joint_axes: List[np.ndarray] = []
    joint_indices: List[int] = []

    for joint in chain:
        transform = transform @ xyz_rpy_to_transform(joint.xyz, joint.rpy)
        if joint.joint_type in {"revolute", "continuous"}:
            q_index = joint_name_to_index[joint.name]
            axis_world = transform[:3, :3] @ (joint.axis / np.linalg.norm(joint.axis))
            joint_origins.append(transform[:3, 3].copy())
            joint_axes.append(axis_world.copy())
            joint_indices.append(q_index)
            transform = transform @ make_transform(
                axis_angle_to_matrix(joint.axis, float(q[q_index])), np.zeros(3)
            )

    ee_position = transform[:3, 3]
    jacobian = np.zeros((6, len(q)), dtype=np.float64)
    for origin, axis, q_index in zip(joint_origins, joint_axes, joint_indices):
        jacobian[:3, q_index] = np.cross(axis, ee_position - origin)
        jacobian[3:, q_index] = axis
    return transform, jacobian


def pose_error(current: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float, float]:
    position_error = target[:3, 3] - current[:3, 3]
    rotation_error_matrix = target[:3, :3] @ current[:3, :3].T
    rotation_error = rotation_vector_from_matrix(rotation_error_matrix)
    return np.concatenate([position_error, rotation_error]), float(
        np.linalg.norm(position_error)
    ), float(np.linalg.norm(rotation_error))


def clamp_to_limits(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(q, lower), upper)


def solve_single_ik(
    chain: Sequence[JointSpec],
    joint_name_to_index: Dict[str, int],
    target: np.ndarray,
    seeds: Sequence[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    position_threshold: float,
    rotation_threshold: float,
    max_iterations: int,
    damping: float,
    max_step_rad: float,
) -> IkResult:
    best: Optional[IkResult] = None
    damping_matrix = (damping * damping) * np.eye(6, dtype=np.float64)

    for seed_index, seed_q in enumerate(seeds):
        q = clamp_to_limits(seed_q.astype(np.float64, copy=True), lower, upper)
        last_norm = float("inf")
        pos_error = float("inf")
        rot_error = float("inf")
        iterations_used = 0

        for iteration in range(max_iterations + 1):
            current, jacobian = compute_fk_and_jacobian(chain, joint_name_to_index, q)
            error_vec, pos_error, rot_error = pose_error(current, target)
            err_norm = float(np.linalg.norm(error_vec))
            iterations_used = iteration
            if pos_error <= position_threshold and rot_error <= rotation_threshold:
                return IkResult(True, q, pos_error, rot_error, iterations_used, seed_index)

            if iteration == max_iterations:
                break

            lhs = jacobian @ jacobian.T + damping_matrix
            try:
                step = jacobian.T @ np.linalg.solve(lhs, error_vec)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(jacobian) @ error_vec

            step_norm = float(np.linalg.norm(step, ord=np.inf))
            if step_norm > max_step_rad:
                step *= max_step_rad / step_norm

            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = clamp_to_limits(q + scale * step, lower, upper)
                candidate_fk, _ = compute_fk_and_jacobian(chain, joint_name_to_index, candidate)
                candidate_error, _, _ = pose_error(candidate_fk, target)
                candidate_norm = float(np.linalg.norm(candidate_error))
                if candidate_norm < err_norm or candidate_norm < last_norm:
                    q = candidate
                    last_norm = candidate_norm
                    accepted = True
                    break

            if not accepted:
                q = clamp_to_limits(q + 0.02 * step, lower, upper)
                last_norm = err_norm

        result = IkResult(False, q, pos_error, rot_error, iterations_used, seed_index)
        if best is None:
            best = result
        else:
            best_score = best.position_error_m / position_threshold + best.rotation_error_rad / rotation_threshold
            score = result.position_error_m / position_threshold + result.rotation_error_rad / rotation_threshold
            if score < best_score:
                best = result

    if best is None:
        raise RuntimeError("No IK seeds were provided.")
    return best


def make_seed_bank(
    lower: np.ndarray,
    upper: np.ndarray,
    num_seeds: int,
    rng: np.random.Generator,
    previous_solution: Optional[np.ndarray],
) -> List[np.ndarray]:
    seeds: List[np.ndarray] = []
    center = (lower + upper) * 0.5
    zero = clamp_to_limits(np.zeros_like(center), lower, upper)
    if previous_solution is not None:
        seeds.append(previous_solution.copy())
        for scale in (0.03, 0.08, 0.15):
            seeds.append(clamp_to_limits(previous_solution + rng.normal(0.0, scale, size=center.shape), lower, upper))
    seeds.extend([zero, center])

    seen: List[np.ndarray] = []
    deduped: List[np.ndarray] = []
    for seed in seeds:
        if not any(np.allclose(seed, old, atol=1e-9) for old in seen):
            deduped.append(seed)
            seen.append(seed)

    while len(deduped) < num_seeds:
        deduped.append(rng.uniform(lower, upper))
    return deduped[:num_seeds]


def ros_pose_values(pose: Any) -> np.ndarray:
    return np.asarray(
        [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ],
        dtype=np.float64,
    )


def load_selected_action_samples(
    mcap_path: Path,
    observation_topic: str,
    action_topic: str,
    zero_atol: float,
    left_transform: np.ndarray,
    right_transform: np.ndarray,
    max_samples: int,
) -> Tuple[List[ActionSample], Dict[str, Any]]:
    from mcap.reader import make_reader
    from mcap_ros1.decoder import DecoderFactory

    observations: List[Tuple[int, int, np.ndarray]] = []
    actions: List[Tuple[int, int, np.ndarray, np.ndarray]] = []

    with mcap_path.open("rb") as file_obj:
        reader = make_reader(file_obj, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages(
            topics=[observation_topic, action_topic]
        ):
            if channel.topic == observation_topic:
                pose = decoded.multibody_pose.pose
                observations.append((len(observations), int(message.log_time), ros_pose_values(pose)))
            elif channel.topic == action_topic:
                pose = decoded.pose_command.pose
                values = ros_pose_values(pose)
                actions.append(
                    (
                        len(actions),
                        int(message.log_time),
                        values[:3].copy(),
                        normalize_quaternion_xyzw(values[3:].copy()),
                    )
                )

    paired_count = min(len(observations), len(actions))
    selected: List[ActionSample] = []
    selected_time_deltas_ns: List[int] = []
    inv_left = inverse_transform(left_transform)
    inv_right = inverse_transform(right_transform)
    max_count = None if max_samples <= 0 else max_samples

    for pair_index in range(paired_count):
        obs_index, obs_time, obs_pose_values = observations[pair_index]
        action_index, action_time, action_pos, action_quat = actions[pair_index]
        if obs_index != action_index:
            raise RuntimeError("Internal pairing index mismatch.")
        if not np.all(np.abs(obs_pose_values) <= zero_atol):
            continue

        action_transform = pose_to_transform(action_pos, action_quat)
        target_transform = inv_left @ action_transform @ inv_right
        selected.append(
            ActionSample(
                source_index=pair_index,
                obs_log_time_ns=obs_time,
                action_log_time_ns=action_time,
                action_position=action_pos,
                action_quaternion_xyzw=action_quat,
                target_transform=target_transform,
            )
        )
        selected_time_deltas_ns.append(action_time - obs_time)
        if max_count is not None and len(selected) >= max_count:
            break

    counts: Dict[str, Any] = {
        "observation_messages": len(observations),
        "action_messages": len(actions),
        "paired_by_index": paired_count,
        "zero_observation_pose_frames": int(
            sum(np.all(np.abs(obs[2]) <= zero_atol) for obs in observations[:paired_count])
        ),
        "selected_frames": len(selected),
    }
    if selected_time_deltas_ns:
        deltas_ms = np.asarray(selected_time_deltas_ns, dtype=np.float64) / 1e6
        counts["selected_action_minus_observation_time_delta_ms"] = summarize(deltas_ms)

    return selected, counts


def summarize(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def default_output_dir(mcap_path: Path) -> Path:
    return REPO_ROOT / "mcap" / "results" / f"{mcap_path.stem}_right_zero_obs_action_ik"


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive.")
    if not args.mcap.is_file():
        raise FileNotFoundError(f"MCAP file not found: {args.mcap}")
    if not args.urdf.is_file():
        raise FileNotFoundError(f"URDF file not found: {args.urdf}")

    joint_names = parse_name_list(args.joint_names)
    left_transform = pose_to_transform(
        parse_float_list(args.left_translation, 3, "--left-translation"),
        parse_float_list(args.left_rotation_xyzw, 4, "--left-rotation-xyzw"),
    )
    right_transform = pose_to_transform(
        parse_float_list(args.right_translation, 3, "--right-translation"),
        parse_float_list(args.right_rotation_xyzw, 4, "--right-rotation-xyzw"),
    )

    joints = load_urdf_joints(args.urdf)
    chain = find_joint_chain(joints, args.base_link, args.ee_link)
    active_joints = active_chain_joints(chain)
    chain_joint_names = [joint.name for joint in active_joints]
    if chain_joint_names != joint_names:
        raise ValueError(
            "URDF active joints do not match --joint-names.\n"
            f"chain active joints: {chain_joint_names}\n"
            f"--joint-names: {joint_names}"
        )
    lower, upper = joint_limits(active_joints)
    joint_name_to_index = {name: index for index, name in enumerate(joint_names)}

    samples, counts = load_selected_action_samples(
        args.mcap,
        args.observation_topic,
        args.action_topic,
        args.zero_atol,
        left_transform,
        right_transform,
        args.max_samples,
    )
    if not samples:
        raise RuntimeError("No action poses were selected from zero observation-pose frames.")

    print("=" * 72)
    print("Right Marvin zero-observation action-pose IK check")
    print(f"MCAP: {args.mcap}")
    print(f"observation topic: {args.observation_topic}")
    print(f"action topic: {args.action_topic}")
    print(f"selected zero-pose frames: {len(samples)}")
    print(f"URDF chain: {args.base_link} -> {args.ee_link}")
    print(f"active joints: {joint_names}")
    print("target transform: inverse(T_left) @ T_action @ inverse(T_right)")
    print("collision checking: disabled")
    print(
        f"IK thresholds: position <= {args.position_threshold} m, "
        f"rotation <= {args.rotation_threshold} rad"
    )
    print(f"IK seeds per target: {args.num_seeds}")
    print("=" * 72)

    rng = np.random.default_rng(args.seed)
    previous_solution: Optional[np.ndarray] = None
    rows: List[Dict[str, Any]] = []
    results: List[IkResult] = []
    start_time = time.time()

    for index, sample in enumerate(samples):
        seeds = make_seed_bank(lower, upper, args.num_seeds, rng, previous_solution)
        result = solve_single_ik(
            chain,
            joint_name_to_index,
            sample.target_transform,
            seeds,
            lower,
            upper,
            args.position_threshold,
            args.rotation_threshold,
            args.max_iterations,
            args.damping,
            args.max_step_rad,
        )
        if result.success:
            previous_solution = result.q.copy()
        results.append(result)

        target_quat = matrix_to_quat_xyzw(sample.target_transform[:3, :3])
        row: Dict[str, Any] = {
            "selected_index": index,
            "source_message_index": sample.source_index,
            "obs_log_time_ns": sample.obs_log_time_ns,
            "action_log_time_ns": sample.action_log_time_ns,
            "action_minus_obs_time_delta_ms": (sample.action_log_time_ns - sample.obs_log_time_ns)
            / 1e6,
            "success": bool(result.success),
            "position_error_m": result.position_error_m,
            "rotation_error_rad": result.rotation_error_rad,
            "rotation_error_deg": math.degrees(result.rotation_error_rad),
            "iterations": result.iterations,
            "seed_index": result.seed_index,
            "action_x_m": float(sample.action_position[0]),
            "action_y_m": float(sample.action_position[1]),
            "action_z_m": float(sample.action_position[2]),
            "action_qx": float(sample.action_quaternion_xyzw[0]),
            "action_qy": float(sample.action_quaternion_xyzw[1]),
            "action_qz": float(sample.action_quaternion_xyzw[2]),
            "action_qw": float(sample.action_quaternion_xyzw[3]),
            "target_x_m": float(sample.target_transform[0, 3]),
            "target_y_m": float(sample.target_transform[1, 3]),
            "target_z_m": float(sample.target_transform[2, 3]),
            "target_qx": float(target_quat[0]),
            "target_qy": float(target_quat[1]),
            "target_qz": float(target_quat[2]),
            "target_qw": float(target_quat[3]),
        }
        for joint_name, value in zip(joint_names, result.q.tolist()):
            row[f"q_{joint_name}_rad"] = float(value)
        rows.append(row)

        if (index + 1) % 20 == 0 or index + 1 == len(samples):
            elapsed = time.time() - start_time
            success_count = sum(1 for item in results if item.success)
            print(
                f"  IK {index + 1}/{len(samples)} "
                f"success={success_count}/{index + 1} elapsed={elapsed:.1f}s",
                flush=True,
            )

    success_mask = np.asarray([result.success for result in results], dtype=bool)
    position_errors = np.asarray([result.position_error_m for result in results], dtype=np.float64)
    rotation_errors = np.asarray([result.rotation_error_rad for result in results], dtype=np.float64)
    elapsed = time.time() - start_time
    output_dir = args.output_dir or default_output_dir(args.mcap)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ik_results.csv"
    write_csv(csv_path, rows)

    failed_rows = [row for row in rows if not row["success"]]
    failed_csv_path = output_dir / "ik_failed_results.csv"
    write_csv(failed_csv_path, failed_rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mcap": str(args.mcap),
        "observation_topic": args.observation_topic,
        "action_topic": args.action_topic,
        "urdf": str(args.urdf),
        "base_link": args.base_link,
        "ee_link": args.ee_link,
        "joint_names": joint_names,
        "joint_limits_rad": {
            name: {"lower": float(lo), "upper": float(hi)}
            for name, lo, hi in zip(joint_names, lower.tolist(), upper.tolist())
        },
        "sample_counts": counts,
        "zero_pose_definition": (
            "all components of observation.multibody_pose.pose "
            f"[x,y,z,qx,qy,qz,qw] have abs(value) <= {args.zero_atol}"
        ),
        "pairing": "observation and action are paired by message index after topic filtering",
        "target_transform_definition": "T_goal = inverse(T_left) @ T_action @ inverse(T_right)",
        "T_left_parent_base": {
            "translation_xyz_m": parse_float_list(
                args.left_translation, 3, "--left-translation"
            ).tolist(),
            "rotation_xyzw": normalize_quaternion_xyzw(
                parse_float_list(args.left_rotation_xyzw, 4, "--left-rotation-xyzw")
            ).tolist(),
        },
        "T_right_parent_base": {
            "translation_xyz_m": parse_float_list(
                args.right_translation, 3, "--right-translation"
            ).tolist(),
            "rotation_xyzw": normalize_quaternion_xyzw(
                parse_float_list(args.right_rotation_xyzw, 4, "--right-rotation-xyzw")
            ).tolist(),
        },
        "ik_method": {
            "solver": "CPU damped least-squares geometric-Jacobian IK",
            "self_collision_check": False,
            "position_threshold_m": args.position_threshold,
            "rotation_threshold_rad": args.rotation_threshold,
            "num_seeds": args.num_seeds,
            "max_iterations": args.max_iterations,
            "damping": args.damping,
            "max_step_rad": args.max_step_rad,
            "random_seed": args.seed,
        },
        "success_count": int(success_mask.sum()),
        "total_count": int(success_mask.size),
        "success_rate": float(success_mask.mean()),
        "elapsed_s": elapsed,
        "position_error_m": summarize(position_errors),
        "rotation_error_rad": summarize(rotation_errors),
        "rotation_error_deg": summarize(np.rad2deg(rotation_errors)),
        "successful_position_error_m": summarize(position_errors[success_mask]),
        "successful_rotation_error_rad": summarize(rotation_errors[success_mask]),
        "failed_count": int((~success_mask).sum()),
        "csv": str(csv_path),
        "failed_csv": str(failed_csv_path) if failed_rows else None,
    }
    if failed_rows:
        worst_failed = max(
            failed_rows,
            key=lambda row: row["position_error_m"] / args.position_threshold
            + row["rotation_error_rad"] / args.rotation_threshold,
        )
        summary["worst_failed_sample"] = worst_failed

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print("-" * 72)
    print(
        f"IK success: {summary['success_count']}/{summary['total_count']} "
        f"({100.0 * summary['success_rate']:.2f}%)"
    )
    print(
        "position error [m]: "
        f"mean={summary['position_error_m']['mean']:.6g}, "
        f"p95={summary['position_error_m']['p95']:.6g}, "
        f"max={summary['position_error_m']['max']:.6g}"
    )
    print(
        "rotation error [deg]: "
        f"mean={summary['rotation_error_deg']['mean']:.6g}, "
        f"p95={summary['rotation_error_deg']['p95']:.6g}, "
        f"max={summary['rotation_error_deg']['max']:.6g}"
    )
    print(f"elapsed: {elapsed:.2f}s")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")
    if failed_rows:
        print(f"failed csv: {failed_csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
