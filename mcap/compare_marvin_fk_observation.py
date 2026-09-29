#!/usr/bin/env python3
"""Compare Marvin URDF FK against observation multibody_pose from an MCAP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MCAP = REPO_ROOT / "mcap" / "data" / "20260702-122552_optimized.mcap"
DEFAULT_TOPIC = "/robot/data/marvin_arm_left/observation"
DEFAULT_URDF = (
    REPO_ROOT
    / "src"
    / "curobo"
    / "content"
    / "assets"
    / "robot"
    / "marvin_description"
    / "urdf"
    / "left_marvin.urdf"
)
DEFAULT_BASE_LINK = "Base"
DEFAULT_EE_LINK = "Link7"
DEFAULT_JOINT_NAMES = "Joint1 Joint2 Joint3 Joint4 Joint5 Joint6 Joint7"
DEFAULT_PRE_TRANSLATION = "0 0.061 0"
DEFAULT_PRE_ROTATION_XYZW = "-0.68304129 0.18291100 -0.18291033 0.68303878"


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


@dataclass
class ObservationSample:
    source_index: int
    log_time_ns: int
    q: np.ndarray
    observed_pose: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Marvin observation messages from MCAP, compute left_marvin.urdf "
            "FK from Base to Link7, left-multiply a fixed transform, and compare "
            "against multibody_pose."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mcap", type=Path, default=DEFAULT_MCAP)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--base-link", default=DEFAULT_BASE_LINK)
    parser.add_argument("--ee-link", default=DEFAULT_EE_LINK)
    parser.add_argument(
        "--joint-names",
        default=DEFAULT_JOINT_NAMES,
        help="Space- or comma-separated joint names matching multibody_state.states order.",
    )
    parser.add_argument(
        "--pre-translation",
        default=DEFAULT_PRE_TRANSLATION,
        help="xyz translation of the left-multiplied transform.",
    )
    parser.add_argument(
        "--pre-rotation-xyzw",
        default=DEFAULT_PRE_ROTATION_XYZW,
        help="xyzw quaternion of the left-multiplied transform.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for csv/json/png outputs. Defaults under mcap/results.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Use <=0 for all valid frames.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def parse_float_list(value: str, expected: int, name: str) -> np.ndarray:
    parts = value.replace(",", " ").split()
    if len(parts) != expected:
        raise ValueError(f"{name} expects {expected} numbers, got {value!r}")
    return np.asarray([float(part) for part in parts], dtype=np.float64)


def parse_name_list(value: str) -> List[str]:
    names = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if not names:
        raise ValueError("Expected at least one joint name")
    return names


def normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise ValueError("Encountered zero-length quaternion")
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
    norm = np.linalg.norm(axis)
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


def xyz_rpy_to_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    return make_transform(rpy_to_matrix(rpy), xyz)


def pose_to_transform(position_xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    return make_transform(quat_xyzw_to_matrix(quat_xyzw), position_xyz)


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


def load_urdf_joints(urdf_path: Path) -> Dict[str, JointSpec]:
    root = ET.parse(urdf_path).getroot()
    joints: Dict[str, JointSpec] = {}
    for joint_elem in root.findall("joint"):
        parent = joint_elem.find("parent")
        child = joint_elem.find("child")
        if parent is None or child is None:
            continue
        xyz, rpy = parse_origin(joint_elem)
        spec = JointSpec(
            name=joint_elem.attrib["name"],
            joint_type=joint_elem.attrib.get("type", "fixed"),
            parent=parent.attrib["link"],
            child=child.attrib["link"],
            xyz=xyz,
            rpy=rpy,
            axis=parse_axis(joint_elem),
        )
        joints[spec.name] = spec
    return joints


def find_joint_chain(joints: Dict[str, JointSpec], base_link: str, ee_link: str) -> List[JointSpec]:
    by_parent: Dict[str, List[JointSpec]] = {}
    for joint in joints.values():
        by_parent.setdefault(joint.parent, []).append(joint)

    queue = deque([(base_link, [])])
    visited = set()
    while queue:
        link, chain = queue.popleft()
        if link == ee_link:
            return chain
        if link in visited:
            continue
        visited.add(link)
        for joint in by_parent.get(link, []):
            queue.append((joint.child, chain + [joint]))
    raise ValueError(f"Could not find a joint chain from {base_link!r} to {ee_link!r}")


def compute_fk(chain: List[JointSpec], q_by_joint: Dict[str, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    for joint in chain:
        transform = transform @ xyz_rpy_to_transform(joint.xyz, joint.rpy)
        if joint.joint_type in {"revolute", "continuous"}:
            angle = q_by_joint.get(joint.name)
            if angle is None:
                raise KeyError(f"Missing q for joint {joint.name}")
            transform = transform @ make_transform(axis_angle_to_matrix(joint.axis, angle), np.zeros(3))
    return transform


def ros_time_to_ns(stamp: Any) -> int:
    return int(getattr(stamp, "secs", 0)) * 1_000_000_000 + int(getattr(stamp, "nsecs", 0))


def load_observation_samples(
    mcap_path: Path,
    topic: str,
    dof: int,
    max_samples: int,
) -> Tuple[List[ObservationSample], Dict[str, int]]:
    from mcap.reader import make_reader
    from mcap_ros1.decoder import DecoderFactory

    samples: List[ObservationSample] = []
    counts = {"messages": 0, "empty_state": 0, "wrong_state_length": 0, "used": 0}
    max_count = None if max_samples <= 0 else max_samples

    with mcap_path.open("rb") as file_obj:
        reader = make_reader(file_obj, decoder_factories=[DecoderFactory()])
        for _, _, message, decoded in reader.iter_decoded_messages(topics=[topic]):
            counts["messages"] += 1
            states = decoded.multibody_state.states
            if len(states) == 0:
                counts["empty_state"] += 1
                continue
            if len(states) < dof:
                counts["wrong_state_length"] += 1
                continue

            pose = decoded.multibody_pose.pose
            position = np.asarray(
                [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
                dtype=np.float64,
            )
            quat = np.asarray(
                [
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ],
                dtype=np.float64,
            )
            q = np.asarray([float(state.q) for state in states[:dof]], dtype=np.float64)
            samples.append(
                ObservationSample(
                    source_index=counts["messages"] - 1,
                    log_time_ns=int(message.log_time),
                    q=q,
                    observed_pose=pose_to_transform(position, quat),
                )
            )
            counts["used"] += 1
            if max_count is not None and len(samples) >= max_count:
                break

    return samples, counts


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def default_output_dir(mcap_path: Path, topic: str) -> Path:
    topic_name = topic.strip("/").replace("/", "_")
    return REPO_ROOT / "mcap" / "results" / f"{mcap_path.stem}_{topic_name}_fk_pose_compare"


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_plots(output_dir: Path, translation_mm: np.ndarray, rotation_deg: np.ndarray) -> List[str]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional.
        print(f"Plotting skipped: {exc}")
        return []

    plot_paths: List[str] = []
    x = np.arange(len(translation_mm))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(x, translation_mm, linewidth=0.8)
    axes[0].axhline(float(np.mean(translation_mm)), color="tab:green", linestyle="--", label="mean")
    axes[0].axhline(float(np.max(translation_mm)), color="tab:red", linestyle=":", label="max")
    axes[0].set_ylabel("translation error [mm]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(x, rotation_deg, linewidth=0.8)
    axes[1].axhline(float(np.mean(rotation_deg)), color="tab:green", linestyle="--", label="mean")
    axes[1].axhline(float(np.max(rotation_deg)), color="tab:red", linestyle=":", label="max")
    axes[1].set_xlabel("valid state frame index")
    axes[1].set_ylabel("rotation error [deg]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    path = output_dir / "fk_pose_error_by_frame.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(translation_mm, bins=80)
    axes[0].axvline(float(np.mean(translation_mm)), color="tab:green", linestyle="--", label="mean")
    axes[0].axvline(float(np.max(translation_mm)), color="tab:red", linestyle=":", label="max")
    axes[0].set_xlabel("translation error [mm]")
    axes[0].set_ylabel("count")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].hist(rotation_deg, bins=80)
    axes[1].axvline(float(np.mean(rotation_deg)), color="tab:green", linestyle="--", label="mean")
    axes[1].axvline(float(np.max(rotation_deg)), color="tab:red", linestyle=":", label="max")
    axes[1].set_xlabel("rotation error [deg]")
    axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    path = output_dir / "fk_pose_error_histogram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(path))

    return plot_paths


def main() -> None:
    args = parse_args()
    if not args.mcap.is_file():
        raise FileNotFoundError(f"MCAP file not found: {args.mcap}")
    if not args.urdf.is_file():
        raise FileNotFoundError(f"URDF file not found: {args.urdf}")

    joint_names = parse_name_list(args.joint_names)
    pre_translation = parse_float_list(args.pre_translation, 3, "--pre-translation")
    pre_rotation_xyzw = parse_float_list(args.pre_rotation_xyzw, 4, "--pre-rotation-xyzw")
    pre_transform = pose_to_transform(pre_translation, pre_rotation_xyzw)

    joints = load_urdf_joints(args.urdf)
    chain = find_joint_chain(joints, args.base_link, args.ee_link)
    chain_joint_names = [joint.name for joint in chain if joint.joint_type in {"revolute", "continuous"}]
    if chain_joint_names != joint_names:
        raise ValueError(
            "The active joint names in the URDF chain do not match --joint-names.\n"
            f"chain active joints: {chain_joint_names}\n"
            f"--joint-names: {joint_names}"
        )

    samples, counts = load_observation_samples(args.mcap, args.topic, len(joint_names), args.max_samples)
    if not samples:
        raise RuntimeError("No valid observation samples with joint states were found.")

    rows: List[Dict[str, Any]] = []
    translation_errors_m: List[float] = []
    rotation_errors_rad: List[float] = []
    for valid_index, sample in enumerate(samples):
        q_by_joint = dict(zip(joint_names, sample.q.tolist()))
        fk_base_to_link7 = compute_fk(chain, q_by_joint)
        predicted_pose = pre_transform @ fk_base_to_link7
        error_transform = inverse_transform(sample.observed_pose) @ predicted_pose
        translation_error = error_transform[:3, 3]
        rotation_error = rotation_angle_rad(error_transform[:3, :3])
        translation_norm = float(np.linalg.norm(translation_error))

        translation_errors_m.append(translation_norm)
        rotation_errors_rad.append(rotation_error)
        row: Dict[str, Any] = {
            "valid_index": valid_index,
            "source_message_index": sample.source_index,
            "log_time_ns": sample.log_time_ns,
            "translation_error_m": translation_norm,
            "translation_error_mm": translation_norm * 1000.0,
            "rotation_error_rad": rotation_error,
            "rotation_error_deg": math.degrees(rotation_error),
            "relative_error_x_m": float(translation_error[0]),
            "relative_error_y_m": float(translation_error[1]),
            "relative_error_z_m": float(translation_error[2]),
            "observed_x_m": float(sample.observed_pose[0, 3]),
            "observed_y_m": float(sample.observed_pose[1, 3]),
            "observed_z_m": float(sample.observed_pose[2, 3]),
            "predicted_x_m": float(predicted_pose[0, 3]),
            "predicted_y_m": float(predicted_pose[1, 3]),
            "predicted_z_m": float(predicted_pose[2, 3]),
        }
        for name, value in zip(joint_names, sample.q.tolist()):
            row[f"q_{name}_rad"] = float(value)
        rows.append(row)

    translation_errors_m_arr = np.asarray(translation_errors_m, dtype=np.float64)
    rotation_errors_rad_arr = np.asarray(rotation_errors_rad, dtype=np.float64)
    translation_errors_mm = translation_errors_m_arr * 1000.0
    rotation_errors_deg = np.rad2deg(rotation_errors_rad_arr)

    output_dir = args.output_dir or default_output_dir(args.mcap, args.topic)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fk_pose_comparison.csv"
    write_csv(csv_path, rows)

    max_t_idx = int(np.argmax(translation_errors_m_arr))
    max_r_idx = int(np.argmax(rotation_errors_rad_arr))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mcap": str(args.mcap),
        "topic": args.topic,
        "urdf": str(args.urdf),
        "base_link": args.base_link,
        "ee_link": args.ee_link,
        "joint_names": joint_names,
        "chain_joints": [joint.name for joint in chain],
        "left_multiply_transform": {
            "translation_xyz_m": pre_translation.tolist(),
            "rotation_xyzw": normalize_quaternion_xyzw(pre_rotation_xyzw).tolist(),
        },
        "sample_counts": counts,
        "comparison_definition": "T_error = inverse(T_multibody_pose) @ (T_left_multiply @ FK_Base_to_Link7)",
        "translation_error_m": summarize(translation_errors_m_arr),
        "translation_error_mm": summarize(translation_errors_mm),
        "rotation_error_rad": summarize(rotation_errors_rad_arr),
        "rotation_error_deg": summarize(rotation_errors_deg),
        "max_translation_error_sample": rows[max_t_idx],
        "max_rotation_error_sample": rows[max_r_idx],
    }

    plot_paths: List[str] = []
    if not args.no_plots:
        plot_paths = maybe_write_plots(output_dir, translation_errors_mm, rotation_errors_deg)
    summary["plots"] = plot_paths

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print("=" * 72)
    print("Marvin FK vs multibody_pose comparison")
    print(f"MCAP: {args.mcap}")
    print(f"topic: {args.topic}")
    print(f"URDF chain: {args.base_link} -> {args.ee_link}")
    print(f"valid samples: {len(samples)} / messages: {counts['messages']}")
    print(f"skipped empty states: {counts['empty_state']}")
    print(f"left-multiply translation [m]: {pre_translation.tolist()}")
    print(f"left-multiply rotation xyzw: {normalize_quaternion_xyzw(pre_rotation_xyzw).tolist()}")
    print("-" * 72)
    print(
        "translation error [mm]: "
        f"mean={summary['translation_error_mm']['mean']:.6f}, "
        f"p95={summary['translation_error_mm']['p95']:.6f}, "
        f"max={summary['translation_error_mm']['max']:.6f}"
    )
    print(
        "rotation error [deg]: "
        f"mean={summary['rotation_error_deg']['mean']:.6f}, "
        f"p95={summary['rotation_error_deg']['p95']:.6f}, "
        f"max={summary['rotation_error_deg']['max']:.6f}"
    )
    print("-" * 72)
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")
    for path in plot_paths:
        print(f"plot: {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
