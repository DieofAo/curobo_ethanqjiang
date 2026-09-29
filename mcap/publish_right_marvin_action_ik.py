#!/usr/bin/env python3
"""Solve every right Marvin action pose with the calibrated transform and publish JointState."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from check_right_marvin_zero_obs_action_ik import (
    active_chain_joints,
    compute_fk_and_jacobian,
    find_joint_chain,
    joint_limits,
    load_urdf_joints,
    make_seed_bank,
    matrix_to_quat_xyzw,
    normalize_quaternion_xyzw,
    parse_float_list,
    parse_name_list,
    pose_error,
    pose_to_transform,
    solve_single_ik,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MCAP = REPO_ROOT / "mcap" / "data" / "20260702-191242_optimized.mcap"
DEFAULT_ACTION_TOPIC = "/robot/data/marvin_arm_right/action"
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "mcap" / "results" / "right_marvin_action_ik_publish"
DEFAULT_JOINT_NAMES = "Joint1 Joint2 Joint3 Joint4 Joint5 Joint6 Joint7"
DEFAULT_LEFT_TRANSLATION = "0 -0.061 0"
DEFAULT_LEFT_ROTATION_XYZW = "0.68304129 0.18291100 0.18291033 0.68303878"
DEFAULT_RIGHT_TRANSLATION = "0 -0.3202976292452259 0"
DEFAULT_RIGHT_ROTATION_XYZW = "0.5 -0.5 0.5 0.5"


@dataclass
class ActionSamples:
    timestamps_ns: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    targets: np.ndarray
    topic: str


@dataclass
class IkSequence:
    solutions: np.ndarray
    publish_solutions: np.ndarray
    success: np.ndarray
    position_errors_m: np.ndarray
    rotation_errors_rad: np.ndarray
    iterations: np.ndarray
    seed_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read /robot/data/marvin_arm_right/action poses, transform each pose as "
            "inv(T_left_parent_base) @ T_action @ inv(T_right_parent_base), solve "
            "right_marvin.urdf Base->Link7 IK without collision checking, and publish "
            "the resulting trajectory to /joint_states for RViz."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mcap", type=Path, default=DEFAULT_MCAP)
    parser.add_argument("--action-topic", default=DEFAULT_ACTION_TOPIC)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--base-link", default="Base")
    parser.add_argument("--ee-link", default="Link7")
    parser.add_argument("--joint-names", default=DEFAULT_JOINT_NAMES)
    parser.add_argument("--left-translation", default=DEFAULT_LEFT_TRANSLATION)
    parser.add_argument("--left-rotation-xyzw", default=DEFAULT_LEFT_ROTATION_XYZW)
    parser.add_argument("--right-translation", default=DEFAULT_RIGHT_TRANSLATION)
    parser.add_argument("--right-rotation-xyzw", default=DEFAULT_RIGHT_ROTATION_XYZW)
    parser.add_argument("--position-threshold", type=float, default=0.001)
    parser.add_argument("--rotation-threshold", type=float, default=0.01)
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=600)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--max-step-rad", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-poses", type=int, default=0, help="Use <=0 for all action poses.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--publish-topic", default="/joint_states")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--publish-rate-hz", type=float, default=0.0)
    parser.add_argument("--no-loop", action="store_true", help="Publish the trajectory once.")
    parser.add_argument(
        "--publish-failed-best",
        action="store_true",
        help="Publish each failed frame's best IK result instead of holding the previous success.",
    )
    parser.add_argument("--solve-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args()
    unexpected = [
        item
        for item in unknown
        if ":=" not in item and Path(item).name != "robot_state_publisher"
    ]
    if unexpected:
        parser.error(f"unrecognized arguments: {' '.join(unexpected)}")
    return args


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ translation)
    return result


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


def ros_pose_to_arrays(pose: Any) -> Tuple[List[float], List[float]]:
    position = [float(pose.position.x), float(pose.position.y), float(pose.position.z)]
    quaternion = [
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]
    return position, quaternion


def load_action_samples(
    mcap_path: Path,
    action_topic: str,
    left_transform: np.ndarray,
    right_transform: np.ndarray,
    max_poses: int,
) -> ActionSamples:
    from mcap.reader import make_reader
    from mcap_ros1.decoder import DecoderFactory

    inv_left = inverse_transform(left_transform)
    inv_right = inverse_transform(right_transform)
    timestamps_ns: List[int] = []
    positions: List[List[float]] = []
    quaternions_xyzw: List[List[float]] = []
    targets: List[np.ndarray] = []
    max_count = None if max_poses <= 0 else max_poses

    with mcap_path.open("rb") as file_obj:
        reader = make_reader(file_obj, decoder_factories=[DecoderFactory()])
        for _, _, message, decoded in reader.iter_decoded_messages(topics=[action_topic]):
            position, quaternion = ros_pose_to_arrays(decoded.pose_command.pose)
            quat = normalize_quaternion_xyzw(np.asarray(quaternion, dtype=np.float64))
            action_transform = pose_to_transform(np.asarray(position, dtype=np.float64), quat)
            target = inv_left @ action_transform @ inv_right

            timestamps_ns.append(int(message.log_time))
            positions.append(position)
            quaternions_xyzw.append(quat.tolist())
            targets.append(target)

            if max_count is not None and len(timestamps_ns) >= max_count:
                break

    if not timestamps_ns:
        raise RuntimeError(f"No action poses found on topic: {action_topic}")

    return ActionSamples(
        timestamps_ns=np.asarray(timestamps_ns, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.float64),
        quaternions_xyzw=np.asarray(quaternions_xyzw, dtype=np.float64),
        targets=np.stack(targets, axis=0),
        topic=action_topic,
    )


def build_ik_model(args: argparse.Namespace):
    joint_names = parse_name_list(args.joint_names)
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
    return joint_names, chain, joint_name_to_index, lower, upper


def solve_ik_sequence(
    samples: ActionSamples,
    joint_names: Sequence[str],
    chain: Sequence[Any],
    joint_name_to_index: Dict[str, int],
    lower: np.ndarray,
    upper: np.ndarray,
    args: argparse.Namespace,
) -> IkSequence:
    rng = np.random.default_rng(args.seed)
    solutions = np.zeros((samples.targets.shape[0], len(joint_names)), dtype=np.float64)
    publish_solutions = np.zeros_like(solutions)
    success = np.zeros(samples.targets.shape[0], dtype=bool)
    position_errors = np.zeros(samples.targets.shape[0], dtype=np.float64)
    rotation_errors = np.zeros(samples.targets.shape[0], dtype=np.float64)
    iterations = np.zeros(samples.targets.shape[0], dtype=np.int32)
    seed_indices = np.zeros(samples.targets.shape[0], dtype=np.int32)

    previous_success: Optional[np.ndarray] = None
    start_time = time.time()

    for index, target in enumerate(samples.targets):
        seeds = make_seed_bank(lower, upper, args.num_seeds, rng, previous_success)
        result = solve_single_ik(
            chain,
            joint_name_to_index,
            target,
            seeds,
            lower,
            upper,
            args.position_threshold,
            args.rotation_threshold,
            args.max_iterations,
            args.damping,
            args.max_step_rad,
        )
        solutions[index] = result.q
        success[index] = result.success
        position_errors[index] = result.position_error_m
        rotation_errors[index] = result.rotation_error_rad
        iterations[index] = result.iterations
        seed_indices[index] = result.seed_index

        if result.success:
            previous_success = result.q.copy()
            publish_solutions[index] = result.q
        elif previous_success is not None and not args.publish_failed_best:
            publish_solutions[index] = previous_success
        else:
            publish_solutions[index] = result.q

        if (index + 1) % 100 == 0 or index + 1 == samples.targets.shape[0]:
            elapsed = time.time() - start_time
            print(
                f"  IK {index + 1}/{samples.targets.shape[0]} "
                f"success={int(success[:index + 1].sum())}/{index + 1} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    return IkSequence(
        solutions=solutions.astype(np.float32),
        publish_solutions=publish_solutions.astype(np.float32),
        success=success,
        position_errors_m=position_errors,
        rotation_errors_rad=rotation_errors,
        iterations=iterations,
        seed_indices=seed_indices,
    )


def verify_publish_solutions(
    samples: ActionSamples,
    publish_solutions: np.ndarray,
    chain: Sequence[Any],
    joint_name_to_index: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    pos_errors = np.zeros(publish_solutions.shape[0], dtype=np.float64)
    rot_errors = np.zeros(publish_solutions.shape[0], dtype=np.float64)
    for index, q in enumerate(publish_solutions):
        fk, _ = compute_fk_and_jacobian(chain, joint_name_to_index, q.astype(np.float64))
        _, pos_error, rot_error = pose_error(fk, samples.targets[index])
        pos_errors[index] = pos_error
        rot_errors[index] = rot_error
    return pos_errors, rot_errors


def write_outputs(
    args: argparse.Namespace,
    samples: ActionSamples,
    sequence: IkSequence,
    joint_names: Sequence[str],
    publish_position_errors: np.ndarray,
    publish_rotation_errors: np.ndarray,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "right_marvin_action_ik_solutions.npz"
    np.savez_compressed(
        npz_path,
        timestamps_ns=samples.timestamps_ns,
        action_positions=samples.positions,
        action_quaternions_xyzw=samples.quaternions_xyzw,
        target_transforms=samples.targets,
        solutions=sequence.solutions,
        publish_solutions=sequence.publish_solutions,
        success=sequence.success,
        position_errors_m=sequence.position_errors_m,
        rotation_errors_rad=sequence.rotation_errors_rad,
        publish_position_errors_m=publish_position_errors,
        publish_rotation_errors_rad=publish_rotation_errors,
        iterations=sequence.iterations,
        seed_indices=sequence.seed_indices,
        joint_names=np.asarray(joint_names),
    )

    success = sequence.success
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mcap": str(args.mcap),
        "action_topic": samples.topic,
        "urdf": str(args.urdf),
        "base_link": args.base_link,
        "ee_link": args.ee_link,
        "joint_names": list(joint_names),
        "target_transform_definition": "T_goal = inverse(T_left) @ T_action @ inverse(T_right)",
        "T_left_parent_base": {
            "translation_xyz_m": parse_float_list(args.left_translation, 3, "--left-translation").tolist(),
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
        "total_count": int(success.size),
        "success_count": int(success.sum()),
        "success_rate": float(success.mean()),
        "position_error_m": summarize(sequence.position_errors_m),
        "rotation_error_rad": summarize(sequence.rotation_errors_rad),
        "rotation_error_deg": summarize(np.rad2deg(sequence.rotation_errors_rad)),
        "publish_position_error_m": summarize(publish_position_errors),
        "publish_rotation_error_rad": summarize(publish_rotation_errors),
        "publish_rotation_error_deg": summarize(np.rad2deg(publish_rotation_errors)),
        "npz": str(npz_path),
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print("-" * 72)
    print(
        f"IK success: {summary['success_count']}/{summary['total_count']} "
        f"({100.0 * summary['success_rate']:.2f}%)"
    )
    print(
        "IK position error [m]: "
        f"mean={summary['position_error_m']['mean']:.6g}, "
        f"p95={summary['position_error_m']['p95']:.6g}, "
        f"max={summary['position_error_m']['max']:.6g}"
    )
    print(
        "IK rotation error [deg]: "
        f"mean={summary['rotation_error_deg']['mean']:.6g}, "
        f"p95={summary['rotation_error_deg']['p95']:.6g}, "
        f"max={summary['rotation_error_deg']['max']:.6g}"
    )
    print(f"npz: {npz_path}")
    print(f"summary: {summary_path}")


def sleep_for_frame(rospy: Any, timestamps_ns: np.ndarray, index: int, speed: float, rate_hz: float) -> None:
    if rate_hz > 0.0:
        rospy.sleep(1.0 / rate_hz)
        return
    if index >= len(timestamps_ns) - 1:
        return
    dt_s = float(timestamps_ns[index + 1] - timestamps_ns[index]) / 1e9
    if 0.0 < dt_s < 10.0:
        rospy.sleep(dt_s / speed)


def publish_joint_states(
    joint_names: Sequence[str],
    solutions: np.ndarray,
    timestamps_ns: np.ndarray,
    publish_topic: str,
    speed: float,
    loop: bool,
    publish_rate_hz: float,
) -> None:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header

    rospy.init_node("right_marvin_action_ik_joint_state_publisher", anonymous=True)
    pub = rospy.Publisher(publish_topic, JointState, queue_size=10)
    rospy.sleep(0.5)

    if publish_rate_hz > 0.0:
        timing = f"fixed {publish_rate_hz:.2f} Hz"
    else:
        dt = np.diff(timestamps_ns).astype(np.float64) / 1e9
        mean_hz = 1.0 / float(dt.mean()) if dt.size and dt.mean() > 0.0 else 0.0
        timing = f"MCAP timing, mean {mean_hz:.2f} Hz, speed={speed:g}x"

    print(f"Publishing {len(solutions)} right Marvin frames to {publish_topic}")
    print(f"joint names: {list(joint_names)}")
    print(f"timing: {timing}, loop={loop}")

    try:
        loop_count = 0
        while not rospy.is_shutdown():
            loop_count += 1
            print(f"  publish loop {loop_count}", flush=True)
            for index, solution in enumerate(solutions):
                if rospy.is_shutdown():
                    break
                msg = JointState()
                msg.header = Header()
                msg.header.stamp = rospy.Time.now()
                msg.name = list(joint_names)
                msg.position = solution.astype(float).tolist()
                msg.velocity = []
                msg.effort = []
                pub.publish(msg)
                sleep_for_frame(rospy, timestamps_ns, index, speed, publish_rate_hz)
            if not loop:
                break
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass


def main() -> None:
    args = parse_args()
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive.")
    if args.publish_rate_hz < 0.0:
        raise ValueError("--publish-rate-hz must be non-negative.")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive.")
    if not args.mcap.is_file():
        raise FileNotFoundError(f"MCAP file not found: {args.mcap}")
    if not args.urdf.is_file():
        raise FileNotFoundError(f"URDF file not found: {args.urdf}")

    left_transform = pose_to_transform(
        parse_float_list(args.left_translation, 3, "--left-translation"),
        parse_float_list(args.left_rotation_xyzw, 4, "--left-rotation-xyzw"),
    )
    right_transform = pose_to_transform(
        parse_float_list(args.right_translation, 3, "--right-translation"),
        parse_float_list(args.right_rotation_xyzw, 4, "--right-rotation-xyzw"),
    )

    joint_names, chain, joint_name_to_index, lower, upper = build_ik_model(args)
    samples = load_action_samples(
        args.mcap,
        args.action_topic,
        left_transform,
        right_transform,
        args.max_poses,
    )

    print("=" * 72)
    print("Right Marvin action IK JointState publisher")
    print(f"MCAP: {args.mcap}")
    print(f"action topic: {samples.topic}")
    print(f"action poses: {len(samples.timestamps_ns)}")
    print(f"URDF chain: {args.base_link} -> {args.ee_link}")
    print(f"target transform: inverse(T_left) @ T_action @ inverse(T_right)")
    print("collision checking: disabled")
    print(f"joint names: {joint_names}")
    print("=" * 72)

    if args.dry_run:
        first_target_quat = matrix_to_quat_xyzw(samples.targets[0, :3, :3])
        print(f"first action position: {samples.positions[0].tolist()}")
        print(f"first action quaternion xyzw: {samples.quaternions_xyzw[0].tolist()}")
        print(f"first target position: {samples.targets[0, :3, 3].tolist()}")
        print(f"first target quaternion xyzw: {first_target_quat.tolist()}")
        return

    sequence = solve_ik_sequence(
        samples,
        joint_names,
        chain,
        joint_name_to_index,
        lower,
        upper,
        args,
    )
    publish_position_errors, publish_rotation_errors = verify_publish_solutions(
        samples,
        sequence.publish_solutions,
        chain,
        joint_name_to_index,
    )
    write_outputs(
        args,
        samples,
        sequence,
        joint_names,
        publish_position_errors,
        publish_rotation_errors,
    )

    if args.solve_only:
        return

    publish_joint_states(
        joint_names,
        sequence.publish_solutions,
        samples.timestamps_ns,
        args.publish_topic,
        args.speed,
        not args.no_loop,
        args.publish_rate_hz,
    )


if __name__ == "__main__":
    sys.exit(main())
