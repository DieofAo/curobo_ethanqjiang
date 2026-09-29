#!/usr/bin/env python3
"""Solve Marvin arm IK from an MCAP pose stream and publish JointState."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_ARM = "left"
DEFAULT_ROBOT_CONFIG = "marvin_left_arm.yml"
DEFAULT_BASE_LINK = "robot_stand"
DEFAULT_EE_LINK = "Link7"
DEFAULT_RIGHT_OFFSET = "0 0 0.00"
DEFAULT_PUBLISH_TOPIC = "/joint_states"
DEFAULT_JOINT_NAMES = "Joint1 Joint2 Joint3 Joint4 Joint5 Joint6 Joint7"
ARM_DEFAULTS = {
    "left": {
        "mcap_path": "/retargeted/jaka_k1/left_arm/pose/state.pose",
        "urdf": "robot/marvin_description/urdf/left_marvin.urdf",
    },
    "right": {
        "mcap_path": "/retargeted/jaka_k1/right_arm/pose/state.pose",
        "urdf": "robot/marvin_description/urdf/right_marvin.urdf",
    },
}


@dataclass
class PoseSamples:
    timestamps_ns: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    requested_path: str
    channel_topic: str
    pose_field: str
    visible_topics: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read MCAP pose samples, right-multiply a local translation offset, solve IK for "
            "a Marvin arm base_link->Link7, and publish /joint_states."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arm",
        choices=sorted(ARM_DEFAULTS),
        default=DEFAULT_ARM,
        help="Marvin arm preset used to choose the default URDF and MCAP pose path.",
    )
    parser.add_argument("--mcap", type=Path, required=True, help="Input MCAP file.")
    parser.add_argument(
        "--mcap-path",
        default=None,
        help=(
            "MCAP topic or topic plus decoded field path. Defaults to the selected arm's "
            "/retargeted/jaka_k1/<arm>_arm/pose/state.pose."
        ),
    )
    parser.add_argument(
        "--pose-field",
        default=None,
        help="Override decoded ROS message field containing geometry_msgs/Pose.",
    )
    parser.add_argument("--robot-config", default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--base-link", default=None)
    parser.add_argument("--ee-link", default=None)
    parser.add_argument(
        "--joint-names",
        default=None,
        help="Space- or comma-separated active joint names in the URDF chain.",
    )
    parser.add_argument(
        "--right-offset",
        default=DEFAULT_RIGHT_OFFSET,
        help=(
            "Local xyz translation right-multiplied onto the MCAP pose. "
            "Use '0 0 -0.10' for 10cm along negative local z."
        ),
    )
    parser.add_argument("--publish-topic", default=DEFAULT_PUBLISH_TOPIC)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cuda:1, or cpu.")
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--local-seed-noise-std", type=float, default=0.05)
    parser.add_argument("--position-threshold", type=float, default=0.005)
    parser.add_argument("--rotation-threshold", type=float, default=0.05)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--loop",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Publish the solved JointState trajectory once and exit.",
    )
    parser.add_argument(
        "--max-poses",
        type=int,
        default=None,
        help="Only process the first N pose samples. Useful for quick IK checks.",
    )
    parser.add_argument(
        "--solve-only",
        action="store_true",
        help="Solve IK and print success statistics without publishing JointState.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read poses and print diagnostics without creating IK solver or publishing.",
    )
    args = parser.parse_args()
    apply_arm_defaults(args)
    args.loop = not args.no_loop
    return args


def apply_arm_defaults(args: argparse.Namespace) -> None:
    defaults = ARM_DEFAULTS[args.arm]
    if args.mcap_path is None:
        args.mcap_path = defaults["mcap_path"]
    if args.urdf is None:
        args.urdf = defaults["urdf"]
    if args.base_link is None:
        args.base_link = DEFAULT_BASE_LINK
    if args.ee_link is None:
        args.ee_link = DEFAULT_EE_LINK
    if args.joint_names is None:
        args.joint_names = DEFAULT_JOINT_NAMES


def parse_xyz(value: str) -> np.ndarray:
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"Expected three xyz values, got: {value!r}")
    return np.asarray([float(part) for part in parts], dtype=np.float64)


def parse_name_list(value: str) -> List[str]:
    names = [part.strip() for part in value.replace(",", " ").split()]
    if not names:
        raise ValueError("Expected at least one joint name")
    return names


def get_nested_attr(obj: Any, field_path: str) -> Any:
    value = obj
    for part in field_path.split("."):
        if not part:
            continue
        value = getattr(value, part)
    return value


def collect_mcap_topics(mcap_path: Path) -> List[str]:
    from mcap.reader import make_reader

    with mcap_path.open("rb") as file_obj:
        summary = make_reader(file_obj).get_summary()
    if summary is None:
        return []
    return sorted(channel.topic for channel in summary.channels.values())


def resolve_topic_and_pose_field(
    requested_path: str,
    visible_topics: Iterable[str],
    pose_field_override: Optional[str],
) -> Tuple[str, str]:
    topics = sorted(set(visible_topics))
    topic_set = set(topics)
    requested_path = requested_path.strip()

    if " plus " in requested_path:
        topic, suffix = requested_path.split(" plus ", 1)
        topic = topic.strip()
        if topic in topic_set:
            return topic, pose_field_override or suffix.strip().replace("/", ".")

    candidates = [
        topic
        for topic in topics
        if requested_path.startswith(topic + "/") or requested_path.startswith(topic + ".")
    ]
    if candidates:
        topic = max(candidates, key=len)
        suffix = requested_path[len(topic) + 1 :].replace("/", ".")
        return topic, pose_field_override or suffix

    if requested_path in topic_set:
        return requested_path, pose_field_override or "state.pose"

    visible = "\n  ".join(topics)
    raise ValueError(
        f"Could not resolve '{requested_path}' to an MCAP topic.\nVisible topics:\n  {visible}"
    )


def timestamp_ns(log_time: Any) -> int:
    if hasattr(log_time, "timestamp"):
        return int(log_time.timestamp() * 1e9)
    return int(log_time)


def normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise ValueError("Encountered zero-length quaternion")
    return quat / norm


def rotate_vector_xyzw(quat_xyzw: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    q_vec = np.asarray([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(q_vec, vector)
    return vector + w * t + np.cross(q_vec, t)


def right_multiply_translation(
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    offset_xyz: np.ndarray,
) -> np.ndarray:
    shifted = np.empty_like(positions, dtype=np.float64)
    for idx, (pos, quat) in enumerate(zip(positions, quaternions_xyzw)):
        shifted[idx] = pos + rotate_vector_xyzw(quat, offset_xyz)
    return shifted


def load_pose_samples(
    mcap_path: Path,
    requested_path: str,
    pose_field_override: Optional[str],
) -> PoseSamples:
    from mcap.reader import make_reader
    from mcap_ros1.decoder import DecoderFactory as Ros1DecoderFactory
    from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

    visible_topics = collect_mcap_topics(mcap_path)
    if not visible_topics:
        raise RuntimeError(f"No MCAP topics found in {mcap_path}")

    topic, pose_field = resolve_topic_and_pose_field(
        requested_path, visible_topics, pose_field_override
    )

    timestamps_ns: List[int] = []
    positions: List[List[float]] = []
    quaternions_xyzw: List[List[float]] = []

    with mcap_path.open("rb") as file_obj:
        reader = make_reader(
            file_obj,
            decoder_factories=[Ros1DecoderFactory(), Ros2DecoderFactory()],
        )
        for _, _, message, decoded_msg in reader.iter_decoded_messages(topics=[topic]):
            pose = get_nested_attr(decoded_msg, pose_field)
            p = pose.position
            q = pose.orientation
            timestamps_ns.append(timestamp_ns(message.log_time))
            positions.append([float(p.x), float(p.y), float(p.z)])
            quaternions_xyzw.append([float(q.x), float(q.y), float(q.z), float(q.w)])

    if not positions:
        raise RuntimeError(f"No pose samples found on topic={topic}, field={pose_field}")

    quats = np.asarray(quaternions_xyzw, dtype=np.float64)
    for idx in range(quats.shape[0]):
        quats[idx] = normalize_quaternion_xyzw(quats[idx])

    return PoseSamples(
        timestamps_ns=np.asarray(timestamps_ns, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.float64),
        quaternions_xyzw=quats,
        requested_path=requested_path,
        channel_topic=topic,
        pose_field=pose_field,
        visible_topics=visible_topics,
    )


def configure_torch(device_arg: str):
    import torch
    from curobo.types.base import TensorDeviceType

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device_arg == "auto":
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device_arg)

    if device.type != "cuda":
        raise RuntimeError("cuRobo IK requires CUDA for this script.")
    return torch, TensorDeviceType(device=device, dtype=torch.float32)


def resolve_yaml_path(name_or_path: str, config_dir: str) -> str:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return str(candidate)
    from curobo.util_file import join_path

    return join_path(config_dir, name_or_path)


def build_ik_solver(args: argparse.Namespace, tensor_args: Any):
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_configs_path, load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    cfg_path = resolve_yaml_path(args.robot_config, get_robot_configs_path())
    cfg = copy.deepcopy(load_yaml(cfg_path)["robot_cfg"])
    kinematics = cfg["kinematics"]
    kinematics["urdf_path"] = args.urdf
    if Path(args.urdf).is_absolute():
        kinematics["asset_root_path"] = str(Path(args.urdf).resolve().parents[1])
    else:
        kinematics["asset_root_path"] = "robot/marvin_description"
    kinematics["base_link"] = args.base_link
    kinematics["ee_link"] = args.ee_link
    kinematics["link_names"] = None
    kinematics["extra_links"] = None
    kinematics["collision_link_names"] = None
    kinematics["collision_spheres"] = None
    kinematics["self_collision_ignore"] = {}
    kinematics["self_collision_buffer"] = {}
    kinematics["mesh_link_names"] = None
    kinematics.setdefault("cspace", {})["joint_names"] = parse_name_list(args.joint_names)

    robot_cfg = RobotConfig.from_dict(cfg, tensor_args=tensor_args)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=args.rotation_threshold,
        position_threshold=args.position_threshold,
        num_seeds=args.num_seeds,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    solver = IKSolver(ik_config)
    return solver


def make_local_seed_config(torch: Any, prev_solution: Any, num_seeds: int, noise_std: float):
    if prev_solution is None:
        return None
    seeds = prev_solution.unsqueeze(0).repeat(num_seeds, 1)
    if num_seeds > 1 and noise_std > 0.0:
        seeds[1:] += torch.randn_like(seeds[1:]) * noise_std
    return seeds.unsqueeze(0)


def select_continuous_solution(torch: Any, result: Any, prev_solution: Any):
    candidates = result.solution[0]
    success = result.success[0].bool()

    if success.any():
        success_indices = torch.where(success)[0]
        success_solutions = candidates[success_indices]
        if prev_solution is None:
            selected_idx = success_indices[0]
        else:
            distances = torch.linalg.norm(success_solutions - prev_solution.unsqueeze(0), dim=-1)
            selected_idx = success_indices[torch.argmin(distances)]
        return candidates[selected_idx], True

    return candidates[0], False


def solve_ik_sequence(
    torch: Any,
    solver: Any,
    tensor_args: Any,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    num_seeds: int,
    seed_noise_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    from curobo.types.math import Pose

    joint_names = list(solver.joint_names)
    solutions = np.zeros((positions.shape[0], len(joint_names)), dtype=np.float32)
    success = np.zeros(positions.shape[0], dtype=bool)

    quats_wxyz = np.empty_like(quaternions_xyzw, dtype=np.float32)
    quats_wxyz[:, 0] = quaternions_xyzw[:, 3]
    quats_wxyz[:, 1] = quaternions_xyzw[:, 0]
    quats_wxyz[:, 2] = quaternions_xyzw[:, 1]
    quats_wxyz[:, 3] = quaternions_xyzw[:, 2]

    pos_tensor = torch.as_tensor(positions, device=tensor_args.device, dtype=tensor_args.dtype)
    quat_tensor = torch.as_tensor(quats_wxyz, device=tensor_args.device, dtype=tensor_args.dtype)

    prev_solution = None
    start_time = time.time()
    for idx in range(positions.shape[0]):
        goal = Pose(pos_tensor[idx : idx + 1], quat_tensor[idx : idx + 1])
        seed_config = make_local_seed_config(torch, prev_solution, num_seeds, seed_noise_std)
        result = solver.solve_batch(
            goal,
            seed_config=seed_config,
            return_seeds=num_seeds,
            num_seeds=num_seeds,
        )
        solution, frame_success = select_continuous_solution(torch, result, prev_solution)
        if frame_success:
            prev_solution = solution.clone()
        elif prev_solution is not None:
            solution = prev_solution

        solutions[idx] = solution.detach().cpu().numpy()
        success[idx] = frame_success

        if (idx + 1) % 100 == 0 or idx + 1 == positions.shape[0]:
            elapsed = time.time() - start_time
            print(
                f"  IK {idx + 1}/{positions.shape[0]} "
                f"success={int(success[:idx + 1].sum())}/{idx + 1} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return solutions, success


def sleep_for_mcap_dt(rospy: Any, dt_s: float, speed: float) -> None:
    if dt_s > 0.0 and dt_s < 10.0:
        rospy.sleep(dt_s / speed)


def publish_joint_states(
    joint_names: List[str],
    solutions: np.ndarray,
    timestamps_ns: np.ndarray,
    publish_topic: str,
    node_name: str,
    loop: bool,
    speed: float,
) -> None:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header

    rospy.init_node(node_name, anonymous=True)
    pub = rospy.Publisher(publish_topic, JointState, queue_size=10)
    rospy.sleep(0.5)

    dt_s = np.diff(timestamps_ns).astype(np.float64) / 1e9
    mean_hz = 1.0 / dt_s.mean() if dt_s.size and dt_s.mean() > 0 else 0.0
    print(f"Publishing {len(solutions)} frames to {publish_topic}")
    print(f"Joint names: {joint_names}")
    print(f"MCAP mean rate: {mean_hz:.2f} Hz, speed={speed}x, loop={loop}")
    print("Press Ctrl+C to stop.")

    try:
        loop_count = 0
        while not rospy.is_shutdown():
            loop_count += 1
            print(f"  publish loop {loop_count}")
            for idx, solution in enumerate(solutions):
                if rospy.is_shutdown():
                    break
                msg = JointState()
                msg.header = Header()
                msg.header.stamp = rospy.Time.now()
                msg.name = joint_names
                msg.position = solution.astype(float).tolist()
                msg.velocity = []
                msg.effort = []
                pub.publish(msg)

                if idx < len(solutions) - 1 and idx < len(dt_s):
                    sleep_for_mcap_dt(rospy, float(dt_s[idx]), speed)

            if not loop:
                break
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    if args.max_poses is not None and args.max_poses <= 0:
        raise ValueError("--max-poses must be positive")

    mcap_path = args.mcap.resolve()
    if not mcap_path.is_file():
        raise FileNotFoundError(f"MCAP file not found: {mcap_path}")

    right_offset = parse_xyz(args.right_offset)
    samples = load_pose_samples(mcap_path, args.mcap_path, args.pose_field)
    if args.max_poses is not None and args.max_poses < len(samples.timestamps_ns):
        samples = PoseSamples(
            timestamps_ns=samples.timestamps_ns[: args.max_poses],
            positions=samples.positions[: args.max_poses],
            quaternions_xyzw=samples.quaternions_xyzw[: args.max_poses],
            requested_path=samples.requested_path,
            channel_topic=samples.channel_topic,
            pose_field=samples.pose_field,
            visible_topics=samples.visible_topics,
        )
    shifted_positions = right_multiply_translation(
        samples.positions, samples.quaternions_xyzw, right_offset
    )

    print("=" * 72)
    print(f"{args.arm.title()} Marvin Link7 IK publisher")
    print(f"MCAP: {mcap_path}")
    print(f"resolved topic: {samples.channel_topic}")
    print(f"resolved pose field: {samples.pose_field}")
    print(f"samples: {len(samples.timestamps_ns)}")
    print(f"right-multiplied local translation offset xyz [m]: {right_offset.tolist()}")
    print(f"URDF: {args.urdf}")
    print(f"IK frame: {args.base_link} -> {args.ee_link}")
    print("=" * 72)

    if args.dry_run:
        print(f"first raw position: {samples.positions[0].tolist()}")
        print(f"first shifted position: {shifted_positions[0].tolist()}")
        print(f"first quaternion xyzw: {samples.quaternions_xyzw[0].tolist()}")
        return

    torch, tensor_args = configure_torch(args.device)
    solver = build_ik_solver(args, tensor_args)
    joint_names = list(solver.joint_names)
    print(f"solver joints: {joint_names}")

    solutions, success = solve_ik_sequence(
        torch,
        solver,
        tensor_args,
        shifted_positions.astype(np.float32),
        samples.quaternions_xyzw.astype(np.float32),
        args.num_seeds,
        args.local_seed_noise_std,
    )
    print(
        f"IK success: {int(success.sum())}/{len(success)} "
        f"({100.0 * float(success.mean()):.1f}%)"
    )
    if args.solve_only:
        return

    publish_joint_states(
        joint_names,
        solutions,
        samples.timestamps_ns,
        args.publish_topic,
        f"{args.arm}_marvin_link7_ik_publisher",
        args.loop,
        args.speed,
    )


if __name__ == "__main__":
    main()
