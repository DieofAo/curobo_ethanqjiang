#!/usr/bin/env python3
"""Generate workspace-limited, self-collision-free JAKA left-arm joint samples."""

from __future__ import annotations

import argparse
import copy
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.rollout.cost.self_collision_cost import SelfCollisionCost, SelfCollisionCostConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "jaka" / "left_jaka.urdf"
DEFAULT_ASSET_ROOT = REPO_ROOT / "jaka"
DEFAULT_OUTPUT = REPO_ROOT / "jaka" / "left_jaka_collision_free_q_20000.txt"
JOINT_NAMES = ["J_1", "J_2", "J_3", "J_4", "J_5", "J_6", "J_7"]


def parse_joint_limits_from_urdf(urdf_path: Path, joint_names: list[str]) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    limits_by_name: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in joint_names:
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"Joint {name} has no lower/upper limit in {urdf_path}")
        limits_by_name[name] = (float(limit.get("lower")), float(limit.get("upper")))

    missing = [name for name in joint_names if name not in limits_by_name]
    if missing:
        raise ValueError(f"Missing joints in {urdf_path}: {missing}")

    return np.asarray([limits_by_name[name] for name in joint_names], dtype=np.float32)


def load_left_jaka_robot_config(
    robot_file: str,
    urdf_path: Path,
    asset_root_path: Path,
    tensor_args: TensorDeviceType,
) -> RobotConfig:
    cfg = copy.deepcopy(load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"])
    cfg["kinematics"]["urdf_path"] = os.fspath(urdf_path)
    cfg["kinematics"]["asset_root_path"] = os.fspath(asset_root_path)
    return RobotConfig.from_dict(cfg, tensor_args=tensor_args)


def make_self_collision_cost(
    kinematics: CudaRobotModel,
    tensor_args: TensorDeviceType,
) -> SelfCollisionCost:
    config = SelfCollisionCostConfig(
        weight=tensor_args.to_device([1.0]),
        tensor_args=tensor_args,
        return_loss=True,
        self_collision_kin_config=kinematics.get_self_collision_config(),
    )
    return SelfCollisionCost(config)


def format_ee_workspace(args: argparse.Namespace) -> str:
    parts: list[str] = []
    bounds = (
        ("x", args.ee_min_x, args.ee_max_x),
        ("y", args.ee_min_y, args.ee_max_y),
        ("z", args.ee_min_z, args.ee_max_z),
    )
    for axis, lower, upper in bounds:
        if lower is not None:
            parts.append(f"{axis}>{lower:g}m")
        if upper is not None:
            parts.append(f"{axis}<{upper:g}m")
    return ", ".join(parts) if parts else "unbounded"


def validate_ee_workspace(args: argparse.Namespace) -> None:
    for axis in ("x", "y", "z"):
        lower = getattr(args, f"ee_min_{axis}")
        upper = getattr(args, f"ee_max_{axis}")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"Invalid EE workspace bound: ee-min-{axis} must be < ee-max-{axis}")


def parse_link_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_sphere_workspace_mask(
    kinematics: CudaRobotModel,
    args: argparse.Namespace,
) -> torch.Tensor | None:
    if not args.require_spheres_in_workspace:
        return None

    link_sphere_idx_map = kinematics.kinematics_config.link_sphere_idx_map
    if link_sphere_idx_map is None:
        raise ValueError("Robot config does not provide collision spheres for sphere workspace filtering.")

    keep_mask = kinematics.kinematics_config.link_spheres[:, 3] > 0.0
    ignored_links = parse_link_list(args.sphere_workspace_ignore_links)
    for link_name in ignored_links:
        if link_name not in kinematics.kinematics_config.link_name_to_idx_map:
            raise ValueError(f"Unknown link in --sphere-workspace-ignore-links: {link_name}")
        link_idx = kinematics.kinematics_config.link_name_to_idx_map[link_name]
        keep_mask &= link_sphere_idx_map != link_idx

    if not torch.any(keep_mask):
        raise ValueError("No collision spheres remain after applying --sphere-workspace-ignore-links.")
    return keep_mask


def filter_ee_workspace(ee_position: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    mask = torch.ones(ee_position.shape[:-1], device=ee_position.device, dtype=torch.bool)
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    for axis, idx in axis_to_index.items():
        lower = getattr(args, f"ee_min_{axis}")
        upper = getattr(args, f"ee_max_{axis}")
        if lower is not None:
            mask &= ee_position[..., idx] > lower
        if upper is not None:
            mask &= ee_position[..., idx] < upper
    return mask


@torch.inference_mode()
def filter_valid_samples(
    q: torch.Tensor,
    kinematics: CudaRobotModel,
    cost_fn: SelfCollisionCost,
    args: argparse.Namespace,
    sphere_workspace_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = kinematics.get_state(q)
    spheres = state.link_spheres_tensor.view(q.shape[0], 1, -1, 4).contiguous()
    collision_cost = cost_fn.forward(spheres).view(-1)
    valid_mask = collision_cost <= args.collision_eps
    valid_mask &= filter_ee_workspace(state.ee_position, args)
    if sphere_workspace_mask is not None:
        sphere_centers = state.link_spheres_tensor[:, sphere_workspace_mask, :3]
        valid_mask &= filter_ee_workspace(sphere_centers, args).all(dim=-1)
    return valid_mask, state.ee_position


def generate_samples(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    validate_ee_workspace(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuRobo self-collision checking.")

    tensor_args = TensorDeviceType(device=torch.device("cuda", args.cuda_device), dtype=torch.float32)
    robot_cfg = load_left_jaka_robot_config(
        args.robot_file,
        args.urdf_path,
        args.asset_root_path,
        tensor_args,
    )
    kinematics = CudaRobotModel(robot_cfg.kinematics)
    cost_fn = make_self_collision_cost(kinematics, tensor_args)
    sphere_workspace_mask = make_sphere_workspace_mask(kinematics, args)

    urdf_limits = parse_joint_limits_from_urdf(args.urdf_path, JOINT_NAMES)
    low = torch.as_tensor(urdf_limits[:, 0], device=tensor_args.device, dtype=tensor_args.dtype)
    high = torch.as_tensor(urdf_limits[:, 1], device=tensor_args.device, dtype=tensor_args.dtype)

    rng = np.random.default_rng(args.seed)
    accepted: list[np.ndarray] = []
    accepted_ee: list[np.ndarray] = []
    accepted_count = 0
    tried_count = 0
    start_time = time.time()

    while accepted_count < args.num_samples:
        random_batch = rng.random((args.batch_size, len(JOINT_NAMES)), dtype=np.float32)
        q = low + torch.as_tensor(random_batch, device=tensor_args.device) * (high - low)
        valid_mask, ee_position = filter_valid_samples(
            q,
            kinematics,
            cost_fn,
            args,
            sphere_workspace_mask,
        )
        valid_q = q[valid_mask]

        if valid_q.numel() > 0:
            valid_np = valid_q.detach().cpu().numpy()
            valid_ee_np = ee_position[valid_mask].detach().cpu().numpy()
            remaining = args.num_samples - accepted_count
            selected_count = min(valid_np.shape[0], remaining)
            accepted.append(valid_np[:selected_count])
            accepted_ee.append(valid_ee_np[:selected_count])
            accepted_count += selected_count

        tried_count += args.batch_size
        if args.verbose:
            elapsed = time.time() - start_time
            rate = 100.0 * accepted_count / max(tried_count, 1)
            print(
                f"accepted {accepted_count}/{args.num_samples} "
                f"from {tried_count} candidates ({rate:.2f}%), {elapsed:.1f}s",
                flush=True,
            )

        if tried_count >= args.max_candidates and accepted_count < args.num_samples:
            raise RuntimeError(
                f"Only found {accepted_count} valid samples after {tried_count} candidates. "
                "Increase --max-candidates or relax workspace/collision settings."
            )

    samples = np.concatenate(accepted, axis=0)
    ee_positions = np.concatenate(accepted_ee, axis=0)
    return samples[: args.num_samples], ee_positions[: args.num_samples]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-file", default="jaka.yml")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--asset-root-path", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--max-candidates", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--collision-eps", type=float, default=0.0)
    parser.add_argument("--ee-min-x", type=float, default=0.0, help="Strict lower EE x bound in m.")
    parser.add_argument("--ee-max-x", type=float, default=None, help="Strict upper EE x bound in m.")
    parser.add_argument("--ee-min-y", type=float, default=0.0, help="Strict lower EE y bound in m.")
    parser.add_argument("--ee-max-y", type=float, default=0.8, help="Strict upper EE y bound in m.")
    parser.add_argument("--ee-min-z", type=float, default=None, help="Strict lower EE z bound in m.")
    parser.add_argument("--ee-max-z", type=float, default=0.0, help="Strict upper EE z bound in m.")
    parser.add_argument(
        "--require-spheres-in-workspace",
        action="store_true",
        help="Also require selected collision sphere centers to satisfy the workspace bounds.",
    )
    parser.add_argument(
        "--sphere-workspace-ignore-links",
        default="LINK_BASE,LINK_1",
        help=(
            "Comma-separated links ignored by --require-spheres-in-workspace. "
            "Use an empty string to include every collision sphere."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    samples, ee_positions = generate_samples(args)
    header = (
        " ".join(JOINT_NAMES)
        + f"  # radians, self-collision-free by cuRobo; EE workspace: {format_ee_workspace(args)}"
    )
    np.savetxt(args.output, samples, fmt="%.9f", header=header)

    limits = parse_joint_limits_from_urdf(args.urdf_path, JOINT_NAMES)
    print(f"wrote {samples.shape[0]} samples to {args.output}")
    print("joint order:", " ".join(JOINT_NAMES))
    print("min:", " ".join(f"{v:.6f}" for v in samples.min(axis=0)))
    print("max:", " ".join(f"{v:.6f}" for v in samples.max(axis=0)))
    print("urdf lower:", " ".join(f"{v:.6f}" for v in limits[:, 0]))
    print("urdf upper:", " ".join(f"{v:.6f}" for v in limits[:, 1]))
    print("ee workspace:", format_ee_workspace(args))
    if args.require_spheres_in_workspace:
        ignored = args.sphere_workspace_ignore_links or "none"
        print("sphere workspace:", f"enabled, ignored links: {ignored}")
    else:
        print("sphere workspace: disabled")
    print("ee min xyz:", " ".join(f"{v:.6f}" for v in ee_positions.min(axis=0)))
    print("ee max xyz:", " ".join(f"{v:.6f}" for v in ee_positions.max(axis=0)))


if __name__ == "__main__":
    main()
