#!/usr/bin/env python3
"""Analyze relative end-effector offsets for calibrated and nominal xtrainer URDFs.

The calibrated URDF is treated as ground truth. This script runs three analyses:

1. FK effect: FK(q) is computed with both URDFs and the raw EE pose error is reported.
2. Relative FK effect: FK(q) is computed with both URDFs, the same relative EE
   transform is right-multiplied, and the resulting pose error is reported.
3. IK effect: each URDF solves IK for its own right-multiplied target using q as
   the seed/retract configuration. Both IK solutions are evaluated with the
   ground-truth URDF and compared in ground-truth EE pose space.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_MCAP_PATH = Path(__file__).resolve().parent / "data" / "20260618-184047_optimized.mcap"
DEFAULT_MCAP_STATE_PATH = "/robot/data/left_arm/observation/multibody_state/states"
DEFAULT_STATE_FIELD = "multibody_state.states"
DEFAULT_VALUE_FIELD = "q"


@dataclass
class QSamples:
    q: np.ndarray
    timestamps_ns: np.ndarray
    requested_path: str
    channel_topic: str
    state_field: str
    value_field: str
    visible_topics: List[str]


@dataclass
class RobotBundle:
    name: str
    solver: Any
    joint_names: List[str]


@dataclass
class JointLimitBundle:
    lower: np.ndarray
    upper: np.ndarray
    source: str
    gt_urdf_path: str
    nominal_urdf_path: str
    gt_lower: np.ndarray
    gt_upper: np.ndarray
    nominal_lower: np.ndarray
    nominal_upper: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze calibrated vs nominal xtrainer relative EE behavior with cuRobo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mcap", type=Path, default=DEFAULT_MCAP_PATH)
    parser.add_argument(
        "--mcap-path",
        default=DEFAULT_MCAP_STATE_PATH,
        help=(
            "MCAP channel or channel plus message field path. The default resolves to "
            "/robot/data/left_arm/observation plus multibody_state.states."
        ),
    )
    parser.add_argument(
        "--state-field",
        default=None,
        help="Override the decoded ROS message field that contains joint states.",
    )
    parser.add_argument(
        "--value-field",
        default=DEFAULT_VALUE_FIELD,
        help="Field read from each joint state, usually q.",
    )
    parser.add_argument("--robot-config", default="xtrainer.yml")
    parser.add_argument("--nominal-urdf", default="robot/ur_description/xtrainer.urdf")
    parser.add_argument("--gt-urdf", default="robot/ur_description/xtrainer_cali.urdf")
    parser.add_argument(
        "--world-config",
        default=None,
        help="Optional cuRobo world config yaml. If omitted, only robot self-collision is used.",
    )
    parser.add_argument(
        "--max-q",
        type=int,
        default=0,
        help="Maximum q samples to read. Use <=0 to read all q samples.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rel-samples-per-q", type=int, default=20)
    parser.add_argument(
        "--translation-range-m",
        type=float,
        default=0.2,
        help="Uniform relative translation bound for each xyz axis.",
    )
    parser.add_argument(
        "--rotation-range-deg",
        type=float,
        default=10.0,
        help="Uniform relative rotation angle bound, sampled around random axes.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cuda:1, or cpu.")
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ik-position-threshold", type=float, default=0.005)
    parser.add_argument("--ik-rotation-threshold", type=float, default=0.05)
    parser.add_argument(
        "--fk-sample-only",
        action="store_true",
        help="Skip MCAP/relative/IK analyses and run only joint-limit FK sampling.",
    )
    parser.add_argument(
        "--fk-sample-method",
        choices=["sobol", "random", "grid"],
        default="sobol",
        help="Joint-space sampling strategy for --fk-sample-only.",
    )
    parser.add_argument(
        "--fk-sample-count",
        type=int,
        default=200000,
        help="Number of q samples for sobol/random FK sampling.",
    )
    parser.add_argument(
        "--fk-sample-batch-size",
        type=int,
        default=65536,
        help="FK batch size for --fk-sample-only.",
    )
    parser.add_argument(
        "--fk-grid-step-rad",
        type=float,
        default=0.01,
        help="Per-joint step for exact Cartesian grid FK sampling.",
    )
    parser.add_argument(
        "--fk-grid-max-samples",
        type=int,
        default=1000000,
        help=(
            "Safety cap for exact grid samples. Use <=0 to disable the cap, "
            "but full 0.01 rad grids can be astronomically large."
        ),
    )
    parser.add_argument(
        "--fk-limit-source",
        choices=["intersection", "gt", "nominal"],
        default="intersection",
        help="Joint limits used by --fk-sample-only.",
    )
    parser.add_argument(
        "--fk-sample-top-k",
        type=int,
        default=20,
        help="Number of worst FK samples to keep in summary.json.",
    )
    parser.add_argument(
        "--fk-sample-write-csv",
        dest="fk_sample_write_csv",
        action="store_true",
        default=True,
        help="Write fk_sample_metrics.csv for --fk-sample-only.",
    )
    parser.add_argument(
        "--no-fk-sample-write-csv",
        dest="fk_sample_write_csv",
        action="store_false",
        help="Do not write per-sample FK metrics CSV.",
    )
    parser.add_argument("--skip-target3", action="store_true", help="Skip IK target 3.")
    parser.add_argument(
        "--skip-target2",
        action="store_true",
        dest="skip_target3",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run-mcap",
        action="store_true",
        help="Only read q samples and print MCAP/channel diagnostics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to relative_ee_analysis/results/<timestamp>.",
    )
    return parser.parse_args()


def get_nested_attr(obj: Any, field_path: str) -> Any:
    value = obj
    for part in field_path.split("."):
        if not part:
            continue
        value = getattr(value, part)
    return value


def normalize_state_field(field_path: str, value_field: str) -> str:
    field_path = field_path.strip().replace("/", ".").replace("[:]", "")
    suffix = f".{value_field}"
    if field_path.endswith(suffix):
        field_path = field_path[: -len(suffix)]
    return field_path


def collect_mcap_catalog(mcap_path: Path) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    from mcap.records import Channel, Schema
    from mcap.stream_reader import StreamReader

    schemas: Dict[int, Any] = {}
    channels: Dict[int, Any] = {}
    with mcap_path.open("rb") as file_obj:
        # The sample mcap stores schemas/channels in the summary section. emit_chunks=True
        # avoids expanding every image message while still scanning to the summary.
        for record in StreamReader(file_obj, emit_chunks=True, record_size_limit=None).records:
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                channels[record.id] = record
    return schemas, channels


def resolve_channel_and_state_field(
    requested_path: str,
    channels: Dict[int, Any],
    state_field_override: Optional[str],
    value_field: str,
) -> Tuple[Any, str]:
    topic_to_channel = {channel.topic: channel for channel in channels.values()}
    requested_path = requested_path.strip()
    if " plus " in requested_path:
        topic, suffix = requested_path.split(" plus ", 1)
        topic = topic.strip()
        if topic in topic_to_channel:
            return topic_to_channel[topic], normalize_state_field(
                state_field_override or suffix, value_field
            )

    if requested_path in topic_to_channel:
        return topic_to_channel[requested_path], normalize_state_field(
            state_field_override or DEFAULT_STATE_FIELD, value_field
        )

    candidates = [
        topic
        for topic in topic_to_channel
        if requested_path.startswith(topic + "/") and len(requested_path) > len(topic) + 1
    ]
    if candidates:
        topic = max(candidates, key=len)
        suffix = requested_path[len(topic) + 1 :].replace("/", ".")
        return topic_to_channel[topic], normalize_state_field(
            state_field_override or suffix, value_field
        )

    visible = "\n  ".join(sorted(topic_to_channel))
    raise ValueError(
        f"Could not resolve '{requested_path}' to an MCAP channel.\nVisible topics:\n  {visible}"
    )


def load_q_from_mcap(
    mcap_path: Path,
    requested_path: str,
    state_field_override: Optional[str],
    value_field: str,
    dof: int,
    max_q: int,
    start_index: int,
    stride: int,
) -> QSamples:
    from mcap.records import Message
    from mcap.stream_reader import StreamReader
    from mcap_ros1.decoder import DecoderFactory

    if stride <= 0:
        raise ValueError("--stride must be positive")
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")

    schemas, channels = collect_mcap_catalog(mcap_path)
    if not channels:
        raise RuntimeError(f"No MCAP channels found in {mcap_path}")

    channel, state_field = resolve_channel_and_state_field(
        requested_path, channels, state_field_override, value_field
    )
    schema = schemas.get(channel.schema_id)
    if schema is None:
        raise RuntimeError(f"Channel {channel.topic} references missing schema {channel.schema_id}")

    decoder = DecoderFactory().decoder_for(channel.message_encoding, schema)
    if decoder is None:
        raise RuntimeError(
            f"No ROS1 decoder for topic={channel.topic}, encoding={channel.message_encoding}, "
            f"schema={schema.name}/{schema.encoding}"
        )

    q_rows: List[List[float]] = []
    timestamps_ns: List[int] = []
    valid_seen = 0
    max_count = None if max_q <= 0 else max_q

    with mcap_path.open("rb") as file_obj:
        for record in StreamReader(file_obj, emit_chunks=False, record_size_limit=None).records:
            if not isinstance(record, Message) or record.channel_id != channel.id:
                continue

            decoded = decoder(record.data)
            states = get_nested_attr(decoded, state_field)
            if len(states) < dof:
                continue

            q = [float(getattr(state, value_field)) for state in states[:dof]]
            if valid_seen >= start_index and (valid_seen - start_index) % stride == 0:
                q_rows.append(q)
                timestamps_ns.append(int(record.log_time))
                if max_count is not None and len(q_rows) >= max_count:
                    break
            valid_seen += 1

    if not q_rows:
        visible = [channel.topic for channel in sorted(channels.values(), key=lambda c: c.id)]
        raise RuntimeError(
            "No q samples were extracted. Check --mcap-path, --state-field, "
            f"--start-index, and --stride. Visible topics: {visible}"
        )

    return QSamples(
        q=np.asarray(q_rows, dtype=np.float32),
        timestamps_ns=np.asarray(timestamps_ns, dtype=np.int64),
        requested_path=requested_path,
        channel_topic=channel.topic,
        state_field=state_field,
        value_field=value_field,
        visible_topics=[channel.topic for channel in sorted(channels.values(), key=lambda c: c.id)],
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def summarize_array(values: np.ndarray) -> Dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "min": float(np.min(finite)),
    }


def summarize_q(q: np.ndarray, joint_names: Iterable[str]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for idx, name in enumerate(joint_names):
        values = q[:, idx]
        summary[name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "start": float(values[0]),
            "end": float(values[-1]),
            "delta_end_start": float(values[-1] - values[0]),
        }
    return summary


def print_q_summary(q: np.ndarray, joint_names: Iterable[str]) -> None:
    print("q joint ranges:", flush=True)
    for name, stats in summarize_q(q, joint_names).items():
        print(
            f"  {name}: start={stats['start']:.6f}, end={stats['end']:.6f}, "
            f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
            f"range={stats['range']:.6f} rad",
            flush=True,
        )


def make_output_dir(output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path(__file__).resolve().parent / "results" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_q_csv(path: Path, q_samples: QSamples, joint_names: Iterable[str]) -> None:
    fieldnames = ["q_index", "timestamp_ns"] + list(joint_names)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (timestamp_ns, q) in enumerate(zip(q_samples.timestamps_ns, q_samples.q)):
            row = {"q_index": idx, "timestamp_ns": int(timestamp_ns)}
            row.update({name: float(value) for name, value in zip(joint_names, q)})
            writer.writerow(row)


def write_target1_csv(
    path: Path,
    q_indices: np.ndarray,
    timestamps_ns: np.ndarray,
    translation_m: np.ndarray,
    rotation_rad: np.ndarray,
) -> None:
    translation_m = array_1d(translation_m)
    rotation_rad = array_1d(rotation_rad)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "q_index",
                "timestamp_ns",
                "translation_error_m",
                "translation_error_mm",
                "rotation_error_rad",
                "rotation_error_deg",
            ],
        )
        writer.writeheader()
        for idx in range(len(q_indices)):
            writer.writerow(
                {
                    "q_index": int(q_indices[idx]),
                    "timestamp_ns": int(timestamps_ns[idx]),
                    "translation_error_m": float(translation_m[idx]),
                    "translation_error_mm": float(translation_m[idx] * 1000.0),
                    "rotation_error_rad": float(rotation_rad[idx]),
                    "rotation_error_deg": float(math.degrees(rotation_rad[idx])),
                }
            )


def write_target2_csv(
    path: Path,
    q_indices: np.ndarray,
    rel_indices: np.ndarray,
    translation_m: np.ndarray,
    rotation_rad: np.ndarray,
) -> None:
    translation_m = array_1d(translation_m)
    rotation_rad = array_1d(rotation_rad)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "sample_index",
                "q_index",
                "relative_index",
                "translation_error_m",
                "translation_error_mm",
                "rotation_error_rad",
                "rotation_error_deg",
            ],
        )
        writer.writeheader()
        for idx in range(len(q_indices)):
            writer.writerow(
                {
                    "sample_index": idx,
                    "q_index": int(q_indices[idx]),
                    "relative_index": int(rel_indices[idx]),
                    "translation_error_m": float(translation_m[idx]),
                    "translation_error_mm": float(translation_m[idx] * 1000.0),
                    "rotation_error_rad": float(rotation_rad[idx]),
                    "rotation_error_deg": float(math.degrees(rotation_rad[idx])),
                }
            )


def write_target3_csv(
    path: Path,
    q_indices: np.ndarray,
    rel_indices: np.ndarray,
    target3: Dict[str, np.ndarray],
) -> None:
    gt_eval_translation = array_1d(target3["gt_eval_translation_error_m"])
    gt_eval_rotation = array_1d(target3["gt_eval_rotation_error_rad"])
    gt_ik_success = array_1d(target3["gt_ik_success"]).astype(bool)
    nominal_ik_success = array_1d(target3["nominal_ik_success"]).astype(bool)
    both_ik_success = array_1d(target3["both_ik_success"]).astype(bool)
    gt_ik_position_error = array_1d(target3["gt_ik_position_error_m"])
    gt_ik_rotation_error = array_1d(target3["gt_ik_rotation_error_rad"])
    nominal_ik_position_error = array_1d(target3["nominal_ik_position_error_m"])
    nominal_ik_rotation_error = array_1d(target3["nominal_ik_rotation_error_rad"])
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "sample_index",
                "q_index",
                "relative_index",
                "gt_ik_success",
                "nominal_ik_success",
                "both_ik_success",
                "gt_eval_translation_error_m",
                "gt_eval_translation_error_mm",
                "gt_eval_rotation_error_rad",
                "gt_eval_rotation_error_deg",
                "gt_ik_position_error_m",
                "gt_ik_rotation_error_rad",
                "nominal_ik_position_error_m",
                "nominal_ik_rotation_error_rad",
            ],
        )
        writer.writeheader()
        for idx in range(len(q_indices)):
            rot = gt_eval_rotation[idx]
            trans = gt_eval_translation[idx]
            writer.writerow(
                {
                    "sample_index": idx,
                    "q_index": int(q_indices[idx]),
                    "relative_index": int(rel_indices[idx]),
                    "gt_ik_success": bool(gt_ik_success[idx]),
                    "nominal_ik_success": bool(nominal_ik_success[idx]),
                    "both_ik_success": bool(both_ik_success[idx]),
                    "gt_eval_translation_error_m": float(trans),
                    "gt_eval_translation_error_mm": float(trans * 1000.0),
                    "gt_eval_rotation_error_rad": float(rot),
                    "gt_eval_rotation_error_deg": float(math.degrees(rot))
                    if np.isfinite(rot)
                    else float("nan"),
                    "gt_ik_position_error_m": float(gt_ik_position_error[idx]),
                    "gt_ik_rotation_error_rad": float(gt_ik_rotation_error[idx]),
                    "nominal_ik_position_error_m": float(nominal_ik_position_error[idx]),
                    "nominal_ik_rotation_error_rad": float(nominal_ik_rotation_error[idx]),
                }
            )


def resolve_yaml_path(name_or_path: str, config_dir: str) -> str:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return str(candidate)
    from curobo.util_file import join_path

    return join_path(config_dir, name_or_path)


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
        raise RuntimeError(
            "cuRobo fused kinematics/IK requires a CUDA device for this analysis. "
            "Run on a machine/session where torch.cuda.is_available() is True, or pass "
            "--dry-run-mcap to inspect only the MCAP input."
        )
    return torch, TensorDeviceType(device=device, dtype=torch.float32)


def build_solver(
    name: str,
    robot_config_file: str,
    urdf_path: str,
    world_config_file: Optional[str],
    tensor_args: Any,
    args: argparse.Namespace,
) -> RobotBundle:
    from curobo.geom.types import WorldConfig
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_configs_path, get_world_configs_path, load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    cfg_path = resolve_yaml_path(robot_config_file, get_robot_configs_path())
    cfg = copy.deepcopy(load_yaml(cfg_path)["robot_cfg"])
    cfg["kinematics"]["urdf_path"] = urdf_path
    robot_cfg = RobotConfig.from_dict(cfg, tensor_args=tensor_args)

    world_cfg = None
    if world_config_file:
        world_path = resolve_yaml_path(world_config_file, get_world_configs_path())
        world_cfg = WorldConfig.from_dict(load_yaml(world_path))

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=args.ik_rotation_threshold,
        position_threshold=args.ik_position_threshold,
        num_seeds=args.num_seeds,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        seed=args.seed,
    )
    solver = IKSolver(ik_config)
    return RobotBundle(name=name, solver=solver, joint_names=list(solver.joint_names))


def resolve_asset_file(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    candidates: List[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(REPO_ROOT / candidate)
        candidates.append(SRC_ROOT / "curobo" / "content" / "assets" / candidate)
        try:
            from curobo.util_file import get_assets_path

            candidates.append(Path(get_assets_path()) / candidate)
        except Exception:
            pass

    for path in candidates:
        if path.is_file():
            return path.resolve()

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve asset file '{name_or_path}'. Searched:\n  {searched}")


def read_urdf_joint_limits(
    urdf_path: Path, joint_names: Iterable[str]
) -> Tuple[np.ndarray, np.ndarray]:
    joint_name_list = list(joint_names)
    wanted = set(joint_name_list)
    root = ET.parse(urdf_path).getroot()
    limits: Dict[str, Tuple[float, float]] = {}

    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in wanted:
            continue
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise RuntimeError(f"Joint {name} in {urdf_path} does not have lower/upper limits")
        limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))

    missing = [name for name in joint_name_list if name not in limits]
    if missing:
        raise RuntimeError(f"URDF {urdf_path} is missing limits for joints: {missing}")

    lower = np.asarray([limits[name][0] for name in joint_name_list], dtype=np.float32)
    upper = np.asarray([limits[name][1] for name in joint_name_list], dtype=np.float32)
    return lower, upper


def resolve_joint_limits(
    gt_urdf: str,
    nominal_urdf: str,
    joint_names: Iterable[str],
    source: str,
) -> JointLimitBundle:
    gt_urdf_path = resolve_asset_file(gt_urdf)
    nominal_urdf_path = resolve_asset_file(nominal_urdf)
    gt_lower, gt_upper = read_urdf_joint_limits(gt_urdf_path, joint_names)
    nominal_lower, nominal_upper = read_urdf_joint_limits(nominal_urdf_path, joint_names)

    if source == "intersection":
        lower = np.maximum(gt_lower, nominal_lower)
        upper = np.minimum(gt_upper, nominal_upper)
    elif source == "gt":
        lower = gt_lower.copy()
        upper = gt_upper.copy()
    elif source == "nominal":
        lower = nominal_lower.copy()
        upper = nominal_upper.copy()
    else:
        raise ValueError(f"Unknown joint limit source: {source}")

    invalid = np.where(lower >= upper)[0]
    if invalid.size:
        raise RuntimeError(
            f"Selected joint limits are empty for joint indices {invalid.tolist()}: "
            f"lower={lower.tolist()}, upper={upper.tolist()}"
        )

    return JointLimitBundle(
        lower=lower,
        upper=upper,
        source=source,
        gt_urdf_path=str(gt_urdf_path),
        nominal_urdf_path=str(nominal_urdf_path),
        gt_lower=gt_lower,
        gt_upper=gt_upper,
        nominal_lower=nominal_lower,
        nominal_upper=nominal_upper,
    )


def make_grid_axes(lower: np.ndarray, upper: np.ndarray, step: float) -> List[np.ndarray]:
    if step <= 0:
        raise ValueError("--fk-grid-step-rad must be positive")

    axes: List[np.ndarray] = []
    for lo, hi in zip(lower, upper):
        span = float(hi - lo)
        count = int(math.floor(span / step)) + 1
        values = float(lo) + np.arange(count, dtype=np.float64) * step
        values = values[values <= float(hi) + 1e-9]
        if values.size == 0 or values[-1] < float(hi) - 1e-7:
            values = np.append(values, float(hi))
        else:
            values[-1] = min(values[-1], float(hi))
        axes.append(values.astype(np.float32))
    return axes


def product_count(counts: Iterable[int]) -> int:
    total = 1
    for count in counts:
        total *= int(count)
    return total


def iter_grid_q_batches(axes: List[np.ndarray], batch_size: int):
    counts = np.asarray([axis.size for axis in axes], dtype=np.int64)
    total = product_count(counts)
    dof = len(axes)
    start = 0
    while start < total:
        end = min(start + batch_size, total)
        linear = np.arange(start, end, dtype=np.int64)
        q_batch = np.empty((end - start, dof), dtype=np.float32)
        remainder = linear.copy()
        for joint_idx in range(dof - 1, -1, -1):
            axis_indices = remainder % counts[joint_idx]
            remainder //= counts[joint_idx]
            q_batch[:, joint_idx] = axes[joint_idx][axis_indices]
        yield start, q_batch
        start = end


def iter_random_q_batches(
    lower: np.ndarray,
    upper: np.ndarray,
    sample_count: int,
    batch_size: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    span = upper - lower
    start = 0
    while start < sample_count:
        count = min(batch_size, sample_count - start)
        unit = rng.random((count, lower.size), dtype=np.float32)
        yield start, lower + unit * span
        start += count


def iter_sobol_q_batches(
    torch: Any,
    lower: np.ndarray,
    upper: np.ndarray,
    sample_count: int,
    batch_size: int,
    seed: int,
):
    engine = torch.quasirandom.SobolEngine(dimension=int(lower.size), scramble=True, seed=seed)
    span = upper - lower
    start = 0
    while start < sample_count:
        count = min(batch_size, sample_count - start)
        unit = engine.draw(count).cpu().numpy().astype(np.float32)
        yield start, lower + unit * span
        start += count


def fk_sample_entry(
    sample_index: int,
    q: np.ndarray,
    translation_m: float,
    rotation_rad: float,
    gt_radius_m: float,
    nominal_radius_m: float,
    joint_names: Iterable[str],
) -> Dict[str, Any]:
    return {
        "sample_index": int(sample_index),
        "gt_ee_radius_m": float(gt_radius_m),
        "gt_ee_radius_mm": float(gt_radius_m * 1000.0),
        "nominal_ee_radius_m": float(nominal_radius_m),
        "nominal_ee_radius_mm": float(nominal_radius_m * 1000.0),
        "translation_error_m": float(translation_m),
        "translation_error_mm": float(translation_m * 1000.0),
        "rotation_error_rad": float(rotation_rad),
        "rotation_error_deg": float(math.degrees(rotation_rad)),
        "q": {name: float(value) for name, value in zip(joint_names, q)},
    }


def update_top_fk_samples(
    top_samples: List[Dict[str, Any]],
    sort_key: str,
    q_batch: np.ndarray,
    translation_m: np.ndarray,
    rotation_rad: np.ndarray,
    gt_radius_m: np.ndarray,
    nominal_radius_m: np.ndarray,
    sample_start: int,
    joint_names: Iterable[str],
    top_k: int,
) -> None:
    if top_k <= 0 or q_batch.size == 0:
        return

    values = translation_m if sort_key == "translation_error_m" else rotation_rad
    candidate_count = min(top_k, values.size)
    if values.size > candidate_count:
        candidate_indices = np.argpartition(values, -candidate_count)[-candidate_count:]
    else:
        candidate_indices = np.arange(values.size)

    for idx in candidate_indices:
        top_samples.append(
            fk_sample_entry(
                sample_start + int(idx),
                q_batch[idx],
                float(translation_m[idx]),
                float(rotation_rad[idx]),
                float(gt_radius_m[idx]),
                float(nominal_radius_m[idx]),
                joint_names,
            )
        )

    top_samples.sort(key=lambda item: item[sort_key], reverse=True)
    del top_samples[top_k:]


def run_fk_difference_batch(
    gt: RobotBundle,
    nominal: RobotBundle,
    q_tensor: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from curobo.types.math import Pose

    gt_fk = gt.solver.fk(q_tensor)
    nominal_fk = nominal.solver.fk(q_tensor)
    gt_pose = Pose(gt_fk.ee_position, gt_fk.ee_quaternion, normalize_rotation=False)
    nominal_pose = Pose(
        nominal_fk.ee_position,
        nominal_fk.ee_quaternion,
        normalize_rotation=False,
    )
    translation_error, rotation_error = gt_pose.distance(nominal_pose)
    gt_radius = gt_fk.ee_position.norm(dim=-1)
    nominal_radius = nominal_fk.ee_position.norm(dim=-1)
    return (
        tensor_to_numpy_1d(translation_error),
        tensor_to_numpy_1d(rotation_error),
        tensor_to_numpy_1d(gt_radius),
        tensor_to_numpy_1d(nominal_radius),
    )


def joint_limit_summary(
    joint_names: Iterable[str],
    lower: np.ndarray,
    upper: np.ndarray,
    gt_lower: np.ndarray,
    gt_upper: np.ndarray,
    nominal_lower: np.ndarray,
    nominal_upper: np.ndarray,
) -> Dict[str, Any]:
    summary = {}
    for idx, name in enumerate(joint_names):
        summary[name] = {
            "sample_lower": float(lower[idx]),
            "sample_upper": float(upper[idx]),
            "sample_range": float(upper[idx] - lower[idx]),
            "gt_lower": float(gt_lower[idx]),
            "gt_upper": float(gt_upper[idx]),
            "nominal_lower": float(nominal_lower[idx]),
            "nominal_upper": float(nominal_upper[idx]),
        }
    return summary


def metric_stats_row(name: str, unit: str, values: np.ndarray) -> Dict[str, Any]:
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return {
            "metric": name,
            "unit": unit,
            "count": 0,
            "mean": "",
            "max": "",
            "max_sample_index": "",
            "median": "",
            "p95": "",
            "min": "",
        }

    finite_indices = np.where(finite_mask)[0]
    local_max_idx = int(np.argmax(finite_values))
    return {
        "metric": name,
        "unit": unit,
        "count": int(finite_values.size),
        "mean": float(np.mean(finite_values)),
        "max": float(np.max(finite_values)),
        "max_sample_index": int(finite_indices[local_max_idx]),
        "median": float(np.median(finite_values)),
        "p95": float(np.percentile(finite_values, 95)),
        "min": float(np.min(finite_values)),
    }


def write_fk_sample_stats_csv(
    path: Path,
    translation_m: np.ndarray,
    rotation_rad: np.ndarray,
    gt_radius_m: np.ndarray,
    nominal_radius_m: np.ndarray,
) -> None:
    rows = [
        metric_stats_row("translation_error", "mm", translation_m * 1000.0),
        metric_stats_row("rotation_error", "deg", np.degrees(rotation_rad)),
        metric_stats_row("gt_ee_radius", "mm", gt_radius_m * 1000.0),
        metric_stats_row("nominal_ee_radius", "mm", nominal_radius_m * 1000.0),
    ]
    fieldnames = ["metric", "unit", "count", "mean", "max", "max_sample_index", "median", "p95", "min"]
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_fk_sampling_experiment(
    torch: Any,
    tensor_args: Any,
    gt: RobotBundle,
    nominal: RobotBundle,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    if args.fk_sample_batch_size <= 0:
        raise ValueError("--fk-sample-batch-size must be positive")
    if args.fk_sample_top_k < 0:
        raise ValueError("--fk-sample-top-k must be non-negative")

    limits = resolve_joint_limits(
        args.gt_urdf,
        args.nominal_urdf,
        gt.joint_names,
        args.fk_limit_source,
    )
    lower = limits.lower.astype(np.float32)
    upper = limits.upper.astype(np.float32)

    grid_metadata: Dict[str, Any] = {}
    if args.fk_sample_method == "grid":
        axes = make_grid_axes(lower, upper, args.fk_grid_step_rad)
        axis_counts = [int(axis.size) for axis in axes]
        total_samples = product_count(axis_counts)
        grid_metadata = {
            "step_rad": args.fk_grid_step_rad,
            "axis_counts": {name: count for name, count in zip(gt.joint_names, axis_counts)},
            "total_cartesian_samples": int(total_samples),
            "max_samples_cap": int(args.fk_grid_max_samples),
        }
        if args.fk_grid_max_samples > 0 and total_samples > args.fk_grid_max_samples:
            raise RuntimeError(
                f"Exact grid would create {total_samples:,} q samples, exceeding "
                f"--fk-grid-max-samples={args.fk_grid_max_samples:,}. "
                "Use --fk-sample-method sobol/random, increase the grid step, or raise the cap."
            )
        q_batches = iter_grid_q_batches(axes, args.fk_sample_batch_size)
    else:
        if args.fk_sample_count <= 0:
            raise ValueError("--fk-sample-count must be positive for sobol/random sampling")
        total_samples = int(args.fk_sample_count)
        if args.fk_sample_method == "sobol":
            q_batches = iter_sobol_q_batches(
                torch, lower, upper, total_samples, args.fk_sample_batch_size, args.seed
            )
        elif args.fk_sample_method == "random":
            q_batches = iter_random_q_batches(
                lower, upper, total_samples, args.fk_sample_batch_size, args.seed
            )
        else:
            raise ValueError(f"Unknown FK sample method: {args.fk_sample_method}")

    print(
        f"Running FK sampling: method={args.fk_sample_method}, samples={total_samples:,}, "
        f"limit_source={args.fk_limit_source}",
        flush=True,
    )

    csv_file = None
    csv_writer = None
    if args.fk_sample_write_csv:
        csv_file = (output_dir / "fk_sample_metrics.csv").open("w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["sample_index"]
            + gt.joint_names
            + [
                "translation_error_m",
                "translation_error_mm",
                "rotation_error_rad",
                "rotation_error_deg",
            ]
        )

    translation_chunks: List[np.ndarray] = []
    rotation_chunks: List[np.ndarray] = []
    top_translation: List[Dict[str, Any]] = []
    top_rotation: List[Dict[str, Any]] = []
    seen = 0
    next_report = 0
    report_every = max(total_samples // 20, 1)

    try:
        for sample_start, q_batch in q_batches:
            q_tensor = torch.as_tensor(
                q_batch, device=tensor_args.device, dtype=tensor_args.dtype
            )
            translation_m, rotation_rad = run_fk_difference_batch(gt, nominal, q_tensor)
            translation_m = translation_m.astype(np.float32, copy=False)
            rotation_rad = rotation_rad.astype(np.float32, copy=False)

            translation_chunks.append(translation_m)
            rotation_chunks.append(rotation_rad)
            update_top_fk_samples(
                top_translation,
                "translation_error_m",
                q_batch,
                translation_m,
                rotation_rad,
                sample_start,
                gt.joint_names,
                args.fk_sample_top_k,
            )
            update_top_fk_samples(
                top_rotation,
                "rotation_error_rad",
                q_batch,
                translation_m,
                rotation_rad,
                sample_start,
                gt.joint_names,
                args.fk_sample_top_k,
            )

            if csv_writer is not None:
                for local_idx, q in enumerate(q_batch):
                    trans = float(translation_m[local_idx])
                    rot = float(rotation_rad[local_idx])
                    csv_writer.writerow(
                        [sample_start + local_idx]
                        + [float(value) for value in q]
                        + [trans, trans * 1000.0, rot, math.degrees(rot)]
                    )

            seen += q_batch.shape[0]
            if seen >= next_report or seen == total_samples:
                print(f"  FK sampled {seen:,}/{total_samples:,}", flush=True)
                next_report = seen + report_every
    finally:
        if csv_file is not None:
            csv_file.close()

    if torch.cuda.is_available() and tensor_args.device.type == "cuda":
        torch.cuda.synchronize()

    translation_all = np.concatenate(translation_chunks, axis=0)
    rotation_all = np.concatenate(rotation_chunks, axis=0)

    np.savez_compressed(
        output_dir / "fk_sample_results.npz",
        translation_error_m=translation_all,
        rotation_error_rad=rotation_all,
        joint_limit_lower=lower,
        joint_limit_upper=upper,
    )

    summary: Dict[str, Any] = {
        "mode": "fk_sample_only",
        "inputs": {
            "robot_config": args.robot_config,
            "gt_urdf": args.gt_urdf,
            "nominal_urdf": args.nominal_urdf,
            "gt_urdf_path": limits.gt_urdf_path,
            "nominal_urdf_path": limits.nominal_urdf_path,
            "device": str(tensor_args.device),
            "seed": args.seed,
            "sample_method": args.fk_sample_method,
            "sample_count": int(total_samples),
            "sample_batch_size": int(args.fk_sample_batch_size),
            "limit_source": limits.source,
            "write_csv": bool(args.fk_sample_write_csv),
        },
        "robot": {"joint_names": gt.joint_names},
        "joint_limits": joint_limit_summary(
            gt.joint_names,
            lower,
            upper,
            limits.gt_lower,
            limits.gt_upper,
            limits.nominal_lower,
            limits.nominal_upper,
        ),
        "translation_error_m": summarize_array(translation_all),
        "translation_error_mm": summarize_array(translation_all * 1000.0),
        "rotation_error_rad": summarize_array(rotation_all),
        "rotation_error_deg": summarize_array(np.degrees(rotation_all)),
        "top_translation_error": top_translation,
        "top_rotation_error": top_rotation,
    }
    if grid_metadata:
        summary["grid"] = grid_metadata

    with (output_dir / "summary.json").open("w") as file_obj:
        json.dump(jsonable(summary), file_obj, indent=2, sort_keys=True)

    print(f"Wrote FK sampling results to {output_dir}", flush=True)
    print(
        "FK sampling max: "
        f"{summary['translation_error_mm']['max']:.3f} mm, "
        f"{summary['rotation_error_deg']['max']:.4f} deg",
        flush=True,
    )


def sample_relative_poses(
    torch: Any,
    tensor_args: Any,
    n_q: int,
    rel_samples_per_q: int,
    translation_range_m: float,
    rotation_range_deg: float,
    seed: int,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from curobo.types.math import Pose

    if rel_samples_per_q <= 0:
        raise ValueError("--rel-samples-per-q must be positive")

    rng = np.random.default_rng(seed)
    total = n_q * rel_samples_per_q
    rel_pos = rng.uniform(-translation_range_m, translation_range_m, size=(total, 3)).astype(
        np.float32
    )

    axes = rng.normal(size=(total, 3)).astype(np.float32)
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / np.maximum(norms, 1e-8)
    angles = rng.uniform(
        -math.radians(rotation_range_deg), math.radians(rotation_range_deg), size=(total, 1)
    ).astype(np.float32)
    half_angles = 0.5 * angles
    rel_quat = np.concatenate([np.cos(half_angles), axes * np.sin(half_angles)], axis=1).astype(
        np.float32
    )

    q_indices = np.repeat(np.arange(n_q, dtype=np.int64), rel_samples_per_q)
    rel_indices = np.tile(np.arange(rel_samples_per_q, dtype=np.int64), n_q)

    rel_pose = Pose(
        torch.as_tensor(rel_pos, device=tensor_args.device, dtype=tensor_args.dtype),
        torch.as_tensor(rel_quat, device=tensor_args.device, dtype=tensor_args.dtype),
    )
    return rel_pose, rel_pos, rel_quat, q_indices, rel_indices


def pose_from_fk(fk_state: Any, q_indices_t: Any) -> Any:
    from curobo.types.math import Pose

    return Pose(
        fk_state.ee_position[q_indices_t],
        fk_state.ee_quaternion[q_indices_t],
        normalize_rotation=False,
    )


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def tensor_to_numpy_1d(tensor: Any) -> np.ndarray:
    return np.asarray(tensor_to_numpy(tensor)).reshape(-1)


def array_1d(array: np.ndarray) -> np.ndarray:
    return np.asarray(array).reshape(-1)


def run_target1(
    torch: Any,
    gt: RobotBundle,
    nominal: RobotBundle,
    q_tensor: Any,
) -> Dict[str, np.ndarray]:
    from curobo.types.math import Pose

    gt_fk = gt.solver.fk(q_tensor)
    nominal_fk = nominal.solver.fk(q_tensor)
    gt_pose = Pose(gt_fk.ee_position, gt_fk.ee_quaternion, normalize_rotation=False)
    nominal_pose = Pose(
        nominal_fk.ee_position,
        nominal_fk.ee_quaternion,
        normalize_rotation=False,
    )
    translation_error, rotation_error = gt_pose.distance(nominal_pose)

    return {
        "gt_ee_position": tensor_to_numpy(gt_pose.position),
        "gt_ee_quaternion_wxyz": tensor_to_numpy(gt_pose.quaternion),
        "nominal_ee_position": tensor_to_numpy(nominal_pose.position),
        "nominal_ee_quaternion_wxyz": tensor_to_numpy(nominal_pose.quaternion),
        "translation_error_m": tensor_to_numpy_1d(translation_error),
        "rotation_error_rad": tensor_to_numpy_1d(rotation_error),
    }


def run_target2(
    torch: Any,
    gt: RobotBundle,
    nominal: RobotBundle,
    q_tensor: Any,
    rel_pose: Any,
    q_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    from curobo.types.math import Pose

    q_indices_t = torch.as_tensor(q_indices, device=q_tensor.device, dtype=torch.long)
    gt_fk = gt.solver.fk(q_tensor)
    nominal_fk = nominal.solver.fk(q_tensor)
    gt_pose = pose_from_fk(gt_fk, q_indices_t)
    nominal_pose = pose_from_fk(nominal_fk, q_indices_t)

    gt_target = gt_pose.multiply(rel_pose)
    nominal_target = nominal_pose.multiply(rel_pose)
    translation_error, rotation_error = gt_target.distance(nominal_target)

    return {
        "gt_target_position": tensor_to_numpy(gt_target.position),
        "gt_target_quaternion_wxyz": tensor_to_numpy(gt_target.quaternion),
        "nominal_target_position": tensor_to_numpy(nominal_target.position),
        "nominal_target_quaternion_wxyz": tensor_to_numpy(nominal_target.quaternion),
        "translation_error_m": tensor_to_numpy_1d(translation_error),
        "rotation_error_rad": tensor_to_numpy_1d(rotation_error),
    }


def solve_ik_batches(
    torch: Any,
    solver: Any,
    goals: Any,
    seed_q: Any,
    batch_size: int,
    label: str,
) -> Dict[str, np.ndarray]:
    from curobo.types.math import Pose

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    total = goals.position.shape[0]
    q_solutions: List[np.ndarray] = []
    success_values: List[np.ndarray] = []
    pos_errors: List[np.ndarray] = []
    rot_errors: List[np.ndarray] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        goal_batch = Pose(
            goals.position[start:end].contiguous(),
            goals.quaternion[start:end].contiguous(),
            normalize_rotation=False,
        )
        q_batch = seed_q[start:end].contiguous()
        seed_config = q_batch.unsqueeze(1)
        result = solver.solve_batch(
            goal_batch,
            retract_config=q_batch,
            seed_config=seed_config,
            use_nn_seed=False,
        )
        q_solutions.append(tensor_to_numpy(result.solution[:, 0, :]))
        success_values.append(tensor_to_numpy_1d(result.success[:, 0]).astype(bool))
        pos_errors.append(tensor_to_numpy_1d(result.position_error[:, 0]))
        rot_errors.append(tensor_to_numpy_1d(result.rotation_error[:, 0]))
        print(
            f"  {label}: solved [{start}:{end}] "
            f"success={int(success_values[-1].sum())}/{end - start}",
            flush=True,
        )

    if torch.cuda.is_available() and seed_q.device.type == "cuda":
        torch.cuda.synchronize()

    return {
        "solution": np.concatenate(q_solutions, axis=0),
        "success": np.concatenate(success_values, axis=0),
        "position_error_m": np.concatenate(pos_errors, axis=0),
        "rotation_error_rad": np.concatenate(rot_errors, axis=0),
    }


def run_target3(
    torch: Any,
    gt: RobotBundle,
    nominal: RobotBundle,
    q_tensor: Any,
    q_indices: np.ndarray,
    target2: Dict[str, np.ndarray],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    from curobo.types.math import Pose

    q_indices_t = torch.as_tensor(q_indices, device=q_tensor.device, dtype=torch.long)
    seed_q = q_tensor[q_indices_t].contiguous()

    gt_goals = Pose(
        torch.as_tensor(
            target2["gt_target_position"], device=q_tensor.device, dtype=q_tensor.dtype
        ),
        torch.as_tensor(
            target2["gt_target_quaternion_wxyz"], device=q_tensor.device, dtype=q_tensor.dtype
        ),
        normalize_rotation=False,
    )
    nominal_goals = Pose(
        torch.as_tensor(
            target2["nominal_target_position"], device=q_tensor.device, dtype=q_tensor.dtype
        ),
        torch.as_tensor(
            target2["nominal_target_quaternion_wxyz"],
            device=q_tensor.device,
            dtype=q_tensor.dtype,
        ),
        normalize_rotation=False,
    )

    print("Solving target 3 IK with ground-truth URDF...", flush=True)
    gt_ik = solve_ik_batches(torch, gt.solver, gt_goals, seed_q, batch_size, "gt IK")
    print("Solving target 3 IK with nominal URDF...", flush=True)
    nominal_ik = solve_ik_batches(
        torch, nominal.solver, nominal_goals, seed_q, batch_size, "nominal IK"
    )

    total = len(q_indices)
    both_success = gt_ik["success"] & nominal_ik["success"]
    gt_eval_translation = np.full(total, np.nan, dtype=np.float32)
    gt_eval_rotation = np.full(total, np.nan, dtype=np.float32)

    success_idx = np.where(both_success)[0]
    if success_idx.size > 0:
        gt_q = torch.as_tensor(
            gt_ik["solution"][success_idx], device=q_tensor.device, dtype=q_tensor.dtype
        )
        nominal_q = torch.as_tensor(
            nominal_ik["solution"][success_idx], device=q_tensor.device, dtype=q_tensor.dtype
        )
        gt_fk = gt.solver.fk(gt_q)
        nominal_on_gt_fk = gt.solver.fk(nominal_q)
        gt_pose = Pose(gt_fk.ee_position, gt_fk.ee_quaternion, normalize_rotation=False)
        nominal_on_gt_pose = Pose(
            nominal_on_gt_fk.ee_position,
            nominal_on_gt_fk.ee_quaternion,
            normalize_rotation=False,
        )
        trans, rot = gt_pose.distance(nominal_on_gt_pose)
        gt_eval_translation[success_idx] = tensor_to_numpy_1d(trans).astype(np.float32)
        gt_eval_rotation[success_idx] = tensor_to_numpy_1d(rot).astype(np.float32)

    return {
        "gt_ik_solution": gt_ik["solution"],
        "nominal_ik_solution": nominal_ik["solution"],
        "gt_ik_success": gt_ik["success"],
        "nominal_ik_success": nominal_ik["success"],
        "both_ik_success": both_success,
        "gt_ik_position_error_m": gt_ik["position_error_m"],
        "gt_ik_rotation_error_rad": gt_ik["rotation_error_rad"],
        "nominal_ik_position_error_m": nominal_ik["position_error_m"],
        "nominal_ik_rotation_error_rad": nominal_ik["rotation_error_rad"],
        "gt_eval_translation_error_m": gt_eval_translation,
        "gt_eval_rotation_error_rad": gt_eval_rotation,
    }


def main() -> None:
    args = parse_args()
    output_dir = make_output_dir(args.output_dir)

    if args.fk_sample_only:
        torch, tensor_args = configure_torch(args.device)
        gt = build_solver(
            "ground_truth",
            args.robot_config,
            args.gt_urdf,
            args.world_config,
            tensor_args,
            args,
        )
        nominal = build_solver(
            "nominal",
            args.robot_config,
            args.nominal_urdf,
            args.world_config,
            tensor_args,
            args,
        )
        if gt.joint_names != nominal.joint_names:
            raise RuntimeError(
                f"Joint name mismatch: gt={gt.joint_names}, nominal={nominal.joint_names}"
            )
        run_fk_sampling_experiment(torch, tensor_args, gt, nominal, args, output_dir)
        return

    # The mcap states have 6 arm joints. The cuRobo config is loaded later and checked.
    q_samples = load_q_from_mcap(
        mcap_path=args.mcap,
        requested_path=args.mcap_path,
        state_field_override=args.state_field,
        value_field=args.value_field,
        dof=6,
        max_q=args.max_q,
        start_index=args.start_index,
        stride=args.stride,
    )
    print(
        f"Loaded q samples: {len(q_samples.q)} from channel={q_samples.channel_topic}, "
        f"field={q_samples.state_field}",
        flush=True,
    )

    if args.dry_run_mcap:
        print("Visible MCAP topics:")
        for topic in q_samples.visible_topics:
            print(f"  {topic}")
        print(f"First q: {q_samples.q[0].tolist()}")
        dry_joint_names = [f"q{i+1}" for i in range(q_samples.q.shape[1])]
        print_q_summary(q_samples.q, dry_joint_names)
        write_q_csv(output_dir / "q_samples.csv", q_samples, dry_joint_names)
        print(f"Wrote dry-run q_samples.csv to {output_dir}")
        return

    torch, tensor_args = configure_torch(args.device)
    gt = build_solver(
        "ground_truth",
        args.robot_config,
        args.gt_urdf,
        args.world_config,
        tensor_args,
        args,
    )
    nominal = build_solver(
        "nominal",
        args.robot_config,
        args.nominal_urdf,
        args.world_config,
        tensor_args,
        args,
    )

    if gt.joint_names != nominal.joint_names:
        raise RuntimeError(f"Joint name mismatch: gt={gt.joint_names}, nominal={nominal.joint_names}")
    if q_samples.q.shape[1] != len(gt.joint_names):
        raise RuntimeError(
            f"q dimension {q_samples.q.shape[1]} does not match cuRobo DOF {len(gt.joint_names)}"
        )

    q_tensor = torch.as_tensor(q_samples.q, device=tensor_args.device, dtype=tensor_args.dtype)
    rel_pose, rel_pos, rel_quat, q_indices, rel_indices = sample_relative_poses(
        torch,
        tensor_args,
        n_q=len(q_samples.q),
        rel_samples_per_q=args.rel_samples_per_q,
        translation_range_m=args.translation_range_m,
        rotation_range_deg=args.rotation_range_deg,
        seed=args.seed,
    )

    print(f"Running target 1: raw FK q={len(q_samples.q)}", flush=True)
    target1 = run_target1(torch, gt, nominal, q_tensor)

    print(
        f"Running target 2: relative FK q={len(q_samples.q)}, "
        f"rel_per_q={args.rel_samples_per_q}, total={len(q_indices)}",
        flush=True,
    )
    target2 = run_target2(torch, gt, nominal, q_tensor, rel_pose, q_indices)

    target3: Dict[str, np.ndarray] = {}
    if not args.skip_target3:
        target3 = run_target3(torch, gt, nominal, q_tensor, q_indices, target2, args.batch_size)

    write_q_csv(output_dir / "q_samples.csv", q_samples, gt.joint_names)
    write_target1_csv(
        output_dir / "target1_metrics.csv",
        np.arange(len(q_samples.q), dtype=np.int64),
        q_samples.timestamps_ns,
        target1["translation_error_m"],
        target1["rotation_error_rad"],
    )
    write_target2_csv(
        output_dir / "target2_metrics.csv",
        q_indices,
        rel_indices,
        target2["translation_error_m"],
        target2["rotation_error_rad"],
    )
    if target3:
        write_target3_csv(output_dir / "target3_metrics.csv", q_indices, rel_indices, target3)

    npz_payload: Dict[str, Any] = {
        "q": q_samples.q,
        "timestamps_ns": q_samples.timestamps_ns,
        "relative_position": rel_pos,
        "relative_quaternion_wxyz": rel_quat,
        "q_indices": q_indices,
        "relative_indices": rel_indices,
        "target1_translation_error_m": target1["translation_error_m"],
        "target1_rotation_error_rad": target1["rotation_error_rad"],
        "target1_gt_ee_position": target1["gt_ee_position"],
        "target1_gt_ee_quaternion_wxyz": target1["gt_ee_quaternion_wxyz"],
        "target1_nominal_ee_position": target1["nominal_ee_position"],
        "target1_nominal_ee_quaternion_wxyz": target1["nominal_ee_quaternion_wxyz"],
        "target2_translation_error_m": target2["translation_error_m"],
        "target2_rotation_error_rad": target2["rotation_error_rad"],
        "target2_gt_target_position": target2["gt_target_position"],
        "target2_gt_target_quaternion_wxyz": target2["gt_target_quaternion_wxyz"],
        "target2_nominal_target_position": target2["nominal_target_position"],
        "target2_nominal_target_quaternion_wxyz": target2["nominal_target_quaternion_wxyz"],
    }
    for key, value in target3.items():
        npz_payload[f"target3_{key}"] = value
    np.savez_compressed(output_dir / "results.npz", **npz_payload)

    summary: Dict[str, Any] = {
        "inputs": {
            "mcap": args.mcap,
            "requested_mcap_path": q_samples.requested_path,
            "resolved_channel_topic": q_samples.channel_topic,
            "resolved_state_field": q_samples.state_field,
            "value_field": q_samples.value_field,
            "robot_config": args.robot_config,
            "gt_urdf": args.gt_urdf,
            "nominal_urdf": args.nominal_urdf,
            "world_config": args.world_config,
            "device": str(tensor_args.device),
            "num_seeds": args.num_seeds,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "relative_translation_range_m_per_axis": args.translation_range_m,
            "relative_rotation_range_deg": args.rotation_range_deg,
        },
        "counts": {
            "q_samples": int(len(q_samples.q)),
            "relative_samples_per_q": int(args.rel_samples_per_q),
            "total_relative_targets": int(len(q_indices)),
        },
        "robot": {"joint_names": gt.joint_names},
        "q": summarize_q(q_samples.q, gt.joint_names),
        "target1": {
            "translation_error_m": summarize_array(target1["translation_error_m"]),
            "translation_error_mm": summarize_array(target1["translation_error_m"] * 1000.0),
            "rotation_error_rad": summarize_array(target1["rotation_error_rad"]),
            "rotation_error_deg": summarize_array(
                np.degrees(target1["rotation_error_rad"])
            ),
        },
    }
    if target2:
        summary["target2"] = {
            "translation_error_m": summarize_array(target2["translation_error_m"]),
            "translation_error_mm": summarize_array(target2["translation_error_m"] * 1000.0),
            "rotation_error_rad": summarize_array(target2["rotation_error_rad"]),
            "rotation_error_deg": summarize_array(
                np.degrees(target2["rotation_error_rad"])
            ),
        }
    if target3:
        summary["target3"] = {
            "gt_ik_success_rate": float(np.mean(target3["gt_ik_success"])),
            "nominal_ik_success_rate": float(np.mean(target3["nominal_ik_success"])),
            "both_ik_success_rate": float(np.mean(target3["both_ik_success"])),
            "gt_eval_translation_error_m": summarize_array(
                target3["gt_eval_translation_error_m"]
            ),
            "gt_eval_translation_error_mm": summarize_array(
                target3["gt_eval_translation_error_m"] * 1000.0
            ),
            "gt_eval_rotation_error_rad": summarize_array(
                target3["gt_eval_rotation_error_rad"]
            ),
            "gt_eval_rotation_error_deg": summarize_array(
                np.degrees(target3["gt_eval_rotation_error_rad"])
            ),
            "gt_ik_position_error_m": summarize_array(target3["gt_ik_position_error_m"]),
            "gt_ik_rotation_error_rad": summarize_array(target3["gt_ik_rotation_error_rad"]),
            "nominal_ik_position_error_m": summarize_array(
                target3["nominal_ik_position_error_m"]
            ),
            "nominal_ik_rotation_error_rad": summarize_array(
                target3["nominal_ik_rotation_error_rad"]
            ),
        }

    with (output_dir / "summary.json").open("w") as file_obj:
        json.dump(jsonable(summary), file_obj, indent=2, sort_keys=True)

    print(f"Wrote results to {output_dir}", flush=True)
    print(
        "Target 1 mean: "
        f"{summary['target1']['translation_error_mm']['mean']:.3f} mm, "
        f"{summary['target1']['rotation_error_deg']['mean']:.4f} deg",
        flush=True,
    )
    if target2:
        target2_summary = summary["target2"]
        print(
            "Target 2 mean: "
            f"{target2_summary['translation_error_mm']['mean']:.3f} mm, "
            f"{target2_summary['rotation_error_deg']['mean']:.4f} deg",
            flush=True,
        )
    if target3:
        target3_summary = summary["target3"]
        print(
            "Target 3 both-success rate: "
            f"{target3_summary['both_ik_success_rate']:.3f}, "
            "mean gt-eval error: "
            f"{target3_summary['gt_eval_translation_error_mm'].get('mean', float('nan')):.3f} mm, "
            f"{target3_summary['gt_eval_rotation_error_deg'].get('mean', float('nan')):.4f} deg",
            flush=True,
        )


if __name__ == "__main__":
    main()
