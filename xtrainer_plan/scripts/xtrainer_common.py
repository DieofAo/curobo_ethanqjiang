#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XTrainer 轨迹规划任务的公共工具:
  - rpy(度, ROS 固定轴 XYZ 外旋) <-> 四元数(wxyz / xyzw) 互转
  - 任务 yaml 的加载与深层合并
  - 工作空间 6 面墙 cuboid 的构造(含 base 让位孔)
  - 轨迹 npz/json 的读写约定

不依赖 curobo / rospy, 两个环境(conda py3.11 与 ROS py3.8)都能 import。
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parents[1]


# ============================== 旋转约定 ==============================
# 全仓库统一: rpy 单位为「度」, 约定为 ROS/URDF 标准固定轴 XYZ 外旋,
#            即 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。
# curobo Pose 的四元数顺序是 (w, x, y, z); ROS geometry_msgs 是 (x, y, z, w)。


def rpy_deg_to_matrix(rpy_deg: Sequence[float]) -> np.ndarray:
    """rpy(度, 固定轴 XYZ 外旋) -> 3x3 旋转矩阵。R = Rz @ Ry @ Rx"""
    r, p, y = (math.radians(float(v)) for v in rpy_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def matrix_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> 四元数 (w, x, y, z), 数值稳定的 Shepperd 分支法。"""
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:  # 规范化到 w >= 0, 避免同一姿态的双重表示
        q = -q
    return q


def quat_wxyz_to_matrix(q_wxyz: Sequence[float]) -> np.ndarray:
    """四元数 (w, x, y, z) -> 3x3 旋转矩阵。"""
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rpy_deg_to_quat_wxyz(rpy_deg: Sequence[float]) -> np.ndarray:
    """rpy(度, 固定轴 XYZ) -> 四元数 (w, x, y, z), curobo 顺序。"""
    return matrix_to_quat_wxyz(rpy_deg_to_matrix(rpy_deg))


def quat_wxyz_to_xyzw(q_wxyz: Sequence[float]) -> np.ndarray:
    """curobo (w,x,y,z) -> ROS (x,y,z,w)。"""
    w, x, y, z = (float(v) for v in q_wxyz)
    return np.array([x, y, z, w], dtype=np.float64)


def quat_xyzw_to_wxyz(q_xyzw: Sequence[float]) -> np.ndarray:
    """ROS (x,y,z,w) -> curobo (w,x,y,z)。"""
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array([w, x, y, z], dtype=np.float64)


def matrix_to_rpy_deg(m: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> rpy(度, 固定轴 XYZ 外旋)。R = Rz @ Ry @ Rx 的反解。"""
    m = np.asarray(m, dtype=np.float64)
    sp = -m[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = math.asin(sp)
    if abs(sp) > 1.0 - 1e-9:  # 万向锁
        roll = 0.0
        yaw = math.atan2(-m[0, 1], m[1, 1]) if sp > 0 else math.atan2(m[0, 1], m[1, 1])
    else:
        roll = math.atan2(m[2, 1], m[2, 2])
        yaw = math.atan2(m[1, 0], m[0, 0])
    return np.degrees([roll, pitch, yaw])


def quat_wxyz_to_rpy_deg(q_wxyz: Sequence[float]) -> np.ndarray:
    """四元数 (w,x,y,z) -> rpy(度, 固定轴 XYZ)。"""
    return matrix_to_rpy_deg(quat_wxyz_to_matrix(q_wxyz))


def quat_angle_deg(q_a_wxyz: Sequence[float], q_b_wxyz: Sequence[float]) -> float:
    """两个四元数之间的最小旋转夹角(度)。"""
    a = np.asarray(q_a_wxyz, dtype=np.float64)
    b = np.asarray(q_b_wxyz, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    d = float(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(d))


def tool_z_axis(q_wxyz: Sequence[float]) -> np.ndarray:
    """工具坐标系 +Z 在 base 系下的方向向量(旋转矩阵第 3 列)。"""
    return quat_wxyz_to_matrix(q_wxyz)[:, 2]


def parse_rigid_transform_matrix(
    value: Any,
    name: str = "transform",
) -> np.ndarray:
    """严格解析一个刚体变换，返回齐次 ``float64`` 4x4 矩阵。

    推荐配置格式使用位置向量和轴角旋转::

        position: [x, y, z]
        rotation:
          axis: [ax, ay, az]  # 也可写成 x / y / z
          angle_deg: 90.0

    ``translation`` 可作为 ``position`` 的兼容别名；若二者同时出现，
    必须表示相同向量。轴向量会自动归一化，并且即使角度为零也必须非零，
    以便尽早发现配置笔误。为兼容已有配置，也继续接受齐次 4x4 数字矩阵。
    矩阵旋转块必须为右手正交矩阵，齐次末行必须为
    ``[0, 0, 0, 1]``。本函数不对近似旋转做投影或静默修正，避免
    配置错误改变目标位姿。
    """
    if isinstance(value, dict):
        return _parse_axis_angle_transform(value, name)

    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 4x4 数字矩阵") from exc
    if transform.shape != (4, 4):
        raise ValueError(
            f"{name} 必须是 4x4 矩阵，得到 shape={transform.shape}"
        )
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} 必须只包含有限数字")
    if not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-9
    ):
        raise ValueError(f"{name} 末行必须是 [0, 0, 0, 1]")

    rotation = transform[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6
    ):
        raise ValueError(f"{name} 的 3x3 旋转块必须正交")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"{name} 的 3x3 旋转块行列式必须为 +1，得到 {determinant:.9g}"
        )
    return transform.copy()


def _parse_axis_angle_transform(value: Dict[str, Any], name: str) -> np.ndarray:
    """解析 ``position + rotation(axis, angle_deg)`` 形式的刚体变换。"""
    allowed_fields = {"position", "translation", "rotation"}
    unknown_fields = set(value) - allowed_fields
    if unknown_fields:
        fields = ", ".join(
            repr(field) for field in sorted(unknown_fields, key=str)
        )
        raise ValueError(f"{name} 包含未知字段: {fields}")
    if "rotation" not in value:
        raise ValueError(f"{name} 缺少必填字段 rotation")
    if "position" not in value and "translation" not in value:
        raise ValueError(f"{name} 缺少必填字段 position")

    rotation_cfg = value["rotation"]
    if not isinstance(rotation_cfg, dict):
        raise ValueError(f"{name}.rotation 必须是 mapping")
    allowed_rotation_fields = {"axis", "angle_deg"}
    unknown_rotation_fields = set(rotation_cfg) - allowed_rotation_fields
    if unknown_rotation_fields:
        fields = ", ".join(
            repr(field)
            for field in sorted(unknown_rotation_fields, key=str)
        )
        raise ValueError(f"{name}.rotation 包含未知字段: {fields}")
    missing_rotation_fields = allowed_rotation_fields - set(rotation_cfg)
    if missing_rotation_fields:
        fields = ", ".join(sorted(missing_rotation_fields))
        raise ValueError(f"{name}.rotation 缺少必填字段: {fields}")

    position = (
        _parse_finite_vector3(value.get("position"), f"{name}.position")
        if "position" in value
        else None
    )
    translation = (
        _parse_finite_vector3(value.get("translation"), f"{name}.translation")
        if "translation" in value
        else None
    )
    if position is not None and translation is not None:
        if not np.array_equal(position, translation):
            raise ValueError(
                f"{name}.position 和兼容别名 {name}.translation 必须相同"
            )
    offset = position if position is not None else translation
    assert offset is not None  # 上方已检查两个位置字段至少存在一个。

    raw_axis = rotation_cfg["axis"]
    if isinstance(raw_axis, str):
        axis_name = raw_axis.strip().lower()
        named_axes = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        if axis_name not in named_axes:
            raise ValueError(
                f"{name}.rotation.axis 必须是 x/y/z 或三维非零向量"
            )
        axis = named_axes[axis_name]
    else:
        axis = _parse_finite_vector3(raw_axis, f"{name}.rotation.axis")
        scale = float(np.max(np.abs(axis)))
        if scale == 0.0:
            raise ValueError(
                f"{name}.rotation.axis 必须是非零向量（angle_deg 为 0 时也必须指定）"
            )
        # 先缩放再求范数，避免有限但很大的轴向量在平方求和时溢出。
        scaled_axis = axis / scale
        axis = scaled_axis / np.linalg.norm(scaled_axis)

    raw_angle = rotation_cfg["angle_deg"]
    if isinstance(
        raw_angle,
        (bool, np.bool_, str, bytes, list, tuple, dict, np.ndarray),
    ):
        raise ValueError(f"{name}.rotation.angle_deg 必须是有限标量")
    try:
        angle_deg = float(raw_angle)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}.rotation.angle_deg 必须是有限标量") from exc
    if not math.isfinite(angle_deg):
        raise ValueError(f"{name}.rotation.angle_deg 必须是有限标量")

    angle_rad = math.radians(angle_deg)
    x, y, z = axis
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    sine = math.sin(angle_rad)
    cosine = math.cos(angle_rad)
    rotation = (
        np.eye(3, dtype=np.float64)
        + sine * skew
        + (1.0 - cosine) * (skew @ skew)
    )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = offset
    return transform


def _parse_finite_vector3(value: Any, name: str) -> np.ndarray:
    """严格解析有限三维数字向量。"""
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3,):
        raise ValueError(f"{name} 必须是三维有限数字向量")
    if any(
        isinstance(item, (bool, np.bool_, str, bytes)) for item in raw
    ):
        raise ValueError(f"{name} 必须是三维有限数字向量")
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是三维有限数字向量") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 必须是三维有限数字向量")
    return vector.copy()


def build_place_position_tf_specs(
    meta: Dict[str, Any],
) -> List[Tuple[str, List[float], List[float]]]:
    """从轨迹 metadata 提取任务级 place 位置 TF。

    双臂规划的 ``place_positions_arm1_base`` 已统一在一号臂基坐标系
    （联合规划 root）下表达，播放器不得再对二号臂位置应用安装变换。
    单臂结果从 ``place_position`` 读取实际规划 root 坐标。所有这些 frame
    只标记固定放置位置，不代表逐件搜索后选中的 TCP 姿态；新单/双臂结果若
    记录了 ``robot.link0_target_transform``，frame 朝向只显示这层任务坐标
    修正，未记录该字段的旧结果仍使用单位四元数。
    """
    positions = meta.get("place_positions_arm1_base")
    marker_quat = [1.0, 0.0, 0.0, 0.0]
    robot = meta.get("robot")
    if (
        isinstance(robot, dict)
        and robot.get("link0_target_transform") is not None
    ):
        correction = parse_rigid_transform_matrix(
            robot["link0_target_transform"],
            "metadata robot.link0_target_transform",
        )
        marker_quat = matrix_to_quat_wxyz(correction[:3, :3]).tolist()
    candidates: List[Tuple[str, Any]] = []
    if isinstance(positions, dict):
        candidates = [
            ("xtrainer_arm1_place", positions.get("arm1")),
            ("xtrainer_arm2_place", positions.get("arm2")),
        ]
    elif meta.get("place_position") is not None:
        candidates = [("xtrainer_arm1_place", meta.get("place_position"))]

    specs: List[Tuple[str, List[float], List[float]]] = []
    for child, raw_position in candidates:
        try:
            position = np.asarray(raw_position, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            continue
        specs.append((child, position.tolist(), marker_quat.copy()))
    return specs


# ============================== 配置加载 ==============================


def normalize_link0_target_transform_config_layer(
    layer: Dict[str, Any],
    source: str = "config",
) -> Dict[str, Any]:
    """将单层 YAML 中的 LINK_0 目标变换归一到 canonical 路径。

    canonical 路径是 ``pick_place.link0_target_transform``。早期版本曾把
    同一配置放在 ``dual_arm.link0_target_transform``，因此在每层配置参与
    ``deep_update`` 之前做迁移。逐层处理很重要：若等整个默认配置和
    用户差异配置合并后再处理，默认的 canonical 配置值会遮住后层的
    legacy 配置。

    同一层若同时写了新旧两个路径，只有两个刚体矩阵相同才
    接受，避免用隐式优先级猜测用户意图。返回深拷贝，不修改调用者
    持有的 YAML 字典，且返回值中只保留 canonical 路径。
    """
    if not isinstance(layer, dict):
        raise TypeError(f"{source} 顶层必须是 mapping")

    out = copy.deepcopy(layer)
    pick_place = out.get("pick_place")
    dual_arm = out.get("dual_arm")
    if pick_place is not None and not isinstance(pick_place, dict):
        raise TypeError(f"{source} 的 pick_place 必须是 mapping")
    if dual_arm is not None and not isinstance(dual_arm, dict):
        raise TypeError(f"{source} 的 dual_arm 必须是 mapping")

    missing = object()
    canonical_value = (
        pick_place.get("link0_target_transform", missing)
        if isinstance(pick_place, dict)
        else missing
    )
    legacy_value = (
        dual_arm.get("link0_target_transform", missing)
        if isinstance(dual_arm, dict)
        else missing
    )
    # 单层先把平移别名改成 canonical ``position``。否则后续 deep_update
    # 会把用户层的 ``translation`` 和默认层已有的 ``position`` 同时保留，
    # 最终看起来像用户配置了两个互相冲突的平移来源。
    if canonical_value is not missing:
        canonical_value = _normalize_transform_position_alias(
            canonical_value,
            f"{source} pick_place.link0_target_transform",
        )
        out["pick_place"]["link0_target_transform"] = canonical_value
    if legacy_value is not missing:
        legacy_value = _normalize_transform_position_alias(
            legacy_value,
            f"{source} dual_arm.link0_target_transform",
        )
        out["dual_arm"]["link0_target_transform"] = legacy_value
    if legacy_value is missing:
        return out

    if canonical_value is not missing:
        canonical = parse_rigid_transform_matrix(
            canonical_value,
            f"{source} pick_place.link0_target_transform",
        )
        legacy = parse_rigid_transform_matrix(
            legacy_value,
            f"{source} dual_arm.link0_target_transform",
        )
        if not np.allclose(canonical, legacy, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{source} 同时配置了 "
                "pick_place.link0_target_transform 和已废弃的 "
                "dual_arm.link0_target_transform，但两者不同"
            )
    else:
        out.setdefault("pick_place", {})["link0_target_transform"] = copy.deepcopy(
            legacy_value
        )

    # 归一后只留 canonical 路径，避免后续合并再产生两个真值源。
    out["dual_arm"].pop("link0_target_transform", None)
    return out


def _normalize_transform_position_alias(value: Any, name: str) -> Any:
    """在单层配置中把 ``translation`` 别名归一为 ``position``。"""
    if not isinstance(value, dict) or "translation" not in value:
        return copy.deepcopy(value)

    out = copy.deepcopy(value)
    translation = _parse_finite_vector3(
        out["translation"], f"{name}.translation"
    )
    if "position" in out:
        position = _parse_finite_vector3(out["position"], f"{name}.position")
        if not np.array_equal(position, translation):
            raise ValueError(
                f"{name}.position 和兼容别名 {name}.translation 必须相同"
            )
    else:
        out["position"] = copy.deepcopy(out["translation"])
    out.pop("translation")
    return out


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并 dict, override 覆盖 base。返回新 dict。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_task_config(path: Optional[str] = None) -> Dict[str, Any]:
    """加载任务 yaml。未指定则用 config/task_default.yaml。

    若指定了自定义配置, 会先加载 default 再深层合并, 因此自定义文件可以只写差异项。
    """
    import yaml  # noqa: PLC0415  两个环境都装了 pyyaml

    default_path = TASK_ROOT / "config" / "task_default.yaml"
    with open(default_path, "r") as f:
        cfg = yaml.safe_load(f)
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            for cand in (Path.cwd() / p, TASK_ROOT / p, TASK_ROOT / "config" / p):
                if cand.exists():
                    p = cand
                    break
        if not p.exists():
            raise FileNotFoundError(f"config not found: {path}")
        if p.resolve() != default_path.resolve():
            with open(p, "r") as f:
                cfg = deep_update(cfg, yaml.safe_load(f) or {})
    return cfg


# ============================== 位姿容器 ==============================


class PoseSpec:
    """一个带名字的目标位姿, 内部统一用 (position, quat_wxyz) 存储。

    joint_config 可选: 若给出(6 个关节角, rad), 规划该路点时走关节空间目标
    (plan_single_js), 从而保证构型/IK 分支与相邻路点一致。position/quat 此时
    仅用于 tf 显示与报表。
    """

    __slots__ = ("name", "position", "quat_wxyz", "kind", "joint_config")

    def __init__(
        self,
        name: str,
        position: Sequence[float],
        quat_wxyz: Sequence[float],
        kind: str = "waypoint",
        joint_config: Optional[Sequence[float]] = None,
    ):
        self.name = str(name)
        self.position = np.asarray(position, dtype=np.float64).reshape(3)
        q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        self.quat_wxyz = q / max(np.linalg.norm(q), 1e-12)
        self.kind = kind
        self.joint_config = (
            None if joint_config is None
            else np.asarray(joint_config, dtype=np.float64).reshape(-1)
        )

    @classmethod
    def from_rpy_deg(
        cls,
        name: str,
        position: Sequence[float],
        rpy_deg: Sequence[float],
        kind: str = "waypoint",
        joint_config: Optional[Sequence[float]] = None,
    ) -> "PoseSpec":
        return cls(name, position, rpy_deg_to_quat_wxyz(rpy_deg), kind, joint_config)

    @property
    def rpy_deg(self) -> np.ndarray:
        return quat_wxyz_to_rpy_deg(self.quat_wxyz)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "position": self.position.tolist(),
            "quat_wxyz": self.quat_wxyz.tolist(),
            "quat_xyzw": quat_wxyz_to_xyzw(self.quat_wxyz).tolist(),
            "rpy_deg": self.rpy_deg.tolist(),
        }
        if self.joint_config is not None:
            d["joint_config"] = self.joint_config.tolist()
            d["joint_config_deg"] = np.degrees(self.joint_config).round(3).tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PoseSpec":
        return cls(
            d["name"], d["position"], d["quat_wxyz"], d.get("kind", "waypoint"),
            d.get("joint_config"),
        )

    def __repr__(self) -> str:
        p = self.position
        r = self.rpy_deg
        js = "" if self.joint_config is None else ", js=True"
        return (
            f"PoseSpec({self.name}, xyz=[{p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}], "
            f"rpy_deg=[{r[0]:+.2f},{r[1]:+.2f},{r[2]:+.2f}]{js})"
        )


def build_pose_sequence(task_cfg: Dict[str, Any]) -> List[PoseSpec]:
    """按配置组装完整的位姿序列: start -> [start_lift] -> [extra...] -> [goal_lift] -> goal。"""
    start_cfg = task_cfg["start"]
    goal_cfg = task_cfg["goal"]
    start = PoseSpec.from_rpy_deg("start", start_cfg["position"], start_cfg["rpy_deg"], "start")
    goal = PoseSpec.from_rpy_deg("goal", goal_cfg["position"], goal_cfg["rpy_deg"], "goal")

    seq: List[PoseSpec] = [start]
    wp_cfg = task_cfg.get("waypoints") or {}
    lift_cfg = wp_cfg.get("auto_lift") or {}

    start_lift: Optional[PoseSpec] = None
    goal_lift: Optional[PoseSpec] = None
    if lift_cfg.get("enable", False):
        axis_mode = str(lift_cfg.get("lift_axis", "base_z"))
        keep = bool(lift_cfg.get("keep_orientation", True))

        def _lift(src: PoseSpec, dz: float, rpy_override, tag: str) -> Optional[PoseSpec]:
            if dz is None or abs(float(dz)) < 1e-9:
                return None
            if axis_mode == "tool_z_neg":
                direction = -tool_z_axis(src.quat_wxyz)
            elif axis_mode == "base_z":
                direction = np.array([0.0, 0.0, 1.0])
            else:
                raise ValueError(f"unknown lift_axis: {axis_mode}")
            pos = src.position + float(dz) * direction
            if keep or rpy_override is None:
                return PoseSpec(tag, pos, src.quat_wxyz, "lift")
            return PoseSpec.from_rpy_deg(tag, pos, rpy_override, "lift")

        start_lift = _lift(
            start, lift_cfg.get("start_lift_z"), lift_cfg.get("start_lift_rpy_deg"), "start_lift"
        )
        goal_lift = _lift(
            goal, lift_cfg.get("goal_lift_z"), lift_cfg.get("goal_lift_rpy_deg"), "goal_lift"
        )

    if start_lift is not None:
        seq.append(start_lift)
    for i, e in enumerate(wp_cfg.get("extra") or []):
        # extra 路点支持两种写法:
        #   1) position + rpy_deg          -> 位姿目标(IK 可能选到别的分支)
        #   2) joint_deg 或 joint_rad      -> 关节空间目标(保证构型连续, 推荐用于分解大翻转)
        jc = None
        if e.get("joint_deg") is not None:
            jc = np.radians(np.asarray(e["joint_deg"], dtype=np.float64))
        elif e.get("joint_rad") is not None:
            jc = np.asarray(e["joint_rad"], dtype=np.float64)

        if jc is not None:
            # 关节目标模式下 position/rpy 可省略(仅用于显示), 缺省则填 0
            pos = e.get("position") or [0.0, 0.0, 0.0]
            rpy = e.get("rpy_deg") or [0.0, 0.0, 0.0]
            seq.append(PoseSpec.from_rpy_deg(f"extra_{i}", pos, rpy, "extra", jc))
        else:
            seq.append(
                PoseSpec.from_rpy_deg(f"extra_{i}", e["position"], e["rpy_deg"], "extra")
            )
    if goal_lift is not None:
        seq.append(goal_lift)
    seq.append(goal)
    return seq


def compose_right(q_wxyz: Sequence[float], delta_rpy_deg: Sequence[float]) -> np.ndarray:
    """在目标姿态上「右乘」一个增量旋转, 即绕工具自身轴转动。

        R_new = R_base @ R_delta

    右乘 = 内旋 = 绕转动后的新轴转; 左乘 = 外旋 = 绕固定的 base 轴转。
    delta_rpy_deg 自身仍按本项目统一约定解释(固定轴 XYZ 外旋 -> R=Rz@Ry@Rx),
    但整体是作用在工具坐标系下的。

    例: delta=[-30,0,0] 表示绕工具自身的 X 轴转 -30 度。
    """
    r_base = quat_wxyz_to_matrix(q_wxyz)
    r_delta = rpy_deg_to_matrix(delta_rpy_deg)
    return matrix_to_quat_wxyz(r_base @ r_delta)


def compose_left(q_wxyz: Sequence[float], delta_rpy_deg: Sequence[float]) -> np.ndarray:
    """在目标姿态上「左乘」一个增量旋转, 即绕 base 固定轴转动。

        R_new = R_delta @ R_base
    """
    r_base = quat_wxyz_to_matrix(q_wxyz)
    r_delta = rpy_deg_to_matrix(delta_rpy_deg)
    return matrix_to_quat_wxyz(r_delta @ r_base)


def build_pose_variants(
    task_cfg: Dict[str, Any], which: str, variants_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """展开某个端点(start 或 goal)的姿态变体列表。

    读取 task.<which>_variants:
        enable: true
        mode: right            # right=绕工具自身轴(默认); left=绕 base 固定轴
        rotations:             # 每一项是一个 rpy 增量(度), 对应一个独立端点
          - [-30, 0, 0]
          - [  0, 0, 0]
        position_delta: null   # 可选, 同时给位置加偏移 [dx,dy,dz](m)
                               # 或 [[..],[..]] 逐变体指定

    Returns:
        变体列表, 每项 {which, name, index, mode, delta_rpy_deg, delta_position,
                       position, quat_wxyz, rpy_deg}
        未启用时返回单个 "base" 变体(即原始端点), 便于上层统一按列表处理。
    """
    if which not in ("start", "goal"):
        raise ValueError(f"which 只支持 start/goal, 得到 {which}")
    cfg = task_cfg[which]
    base_pos = np.asarray(cfg["position"], dtype=np.float64).reshape(3)
    base_quat = rpy_deg_to_quat_wxyz(cfg["rpy_deg"])

    key = variants_key or f"{which}_variants"
    gv = task_cfg.get(key) or {}
    rotations = gv.get("rotations") or []
    if not gv.get("enable", False) or not rotations:
        return [
            {
                "which": which,
                "name": "base",
                "index": 0,
                "mode": "none",
                "delta_rpy_deg": [0.0, 0.0, 0.0],
                "delta_position": [0.0, 0.0, 0.0],
                "position": base_pos.tolist(),
                "quat_wxyz": base_quat.tolist(),
                "rpy_deg": quat_wxyz_to_rpy_deg(base_quat).tolist(),
            }
        ]

    mode = str(gv.get("mode", "right")).lower()
    if mode not in ("right", "left"):
        raise ValueError(f"{key}.mode 只支持 right/left, 得到 {mode}")
    dpos_all = gv.get("position_delta")

    out: List[Dict[str, Any]] = []
    for i, rot in enumerate(rotations):
        d = np.asarray(rot, dtype=np.float64).reshape(3)
        q = compose_right(base_quat, d) if mode == "right" else compose_left(base_quat, d)
        dp = np.zeros(3)
        if dpos_all is not None:
            arr = np.asarray(dpos_all, dtype=np.float64)
            dp = arr[i] if arr.ndim == 2 else arr.reshape(3)
        pos = base_pos + dp
        tag = "_".join(f"{v:+.0f}" for v in d).replace("+", "p").replace("-", "m")
        out.append(
            {
                "which": which,
                "name": f"{which[0]}{i}_rpy_{tag}",
                "index": i,
                "mode": mode,
                "delta_rpy_deg": d.tolist(),
                "delta_position": dp.tolist(),
                "position": pos.tolist(),
                "quat_wxyz": q.tolist(),
                "rpy_deg": quat_wxyz_to_rpy_deg(q).tolist(),
            }
        )
    return out


def build_goal_variants(task_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """展开目标姿态变体(兼容旧接口)。等价于 build_pose_variants(cfg, 'goal')。"""
    return build_pose_variants(task_cfg, "goal")


def build_variant_combos(task_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """对 start 变体与 goal 变体做笛卡尔积, 每个组合 = 一次独立规划。

    Returns:
        组合列表, 每项 {name, index, start, goal, n_start, n_goal},
        其中 start/goal 是 build_pose_variants 的单个元素。
    """
    s_list = build_pose_variants(task_cfg, "start")
    g_list = build_pose_variants(task_cfg, "goal")
    combos: List[Dict[str, Any]] = []
    k = 0
    for sv in s_list:
        for gv in g_list:
            # 命名: 两端都是 base 时用 base; 否则拼接非 base 的一侧
            if sv["name"] == "base" and gv["name"] == "base":
                name = "base"
            elif sv["name"] == "base":
                name = gv["name"]
            elif gv["name"] == "base":
                name = sv["name"]
            else:
                name = f"{sv['name']}__{gv['name']}"
            combos.append(
                {
                    "name": name,
                    "index": k,
                    "start": sv,
                    "goal": gv,
                    "n_start": len(s_list),
                    "n_goal": len(g_list),
                }
            )
            k += 1
    return combos


def apply_variant_combo(task_cfg: Dict[str, Any], combo: Dict[str, Any]) -> Dict[str, Any]:
    """返回把某个组合(start+goal)写入后的 task 配置副本。"""
    t = copy.deepcopy(task_cfg)
    for which in ("start", "goal"):
        v = combo[which]
        t[which]["position"] = list(v["position"])
        t[which]["rpy_deg"] = list(v["rpy_deg"])
    # 变体已展开, 避免下游再次展开
    t["start_variants"] = {"enable": False}
    t["goal_variants"] = {"enable": False}
    # start 姿态被改动后, 预设的 start_joint_state 不再对应该位姿, 必须让 IK 重解
    if combo["start"]["name"] != "base":
        t["start_joint_state"] = None
    return t


def apply_goal_variant(task_cfg: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    """返回把某个变体写入 goal 后的 task 配置副本(兼容旧接口)。"""
    t = copy.deepcopy(task_cfg)
    t["goal"]["position"] = list(variant["position"])
    t["goal"]["rpy_deg"] = list(variant["rpy_deg"])
    t["goal_variants"] = {"enable": False}
    return t


# ============================== 工作空间墙 ==============================


def build_workspace_wall_cuboids(ws_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把工作空间 bounds 转成 6 面 cuboid 墙的描述列表。

    每个元素: {name, dims:[dx,dy,dz], pose:[x,y,z,qw,qx,qy,qz]}, pose 是盒中心。
    墙从边界面向外延伸 thickness, 因此盒内部即为合法工作空间。

    base 让位孔: base 立柱位于原点, 其碰撞球必然穿过 x_max(x=0) 与 z_min(z=0.1) 面。
    curobo 无法「按 link 关闭世界碰撞」(disable_link_spheres 会连自碰撞一起失效),
    因此把这两个面各拆成若干块, 在底座周围留出空洞, 从而完整保留自碰撞检测。
    """
    b = ws_cfg["bounds"]
    wall = ws_cfg.get("wall") or {}
    # Mounted configurations can carry exact wall cuboids in current LINK_0.
    # Rebuilding from the displayed AABB would alter tilted collision walls.
    explicit = wall.get("cuboids_override")
    if (explicit is None) != (ws_cfg.get("oriented_bounds") is None):
        raise ValueError(
            "workspace.oriented_bounds and wall.cuboids_override must be paired")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise ValueError("workspace.wall.cuboids_override must be a list")
        boxes = []
        names = set()
        for raw in explicit:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                raise ValueError("each explicit wall cuboid needs a name")
            name = raw["name"]
            if name in names:
                raise ValueError(f"duplicate explicit wall cuboid: {name}")
            names.add(name)
            dims = np.asarray(raw.get("dims"), dtype=np.float64)
            pose = np.asarray(raw.get("pose"), dtype=np.float64)
            if (dims.shape != (3,) or pose.shape != (7,)
                    or not np.all(np.isfinite(dims))
                    or not np.all(np.isfinite(pose))
                    or np.any(dims <= 0)
                    or not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-6)):
                raise ValueError(f"invalid explicit wall cuboid: {name}")
            boxes.append({"name": name, "dims": dims.tolist(), "pose": pose.tolist()})
        return boxes
    # 墙可以用一个放宽后的盒子(bounds_override), 以容纳伸出末端的夹爪;
    # 而 bounds 本身保持用户声明的真实工作空间, 仅用于规划后的软校验。
    ob = wall.get("bounds_override") or {}
    bx = ob.get("x", b["x"])
    by = ob.get("y", b["y"])
    bz = ob.get("z", b["z"])
    x0, x1 = float(bx[0]), float(bx[1])
    y0, y1 = float(by[0]), float(by[1])
    z0, z1 = float(bz[0]), float(bz[1])
    t = float(wall.get("thickness", 0.05))
    faces = wall.get("faces") or {}
    clr = wall.get("base_clearance") or {}
    clr_on = bool(clr.get("enable", False))
    cy = float(clr.get("y_half", 0.13))
    cz = float(clr.get("z_max", 0.15))
    cx = float(clr.get("x_min", -0.13))

    ident = [1.0, 0.0, 0.0, 0.0]
    out: List[Dict[str, Any]] = []

    def add(name: str, lo: Tuple[float, float, float], hi: Tuple[float, float, float]) -> None:
        dims = [hi[i] - lo[i] for i in range(3)]
        if min(dims) <= 1e-6:
            return
        center = [(lo[i] + hi[i]) * 0.5 for i in range(3)]
        out.append({"name": name, "dims": dims, "pose": center + ident})

    # 墙面覆盖范围向外扩 t, 保证 6 面在角上互相咬合、无缝隙
    ex0, ex1 = x0 - t, x1 + t
    ey0, ey1 = y0 - t, y1 + t
    ez0, ez1 = z0 - t, z1 + t

    if faces.get("x_min", True):
        add("wall_x_min", (ex0, ey0, ez0), (x0, ey1, ez1))

    if faces.get("x_max", True):
        if clr_on:
            # x=0 面: 在 |y|<cy 且 z<cz 的区域开孔给底座让位
            yc0 = max(-cy, ey0)
            yc1 = min(cy, ey1)
            zc1 = min(cz, ez1)
            add("wall_x_max_ylo", (x1, ey0, ez0), (ex1, yc0, ez1))
            add("wall_x_max_yhi", (x1, yc1, ez0), (ex1, ey1, ez1))
            add("wall_x_max_zhi", (x1, yc0, zc1), (ex1, yc1, ez1))
        else:
            add("wall_x_max", (x1, ey0, ez0), (ex1, ey1, ez1))

    if faces.get("y_min", True):
        add("wall_y_min", (ex0, ey0, ez0), (ex1, y0, ez1))
    if faces.get("y_max", True):
        add("wall_y_max", (ex0, y1, ez0), (ex1, ey1, ez1))
    if faces.get("z_max", True):
        add("wall_z_max", (ex0, ey0, z1), (ex1, ey1, ez1))

    if faces.get("z_min", True):
        if clr_on:
            # z=z0 面: 在 x>cx 且 |y|<cy 的区域开孔给底座让位
            yc0 = max(-cy, ey0)
            yc1 = min(cy, ey1)
            xc0 = max(cx, ex0)
            add("wall_z_min_xlo", (ex0, ey0, ez0), (xc0, ey1, z0))
            add("wall_z_min_ylo", (xc0, ey0, ez0), (ex1, yc0, z0))
            add("wall_z_min_yhi", (xc0, yc1, ez0), (ex1, ey1, z0))
        else:
            add("wall_z_min", (ex0, ey0, ez0), (ex1, ey1, z0))

    return out


def check_in_bounds(
    positions: np.ndarray, ws_cfg: Dict[str, Any], margin: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """逐点检查是否在 bounds 内。

    Args:
        positions: [N, 3]
        margin: 允许的越界容差 (m)

    Returns:
        (inside[N] bool, violation[N] float) violation 为最大越界距离(m), 0 表示合法。
    """
    oriented = ws_cfg.get("oriented_bounds")
    if (oriented is None) != (
            (ws_cfg.get("wall") or {}).get("cuboids_override") is None):
        raise ValueError(
            "workspace.oriented_bounds and wall.cuboids_override must be paired")
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if oriented is not None:
        # frame_transform maps original task bounds into current LINK_0.
        # Evaluate in that frame; workspace.bounds is only its display AABB.
        frame = parse_rigid_transform_matrix(
            oriented["frame_transform"], "workspace.oriented_bounds.frame_transform"
        )
        p = (p - frame[:3, 3]) @ frame[:3, :3]
        b = oriented["reference_bounds"]
    else:
        b = ws_cfg["bounds"]
    lo = np.array([b["x"][0], b["y"][0], b["z"][0]], dtype=np.float64) - margin
    hi = np.array([b["x"][1], b["y"][1], b["z"][1]], dtype=np.float64) + margin
    under = np.maximum(lo[None, :] - p, 0.0)
    over = np.maximum(p - hi[None, :], 0.0)
    viol = np.maximum(under, over).max(axis=1)
    return viol <= 1e-12, viol


# ============================== 轨迹 I/O ==============================

TRAJ_NPZ = "trajectory.npz"
TRAJ_META = "trajectory_meta.json"


def _json_default(o: Any) -> Any:
    """让 json.dump 能处理 numpy 标量/数组。

    规划端会把 torch/numpy 取出的数值塞进 meta, 若不转换会抛 TypeError,
    导致 meta json 被写残(截断), 后续读取直接失败。
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def dump_json(obj: Any, path: Path) -> None:
    """写 json, 自动处理 numpy 类型。"""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def save_trajectory(
    out_dir: Path,
    joint_names: List[str],
    positions: np.ndarray,
    velocities: Optional[np.ndarray],
    accelerations: Optional[np.ndarray],
    times: np.ndarray,
    ee_positions: np.ndarray,
    ee_quats_wxyz: np.ndarray,
    meta: Dict[str, Any],
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Path, Path]:
    """保存轨迹。npz 存数值, json 存元信息(位姿目标/工作空间/统计)。

    注意: 规划端是 conda py3.11 + numpy>=2, 播放端是 ROS py3.8 + numpy<2,
    两者的 pickle 不兼容。因此 npz 中一律不放 object dtype 数组
    (joint_names 改为定长 ASCII 的 'S' dtype), 保证 allow_pickle=False 即可读取。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / TRAJ_NPZ
    payload: Dict[str, Any] = {
        "joint_names": np.array([str(s) for s in joint_names], dtype="S64"),
        "positions": np.asarray(positions, dtype=np.float64),
        "times": np.asarray(times, dtype=np.float64),
        "ee_positions": np.asarray(ee_positions, dtype=np.float64),
        "ee_quats_wxyz": np.asarray(ee_quats_wxyz, dtype=np.float64),
    }
    if velocities is not None:
        payload["velocities"] = np.asarray(velocities, dtype=np.float64)
    if accelerations is not None:
        payload["accelerations"] = np.asarray(accelerations, dtype=np.float64)
    for name, value in (extra_arrays or {}).items():
        if name in payload:
            raise ValueError(f"extra_arrays 不能覆盖内置字段: {name}")
        payload[str(name)] = np.asarray(value)
    np.savez_compressed(npz_path, **payload)

    meta_path = out_dir / TRAJ_META
    dump_json(meta, meta_path)
    return npz_path, meta_path


def load_trajectory(traj_dir_or_npz: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """加载轨迹。参数可以是目录或 npz 文件路径。返回 (data, meta)。"""
    p = Path(traj_dir_or_npz)
    npz_path = p if p.suffix == ".npz" else p / TRAJ_NPZ
    if not npz_path.exists():
        raise FileNotFoundError(f"trajectory npz not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        data: Dict[str, Any] = {k: z[k] for k in z.files}
    data["joint_names"] = [
        s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s) for s in data["joint_names"]
    ]

    meta_path = npz_path.parent / TRAJ_META
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return data, meta


def resolve_repo_path(p: str) -> Path:
    """把相对仓库根的路径转成绝对路径。"""
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)
