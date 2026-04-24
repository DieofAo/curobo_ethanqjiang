#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 sample.txt 中提取 jaka_arm_left 的目标位姿数据，
使用 CuRobo IK 求解器进行批量逆运动学求解。

与 `jaka_left_ik_from_sample.py` 的区别：
  * 使用 `jaka_mobile.yml` 配置（10 DOF）：
      [base_x, base_y, base_yaw, J_1, ..., J_7]
  * base_link = world，目标 pose 视为表达在 world 系下（与初始 LINK_BASE 重合）
  * 通过 IK 同时优化底盘平移/旋转 + 机械臂 7 关节
  * 用于对比：纯 7-DOF vs. 引入底盘 3 自由度后 IK 成功率

数据格式 (每行17个字段，与原脚本一致):
  字段1:  时间戳
  字段2:  机械臂名称 (jaka_arm_left / jaka_arm_right)
  字段3:  状态 (solver_failed / large_joint_change)
  字段4-6:  目标位置 (x, y, z)
  字段7-10: 目标四元数 (x, y, z, w)
  字段11-17: 7个关节角度 (seed)

使用方法:
  # 仅求解并统计
  python examples/jaka_left_ik_mobile_from_sample.py

  # 求解后发布到 ROS 话题 (需要先启动 roscore 和 robot_state_publisher)
  python examples/jaka_left_ik_mobile_from_sample.py --publish

  # 指定发布速率倍数 (默认1.0x)
  python examples/jaka_left_ik_mobile_from_sample.py --publish --speed 2.0

  # 循环回放
  python examples/jaka_left_ik_mobile_from_sample.py --publish --loop

  # 仅回放失败片段
  python examples/jaka_left_ik_mobile_from_sample.py --publish --focus-failed

RViz 可视化前置步骤:
  1. 启动 roscore
  2. 用 left_jaka_mobile.urdf 作为 robot_description 启动 robot_state_publisher
     (URDF 根 link 为 world，含虚拟 base_x/base_y/base_yaw + J_1~J_7)
  3. 在 RViz 中添加 RobotModel 显示，Fixed Frame 设为 world
     可同时添加 Pose 显示，订阅 /ik_target_pose 查看目标位姿
"""

import os
import re
import time
import argparse

import torch
import numpy as np

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml, get_assets_path
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 10 DOF 顺序，与 jaka_mobile.yml 中 cspace.joint_names 保持一致
MOBILE_JOINT_NAMES = [
    "base_x", "base_y", "base_yaw",
    "J_1", "J_2", "J_3", "J_4", "J_5", "J_6", "J_7",
]
BASE_DOF = 3
ARM_DOF = 7

# 7-DOF 纯机械臂配置（用于自适应底盘控制中的 LOCKED 分支）
ARM_ROBOT_FILE = "jaka.yml"
ARM_URDF_REL_PATH = "robot/jaka_description/left_jaka.urdf"
ARM_JOINT_NAMES = ["J_1", "J_2", "J_3", "J_4", "J_5", "J_6", "J_7"]


def _parse_joint_limits_from_urdf(urdf_path, joint_names):
    """从 URDF 文本里解析指定关节的 (lower, upper)。

    返回 shape=(len(joint_names), 2) 的 numpy 数组。
    简单的正则解析，足以处理形如:
        <joint name="J_1" type="revolute">
            ...
            <limit lower="-6.1927" upper="6.1927" .../>
        </joint>
    """
    with open(urdf_path, "r") as f:
        text = f.read()
    limits = np.zeros((len(joint_names), 2), dtype=np.float32)
    for k, jn in enumerate(joint_names):
        pattern = (
            r'<joint\s+name="' + re.escape(jn) + r'"[^>]*>'
            r'(.*?)</joint>'
        )
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            raise ValueError(f"URDF 中找不到关节 {jn}: {urdf_path}")
        body = m.group(1)
        ml = re.search(r'<limit\s+lower="([-0-9.eE+]+)"\s+upper="([-0-9.eE+]+)"', body)
        if not ml:
            raise ValueError(f"关节 {jn} 未找到 <limit lower=... upper=...>")
        limits[k, 0] = float(ml.group(1))
        limits[k, 1] = float(ml.group(2))
    return limits


def _world_pose_to_linkbase_pose(pos_world, quat_world_wxyz, base_xyyaw):
    """把 world 系下的目标 pose 变换到 LINK_BASE 系下。

    base_xyyaw: [bx, by, byaw] 描述 world→LINK_BASE 的 SE(2) 变换
        (LINK_BASE 相对 world 的位置/朝向)

    返回 (pos_base[3], quat_base_wxyz[4])，都是 numpy。
    """
    bx, by, byaw = float(base_xyyaw[0]), float(base_xyyaw[1]), float(base_xyyaw[2])
    c, s = np.cos(byaw), np.sin(byaw)
    # 位置：先平移再绕 Z 逆旋转
    dx = pos_world[0] - bx
    dy = pos_world[1] - by
    px = c * dx + s * dy
    py = -s * dx + c * dy
    pz = pos_world[2]
    pos_base = np.array([px, py, pz], dtype=np.float32)

    # 四元数：world→base 的逆旋转 = 绕 Z -byaw
    # q_base = q_zinv(byaw) * q_world  (wxyz 顺序, 四元数 Hamilton 乘法)
    qw_b = np.cos(-byaw / 2.0)
    qz_b = np.sin(-byaw / 2.0)
    # q_world
    qw_w = float(quat_world_wxyz[0])
    qx_w = float(quat_world_wxyz[1])
    qy_w = float(quat_world_wxyz[2])
    qz_w = float(quat_world_wxyz[3])
    # (w1,0,0,z1) * (w2,x2,y2,z2)
    ow = qw_b * qw_w - qz_b * qz_w
    ox = qw_b * qx_w - qz_b * qy_w
    oy = qw_b * qy_w + qz_b * qx_w
    oz = qw_b * qz_w + qz_b * qw_w
    quat_base = np.array([ow, ox, oy, oz], dtype=np.float32)
    return pos_base, quat_base


class AdaptiveBaseController:
    """自适应底盘控制器（状态机 + 双 IK 求解器）。

    LOCKED: 只用 7-DOF IK，base 冻结在 prev_base（初始 0）。
    ACTIVE: 走 10-DOF IK，base 自由。

    触发 LOCKED → ACTIVE 的条件（任一满足）:
      - 7-DOF IK 失败
      - 7-DOF 成功但关节距任意限位 < margin_trigger
      - 7-DOF 成功但 |J_5| < singular_j5 或 |J_3| < singular_j3（启发式奇异）

    ACTIVE → LOCKED 的条件（全部满足）:
      - 10-DOF 成功
      - 关节距任意限位 > margin_release
      - 远离奇异
      - 连续满足 >= release_hold 帧
    """

    def __init__(
        self,
        ik_solver_full,
        ik_solver_arm,
        arm_joint_limits,      # np.array shape=(7,2)
        tensor_args,
        margin_trigger=np.deg2rad(5.0),
        margin_release=np.deg2rad(15.0),
        release_hold=10,
        singular_j5=np.deg2rad(3.0),
        singular_j3=np.deg2rad(5.0),
    ):
        self.ik_full = ik_solver_full
        self.ik_arm = ik_solver_arm
        self.limits = arm_joint_limits  # (7,2)
        self.tensor_args = tensor_args

        self.margin_trigger = float(margin_trigger)
        self.margin_release = float(margin_release)
        self.release_hold = int(release_hold)
        self.singular_j5 = float(singular_j5)
        self.singular_j3 = float(singular_j3)

        self.state = "LOCKED"
        self.prev_base = np.zeros(BASE_DOF, dtype=np.float32)
        # 上一帧 7-DOF IK 解出的 arm (warm-start 用，降低多解跳变带来的 arm 抖动)
        self.prev_arm_q = None
        self.safe_cnt = 0

        # 统计
        self.n_locked = 0
        self.n_active = 0
        self.n_switches = 0
        self.switch_log = []  # [(frame_idx, from, to, reason)]
        # freeze 降级 (ACTIVE 下末端静止时尝试 7-DOF 数学正确地冻结 base)
        self.n_freeze_downgrade_ok = 0
        self.n_freeze_downgrade_fail = 0

    # ---- 工具方法 ----
    def _min_joint_margin(self, q_arm):
        """返回 7 关节中距离任一端的最小裕度 (rad)。q_arm: np.ndarray shape=(7,)"""
        lo = self.limits[:, 0]
        up = self.limits[:, 1]
        d_lo = q_arm - lo
        d_up = up - q_arm
        return float(min(d_lo.min(), d_up.min()))

    def _is_singular(self, q_arm):
        """启发式奇异检测：J_5 接近 0 (腕奇异) 或 J_3 接近 0 (肘奇异)。"""
        if abs(float(q_arm[4])) < self.singular_j5:  # J_5 index=4
            return True, "wrist_singular(J5~0)"
        if abs(float(q_arm[2])) < self.singular_j3:  # J_3 index=2
            return True, "elbow_singular(J3~0)"
        return False, ""

    def _solve_arm_7dof(self, pos_world, quat_world, seed_arm_7):
        """在 LINK_BASE 系下用 7-DOF IK 求解。返回 (success, q_arm_7, pos_err, rot_err)。"""
        pos_base, quat_base = _world_pose_to_linkbase_pose(
            pos_world, quat_world, self.prev_base
        )
        pos_t = torch.tensor(
            pos_base.reshape(1, 3),
            device=self.tensor_args.device, dtype=self.tensor_args.dtype,
        )
        quat_t = torch.tensor(
            quat_base.reshape(1, 4),
            device=self.tensor_args.device, dtype=self.tensor_args.dtype,
        )
        seed_t = torch.tensor(
            np.asarray(seed_arm_7, dtype=np.float32).reshape(1, 1, ARM_DOF),
            device=self.tensor_args.device, dtype=self.tensor_args.dtype,
        )
        res = self.ik_arm.solve_single(Pose(pos_t, quat_t), seed_config=seed_t)
        torch.cuda.synchronize()
        ok = bool(res.success.item())
        pe = float(res.position_error.item())
        re_ = float(res.rotation_error.item())
        if ok:
            q_arm = res.js_solution.position.cpu().numpy().flatten()  # (7,)
        else:
            q_arm = None
        return ok, q_arm, pe, re_

    def _solve_full_10dof(self, pos_world, quat_world, seed_full_10):
        """10-DOF IK 求解（world 系）。返回 (success, q_full_10, pos_err, rot_err)。"""
        pos_t = torch.tensor(
            np.asarray(pos_world, dtype=np.float32).reshape(1, 3),
            device=self.tensor_args.device, dtype=self.tensor_args.dtype,
        )
        quat_t = torch.tensor(
            np.asarray(quat_world, dtype=np.float32).reshape(1, 4),
            device=self.tensor_args.device, dtype=self.tensor_args.dtype,
        )
        seed_t = torch.tensor(
            np.asarray(seed_full_10, dtype=np.float32).reshape(1, 1, BASE_DOF + ARM_DOF),
            device=self.tensor_args.device, dtype=self.tensor_args.dtype,
        )
        res = self.ik_full.solve_single(Pose(pos_t, quat_t), seed_config=seed_t)
        torch.cuda.synchronize()
        ok = bool(res.success.item())
        pe = float(res.position_error.item())
        re_ = float(res.rotation_error.item())
        if ok:
            q_full = res.js_solution.position.cpu().numpy().flatten()  # (10,)
        else:
            q_full = None
        return ok, q_full, pe, re_

    def _switch(self, new_state, frame_idx, reason):
        if new_state != self.state:
            self.switch_log.append((frame_idx, self.state, new_state, reason))
            self.n_switches += 1
            print(f"  [frame {frame_idx+1}] {self.state}→{new_state}  reason={reason}")
            self.state = new_state
            self.safe_cnt = 0

    # ---- 对外主入口 ----
    def step(self, frame_idx, pos_world, quat_world, seed_arm_7,
             freeze_if_stationary=False, end_stationary=False):
        """处理单帧。

        关键不变量（严禁违反）：
          - LOCKED 分支输出 [prev_base, q_arm_7dof]，其中 7-DOF IK 的 pose 变换用的
            就是 self.prev_base。base 与 arm 一一匹配。
          - ACTIVE 分支输出 10-DOF IK 原始解 [bx_new, by_new, byaw_new, q_arm_new]，
            绝不允许事后替换 base。
          - freeze 降级：ACTIVE 下若 end_stationary 且 freeze_if_stationary，
            先试 7-DOF（用 prev_base 做 world→LINK_BASE 变换），成功则按 LOCKED
            的方式输出（数学正确地冻结 base）；失败则退回 10-DOF 正常求解。
            该帧 state 标签保持 ACTIVE，不影响 safe_cnt / release_hold 节奏。

        Returns:
            sol_full_10: np.ndarray(10,) 或 None
            success: bool
            state: 'LOCKED'/'ACTIVE'
            pos_err, rot_err: float (用于统计)
            solve_time: float
        """
        t0 = time.time()

        # arm warm-start：优先用上一帧 7-DOF 解，降低 multi-seed 选解跳变
        seed_arm_eff = (
            self.prev_arm_q if self.prev_arm_q is not None
            else np.asarray(seed_arm_7, dtype=np.float32)
        )

        if self.state == "LOCKED":
            ok, q_arm, pe, re_ = self._solve_arm_7dof(
                pos_world, quat_world, seed_arm_eff
            )
            trigger_reason = None
            if not ok:
                trigger_reason = f"arm_ik_failed(pe={pe*1000:.1f}mm,re={re_:.3f})"
            else:
                margin = self._min_joint_margin(q_arm)
                is_sg, sg_name = self._is_singular(q_arm)
                if margin < self.margin_trigger:
                    trigger_reason = (
                        f"limit_margin={np.rad2deg(margin):.2f}deg"
                        f"<trigger={np.rad2deg(self.margin_trigger):.1f}deg"
                    )
                elif is_sg:
                    trigger_reason = sg_name

            if trigger_reason is None:
                # 保持 LOCKED，base 锁在 prev_base，两者匹配
                self.n_locked += 1
                self.prev_arm_q = q_arm.astype(np.float32).copy()
                sol_full = np.concatenate([self.prev_base, q_arm]).astype(np.float32)
                return sol_full, True, "LOCKED", pe, re_, time.time() - t0

            # 触发 → 切到 ACTIVE，继续走下面 ACTIVE 分支
            self._switch("ACTIVE", frame_idx, trigger_reason)

        # state == "ACTIVE"
        # ---- freeze 降级：末端静止时优先尝试 7-DOF 冻结 base (数学正确) ----
        if freeze_if_stationary and end_stationary:
            ok7, q_arm7, pe7, re7 = self._solve_arm_7dof(
                pos_world, quat_world, seed_arm_eff
            )
            if ok7:
                # 输出 [prev_base, q_arm7]，base/arm 严格匹配，无 FK 漂移
                self.n_freeze_downgrade_ok += 1
                self.n_active += 1
                self.safe_cnt = 0  # 本帧未走 10-DOF，不累计 release 节奏
                self.prev_arm_q = q_arm7.astype(np.float32).copy()
                sol_full = np.concatenate(
                    [self.prev_base, q_arm7]
                ).astype(np.float32)
                return sol_full, True, "ACTIVE", pe7, re7, time.time() - t0
            # 降级失败，回退到正常 10-DOF
            self.n_freeze_downgrade_fail += 1

        seed_full = np.concatenate(
            [self.prev_base, seed_arm_eff.astype(np.float32)]
        ).astype(np.float32)
        ok, q_full, pe, re_ = self._solve_full_10dof(pos_world, quat_world, seed_full)

        if not ok:
            self.safe_cnt = 0
            self.n_active += 1
            return None, False, "ACTIVE", pe, re_, time.time() - t0

        # 成功：更新 prev_base + 判断能否回 LOCKED
        q_arm = q_full[BASE_DOF:]
        margin = self._min_joint_margin(q_arm)
        is_sg, _ = self._is_singular(q_arm)
        very_safe = (margin > self.margin_release) and (not is_sg)
        if very_safe:
            self.safe_cnt += 1
            if self.safe_cnt >= self.release_hold:
                # 先记录新的 base，再切回 LOCKED
                self.prev_base = q_full[:BASE_DOF].copy().astype(np.float32)
                self.prev_arm_q = q_arm.astype(np.float32).copy()
                self._switch("LOCKED", frame_idx,
                             f"safe_hold_reached(cnt={self.safe_cnt})")
                self.n_locked += 1
                return q_full.astype(np.float32), True, "LOCKED", pe, re_, time.time() - t0
        else:
            self.safe_cnt = 0

        self.prev_base = q_full[:BASE_DOF].copy().astype(np.float32)
        self.prev_arm_q = q_arm.astype(np.float32).copy()
        self.n_active += 1
        return q_full.astype(np.float32), True, "ACTIVE", pe, re_, time.time() - t0

    def summary_str(self):
        total = self.n_locked + self.n_active
        if total == 0:
            return "[自适应底盘] 无数据"
        return (
            f"[自适应底盘] LOCKED={self.n_locked}/{total} ({self.n_locked/total*100:.1f}%)  "
            f"ACTIVE={self.n_active}/{total} ({self.n_active/total*100:.1f}%)  "
            f"切换次数={self.n_switches}  "
            f"freeze降级 ok/fail={self.n_freeze_downgrade_ok}/{self.n_freeze_downgrade_fail}"
        )


def parse_sample_file(filepath: str):
    """解析 sample.txt，提取 jaka_arm_left 的数据。"""
    timestamps, statuses, positions, quaternions, seed_joints = [], [], [], [], []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 17:
                continue
            if parts[1] != "jaka_arm_left":
                continue

            timestamps.append(float(parts[0]))
            statuses.append(parts[2])
            pos = [float(parts[3]), float(parts[4]), float(parts[5])]
            # 原始数据四元数顺序 xyzw -> CuRobo 需要 wxyz
            qx, qy, qz, qw = (
                float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])
            )
            quat = [qw, qx, qy, qz]
            joints = [float(parts[i]) for i in range(10, 17)]

            positions.append(pos)
            quaternions.append(quat)
            seed_joints.append(joints)

    return (
        timestamps,
        statuses,
        np.array(positions, dtype=np.float32),
        np.array(quaternions, dtype=np.float32),
        np.array(seed_joints, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(
        description="JAKA Left Arm IK (with 3 virtual base DOFs)"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="输入数据文件路径 (默认为项目根目录的 sample.txt)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=500,
        help="IK 种子数 (默认 500，与 7-DOF 脚本一致)"
    )
    parser.add_argument(
        "--self-collision", action="store_true",
        help="启用自碰撞检查 (默认关闭，与 7-DOF 脚本一致)"
    )
    parser.add_argument(
        "--publish", action="store_true",
        help="求解后将关节角发布到 ROS /joint_states 话题"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="回放速率倍数 (默认 1.0x，2.0 表示两倍速)"
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="循环回放"
    )
    parser.add_argument(
        "--topic", type=str, default="/joint_states",
        help="发布的 ROS 话题名 (默认 /joint_states)"
    )
    parser.add_argument(
        "--focus-failed", action="store_true",
        help="仅播放 IK 失败片段及其前后正常轨迹 (慢速模式)"
    )
    parser.add_argument(
        "--context-size", type=int, default=5,
        help="--focus-failed 模式下，失败片段前后各保留的正常帧数 (默认 5)"
    )
    # --- 抑制底盘漂移相关 ---
    parser.add_argument(
        "--no-warm-start", action="store_true",
        help="关闭时序热启动 (默认开启：用上一帧解作为下一帧 seed)"
    )
    parser.add_argument(
        "--no-freeze-base", action="store_true",
        help="关闭末端静止时底盘冻结 (默认开启：末端几乎没动时强制复用上一帧底盘)"
    )
    parser.add_argument(
        "--freeze-base-eps-pos", type=float, default=0.003,
        help="末端位移小于此值(米)视为静止 (默认 0.003m = 3mm)"
    )
    parser.add_argument(
        "--freeze-base-eps-rot", type=float, default=0.01,
        help="末端转角小于此值(rad)视为静止 (默认 0.01rad ≈ 0.57°)"
    )
    # --- 自适应底盘 (分层控制) ---
    parser.add_argument(
        "--adaptive-base", action="store_true",
        help="启用自适应底盘：平时用 7-DOF (base 锁死)，接近限位/奇异/IK失败时切到 10-DOF"
    )
    parser.add_argument(
        "--adaptive-margin-trigger", type=float, default=5.0,
        help="LOCKED→ACTIVE 触发阈值：关节距限位裕度(度) 默认 5°"
    )
    parser.add_argument(
        "--adaptive-margin-release", type=float, default=15.0,
        help="ACTIVE→LOCKED 释放阈值：关节距限位裕度(度) 默认 15°"
    )
    parser.add_argument(
        "--adaptive-release-hold", type=int, default=10,
        help="释放条件连续满足的帧数 默认 10"
    )
    parser.add_argument(
        "--adaptive-singular-j5", type=float, default=3.0,
        help="腕奇异阈值(度)：|J_5|<此值时触发 ACTIVE 默认 3°"
    )
    parser.add_argument(
        "--adaptive-singular-j3", type=float, default=5.0,
        help="肘奇异阈值(度)：|J_3|<此值时触发 ACTIVE 默认 5°"
    )
    args = parser.parse_args()

    warm_start = not args.no_warm_start
    freeze_base = not args.no_freeze_base
    eps_pos = float(args.freeze_base_eps_pos)
    eps_rot = float(args.freeze_base_eps_rot)

    # ========== 1. 解析数据 ==========
    if args.file:
        sample_path = os.path.abspath(args.file)
    else:
        sample_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "sample.txt")
        )
    print(f"读取数据文件: {sample_path}")

    timestamps, statuses, positions, quaternions, seed_joints = parse_sample_file(
        sample_path
    )
    total_count = len(timestamps)
    print(f"jaka_arm_left 总数据量: {total_count}")

    status_counts = {}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"原始状态分布: {status_counts}")

    # ========== 2. 初始化 CuRobo IK 求解器 (10 DOF) ==========
    print("\n初始化 CuRobo IK 求解器 (jaka_mobile.yml, 10 DOF)...")
    tensor_args = TensorDeviceType()

    robot_file = "jaka_mobile.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=args.num_seeds,
        self_collision_check=args.self_collision,
        self_collision_opt=args.self_collision,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    print(f"IK 求解器初始化完成，关节顺序: {ik_solver.joint_names}")
    print(f"总自由度: {len(ik_solver.joint_names)} "
          f"(base={BASE_DOF}, arm={ARM_DOF})")

    # ---- 可选：初始化 7-DOF IK 求解器 (供自适应底盘使用) ----
    adaptive_ctrl = None
    if args.adaptive_base:
        print("\n[自适应底盘] 初始化 7-DOF IK 求解器 (jaka.yml)...")
        arm_robot_cfg = RobotConfig.from_dict(
            load_yaml(join_path(get_robot_configs_path(), ARM_ROBOT_FILE))["robot_cfg"]
        )
        arm_ik_config = IKSolverConfig.load_from_robot_config(
            arm_robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=args.num_seeds,
            self_collision_check=args.self_collision,
            self_collision_opt=args.self_collision,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        ik_solver_arm = IKSolver(arm_ik_config)
        print(f"  7-DOF 关节顺序: {ik_solver_arm.joint_names}")

        # 从 URDF 读取真实关节限位
        urdf_abs = join_path(get_assets_path(), ARM_URDF_REL_PATH)
        arm_limits = _parse_joint_limits_from_urdf(urdf_abs, ARM_JOINT_NAMES)
        print(f"  从 URDF 读取关节限位 ({ARM_URDF_REL_PATH}):")
        for jn, (lo, up) in zip(ARM_JOINT_NAMES, arm_limits):
            print(f"    {jn}: [{lo:+.4f}, {up:+.4f}] rad")

        adaptive_ctrl = AdaptiveBaseController(
            ik_solver_full=ik_solver,
            ik_solver_arm=ik_solver_arm,
            arm_joint_limits=arm_limits,
            tensor_args=tensor_args,
            margin_trigger=np.deg2rad(args.adaptive_margin_trigger),
            margin_release=np.deg2rad(args.adaptive_margin_release),
            release_hold=args.adaptive_release_hold,
            singular_j5=np.deg2rad(args.adaptive_singular_j5),
            singular_j3=np.deg2rad(args.adaptive_singular_j3),
        )
        print(
            f"  阈值: trigger={args.adaptive_margin_trigger}°  "
            f"release={args.adaptive_margin_release}°  "
            f"hold={args.adaptive_release_hold}帧  "
            f"|J5|<{args.adaptive_singular_j5}°  |J3|<{args.adaptive_singular_j3}°"
        )

        # 预热 7-DOF CUDA Graph
        pos_b0, quat_b0 = _world_pose_to_linkbase_pose(
            positions[0], quaternions[0], np.zeros(3, dtype=np.float32)
        )
        _pos0 = torch.tensor(pos_b0.reshape(1, 3),
                             device=tensor_args.device, dtype=tensor_args.dtype)
        _quat0 = torch.tensor(quat_b0.reshape(1, 4),
                              device=tensor_args.device, dtype=tensor_args.dtype)
        _seed0 = torch.tensor(
            seed_joints[0:1].reshape(1, 1, ARM_DOF),
            device=tensor_args.device, dtype=tensor_args.dtype,
        )
        _ = ik_solver_arm.solve_single(Pose(_pos0, _quat0), seed_config=_seed0)
        torch.cuda.synchronize()
        print("  7-DOF CUDA Graph 预热完成")

    # 校验：确保 CuRobo 内部顺序与我们假设的 MOBILE_JOINT_NAMES 一致
    if list(ik_solver.joint_names) != MOBILE_JOINT_NAMES:
        print("[警告] CuRobo 解析出的关节顺序与脚本预期不一致！")
        print(f"  预期: {MOBILE_JOINT_NAMES}")
        print(f"  实际: {ik_solver.joint_names}")
        print("  后续 seed 拼接可能会错位，请检查 URDF / YAML。")

    # ========== 3. 构造 seed：前 3 维补 0 (底盘初始位于原点) ==========
    # CuRobo 期望 seed_config shape: (num_seeds_provided, batch, dof)
    # 这里单条求解 batch=1，num_seeds_provided=1
    arm_seed_all = torch.tensor(
        seed_joints, device=tensor_args.device, dtype=tensor_args.dtype
    )  # (N, 7)
    base_seed_zero = torch.zeros(
        (total_count, BASE_DOF), device=tensor_args.device, dtype=tensor_args.dtype
    )
    full_seed_all = torch.cat([base_seed_zero, arm_seed_all], dim=1)  # (N, 10)

    # ========== 4. 预热 CUDA Graph ==========
    print(f"\n开始逐条 IK 求解 ({total_count} 条)...")
    pos0 = torch.tensor(
        positions[0:1], device=tensor_args.device, dtype=tensor_args.dtype
    )
    quat0 = torch.tensor(
        quaternions[0:1], device=tensor_args.device, dtype=tensor_args.dtype
    )
    seed0 = full_seed_all[0:1].unsqueeze(0)  # (1, 1, 10)
    _ = ik_solver.solve_single(Pose(pos0, quat0), seed_config=seed0)
    torch.cuda.synchronize()
    print("CUDA Graph 预热完成\n")

    # ========== 5. 批量求解 ==========
    results_success = []
    results_pos_error = []
    results_rot_error = []
    results_solve_time = []
    results_full_solutions = []  # 保存 10 维解
    results_base_xyyaw = []      # 保存 (base_x, base_y, base_yaw) 方便分析

    # 抑制底盘漂移状态
    print(
        f"[抑制底盘漂移] warm_start={warm_start}  freeze_base={freeze_base}  "
        f"eps_pos={eps_pos*1000:.1f}mm  eps_rot={eps_rot:.4f}rad"
    )
    prev_base = None          # 上一帧成功解的底盘 [bx, by, byaw]
    prev_pos = None           # 上一帧目标末端位置 (3,)
    prev_quat = None          # 上一帧目标末端四元数 (wxyz,)
    freeze_count = 0          # 被冻结底盘的次数

    total_start = time.time()

    for i in range(total_count):
        pos_t = torch.tensor(
            positions[i: i + 1], device=tensor_args.device, dtype=tensor_args.dtype
        )
        quat_t = torch.tensor(
            quaternions[i: i + 1], device=tensor_args.device, dtype=tensor_args.dtype
        )
        goal = Pose(pos_t, quat_t)

        # ---- 构造 seed：默认底盘=0 + 原始 arm seed ----
        seed_vec = full_seed_all[i].clone()  # (10,)

        # ---- 末端差分：判断当前帧相对上一帧是否“几乎没动” ----
        end_stationary = False
        if prev_pos is not None and prev_quat is not None:
            dpos = float(np.linalg.norm(positions[i] - prev_pos))
            # 四元数夹角：angle = 2*acos(|<q1,q2>|)
            dot = float(abs(np.dot(quaternions[i], prev_quat)))
            dot = min(1.0, max(-1.0, dot))
            drot = 2.0 * float(np.arccos(dot))
            if dpos < eps_pos and drot < eps_rot:
                end_stationary = True

        # ---- 热启动：用上一帧底盘值覆盖 seed 的前 3 维 ----
        if warm_start and prev_base is not None:
            seed_vec[:BASE_DOF] = torch.tensor(
                prev_base, device=tensor_args.device, dtype=tensor_args.dtype
            )

        seed = seed_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, 10)

        # =========================
        # 分支：自适应底盘 vs 原 10-DOF 流程
        # =========================
        if adaptive_ctrl is not None:
            # 用 7-DOF 书写的 seed（原始不变）；prev_base 由控制器内部维护
            # freeze 语义由控制器内部接管：ACTIVE 下末端静止优先走 7-DOF 降级，
            # 成功则输出 [prev_base, q_arm7]，保证 base/arm 数学匹配 (无 FK 漂移)
            seed_arm_7_np = seed_joints[i]  # shape (7,)
            sol_full, success, _state, pos_err, rot_err, solve_t = adaptive_ctrl.step(
                frame_idx=i,
                pos_world=positions[i],
                quat_world=quaternions[i],
                seed_arm_7=seed_arm_7_np,
                freeze_if_stationary=freeze_base,
                end_stationary=end_stationary,
            )
            results_success.append(bool(success))
            results_pos_error.append(float(pos_err))
            results_rot_error.append(float(rot_err))
            results_solve_time.append(float(solve_t))
            if success and sol_full is not None:
                sol = np.asarray(sol_full, dtype=np.float32).copy()
                # 注：绝不在这里事后替换 base。控制器已保证返回的 [base, arm]
                # 是严格匹配的，如果这里再改 base 会直接破坏末端精度。
                results_full_solutions.append(sol)
                results_base_xyyaw.append(sol[:BASE_DOF].copy())
                prev_base = sol[:BASE_DOF].copy()
            else:
                results_full_solutions.append(None)
                results_base_xyyaw.append(None)
        else:
            st = time.time()
            result = ik_solver.solve_single(goal, seed_config=seed)
            torch.cuda.synchronize()
            solve_t = time.time() - st

            success = result.success.item()
            pos_err = result.position_error.item()
            rot_err = result.rotation_error.item()

            results_success.append(success)
            results_pos_error.append(pos_err)
            results_rot_error.append(rot_err)
            results_solve_time.append(solve_t)

            if success:
                sol = result.js_solution.position.cpu().numpy().flatten()  # (10,)
                # 注：原来这里有“末端静止时强制把 sol[:3] 改为 prev_base”的逻辑，
                # 但这会造成 base 与 arm 不匹配 (arm 是在 base_new 下优化出的)，
                # 直接导致末端 FK 偏移。已删除。warm_start 仍然会引导 IK
                # 自己选择“少动 base”的解，起到类似的抗抖动效果。
                results_full_solutions.append(sol)
                results_base_xyyaw.append(sol[:BASE_DOF].copy())
                prev_base = sol[:BASE_DOF].copy()
            else:
                results_full_solutions.append(None)
                results_base_xyyaw.append(None)
                # 失败帧不更新 prev_base，保留上一次成功解作为锚点

        # 更新末端参考 (不论成功与否，都基于“目标”做差分)
        prev_pos = positions[i].copy()
        prev_quat = quaternions[i].copy()

        if (i + 1) % 100 == 0 or (i + 1) == total_count:
            cur_ok = sum(results_success[: i + 1])
            print(
                f"  进度: {i+1}/{total_count}, "
                f"当前成功: {cur_ok}/{i+1} ({cur_ok/(i+1)*100:.1f}%)"
            )

    total_time = time.time() - total_start

    # ========== 6. 统计结果 ==========
    print("\n" + "=" * 70)
    print("        JAKA Left Arm IK (Mobile Base, 10 DOF) 求解统计")
    print("=" * 70)

    # 自适应底盘汇总
    if adaptive_ctrl is not None:
        print(f"\n{adaptive_ctrl.summary_str()}")
        if adaptive_ctrl.switch_log:
            print(f"  切换记录 (共 {len(adaptive_ctrl.switch_log)} 次):")
            for (fi, s_from, s_to, reason) in adaptive_ctrl.switch_log:
                print(f"    [frame {fi+1}] {s_from}→{s_to}  {reason}")

    success_count = sum(results_success)
    fail_count = total_count - success_count
    success_rate = success_count / total_count * 100

    print(f"\n【总体统计】")
    print(f"  总数据量:       {total_count}")
    print(f"  求解成功:       {success_count} ({success_rate:.2f}%)")
    print(f"  求解失败:       {fail_count} ({100 - success_rate:.2f}%)")
    print(f"  总耗时:         {total_time:.2f}s")
    print(f"  平均每条耗时:   {total_time/total_count*1000:.2f}ms")
    print(f"  求解频率:       {total_count/total_time:.1f}Hz")

    # 按原始状态分组
    print(f"\n【按原始状态分组统计】")
    for status in sorted(status_counts.keys()):
        idxs = [j for j, s in enumerate(statuses) if s == status]
        sub_ok = sum(results_success[j] for j in idxs)
        print(
            f"  {status:25s}: {sub_ok}/{len(idxs)} 成功 "
            f"({sub_ok/len(idxs)*100:.2f}%)"
        )

    # 底盘自由度利用情况分析
    used_bases = [b for b in results_base_xyyaw if b is not None]
    if used_bases:
        used_arr = np.stack(used_bases, axis=0)  # (K, 3)
        print(f"\n【底盘自由度使用情况】(仅成功样本, 共 {len(used_bases)} 条)")
        labels = ["base_x (m)", "base_y (m)", "base_yaw (rad)"]
        for k, lab in enumerate(labels):
            col = used_arr[:, k]
            print(
                f"  {lab:15s}  mean_abs={np.mean(np.abs(col)):.4f}  "
                f"min={col.min():.4f}  max={col.max():.4f}  "
                f"std={col.std():.4f}"
            )
        # 有多少样本底盘几乎没动
        near_zero = np.all(np.abs(used_arr) < np.array([0.01, 0.01, 0.01]), axis=1)
        print(f"  底盘几乎未移动 (|x|,|y|<1cm 且 |yaw|<0.57°): "
              f"{int(near_zero.sum())}/{len(used_bases)} 条")

        # ---- 底盘帧间跳变统计 (仅统计“相邻两帧都成功”的帧对) ----
        diffs = []
        for k in range(1, total_count):
            b_prev = results_base_xyyaw[k - 1]
            b_cur = results_base_xyyaw[k]
            if b_prev is None or b_cur is None:
                continue
            diffs.append(np.abs(b_cur - b_prev))
        if diffs:
            diffs_arr = np.stack(diffs, axis=0)  # (M, 3)
            print(
                f"  相邻帧底盘跳变 (|Δ|, 共 {len(diffs)} 对): "
                f"mean_dx={np.mean(diffs_arr[:,0])*1000:.2f}mm  "
                f"mean_dy={np.mean(diffs_arr[:,1])*1000:.2f}mm  "
                f"mean_dyaw={np.degrees(np.mean(diffs_arr[:,2])):.3f}°"
            )
            print(
                f"                                  "
                f"max_dx={np.max(diffs_arr[:,0])*1000:.2f}mm  "
                f"max_dy={np.max(diffs_arr[:,1])*1000:.2f}mm  "
                f"max_dyaw={np.degrees(np.max(diffs_arr[:,2])):.3f}°"
            )
        if adaptive_ctrl is not None:
            print(
                f"  底盘freeze降级 (ACTIVE下末端静止→7DOF,base/arm严格匹配): "
                f"成功={adaptive_ctrl.n_freeze_downgrade_ok}  "
                f"回退10DOF={adaptive_ctrl.n_freeze_downgrade_fail}"
            )
        else:
            print(f"  底盘被冻结次数 (旧逻辑已移除, 始终为 0): {freeze_count}")

    # 误差统计（与 7-DOF 脚本一致的格式）
    ok_idx = [j for j in range(total_count) if results_success[j]]
    if ok_idx:
        pe = [results_pos_error[j] for j in ok_idx]
        re = [results_rot_error[j] for j in ok_idx]
        tt = [results_solve_time[j] for j in ok_idx]
        print(f"\n【成功样本误差统计】(共 {len(ok_idx)} 条)")
        print(f"  位置误差 (mm): mean={np.mean(pe)*1000:.4f}  "
              f"max={np.max(pe)*1000:.4f}  median={np.median(pe)*1000:.4f}")
        print(f"  旋转误差 (rad): mean={np.mean(re):.6f}  "
              f"max={np.max(re):.6f}  median={np.median(re):.6f}")
        print(f"  求解时间 (ms): mean={np.mean(tt)*1000:.2f}  "
              f"max={np.max(tt)*1000:.2f}")

    fail_idx = [j for j in range(total_count) if not results_success[j]]
    if fail_idx:
        pe = [results_pos_error[j] for j in fail_idx]
        re = [results_rot_error[j] for j in fail_idx]
        print(f"\n【失败样本误差统计】(共 {len(fail_idx)} 条)")
        print(f"  位置误差 (mm): mean={np.mean(pe)*1000:.4f}  "
              f"max={np.max(pe)*1000:.4f}")
        print(f"  旋转误差 (rad): mean={np.mean(re):.6f}  max={np.max(re):.6f}")

        print(f"\n【前10条失败样本详情】")
        for k, j in enumerate(fail_idx[:10]):
            print(
                f"  [{k+1}] t={timestamps[j]:.6f}, "
                f"status={statuses[j]}, "
                f"pos=({positions[j][0]:.4f}, {positions[j][1]:.4f}, "
                f"{positions[j][2]:.4f}), "
                f"pos_err={results_pos_error[j]*1000:.4f}mm, "
                f"rot_err={results_rot_error[j]:.6f}rad"
            )

    print("\n" + "=" * 70)
    print("求解完成")
    print("=" * 70)

    # ========== 7. ROS 发布 (可选) ==========
    if args.publish:
        publish_to_ros(
            timestamps=timestamps,
            statuses=statuses,
            positions=positions,
            quaternions=quaternions,
            seed_joints=seed_joints,  # 原始 7 维 seed，内部再补底盘 3 维 0
            results_success=results_success,
            results_full_solutions=results_full_solutions,  # 10 维解
            topic=args.topic,
            speed=args.speed,
            loop=args.loop,
            focus_failed=args.focus_failed,
            context_size=args.context_size,
        )


def _build_focus_failed_segments(statuses, results_success, context_size, total):
    """构建 focus-failed 模式下需要播放的帧索引列表和片段信息。

    找出所有 CuRobo IK 求解失败的帧，将其及前后 context_size 帧合并为连续片段。
    """
    failed_indices = [i for i in range(total) if not results_success[i]]
    if not failed_indices:
        return [], set()

    raw_ranges = []
    for fi in failed_indices:
        start = max(0, fi - context_size)
        end = min(total - 1, fi + context_size)
        raw_ranges.append((start, end))

    merged = [raw_ranges[0]]
    for s, e in raw_ranges[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e + 1:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))

    segments = []
    play_indices_set = set()
    for seg_start, seg_end in merged:
        seg_failed = [fi for fi in failed_indices if seg_start <= fi <= seg_end]
        segments.append({
            "start": seg_start,
            "end": seg_end,
            "failed_indices": seg_failed,
        })
        for idx in range(seg_start, seg_end + 1):
            play_indices_set.add(idx)

    return segments, play_indices_set


def publish_to_ros(
    timestamps,
    statuses,
    positions,
    quaternions,
    seed_joints,
    results_success,
    results_full_solutions,
    topic="/joint_states",
    speed=1.0,
    loop=False,
    focus_failed=False,
    context_size=5,
):
    """将 10-DOF IK 求解结果发布到 ROS JointState 话题，用于 RViz 可视化。

    关节名顺序: [base_x, base_y, base_yaw, J_1, ..., J_7]
    - 成功样本: 发布 CuRobo 求解得到的 10 维关节角 (含底盘自由度)
    - 失败样本: 发布 [0,0,0, seed_arm_7] (底盘补零，机械臂用原始 seed)
    同时在 /ik_target_pose 话题上以 world 系发布目标位姿。
    """
    try:
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Header
        from geometry_msgs.msg import PoseStamped
    except ImportError:
        print("\n[错误] 未找到 rospy / sensor_msgs，请确保已安装 ROS 并 source 了环境。")
        print("  例如: source /opt/ros/noetic/setup.bash")
        return

    rospy.init_node("jaka_left_ik_mobile_visualizer", anonymous=True)
    joint_pub = rospy.Publisher(topic, JointState, queue_size=10)
    pose_pub = rospy.Publisher("/ik_target_pose", PoseStamped, queue_size=10)

    total = len(timestamps)

    # ---- 构建播放列表 ----
    if focus_failed:
        segments, play_set = _build_focus_failed_segments(
            statuses, results_success, context_size, total
        )
        if not segments:
            print("\n[提示] 所有样本均求解成功，无失败片段可播放。")
            return
        play_indices = sorted(play_set)
        effective_speed = speed * 0.3
        mode_name = "失败片段模式 (focus-failed)"
    else:
        play_indices = list(range(total))
        segments = None
        effective_speed = speed
        mode_name = "全量模式"

    # ---- 统计 txt 中失败但 CuRobo 成功的 case ----
    txt_failed_curobo_success = [
        i for i in range(total)
        if statuses[i] in ("solver_failed", "large_joint_change") and results_success[i]
    ]
    txt_failed_curobo_success_set = set(txt_failed_curobo_success)

    print(f"\n{'='*70}")
    print(f"  播放模式: {mode_name}")
    print(f"  发布话题: {topic}  (10 个关节)")
    print(f"  关节名顺序: {MOBILE_JOINT_NAMES}")
    print(f"  回放速率: {effective_speed:.2f}x | 循环: {loop}")
    if focus_failed:
        print(f"  失败片段数: {len(segments)}, 总播放帧数: {len(play_indices)}/{total}")
        print(f"  上下文帧数: 前后各 {context_size} 帧")
    print(f"  成功样本: 发布 CuRobo 求解 10 维关节角")
    print(f"  失败样本: 底盘补 0 + 原始 7 维 seed (终端标注 [SEED])")
    if txt_failed_curobo_success:
        print(f"  ★ txt标记失败但CuRobo成功: {len(txt_failed_curobo_success)} 条 (终端高亮 ★)")
    print(f"  目标位姿同步发布到: /ik_target_pose  (frame_id=world)")
    print(f"  按 Ctrl+C 停止")
    print(f"{'='*70}\n")

    rate_hz = 50
    ros_rate = rospy.Rate(rate_hz)

    # ANSI 颜色
    COLOR_RED = "\033[91m"
    COLOR_GREEN = "\033[92m"
    COLOR_YELLOW = "\033[93m"
    COLOR_CYAN = "\033[96m"
    COLOR_RESET = "\033[0m"

    try:
        while not rospy.is_shutdown():
            current_segment_idx = 0

            for play_pos, i in enumerate(play_indices):
                if rospy.is_shutdown():
                    break

                # focus-failed 模式下，打印片段分隔线
                if focus_failed and segments:
                    seg = segments[current_segment_idx]
                    if i == seg["start"]:
                        n_failed_in_seg = len(seg["failed_indices"])
                        print(
                            f"\n{COLOR_CYAN}{'─'*70}\n"
                            f"  ▶ 片段 {current_segment_idx + 1}/{len(segments)}  "
                            f"帧范围 [{seg['start']+1}~{seg['end']+1}]  "
                            f"含 {n_failed_in_seg} 个失败帧\n"
                            f"{'─'*70}{COLOR_RESET}"
                        )
                    if i > seg["end"] and current_segment_idx + 1 < len(segments):
                        current_segment_idx += 1

                # 确定要发布的 10 维关节值
                if results_success[i] and results_full_solutions[i] is not None:
                    joint_values = list(map(float, results_full_solutions[i]))
                    source_tag = "IK"
                else:
                    # 底盘补 0 + 原始 7 维 seed
                    joint_values = [0.0, 0.0, 0.0] + list(map(float, seed_joints[i]))
                    source_tag = "SEED"

                # 构造 JointState
                js_msg = JointState()
                js_msg.header = Header()
                js_msg.header.stamp = rospy.Time.now()
                js_msg.name = MOBILE_JOINT_NAMES
                js_msg.position = joint_values

                # 构造目标位姿 (世界系)
                pose_msg = PoseStamped()
                pose_msg.header.stamp = js_msg.header.stamp
                pose_msg.header.frame_id = "world"
                pose_msg.pose.position.x = float(positions[i][0])
                pose_msg.pose.position.y = float(positions[i][1])
                pose_msg.pose.position.z = float(positions[i][2])
                pose_msg.pose.orientation.w = float(quaternions[i][0])
                pose_msg.pose.orientation.x = float(quaternions[i][1])
                pose_msg.pose.orientation.y = float(quaternions[i][2])
                pose_msg.pose.orientation.z = float(quaternions[i][3])

                joint_pub.publish(js_msg)
                pose_pub.publish(pose_msg)

                # ---- 终端打印 ----
                status_str = statuses[i]
                base_str = "[{:+.3f}, {:+.3f}, {:+.3f}]".format(
                    joint_values[0], joint_values[1], joint_values[2]
                )
                arm_str = ", ".join(f"{v:.3f}" for v in joint_values[3:])

                is_highlight = (i in txt_failed_curobo_success_set)
                if is_highlight:
                    color = COLOR_YELLOW
                    prefix = "★ TXT失败/CuRobo成功"
                    success_mark = "✓★"
                elif not results_success[i]:
                    color = COLOR_RED
                    prefix = ""
                    success_mark = "✗"
                else:
                    color = COLOR_GREEN
                    prefix = ""
                    success_mark = "✓"

                line = (
                    f"  [{i+1:4d}/{total}] [{source_tag:4s}] {success_mark} "
                    f"状态={status_str:20s} "
                    f"base={base_str}  arm=[{arm_str}]"
                )
                if prefix:
                    line += f"  {prefix}"

                print(f"{color}{line}{COLOR_RESET}")

                # ---- 控制回放速度 ----
                if play_pos + 1 < len(play_indices):
                    next_i = play_indices[play_pos + 1]
                    dt = timestamps[next_i] - timestamps[i]
                    if dt > 0:
                        sleep_time = dt / effective_speed
                        sleep_time = min(sleep_time, 2.0)
                        rospy.sleep(sleep_time)
                    else:
                        ros_rate.sleep()
                else:
                    ros_rate.sleep()

            if not loop:
                print("\n回放完成。")
                break
            else:
                print(f"\n--- 循环回放，重新开始 ---\n")

    except rospy.ROSInterruptException:
        print("\n[收到 ROS 中断信号] 停止发布。")
    except KeyboardInterrupt:
        print("\n[收到键盘中断] 停止发布。")


if __name__ == "__main__":
    main()
