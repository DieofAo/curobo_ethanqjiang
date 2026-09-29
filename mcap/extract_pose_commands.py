#!/usr/bin/env python3
"""
从mcap文件中提取 /robot/data/jaka_arm_left/action 的 pose_command 数据，
保存为numpy文件以便后续使用。
"""
import numpy as np
from mcap_ros1.reader import read_ros1_messages

MCAP_PATH = "/home/ethanqjiang/workspace/curobo/mcap/20260525-100300_optimized.mcap"
OUTPUT_PATH = "/home/ethanqjiang/workspace/curobo/mcap/left_arm_pose_commands.npz"
TOPIC = "/robot/data/jaka_arm_left/action"


def main():
    print(f"读取 mcap 文件: {MCAP_PATH}")
    print(f"话题: {TOPIC}")

    timestamps_ns = []  # log_time in nanoseconds
    positions = []      # [x, y, z]
    quaternions_xyzw = []  # [x, y, z, w] (原始格式)

    count = 0
    for msg in read_ros1_messages(MCAP_PATH, topics=[TOPIC]):
        decoded_msg = msg.ros_msg
        p = decoded_msg.pose_command.pose.position
        o = decoded_msg.pose_command.pose.orientation

        # log_time 可能是 datetime 或 int(ns)
        lt = msg.log_time
        if hasattr(lt, 'timestamp'):
            # datetime -> nanoseconds
            timestamps_ns.append(int(lt.timestamp() * 1e9))
        else:
            timestamps_ns.append(int(lt))
        positions.append([p.x, p.y, p.z])
        quaternions_xyzw.append([o.x, o.y, o.z, o.w])

        count += 1
        if count % 1000 == 0:
            print(f"  已读取 {count} 条消息...")

    print(f"总共读取 {count} 条消息")

    if count == 0:
        print("未找到任何消息!")
        return

    timestamps_ns = np.array(timestamps_ns, dtype=np.int64)
    positions = np.array(positions, dtype=np.float64)
    quaternions_xyzw = np.array(quaternions_xyzw, dtype=np.float64)

    # 打印前几条数据
    print(f"\n前3条数据:")
    for i in range(min(3, count)):
        print(f"  [{i}] t={timestamps_ns[i]}, pos={positions[i]}, quat_xyzw={quaternions_xyzw[i]}")

    # 时间间隔统计
    if count > 1:
        dt_ns = np.diff(timestamps_ns)
        dt_ms = dt_ns / 1e6
        print(f"\n时间间隔统计:")
        print(f"  mean={dt_ms.mean():.2f} ms, std={dt_ms.std():.2f} ms")
        print(f"  min={dt_ms.min():.2f} ms, max={dt_ms.max():.2f} ms")
        print(f"  频率约 {1000.0 / dt_ms.mean():.1f} Hz")

    # 保存
    np.savez_compressed(
        OUTPUT_PATH,
        timestamps_ns=timestamps_ns,
        positions=positions,
        quaternions_xyzw=quaternions_xyzw,
    )
    print(f"\n数据已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
