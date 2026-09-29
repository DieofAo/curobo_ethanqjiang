#!/usr/bin/env python3
"""检查mcap文件的话题和消息格式"""
import sys
from mcap.reader import make_reader

mcap_path = "20260525-100300_optimized.mcap"

with open(mcap_path, 'rb') as f:
    reader = make_reader(f)
    summary = reader.get_summary()
    if summary is None:
        print("No summary available, trying to iterate...")
        sys.exit(1)

    print("=== Schemas ===")
    for s_id, schema in summary.schemas.items():
        print(f"  ID={s_id}, name={schema.name}, encoding={schema.encoding}")
        if schema.encoding in ('jsonschema', 'ros2msg', 'ros1msg'):
            print(f"    data: {schema.data.decode('utf-8', errors='replace')[:300]}")

    print("\n=== Channels (filtered for pose_command/jaka_arm_left) ===")
    for ch_id, ch in summary.channels.items():
        if 'pose_command' in ch.topic or 'jaka_arm_left' in ch.topic:
            print(f"  ID={ch_id}, topic={ch.topic}, schema_id={ch.schema_id}, encoding={ch.message_encoding}")

    print("\n=== All topics ===")
    for ch_id, ch in summary.channels.items():
        print(f"  {ch.topic} (encoding={ch.message_encoding})")
