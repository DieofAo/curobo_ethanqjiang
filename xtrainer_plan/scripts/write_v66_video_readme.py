#!/usr/bin/env python3
"""Write a source-grounded video index for one audited V66 full mount run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--result', type=Path, required=True)
    ap.add_argument('--videos', type=Path, required=True)
    args = ap.parse_args()
    run = args.result.resolve()
    videos = args.videos.resolve()
    label = run.name
    meta = json.loads((run / 'trajectory_meta.json').read_text())
    full = json.loads((videos / 'full_40x/video_decode_audit.json').read_text())
    extreme_root = videos / 'joint_span_extremes'
    manifest = json.loads((extreme_root / 'manifest.json').read_text())
    comparison = json.loads((extreme_root / 'side_by_side_manifest.json').read_text())
    comparison_audit = json.loads((extreme_root / 'side_by_side_video_audit.json').read_text())
    if full['source_coverage']['successful_cases'] != meta['n_items_success']:
        raise ValueError('Full video audit differs from run success count')
    if comparison_audit['sha256'] != comparison['sha256']:
        raise ValueError('Comparison video audit differs from manifest')
    selected = {row['label']: row for row in manifest['selected']}
    first_item = next(item for item in meta['items'] if item.get('success'))
    first_success = int(first_item['index']) + 1
    with np.load(run / 'trajectory.npz', allow_pickle=False) as archive:
        first_q = np.asarray(archive['positions'][:int(first_item['n_points'])], dtype=float)
    first_spans = np.degrees(np.ptp(first_q, axis=0))
    first_joint = int(np.argmax(first_spans)) + 1
    first_span = float(np.max(first_spans))
    if manifest['excluded_original_case_numbers_1based'] != [first_success]:
        raise ValueError('Extreme selection does not exclude first saved trajectory')
    option = meta['config']['overhead']['variant_options']
    position = option['mount_position']
    tilt = option['local_y_tilt_deg']
    parts = [
        f'# {label} 保存轨迹视频', '',
        f'基座原点 `({position[0]:+.2f}, {position[1]:+.2f}, {position[2]:+.2f}) m`，绕原始基座局部 `+Y` 轴倾斜 `{tilt:g}°`。', '',
        f'- [完整全量视频](full_40x/{label}_complete_40x.mp4)：`{meta["n_items_success"]}/400` 个成功 case，原轨迹 `{meta["n_points"]}` 个 20 ms 采样点；每 40 点取一帧并额外保留最后一帧，50 fps，名义 40 倍速，成片 `{full["n_frames"]}` 帧、`{full["duration_s"]:.2f}` 秒。失败 case 没有轨迹，因此画面编号会跳过。',
        f'- [最大/最小关节跨度左右对比](joint_span_extremes/{Path(comparison["output"]).name})：左右都包含完整六阶段抓放原始采样；每原始采样一帧，50 fps，1 倍速。较短一侧在末帧保持至两侧等长。',
        '- [首帧](full_40x/preview.png) · [完整视频逐帧审计](full_40x/video_decode_audit.json) · [极值抽取清单](joint_span_extremes/manifest.json) · [对比视频逐帧审计](joint_span_extremes/side_by_side_video_audit.json)', '',
        '| 画面 | 原始 case | 抓取点 (m) | 完整循环的最大单关节角跨度 | 原始采样点数 |',
        '| --- | ---: | --- | --- | ---: |',
    ]
    for side, name in (('左：最大','max'),('右：最小','min')):
        row = selected[name]
        p = row['position_raw']
        parts.append(f'| {side} | {row["original_case_number_1based"]} | '
                     f'({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) | '
                     f'{row["max_span_joint"]} {row["max_joint_span_deg"]:.3f}° | {row["n_points"]} |')
    parts.extend(['',
        f'极值选择排除首条**实际保存的成功轨迹**（原始 case {first_success}，含从共同 Home 入场的运动，'
        f'其最大单关节角跨度为 J{first_joint} {first_span:.3f}°）；失败 case 本来就没有轨迹。'
        f'比较范围为剩余 `{manifest["n_eligible_successful_cases"]}` 条成功轨迹。指标是每个 case 完整六阶段中，'
        '先对六个关节分别计算保存角度的最大值减最小值，再取六者中的最大值；这不是累计转角。', '',
        '录制使用独立 Xephyr 显示、独立 ROS master 和软件 OpenGL。每个画面在对应的关节 TF 到达且 RViz 至少完成两次渲染后截取；'
        '这是已保存规划轨迹的离线可视化，不代表真机执行。三个视频已全部逐帧解码，场景黑帧和桌面面板遮挡帧均为 0。', '',
        '单独播放：', '',
        '```bash',
        f'ffplay -autoexit {videos / "full_40x" / (label + "_complete_40x.mp4")}',
        f'ffplay -autoexit {comparison["output"]}',
        '```', '',
    ])
    target = videos / 'README.md'
    target.write_text('\n'.join(parts), encoding='utf-8')
    print(target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
