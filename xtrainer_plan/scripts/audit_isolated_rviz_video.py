#!/usr/bin/env python3
"""Decode every RViz video frame and check scene visibility and run coverage.

The capture is a planned-trajectory visualization. This audit detects a blank
RViz scene and the desktop-window contamination seen in earlier XTrainer clips.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def ffprobe(path: Path) -> tuple[int, int, int, str]:
    cmd = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate',
           '-of', 'json', str(path)]
    value = json.loads(subprocess.check_output(cmd, text=True))['streams'][0]
    return int(value['width']), int(value['height']), int(value['nb_read_frames']), value['r_frame_rate']


def scan_scene(path: Path, x: int, top: int) -> dict:
    crop = f'crop=1280:800:{x}:{top},scale=160:100,format=gray'
    cmd = ['ffmpeg', '-v', 'error', '-i', str(path), '-vf', crop,
           '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    black, blocked = [], []
    bright_min = 16000
    bright_max = white_max = 0
    index = 0
    assert process.stdout is not None
    while True:
        frame = process.stdout.read(16000)
        if not frame:
            break
        if len(frame) != 16000:
            process.kill()
            raise ValueError(f'Incomplete decoded frame {index}: {len(frame)} bytes')
        gray = np.frombuffer(frame, dtype=np.uint8)
        bright = int(np.count_nonzero(gray > 70))
        white = int(np.count_nonzero(gray > 230))
        bright_min = min(bright_min, bright)
        bright_max = max(bright_max, bright)
        white_max = max(white_max, white)
        if bright < 100:
            black.append(index)
        if white > 800:
            blocked.append(index)
        index += 1
    error = process.stderr.read().decode('utf-8', 'replace') if process.stderr else ''
    if process.wait() != 0:
        raise RuntimeError(f'ffmpeg decode failed: {error[-1000:]}')
    return {'decoded_frames': index, 'bright_pixel_count_min': bright_min,
            'bright_pixel_count_max': bright_max, 'near_white_pixel_count_max': white_max,
            'black_scene_frame_indices': black, 'occluded_scene_frame_indices': blocked,
            'scene_scan': '1280x800 RViz area reduced to 160x100 grayscale; black if <100 pixels >70; occluded if >800 pixels >230'}


def verify_full_source(audit: dict, run: Path, decoded: int) -> dict:
    with (run / 'trajectory.npz').open('rb') as stream:
        source_sha = hashlib.sha256(stream.read()).hexdigest()
    assert source_sha == audit['source_npz_sha256']
    meta = json.loads((run / 'trajectory_meta.json').read_text())
    assert sha256(run / 'trajectory_meta.json') == audit['source_meta_sha256']
    successes = [item for item in meta['items'] if item.get('success')]
    assert len(successes) == audit['n_successful_cases']
    n = int(meta['n_points'])
    selected = list(range(0, n, 40))
    if selected[-1] != n - 1:
        selected.append(n - 1)
    assert len(selected) == decoded == audit['n_video_frames']
    assert len(audit['frames']) == decoded
    assert [f['source_sample_index'] for f in audit['frames']] == selected
    assert [row['case_number'] for row in audit['case_coverage']] == [int(item['index'])+1 for item in successes]
    assert all(row['n_video_frames'] > 0 for row in audit['case_coverage'])
    assert sum(row['n_video_frames'] for row in audit['case_coverage']) == decoded
    assert sum(int(item['n_points']) for item in successes) == n
    assert all(f['tf_stamp_ns'] > 0 and f['rviz_render_count'] > 0 for f in audit['frames'])
    return {'source_samples': n, 'successful_cases': len(successes),
            'min_frames_per_case': min(row['n_video_frames'] for row in audit['case_coverage']),
            'max_frames_per_case': max(row['n_video_frames'] for row in audit['case_coverage']),
            'first_source_index': selected[0], 'last_source_index': selected[-1],
            'source_sha256_match': True, 'all_successful_cases_have_frames': True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--mode', choices=['full', 'clip', 'comparison'], required=True)
    parser.add_argument('--render-audit', type=Path)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    video = args.video.resolve()
    width, height, frames, fps = ffprobe(video)
    if fps != '50/1':
        raise ValueError(f'Expected exactly 50 fps, got {fps}')
    expected_width = 2560 if args.mode == 'comparison' else 1280
    expected_height = 936 if args.mode != 'full' else 916
    if (width, height) != (expected_width, expected_height):
        raise ValueError(f'Unexpected video size {width}x{height}')
    top = 81 if args.mode == 'full' else 92
    scans = {'left': scan_scene(video, 0, top)}
    if args.mode == 'comparison':
        scans['right'] = scan_scene(video, 1280, top)
    if any(scan['decoded_frames'] != frames or scan['black_scene_frame_indices']
           or scan['occluded_scene_frame_indices'] for scan in scans.values()):
        raise RuntimeError(f'Video has missing, blank, or occluded scene frames: {scans}')
    summary = {'video': str(video), 'sha256': sha256(video), 'mode': args.mode,
               'width': width, 'height': height, 'fps': fps, 'n_frames': frames,
               'duration_s': frames / 50, 'scans': scans}
    if args.mode == 'full':
        if not args.render_audit or not args.source_run:
            raise ValueError('--render-audit and --source-run required for full video')
        audit = json.loads(args.render_audit.read_text())
        if audit['video_sha256'] != summary['sha256']:
            raise ValueError('Video differs from render audit SHA256')
        summary['source_coverage'] = verify_full_source(audit, args.source_run.resolve(), frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f'PASS {video}: {frames} frames; blank=0, occluded=0')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
