#!/usr/bin/env python3
"""Combine exact saved max/min XTrainer case recordings, with frame-count checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def frames(path: Path) -> tuple[int, int, int, str]:
    cmd = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate',
           '-of', 'json', str(path)]
    info = json.loads(subprocess.check_output(cmd, text=True))['streams'][0]
    return int(info['nb_read_frames']), int(info['width']), int(info['height']), info['r_frame_rate']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    selected = {row['label']: row for row in manifest['selected']}
    if set(selected) != {'min', 'max'}:
        raise ValueError('Expected exactly min and max extracted cases')
    sources = {}
    for label in ('max', 'min'):
        number = selected[label]['original_case_number_1based']
        path = root / f'{label}_case_{number:03d}.mp4'
        audit = json.loads((root / label / 'video_render_audit.json').read_text())
        n, width, height, fps = frames(path)
        if (width, height, fps) != (1280, 936, '50/1') or n != audit['n_video_frames']:
            raise ValueError(f'Unexpected {label} clip frames/size/fps')
        if audit['source_npz_sha256'] != selected[label]['output_sha256']['trajectory.npz']:
            raise ValueError(f'{label} rendered source differs from extracted source')
        if sha256(path) != audit['video_sha256']:
            raise ValueError(f'{label} clip differs from renderer SHA256')
        sources[label] = {'file': path, 'frames': n, 'case': number,
                          'joint': selected[label]['max_span_joint'],
                          'span_deg': selected[label]['max_joint_span_deg']}
    target_frames = max(row['frames'] for row in sources.values())
    output = root / f'max{sources["max"]["case"]:03d}_left_min{sources["min"]["case"]:03d}_right.mp4'
    if output.exists():
        raise FileExistsError(output)
    parts = []
    for n, label in enumerate(('max', 'min')):
        pad = (target_frames - sources[label]['frames']) / 50 + 0.1
        parts.append(f'[{n}:v]tpad=stop_mode=clone:stop_duration={pad:.4f},'
                     f'trim=end_frame={target_frames},setpts=N/(50*TB)[{label}]')
    parts.append('[max][min]hstack=inputs=2[v]')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n',
               '-i', str(sources['max']['file']), '-i', str(sources['min']['file']),
               '-filter_complex', ';'.join(parts), '-map', '[v]', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    subprocess.run(command, check=True)
    actual_frames, width, height, fps = frames(output)
    if (actual_frames, width, height, fps) != (target_frames, 2560, 936, '50/1'):
        raise ValueError('Unexpected combined video frames/size/fps')
    combined = {'output': str(output), 'sha256': sha256(output), 'frames': actual_frames,
                'duration_s': actual_frames/50, 'fps': 50, 'resolution': [width, height],
                'left': {k: str(v) if isinstance(v, Path) else v for k,v in sources['max'].items()},
                'right': {k: str(v) if isinstance(v, Path) else v for k,v in sources['min'].items()},
                'method': 'Exact original 20 ms samples at 50 fps; one-second holds at each end; shorter clip holds last frame to match longer clip.'}
    (root / 'side_by_side_manifest.json').write_text(json.dumps(combined, indent=2))
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
