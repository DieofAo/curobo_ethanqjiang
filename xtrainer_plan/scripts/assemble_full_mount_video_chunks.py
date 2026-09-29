#!/usr/bin/env python3
"""Assemble fresh-RViz video chunks with exact source-frame and TF audits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from audit_isolated_rviz_video import ffprobe, scan_scene


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def decoded_frame_hashes(path: Path) -> list[str]:
    """Hash actual decoded pixels and require exact 50 Hz presentation timestamps."""
    output = subprocess.check_output(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(path),
         '-f', 'framemd5', '-'], text=True)
    if '#tb 0: 1/50' not in output:
        raise ValueError(f'Unexpected decoded frame time base: {path}')
    hashes = []
    for line in output.splitlines():
        if not line or line.startswith('#'):
            continue
        columns = [part.strip() for part in line.split(',')]
        if len(columns) != 6:
            raise ValueError(f'Invalid frame MD5 row: {line}')
        _, dts, pts, duration, size, md5 = columns
        if int(pts) != len(hashes) or int(duration) != 1 or int(size) <= 0:
            raise ValueError(f'Non-contiguous 50 Hz PTS at decoded frame {len(hashes)} of {path}')
        if len(md5) != 32:
            raise ValueError(f'Invalid decoded-frame MD5 at frame {len(hashes)} of {path}')
        hashes.append(md5)
    return hashes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    out = args.out_dir.resolve()
    segments = out / 'segments'
    meta = json.loads((run / 'trajectory_meta.json').read_text())
    n_source = int(meta['n_points'])
    selected = list(range(0, n_source, 40))
    if selected[-1] != n_source - 1:
        selected.append(n_source - 1)
    audits = [json.loads(path.read_text()) for path in segments.glob('video_render_audit*.json')]
    audits.sort(key=lambda row: row['video_frame_range_half_open'][0])
    if not audits:
        raise ValueError('No chunk render audits')
    cursor = 0
    parts = []
    full_evidence = []
    expected_pixel_hashes = []
    source_npz_sha = sha256(run / 'trajectory.npz')
    source_meta_sha = sha256(run / 'trajectory_meta.json')
    coverage = audits[0]['case_coverage']
    for audit in audits:
        lo, hi = audit['video_frame_range_half_open']
        if lo != cursor or hi <= lo or hi > len(selected):
            raise ValueError(f'Chunk frame ranges overlap or skip: {lo}:{hi} after {cursor}')
        path = Path(audit['output'])
        if not path.is_file() or sha256(path) != audit['video_sha256']:
            raise ValueError(f'Chunk MP4 changed: {path}')
        width, height, frames, fps = ffprobe(path)
        if (width, height, frames, fps) != (1280, 916, hi-lo, '50/1'):
            raise ValueError(f'Invalid encoded chunk: {path}')
        scan = scan_scene(path, 0, 81)
        if scan['decoded_frames'] != frames or scan['black_scene_frame_indices'] or scan['occluded_scene_frame_indices']:
            raise ValueError(f'Chunk contains missing or invalid frames: {path}')
        pixel_hashes = decoded_frame_hashes(path)
        if len(pixel_hashes) != frames:
            raise ValueError(f'Chunk decoded pixel count differs: {path}')
        expected_pixel_hashes.extend(pixel_hashes)
        if audit['source_npz_sha256'] != source_npz_sha or audit['source_meta_sha256'] != source_meta_sha:
            raise ValueError('Chunk source hashes differ from audited run')
        if audit['case_coverage'] != coverage:
            raise ValueError('Chunk case coverage geometry differs')
        evidence = audit['frames']
        if len(evidence) != frames or [row['source_sample_index'] for row in evidence] != selected[lo:hi]:
            raise ValueError('Chunk source sample indices differ from expected selected frames')
        if any(row['tf_stamp_ns'] <= 0 or row['rviz_render_count'] <= 0 for row in evidence):
            raise ValueError('Missing TF/render evidence for a chunk frame')
        full_evidence.extend(evidence)
        parts.append({'file': str(path), 'sha256': audit['video_sha256'], 'frame_range_half_open': [lo,hi],
                      'source_sample_range_inclusive': [selected[lo], selected[hi-1]],
                      'scan': scan})
        cursor = hi
    if cursor != len(selected):
        raise ValueError(f'Missing final frames {cursor}..{len(selected)-1}')
    concat_file = out / 'segments.ffconcat'
    concat_file.write_text('ffconcat version 1.0\n' + ''.join("file '" + row['file'].replace("'", "'\\''") + "'\n" for row in parts))
    output = out / f'{args.label}_complete_40x.mp4'
    if output.exists():
        raise FileExistsError(output)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','warning','-n','-f','concat','-safe','0',
                    '-i',str(concat_file),'-c','copy','-movflags','+faststart',str(output)],check=True)
    width,height,frames,fps = ffprobe(output)
    if (width,height,frames,fps) != (1280,916,len(selected),'50/1'):
        raise ValueError('Assembled video frame count/format differs')
    final_pixel_hashes = decoded_frame_hashes(output)
    if final_pixel_hashes != expected_pixel_hashes:
        mismatch = next((index for index,(a,b) in enumerate(zip(final_pixel_hashes,expected_pixel_hashes)) if a != b), None)
        raise ValueError(f'Assembled decoded frames differ from source chunks at video frame {mismatch}')
    decoded_content_sha = hashlib.sha256('\n'.join(final_pixel_hashes).encode()).hexdigest()
    successes = [item for item in meta['items'] if item.get('success')]
    full_audit = {'output': str(output), 'source_run': str(run),
                  'source_npz_sha256': source_npz_sha, 'source_meta_sha256': source_meta_sha,
                  'video_sha256': sha256(output), 'n_source_samples': n_source,
                  'source_duration_s': float(meta['total_duration_s']),
                  'source_sample_period_s': .02,
                  'selection': 'Every 40th source sample plus final sample',
                  'n_video_frames': frames, 'fps': 50,
                  'video_frame_range_half_open': [0, frames],
                  'nominal_speed': 40, 'video_duration_s': frames / 50,
                  'first_source_sample': selected[0], 'last_source_sample': selected[-1],
                  'n_successful_cases': len(successes),
                  'minimum_video_frames_per_case': min(row['n_video_frames'] for row in coverage),
                  'maximum_video_frames_per_case': max(row['n_video_frames'] for row in coverage),
                  'case_coverage': coverage, 'frames': full_evidence,
                  'parts': parts,
                  'frame_content_proof': {'all_final_decoded_frame_md5_match_concatenated_chunk_frames': True,
                                          'all_decoded_pts_exactly_0_to_n_minus_1_at_50hz': True,
                                          'decoded_frame_hashes_sha256': decoded_content_sha,
                                          'n_compared_decoded_frames': frames},
                  'render_verification': 'Every selected sample has a matching TF stamp and at least two RViz renders in its fresh-instance chunk audit; every chunk decoded without black or occluded scene frames.'}
    (out / 'video_render_audit.json').write_text(json.dumps(full_audit, indent=2))
    print(f'ASSEMBLED {output}: {frames} frames from {len(parts)} fresh RViz instances')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
