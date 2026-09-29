#!/usr/bin/env python3
"""Export and verify first/last and adjacent chunk-boundary RViz frames."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--render-audit', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    video = args.video.resolve()
    audit = json.loads(args.render_audit.read_text())
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    if digest != audit['video_sha256']:
        raise ValueError('Video SHA256 differs from assembled render audit')
    n_source = int(audit['n_source_samples'])
    selected = list(range(0, n_source, 40))
    if selected[-1] != n_source - 1:
        selected.append(n_source - 1)
    if len(selected) != audit['n_video_frames'] or [row['source_sample_index'] for row in audit['frames']] != selected:
        raise ValueError('Audit source sample sequence differs from deterministic 40-sample selection')
    parts = audit['parts']
    if len(parts) < 2:
        raise ValueError('Boundary review requires at least two fresh-RViz chunks')
    last = audit['n_video_frames'] - 1
    indices = [0]
    boundaries = []
    for previous, following in zip(parts, parts[1:]):
        before = previous['frame_range_half_open'][1] - 1
        after = following['frame_range_half_open'][0]
        if after != before + 1:
            raise ValueError('Missing video frame across chunks')
        sample_before = audit['frames'][before]['source_sample_index']
        sample_after = audit['frames'][after]['source_sample_index']
        expected_step = selected[after] - selected[before]
        if sample_after - sample_before != expected_step or not 0 < expected_step <= 40:
            raise ValueError('Source sample discontinuity across chunks')
        boundaries.append({'video_frames': [before, after],
                           'source_samples': [sample_before, sample_after],
                           'source_sample_step': expected_step})
        indices.extend([before, after])
    indices.append(last)
    indices = list(dict.fromkeys(indices))  # final chunk may contain one frame

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='xtrainer_video_boundary_') as temp:
        pattern = Path(temp) / 'frame_%02d.png'
        expression = '+'.join(f'eq(n\\,{number})' for number in indices)
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),
                        '-vf',f'select={expression},scale=640:458',
                        '-vsync','0',str(pattern)], check=True)
        images = sorted(Path(temp).glob('frame_*.png'))
        if len(images) != len(indices):
            raise ValueError(f'Could not decode all review frames: {len(images)} vs {len(indices)}')
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
        cols = 2
        rows = (len(indices) + cols - 1) // cols
        montage = Image.new('RGB',(cols*640,rows*492),(18,22,29))
        draw = ImageDraw.Draw(montage)
        for order,(index,path) in enumerate(zip(indices,images)):
            x=(order%cols)*640;y=(order//cols)*492
            frame=Image.open(path).convert('RGB')
            montage.paste(frame,(x,y+28))
            sample=audit['frames'][index]['source_sample_index']
            draw.text((x+10,y+5),f'video frame {index}  |  source sample {sample}',font=font,fill=(245,245,245))
        montage.save(args.out_dir/'boundary_review.png')
    (args.out_dir/'boundary_review.json').write_text(json.dumps({
        'video':str(video),'selected_video_frames':indices,'boundaries':boundaries,
        'first_source_sample':audit['frames'][0]['source_sample_index'],
        'last_source_sample':audit['frames'][-1]['source_sample_index'],
        'boundary_source_indices_contiguous':True,
        'all_selected_review_frames_decoded':True,
        'video_sha256_match':True,
        'framemd5_content_equality_verified_by_assembly':audit['frame_content_proof']['all_final_decoded_frame_md5_match_concatenated_chunk_frames']},indent=2))
    print(args.out_dir/'boundary_review.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
