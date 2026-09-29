#!/usr/bin/env python3
"""Extract saved min/max full-cycle joint-span cases after explicit exclusions."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from extract_pick_place_motion_cases import (
    METRIC, clip_payload, load_source, rank_cases, sha256, write_json_exclusive,
)


def extract_extremes(result, out, exclude_case_numbers=()):
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing to replace existing output: {out}")
    meta, arrays, source = load_source(result)
    ranked, _, _ = rank_cases(meta, arrays)
    excluded = set(exclude_case_numbers)
    if (len(excluded) != len(exclude_case_numbers) or
            any(not isinstance(i, int) or i < 1 or i > meta['n_items_total'] for i in excluded)):
        raise ValueError('Excluded case numbers must be unique valid 1-based integers')
    eligible = [row for row in ranked if row['original_case_number_1based'] not in excluded]
    if not eligible:
        raise ValueError('No successful cases remain after exclusions')
    selected = {'min': eligible[0], 'max': eligible[-1]}
    payloads = {label: clip_payload(meta, arrays, row, label, source)
                for label, row in selected.items()}
    for name, digest in source['sha256'].items():
        if sha256(Path(source['files'][name]).read_bytes()) != digest:
            raise ValueError('Source changed while extracting')
    for _, clip_meta in payloads.values():
        json.dumps(clip_meta, allow_nan=False)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for label, row in selected.items():
        clip = out / label
        clip.mkdir()
        data, clip_meta = payloads[label]
        with (clip / 'trajectory.npz').open('xb') as stream:
            np.savez_compressed(stream, **data)
        write_json_exclusive(clip / 'trajectory_meta.json', clip_meta)
        rows.append({'label': label, 'result': str(clip.resolve()), **row,
                     'output_sha256': {name: sha256((clip / name).read_bytes())
                                       for name in ('trajectory.npz', 'trajectory_meta.json')}})
    manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'source': source, 'metric': METRIC,
                'excluded_original_case_numbers_1based': sorted(excluded),
                'n_eligible_successful_cases': len(eligible), 'selected': rows,
                'scope': 'Exact saved successful full-cycle samples; no angle wrapping, interpolation, smoothing or replanning.',
                'extractor': {'path': str(Path(__file__).resolve()),
                              'sha256': sha256(Path(__file__).read_bytes())}}
    write_json_exclusive(out / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exclude-case-number', type=int, action='append', default=[],
                        help='Original 1-based case number to exclude from min/max selection')
    args = parser.parse_args()
    manifest = extract_extremes(args.result, args.out, args.exclude_case_number)
    for row in manifest['selected']:
        print(f"{row['label']}: case {row['original_case_number_1based']}, "
              f"{row['max_span_joint']} {row['max_joint_span_deg']:.6f} deg")
    print(Path(args.out).absolute() / 'manifest.json')


if __name__ == '__main__':
    main()
