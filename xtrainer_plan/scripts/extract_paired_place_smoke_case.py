#!/usr/bin/env python3
"""Extract the same central successful case from paired place-pose smoke runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from extract_pick_place_motion_cases import load_source, rank_cases, clip_payload, write_json_exclusive, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', required=True, type=Path)
    ap.add_argument('--rotated', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    sources = [load_source(p) for p in (args.baseline, args.rotated)]
    ranked = [rank_cases(meta, arrays)[0] for meta, arrays, _ in sources]
    maps = [{row['index']: row for row in rows} for rows in ranked]
    common = maps[0].keys() & maps[1].keys()
    if not common:
        raise ValueError('No common successful case to compare')
    grid = sources[0][0]['config']['pick_place']['grasp_grid']
    assert grid == sources[1][0]['config']['pick_place']['grasp_grid']
    center = np.array([np.mean(grid['x_range']), np.mean(grid['y_range'])])
    index = min(common, key=lambda i: (np.linalg.norm(np.asarray(maps[0][i]['position_raw'])[:2] - center), i))
    assert maps[0][index]['position_raw'] == maps[1][index]['position_raw']
    manifest = {'selection_rule': 'Nearest to grasp-grid center among common successful cases, then lower index',
                'index': index, 'case_number_1based': index + 1, 'center_xy': center.tolist(),
                'position_raw': maps[0][index]['position_raw'], 'clips': []}
    payloads = []
    for label, (meta, arrays, source), rows in zip(['baseline', 'rotated'], sources, maps):
        clipped, clip_meta = clip_payload(meta, arrays, rows[index], label, source)
        clip_meta['extraction']['selection_rule'] = manifest['selection_rule']
        payloads.append((label, clipped, clip_meta))
    out.mkdir(parents=True)
    for label, clipped, clip_meta in payloads:
        folder = out / label
        folder.mkdir()
        with (folder / 'trajectory.npz').open('xb') as stream:
            np.savez_compressed(stream, **clipped)
        write_json_exclusive(folder / 'trajectory_meta.json', clip_meta)
        manifest['clips'].append({'label': label, 'result': str(folder),
                                  'selection': clip_meta['extraction']['selection'],
                                  'source': clip_meta['extraction']['source'],
                                  'output_sha256': {name: sha256((folder / name).read_bytes()) for name in
                                                    ('trajectory.npz', 'trajectory_meta.json')}})
    write_json_exclusive(out / 'manifest.json', manifest)
    print(json.dumps({'index': index, 'position': manifest['position_raw'], 'out': str(out)}))


if __name__ == '__main__':
    main()
