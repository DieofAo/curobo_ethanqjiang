#!/usr/bin/env python3
"""Record three controlled full-grid variants of the approved V59 smoke."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from plan_pick_place import angle_combos, make_round_poses
from xtrainer_common import quat_wxyz_to_matrix


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out-root', type=Path, required=True)
    args = ap.parse_args()
    source, root = args.source.resolve(), args.out_root.resolve()
    cfg = json.loads(source.read_text())
    pp = cfg['pick_place']
    assert cfg['robot']['joint_limit_clip'] == .14
    assert not cfg['robot']['dual_arm_prefix']
    assert cfg['robot']['mount_transform'] == np.eye(4).tolist()
    assert pp['link0_target_transform'] == np.eye(4).tolist()
    assert pp['place']['tool_z_rotation_deg'] == 180.
    assert pp['angle_search']['order'] == 'asc'
    assert not pp['angle_search']['reuse_last_success']
    assert not pp['angle_search']['stage2']['enable']
    assert not pp['criterion']['prescreen_by_ik']
    assert pp['grasp_grid']['x_range'] == [-.62, -.20]
    assert pp['grasp_grid']['y_range'] == [-.10, .40]
    assert pp['grasp_grid']['z'] == .03
    assert pp['place']['position'] == [-.36, -.12, .10]
    baseline = copy.deepcopy(cfg)
    baseline['pick_place']['grasp_grid'].update(rows=20, cols=20)
    variants = []
    options = [('v60_00_place180_asc', 180., 'asc'),
               ('v60_01_place0_desc', 0., 'desc'),
               ('v60_02_place0_asc', 0., 'asc')]
    repo = Path(__file__).resolve().parents[2]
    inputs = [repo / cfg['robot']['urdf'],
              repo / 'src/curobo/content/configs/robot/xtrainer.yml',
              repo / 'src/curobo/content/configs/robot/spheres/xtrainer.yml',
              Path(__file__).with_name('plan_pick_place.py'),
              Path(__file__).with_name('plan_trajectory.py'),
              Path(__file__).with_name('xtrainer_common.py')]
    assert digest(inputs[0]) == cfg['overhead']['urdf_sha256']
    for name, rotation, order in options:
        candidate = copy.deepcopy(baseline)
        candidate['pick_place']['place']['tool_z_rotation_deg'] = rotation
        candidate['pick_place']['angle_search']['order'] = order
        angles = list(range(-30, 31, 2))
        if order == 'desc':
            angles.reverse()
        assert angle_combos(candidate['pick_place']['angle_search']) == [(float(a), float(a)) for a in angles]
        # Normalize the two experimental factors before comparing everything else.
        normalized = copy.deepcopy(candidate)
        normalized['pick_place']['place']['tool_z_rotation_deg'] = 180.
        normalized['pick_place']['angle_search']['order'] = 'asc'
        assert normalized == baseline
        variants.append((name, rotation, order, angles, candidate))
    unrotated = variants[2][4]['pick_place']
    rotated = variants[0][4]['pick_place']
    max_error = 0.
    for angle in range(-30, 31, 2):
        poses = [make_round_poses([-.41, .15, .03], p['place']['position'], angle, angle, p, 0)
                 for p in (unrotated, rotated)]
        assert len(poses[0]) == len(poses[1]) == 6
        for index, (a, b) in enumerate(zip(*poses)):
            assert np.array_equal(a.position, b.position)
            ra, rb = quat_wxyz_to_matrix(a.quat_wxyz), quat_wxyz_to_matrix(b.quat_wxyz)
            expected = ra if index < 3 else ra @ np.diag([-1., -1., 1.])
            error = float(np.max(np.abs(rb - expected)))
            assert error < 1e-12
            assert np.allclose(ra[:, 2], rb[:, 2], atol=1e-12, rtol=0)
            max_error = max(max_error, error)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'configs').mkdir()
    manifest = {
        'schema_version': 1, 'n_configs': 3, 'source': str(source),
        'source_sha256': digest(source), 'scope': 'User-approved three sequential 20x20 full runs',
        'input_sha256': {str(p): digest(p) for p in inputs},
        'controlled_factors': ['pick_place.place.tool_z_rotation_deg', 'pick_place.angle_search.order'],
        'other_settings_identical': True, 'original_mount': True,
        'pose_rotation_precheck': {'passed': True, 'n_angles': 31, 'max_matrix_error': max_error},
        'candidates': [],
    }
    for index, (name, rotation, order, angles, candidate) in enumerate(variants):
        path, result = root / 'configs' / (name + '.json'), root / 'runs' / name
        candidate['output'].update(dir=str(result), add_timestamp=False)
        path.write_text(json.dumps(candidate, indent=2) + '\n')
        manifest['candidates'].append({
            'index': index, 'name': name, 'config': str(path), 'config_sha256': digest(path),
            'result': str(result), 'base_xyz_m': [0., 0., 0.],
            'place_xyz_m': candidate['pick_place']['place']['position'],
            'place_tool_z_rotation_deg': rotation, 'search_order': order,
            'grasp_angle_order_deg': angles,
        })
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(root / 'manifest.json')


if __name__ == '__main__':
    main()
