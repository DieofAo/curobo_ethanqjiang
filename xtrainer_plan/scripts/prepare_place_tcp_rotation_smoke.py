#!/usr/bin/env python3
"""Record a paired original-place/local-TCP-Z-180 smoke test, without planning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from plan_pick_place import angle_combos, make_round_poses, transform_pose
from xtrainer_common import quat_wxyz_to_matrix


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out-root', required=True, type=Path)
    parser.add_argument('--size', type=int, default=5)
    args = parser.parse_args()
    if args.size < 2:
        raise ValueError('Smoke grid must cover both edges')
    source, root = args.source.resolve(), args.out_root.resolve()
    cfg = json.loads(source.read_text())
    assert cfg['robot']['joint_limit_clip'] == .14
    assert cfg['pick_place']['angle_search']['stage2']['enable'] is False
    assert cfg['pick_place']['place'].get('tool_z_rotation_deg', 0) == 0
    assert cfg['pick_place']['angle_search']['order'] == 'asc'
    assert cfg['pick_place']['angle_search']['reuse_last_success'] is False
    baseline = copy.deepcopy(cfg)
    baseline['pick_place']['grasp_grid'].update(rows=args.size, cols=args.size)
    rotated = copy.deepcopy(baseline)
    rotated['pick_place']['place']['tool_z_rotation_deg'] = 180.0
    oldpp, newpp = baseline['pick_place'], rotated['pick_place']
    combos = angle_combos(oldpp['angle_search'])
    assert combos == [(float(g), float(g)) for g in range(-30, 31, 2)]
    assert combos == angle_combos(newpp['angle_search'])
    target_c = np.asarray(oldpp['link0_target_transform'])
    expected = np.diag([-1., -1., 1.])
    checks = []
    for grasp, place in combos:
        seq_old = make_round_poses([-.41, .15, .03], oldpp['place']['position'],
                                   grasp, place, oldpp, 0)
        seq_new = make_round_poses([-.41, .15, .03], newpp['place']['position'],
                                   grasp, place, newpp, 0)
        assert len(seq_old) == len(seq_new) == 6
        errors = []
        for j, (old, new) in enumerate(zip(seq_old, seq_new)):
            assert np.array_equal(old.position, new.position)
            old, new = transform_pose(old, target_c), transform_pose(new, target_c)
            r_old, r_new = quat_wxyz_to_matrix(old.quat_wxyz), quat_wxyz_to_matrix(new.quat_wxyz)
            wanted = r_old if j < 3 else r_old @ expected
            error = float(np.max(np.abs(r_new - wanted)))
            assert error < 1e-12, (grasp, j, error)
            assert np.allclose(r_new[:, 2], r_old[:, 2], atol=1e-12, rtol=0)
            errors.append(error)
        checks.append({'grasp_angle_deg': grasp, 'place_angle_deg': place,
                       'max_matrix_error': max(errors), 'all_positions_unchanged': True,
                       'all_tcp_z_axes_unchanged': True, 'grasp_orientations_unchanged': True})
    root.mkdir(parents=True, exist_ok=False)
    (root / 'configs').mkdir()
    repo = Path(__file__).resolve().parents[2]
    inputs = [repo / 'src/curobo/content/assets/robot/ur_description/xtrainer.urdf',
              repo / 'src/curobo/content/configs/robot/xtrainer.yml',
              repo / 'src/curobo/content/configs/robot/spheres/xtrainer.yml',
              Path(__file__).with_name('plan_pick_place.py')]
    manifest = {'schema_version': 1, 'source': str(source), 'source_sha256': digest(source),
                'scope': 'Paired 5x5 full-cycle smoke; only place final local TCP Z +180 deg differs between candidates',
                'grid_size': args.size, 'n_configs': 2, 'candidates': [],
                'input_sha256': {str(p): digest(p) for p in inputs}}
    names = ['v58_00_baseline', 'v58_01_place_tcp180']
    for index, (experiment, name) in enumerate(zip((baseline, rotated), names)):
        result = root / 'runs' / name
        experiment['output'].update(dir=str(result), add_timestamp=False)
        experiment['overhead'].update(derived_from=str(source), variant_options={
            'smoke_grid': args.size, 'place_tool_z_rotation_deg': index * 180,
            'note': 'Postmultiply each final place orientation by local Rz, before task-to-mounted-base transform'})
        config_path = root / 'configs' / (name + '.json')
        config_path.write_text(json.dumps(experiment, indent=2) + '\n')
        manifest['candidates'].append({'index': index, 'name': name, 'config': str(config_path),
                                      'config_sha256': digest(config_path), 'result': str(result),
                                      'base_xyz_m': np.asarray(experiment['robot']['mount_transform'])[:3, 3].tolist(),
                                      'place_xyz_m': experiment['pick_place']['place']['position'],
                                      'place_tool_z_rotation_deg': index * 180})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (root / 'target_rotation_precheck.json').write_text(json.dumps({
        'passed': True, 'formula': 'R_place_new(g) = R_place_old(g) @ Rz_local(180deg)',
        'n_candidates': len(combos), 'checks': checks,
        'candidate_pairs_unchanged': True, 'home_unchanged': oldpp['home'] == newpp['home'],
        'robot_and_collision_constraints_unchanged': baseline['robot'] == rotated['robot'] and
        baseline['workspace'] == rotated['workspace'] and baseline['planner'] == rotated['planner'],
    }, indent=2) + '\n')
    print(root / 'manifest.json')


if __name__ == '__main__':
    main()
