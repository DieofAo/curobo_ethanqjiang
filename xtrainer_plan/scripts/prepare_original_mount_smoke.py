#!/usr/bin/env python3
"""Derive an isolated original-LINK0 smoke from the recorded current task.

No planning or default/model edits. The overhead provenance block is retained
only for compatibility with the existing frame/FK audit, with identity mounting.
"""
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
    ap.add_argument('--ground-source', type=Path, required=True)
    ap.add_argument('--out-root', type=Path, required=True)
    args = ap.parse_args()
    source, ground_source, root = (p.resolve() for p in
                                  (args.source, args.ground_source, args.out_root))
    current = json.loads(source.read_text())
    ground = json.loads(ground_source.read_text())['config']
    cfg = copy.deepcopy(current)
    pp = cfg['pick_place']
    assert cfg['robot']['joint_limit_clip'] == .14
    assert not cfg['robot']['dual_arm_prefix']
    assert pp['grasp_grid']['x_range'] == [-.62, -.20]
    assert pp['grasp_grid']['y_range'] == [-.10, .40]
    assert pp['grasp_grid']['z'] == .03
    assert pp['place']['position'] == [-.36, -.12, .10]
    identity = np.eye(4).tolist()
    cfg['robot'].update(mount_transform=identity, task_frame='LINK_0',
                        legacy_task_frame='original_LINK_0')
    pp['link0_target_transform'] = identity
    # Preserve physical walls, not their old coordinates in the suspended base.
    cfg['workspace'] = copy.deepcopy(current['overhead']['task_workspace'])
    pp['home']['ik_seed_joint_deg'] = copy.deepcopy(
        ground['pick_place']['home']['ik_seed_joint_deg'])
    pp['grasp_grid'].update(rows=5, cols=5)
    pp['angle_search'].update(order='asc', reuse_last_success=False)
    pp['angle_search']['grasp'].update(min_deg=-30., max_deg=30., step_deg=2.)
    pp['place']['tool_z_rotation_deg'] = 180.
    assert not pp['criterion']['prescreen_by_ik']
    assert not pp['angle_search']['stage2']['enable']
    combos = angle_combos(pp['angle_search'])
    assert combos == [(float(g), float(g)) for g in range(-30, 31, 2)]
    # Check the fixed rotation is applied after each angle-dependent pose.
    unrotated = copy.deepcopy(pp)
    unrotated['place']['tool_z_rotation_deg'] = 0.
    checks = []
    for g, p in combos:
        old = make_round_poses([-.41, .15, .03], pp['place']['position'], g, p, unrotated, 0)
        new = make_round_poses([-.41, .15, .03], pp['place']['position'], g, p, pp, 0)
        assert len(old) == len(new) == 6
        for index, (a, b) in enumerate(zip(old, new)):
            ra, rb = quat_wxyz_to_matrix(a.quat_wxyz), quat_wxyz_to_matrix(b.quat_wxyz)
            expected = ra if index < 3 else ra @ np.diag([-1., -1., 1.])
            assert np.array_equal(a.position, b.position)
            error = float(np.max(np.abs(rb - expected)))
            assert error < 1e-12
            assert np.allclose(ra[:, 2], rb[:, 2], atol=1e-12, rtol=0)
            checks.append({'angle_deg': g, 'phase': index, 'rotation_error': error})
    repo = Path(__file__).resolve().parents[2]
    urdf = repo / cfg['robot']['urdf']
    cfg['overhead'] = {
        'mode': 'original_link0_mount_not_overhead', 'mount_rpy_deg': [0., 0., 0.],
        'task_workspace': copy.deepcopy(cfg['workspace']),
        'original_target_transform': identity, 'tcp_offset_m': .19,
        'urdf_sha256': digest(urdf), 'derived_from': str(source),
        'note': 'Identity mount for reusable coordinate audit only; display with run_rviz.sh, not overhead scene builder.',
    }
    name = 'v59_00_original_base_place_tcp180'
    result = root / 'runs' / name
    cfg['output'].update(dir=str(result), add_timestamp=False)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'configs').mkdir()
    config_path = root / 'configs' / (name + '.json')
    config_path.write_text(json.dumps(cfg, indent=2) + '\n')
    inputs = [urdf, repo / 'src/curobo/content/configs/robot/xtrainer.yml',
              repo / 'src/curobo/content/configs/robot/spheres/xtrainer.yml',
              Path(__file__).with_name('plan_pick_place.py')]
    manifest = {
        'scope': '5x5 full-cycle smoke only; full 20x20 awaits user RViz confirmation',
        'source': str(source), 'source_sha256': digest(source),
        'ground_seed_source': str(ground_source), 'ground_seed_source_sha256': digest(ground_source),
        'config': str(config_path), 'config_sha256': digest(config_path), 'result': str(result),
        'input_sha256': {str(p): digest(p) for p in inputs},
        'home_position_unchanged': pp['home']['position'] == current['pick_place']['home']['position'],
        'home_seed_only_replaced_with_original_mount_seed': True,
        'workspace_in_original_frame': True,
        'grasp_angles_in_order_deg': [g for g, p in combos],
        'place_rotation_formula': 'R_new_place(g) = R_old_place(g) @ Rz_local(180deg)',
        'pose_precheck_passed': True, 'pose_precheck': checks,
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(config_path)
    print(result)


if __name__ == '__main__':
    main()
