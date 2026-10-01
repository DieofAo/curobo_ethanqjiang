#!/usr/bin/env python3
"""Replay the saved V78 bundle in its own ROS1 master; no planning imports.

--validate-only checks exported hashes, events, and exact saved samples. It
does not repeat the original collision audit. EE first-piece exclusions affect
only the static path display; robot playback always uses the full 400 picks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DT = .02
NAMES = [f'{side}_J{i}' for side in ('near', 'far') for i in range(1, 7)]
ROS_NAMES = [f'{prefix}J_{i}' for prefix in ('', 'far_') for i in range(1, 7)]
COLORS = {'near': (.12, .42, 1.), 'far': (1., .90, .02)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'manifest path must be relative: ' + str(relative))
    return root / path


def relocated_urdf(path, repo):
    root = ET.parse(path).getroot()
    active = [j.attrib['name'] for j in root.findall('joint') if j.attrib['type'] != 'fixed']
    require(len(active) == 12 and set(active) == set(ROS_NAMES), 'URDF active joints differ from saved joint order')
    for mesh in root.iter('mesh'):
        old = mesh.attrib['filename']
        require('/src/curobo/' in old, 'unsupported mesh reference: ' + old)
        resource = repo / 'src/curobo' / old.split('/src/curobo/', 1)[1]
        require(resource.is_file(), 'missing repository mesh: ' + str(resource))
        mesh.set('filename', resource.resolve().as_uri())
    return ET.tostring(root, encoding='unicode')


def validate_bundle(bundle=HERE, repo=REPO):
    manifest = read_json(bundle / 'manifest.json')
    files = manifest['files']
    required = ('data/full_trajectory.npz', 'data/summary.json', 'scene/dual_arm.urdf',
                'scene/scene.json', 'scene/ee_paths.rviz', 'ee_paths/ee_paths_excluding_first_pieces.npz')
    require(all(name in files for name in required), 'manifest lacks required replay files')
    require('play_in_rviz.py' in files, 'manifest does not bind the replay script')
    checked = [(local_path(bundle, name), sha) for name, sha in files.items()]
    checked += [(local_path(repo, name), sha) for name, sha in manifest.get('assets', {}).items()]
    for path, sha in checked:
        require(digest(path) == sha, 'SHA-256 mismatch: ' + str(path))
    with np.load(bundle / required[0], allow_pickle=False) as archive:
        q, times = archive['positions'].copy(), archive['times'].copy()
        names = archive['joint_names'].tolist()
    require(q.shape == (145141, 12) and np.isfinite(q).all(), 'invalid full joint trajectory')
    require(names == NAMES, 'full trajectory joint names/order differ')
    require(times.shape == (len(q),) and np.array_equal(times, np.arange(len(q)) * DT), 'full trajectory time samples differ')
    summary, scene = read_json(bundle / required[1]), read_json(bundle / required[3])
    require(summary['total_picks'] == 400 and summary['all_window_audits_pass']
            and summary['all_boundaries_exactly_shared'], 'summary is not the audited full 400-case result')
    require(summary['samples'] == len(q) and summary['duration_s'] == float(times[-1]), 'summary sample count/time differs')
    windows = {row['id']: row for row in summary['windows']}
    require(len(windows) == len(summary['windows']), 'duplicate window ids')
    end = 0
    for ordinal, row in enumerate(summary['windows']):
        require(row['start_global_sample'] == end and row['end_global_sample'] > end, 'window boundary/gap differs')
        end = row['end_global_sample']
    require(end == len(q) - 1, 'windows do not cover the full trajectory')
    events = summary['events']
    require(len(events) == 800, 'expected exactly 800 grasp/place events')
    bycase = {}
    for event in events:
        side, kind, case, sample = [event[k] for k in ('side', 'kind', 'grasp_case_1based', 'global_sample')]
        require(side in COLORS and kind in ('grasp', 'place') and isinstance(sample, int)
                and not isinstance(sample, bool) and 0 <= sample < len(q), 'invalid event')
        require(np.isclose(event['global_time_s'], times[sample], atol=1e-9, rtol=0), 'event time differs from saved sample')
        window = windows[event['window']]
        require(window['start_global_sample'] <= sample <= window['end_global_sample'], 'event outside its window')
        target = np.asarray(event['target_world_m'], dtype=float)
        require(target.shape == (3,) and np.isfinite(target).all(), 'invalid event target')
        if kind == 'place':
            require(np.array_equal(target, scene[side + '_place_task_world']), 'place differs from fixed scene target')
        bycase.setdefault(case, []).append(event)
    require(set(bycase) == set(range(1, 401)), 'grasp/place coverage differs from cases 1..400')
    require(all(len(rows) == 2 and {r['kind'] for r in rows} == {'grasp', 'place'}
                and len({r['side'] for r in rows}) == 1
                and next(r['global_sample'] for r in rows if r['kind'] == 'grasp')
                < next(r['global_sample'] for r in rows if r['kind'] == 'place')
                for rows in bycase.values()), 'duplicate/mismatched grasp-place events')
    grasps = [e for e in events if e['kind'] == 'grasp']
    counts = dict(Counter(e['side'] for e in grasps))
    stats = manifest['stats']
    require(stats['samples'] == len(q) and stats['duration_s'] == float(times[-1])
            and stats['unique_cases'] == 400 and stats['events'] == 800
            and stats['counts_by_executing_arm'] == counts, 'manifest statistics differ')
    coordinates = manifest['scene']
    require(coordinates['coordinate_frame'] == scene['task_frame'] == 'task_world', 'unexpected coordinate frame')
    for side in COLORS:
        mount = np.asarray(scene[side + '_mount_task_world'], dtype=float)
        require(mount.shape == (4, 4) and np.isfinite(mount).all(), 'invalid base transform')
        require(np.allclose(mount[:3, 3], coordinates[side + '_base_task_world'], atol=1e-9, rtol=0)
                and np.allclose(scene[side + '_place_task_world'], coordinates[side + '_place_task_world'], atol=1e-9, rtol=0), 'scene coordinates differ')
    xyz = np.asarray([e['target_world_m'] for e in grasps])
    require(len(np.unique(xyz, axis=0)) == 400, 'duplicate grasp targets')
    bounds = coordinates['grasp_bounds_task_world']
    require(np.allclose(xyz.min(axis=0), bounds['minimum'], atol=1e-9, rtol=0)
            and np.allclose(xyz.max(axis=0), bounds['maximum'], atol=1e-9, rtol=0), 'grasp region differs')
    paths = {}
    with np.load(bundle / required[5], allow_pickle=False) as archive:
        for side, columns, cut in [('near', slice(0, 6), 716), ('far', slice(6, 12), 488)]:
            exclusion = manifest['ee_path_exclusion'][side]
            require(exclusion['retained_start_global_sample'] == cut
                    and exclusion['excluded_global_samples'] == [0, cut - 1]
                    and exclusion['shared_end_boundary_retained'] is True, 'EE first-piece cutoff differs')
            source = archive[side + '_source_global_sample']
            tcp = archive[side + '_tcp_task_world']
            moving = archive[side + '_moving_source_global_sample']
            require(np.array_equal(source, np.arange(cut, len(q))), 'EE retained source indices differ')
            require(np.array_equal(archive[side + '_q_rad'], q[cut:, columns])
                    and np.array_equal(archive[side + '_times_s'], times[cut:]), 'EE q/times differ from exact full source slice')
            require(tcp.shape == (len(source), 3) and np.isfinite(tcp).all(), 'invalid saved EE positions')
            keep = np.r_[True, np.linalg.norm(np.diff(tcp, axis=0), axis=1) > 1e-9]
            require(np.array_equal(moving, source[keep]), 'EE display deletes more than stationary consecutive duplicates')
            paths[side] = tcp[keep].copy()
    video = bundle / 'data/video_timeline.npz'
    if video.exists():
        require('data/video_timeline.npz' in files, 'video timeline is not hash bound')
        with np.load(video, allow_pickle=False) as archive:
            require(np.array_equal(archive['positions'], q) and np.array_equal(archive['times'], times), 'video timeline differs from full source')
    display = 'scene/video_visual.urdf'
    if (bundle / display).exists():
        require(display in files, 'visual URDF is not hash bound')
        original = ET.parse(bundle / required[2]).getroot()
        visual = ET.parse(bundle / display).getroot()
        require([ET.tostring(j) for j in original.findall('joint')] ==
                [ET.tostring(j) for j in visual.findall('joint')], 'visual URDF changes robot kinematics')
    else:
        display = required[2]
    urdf = relocated_urdf(bundle / display, repo)
    for path, sha in checked:
        require(digest(path) == sha, 'file changed during validation: ' + str(path))
    return q, times, summary, scene, paths, urdf


def choose_port(requested):
    if requested is not None:
        require(1 <= requested <= 65535, '--master-port must be 1..65535')
        try:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', requested))
            return requested
        except OSError:
            pass
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def stop_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


def replay(args, q, times, summary, scene, paths, urdf):
    port = choose_port(args.master_port)
    env = os.environ.copy()
    env.update(ROS_MASTER_URI=f'http://127.0.0.1:{port}', ROS_HOSTNAME='127.0.0.1',
               ROS_IP='127.0.0.1', DISPLAY=args.display)
    os.environ.update({k: env[k] for k in ('ROS_MASTER_URI', 'ROS_HOSTNAME', 'ROS_IP', 'DISPLAY')})
    processes = []
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(*_):
        raise KeyboardInterrupt
    for sig in previous:
        signal.signal(sig, interrupted)
    log = Path(f'/tmp/v78_portable_rviz_{port}.log')
    try:
        with log.open('ab') as stream:
            def start(command):
                proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(proc)
                return proc
            master = start(['roscore', '-p', str(port)])
            deadline = time.monotonic() + 15
            ready = False
            while time.monotonic() < deadline:
                require(master.poll() is None, 'own roscore exited; see ' + str(log))
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.2):
                        ready = True
                    if ready:
                        break
                except OSError:
                    time.sleep(.1)
            require(ready, 'own ROS master did not start; see ' + str(log))
            import rospy
            from sensor_msgs.msg import JointState
            from geometry_msgs.msg import Point
            from visualization_msgs.msg import Marker, MarkerArray
            rospy.init_node('v78_portable_replay', disable_signals=True)
            rospy.set_param('/robot_description', urdf)
            state = start(['rosrun', 'robot_state_publisher', 'robot_state_publisher', '__name:=v78_portable_state'])
            joints = rospy.Publisher('/joint_states', JointState, queue_size=1)
            markers = rospy.Publisher('/xtrainer_plan/markers', MarkerArray, queue_size=1, latch=True)
            ee = rospy.Publisher('/xtrainer_plan/ee_paths', MarkerArray, queue_size=1, latch=True)
            def marker(namespace, kind, points, color, size):
                m = Marker(); m.header.frame_id = 'task_world'; m.ns = namespace; m.id = 0
                m.type = kind; m.action = Marker.ADD; m.pose.orientation.w = 1.
                m.scale.x = size; m.scale.y = size; m.scale.z = size
                m.color.r, m.color.g, m.color.b, m.color.a = (*color, 1.)
                m.points = [Point(float(x), float(y), float(z)) for x, y, z in points]
                return m
            static = MarkerArray(); path_markers = MarkerArray()
            for side, color in COLORS.items():
                grasp = [e['target_world_m'] for e in summary['events'] if e['side'] == side and e['kind'] == 'grasp']
                static.markers.append(marker(side + '_grasp', Marker.SPHERE_LIST, grasp, color, .006))
                for label, xyz in [('base', np.asarray(scene[side + '_mount_task_world'])[:3, 3]),
                                   ('place', scene[side + '_place_task_world'])]:
                    static.markers.append(marker(side + '_' + label, Marker.SPHERE_LIST, [xyz], color, .025))
                    text = marker(side + '_' + label + '_label', Marker.TEXT_VIEW_FACING, [], color, .025)
                    text.text = side.upper() + ' ' + label
                    text.pose.position = Point(float(xyz[0]), float(xyz[1]), float(xyz[2]) + .04)
                    static.markers.append(text)
                if args.ee_paths == 'full':
                    path_markers.markers.append(marker(side + '_ee_path', Marker.LINE_STRIP, paths[side], color, .002))
            markers.publish(static); ee.publish(path_markers)
            viewer = start(['rviz', '-d', str(HERE / 'scene/ee_paths.rviz')])
            def healthy():
                if viewer.poll() is not None:
                    return False
                require(master.poll() is None and state.poll() is None, 'an owned ROS process exited; see ' + str(log))
                return True
            def publish(index):
                msg = JointState(); msg.header.stamp = rospy.Time.now(); msg.name = ROS_NAMES
                msg.position = q[index].tolist(); joints.publish(msg)
            for _ in range(30):
                publish(0); time.sleep(.05)
            print(json.dumps({'ROS_MASTER_URI': env['ROS_MASTER_URI'], 'log': str(log), 'playback_cases': 400,
                              'ee_path_display': args.ee_paths, 'speed': args.speed}), flush=True)
            stride = max(1, int(round(args.speed)))
            selected = sorted(set(range(0, len(q), stride)) | {e['global_sample'] for e in summary['events']} | {len(q) - 1})
            while not rospy.is_shutdown():
                began = time.monotonic()
                for index in selected:
                    if rospy.is_shutdown() or not healthy():
                        return
                    delay = began + float(times[index]) / args.speed - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    publish(index)
                for _ in range(20):
                    publish(len(q) - 1); time.sleep(.05)
                if args.once:
                    while not rospy.is_shutdown() and healthy():
                        publish(len(q) - 1); time.sleep(.05)
                    return
    finally:
        for process in reversed(processes):
            stop_group(process)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--speed', type=float, default=1.)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--master-port', type=int)
    ap.add_argument('--display', default=os.environ.get('DISPLAY', ':0'))
    ap.add_argument('--ee-paths', choices=('off', 'full'), default='full')
    args = ap.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        ap.error('--speed must be finite and positive')
    q, times, summary, scene, paths, urdf = validate_bundle()
    if args.validate_only:
        print(json.dumps({'passed': True, 'validation': 'exported_hashes_and_saved_samples_not_new_collision_audit',
                          'samples': len(q), 'events': len(summary['events']), 'cases': 400,
                          'duration_s': float(times[-1]), 'ee_retained_start': {'near': 716, 'far': 488},
                          'ee_moving_points': {side: len(points) for side, points in paths.items()}}, ensure_ascii=False))
        return 0
    try:
        replay(args, q, times, summary, scene, paths, urdf)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
