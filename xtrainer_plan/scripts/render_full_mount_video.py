#!/usr/bin/env python3
"""Render the complete saved full mount motion in isolated RViz at 40x video speed.

Source samples are spaced 20 ms apart. We render indices 0, 40, 80, ... and
the last index; 50 video frames/s therefore advances source time by 40x.
Each selected joint state waits for matching robot TF and two RViz renders.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np


TASK_ROOT = Path(__file__).resolve().parents[1]
FPS = 50
SAMPLE_STEP = 40


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--experiment-label', required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--first-video-frame', type=int, default=0)
    parser.add_argument('--last-video-frame', type=int)
    parser.add_argument('--output-name')
    args = parser.parse_args()
    run = args.run.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_master = f'http://127.0.0.1:{args.master_port}'
    if args.master_port < 11350 or os.environ.get('ROS_MASTER_URI') != expected_master:
        raise RuntimeError(f'Recording requires dedicated ROS master {expected_master} on port >=11350')
    sys.path.insert(0, str(TASK_ROOT / 'scripts'))
    from PyQt5 import QtCore, QtGui, QtWidgets
    from rviz import bindings as rviz
    import rospy
    from sensor_msgs.msg import JointState
    from tf2_msgs.msg import TFMessage
    from visualization_msgs.msg import Marker, MarkerArray
    from xtrainer_common import load_trajectory
    from play_trajectory_ros import build_static_markers
    from build_overhead_scene_urdf import load_resolved_config, scene_settings, build_scene, _serialize

    data, meta = load_trajectory(str(run))
    q = np.asarray(data['positions'])
    times = np.asarray(data['times'])
    if len(q) < 2 or q.shape[1] != 6 or not np.isfinite(q).all():
        raise RuntimeError('Invalid six-joint source trajectory')
    if not np.allclose(np.diff(times), 0.02, rtol=0, atol=1e-7):
        raise RuntimeError('Source sample period is not uniform 20 ms')
    selected = list(range(0, len(q), SAMPLE_STEP))
    if selected[-1] != len(q) - 1:
        selected.append(len(q) - 1)
    final_frame = len(selected) - 1 if args.last_video_frame is None else args.last_video_frame
    if not 0 <= args.first_video_frame <= final_frame < len(selected):
        raise ValueError('Invalid video frame slice')
    frame_ids = range(args.first_video_frame, final_frame + 1)

    successes = [item for item in meta['items'] if item.get('success')]
    if not successes or sum(int(item['n_points']) for item in successes) != len(q):
        raise RuntimeError('Successful item lengths differ from source trajectory')
    starts = [0]
    for item in successes:
        starts.append(starts[-1] + int(item['n_points']))
    coverage = []
    for ordinal, item in enumerate(successes):
        lo, hi = starts[ordinal:ordinal + 2]
        from_frame = bisect.bisect_left(selected, lo)
        to_frame = bisect.bisect_left(selected, hi)
        count = to_frame - from_frame
        if count == 0:
            raise RuntimeError(f'No video frame in saved item {item["index"] + 1}')
        coverage.append({'case_number': int(item['index']) + 1,
                         'case_index_zero_based': int(item['index']),
                         'source_range_half_open': [lo, hi],
                         'video_frame_range_half_open': [from_frame, to_frame],
                         'n_video_frames': count})

    cfg, source = load_resolved_config(None, str(run))
    scene, summary = build_scene(scene_settings(cfg, source))
    urdf = _serialize(scene)
    (out_dir / 'recording_scene.urdf').write_text(urdf)
    (out_dir / 'recording_scene.json').write_text(json.dumps(summary, indent=2))
    children = [joint.find('child').get('link') for joint in ET.fromstring(urdf).findall('joint')
                if joint.get('type') in ('revolute', 'continuous')]
    if len(children) != 6:
        raise RuntimeError('Expected exactly six active arm joints')

    rospy.init_node('full_mount_video_renderer', anonymous=False, disable_signals=True)
    rospy.set_param('/robot_description', urdf)
    rsp = subprocess.Popen(['rosrun', 'robot_state_publisher', 'robot_state_publisher',
                            '__name:=full_video_robot_state_publisher', '_publish_frequency:=1000'])
    joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=1)
    marker_pub = rospy.Publisher('/xtrainer_plan/markers', MarkerArray, queue_size=1, latch=True)
    latest_tf = {}
    lock = threading.Lock()

    def receive_tf(msg):
        with lock:
            for transform in msg.transforms:
                latest_tf[transform.child_frame_id] = transform.header.stamp.to_nsec()

    tf_sub = rospy.Subscriber('/tf', TFMessage, receive_tf, queue_size=100)
    app = QtWidgets.QApplication(sys.argv[:1])
    window = rviz.VisualizationFrame()
    window.setSplashPath('')
    window.initialize()
    config = rviz.Config()
    rviz.YamlConfigReader().readFile(config, str(TASK_ROOT / 'config/xtrainer_overhead.rviz'))
    window.load(config)
    window.setHideButtonVisibility(False)
    for widget in window.findChildren(QtWidgets.QDockWidget):
        widget.hide()
    for widget in window.findChildren(QtWidgets.QToolBar):
        widget.hide()
    window.menuBar().hide()
    window.statusBar().hide()
    window.setWindowTitle(f'{args.experiment_label} complete saved trajectory - isolated RViz recording')
    window.resize(1280, 800)
    window.move(100, 100)
    window.show()
    manager = window.getManager()
    manager.setFixedFrame('task_world')
    view = manager.getViewManager().getCurrent()
    view.subProp('Distance').setValue(1.65)
    view.subProp('Pitch').setValue(0.42)
    view.subProp('Yaw').setValue(2.35)
    manager.startUpdate()

    def pump(seconds=0.0):
        deadline = time.monotonic() + seconds
        while True:
            app.processEvents(QtCore.QEventLoop.AllEvents, 10)
            if time.monotonic() >= deadline:
                return
            time.sleep(0.002)

    def pose(source_index):
        stamp = rospy.Time.now()
        message = JointState()
        message.header.stamp = stamp
        message.name = list(data['joint_names'])
        message.position = [float(value) for value in q[source_index]]
        joint_pub.publish(message)
        deadline = time.monotonic() + 6
        while True:
            with lock:
                ready = all(latest_tf.get(child, -1) == stamp.to_nsec() for child in children)
            if ready:
                break
            pump(.003)
            if time.monotonic() > deadline:
                raise RuntimeError(f'Timed out waiting for source sample {source_index} TF')
        before = manager.getFrameCount()
        while manager.getFrameCount() < before + 2:
            manager.queueRender()
            pump(.005)
            if time.monotonic() > deadline:
                raise RuntimeError(f'Timed out waiting for source sample {source_index} RViz render')
        return {'source_sample_index': source_index,
                'source_time_s': float(times[source_index]),
                'tf_stamp_ns': stamp.to_nsec(),
                'rviz_render_count': manager.getFrameCount()}

    def capture():
        pixmap = app.primaryScreen().grabWindow(int(window.winId()))
        if pixmap.isNull() or pixmap.width() < 1000 or pixmap.height() < 600:
            raise RuntimeError('Invalid isolated RViz capture')
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
        raw = image.bits()
        raw.setsize(image.byteCount())
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
        pixels = pixels[:, :image.width() * 3].reshape(image.height(), image.width(), 3)
        visible = int(np.count_nonzero(np.max(pixels[::10, ::10], axis=2) > 70))
        if visible < 100:
            raise RuntimeError(f'RViz scene appears black before encoding: {visible} bright sample pixels')
        return image

    def decorate(source, frame_index, source_index):
        width, height = source.width(), source.height()
        canvas = QtGui.QImage(width, height + 116, QtGui.QImage.Format_RGB888)
        canvas.fill(QtGui.QColor(22, 25, 31))
        painter = QtGui.QPainter(canvas)
        painter.drawImage(0, 81, source)
        ordinal = bisect.bisect_right(starts, source_index) - 1
        ordinal = min(ordinal, len(successes) - 1)
        item = successes[ordinal]
        grasp = item.get('position_raw', item['position'])
        painter.setPen(QtGui.QColor(241, 245, 249))
        painter.setFont(QtGui.QFont('DejaVu Sans', 18, QtGui.QFont.Bold))
        painter.drawText(24, 30, f'{args.experiment_label} COMPLETE TRAJECTORY  |  40x  |  Case {item["index"] + 1}/{meta["n_items_total"]}')
        painter.setFont(QtGui.QFont('DejaVu Sans', 12))
        painter.drawText(24, 59,
                         f'Successful case {ordinal + 1}/{len(successes)}    Grasp (x,y)=({grasp[0]:+.3f},{grasp[1]:+.3f}) m'
                         f'    Source t={times[source_index]:.2f}/{times[-1]:.2f} s')
        painter.setPen(QtGui.QColor(174, 182, 195))
        painter.drawText(24, height + 104,
                         f'Saved plan, sampled every 40 x 20 ms | source sample {source_index}/{len(q) - 1}'
                         f' | video frame {frame_index + 1}/{len(selected)}')
        painter.end()
        return canvas

    encoder = None
    output = out_dir / (args.output_name or f'{args.experiment_label}_complete_40x.mp4')
    try:
        pump(3)
        clear = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        clear.markers = [marker]
        marker_pub.publish(clear)
        pump(.1)
        markers = build_static_markers(meta, 'LINK_0', data['ee_positions'][::100],
                                       argparse.Namespace(show_walls=False, wall_alpha=.12))
        markers.markers = [marker for marker in markers.markers if marker.ns == 'ee_path']
        marker_pub.publish(markers)
        window.raise_()
        first = pose(selected[args.first_video_frame])
        pump(.25)
        image = decorate(capture(), args.first_video_frame, selected[args.first_video_frame])
        preview_name = 'preview.png' if args.first_video_frame == 0 else f'preview_{args.first_video_frame}.png'
        image.save(str(out_dir / preview_name))
        print(f'PREVIEW {out_dir / preview_name}', flush=True)
        if args.preview_only:
            return 0
        if output.exists():
            raise FileExistsError(output)
        width, height = image.width(), image.height()
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n', '-f', 'rawvideo',
                   '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                   '-framerate', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
                   '-preset', 'fast', '-crf', '21', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(output)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)

        def write(frame):
            if frame.bytesPerLine() != width * 3:
                raise RuntimeError('Unexpected RGB line padding')
            bits = frame.bits()
            bits.setsize(frame.byteCount())
            encoder.stdin.write(bytes(bits))

        audit_frames = []
        for local_frame, frame_index in enumerate(frame_ids):
            source_index = selected[frame_index]
            evidence = first if local_frame == 0 else pose(source_index)
            evidence['video_frame_index'] = frame_index
            audit_frames.append(evidence)
            try:
                image = decorate(capture(), frame_index, source_index)
            except Exception as error:
                raise RuntimeError(f'Capture failed at video frame {frame_index}, source sample {source_index}: {error}') from error
            write(image)
            if frame_index in {len(selected) // 4, len(selected) // 2,
                               3 * len(selected) // 4, len(selected) - 1}:
                image.save(str(out_dir / f'frame_{frame_index:04d}.png'))
            if frame_index % 100 == 0 or frame_index == final_frame:
                print(f'RENDER {frame_index + 1}/{len(selected)} source={source_index}', flush=True)
        encoder.stdin.close()
        if encoder.wait(timeout=120):
            raise RuntimeError('Video encoder failed')
        encoder = None
        audit = {'output': str(output), 'source_run': str(run),
                 'source_npz_sha256': sha256(run / 'trajectory.npz'),
                 'source_meta_sha256': sha256(run / 'trajectory_meta.json'),
                 'video_sha256': sha256(output), 'n_source_samples': len(q),
                 'source_duration_s': float(times[-1]), 'source_sample_period_s': .02,
                 'selection': 'Every 40th source sample plus final sample',
                 'n_video_frames': len(frame_ids), 'fps': FPS,
                 'video_frame_range_half_open': [args.first_video_frame, final_frame + 1],
                 'nominal_speed': SAMPLE_STEP, 'video_duration_s': len(frame_ids) / FPS,
                 'first_source_sample': selected[args.first_video_frame], 'last_source_sample': selected[final_frame],
                 'n_successful_cases': len(successes),
                 'minimum_video_frames_per_case': min(row['n_video_frames'] for row in coverage),
                 'maximum_video_frames_per_case': max(row['n_video_frames'] for row in coverage),
                 'case_coverage': coverage, 'frames': audit_frames,
                 'render_verification': 'Each selected joint state had matching TF timestamp and at least two RViz renders before its window capture'}
        audit_name = 'video_render_audit.json' if args.first_video_frame == 0 else f'video_render_audit_{args.first_video_frame}.json'
        (out_dir / audit_name).write_text(json.dumps(audit, indent=2))
        print(f'COMPLETE {output}', flush=True)
        return 0
    finally:
        if encoder is not None:
            # Preserve a decodable prefix if a later RViz frame fails, so only
            # the missing suffix needs a fresh isolated RViz rendering pass.
            encoder.stdin.close()
            try:
                encoder.wait(timeout=120)
            except subprocess.TimeoutExpired:
                encoder.terminate()
                encoder.wait(timeout=10)
        manager.stopUpdate()
        window.hide()
        rsp.terminate()
        rsp.wait(timeout=10)
        tf_sub.unregister()
        rospy.signal_shutdown('Full video recording completed')


if __name__ == '__main__':
    raise SystemExit(main())
