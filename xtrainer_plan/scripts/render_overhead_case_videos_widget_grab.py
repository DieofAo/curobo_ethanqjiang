#!/usr/bin/env python3
"""Render saved case samples in an isolated RViz instance (no robot execution).

Each original 20 ms sample is shown, its robot_state_publisher TF is acknowledged,
then two RViz renders complete before capture. Encoding at 50 fps preserves 1x
trajectory timing even when offline rendering takes longer than real time.
Capture this program's QWidget directly rather than any desktop X11 pixels.
"""
from __future__ import annotations

import argparse
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--preview-only', action='store_true')
    ap.add_argument('--tiers', nargs='+', default=['min', 'median', 'max'])
    ap.add_argument('--experiment-label', default='V57')
    ap.add_argument('--title-mode', choices=['joint-range', 'place-roll'], default='joint-range')
    args = ap.parse_args()
    root = args.root.resolve()
    # Never attach this publisher to an existing/default user's ROS master.
    if os.environ.get('ROS_MASTER_URI') != 'http://127.0.0.1:11339':
        raise RuntimeError('Use the dedicated recording master on port 11339')
    from PyQt5 import QtCore, QtGui, QtWidgets
    from rviz import bindings as rviz
    import rospy
    from sensor_msgs.msg import JointState
    from tf2_msgs.msg import TFMessage
    from visualization_msgs.msg import Marker, MarkerArray
    from xtrainer_common import load_trajectory
    from play_trajectory_ros import build_static_markers
    from build_overhead_scene_urdf import load_resolved_config, scene_settings, build_scene, _serialize

    initial = root / args.tiers[0]
    cfg, src = load_resolved_config(None, str(initial))
    scene, summary = build_scene(scene_settings(cfg, src))
    urdf = _serialize(scene)
    (root / 'recording_scene.urdf').write_text(urdf)
    (root / 'recording_scene.json').write_text(json.dumps(summary, indent=2))
    children = [j.find('child').get('link') for j in ET.fromstring(urdf).findall('joint')
                if j.get('type') in ('revolute', 'continuous')]
    if len(children) != 6:
        raise RuntimeError('Expected precisely six active single-arm joints')
    rospy.init_node('case_video_renderer', anonymous=False, disable_signals=True)
    rospy.set_param('/robot_description', urdf)
    rsp = subprocess.Popen(['rosrun', 'robot_state_publisher', 'robot_state_publisher',
                            '__name:=video_robot_state_publisher', '_publish_frequency:=1000'])
    pub = rospy.Publisher('/joint_states', JointState, queue_size=1)
    markers = rospy.Publisher('/xtrainer_plan/markers', MarkerArray, queue_size=1, latch=True)
    latest = {}
    lock = threading.Lock()

    def receive_tf(msg):
        with lock:
            for tr in msg.transforms:
                latest[tr.child_frame_id] = tr.header.stamp.to_nsec()

    tf_sub = rospy.Subscriber('/tf', TFMessage, receive_tf, queue_size=100)
    app = QtWidgets.QApplication(sys.argv[:1])
    window = rviz.VisualizationFrame()
    window.setSplashPath('')
    window.initialize()
    config = rviz.Config()
    rviz.YamlConfigReader().readFile(config, str(Path(__file__).resolve().parents[1] /
                                               'config/xtrainer_overhead.rviz'))
    window.load(config)
    window.setHideButtonVisibility(False)
    for widget in window.findChildren(QtWidgets.QDockWidget):
        widget.hide()
    for widget in window.findChildren(QtWidgets.QToolBar):
        widget.hide()
    window.menuBar().hide()
    window.statusBar().hide()
    window.setWindowTitle(f'{args.experiment_label} grasp-place video recording (isolated visualization)')
    window.resize(1280, 800)
    window.move(100, 100)
    window.show()
    manager = window.getManager()
    manager.setFixedFrame('task_world')
    # Same camera for every tier; retain the complete CAD assembly and task grid.
    view = manager.getViewManager().getCurrent()
    view.subProp('Distance').setValue(1.65)
    view.subProp('Pitch').setValue(0.42)
    view.subProp('Yaw').setValue(2.35)
    manager.startUpdate()

    def pump(seconds=0.0):
        end = time.monotonic() + seconds
        while True:
            app.processEvents(QtCore.QEventLoop.AllEvents, 10)
            if time.monotonic() >= end:
                return
            time.sleep(0.002)

    def pose(q, names):
        stamp = rospy.Time.now()
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(names)
        msg.position = [float(x) for x in q]
        pub.publish(msg)
        deadline = time.monotonic() + 5
        while True:
            with lock:
                ready = all(latest.get(c, -1) == stamp.to_nsec() for c in children)
            if ready:
                break
            pump(0.003)
            if time.monotonic() > deadline:
                raise RuntimeError('Timed out waiting for this sample TF: ' + str(latest))
        before = manager.getFrameCount()
        while manager.getFrameCount() < before + 2:
            manager.queueRender()
            pump(0.005)
            if time.monotonic() > deadline:
                raise RuntimeError('Timed out waiting for RViz renders')
        return {'tf_stamp_ns': stamp.to_nsec(), 'render_count': manager.getFrameCount()}

    def capture():
        # X11 grabs only our known RViz window. Abort rather than substitute desktop.
        pix = window.grab()
        if pix.isNull() or pix.width() < 1000 or pix.height() < 600:
            raise RuntimeError('Invalid RViz window capture')
        return pix.toImage().convertToFormat(QtGui.QImage.Format_RGB888)

    phase_names = {'g_lift_in': 'APPROACH GRASP', 'grasp': 'GRASP',
                   'g_lift_out': 'LIFT AFTER GRASP', 'p_lift_in': 'TRANSFER TO PLACE',
                   'place': 'PLACE', 'p_lift_out': 'LIFT AFTER PLACE'}
    encoder = None
    try:
        pump(3)
        for tier in args.tiers:
            clip = root / tier
            data, meta = load_trajectory(str(clip))
            q = np.asarray(data['positions'])
            ts = np.asarray(data['times'])
            if not np.allclose(np.diff(ts), .02, rtol=0, atol=1e-7):
                raise RuntimeError('Expected uniform original 20 ms samples')
            if len(q) < 2 or not np.isfinite(q).all():
                raise RuntimeError('Invalid saved trajectory')
            items = [i for i in meta['items'] if i.get('success')]
            if len(items) != 1:
                raise RuntimeError('Expected one successful complete case')
            item = items[0]
            idx = int(item['index'])
            spans = np.rad2deg(np.ptp(q, axis=0))
            joint = int(np.argmax(spans)) + 1
            pos = item.get('position_raw', item['position'])
            clear = MarkerArray()
            marker = Marker()
            marker.action = Marker.DELETEALL
            clear.markers = [marker]
            markers.publish(clear)
            pump(.1)
            arr = build_static_markers(meta, 'LINK_0', data['ee_positions'],
                                       argparse.Namespace(show_walls=False, wall_alpha=.12))
            # Show the physical task plane/model and path, without large bounds labels.
            arr.markers = [m for m in arr.markers if m.ns in ('ee_path',)]
            markers.publish(arr)
            window.raise_()
            pose(q[0], data['joint_names'])
            pump(.25)
            image = capture()
            width, height = image.width(), image.height()

            def decorate(source, sample, hold_label=''):
                canvas = QtGui.QImage(width, height + 136, QtGui.QImage.Format_RGB888)
                canvas.fill(QtGui.QColor(22, 25, 31))
                painter = QtGui.QPainter(canvas)
                painter.drawImage(0, 92, source)
                painter.setPen(QtGui.QColor(241, 245, 249))
                painter.setFont(QtGui.QFont('DejaVu Sans', 18, QtGui.QFont.Bold))
                if args.title_mode == 'place-roll':
                    roll = float(meta['config']['pick_place']['place'].get('tool_z_rotation_deg', 0))
                    heading = f'PLACE TCP ROLL {roll:+.0f} deg'
                else:
                    heading = f'{tier.upper()} JOINT RANGE'
                painter.drawText(24, 31, f'{heading}  |  {args.experiment_label} case {idx + 1} (index {idx})  |  1x')
                painter.setFont(QtGui.QFont('DejaVu Sans', 13))
                painter.drawText(24, 60, f'Max joint span: J{joint} {spans[joint - 1]:.2f} deg    Grasp (x, y): ({pos[0]:.3f}, {pos[1]:.3f}) m')
                painter.setPen(QtGui.QColor(149, 214, 242))
                label = next((s['name'] for s in meta['extraction']['selection']['segments']
                              if s['clip_sample_range_half_open'][0] <= sample <
                              s['clip_sample_range_half_open'][1]), '')
                phase = next((text for key, text in phase_names.items() if label.endswith('_' + key)), label)
                painter.drawText(24, 84, f'{hold_label or phase}    t = {ts[sample]:.2f} / {ts[-1]:.2f} s')
                painter.setPen(QtGui.QColor(174, 182, 195))
                painter.setFont(QtGui.QFont('DejaVu Sans', 11))
                painter.drawText(24, height + 120, 'Saved planned samples | TCP -3 cm | joint limits inset 0.14 rad | visualization only, not execution approval')
                painter.end()
                return canvas

            preview = decorate(image, 0, 'START POSE')
            preview.save(str(clip / 'preview.png'))
            if args.preview_only:
                print(f'PREVIEW {clip / "preview.png"}', flush=True)
                continue
            output = root / f'{tier}_case_{idx + 1:03d}.mp4'
            if output.exists():
                raise FileExistsError(output)
            cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n', '-f', 'rawvideo',
                   '-pixel_format', 'rgb24', '-video_size', f'{width}x{height + 136}',
                   '-framerate', '50', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
                   '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
            encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)

            def write(frame):
                if frame.bytesPerLine() != width * 3:
                    raise RuntimeError('Unexpected RGB row padding')
                bits = frame.bits()
                bits.setsize(frame.byteCount())
                encoder.stdin.write(bytes(bits))

            for _ in range(50):
                write(preview)
            audit = []
            for sample, values in enumerate(q):
                evidence = pose(values, data['joint_names'])
                evidence.update(sample=sample, trajectory_time_s=float(ts[sample]))
                audit.append(evidence)
                frame_image = capture()
                painted = decorate(frame_image, sample)
                write(painted)
                if sample in {len(q) // 2, len(q) - 1}:
                    painted.save(str(clip / f'frame_{sample:04d}.png'))
                for segment in meta['extraction']['selection']['segments']:
                    if sample == segment['clip_sample_range_half_open'][1] - 1:
                        painted.save(str(clip / f"phase_{segment['phase']}.png"))
                if sample % 100 == 0:
                    print(f'RENDER {tier}: {sample}/{len(q)}', flush=True)
            for _ in range(50):
                write(decorate(frame_image, len(q) - 1, 'COMPLETE - LIFT AFTER PLACE'))
            encoder.stdin.close()
            if encoder.wait(timeout=30):
                raise RuntimeError('ffmpeg encoding failed')
            encoder = None
            evidence = {'output': str(output), 'case_index': idx, 'fps': 50,
                        'n_source_samples': len(q), 'n_video_frames': len(q) + 100,
                        'trajectory_duration_s': float(ts[-1]), 'speed': 1.0,
                        'start_hold_s': 1, 'end_hold_s': 1, 'width': width, 'height': height + 136,
                        'sampling': 'Every original sample, unchanged joints; two confirmed RViz renders after matching TF stamp',
                        'source_npz_sha256': hashlib.sha256((clip / 'trajectory.npz').read_bytes()).hexdigest(),
                        'video_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'frames': audit}
            (clip / 'video_render_audit.json').write_text(json.dumps(evidence, indent=2))
            print(f'COMPLETE {output}', flush=True)
    finally:
        if encoder is not None:
            encoder.stdin.close()
            encoder.terminate()
            encoder.wait(timeout=10)
        manager.stopUpdate()
        window.hide()
        rsp.terminate()
        rsp.wait(timeout=10)
        tf_sub.unregister()
        rospy.signal_shutdown('recording completed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
