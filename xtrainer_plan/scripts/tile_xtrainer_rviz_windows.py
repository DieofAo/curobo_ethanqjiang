#!/usr/bin/env python3
"""Label and tile only the explicitly listed XTrainer RViz X11 windows."""
from __future__ import annotations

import argparse
import ctypes
import time


class ClientMessage(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong),
                ('send_event', ctypes.c_int), ('display', ctypes.c_void_p),
                ('window', ctypes.c_ulong), ('message_type', ctypes.c_ulong),
                ('format', ctypes.c_int), ('data', ctypes.c_long * 5)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('windows', nargs=4, metavar='WINDOW_ID:LABEL')
    args = ap.parse_args()
    lib = ctypes.CDLL('libX11.so.6')
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.XInternAtom.restype = ctypes.c_ulong
    lib.XRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.XRootWindow.restype = ctypes.c_ulong
    lib.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                               ctypes.c_long, ctypes.POINTER(ClientMessage)]
    lib.XMoveResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                      ctypes.c_int, ctypes.c_int,
                                      ctypes.c_uint, ctypes.c_uint]
    lib.XStoreName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_char_p]
    lib.XFlush.argtypes = [ctypes.c_void_p]
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = lib.XOpenDisplay(b':0')
    if not display:
        raise RuntimeError('Unable to open user desktop :0')
    try:
        root = lib.XRootWindow(display, 0)
        wm_state = lib.XInternAtom(display, b'_NET_WM_STATE', 0)
        max_h = lib.XInternAtom(display, b'_NET_WM_STATE_MAXIMIZED_HORZ', 0)
        max_v = lib.XInternAtom(display, b'_NET_WM_STATE_MAXIMIZED_VERT', 0)
        values = []
        for spec in args.windows:
            raw, label = spec.split(':', 1)
            values.append((int(raw, 0), label))
        if len({window for window, _ in values}) != 4:
            raise ValueError('Expected four distinct window IDs')
        for window, _ in values:
            event = ClientMessage()
            event.type = 33  # X11 ClientMessage
            event.send_event = 1
            event.display = display
            event.window = window
            event.message_type = wm_state
            event.format = 32
            event.data[0] = 0  # remove maximize state
            event.data[1] = max_h
            event.data[2] = max_v
            event.data[3] = 1  # application request
            lib.XSendEvent(display, root, 0, (1 << 20) | (1 << 19), ctypes.byref(event))
        lib.XFlush(display)
        time.sleep(.4)
        # Current desktop is 3840x2160; place four 1880x1000 windows with gaps.
        positions = [(32, 58), (1944, 58), (32, 1100), (1944, 1100)]
        for (window, label), (x, y) in zip(values, positions):
            lib.XMoveResizeWindow(display, window, x, y, 1880, 1000)
            lib.XStoreName(display, window, label.encode('utf-8'))
            print(f'{label}: window {window:#x}, geometry 1880x1000+{x}+{y}', flush=True)
        lib.XFlush(display)
    finally:
        lib.XCloseDisplay(display)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
