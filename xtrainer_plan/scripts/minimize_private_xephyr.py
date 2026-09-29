#!/usr/bin/env python3
"""Minimize only the newly started private Xephyr window on the user desktop."""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('nested_display')
    args = ap.parse_args()
    needle = 'Xephyr on ' + args.nested_display
    lib = ctypes.CDLL('libX11.so.6')
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XIconifyWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int]
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    env = dict(os.environ, DISPLAY=':0', XAUTHORITY='/home/ethanqjiang/.Xauthority')
    for _ in range(30):
        output = subprocess.check_output(['xwininfo', '-root', '-tree'], env=env, text=True)
        match = next((re.match(r'\s*(0x[0-9a-f]+)', line)
                      for line in output.splitlines() if needle in line), None)
        if match:
            display = lib.XOpenDisplay(b':0')
            if display:
                lib.XIconifyWindow(display, int(match.group(1), 16), 0)
                lib.XCloseDisplay(display)
                print(f'Minimized isolated {needle}', flush=True)
                return 0
        time.sleep(.3)
    raise RuntimeError(f'Private Xephyr window not found for minimization: {needle}')


if __name__ == '__main__':
    raise SystemExit(main())
