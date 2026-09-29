"""Wall-clock trajectory display scheduling, independent of ROS.

Timestamp mode skips stale *display* samples; it never edits trajectory data.
Explicit fixed-rate mode retains every sample and compensates deadlines for the
time the caller spends publishing each one.
"""

import math
import numbers
import time

import numpy as np


def validate_playback_timing(
    times, *, speed=1.0, rate_hz=0.0, display_hz=50.0, collision_index=None,
    start_index=0,
):
    """Validate settings and return a copied, zero-origin 1-D time array.

    This can be called before initializing ROS. ``collision_index`` is either
    None or an in-bounds integer identifying the first frame that must stop
    playback; it is not permitted to be skipped at high playback speeds.
    """
    timestamps = np.array(times, dtype=np.float64, copy=True)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError("times must be a nonempty 1-D array")
    if not np.isfinite(timestamps).all():
        raise ValueError("times must contain only finite values")
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("times must be nondecreasing")
    for name, value, allow_zero in (
        ("speed", speed, False),
        ("rate_hz", rate_hz, True),
        ("display_hz", display_hz, False),
    ):
        try:
            valid = math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
        except TypeError:
            valid = False
        if not valid:
            bound = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {bound}")
    if (
        isinstance(start_index, (bool, np.bool_))
        or not isinstance(start_index, numbers.Integral)
        or not 0 <= start_index < timestamps.size
    ):
        raise ValueError("start_index must be an in-bounds integer")
    if collision_index is not None:
        if (
            isinstance(collision_index, (bool, np.bool_))
            or not isinstance(collision_index, numbers.Integral)
            or not 0 <= collision_index < timestamps.size
        ):
            raise ValueError("collision_index must be an in-bounds integer or None")
        if start_index > collision_index:
            raise ValueError("start_index cannot bypass the first recorded collision")
    with np.errstate(over="ignore", invalid="ignore"):
        timestamps -= timestamps[0]
    if not np.isfinite(timestamps).all():
        raise ValueError("normalized times must remain finite")
    return timestamps


def iter_playback_frames(
    times,
    *,
    speed=1.0,
    rate_hz=0.0,
    display_hz=50.0,
    collision_index=None,
    start_index=0,
    clock=time.monotonic,
    sleep=time.sleep,
    is_shutdown=lambda: False,
):
    """Yield frame indices on a fresh wall-clock timeline for each invocation.

    With ``rate_hz == 0``, publish the newest due timestamp at no more than
    ``display_hz`` ordinary frames per second. ``start_index`` is immediate;
    the final/collision frame is delivered as soon as due, even if that means
    a shorter final display interval. No frame beyond a collision is yielded.

    With ``rate_hz > 0``, yield all frames at ``rate_hz * speed`` with absolute
    deadlines, independent of timestamps and ``display_hz``. If publishing
    itself exceeds that period, this explicit mode cannot attain its requested
    speed; timestamp mode should be used when skipping display frames is desired.
    """
    timestamps = validate_playback_timing(
        times, speed=speed, rate_hz=rate_hz, display_hz=display_hz,
        collision_index=collision_index, start_index=start_index,
    )
    start_index = int(start_index)
    stop_index = timestamps.size - 1 if collision_index is None else int(collision_index)
    if rate_hz > 0:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            offsets = np.arange(stop_index - start_index + 1, dtype=np.float64) / rate_hz / speed
    else:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            offsets = (timestamps[start_index:stop_index + 1] - timestamps[start_index]) / speed
    if not np.isfinite(offsets).all():
        raise ValueError("playback duration must remain finite at the requested speed")

    def wait_until(deadline):
        while not is_shutdown():
            remaining = deadline - clock()
            if remaining <= 0:
                return True
            sleep(min(remaining, 0.05))
        return False

    if is_shutdown():
        return
    started = clock()
    deadlines = started + offsets
    if not np.isfinite(deadlines).all():
        raise ValueError("playback deadlines must remain finite")
    yield start_index
    last_local = stop_index - start_index
    if last_local == 0:
        return

    if rate_hz > 0:
        for index in range(1, last_local + 1):
            if not wait_until(deadlines[index]):
                return
            yield index + start_index
        return

    period = 1.0 / display_hz
    next_display = started + period
    index = 0
    while index < last_local:
        # Wait until both a new sample and a display slot are available, except
        # for the terminal sample, which must not acquire an artificial tail.
        deadline = min(deadlines[last_local], max(deadlines[index + 1], next_display))
        if not wait_until(deadline):
            return
        now = clock()
        due_index = min(last_local, int(np.searchsorted(deadlines, now, side="right")) - 1)
        if due_index <= index:
            continue
        index = due_index
        next_display = now + period
        yield index + start_index
