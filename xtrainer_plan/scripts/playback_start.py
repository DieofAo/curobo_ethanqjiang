"""Resolve an RViz display start point without importing ROS or changing data."""

import math
import numbers
from collections.abc import Mapping

import numpy as np

from playback_timing import validate_playback_timing


def _integer(value, name, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _bounded_real(value, name, maximum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{name} must be finite and in [0, {maximum:g}]")
    return float(value)


def _item_start(meta, frame_count, requested_item):
    """Validate the single-arm concatenation schema and return its prefix sum.

    ``plan_sequence`` removes duplicate boundaries *between segments*, but its
    first segment retains q_cur. ``plan_pick_place`` then concatenates complete
    successful item arrays without removing another frame. Consequently an
    item's first frame is sum(previous successful item.n_points), not sum - 1;
    that stored frame already contains the preceding successful item's end.
    Failed items add no frames. Never infer these offsets from duration_s.
    """
    item_number = _integer(requested_item, "start_item (1-based original case)", 1)
    if not isinstance(meta, Mapping):
        raise ValueError("start_item requires single-arm pick/place trajectory metadata")
    task_type = meta.get("task_type")
    if task_type not in (None, "pick_place_cycle"):
        raise ValueError(
            f"start_item does not support task_type={task_type!r}; "
            "use start_percent, start_time, or start_frame"
        )
    # Older single-arm incremental saves omit task_type. Their validated items
    # and frame counts still define the same unambiguous concatenation schema.
    items = meta.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("start_item requires a nonempty metadata items list")
    metadata_count = _integer(meta.get("n_points"), "metadata n_points", 1)
    if metadata_count != frame_count:
        raise ValueError(
            f"metadata n_points={metadata_count} does not match NPZ frames={frame_count}"
        )
    total_items = meta.get("n_items_total")
    if total_items is not None:
        total_items = _integer(total_items, "metadata n_items_total", 1)

    starts = {}
    frame_offset = 0
    success_count = 0
    failed_count = 0
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("metadata items must contain objects")
        index = _integer(item.get("index"), "metadata item index")
        if index in starts:
            raise ValueError(f"metadata contains duplicate item index={index}")
        if total_items is not None and index >= total_items:
            raise ValueError(f"metadata item index={index} exceeds n_items_total")
        success = item.get("success")
        if not isinstance(success, (bool, np.bool_)):
            raise ValueError(f"metadata item index={index} requires boolean success")
        if not success:
            if "n_points" in item and _integer(
                item["n_points"], f"failed item index={index} n_points"
            ) != 0:
                raise ValueError(f"failed item index={index} must not contribute frames")
            starts[index] = None
            failed_count += 1
            continue
        count = _integer(item.get("n_points"), f"item index={index} n_points", 1)
        if "segments" in item:
            segments = item["segments"]
            if not isinstance(segments, list) or not segments:
                raise ValueError(f"item index={index} segments must be a nonempty list")
            segment_total = 0
            for segment in segments:
                if not isinstance(segment, Mapping):
                    raise ValueError(f"item index={index} segments must contain objects")
                segment_total += _integer(
                    segment.get("n_points"), f"item index={index} segment n_points", 1
                )
            if segment_total - (len(segments) - 1) != count:
                raise ValueError(f"item index={index} segment frame counts do not match n_points")
        starts[index] = frame_offset
        frame_offset += count
        success_count += 1

    if frame_offset != frame_count:
        raise ValueError(
            f"sum of successful item n_points={frame_offset} does not match "
            f"NPZ frames={frame_count}; incompatible metadata/trajectory"
        )
    for field, expected in (
        ("n_items_success", success_count), ("n_items_skipped", failed_count),
        ("n_items_done", len(items)),
    ):
        if field in meta and _integer(meta[field], f"metadata {field}") != expected:
            raise ValueError(f"metadata {field} does not match items")
    index = item_number - 1
    if index not in starts:
        raise ValueError(f"original case #{item_number} is not present in metadata items")
    if starts[index] is None:
        raise ValueError(f"original case #{item_number} failed planning and has no trajectory")
    return starts[index]


def resolve_playback_start(
    times, meta, *, start_percent=None, start_time=None, start_frame=None, start_item=None
):
    """Return a zero-based frame index in the original, unmodified NPZ arrays.

    Selectors are mutually exclusive. Percentages [0, 100] refer to elapsed
    trajectory time, not sample count; times are seconds relative to times[0].
    These selectors choose the first sample at or after the requested time.
    Explicit 0%/100% select the first/last frame, even for duplicate timestamps.
    A positive start_time equal to duration selects the last frame; zero always
    selects the first (including a zero-duration trajectory).
    Frame numbers are zero-based. Item numbers are one-based *original* case
    indices (metadata items.index + 1), including failed/missing cases which
    raise an error rather than silently selecting the next successful case.
    With no selector, return the first frame. Timing validation always runs.
    """
    timestamps = validate_playback_timing(times)
    selectors = (start_percent, start_time, start_frame, start_item)
    if sum(value is not None for value in selectors) > 1:
        raise ValueError("start_percent, start_time, start_frame, and start_item are mutually exclusive")
    if start_item is not None:
        return _item_start(meta, len(timestamps), start_item)
    if start_frame is not None:
        index = _integer(start_frame, "start_frame (zero-based)")
        if index >= len(timestamps):
            raise ValueError(f"start_frame must be in [0, {len(timestamps) - 1}]")
        return index
    duration = float(timestamps[-1])
    if start_percent is not None:
        percent = _bounded_real(start_percent, "start_percent", 100.0)
        if percent == 0:
            return 0
        if percent == 100:
            return len(timestamps) - 1
        target = duration * (percent / 100.0)
    elif start_time is not None:
        target = _bounded_real(start_time, "start_time (seconds from trajectory start)", duration)
        if target == 0:
            return 0
        if target == duration:
            return len(timestamps) - 1
    else:
        return 0
    return int(np.searchsorted(timestamps, target, side="left"))
