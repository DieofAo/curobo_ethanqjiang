"""CPU-only deterministic tests for RViz wall-clock playback scheduling."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from playback_timing import iter_playback_frames, validate_playback_timing  # noqa: E402


class FakeClock:
    def __init__(self, now=1234.0):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        if duration <= 0:
            raise AssertionError("scheduler must only request positive sleeps")
        self.sleeps.append(duration)
        self.now += duration


class PlaybackTimingTest(unittest.TestCase):
    def collect(self, times, work=0.0, **settings):
        fake = FakeClock()
        started = fake()
        frames = []
        for index in iter_playback_frames(times, clock=fake, sleep=fake.sleep, **settings):
            frames.append((index, fake() - started))
            fake.now += work
        return frames, fake

    def test_requested_speed_and_display_limit(self):
        times = np.linspace(0.0, 10.0, 10001)
        for speed in (0.5, 1.0, 2.0, 100.0):
            with self.subTest(speed=speed):
                frames, _ = self.collect(times, speed=speed)
                self.assertEqual(frames[0], (0, 0.0))
                self.assertEqual(frames[-1][0], len(times) - 1)
                self.assertAlmostEqual(frames[-1][1], 10.0 / speed, places=9)
                self.assertLessEqual(len(frames), int(10.0 / speed * 50) + 2)
                for (_, prior), (_, current) in zip(frames[:-2], frames[1:-1]):
                    self.assertGreaterEqual(current - prior, 0.02 - 1e-10)
                for index, wall in frames:
                    self.assertLessEqual(times[index], wall * speed + 1e-9)

    def test_cpu_delay_is_compensated_and_stale_samples_skipped(self):
        times = np.linspace(0.0, 1.0, 1001)
        frames, _ = self.collect(times, speed=2, work=0.037)
        self.assertEqual(frames[-1][0], 1000)
        self.assertGreaterEqual(frames[-1][1], 0.5)
        self.assertLess(frames[-1][1], 0.5 + 0.037 + 1e-10)
        self.assertLess(len(frames), 20)
        self.assertGreater(frames[1][0], 1)
        for index, wall in frames[:-1]:
            self.assertLessEqual(times[index], wall * 2 + 1e-9)
            self.assertLess(wall * 2 - times[index], 0.001 + 1e-9)

    def test_small_cpu_delay_does_not_accumulate(self):
        frames, _ = self.collect(np.linspace(0, 1, 101), work=0.005)
        # A final publish can wait for the preceding in-flight publish, but the
        # cost must not accumulate across all ordinary display samples.
        self.assertGreaterEqual(frames[-1][1], 1.0)
        self.assertLessEqual(frames[-1][1], 1.005 + 1e-10)

    def test_nonuniform_timestamps_never_publish_future_sample(self):
        times = np.array([0.0, 0.003, 0.19, 0.191, 0.8, 1.0])
        frames, _ = self.collect(times, display_hz=100)
        self.assertEqual([index for index, _ in frames], list(range(6)))
        for index, wall in frames:
            self.assertLessEqual(times[index], wall + 1e-10)
        self.assertAlmostEqual(frames[-1][1], 1.0)

    def test_nonzero_origin_is_normalized_without_modifying_input(self):
        times = np.array([42.0, 42.25, 42.5])
        original = times.copy()
        normalized = validate_playback_timing(times)
        np.testing.assert_array_equal(normalized, [0, 0.25, 0.5])
        self.assertFalse(np.shares_memory(times, normalized))
        frames, _ = self.collect(times, speed=2)
        self.assertAlmostEqual(frames[-1][1], 0.25)
        np.testing.assert_array_equal(times, original)

    def test_single_sample_and_zero_duration(self):
        self.assertEqual(self.collect([17])[0], [(0, 0.0)])
        self.assertEqual(self.collect([17, 17, 17])[0], [(0, 0.0), (2, 0.0)])

    def test_duplicate_timestamps_choose_latest_due(self):
        frames, _ = self.collect([0, 0.1, 0.1, 0.1, 0.2])
        self.assertEqual([index for index, _ in frames], [0, 3, 4])

    def test_very_fast_terminal_frame_not_delayed_by_display_period(self):
        frames, _ = self.collect([0, 0.1, 0.2], speed=100)
        self.assertEqual([index for index, _ in frames], [0, 2])
        self.assertAlmostEqual(frames[-1][1], 0.002)

    def test_collision_zero_stops_at_first_frame(self):
        frames, _ = self.collect([0, 1, 2], collision_index=0, speed=100)
        self.assertEqual(frames, [(0, 0.0)])

    def test_collision_cannot_be_skipped_when_work_overruns(self):
        frames, _ = self.collect([0, 0.1, 0.2, 0.3], collision_index=2,
                                 speed=100, work=0.04)
        self.assertEqual([index for index, _ in frames], [0, 2])
        self.assertAlmostEqual(frames[-1][1], 0.04)

    def test_collision_is_not_published_early(self):
        frames, _ = self.collect([0, 0.1, 0.2, 0.3], collision_index=2, speed=2)
        self.assertEqual(frames[-1][0], 2)
        self.assertAlmostEqual(frames[-1][1], 0.1)

    def test_collision_among_duplicate_timestamps(self):
        frames, _ = self.collect([0, 0, 0, 0.1], collision_index=1)
        self.assertEqual(frames, [(0, 0.0), (1, 0.0)])

    def test_fixed_rate_retains_all_frames_and_ignores_display_limit(self):
        frames, _ = self.collect([0, 0, 9, 10, 100], rate_hz=10, speed=2,
                                 display_hz=1)
        self.assertEqual([index for index, _ in frames], list(range(5)))
        np.testing.assert_allclose([wall for _, wall in frames], np.arange(5) * 0.05)

    def test_fixed_rate_compensates_publish_time_without_skipping(self):
        frames, _ = self.collect([0, 1, 2, 3], rate_hz=10, speed=2, work=0.03)
        np.testing.assert_allclose([wall for _, wall in frames], [0, 0.05, 0.1, 0.15])
        overloaded, _ = self.collect([0, 1, 2, 3], rate_hz=10, speed=2, work=0.08)
        self.assertEqual([index for index, _ in overloaded], [0, 1, 2, 3])
        np.testing.assert_allclose([wall for _, wall in overloaded], [0, 0.08, 0.16, 0.24])

    def test_fixed_rate_collision_stops_at_requested_frame(self):
        frames, _ = self.collect([0, 100, 200, 300], rate_hz=10,
                                 speed=2, collision_index=2)
        self.assertEqual([index for index, _ in frames], [0, 1, 2])
        self.assertAlmostEqual(frames[-1][1], 0.1)

    def test_shutdown_before_start(self):
        self.assertEqual(self.collect([0, 1], is_shutdown=lambda: True)[0], [])

    def test_shutdown_interrupts_wait(self):
        fake = FakeClock(0)
        frames = list(iter_playback_frames([0, 1, 2], clock=fake, sleep=fake.sleep,
                                          is_shutdown=lambda: fake.now >= 0.12))
        self.assertEqual(frames, [0])
        self.assertLessEqual(fake.now, 0.17)
        self.assertLessEqual(max(fake.sleeps), 0.05)

    def test_each_iteration_resets_playback_origin(self):
        fake = FakeClock()
        for _ in range(2):
            started = fake()
            frames = [(index, fake() - started) for index in iter_playback_frames(
                [40, 41], clock=fake, sleep=fake.sleep)]
            self.assertEqual(frames, [(0, 0), (1, 1)])
            fake.now += 100

    def test_invalid_times(self):
        for times in ([], [[0, 1]], [1, 0], [0, float("nan")],
                      [float("inf")], [float("-inf")], [-1e308, 1e308]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                validate_playback_timing(times)

    def test_invalid_settings(self):
        for name in ("speed", "display_hz", "rate_hz"):
            invalid = [-1, float("inf"), float("nan"), None, "fast"]
            if name != "rate_hz":
                invalid.append(0)
            for value in invalid:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_playback_timing([0, 1], **{name: value})
        for collision in (-1, 2, 0.5, True, float("nan"), "0"):
            with self.subTest(collision=collision), self.assertRaises(ValueError):
                validate_playback_timing([0, 1], collision_index=collision)

    def test_resume_uses_original_indices_and_only_remaining_duration(self):
        frames, _ = self.collect([10, 11, 12, 13, 14], start_index=2, speed=2)
        self.assertEqual([i for i, _ in frames], [2, 3, 4])
        np.testing.assert_allclose([t for _, t in frames], [0, 0.5, 1])

    def test_resume_at_final_frame_is_immediate(self):
        self.assertEqual(self.collect([0, 1, 2], start_index=2)[0], [(2, 0)])

    def test_resume_fixed_rate_preserves_remaining_frames(self):
        frames, _ = self.collect([0, 8, 19, 20], start_index=1, rate_hz=10, speed=2)
        self.assertEqual([i for i, _ in frames], [1, 2, 3])
        np.testing.assert_allclose([t for _, t in frames], [0, 0.05, 0.1])

    def test_resume_and_skip_still_stop_at_original_collision_index(self):
        frames, _ = self.collect(np.arange(10) * 0.1, start_index=3,
                                 collision_index=5, speed=100, work=0.04)
        self.assertEqual([i for i, _ in frames], [3, 5])

    def test_resume_at_collision_stays_there(self):
        frames, _ = self.collect([0, 1, 2], start_index=1, collision_index=1)
        self.assertEqual(frames, [(1, 0)])

    def test_resume_cannot_bypass_prior_collision(self):
        with self.assertRaisesRegex(ValueError, "bypass"):
            self.collect([0, 1, 2], start_index=2, collision_index=1)

    def test_resume_loop_origin_resets(self):
        fake = FakeClock()
        for _ in range(2):
            started = fake()
            frames = [(i, fake() - started) for i in iter_playback_frames(
                [0, 5, 6], start_index=1, clock=fake, sleep=fake.sleep)]
            self.assertEqual(frames, [(1, 0), (2, 1)])
            fake.now += 100

    def test_invalid_resume_indices(self):
        for index in (-1, 2, 0.5, True, None, "0"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_playback_timing([0, 1], start_index=index)


if __name__ == "__main__":
    unittest.main()
