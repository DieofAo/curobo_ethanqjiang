# Relative EE Analysis

This directory contains a standalone analysis script for comparing how the
nominal xtrainer URDF and calibrated xtrainer URDF affect relative
end-effector offsets.

The calibrated URDF is treated as ground truth:

- Ground truth: `robot/ur_description/xtrainer_cali.urdf`
- Nominal: `robot/ur_description/xtrainer.urdf`
- Shared cuRobo config: `xtrainer.yml`

The script reuses `xtrainer.yml` for link names, cspace, collision spheres,
self-collision ignore pairs, and buffers. It only swaps `urdf_path` when
constructing the two cuRobo robot configs.

## Analyses

Target 1:

1. Read a batch of joint vectors `q` from the MCAP.
2. Compute `T_gt(q)` with `xtrainer_cali.urdf`.
3. Compute `T_nominal(q)` with `xtrainer.urdf`.
4. Compare the raw FK poses `T_gt(q)` and `T_nominal(q)`.

Target 2:

1. Sample several random relative transforms `T_rel`.
2. Compare `T_gt(q) * T_rel` with `T_nominal(q) * T_rel`.

Target 3:

1. Build the same two right-multiplied targets as target 2.
2. Solve IK for `T_gt(q) * T_rel` in the ground-truth URDF.
3. Solve IK for `T_nominal(q) * T_rel` in the nominal URDF.
4. Use the original `q` as both `seed_config` and `retract_config`.
5. Enable cuRobo self-collision checking and self-collision optimization.
6. Evaluate both IK solutions with the ground-truth URDF and compare the two
   resulting EE poses.

FK sampling mode:

1. Ignore the MCAP and sample `q` directly inside joint limits.
2. Compute FK with the calibrated and nominal URDFs for every sampled `q`.
3. Report aggregate FK pose differences and the worst sampled configurations.
4. This is intended to estimate the largest end-effector effect introduced by
   the calibrated URDF over a broad joint-space coverage.

## MCAP Input

The default input is:

```bash
relative_ee_analysis/data/20260618-184047_optimized.mcap
```

The joint values are read from the logical path:

```bash
/robot/data/left_arm/observation/multibody_state/states
```

In the MCAP this resolves to the channel:

```bash
/robot/data/left_arm/observation
```

and the decoded ROS1 message field:

```bash
multibody_state.states
```

The first 6 `state.q` values are used as the arm joint vector. The script also
accepts this equivalent wording:

```bash
--mcap-path "/robot/data/left_arm/observation plus multibody_state.states[:].q"
```

## Run

Quick MCAP smoke test:

```bash
python3 relative_ee_analysis/analyze_relative_ee.py --dry-run-mcap --max-q 3
```

Run all analyses with conservative defaults:

```bash
python3 relative_ee_analysis/analyze_relative_ee.py
```

By default the script reads all q samples from the MCAP. Use `--max-q <N>`
only when you want a faster capped run.

For the full cuRobo FK/IK path, run in an environment with CUDA available.
If the local conda environment reports a `GLIBCXX` import error, prepend the
environment library path, for example:

```bash
env LD_LIBRARY_PATH=/home/ethanqjiang/miniconda3/envs/curobo/lib \
  conda run -n curobo python3 relative_ee_analysis/analyze_relative_ee.py
```

Smaller fast run:

```bash
python3 relative_ee_analysis/analyze_relative_ee.py \
  --max-q 10 \
  --rel-samples-per-q 5 \
  --num-seeds 16 \
  --batch-size 32
```

Skip IK target 3:

```bash
python3 relative_ee_analysis/analyze_relative_ee.py --skip-target3
```

Adjust the random relative EE range:

```bash
python3 relative_ee_analysis/analyze_relative_ee.py \
  --translation-range-m 0.03 \
  --rotation-range-deg 5.0
```

Run only the joint-limit FK sampling experiment:

```bash
python3 relative_ee_analysis/analyze_relative_ee.py \
  --fk-sample-only \
  --fk-sample-method sobol \
  --fk-sample-count 1000000 \
  --fk-sample-batch-size 65536
```

The default FK sampling limits are the intersection of calibrated and nominal
URDF limits, so every sampled `q` is valid for both models. Use
`--fk-limit-source gt` or `--fk-limit-source nominal` to force one URDF's
limits.

Exact Cartesian grid sampling is available, but use it carefully:

```bash
python3 relative_ee_analysis/analyze_relative_ee.py \
  --fk-sample-only \
  --fk-sample-method grid \
  --fk-grid-step-rad 0.05
```

At `0.01` rad over the calibrated xtrainer limits, a full 6-DOF Cartesian grid
is about `1.3e16` configurations, so the script has a safety cap
(`--fk-grid-max-samples`) and will suggest Sobol/random sampling when the grid
is too large.

## Outputs

By default, outputs are written to:

```bash
relative_ee_analysis/results/<timestamp>/
```

Files:

- `summary.json`: aggregate statistics and configuration metadata.
- `results.npz`: all arrays for later plotting or notebook analysis.
- `q_samples.csv`: extracted q samples.
- `target1_metrics.csv`: raw FK pose difference for each q.
- `target2_metrics.csv`: relative-FK pose difference for each q and random relative EE.
- `target3_metrics.csv`: IK and ground-truth evaluation metrics, if target 3 is enabled.
- `fk_sample_metrics.csv`: sampled joint vectors and raw FK pose differences, if
  `--fk-sample-only` is enabled and CSV output is not disabled. This also
  includes the calibrated and nominal EE radius for reach-vs-error plots.
- `fk_sample_stats.csv`: compact max/mean statistics for FK sampling metrics,
  if `--fk-sample-only` is enabled.
- `fk_sample_results.npz`: FK sampling error arrays and joint limits, if
  `--fk-sample-only` is enabled.

Translations are reported in meters and millimeters. Rotations are reported in
radians and degrees.

## Plot

Generate PNG plots from one result directory:

```bash
python3 relative_ee_analysis/plot_results.py \
  relative_ee_analysis/results/20260621-170909
```

The images are written to:

```bash
relative_ee_analysis/results/20260621-170909/plots/
```

Generated plots include joint curves, target 1 raw-FK curves and histograms,
target 2 relative-FK curves and histograms, and target 3 IK/GT-evaluation
curves if `target3_metrics.csv` exists. When both target 1 and target 2 CSVs
exist, `target2_extra_error_minus_target1_by_q.png` is also generated to show
the mean target 2 relative-FK error per q minus the target 1 raw-FK error at the
same q. If `fk_sample_metrics.csv` exists, FK sampling curves and histograms are
also generated.
For q inspection, `q_samples_joint_deltas.png` is often clearer than the raw
absolute joint plot because it shows each joint relative to its first sample.

For FK sampling CSV/NPZ outputs, there is also a standalone plotting script:

```bash
python3 relative_ee_analysis/plot_fk_sample_csv.py \
  relative_ee_analysis/results/<timestamp>
```

It reads `fk_sample_metrics.csv` when present, otherwise it reads
`fk_sample_results.npz`. It writes sample-index curves, histograms, and
reach-vs-error plots to:

```bash
relative_ee_analysis/results/<timestamp>/fk_sample_plots/
```

Older NPZ outputs that do not contain `gt_ee_radius_m` can still generate
sample-index curves and histograms, but reach-vs-error plots are skipped.
