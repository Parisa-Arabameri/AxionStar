# Troubleshooting

This file lists common numerical issues and practical adjustments.

## `max_at_edge=True`

The largest mass in the scan occurs at the edge of the scanned parameter
range.  Do not interpret that row as a physical maximum until the scan is
widened.

Recommended actions:

- For leading/first-order and second-order EFT runs, widen `mu_min` or
  `mu_max`.
- For full-GR runs, widen `phi0_min` or `phi0_max`.
- Increase the number of scan points after widening the range.

## Large `tail_fraction`

`tail_fraction` measures the largest scalar amplitude in the outer part of the
radial domain relative to the central amplitude.  A large value means the
profile has not decayed enough by the boundary.

Recommended actions:

- For EFT runs, keep adaptive `r_max` enabled.
- Increase `safety limit in decay lengths`.
- Increase `minimum r_max safety limit`.
- For full-GR complex-field runs, increase `r_max`.
- For full-GR real-field runs, increase `x_max`.

## Many Failed Shooting Attempts

Shooting can fail when adjacent scan points are too far apart or when the
initial continuation guess is poor.

Recommended actions:

- Increase the number of scan points.
- Use a narrower scan interval around the region of interest.
- Use `coarse_then_refine` after a successful broad scan range is chosen, or
  use `refine_existing` with a saved coarse `sequence.csv`.
- For difficult EFT scans, start with `accuracy mode = fast`, a small number
  of coarse `mu` points, and a small consecutive-failure limit. Rerun the
  interesting interval with `standard` or `accurate` before using final values.
- For EFT runs, increase `max_attempts` in the corresponding configuration
  class if more retries are needed.

## Long EFT Runs

Leading/first-order and second-order EFT scans autosave `sequence.csv` and
`attempt_log.csv` after each completed `mu` point.  If a run is stopped before
the full scan finishes, inspect these files before deleting the output folder.

For large attractive couplings in the second-order solver, first locate the
branch with:

```text
accuracy mode = fast
scan mode = coarse
small number of coarse mu points
stop after consecutive failed mu points = 2 or 3
```

Then use `refine_existing` with the saved `sequence.csv` or rerun a narrower
coarse interval with stricter accuracy.

## Abrupt Jumps In The Mass-Radius Curve

Large jumps usually indicate a failed continuation step, a bad branch switch,
or an under-resolved scan.

Recommended actions:

- Inspect `attempt_log.csv`.
- Increase the number of scan points.
- Reduce the scanned interval and rerun around the suspicious region.
- Check the `accepted`, `tail_fraction`, `residual_norm`, and `nodes` columns.
- For leading/first-order EFT, second-order EFT, and full-GR real-field runs,
  use `coarse_then_refine` or rerun with `refine_existing` from a saved coarse
  `sequence.csv`.

## Boundary-Value Solver Reaches `max_nodes`

The full-GR solvers use SciPy's boundary-value solver.  If the solver reaches
`max_nodes`, it could not resolve the solution with the allowed adaptive mesh.

Recommended actions:

- Increase `max_nodes`.
- Relax `BVP tolerance` slightly for exploratory scans.
- Increase `n_mesh` for the initial mesh.
- Use a smaller scan interval and continuation from nearby accepted points.


## Full-GR Real Refined Scan Jumps to a Large-Radius Branch

For real-field oscillaton refinement, the refined grid is close to the maximum
and can be difficult to solve from the analytic initial guess. The package uses
accepted coarse points as warmup continuation points before starting the refined
grid. These are recorded in `attempt_log.csv` with `warmup_only=True`. If a
large-radius or high-harmonic point still appears, check `phi3_over_phi1`,
`R_jump_factor`, and `branch_continuity_ok`; then rerun with a denser coarse scan
or a smaller refinement width factor.

## Refinement Cannot Find `sequence.csv`

For leading/first-order EFT, second-order EFT, and full-GR real-field runs,
the `refine_existing` mode expects an already completed coarse run.  If the
coarse file does not exist, run one of these first:

```bash
python run_axionstar.py --output outputs/coarse_run
```

and choose `coarse`; or choose `coarse_then_refine` to do both stages in one
run.  Then, when using `refine_existing`, enter the path to the coarse
`sequence.csv`, for example:

```text
outputs/coarse_run/sequence.csv
```

## Publication-Quality Runs

Before using a reported maximum in a figure or table:

- Confirm `accepted=True`.
- Confirm `max_at_edge=False`.
- Confirm `tail_fraction` is small.
- Confirm the mass-radius curve is smooth near the maximum.
- Save the corresponding `config.json` with the output data.
