# Input Parameters

This file explains the interactive inputs requested by `run_axionstar.py`.
When a prompt shows a value in brackets, pressing Enter uses that default value.

Example:

```text
mu_min [0.001]:
```

Pressing Enter uses `mu_min = 0.001`.

## General Options

| Input | Meaning |
|---|---|
| `--output` | Directory where `config.json`, `sequence.csv`, `max_summary.csv`, and diagnostics are written. |
| `print progress` | If `y`, print progress messages while the solver scans the sequence. |
| `scan mode` | For leading/first-order EFT, second-order EFT, and full-GR real-field runs: `coarse` runs only the broad scan, `refine_existing` reads an existing `sequence.csv` and runs only the refined scan, and `coarse_then_refine` does both stages in one command. |
| `accuracy mode` | For leading/first-order EFT and second-order EFT: `fast` is for exploratory scans, `standard` is the default balance, and `accurate` is for narrow final scans. |
| `path to previous sequence.csv` | Used only by `refine_existing`. This file should usually come from a previous coarse run with the same physical model and coupling. |
| `number of refined points` | Number of points in the refinement scan. |
| `refinement width factor` | Multiplies the coarse-grid spacing around the maximum to choose the refined scan width. |
| `stop after consecutive failed mu points` | For EFT scans, stop the scan after this many failed neighboring points. Use `0` to disable this stopping rule. |
| `use custom initial shooting seed` | Advanced EFT option. Usually leave this as `n`; use `y` only when continuing from a known nearby successful solution. |

The file `attempt_log.csv` records per-point diagnostic information, including
the shooting residual, relative tail size, elapsed time, and solver-call counts
when available.  For EFT runs, `sequence.csv` and `attempt_log.csv` are updated
after each completed `mu` point, so partial results remain available if a long
run is stopped.

## Leading/First-Order Effective Theory

The first regime contains two related models:

| Option | Meaning |
|---|---|
| `leading` | Nonrelativistic Schrodinger-Poisson limit. |
| `first_order` | First-order relativistic EFT correction. |

The code stores these choices internally as `model="nonrel"` and
`model="relativistic"`.

| Input | Meaning |
|---|---|
| `scan mode` | `coarse` scans the requested broad `mu` interval. `refine_existing` reads a previous `sequence.csv` and builds a narrower `mu` grid around its maximum-mass row. `coarse_then_refine` first runs the broad scan and then automatically refines around the maximum found in that same run. |
| `accuracy mode` | `fast` uses looser tolerances and fewer retry attempts for exploratory scans. `standard` uses the package defaults. `accurate` uses stricter tolerances and more mass-grid points for narrow final scans. |
| `delta = lambda M_Pl^2 / m^2` | Dimensionless quartic self-interaction parameter in the EFT convention. Negative values correspond to attractive self-interaction. |
| `mu_min for coarse scan`, `mu_max for coarse scan` | Range of the dimensionless binding parameter scanned in `coarse` or `coarse_then_refine` mode. These are not requested in `refine_existing` mode because the refined range is read from the previous `sequence.csv`. |
| `number of coarse mu points` | Number of values between `mu_min` and `mu_max` in the coarse scan. |
| `coarse mu spacing` | `geom` uses logarithmic/geometric spacing; `linear` uses equally spaced values. Geometric spacing is useful when small `mu` values need more resolution. |
| `adapt r_max using the scalar decay length` | If `y`, choose the radial box size using the estimated exponential tail length. |
| `minimum starting r_max` | Lower bound for the first radial outer boundary used in a shooting attempt. |
| `minimum r_max safety limit` | Lower bound for the largest radial box allowed during retry attempts. |
| `starting box in decay lengths` | Adaptive initial box size measured in units of `1/Omega`, where `Omega` is the scalar decay rate. |
| `safety limit in decay lengths` | Adaptive upper box limit measured in units of `1/Omega`. This prevents very diffuse points from being rejected only because the radial domain was too short. |
| `path to previous sequence.csv` | Coarse run used to locate the refined `mu` interval in `refine_existing` mode. |
| `number of refined points` | Number of `mu` values in the refined scan. |
| `refinement width factor` | Controls how wide the refined `mu` interval is around the previous maximum. |
| `stop after consecutive failed mu points` | Stops the scan after repeated neighboring failures. This is useful for exploratory scans that have moved outside the solvable branch. |
| `use custom initial shooting seed` | Advanced option for setting `alpha0_seed` and `beta0_seed`. The default seed is usually sufficient. |

For this regime the asymptotic scalar tail is estimated as

```text
f(r) ~ exp(-Omega r) / r,
Omega^2 = 2 mu - mu^2.
```

Smaller `mu` gives a longer tail, so adaptive radial boxes are usually safer
than a single fixed `r_max`.

The output column `tail_fraction` is the largest value of `|f(r)|/|f(0)|` in
the outer part of the radial domain.  It is a scale-independent check that the
solution has decayed before the boundary.

## Second-Order EFT

The second-order solver evolves the coupled radial functions

```text
F, Phi, Psi, A, B
```

with the shifted asymptotic condition

```text
Phi(infinity) = mu + mu^2/2 + mu^3/3.
```

| Input | Meaning |
|---|---|
| `scan mode` | `coarse` scans the requested broad `mu` interval. `refine_existing` reads a previous `sequence.csv` and builds a narrower `mu` grid around its maximum-mass row. `coarse_then_refine` first runs the broad scan and then automatically refines around the maximum found in that same run. |
| `accuracy mode` | `fast` uses looser tolerances, fewer retry attempts, and smaller radial-box defaults for exploratory scans. `standard` uses the package defaults. `accurate` uses stricter tolerances and more mass-grid points for narrow final scans. |
| `delta = lambda M_Pl^2 / m^2` | Dimensionless quartic self-interaction parameter. |
| `eta = m^2 / M_Pl^2` | Dimensionless gravitational coupling parameter appearing in the second-order terms. |
| `mu_min for coarse scan`, `mu_max for coarse scan` | Range of binding parameter values scanned in `coarse` or `coarse_then_refine` mode. These are not requested in `refine_existing` mode because the refined range is read from the previous `sequence.csv`. |
| `number of coarse mu points` | Number of sequence points in the coarse scan. |
| `adapt r_max using the scalar decay length` | If `y`, enlarge the radial box using the second-order decay rate. |
| `minimum starting r_max` | Lower bound for the first radial outer boundary in the shooting solve. |
| `minimum r_max safety limit` | Lower bound for the largest radial box allowed during retries. |
| `starting box in decay lengths` | Initial adaptive radial box size in units of the second-order decay length. |
| `safety limit in decay lengths` | Maximum adaptive radial box size in decay-length units. |
| `path to previous sequence.csv` | Coarse run used to locate the refined `mu` interval in `refine_existing` mode. |
| `number of refined points` | Number of `mu` values in the refined scan. |
| `refinement width factor` | Controls how wide the refined `mu` interval is around the previous maximum. |
| `stop after consecutive failed mu points` | Stops the scan after repeated neighboring failures. A small value such as `2` or `3` is useful for difficult exploratory scans. |
| `use custom initial shooting seed` | Advanced option for setting `F0_seed`, `Phi0_seed`, `Psi0_seed`, `A0_seed`, and `B0_seed`. The default seed is usually sufficient, but a nearby successful seed can help difficult branches. |

The output column `tail_fraction` is the largest value of `|F(r)|/|F(0)|` in
the outer part of the radial domain.

## Full-GR Real Field

The real-field solver computes oscillaton sequences.  The scalar field is
expanded in odd cosine harmonics, while metric functions are expanded in even
harmonics.

| Input | Meaning |
|---|---|
| `scan mode` | `coarse` runs only a broad scan. `refine_existing` reads an existing coarse `sequence.csv` and runs only the refined scan. `coarse_then_refine` runs the broad scan first and then automatically refines around its maximum in the same run. |
| `lambda_tilde in real-field potential` | Quartic coupling in the real-field full-GR potential convention used by the solver. |
| `phi0_min for coarse scan`, `phi0_max for coarse scan` | Range of central scalar amplitudes in `coarse` or `coarse_then_refine` mode. These are not requested in `refine_existing` mode because the refined range is read from the previous `sequence.csv`. |
| `number of coarse phi0 points` | Number of central-amplitude values in the coarse scan. |
| `maximum metric harmonic` | Largest metric harmonic retained in the Fourier expansion. Higher values are more expensive. |
| `n_mesh` | Number of radial mesh points used by the boundary-value solver. |
| `n_theta` | Number of time-phase samples used for Fourier projection. |
| `x_max` | Outer radial boundary for the full-GR real-field solve. |
| `BVP tolerance` | Error tolerance passed to the boundary-value solver. Smaller values are stricter and slower. |
| `max_nodes` | Maximum number of mesh nodes allowed during adaptive BVP refinement. |
| `path to previous sequence.csv` | Coarse run used to locate the region for a refined scan. |
| `number of refined points` | Number of central-amplitude values in the refined scan. |
| `refinement width factor` | Controls how wide the refined interval is around the previous maximum. |

In full-GR real-field `refine_existing` and `coarse_then_refine` modes, the code
first solves accepted coarse points as continuation warmup points before solving
the refined grid. These warmup points are written only to `attempt_log.csv` with
`warmup_only=True`; they are not included as refined `sequence.csv` rows.

The output column `tail_fraction` measures the largest relative scalar tail in
the outer part of the radial domain.

For leading/first-order EFT, second-order EFT, and full-GR real-field runs, use
`coarse_then_refine` when no previous coarse `sequence.csv` exists yet. Use
`refine_existing` only after a coarse run has already produced a valid
`sequence.csv`.

## Full-GR Complex Field

The complex-field solver computes stationary boson-star sequences.

| Input | Meaning |
|---|---|
| `Run single sequence or lambda_scan` | `single` computes one sequence. `lambda_scan` repeats the full sequence for several coupling values and preserves both the individual sequences and their maximum-mass summary. |
| `potential` | Scalar potential: `free`, `quartic`, `liouville`, or `log`. |
| `lambda in comparison convention (lambda = 2 Lambda_GR)` | Asked only in `single` mode. Coupling written in the package comparison convention. Internally the GR equations use `Lambda_GR = lambda/2`. |
| `comma-separated lambda values` | Coupling values used in `lambda_scan` mode, for example `-6,-8,-10,-12,-14`. This input replaces the single-lambda prompt in scan mode. |
| `phi0_min`, `phi0_max` | Range of central scalar amplitudes used independently for every lambda in the scan. |
| `number of phi0 points` | Number of central-amplitude values in the scan. |
| `r_max` | Radial outer boundary. Increase this for diffuse configurations whose mass has not saturated by the boundary. |
| `n_mesh` | Number of radial mesh points used by the BVP solver. |
| `BVP tolerance` | Error tolerance passed to the boundary-value solver. |
| `max_nodes` | Maximum number of mesh nodes allowed during adaptive BVP refinement. |

The output column `tail_fraction` measures the largest relative scalar tail in
the outer part of the radial domain.

For `lambda_scan`, the output directory contains:

```text
config.json             scan settings and lambda_values
sequence.csv           all sequence points from all lambdas
lambda_summary.csv     one accepted maximum-mass row per lambda
max_summary.csv        same per-lambda maximum summary
lambda_m6/             complete individual run for lambda=-6
lambda_m8/             complete individual run for lambda=-8
...
```

Each `lambda_*` subdirectory contains its own `config.json`, `sequence.csv`,
`max_summary.csv`, `attempt_log.csv`, `profiles.npz`, and (when an accepted
solution exists) `max_mass_profile.npz`.  This keeps the complete data needed
for mass-radius curves and for scalar/enclosed-mass profiles at the maximum.

## Choosing Scan Ranges

The default values are starting points, not universal physical limits.  A scan
should be widened or refined when:

- `max_at_edge=True` in `max_summary.csv`.
- `tail_fraction` is large, indicating that the final profile has not decayed
  by the outer boundary.
- The mass-radius curve changes abruptly between neighboring points.
- The solver reports many failed or rejected points near the region of
  interest.

For publication-quality results, inspect the sequence curve, the accepted
flags, and the diagnostic columns before using the reported maximum.
