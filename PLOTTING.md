# Plotting Results

The script `examples/plot_results.py` reads output files from `run_axionstar.py`
and makes common diagnostic plots.

The examples below assume a run directory such as:

```text
outputs/test_run/
  config.json
  sequence.csv
  max_summary.csv
  attempt_log.csv
```

From the repository root, use:

```bash
python examples/plot_results.py --help
```

From inside a run directory, use the full path to the plotting script, for
example:

```bash
cd outputs/test_run
python ../../examples/plot_results.py mass-radius sequence.csv --output mass_radius.png
```

Adjust the number of `..` entries if the run directory is nested more deeply.

## Mass-Radius Curve

Plot one run:

```bash
python examples/plot_results.py mass-radius \
  outputs/test_run/sequence.csv \
  --output outputs/test_run/mass_radius.png
```

By default, axes are labeled with the natural units used in the output files:

```text
R_95 [m^{-1}]
M [M_Pl^2/m]
```

Here `m` is the scalar mass and `M_Pl` is the Planck mass in the convention of
the solver.  If a numerical conversion to another unit is known, rescale only
the plotted values:

```bash
python examples/plot_results.py mass-radius \
  outputs/test_run/sequence.csv \
  --radius-scale 1.0 \
  --radius-unit km \
  --mass-scale 1.0 \
  --mass-unit Msun \
  --output outputs/test_run/mass_radius_physical_units.png
```

Replace `1.0` by the appropriate conversion factors for the regime and
physical model.  The CSV file is not modified.

Compare several regimes:

```bash
python examples/plot_results.py mass-radius \
  outputs/leading_first/sequence.csv \
  outputs/second_order/sequence.csv \
  outputs/full_gr_real/sequence.csv \
  outputs/full_gr_complex/sequence.csv \
  --output outputs/comparison_mass_radius.png
```

If a run contains both coarse and refined points, the plotting script sorts the
accepted rows by `mu` or central amplitude before drawing connected lines.  This
prevents artificial line segments between the end of the coarse scan and the
start of the refined scan.

## Plot Selected Columns

Plot mass and compactness against `mu`:

```bash
python examples/plot_results.py quantity \
  outputs/test_run/sequence.csv \
  --x mu \
  --y mass compactness_95 \
  --output outputs/test_run/mass_compactness_vs_mu.png
```

For two quantities with different scales, use separate y axes:

```bash
python examples/plot_results.py quantity \
  outputs/test_run/sequence.csv \
  --x mu \
  --y mass compactness_95 \
  --secondary-y \
  --output outputs/test_run/mass_compactness_vs_mu.png
```

The default axis labels use rendered mathematical notation where appropriate.
For example, `mass` is plotted as `M [M_Pl^2/m]`, `radius_95` as
`R_95 [m^{-1}]`, and `mu`, `omega`, `lambda_input`, and compactness quantities
are shown without a unit label.

A custom label can be supplied directly:

```bash
python examples/plot_results.py quantity \
  outputs/test_run/sequence.csv \
  --x mu \
  --y mass \
  --ylabel "M [M_Pl^2/m]" \
  --output outputs/test_run/mass_vs_mu.png
```

Or values can be rescaled for display:

```bash
python examples/plot_results.py quantity \
  outputs/test_run/sequence.csv \
  --x mu \
  --y mass \
  --y-scale 1.0 \
  --y-unit Msun \
  --output outputs/test_run/mass_vs_mu_physical_units.png
```

Plot numerical diagnostics:

```bash
python examples/plot_results.py quantity \
  outputs/test_run/sequence.csv \
  --x mu \
  --y residual_norm tail_fraction \
  --output outputs/test_run/diagnostics_vs_mu.png
```

Any numeric column in `sequence.csv` can be used.  Useful choices include:

```text
mass
radius_95
compactness_95
central_amplitude
omega
mu
residual_norm
tail_fraction
r_max
nodes
```

By default, rejected or failed rows are skipped when an `accepted` column is
available.  Add `--all-rows` to include every row in the file.

Full-GR real-field runs may also include diagnostics such as:

```text
C95_areal
M_metric
M_density
R95_areal
R95_iso
tail_phi1
phi3_over_phi1
max_bc_residual
```

Full-GR complex-field runs may also include:

```text
M_tilde_GR
R95_tilde_GR
R99_tilde_GR
C95_GR
Lambda_GR
```

For a combined full-GR complex `lambda_scan`, the mass-radius command
automatically separates the different `lambda_input` values into distinct
curves. Add `--mark-max` to mark the maximum-mass point of each curve.

## Field Or Profile Plot

Plot the profile closest to a selected `mu`:

```bash
python examples/plot_results.py profile \
  outputs/test_run \
  --mu 0.095 \
  --output outputs/test_run/profile_mu_0p095.png
```

Profile radial axes are labeled as `m^-1` by default.  Scalar profiles are
dimensionless; mass profiles are labeled in `M_Pl^2/m` in the legend.  If a
radial conversion factor is available:

```bash
python examples/plot_results.py profile \
  outputs/test_run \
  --mu 0.095 \
  --radius-scale 1.0 \
  --radius-unit km \
  --output outputs/test_run/profile_mu_0p095_km.png
```

Plot the profile closest to a selected central amplitude:

```bash
python examples/plot_results.py profile \
  outputs/test_run \
  --central-amplitude 0.265 \
  --output outputs/test_run/profile_amp_0p265.png
```

For full-GR real-field runs, the script reads `profiles.csv`.  For full-GR
complex-field runs, it reads `profiles.npz`.  For a complex-field run, omit
`--mu` and `--central-amplitude` to select the maximum-mass solution. Use
`--kind field` for the scalar profile, `--kind mass` for the enclosed-mass
profile in the same `mass = 8*pi*M_tilde_GR` convention as `sequence.csv`, or
`--kind both` for both quantities. For leading/first-order EFT and second-order
EFT runs, the script reconstructs the selected profile from `config.json` and
`sequence.csv` by re-solving that single point.

## Maximum Mass Versus Lambda

For a full-GR complex `lambda_scan`, use the root `lambda_summary.csv`, which
contains one accepted maximum-mass row per lambda:

```bash
python examples/plot_results.py lambda-summary \
  outputs/full_gr_complex_lambda_scan/lambda_summary.csv \
  --y mass \
  --output outputs/full_gr_complex_lambda_scan/max_mass_vs_lambda.png
```

The command also accepts the root `sequence.csv`; when several rows exist for
each lambda it automatically selects the accepted maximum-mass row of each
sequence. Using `lambda_summary.csv` is more direct for saved complex scans.

With a mass conversion factor:

```bash
python examples/plot_results.py lambda-summary \
  outputs/full_gr_complex_lambda_scan/lambda_summary.csv \
  --y mass \
  --y-scale 1.0 \
  --y-unit Msun \
  --output outputs/full_gr_complex_lambda_scan/max_mass_vs_lambda_physical_units.png
```

Radius at maximum mass versus lambda:

```bash
python examples/plot_results.py lambda-summary \
  outputs/full_gr_complex_lambda_scan/lambda_summary.csv \
  --y radius_95 \
  --output outputs/full_gr_complex_lambda_scan/radius_at_max_vs_lambda.png
```

If separate runs were made for different lambda values, pass all sequence files:

```bash
python examples/plot_results.py lambda-summary \
  outputs/lambda_m12/sequence.csv \
  outputs/lambda_m20/sequence.csv \
  outputs/lambda_m40/sequence.csv \
  --y mass \
  --output outputs/max_mass_vs_lambda.png
```

The script selects the maximum-mass accepted row from each sequence.

## Full-GR Complex Lambda-Scan Plot Set

A full complex-field lambda scan can generate the standard comparison figures
with one command:

```bash
python examples/plot_results.py complex-scan outputs/full_gr_complex_lambda_scan
```

This creates `outputs/full_gr_complex_lambda_scan/plots/` containing:

```text
mass_radius_by_lambda.png
max_mass_vs_lambda.png
radius_at_max_vs_lambda.png
lambda_m6_field_profile_max_mass.png
lambda_m6_mass_profile_max_mass.png
... one field and enclosed-mass profile pair for each accepted lambda
```

The mass-radius figure uses the full root `sequence.csv`, draws one curve per
lambda, and marks each maximum-mass point. The lambda figures use
`lambda_summary.csv`. The profile figures read the individually saved
`lambda_*/profiles.npz` data and select that lambda's maximum-mass solution.
Use `--no-profiles` if only the three summary figures are wanted.
