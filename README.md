# Axion Star Solvers

Numerical solvers for equilibrium sequences of self-gravitating scalar-field
configurations.  The package includes:

1. **Leading/first-order effective theory**
   - `model="nonrel"`: nonrelativistic Schrodinger-Poisson limit.
   - `model="relativistic"`: first-order relativistic EFT correction.
2. **Second-order EFT**
   - Second-order relativistic EFT correction.
3. **Full-GR real field**
   - Real-field oscillaton solver using a Fourier expansion in time.
4. **Full-GR complex field**
   - Stationary boson-star solver for free, quartic, Liouville, and logarithmic
     scalar potentials.

All solvers write a `sequence.csv` table with the same leading columns, so
mass-radius curves and maximum-mass summaries can be compared across regimes.


## Authors

The software is developed by:

- Parisa Arabameri
- Francisco Colipí-Marchant


## Citation

If this software is used in research, please cite the associated manuscript:

> Parisa Arabameri, Paola Arias, Francisco Colipí-Marchant, and Enrico D. Schiappacasse,
> "A Unified Numerical Study of Axion Stars from the Nonrelativistic Regime to
> General Relativity."

The manuscript does not yet have an arXiv or DOI identifier. Citation metadata
will be updated when a public identifier is available. Machine-readable citation
information is provided in [`CITATION.cff`](CITATION.cff).

Source code repository: `https://github.com/Parisa-Arabameri/AxionStar`


## References

The equations and conventions implemented here are based on:

- Borna Salehian, Hong-Yi Zhang, Mustafa A. Amin, David I. Kaiser, Mohammad
  Hossein Namjoo, "Beyond Schrodinger-Poisson: Nonrelativistic Effective Field
  Theory for Scalar Dark Matter", arXiv:2104.10128.
- Miguel Alcubierre, Ricardo Becerril, F. Siddhartha Guzman, Tonatiuh Matos,
  Dario Nunez, L. Arturo Urena-Lopez, "Numerical studies of Phi^2-Oscillatons",
  arXiv:gr-qc/0301105.
- Gongjun Choi, Hong-Jian He, Enrico D. Schiappacasse, "Probing Dynamics of
  Boson Stars by Fast Radio Bursts and Gravitational Wave Detection",
  arXiv:1906.02094.


## Installation

Clone or download the repository, then install the package from the repository root:

```bash
cd AxionStar
python -m pip install -e .
```

The runtime dependencies are declared in `pyproject.toml` and also listed in
`requirements.txt`. They can be installed directly with:

```bash
python -m pip install -r requirements.txt
```

## Quick Start

Run the guided command-line interface:

```bash
python run_axionstar.py --output outputs/test_run
```

Press Enter at a prompt to accept the value shown in brackets.  For example,
`mu_min [0.001]:` uses `0.001` if no value is typed.

For leading/first-order EFT, second-order EFT, and full-GR real-field runs,
the scan mode can be:

```text
coarse
refine_existing
coarse_then_refine
```

Use `coarse_then_refine` to run a broad scan and then refine around the
maximum found in that scan. Use `refine_existing` when a previous run has
already produced the coarse `sequence.csv`.

For the leading/first-order and second-order EFT solvers, the runner also asks
for an accuracy mode:

```text
fast
standard
accurate
```

Use `fast` for exploratory scans and difficult couplings, then rerun a narrow
window with `standard` or `accurate` before using values in final figures or
tables. EFT runs update `sequence.csv` and `attempt_log.csv` after each
completed `mu` point, so partial results are preserved if a long scan is
stopped.

For a complete explanation of all interactive inputs, see
[`INPUT_PARAMETERS.md`](INPUT_PARAMETERS.md).
For common convergence and scan-quality issues, see
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
For plotting commands and examples, see [`PLOTTING.md`](PLOTTING.md).

Each run writes:

```text
config.json
sequence.csv
max_summary.csv
```

Some regimes also write diagnostic files such as:

```text
attempt_log.csv
profiles.csv
profiles.npz
max_mass_profile.npz
lambda_summary.csv
```

For a full-GR complex `lambda_scan`, the root `sequence.csv` contains all
sequence points from all requested lambda values. `lambda_summary.csv` (and the
root `max_summary.csv`) contains one maximum-mass row per lambda, and every
lambda is also saved as a complete `lambda_*` subdirectory.

## Output Files

`config.json` records the full set of numerical and physical inputs used for a
run.

`sequence.csv` contains one row per accepted or attempted sequence point.  The
leading columns have the same meaning in all regimes:

| Column | Meaning |
|---|---|
| `run_id` | Compact label for the run. |
| `regime` | Approximation regime, for example `effective_first_order` or `full_gr`. |
| `model` | Specific model inside the regime. |
| `field_type` | Real scalar, complex scalar, or real scalar envelope. |
| `potential` | Scalar potential used by the solver. |
| `self_interaction` | Main self-interaction parameter in the solver convention. |
| `lambda_input` | Input coupling value before any convention conversion. |
| `lambda_convention` | Text description of the coupling convention. |
| `mu` | Binding parameter, or `1 - omega` in the full-GR frequency convention. |
| `omega` | Oscillation or eigenfrequency when defined. |
| `central_amplitude` | Central scalar/envelope amplitude. |
| `mass` | Dimensionless mass in the package comparison convention. |
| `radius_95` | Radius enclosing 95 percent of the mass. |
| `compactness_95` | Compactness estimate based on `mass` and `radius_95`. |
| `radius_type` | Radius convention, for example `dimensionless_R95` or `areal_R95`. |
| `success` | Whether the numerical solve converged. |
| `accepted` | Whether validation checks accepted the point for analysis. |
| `is_refined` | Whether the point belongs to an optional refinement scan. |
| `max_at_edge` | Whether the maximum mass lies at the boundary of the scan. |


Additional diagnostic columns are appended after these common columns.
For second-order EFT scans, `sequence.csv` contains only points that pass the
secondary physical, EFT-hierarchy, and residual-order checks. Numerically
usable rejected points remain in `attempt_log.csv` because they may still have
been used to continue the shooting seed toward the physical branch.

`max_summary.csv` stores the finite-mass row with the largest `mass` value in
the sequence.  If `max_at_edge=True`, widen the scan before interpreting that
row as a physical maximum.

## Regimes

### First-order EFT

Select `first_order` in the guided runner.  The self-interaction parameter is

```text
delta = lambda M_Pl^2 / m^2
```

Use `model=nonrel` for the Schrodinger-Poisson limit and
`model=relativistic` for the leading relativistic correction.

The radial outer boundary is adaptive by default.  The scalar tail behaves
approximately as

```text
f(r) ~ exp(-Omega r) / r,
Omega^2 = 2 mu - mu^2.
```

Small `mu` points have longer tails, so the solver enlarges the radial box using
the decay length `1/Omega`.  The settings `r_max_initial` and `r_max_limit` are
lower bounds, while `r_max_initial_decay_lengths` and
`r_max_limit_decay_lengths` set the adaptive domain size.

### Second-order EFT

Select `second_order`.  The solver evolves

```text
F, Phi(Shifted Potential), Psi, A, B
```

with the shifted-potential boundary condition

```text
Phi(infinity) = mu + mu^2/2 + mu^3/3.
```

The parameter

```text
eta = m^2 / M_Pl^2
```

is explicit in the second-order equations and mass-density expression.  The
radial box is adaptive in the same way as the first-order solver, using the
second-order decay rate computed from `Phi(infinity)`.

### Full-GR Real Field

Select `full_gr_real`.  This solver computes real-field oscillaton sequences.
The scalar field is expanded in odd cosine harmonics of the fundamental
frequency, while the metric functions are expanded in even harmonics.

For a broad scan, run without CSV refinement.  For a refined scan, first produce
a coarse `sequence.csv`, then run the solver again and provide that CSV path
when prompted.  The refined run builds a central-amplitude grid around the
previous maximum.

### Full-GR Complex Field

Select `full_gr_complex`.  The scalar field is stationary, so the Einstein-Klein
Gordon system reduces to a radial boundary-value problem for the metric
functions and scalar profile.

Choose `single` to compute one coupling or `lambda_scan` to repeat the complete
central-amplitude sequence for several coupling values. In `lambda_scan` mode,
the coupling values are supplied through a single comma-separated list. For
example:

```text
Run single sequence or lambda_scan [single]: lambda_scan
potential: free, quartic, liouville, log [quartic]:
comma-separated lambda values [-12,-20,-40]: -6,-8,-10,-12,-14
```

The package writes the comparison convention

```text
lambda = 2 Lambda_GR
mass = 8 pi M_tilde_GR
mu = 1 - omega_GR
radius_95 = R95_tilde_GR
```

to the standard output columns.

A `lambda_scan` keeps both the complete combined sequence and the individual
sequences. The root output contains `sequence.csv` with all lambda values and
`lambda_summary.csv` with the maximum-mass row for each lambda. Each
`lambda_*` subdirectory contains its own `sequence.csv`, `max_summary.csv`,
`profiles.npz`, and `max_mass_profile.npz` (when an accepted solution exists).
This makes it possible to plot the full mass-radius curve and the field and
enclosed-mass profile at maximum mass separately for every lambda.

The standard scan plot set is generated with:

```bash
python examples/plot_results.py complex-scan outputs/full_gr_complex_lambda_scan
```

It produces the mass-radius curves by lambda, maximum mass versus lambda,
radius at maximum mass versus lambda, and maximum-mass field and enclosed-mass
profiles for each lambda. See [`PLOTTING.md`](PLOTTING.md) for individual plot
commands and options.

## Comparing Sequences

After producing multiple `sequence.csv` files:

```bash
python examples/compare_sequences.py \
  outputs/leading_first/sequence.csv \
  outputs/second_order/sequence.csv \
  outputs/full_gr_real/sequence.csv \
  --output comparison_mass_radius.png
```

Additional plotting commands are documented in [`PLOTTING.md`](PLOTTING.md).


