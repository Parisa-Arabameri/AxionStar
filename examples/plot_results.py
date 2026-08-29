#!/usr/bin/env python3
"""Plot results produced by ``run_axionstar.py``.

Typical use from a run directory:

    python /path/to/examples/plot_results.py mass-radius sequence.csv
    python /path/to/examples/plot_results.py quantity sequence.csv --x mu --y mass compactness_95
    python /path/to/examples/plot_results.py profile . --mu 0.095

The ``profile`` command reads saved profile files for full-GR runs.  For EFT
runs, profiles are reconstructed by re-solving the selected point from
``config.json`` and ``sequence.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_pyplot():
    """Import matplotlib only when a plot is requested."""

    import matplotlib.pyplot as plt

    return plt


def accepted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return accepted rows when the table contains an accepted column."""

    if "accepted" not in df.columns:
        return df.copy()
    accepted = df["accepted"]
    if accepted.dtype == bool:
        mask = accepted
    else:
        mask = accepted.astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return df[mask].copy()


def label_from_table(df: pd.DataFrame, path: Path) -> str:
    """Build a compact plot label from common output columns."""

    if df.empty:
        return path.parent.name
    parts = []
    for col in ("regime", "model", "lambda_input"):
        if col in df.columns:
            parts.append(f"{col}={df[col].iloc[0]}")
    return ", ".join(parts) if parts else path.parent.name


def read_sequence(path: str | Path) -> tuple[Path, pd.DataFrame]:
    """Read a sequence CSV and return its path and DataFrame."""

    csv_path = Path(path)
    return csv_path, pd.read_csv(csv_path)


DEFAULT_UNITS = {
    "lambda_input": "",
    "mu": "",
    "omega": "",
    "central_amplitude": "",
    "mass": "M_Pl^2/m",
    "radius_95": "m^-1",
    "compactness_95": "",
    "residual_norm": "",
    "tail": "",
    "tail_fraction": "",
    "r_max": "m^-1",
    "M_metric": "M_Pl^2/m",
    "M_density": "M_Pl^2/m",
    "M_tilde_GR": "M_Pl^2/m",
    "M_metric_profile": "M_Pl^2/m",
    "M_density_profile": "M_Pl^2/m",
    "mass_profile": "M_Pl^2/m",
    "R95_areal": "m^-1",
    "R95_iso": "m^-1",
    "R95_density_areal": "m^-1",
    "R95_tilde_GR": "m^-1",
    "R99_tilde_GR": "m^-1",
    "C95_areal": "",
    "C95_GR": "",
    "x_areal": "m^-1",
    "R_iso": "m^-1",
    "r": "m^-1",
    "f": "",
    "F": "",
    "phi": "",
    "phi1": "",
    "Phi": "",
    "Psi": "",
    "A": "",
    "B": "",
}


DISPLAY_NAMES = {
    "lambda_input": r"$\lambda$",
    "mu": r"$\mu$",
    "omega": r"$\omega$",
    "central_amplitude": r"central amplitude",
    "mass": r"$M$",
    "radius_95": r"$R_{95}$",
    "compactness_95": r"$C_{95}$",
    "r_max": r"$r_{\max}$",
    "M_metric": r"$M_{\rm metric}$",
    "M_density": r"$M_{\rm density}$",
    "M_tilde_GR": r"$\tilde{M}_{\rm GR}$",
    "M_metric_profile": r"$M_{\rm metric}$",
    "M_density_profile": r"$M_{\rm density}$",
    "mass_profile": r"$M$",
    "R95_areal": r"$R_{95}^{\rm areal}$",
    "R95_iso": r"$R_{95}^{\rm iso}$",
    "R95_density_areal": r"$R_{95,\rho}^{\rm areal}$",
    "R95_tilde_GR": r"$\tilde{R}_{95,\rm GR}$",
    "R99_tilde_GR": r"$\tilde{R}_{99,\rm GR}$",
    "C95_areal": r"$C_{95}^{\rm areal}$",
    "C95_GR": r"$C_{95,\rm GR}$",
    "x_areal": r"$x_{\rm areal}$",
    "R_iso": r"$R_{\rm iso}$",
    "r": r"$r$",
    "phi": r"$\phi$",
    "phi1": r"$\phi_1$",
    "Phi": r"$\Phi$",
    "Psi": r"$\Psi$",
}


UNIT_LABELS = {
    "M_Pl^2/m": r"$M_{\rm Pl}^2/m$",
    "m^-1": r"$m^{-1}$",
}


def axis_label(name: str, unit: str | None = None) -> str:
    """Return an axis label with a unit in brackets."""

    unit_text = unit if unit is not None else DEFAULT_UNITS.get(name, "")
    display_name = DISPLAY_NAMES.get(name, name)
    rendered_unit = UNIT_LABELS.get(unit_text, unit_text)
    return f"{display_name} [{rendered_unit}]" if rendered_unit else display_name


def scaled(series, factor: float):
    """Return values scaled for plotting."""

    return np.asarray(series, dtype=float) * factor


def sort_for_line_plot(df: pd.DataFrame, preferred_columns: tuple[str, ...]) -> pd.DataFrame:
    """Sort rows before drawing connected lines.

    Run outputs may contain a coarse scan followed by a refined scan.  Sorting
    prevents the plotting routine from connecting the last coarse point to the
    first refined point.
    """

    for col in preferred_columns:
        if col in df.columns:
            return df.sort_values(col).reset_index(drop=True)
    return df.reset_index(drop=True)


def closest_row(df: pd.DataFrame, *, mu: float | None = None, central_amplitude: float | None = None) -> pd.Series:
    """Select the row closest to the requested mu or central amplitude."""

    df = accepted_rows(df)
    if df.empty:
        raise ValueError("No accepted rows are available.")
    if mu is not None:
        if "mu" not in df.columns:
            raise ValueError("This sequence does not contain a mu column.")
        return df.loc[(df["mu"].astype(float) - mu).abs().idxmin()]
    if central_amplitude is not None:
        return df.loc[(df["central_amplitude"].astype(float) - central_amplitude).abs().idxmin()]
    return df.loc[df["mass"].astype(float).idxmax()]


def _mass_radius_groups(df: pd.DataFrame, path: Path):
    """Yield separate curves, including one curve per lambda in combined scans."""

    if "lambda_input" in df.columns:
        lambda_numeric = pd.to_numeric(df["lambda_input"], errors="coerce")
        finite_values = sorted(lambda_numeric[np.isfinite(lambda_numeric)].unique())
        if len(finite_values) > 1:
            for value in finite_values:
                group = df[np.isclose(lambda_numeric, value)].copy()
                yield group, rf"$\lambda={value:g}$"
            return
    yield df, label_from_table(df, path)


def plot_mass_radius(args: argparse.Namespace) -> None:
    """Plot mass-radius curves from one or more sequence files."""

    plt = get_pyplot()
    plt.figure(figsize=(7, 5))
    for item in args.csv:
        path, df = read_sequence(item)
        df = accepted_rows(df)
        if df.empty:
            continue
        for group, label in _mass_radius_groups(df, path):
            group = sort_for_line_plot(group, ("central_amplitude", "mu", "radius_95"))
            line = plt.plot(
                scaled(group["radius_95"], args.radius_scale),
                scaled(group["mass"], args.mass_scale),
                "o-",
                label=label,
            )[0]
            if getattr(args, "mark_max", False) and not group.empty:
                imax = group["mass"].astype(float).idxmax()
                plt.plot(
                    float(group.loc[imax, "radius_95"]) * args.radius_scale,
                    float(group.loc[imax, "mass"]) * args.mass_scale,
                    marker="*",
                    markersize=12,
                    linestyle="none",
                    color=line.get_color(),
                )
    plt.xlabel(args.xlabel or axis_label("radius_95", args.radius_unit))
    plt.ylabel(args.ylabel or axis_label("mass", args.mass_unit))
    plt.title(args.title or "Mass-radius sequence")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output}")


def plot_quantity(args: argparse.Namespace) -> None:
    """Plot selected sequence columns against a chosen x column."""

    plt = get_pyplot()
    path, df = read_sequence(args.csv)
    if not args.all_rows:
        df = accepted_rows(df)
    missing = [col for col in [args.x, *args.y] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = sort_for_line_plot(df, (args.x,))

    plt.figure(figsize=(7, 5))
    x_values = scaled(df[args.x], args.x_scale)
    if args.secondary_y and len(args.y) == 2:
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        line1 = ax1.plot(
            x_values,
            scaled(df[args.y[0]], args.y_scale),
            "o-",
            label=axis_label(args.y[0], args.y_unit),
        )
        line2 = ax2.plot(
            x_values,
            scaled(df[args.y[1]], args.y_scale),
            "s--",
            color="tab:orange",
            label=axis_label(args.y[1], args.y_unit),
        )
        ax1.set_xlabel(args.xlabel or axis_label(args.x, args.x_unit))
        ax1.set_ylabel(args.ylabel or axis_label(args.y[0], args.y_unit))
        ax2.set_ylabel(axis_label(args.y[1], args.y_unit))
        lines = line1 + line2
        ax1.legend(lines, [line.get_label() for line in lines], fontsize=8)
        ax1.grid(alpha=0.3)
        plt.title(args.title or f"{args.y[0]} and {args.y[1]} vs {args.x}")
        plt.tight_layout()
        plt.savefig(args.output, dpi=args.dpi)
        print(f"Saved {args.output}")
        return

    for y_col in args.y:
        plt.plot(x_values, scaled(df[y_col], args.y_scale), "o-", label=axis_label(y_col, args.y_unit))
    plt.xlabel(args.xlabel or axis_label(args.x, args.x_unit))
    if args.ylabel:
        plt.ylabel(args.ylabel)
    elif len(args.y) == 1:
        plt.ylabel(axis_label(args.y[0], args.y_unit))
    else:
        plt.ylabel(f"scaled value [{args.y_unit or 'see legend'}]")
    plt.title(args.title or f"{', '.join(args.y)} vs {args.x}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output}")


def max_rows_from_sequences(csv_paths: list[str]) -> pd.DataFrame:
    """Return one maximum-mass row per sequence file."""

    rows = []
    for item in csv_paths:
        path, df = read_sequence(item)
        df = accepted_rows(df)
        if df.empty:
            continue
        row = dict(df.loc[df["mass"].astype(float).idxmax()])
        row["source_file"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_lambda_summary(args: argparse.Namespace) -> None:
    """Plot maximum quantities against lambda_input.

    This works with either a lambda-scan summary file or several sequence files
    from separate single-lambda runs.
    """

    plt = get_pyplot()
    if len(args.csv) == 1:
        path, df = read_sequence(args.csv[0])
        if "lambda_input" not in df.columns or args.y not in df.columns:
            raise ValueError(f"{path} must contain lambda_input and {args.y}.")
        df = accepted_rows(df)
        if "mass" in df.columns:
            imax = df.groupby("lambda_input")["mass"].idxmax()
            df = df.loc[imax]
    else:
        df = max_rows_from_sequences(args.csv)

    if df.empty:
        raise ValueError("No accepted rows are available for plotting.")

    df = df.sort_values("lambda_input")
    plt.figure(figsize=(7, 5))
    plt.plot(
        scaled(df["lambda_input"], args.lambda_scale),
        scaled(df[args.y], args.y_scale),
        "o-",
    )
    plt.xlabel(args.xlabel or axis_label("lambda_input", args.lambda_unit))
    plt.ylabel(args.ylabel or axis_label(args.y, args.y_unit))
    plt.title(args.title or f"{args.y} at maximum mass vs lambda")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output}")


def plot_real_gr_profile(run_dir: Path, sequence: pd.DataFrame, args: argparse.Namespace) -> bool:
    """Plot a full-GR real-field profile from profiles.csv if available."""

    profile_path = run_dir / "profiles.csv"
    if not profile_path.exists():
        return False
    plt = get_pyplot()
    profiles = pd.read_csv(profile_path)
    row = closest_row(sequence, mu=args.mu, central_amplitude=args.central_amplitude)
    phi0 = float(row["central_amplitude"])
    selected = profiles.iloc[(profiles["phi10"].astype(float) - phi0).abs().argsort()]
    phi0_selected = float(selected["phi10"].iloc[0])
    selected = profiles[np.isclose(profiles["phi10"].astype(float), phi0_selected)]

    plt.figure(figsize=(7, 5))
    x = scaled(selected["x_areal"], args.radius_scale)
    plt.plot(x, scaled(selected["phi1"], args.profile_scale), label=axis_label("phi1", args.profile_unit))
    if "M_metric_profile" in selected.columns:
        plt.plot(
            x,
            scaled(selected["M_metric_profile"], args.profile_scale),
            label=axis_label("M_metric_profile", args.profile_unit),
        )
    plt.xlabel(axis_label("x_areal", args.radius_unit))
    plt.ylabel(axis_label("profile", args.profile_unit) if args.profile_unit else "profile value [see legend]")
    plt.title(args.title or f"Full-GR real profile, phi10={phi0_selected:.6g}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output}")
    return True


def plot_complex_gr_profile(run_dir: Path, sequence: pd.DataFrame, args: argparse.Namespace) -> bool:
    """Plot a full-GR complex-field profile from profiles.npz if available."""

    profile_path = run_dir / "profiles.npz"
    if not profile_path.exists():
        return False
    plt = get_pyplot()
    data = np.load(profile_path, allow_pickle=True)
    row = closest_row(sequence, mu=args.mu, central_amplitude=args.central_amplitude)
    phi0_values = data["phi0"].astype(float)
    idx = int(np.argmin(np.abs(phi0_values - float(row["central_amplitude"]))))

    r = np.asarray(data["r"][idx], dtype=float)
    phi = np.asarray(data["phi"][idx], dtype=float)
    if "mass_profile_comparison" in data.files:
        mass_profile = np.asarray(data["mass_profile_comparison"][idx], dtype=float)
        mass_label = axis_label("mass", args.profile_unit)
    else:
        mass_profile = 8.0 * np.pi * np.asarray(data["mass_profile"][idx], dtype=float)
        mass_label = axis_label("mass", args.profile_unit)

    kind = getattr(args, "kind", "both")
    plt.figure(figsize=(7, 5))
    if kind in {"field", "both"}:
        plt.plot(
            scaled(r, args.radius_scale),
            scaled(phi, args.profile_scale),
            label=axis_label("phi", args.profile_unit if kind == "field" else ""),
        )
    if kind in {"mass", "both"}:
        plt.plot(
            scaled(r, args.radius_scale),
            scaled(mass_profile, args.profile_scale),
            label=mass_label,
        )
    plt.xlabel(axis_label("r", args.radius_unit))
    if kind == "field":
        plt.ylabel(axis_label("phi", args.profile_unit))
        default_title = f"Scalar profile at maximum/selected solution, phi0={phi0_values[idx]:.6g}"
    elif kind == "mass":
        plt.ylabel(axis_label("mass", args.profile_unit))
        default_title = f"Enclosed mass profile, phi0={phi0_values[idx]:.6g}"
    else:
        plt.ylabel("profile value [see legend]")
        default_title = f"Full-GR complex profiles, phi0={phi0_values[idx]:.6g}"
    plt.title(args.title or default_title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output}")
    return True


def plot_eft_profile(run_dir: Path, sequence: pd.DataFrame, args: argparse.Namespace) -> bool:
    """Reconstruct and plot one EFT profile from config.json and sequence.csv."""

    plt = get_pyplot()
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return False
    config_data = json.loads(config_path.read_text())
    row = closest_row(sequence, mu=args.mu, central_amplitude=args.central_amplitude)
    mu = float(row["mu"])

    regime = str(row.get("regime", ""))
    if regime in {"effective_leading_order", "effective_first_order"}:
        from axionstar.effective.first_order import FirstOrderConfig, solve_one_mu as solve_first_order

        config = FirstOrderConfig(**config_data)
        seed = [float(row["alpha0"]), float(row["beta0"])] if {"alpha0", "beta0"}.issubset(row.index) else None
        result = solve_first_order(mu, config, seed=seed)
        if not result.get("success"):
            raise RuntimeError(f"Could not reconstruct first-order profile at mu={mu}.")
        sol = result["solution"]
        r = np.linspace(sol.t[0], sol.t[-1], args.n_profile)
        y = sol.sol(r)
        plt.figure(figsize=(7, 5))
        plt.plot(scaled(r, args.radius_scale), scaled(y[0], args.profile_scale), label=axis_label("f", args.profile_unit))
        plt.plot(scaled(r, args.radius_scale), scaled(y[2], args.profile_scale), label=axis_label("phi", args.profile_unit))
        plt.xlabel(axis_label("r", args.radius_unit))
        plt.ylabel(axis_label("profile", args.profile_unit))
        plt.title(args.title or f"EFT profile, mu={mu:.6g}")
    elif regime == "effective_second_order":
        from axionstar.effective.second_order import SecondOrderConfig, solve_one_mu as solve_second_order

        config = SecondOrderConfig(**config_data)
        seed_names = ["F0", "Phi0", "Psi0", "A0", "B0"]
        seed = [float(row[name]) for name in seed_names] if set(seed_names).issubset(row.index) else None
        result = solve_second_order(mu, config, seed=seed)
        if not result.get("success"):
            raise RuntimeError(f"Could not reconstruct second-order profile at mu={mu}.")
        sol = result["solution"]
        r = np.linspace(sol.t[0], sol.t[-1], args.n_profile)
        y = sol.sol(r)
        plt.figure(figsize=(7, 5))
        for idx, label in [(0, "F"), (2, "Phi"), (4, "Psi"), (6, "A"), (8, "B")]:
            plt.plot(
                scaled(r, args.radius_scale),
                scaled(y[idx], args.profile_scale),
                label=axis_label(label, args.profile_unit),
            )
        plt.xlabel(axis_label("r", args.radius_unit))
        plt.ylabel(axis_label("profile", args.profile_unit))
        plt.title(args.title or f"Second-order EFT profile, mu={mu:.6g}")
    else:
        return False

    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output}")
    return True


def plot_profile(args: argparse.Namespace) -> None:
    """Plot a field/profile from a run directory."""

    run_dir = Path(args.run_dir)
    sequence_path = run_dir / args.sequence
    if not sequence_path.exists():
        raise FileNotFoundError(f"Could not find {sequence_path}")
    sequence = pd.read_csv(sequence_path)

    if plot_real_gr_profile(run_dir, sequence, args):
        return
    if plot_complex_gr_profile(run_dir, sequence, args):
        return
    if plot_eft_profile(run_dir, sequence, args):
        return
    raise FileNotFoundError("No supported profile source found in the run directory.")


def plot_complex_lambda_scan(args: argparse.Namespace) -> None:
    """Create the standard comparison and profile plots for a saved complex-GR lambda scan."""

    run_dir = Path(args.run_dir)
    sequence_path = run_dir / "sequence.csv"
    summary_path = run_dir / "lambda_summary.csv"
    if not sequence_path.exists():
        raise FileNotFoundError(f"Could not find {sequence_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find {summary_path}")

    plot_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    mass_radius_args = argparse.Namespace(
        csv=[str(sequence_path)],
        output=str(plot_dir / "mass_radius_by_lambda.png"),
        title="Full-GR complex-field mass-radius sequences",
        xlabel=None,
        ylabel=None,
        radius_scale=1.0,
        mass_scale=1.0,
        radius_unit=None,
        mass_unit=None,
        mark_max=True,
        dpi=args.dpi,
    )
    plot_mass_radius(mass_radius_args)

    for y_col, filename, title in [
        ("mass", "max_mass_vs_lambda.png", "Maximum mass vs lambda"),
        ("radius_95", "radius_at_max_vs_lambda.png", "Radius at maximum mass vs lambda"),
    ]:
        summary_args = argparse.Namespace(
            csv=[str(summary_path)],
            y=y_col,
            output=str(plot_dir / filename),
            title=title,
            xlabel=None,
            ylabel=None,
            lambda_scale=1.0,
            y_scale=1.0,
            lambda_unit=None,
            y_unit=None,
            dpi=args.dpi,
        )
        plot_lambda_summary(summary_args)

    if args.no_profiles:
        return

    for lambda_dir in sorted(run_dir.glob("lambda_*")):
        if not lambda_dir.is_dir() or not (lambda_dir / "sequence.csv").exists() or not (lambda_dir / "profiles.npz").exists():
            continue
        sequence = pd.read_csv(lambda_dir / "sequence.csv")
        accepted = accepted_rows(sequence)
        if accepted.empty:
            continue
        lambda_value = float(accepted["lambda_input"].iloc[0]) if "lambda_input" in accepted.columns else np.nan
        safe = lambda_dir.name
        for kind, suffix in [("field", "field_profile_max_mass"), ("mass", "mass_profile_max_mass")]:
            profile_args = argparse.Namespace(
                mu=None,
                central_amplitude=None,
                radius_scale=1.0,
                profile_scale=1.0,
                radius_unit=None,
                profile_unit=None,
                title=(
                    rf"$\lambda={lambda_value:g}$: scalar profile at maximum mass"
                    if kind == "field"
                    else rf"$\lambda={lambda_value:g}$: enclosed mass profile at maximum mass"
                ),
                output=str(plot_dir / f"{safe}_{suffix}.png"),
                dpi=args.dpi,
                kind=kind,
            )
            plot_complex_gr_profile(lambda_dir, sequence, profile_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot axion-star solver outputs.")
    parser.add_argument("--dpi", type=int, default=200, help="Output figure resolution.")
    sub = parser.add_subparsers(dest="command", required=True)

    mass_radius = sub.add_parser("mass-radius", help="Plot mass versus radius_95.")
    mass_radius.add_argument("csv", nargs="+", help="One or more sequence.csv files.")
    mass_radius.add_argument("--output", default="mass_radius.png")
    mass_radius.add_argument("--title")
    mass_radius.add_argument("--xlabel")
    mass_radius.add_argument("--ylabel")
    mass_radius.add_argument("--radius-scale", type=float, default=1.0, help="Multiply radius_95 by this factor before plotting.")
    mass_radius.add_argument("--mass-scale", type=float, default=1.0, help="Multiply mass by this factor before plotting.")
    mass_radius.add_argument("--radius-unit", help="Unit label for the radius axis.")
    mass_radius.add_argument("--mass-unit", help="Unit label for the mass axis.")
    mass_radius.add_argument("--mark-max", action="store_true", help="Mark the maximum-mass point on each curve.")
    mass_radius.set_defaults(func=plot_mass_radius)

    quantity = sub.add_parser("quantity", help="Plot selected sequence columns.")
    quantity.add_argument("csv", help="A sequence.csv or summary CSV file.")
    quantity.add_argument("--x", default="mu", help="Column for the x axis.")
    quantity.add_argument("--y", nargs="+", required=True, help="One or more y-axis columns.")
    quantity.add_argument("--output", default="quantity_plot.png")
    quantity.add_argument("--title")
    quantity.add_argument("--xlabel")
    quantity.add_argument("--ylabel")
    quantity.add_argument("--x-scale", type=float, default=1.0, help="Multiply the x column by this factor before plotting.")
    quantity.add_argument("--y-scale", type=float, default=1.0, help="Multiply all y columns by this factor before plotting.")
    quantity.add_argument("--x-unit", help="Unit label for the x axis.")
    quantity.add_argument("--y-unit", help="Unit label for the y axis and legend.")
    quantity.add_argument("--all-rows", action="store_true", help="Include rejected/failed rows when present.")
    quantity.add_argument("--secondary-y", action="store_true", help="Plot two y columns on separate y axes.")
    quantity.set_defaults(func=plot_quantity)

    lam = sub.add_parser("lambda-summary", help="Plot maximum quantities versus lambda_input.")
    lam.add_argument("csv", nargs="+", help="One lambda-scan CSV or several sequence.csv files.")
    lam.add_argument("--y", default="mass", help="Quantity to plot, e.g. mass or radius_95.")
    lam.add_argument("--output", default="lambda_summary.png")
    lam.add_argument("--title")
    lam.add_argument("--xlabel")
    lam.add_argument("--ylabel")
    lam.add_argument("--lambda-scale", type=float, default=1.0, help="Multiply lambda_input by this factor before plotting.")
    lam.add_argument("--y-scale", type=float, default=1.0, help="Multiply the y column by this factor before plotting.")
    lam.add_argument("--lambda-unit", help="Unit label for the lambda axis.")
    lam.add_argument("--y-unit", help="Unit label for the y axis.")
    lam.set_defaults(func=plot_lambda_summary)

    profile = sub.add_parser("profile", help="Plot one field/profile from a run directory.")
    profile.add_argument("run_dir", help="Run directory containing sequence.csv.")
    profile.add_argument("--sequence", default="sequence.csv", help="Sequence filename inside run_dir.")
    profile.add_argument("--mu", type=float, help="Select the profile closest to this mu.")
    profile.add_argument("--central-amplitude", type=float, help="Select the profile closest to this central amplitude.")
    profile.add_argument("--n-profile", type=int, default=2000, help="Number of points for reconstructed EFT profiles.")
    profile.add_argument("--output", default="profile.png")
    profile.add_argument("--title")
    profile.add_argument("--radius-scale", type=float, default=1.0, help="Multiply radial coordinates by this factor before plotting.")
    profile.add_argument("--profile-scale", type=float, default=1.0, help="Multiply plotted profile values by this factor.")
    profile.add_argument("--radius-unit", help="Unit label for the radial axis.")
    profile.add_argument("--profile-unit", help="Unit label for profile values.")
    profile.add_argument(
        "--kind",
        choices=["both", "field", "mass"],
        default="both",
        help="For complex-GR profiles, plot both quantities, only the scalar field, or only enclosed mass.",
    )
    profile.set_defaults(func=plot_profile)

    complex_scan = sub.add_parser(
        "complex-scan",
        help="Create mass-radius, max-mass-vs-lambda, radius-vs-lambda, and maximum-mass profile plots.",
    )
    complex_scan.add_argument("run_dir", help="Full-GR complex lambda-scan output directory.")
    complex_scan.add_argument("--output-dir", help="Directory for figures; defaults to RUN_DIR/plots.")
    complex_scan.add_argument("--no-profiles", action="store_true", help="Skip per-lambda maximum-mass profile figures.")
    complex_scan.set_defaults(func=plot_complex_lambda_scan)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
