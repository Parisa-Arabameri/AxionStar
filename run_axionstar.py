#!/usr/bin/env python3
"""Guided command-line runner for the axion-star solvers.

This script is intentionally simple and interactive.  It asks which physical
regime to run, collects the minimal numerical inputs, and writes a consistent
set of output files:

    config.json
    sequence.csv
    max_summary.csv

Some regimes also write an ``attempt_log.csv`` or profile file.  The leading
columns of every ``sequence.csv`` are shared across all regimes, so one plotting
script can compare the different approximations without renaming columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from axionstar.common import ensure_output_dir, make_run_id, save_config, save_max_summary, save_results
from axionstar.effective.first_order import FirstOrderConfig, scan_mu as scan_first_order
from axionstar.effective.first_order import refine_around_maximum as refine_first_order
from axionstar.effective.second_order import SecondOrderConfig, scan_mu as scan_second_order
from axionstar.effective.second_order import refine_around_maximum as refine_second_order
from axionstar.full_gr.complex_field import ComplexGRConfig, build_sequence as build_complex_sequence
from axionstar.full_gr.complex_field import profiles_to_npz, scan_lambda_detailed as scan_complex_lambda_detailed
from axionstar.full_gr.real_field import RealGRConfig, build_sequence as build_real_sequence
from axionstar.full_gr.real_field import refined_phi0_grid_from_csv, refined_phi0_grid_from_dataframe
from axionstar.full_gr.real_field import warmup_phi0_values_from_csv, warmup_phi0_values_from_dataframe


def ask(prompt: str, default: str) -> str:
    """Prompt with a default value."""

    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default


def ask_float(prompt: str, default: float) -> float:
    return float(ask(prompt, str(default)))


def ask_int(prompt: str, default: int) -> int:
    return int(ask(prompt, str(default)))


def ask_bool(prompt: str, default: bool = False) -> bool:
    default_text = "y" if default else "n"
    value = ask(prompt + " (y/n)", default_text).lower()
    return value.startswith("y")


def save_standard_outputs(config, df: pd.DataFrame, output_dir: Path) -> None:
    """Save common files for every regime."""

    save_config(config, output_dir)
    save_results(df, output_dir, "sequence.csv")
    candidates = accepted_rows(df) if "accepted" in df else df
    if candidates.empty:
        pd.DataFrame().to_csv(output_dir / "max_summary.csv", index=False)
        return
    save_max_summary(candidates, output_dir)


def save_attempt_log(raw_results, output_dir: Path) -> None:
    """Save readable per-point diagnostics without profile arrays."""

    excluded = {"solution", "r", "gamma", "lambda_metric", "phi", "psi", "m_profile"}
    if isinstance(raw_results, pd.DataFrame):
        raw_results.to_csv(output_dir / "attempt_log.csv", index=False)
        return
    rows = [{k: v for k, v in row.items() if k not in excluded} for row in raw_results]
    pd.DataFrame(rows).to_csv(output_dir / "attempt_log.csv", index=False)


def ask_scan_mode(default: str = "coarse") -> str:
    """Ask how the sequence should be scanned."""

    print("Scan modes:")
    print("  1. coarse              Run only a broad coarse scan")
    print("  2. refine_existing     Refine using an existing coarse sequence.csv")
    print("  3. coarse_then_refine  Run a coarse scan, then refine it in this run")
    mode_choice = ask("scan mode", default).lower()
    if mode_choice in {"1", "coarse", "c"}:
        return "coarse"
    if mode_choice in {"2", "refine", "refine_existing", "existing", "r"}:
        return "refine_existing"
    if mode_choice in {"3", "coarse_then_refine", "both", "auto", "cr"}:
        return "coarse_then_refine"
    raise ValueError(f"Unknown scan mode: {mode_choice}")


def accepted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows accepted by the solver, if the table has an accepted flag."""

    if "accepted" not in df.columns:
        return df.copy()
    accepted = df["accepted"]
    if accepted.dtype == bool:
        mask = accepted
    else:
        mask = accepted.astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return df[mask].copy()


def refinement_source_from_sequence(csv_path: str | Path, parameter_col: str) -> pd.DataFrame:
    """Load and clean a saved sequence before using it for refinement."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run a coarse scan first, or choose scan mode "
            "'coarse_then_refine' to do both stages in one run."
        )

    df = accepted_rows(pd.read_csv(path))
    required = {"mass", parameter_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Cannot refine from {path}; missing columns: {', '.join(missing)}.")

    df = df[pd.to_numeric(df["mass"], errors="coerce").notna()]
    df = df[pd.to_numeric(df[parameter_col], errors="coerce").notna()]
    df = df.sort_values(parameter_col).drop_duplicates(parameter_col).reset_index(drop=True)
    if len(df) < 3:
        raise ValueError(f"Need at least three accepted points in {path} to refine.")
    return df


def ask_accuracy_mode() -> str:
    """Ask for numerical accuracy preset used by EFT shooting runs."""

    print("Accuracy modes:")
    print("  1. fast      Exploratory; looser tolerances and fewer retries")
    print("  2. standard  Balanced defaults")
    print("  3. accurate  Stricter tolerances for narrow final scans")
    choice = ask("accuracy mode", "standard").lower()
    if choice in {"1", "fast", "f"}:
        return "fast"
    if choice in {"2", "standard", "std", "s"}:
        return "standard"
    if choice in {"3", "accurate", "a"}:
        return "accurate"
    raise ValueError(f"Unknown accuracy mode: {choice}")


def apply_first_order_accuracy(config: FirstOrderConfig, mode: str) -> None:
    """Apply first-order EFT numerical presets in place."""

    config.accuracy_mode = mode
    if mode == "fast":
        config.rtol_ivp = 1.0e-8
        config.atol_ivp = 1.0e-9
        config.ftol_ls = 1.0e-8
        config.xtol_ls = 1.0e-8
        config.gtol_ls = 1.0e-8
        config.max_nfev_ls = 150
        config.residual_tolerance = 1.0e-4
        config.tail_tolerance = 1.0e-4
        config.tail_fraction_tolerance = 3.0e-3
        config.mass_grid_points = 2000
        config.max_attempts = 12
    elif mode == "accurate":
        config.rtol_ivp = 1.0e-11
        config.atol_ivp = 1.0e-11
        config.ftol_ls = 1.0e-12
        config.xtol_ls = 1.0e-12
        config.gtol_ls = 1.0e-12
        config.max_nfev_ls = 500
        config.residual_tolerance = 3.0e-7
        config.tail_tolerance = 3.0e-7
        config.tail_fraction_tolerance = 5.0e-4
        config.mass_grid_points = 8000
        config.max_attempts = 100


def apply_second_order_accuracy(config: SecondOrderConfig, mode: str) -> None:
    """Apply second-order EFT numerical presets in place."""

    config.accuracy_mode = mode
    if mode == "fast":
        config.rtol_ivp = 1.0e-8
        config.atol_ivp = 1.0e-9
        config.ftol_ls = 1.0e-8
        config.xtol_ls = 1.0e-8
        config.gtol_ls = 1.0e-8
        config.max_nfev_ls = 120
        config.residual_tolerance = 3.0e-3
        config.tail_tolerance = 1.0e-3
        config.tail_fraction_tolerance = 5.0e-3
        config.mass_grid_points = 1500
        config.max_attempts = 4
    elif mode == "accurate":
        config.rtol_ivp = 1.0e-11
        config.atol_ivp = 1.0e-11
        config.ftol_ls = 1.0e-12
        config.xtol_ls = 1.0e-12
        config.gtol_ls = 1.0e-12
        config.max_nfev_ls = 700
        config.residual_tolerance = 1.0e-5
        config.tail_tolerance = 5.0e-4
        config.tail_fraction_tolerance = 5.0e-4
        config.mass_grid_points = 8000
        config.max_attempts = 30


def run_first_order(output_dir: Path) -> None:
    print("\nLeading/first-order effective-theory options:")
    print("  1. leading            Nonrelativistic Schrodinger-Poisson limit")
    print("  2. first_order        First-order relativistic correction")
    print("For input definitions, see INPUT_PARAMETERS.md.")
    choice = ask("Choose effective-theory model", "leading")
    model = "relativistic" if choice in {"2", "first", "first_order", "rel", "relativistic"} else "nonrel"

    scan_mode = ask_scan_mode()
    accuracy_mode = ask_accuracy_mode()
    needs_coarse_grid = scan_mode != "refine_existing"
    fast_defaults = accuracy_mode == "fast"

    config = FirstOrderConfig(
        model=model,
        delta=ask_float("delta = lambda M_Pl^2 / m^2", -20.0),
        mu_min=ask_float("mu_min for coarse scan", 0.001) if needs_coarse_grid else 0.001,
        mu_max=ask_float("mu_max for coarse scan", 0.5 if model == "nonrel" else 0.15)
        if needs_coarse_grid
        else (0.5 if model == "nonrel" else 0.15),
        n_mu=ask_int("number of coarse mu points", 60 if model == "nonrel" else 75)
        if needs_coarse_grid
        else (60 if model == "nonrel" else 75),
        mu_spacing=ask("coarse mu spacing: geom or linear", "geom") if needs_coarse_grid else "geom",
        adaptive_r_max=ask_bool("adapt r_max using the scalar decay length", True),
        r_max_initial=ask_float("minimum starting r_max", 4.0),
        r_max_limit=ask_float("minimum r_max safety limit", 120.0 if fast_defaults else 250.0),
        r_max_initial_decay_lengths=ask_float("starting box in decay lengths", 5.0 if fast_defaults else 8.0),
        r_max_limit_decay_lengths=ask_float("safety limit in decay lengths", 15.0 if fast_defaults else 30.0),
        max_consecutive_failures=ask_int("stop after consecutive failed mu points (0 disables)", 3 if fast_defaults else 0),
        verbose=ask_bool("print progress", True),
    )
    apply_first_order_accuracy(config, accuracy_mode)
    if ask_bool("use custom initial shooting seed", False):
        config.alpha0_seed = ask_float("alpha0 seed", config.alpha0_seed)
        config.beta0_seed = ask_float("beta0 seed", config.beta0_seed)

    if scan_mode == "refine_existing":
        coarse_csv = ask("path to previous sequence.csv", "outputs/leading_first_coarse/sequence.csv")
        coarse_df = refinement_source_from_sequence(coarse_csv, "mu")
        save_config(config, output_dir)
        df, raw = refine_first_order(
            coarse_df,
            config,
            n_refined=ask_int("number of refined points", 30),
            width_factor=ask_float("refinement width factor", 1.5),
            autosave_dir=output_dir,
        )
    elif scan_mode == "coarse_then_refine":
        n_refined = ask_int("number of refined points", 30)
        width_factor = ask_float("refinement width factor", 1.5)
        save_config(config, output_dir)
        coarse_df, coarse_raw = scan_first_order(config, autosave_dir=output_dir)
        if coarse_df.empty:
            save_attempt_log(coarse_raw, output_dir)
            save_standard_outputs(config, coarse_df, output_dir)
            print("\nThe coarse scan produced no accepted sequence points; refinement was skipped.")
            return

        save_standard_outputs(config, coarse_df, output_dir)
        save_attempt_log(coarse_raw, output_dir)
        print(f"\nCoarse scan saved to: {output_dir.resolve()}")
        print("Starting refinement around the coarse maximum.\n")

        refined_df, refined_raw = refine_first_order(
            refinement_source_from_sequence(output_dir / "sequence.csv", "mu"),
            config,
            n_refined=n_refined,
            width_factor=width_factor,
            autosave_dir=output_dir,
        )
        df = pd.concat([coarse_df, refined_df], ignore_index=True)
        raw = coarse_raw + refined_raw
    else:
        save_config(config, output_dir)
        df, raw = scan_first_order(config, autosave_dir=output_dir)

    save_standard_outputs(config, df, output_dir)
    save_attempt_log(raw, output_dir)


def run_second_order(output_dir: Path) -> None:
    print("\nSecond-order EFT options:")
    print("For input definitions, see INPUT_PARAMETERS.md.")
    scan_mode = ask_scan_mode()
    accuracy_mode = ask_accuracy_mode()
    needs_coarse_grid = scan_mode != "refine_existing"
    fast_defaults = accuracy_mode == "fast"

    config = SecondOrderConfig(
        delta=ask_float("delta = lambda M_Pl^2 / m^2", -40.0),
        eta=ask_float("eta = m^2 / M_Pl^2", 1.0),
        mu_min=ask_float("mu_min for coarse scan", 0.01) if needs_coarse_grid else 0.01,
        mu_max=ask_float("mu_max for coarse scan", 0.05) if needs_coarse_grid else 0.05,
        n_mu=ask_int("number of coarse mu points", 15) if needs_coarse_grid else 15,
        adaptive_r_max=ask_bool("adapt r_max using the scalar decay length", True),
        r_max_initial=ask_float("minimum starting r_max", 8.0),
        r_max_limit=ask_float("minimum r_max safety limit", 80.0 if fast_defaults else 120.0),
        r_max_initial_decay_lengths=ask_float("starting box in decay lengths", 5.0 if fast_defaults else 8.0),
        r_max_limit_decay_lengths=ask_float("safety limit in decay lengths", 15.0 if fast_defaults else 30.0),
        max_consecutive_failures=ask_int("stop after consecutive failed mu points (0 disables)", 2 if fast_defaults else 0),
        verbose=ask_bool("print progress", True),
    )
    apply_second_order_accuracy(config, accuracy_mode)
    if ask_bool("use custom initial shooting seed", False):
        config.F0_seed = ask_float("F0 seed", config.F0_seed)
        config.Phi0_seed = ask_float("Phi0 seed", config.Phi0_seed)
        config.Psi0_seed = ask_float("Psi0 seed", config.Psi0_seed)
        config.A0_seed = ask_float("A0 seed", config.A0_seed)
        config.B0_seed = ask_float("B0 seed", config.B0_seed)

    if scan_mode == "refine_existing":
        coarse_csv = ask("path to previous sequence.csv", "outputs/second_order_coarse/sequence.csv")
        coarse_df = refinement_source_from_sequence(coarse_csv, "mu")
        save_config(config, output_dir)
        df, raw = refine_second_order(
            coarse_df,
            config,
            n_refined=ask_int("number of refined points", 30),
            width_factor=ask_float("refinement width factor", 1.5),
            autosave_dir=output_dir,
        )
    elif scan_mode == "coarse_then_refine":
        n_refined = ask_int("number of refined points", 30)
        width_factor = ask_float("refinement width factor", 1.5)
        save_config(config, output_dir)
        coarse_df, coarse_raw = scan_second_order(config, autosave_dir=output_dir)
        if coarse_df.empty:
            save_attempt_log(coarse_raw, output_dir)
            save_standard_outputs(config, coarse_df, output_dir)
            print("\nThe coarse scan produced no accepted sequence points; refinement was skipped.")
            return

        save_standard_outputs(config, coarse_df, output_dir)
        save_attempt_log(coarse_raw, output_dir)
        print(f"\nCoarse scan saved to: {output_dir.resolve()}")
        print("Starting refinement around the coarse maximum.\n")

        refined_df, refined_raw = refine_second_order(
            refinement_source_from_sequence(output_dir / "sequence.csv", "mu"),
            config,
            n_refined=n_refined,
            width_factor=width_factor,
            autosave_dir=output_dir,
        )
        df = pd.concat([coarse_df, refined_df], ignore_index=True)
        raw = coarse_raw + refined_raw
    else:
        save_config(config, output_dir)
        df, raw = scan_second_order(config, autosave_dir=output_dir)

    save_standard_outputs(config, df, output_dir)
    save_attempt_log(raw, output_dir)


def parse_lambda_values(text: str) -> list[float]:
    """Parse a comma-separated lambda list while preserving order."""

    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("lambda_scan requires at least one lambda value.")
    return values


def save_complex_lambda_scan(
    lambda_values: list[float],
    base: ComplexGRConfig,
    output_dir: Path,
) -> None:
    """Run and save a full complex-GR lambda scan without discarding sequences."""

    combined, summary, runs = scan_complex_lambda_detailed(lambda_values, base)

    scan_config = {
        **base.__dict__,
        "run_mode": "lambda_scan",
        "lambda_values": [float(value) for value in lambda_values],
        "lambda_conv": None,
        "Lambda_GR": None,
    }
    save_config(scan_config, output_dir)
    save_results(combined, output_dir, "sequence.csv")
    save_results(summary, output_dir, "lambda_summary.csv")
    # For a lambda scan, max_summary.csv intentionally contains one maximum
    # row per lambda rather than a single global maximum.
    save_results(summary, output_dir, "max_summary.csv")

    for config, df, raw in runs:
        lambda_dir = ensure_output_dir(output_dir / make_run_id("lambda", config.lambda_conv))
        save_standard_outputs(config, df, lambda_dir)
        save_attempt_log(raw, lambda_dir)
        profiles_to_npz(raw, lambda_dir / "profiles.npz")

        accepted = accepted_rows(df)
        if not accepted.empty:
            imax = accepted["mass"].astype(float).idxmax()
            phi0_max_mass = float(accepted.loc[imax, "central_amplitude"])
            profiles_to_npz(
                raw,
                lambda_dir / "max_mass_profile.npz",
                selected_phi0=phi0_max_mass,
            )

    print("\nLambda-scan outputs:")
    print("  sequence.csv        all sequence points for all lambda values")
    print("  lambda_summary.csv  one maximum-mass row per lambda")
    print("  max_summary.csv     same per-lambda maximum summary")
    print("  lambda_*/            complete individual runs with profiles")


def run_full_gr_complex(output_dir: Path) -> None:
    print("\nFull-GR complex-field options:")
    print("For input definitions, see INPUT_PARAMETERS.md.")
    mode = ask("Run single sequence or lambda_scan", "single").strip().lower()
    if mode in {"2", "scan", "lambda", "lambda_scan"}:
        mode = "lambda_scan"
    elif mode in {"1", "single", "s"}:
        mode = "single"
    else:
        raise ValueError(f"Unknown full-GR complex run mode: {mode}")

    potential_type = ask("potential: free, quartic, liouville, log", "quartic")
    if mode == "lambda_scan":
        values = ask("comma-separated lambda values", "-12,-20,-40")
        lambda_values = parse_lambda_values(values)
        lambda_conv = lambda_values[0]  # Initial value; each scan run substitutes its requested coupling.
    else:
        lambda_values = []
        lambda_conv = ask_float("lambda in comparison convention (lambda = 2 Lambda_GR)", -12.0)

    base = ComplexGRConfig(
        potential_type=potential_type,
        lambda_conv=lambda_conv,
        phi0_min=ask_float("phi0_min", 0.03),
        phi0_max=ask_float("phi0_max", 0.35),
        n_phi0=ask_int("number of phi0 points", 35),
        r_max=ask_float("r_max", 100.0),
        n_mesh=ask_int("n_mesh", 700),
        tol=ask_float("BVP tolerance", 1.0e-5),
        max_nodes=ask_int("max_nodes", 50000),
        verbose=ask_bool("print progress", True),
    )

    if mode == "lambda_scan":
        save_complex_lambda_scan(lambda_values, base, output_dir)
        return

    df, raw = build_complex_sequence(base)
    save_standard_outputs(base, df, output_dir)
    save_attempt_log(raw, output_dir)
    profiles_to_npz(raw, output_dir / "profiles.npz")
    accepted = accepted_rows(df)
    if not accepted.empty:
        imax = accepted["mass"].astype(float).idxmax()
        profiles_to_npz(
            raw,
            output_dir / "max_mass_profile.npz",
            selected_phi0=float(accepted.loc[imax, "central_amplitude"]),
        )


def run_full_gr_real(output_dir: Path) -> None:
    print("\nFull-GR real-field options:")
    print("For input definitions, see INPUT_PARAMETERS.md.")
    scan_mode = ask_scan_mode()
    needs_coarse_grid = scan_mode != "refine_existing"

    config = RealGRConfig(
        lambda_tilde=ask_float("lambda_tilde in real-field potential", -4.0),
        phi0_min=ask_float("phi0_min for coarse scan", 0.02) if needs_coarse_grid else 0.02,
        phi0_max=ask_float("phi0_max for coarse scan", 0.40) if needs_coarse_grid else 0.40,
        n_phi0=ask_int("number of coarse phi0 points", 45) if needs_coarse_grid else 45,
        max_metric_harmonic=ask_int("maximum metric harmonic", 4),
        n_mesh=ask_int("n_mesh", 450),
        n_theta=ask_int("n_theta", 128),
        x_max=ask_float("x_max", 250.0),
        tol=ask_float("BVP tolerance", 1.0e-4),
        max_nodes=ask_int("max_nodes", 50000),
        verbose=ask_bool("print progress", True),
    )

    if scan_mode == "refine_existing":
        coarse_csv = ask("path to previous sequence.csv", "outputs/full_gr_real_coarse/sequence.csv")
        coarse_path = Path(coarse_csv)
        if not coarse_path.exists():
            raise FileNotFoundError(
                f"Could not find {coarse_path}. Run a coarse scan first, or choose scan mode "
                "'coarse_then_refine' to do both stages in one run."
            )
        phi0_values = refined_phi0_grid_from_csv(
            coarse_path,
            n_refined=ask_int("number of refined points", 121),
            width_factor=ask_float("refinement width factor", 1.5),
        )
        warmup_phi0 = warmup_phi0_values_from_csv(coarse_path, phi0_values)
        df, raw_df, profiles_df = build_real_sequence(
            config,
            phi0_values=phi0_values,
            is_refined=True,
            warmup_phi0_values=warmup_phi0,
        )
    elif scan_mode == "coarse_then_refine":
        n_refined = ask_int("number of refined points", 121)
        width_factor = ask_float("refinement width factor", 1.5)
        coarse_df, coarse_raw_df, coarse_profiles_df = build_real_sequence(config, is_refined=False)
        if coarse_df.empty:
            save_attempt_log(coarse_raw_df, output_dir)
            raise RuntimeError("The coarse scan produced no accepted sequence points; refinement cannot start.")

        save_standard_outputs(config, coarse_df, output_dir)
        save_attempt_log(coarse_raw_df, output_dir)
        coarse_profiles_df.to_csv(output_dir / "profiles.csv", index=False)
        print(f"\nCoarse scan saved to: {output_dir.resolve()}")
        print("Starting refinement around the coarse maximum.\n")

        phi0_values = refined_phi0_grid_from_dataframe(
            coarse_df,
            n_refined=n_refined,
            width_factor=width_factor,
        )
        warmup_phi0 = warmup_phi0_values_from_dataframe(coarse_df, phi0_values)
        refined_df, refined_raw_df, refined_profiles_df = build_real_sequence(
            config,
            phi0_values=phi0_values,
            is_refined=True,
            warmup_phi0_values=warmup_phi0,
        )
        df = pd.concat([coarse_df, refined_df], ignore_index=True)
        raw_df = pd.concat([coarse_raw_df, refined_raw_df], ignore_index=True)
        profiles_df = pd.concat([coarse_profiles_df, refined_profiles_df], ignore_index=True)
    else:
        df, raw_df, profiles_df = build_real_sequence(config, is_refined=False)

    save_standard_outputs(config, df, output_dir)
    save_attempt_log(raw_df, output_dir)
    profiles_df.to_csv(output_dir / "profiles.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run axion-star equilibrium solvers.")
    parser.add_argument("--output", default="outputs/run", help="Directory for config/results.")
    args = parser.parse_args(argv)

    output_dir = ensure_output_dir(args.output)
    print("\nChoose a regime:")
    print("  1. leading_first    EFT: leading SP limit or first-order correction")
    print("  2. second_order     EFT: second-order correction")
    print("  3. full_gr_real     Real-field KGE oscillaton")
    print("  4. full_gr_complex  Complex-field boson star")
    regime = ask("Regime", "leading_first")

    if regime in {"1", "leading", "leading_first", "first", "first_order"}:
        run_first_order(output_dir)
    elif regime in {"2", "second", "second_order"}:
        run_second_order(output_dir)
    elif regime in {"3", "real", "full_gr_real"}:
        run_full_gr_real(output_dir)
    elif regime in {"4", "complex", "full_gr_complex"}:
        run_full_gr_complex(output_dir)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    print(f"\nDone. Results written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
