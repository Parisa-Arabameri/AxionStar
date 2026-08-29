"""Leading and first-order EFT axion-star solvers.

This module contains two closely related radial shooting problems:

``model="nonrel"``
    The leading Schrodinger-Poisson system with attractive quartic
    self-interaction.

``model="relativistic"``
    The same system including the leading relativistic EFT corrections used
    in Salehian et al.  The variable names follow the standard radial-system
    notation: ``f`` is the scalar envelope, ``phi`` is the Newtonian-like
    potential, ``mu`` is the dimensionless binding parameter, and ``delta`` is
    the dimensionless self-interaction.

The public functions are ``solve_one_mu``, ``scan_mu``, and
``refine_around_maximum``.  They return dictionaries/DataFrames with the
standard columns defined in ``axionstar.common``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import least_squares

from axionstar.common import ensure_output_dir, make_run_id, mark_edge_max, ordered_dataframe


@dataclass
class FirstOrderConfig:
    """Configuration for the leading/first-order EFT sequence.

    Parameters are dimensionless.  Numerical controls are explicit so that a
    published run can be reproduced from the saved ``config.json``.
    """

    model: str = "nonrel"  # "nonrel" or "relativistic"
    accuracy_mode: str = "standard"
    delta: float = -20.0
    mu_min: float = 0.001
    mu_max: float = 0.5
    n_mu: int = 60
    mu_spacing: str = "geom"  # "linear" or "geom"
    alpha0_seed: float = 0.267206211014205
    beta0_seed: float = -0.177148421028479
    r_min: float = 1.0e-5
    # The outer boundary is a numerical domain choice, not a physical input.
    # If adaptive_r_max is True, the code estimates the scalar decay length
    # 1/Omega and uses r_max_initial and r_max_limit only as lower bounds.
    r_max_initial: float = 4.0
    r_max_limit: float = 250.0
    adaptive_r_max: bool = True
    r_max_initial_decay_lengths: float = 8.0
    r_max_limit_decay_lengths: float = 30.0
    r_max_factor: float = 1.4
    max_attempts: int = 80
    seed_growth: float = 1.03
    rtol_ivp: float = 1.0e-10
    atol_ivp: float = 1.0e-10
    ftol_ls: float = 1.0e-11
    xtol_ls: float = 1.0e-11
    gtol_ls: float = 1.0e-11
    max_nfev_ls: int = 300
    residual_tolerance: float = 1.0e-6
    tail_tolerance: float = 1.0e-6
    tail_fraction_tolerance: float = 1.0e-3
    tail_check_fraction: float = 0.10
    mass_grid_points: int = 5000
    max_consecutive_failures: int = 0
    verbose: bool = True


def mu_grid(config: FirstOrderConfig) -> np.ndarray:
    """Build the scan grid for ``mu``."""

    if config.mu_spacing == "geom":
        return np.geomspace(config.mu_min, config.mu_max, config.n_mu)
    if config.mu_spacing == "linear":
        return np.linspace(config.mu_min, config.mu_max, config.n_mu)
    raise ValueError("mu_spacing must be 'geom' or 'linear'.")


def decay_rate_from_mu(mu: float) -> float:
    """Return the asymptotic scalar decay rate Omega.

    For the leading and first-order systems the tail behaves approximately as

        f(r) ~ exp(-Omega r) / r,

    with Omega^2 = 2 mu - mu^2.  Small mu gives a long tail, so the numerical
    domain must grow automatically in the dilute region.
    """

    omega2 = 2.0 * mu - mu**2
    return np.sqrt(omega2) if omega2 > 0 else np.nan


def radial_domain_for_mu(mu: float, config: FirstOrderConfig) -> tuple[float, float]:
    """Choose initial and maximum radial domains for one mu value.

    The configured ``r_max_initial`` and ``r_max_limit`` are safety lower
    bounds.  In adaptive mode they are enlarged according to the decay length
    1/Omega, preventing dilute solutions from being rejected just because the
    box was too small.
    """

    if not config.adaptive_r_max:
        return config.r_max_initial, config.r_max_limit

    omega = decay_rate_from_mu(mu)
    if not np.isfinite(omega) or omega <= 0:
        return config.r_max_initial, config.r_max_limit

    initial = max(config.r_max_initial, config.r_max_initial_decay_lengths / omega)
    limit = max(config.r_max_limit, config.r_max_limit_decay_lengths / omega)
    return initial, limit


def equations_nonrel(r: float, y: np.ndarray, mu: float, delta: float) -> list[float]:
    """Leading Schrodinger-Poisson radial system.

    y = [f, f', phi, phi'].
    """

    f, df, phi, dphi = y
    rr = max(r, 1.0e-30)
    d2f = 2.0 * (mu + phi) * f + (delta / 4.0) * f**3 - 2.0 * df / rr
    d2phi = 0.5 * f**2 - 2.0 * dphi / rr
    return [df, d2f, dphi, d2phi]


def equations_relativistic(r: float, y: np.ndarray, mu: float, delta: float) -> list[float]:
    """First-order relativistic-correction EFT radial system.

    This is the dimensionless radial system for the leading EFT correction.
    """

    f, df, phi, dphi = y
    rr = max(r, 1.0e-30)
    d2f = 2.0 * (
        mu * f
        - 0.5 * mu**2 * f
        - (3.0 / 16.0) * f**3
        + (delta / 8.0) * f**3
        + (delta * mu / 8.0) * f**3
        + (delta**2 / 768.0) * f**5
        + f * phi
        - 4.0 * mu * f * phi
        - 0.25 * delta * f**3 * phi
        - 3.0 * f * phi**2
        - df / rr
    )
    d2phi = (
        0.5 * f**2
        - 1.5 * mu * f**2
        - (delta / 16.0) * f**4
        - 3.0 * f**2 * phi
        - 2.0 * dphi / rr
    )
    return [df, d2f, dphi, d2phi]


def equations_for_model(model: str):
    """Return the RHS function for the requested effective-theory model."""

    if model == "nonrel":
        return equations_nonrel
    if model == "relativistic":
        return equations_relativistic
    raise ValueError("model must be 'nonrel' or 'relativistic'.")


def initial_conditions_nonrel(
    alpha0: float,
    beta0: float,
    mu: float,
    delta: float,
    radius: float,
) -> list[float]:
    """Near-origin Taylor expansion for the nonrelativistic system."""

    alpha2 = (1.0 / 3.0) * alpha0 * (mu + beta0 + (delta / 8.0) * alpha0**2)
    beta2 = (1.0 / 12.0) * alpha0**2
    alpha4 = (1.0 / 10.0) * (
        alpha2 * (mu + beta0 + (3.0 * delta / 8.0) * alpha0**2)
        + alpha0 * beta2
    )
    beta4 = (1.0 / 20.0) * alpha0 * alpha2

    return [
        alpha0 + alpha2 * radius**2 + alpha4 * radius**4,
        2.0 * alpha2 * radius + 4.0 * alpha4 * radius**3,
        beta0 + beta2 * radius**2 + beta4 * radius**4,
        2.0 * beta2 * radius + 4.0 * beta4 * radius**3,
    ]


def initial_conditions_relativistic(
    alpha0: float,
    beta0: float,
    mu: float,
    delta: float,
    radius: float,
) -> list[float]:
    """Near-origin Taylor expansion for the first-order corrected system."""

    a = alpha0
    b = beta0
    d = delta
    alpha2 = (
        -144 * a**3 + 768 * a * b - 2304 * a * b**2 + 96 * a**3 * d
        - 192 * a**3 * b * d + a**5 * d**2 + 768 * a * mu
        - 3072 * a * b * mu + 96 * a**3 * d * mu - 384 * a * mu**2
    ) / 2304.0
    beta2 = (8 * a**2 - 48 * a**2 * b - a**4 * d - 24 * a**2 * mu) / 96.0
    alpha4 = (
        147456 * a**3 + 62208 * a**5 - 2211840 * a**3 * b
        + 589824 * a * b**2 + 6635520 * a**3 * b**2
        - 3538944 * a * b**3 + 5308416 * a * b**4
        - 138240 * a**5 * d + 294912 * a**3 * b * d
        + 497664 * a**5 * b * d - 1474560 * a**3 * b**2 * d
        + 1769472 * a**3 * b**3 * d + 27648 * a**5 * d**2
        + 3456 * a**7 * d**2 - 105984 * a**5 * b * d**2
        + 96768 * a**5 * b**2 * d**2 + 768 * a**7 * d**3
        - 1536 * a**7 * b * d**3 + 5 * a**9 * d**4
        - 1474560 * a**3 * mu + 1179648 * a * b * mu
        + 7962624 * a**3 * b * mu - 8257536 * a * b**2 * mu
        + 14155776 * a * b**3 * mu + 294912 * a**3 * d * mu
        + 101376 * a**5 * d * mu - 1474560 * a**3 * b * d * mu
        + 1474560 * a**3 * b**2 * d * mu + 59904 * a**5 * d**2 * mu
        - 129024 * a**5 * b * d**2 * mu + 768 * a**7 * d**3 * mu
        + 589824 * a * mu**2 + 1990656 * a**3 * mu**2
        - 5308416 * a * b * mu**2 + 11206656 * a * b**2 * mu**2
        + 147456 * a**3 * d * mu**2 - 884736 * a**3 * b * d * mu**2
        + 25344 * a**5 * d**2 * mu**2 - 589824 * a * mu**3
        + 2359296 * a * b * mu**3 - 147456 * a**3 * d * mu**3
        + 147456 * a * mu**4
    ) / 17694720.0
    beta4 = (
        -2880 * a**4 + 3072 * a**2 * b + 17280 * a**4 * b
        - 27648 * a**2 * b**2 + 55296 * a**2 * b**3
        + 384 * a**4 * d + 432 * a**6 * d - 3840 * a**4 * b * d
        + 6912 * a**4 * b**2 * d - 92 * a**6 * d**2
        + 168 * a**6 * b * d**2 - a**8 * d**3
        + 3072 * a**2 * mu + 8640 * a**4 * mu
        - 39936 * a**2 * b * mu + 101376 * a**2 * b**2 * mu
        - 1536 * a**4 * d * mu + 3072 * a**4 * b * d * mu
        - 108 * a**6 * d**2 * mu - 10752 * a**2 * mu**2
        + 46080 * a**2 * b * mu**2 - 768 * a**4 * d * mu**2
        + 4608 * a**2 * mu**3
    ) / 184320.0

    return [
        a + alpha2 * radius**2 + alpha4 * radius**4,
        2.0 * alpha2 * radius + 4.0 * alpha4 * radius**3,
        b + beta2 * radius**2 + beta4 * radius**4,
        2.0 * beta2 * radius + 4.0 * beta4 * radius**3,
    ]


def initial_conditions_for_model(model: str):
    """Return the near-origin expansion for the requested model."""

    if model == "nonrel":
        return initial_conditions_nonrel
    if model == "relativistic":
        return initial_conditions_relativistic
    raise ValueError("model must be 'nonrel' or 'relativistic'.")


def regime_label_for_model(model: str) -> str:
    """Return the standardized output regime label for a model."""

    if model == "nonrel":
        return "effective_leading_order"
    if model == "relativistic":
        return "effective_first_order"
    raise ValueError("model must be 'nonrel' or 'relativistic'.")


def explosion_event(r: float, y: np.ndarray, mu: float, delta: float) -> float:
    """Stop when the scalar amplitude becomes numerically explosive."""

    return 5.0 - abs(y[0])


explosion_event.terminal = True
explosion_event.direction = -1


def node_event(r: float, y: np.ndarray, mu: float, delta: float) -> float:
    """Stop when the ground-state scalar profile crosses zero."""

    return y[0]


node_event.terminal = True
node_event.direction = -1


def monotonicity_event(r: float, y: np.ndarray, mu: float, delta: float) -> float:
    """Stop if the ground-state profile turns upward away from the origin."""

    return y[1]


monotonicity_event.terminal = True
monotonicity_event.direction = 1


def shooting_target(
    params: np.ndarray,
    mu: float,
    delta: float,
    r_max: float,
    config: FirstOrderConfig,
) -> np.ndarray:
    """Boundary residuals for the two-parameter shooting problem."""

    alpha0, beta0 = params
    if alpha0 <= 0 or beta0 > 0:
        return np.array([1.0e6, 1.0e6])

    rhs = equations_for_model(config.model)
    init = initial_conditions_for_model(config.model)
    y0 = init(alpha0, beta0, mu, delta, config.r_min)

    sol = solve_ivp(
        rhs,
        (config.r_min, r_max),
        y0,
        args=(mu, delta),
        method="Radau",
        rtol=config.rtol_ivp,
        atol=config.atol_ivp,
        events=(explosion_event, node_event, monotonicity_event),
    )
    if (not sol.success) and sol.t[-1] < 0.5 * r_max:
        return np.array([1.0e6, 1.0e6])

    f_end, df_end, phi_end, dphi_end = sol.y[:, -1]
    r_end = sol.t[-1]
    omega = decay_rate_from_mu(mu)
    if not np.isfinite(omega):
        return np.array([1.0e6, 1.0e6])

    scalar_bc = f_end + (r_end * df_end) / (1.0 + r_end * omega)
    potential_bc = phi_end + r_end * dphi_end
    return np.array([scalar_bc, potential_bc])


def mass_density_integrand(
    r: np.ndarray,
    y: np.ndarray,
    mu: float,
    delta: float,
    model: str,
) -> np.ndarray:
    """Dimensionless mass integrand dM/dr.

    The nonrelativistic model uses the leading density f^2.  The corrected
    model uses the corresponding first-order EFT energy-density expression.
    """

    f, df, phi, _ = y
    if model == "nonrel":
        density = f**2
    elif model == "relativistic":
        density = f**2 * (1.0 - 3.5 * phi) + 0.5 * df**2 + (delta / 16.0) * f**4
    else:
        raise ValueError("model must be 'nonrel' or 'relativistic'.")
    return 4.0 * np.pi * r**2 * density


def scalar_tail_fraction(sol, central_amplitude: float, config: FirstOrderConfig) -> float:
    """Return the largest relative scalar tail in the outer radial interval."""

    r_start = sol.t[0] + (1.0 - config.tail_check_fraction) * (sol.t[-1] - sol.t[0])
    r_grid = np.linspace(r_start, sol.t[-1], 500)
    f_tail = sol.sol(r_grid)[0]
    return float(np.max(np.abs(f_tail)) / max(abs(central_amplitude), 1.0e-30))


def compute_observables(sol, mu: float, delta: float, config: FirstOrderConfig) -> tuple[float, float, float]:
    """Compute total mass, R95, and compactness from a dense solution."""

    r_grid = np.linspace(sol.t[0], sol.t[-1], config.mass_grid_points)
    y_grid = sol.sol(r_grid)
    integrand = mass_density_integrand(r_grid, y_grid, mu, delta, config.model)
    cumulative_mass = cumulative_trapezoid(integrand, r_grid, initial=0.0)
    total_mass = cumulative_mass[-1]
    if not np.isfinite(total_mass) or total_mass <= 0:
        return np.nan, np.nan, np.nan
    radius_95 = np.interp(0.95 * total_mass, cumulative_mass, r_grid)
    compactness_95 = 0.95 * total_mass / max(radius_95, 1.0e-30)
    return total_mass, radius_95, compactness_95


def solve_one_mu(
    mu: float,
    config: FirstOrderConfig,
    seed: Iterable[float] | None = None,
    is_refined: bool = False,
) -> dict:
    """Solve one value of ``mu`` and return a standardized result row."""

    started = time.perf_counter()
    least_squares_calls = 0
    ivp_calls = 0
    seed_array = np.array(
        list(seed) if seed is not None else [config.alpha0_seed, config.beta0_seed],
        dtype=float,
    )
    regime_label = regime_label_for_model(config.model)
    run_id = make_run_id(regime_label, config.delta)
    rhs = equations_for_model(config.model)
    init = initial_conditions_for_model(config.model)

    for attempt in range(config.max_attempts):
        trial_seed = seed_array * (config.seed_growth**attempt)
        r_max, r_max_limit = radial_domain_for_mu(mu, config)
        while r_max <= r_max_limit:
            lower = np.array([max(1.0e-10, 0.20 * abs(trial_seed[0])), -20.0])
            upper = np.array([max(1.0, 4.00 * abs(trial_seed[0])), 0.0])
            least_squares_calls += 1
            res = least_squares(
                shooting_target,
                trial_seed,
                args=(mu, config.delta, r_max, config),
                bounds=(lower, upper),
                method="trf",
                ftol=config.ftol_ls,
                xtol=config.xtol_ls,
                gtol=config.gtol_ls,
                max_nfev=config.max_nfev_ls,
            )

            alpha0, beta0 = res.x
            if alpha0 < 1.0e-9:
                break

            y0 = init(alpha0, beta0, mu, config.delta, config.r_min)
            ivp_calls += 1
            sol = solve_ivp(
                rhs,
                (config.r_min, r_max),
                y0,
                args=(mu, config.delta),
                dense_output=True,
                method="Radau",
                rtol=config.rtol_ivp,
                atol=config.atol_ivp,
                events=(explosion_event, node_event, monotonicity_event),
            )
            if not sol.success:
                break

            residual = shooting_target(res.x, mu, config.delta, r_max, config)
            residual_norm = float(np.linalg.norm(residual))
            f_end = float(sol.y[0, -1])
            tail_fraction = scalar_tail_fraction(sol, alpha0, config)
            events = sum(len(event) for event in sol.t_events)

            if config.verbose:
                print(
                    f"mu={mu:.6g} attempt={attempt + 1}/{config.max_attempts} "
                    f"r_max={r_max:.2f} residual={residual_norm:.3e} "
                    f"|f_end|={abs(f_end):.3e} tail_fraction={tail_fraction:.3e}"
                )

            if (
                events == 0
                and residual_norm < config.residual_tolerance
                and abs(f_end) < config.tail_tolerance
                and tail_fraction < config.tail_fraction_tolerance
            ):
                mass, radius_95, compactness_95 = compute_observables(sol, mu, config.delta, config)
                accepted = np.isfinite(mass) and mass > 0 and np.isfinite(radius_95)
                return {
                    "run_id": run_id,
                    "regime": regime_label,
                    "model": config.model,
                    "field_type": "real_scalar_envelope",
                    "potential": "quartic_attractive",
                    "self_interaction": config.delta,
                    "lambda_input": config.delta,
                    "lambda_convention": "delta = lambda M_Pl^2 / m^2",
                    "mu": mu,
                    "omega": np.sqrt(max(2.0 * mu - mu**2, 0.0)),
                    "central_amplitude": alpha0,
                    "mass": mass,
                    "radius_95": radius_95,
                    "compactness_95": compactness_95,
                    "radius_type": "dimensionless_R95",
                    "success": True,
                    "accepted": bool(accepted),
                    "is_refined": is_refined,
                    "r_max": r_max,
                    "r_max_limit": r_max_limit,
                    "adaptive_r_max": config.adaptive_r_max,
                    "residual_norm": residual_norm,
                    "cost": float(res.cost),
                    "nodes": 0,
                    "tail": abs(f_end),
                    "tail_fraction": tail_fraction,
                    "elapsed_seconds": time.perf_counter() - started,
                    "attempts_used": attempt + 1,
                    "least_squares_calls": least_squares_calls,
                    "ivp_calls": ivp_calls,
                    "message": "OK",
                    "alpha0": alpha0,
                    "beta0": beta0,
                    "solution": sol,
                }

            trial_seed = res.x
            r_max = max(r_max + 1.0, r_max * config.r_max_factor)

    return {
        "run_id": run_id,
        "regime": regime_label,
        "model": config.model,
        "field_type": "real_scalar_envelope",
        "potential": "quartic_attractive",
        "self_interaction": config.delta,
        "lambda_input": config.delta,
        "lambda_convention": "delta = lambda M_Pl^2 / m^2",
        "mu": mu,
        "success": False,
        "accepted": False,
        "is_refined": is_refined,
        "elapsed_seconds": time.perf_counter() - started,
        "attempts_used": config.max_attempts,
        "least_squares_calls": least_squares_calls,
        "ivp_calls": ivp_calls,
        "message": "shooting failed",
    }


def autosave_scan(rows: list[dict], raw_results: list[dict], output_dir: str | Path) -> None:
    """Write partial EFT scan results after each completed mu point."""

    output_path = ensure_output_dir(output_dir)
    df = ordered_dataframe(rows)
    if not df.empty:
        df = mark_edge_max(df, "mu")
    df.to_csv(output_path / "sequence.csv", index=False)

    excluded = {"solution"}
    raw_rows = [{k: v for k, v in row.items() if k not in excluded} for row in raw_results]
    pd.DataFrame(raw_rows).to_csv(output_path / "attempt_log.csv", index=False)


def scan_mu(
    config: FirstOrderConfig,
    is_refined: bool = False,
    autosave_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Scan the configured ``mu`` values using continuation."""

    rows = []
    raw_results = []
    seed = np.array([config.alpha0_seed, config.beta0_seed], dtype=float)
    consecutive_failures = 0
    for mu in mu_grid(config):
        result = solve_one_mu(mu, config, seed=seed, is_refined=is_refined)
        raw_results.append(result)
        if result.get("success"):
            rows.append({k: v for k, v in result.items() if k != "solution"})
            seed = np.array([result["alpha0"], result["beta0"]], dtype=float)
            consecutive_failures = 0
        elif config.verbose:
            consecutive_failures += 1
            print(f"mu={mu:.6g} failed; keeping previous continuation seed.")
        else:
            consecutive_failures += 1

        if autosave_dir is not None:
            autosave_scan(rows, raw_results, autosave_dir)
        if config.max_consecutive_failures > 0 and consecutive_failures >= config.max_consecutive_failures:
            if config.verbose:
                print(
                    "Stopping scan after "
                    f"{consecutive_failures} consecutive failed mu points."
                )
            break

    df = ordered_dataframe(rows)
    if not df.empty:
        df = mark_edge_max(df, "mu")
    return df, raw_results


def refine_around_maximum(
    coarse_df: pd.DataFrame,
    config: FirstOrderConfig,
    n_refined: int = 30,
    width_factor: float = 1.5,
    autosave_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run a second pass around the coarse maximum-mass point."""

    if len(coarse_df) < 3:
        raise ValueError("Need at least three successful coarse points to refine.")

    masses = coarse_df["mass"].to_numpy(dtype=float)
    mus = coarse_df["mu"].to_numpy(dtype=float)
    imax = int(np.nanargmax(masses))

    if imax == 0:
        left, right = mus[0], mus[1]
    elif imax == len(mus) - 1:
        left, right = mus[-2], mus[-1]
    else:
        left, right = mus[imax - 1], mus[imax + 1]

    center = mus[imax]
    half_width = 0.5 * abs(right - left) * width_factor

    refined = FirstOrderConfig(**{**config.__dict__})
    refined.mu_spacing = "linear"
    refined.mu_min = max(center - half_width, np.finfo(float).eps)
    refined.mu_max = center + half_width
    refined.n_mu = n_refined
    refined.alpha0_seed = float(coarse_df.iloc[imax]["alpha0"])
    refined.beta0_seed = float(coarse_df.iloc[imax]["beta0"])

    return scan_mu(refined, is_refined=True, autosave_dir=autosave_dir)
