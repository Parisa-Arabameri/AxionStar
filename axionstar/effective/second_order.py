"""Second-order EFT axion-star solver.

This module solves the second-order EFT boundary-value problem for five
coupled radial functions by shooting over the central values

    [F(0), Phi(0), Psi(0), A(0), B(0)].

The equations are written in the shifted-potential convention used in Appendix
C of Salehian et al.; the chemical potential enters through the asymptotic
condition

    Phi(infinity) = mu + mu^2/2 + mu^3/3.
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
class SecondOrderConfig:
    """Configuration for a second-order EFT sequence."""

    accuracy_mode: str = "standard"
    delta: float = -40.0
    eta: float = 1.0
    mu_min: float = 0.01
    mu_max: float = 0.05
    n_mu: int = 15
    F0_seed: float = 0.10
    Phi0_seed: float = -0.05
    Psi0_seed: float = -0.05
    A0_seed: float = 0.0
    B0_seed: float = 0.0
    r_min: float = 1.0e-5
    # The radial box is numerical.  In adaptive mode the code estimates the
    # scalar decay length 1/Omega and enlarges the domain for diffuse points.
    r_max_initial: float = 8.0
    r_max_factor: float = 1.4
    r_max_limit: float = 120.0
    adaptive_r_max: bool = True
    r_max_initial_decay_lengths: float = 8.0
    r_max_limit_decay_lengths: float = 30.0
    max_attempts: int = 20
    seed_growth: float = 1.03
    rtol_ivp: float = 1.0e-10
    atol_ivp: float = 1.0e-10
    ftol_ls: float = 1.0e-11
    xtol_ls: float = 1.0e-11
    gtol_ls: float = 1.0e-11
    max_nfev_ls: int = 500
    residual_tolerance: float = 5.0e-5
    tail_tolerance: float = 1.0e-3
    tail_fraction_tolerance: float = 1.0e-3
    tail_check_fraction: float = 0.10
    mass_grid_points: int = 5000
    max_consecutive_failures: int = 0
    verbose: bool = True


def phi_infinity_from_mu(mu: float) -> float:
    """Asymptotic shifted potential for the second-order EFT system."""

    return mu + 0.5 * mu**2 + (1.0 / 3.0) * mu**3


def omega_from_mu(mu: float) -> float:
    """Scalar exponential decay rate at infinity."""

    phi_inf = phi_infinity_from_mu(mu)
    omega2 = 2.0 * phi_inf * (1.0 - phi_inf + (2.0 / 3.0) * phi_inf**2)
    return np.sqrt(omega2) if omega2 > 0 else np.nan


def radial_domain_for_mu(mu: float, config: SecondOrderConfig) -> tuple[float, float]:
    """Choose initial and maximum radial domains for one second-order point.

    The second-order shifted system has the same qualitative tail,
    F ~ exp(-Omega r)/r, but Omega is computed from the shifted asymptotic
    potential.  The fixed config values remain lower bounds and safety guards.
    """

    if not config.adaptive_r_max:
        return config.r_max_initial, config.r_max_limit

    omega = omega_from_mu(mu)
    if not np.isfinite(omega) or omega <= 0:
        return config.r_max_initial, config.r_max_limit

    initial = max(config.r_max_initial, config.r_max_initial_decay_lengths / omega)
    limit = max(config.r_max_limit, config.r_max_limit_decay_lengths / omega)
    return initial, limit


def equations_second_order(r: float, y: np.ndarray, mu: float, delta: float, eta: float) -> list[float]:
    """Second-order EFT radial equations.

    y = [F, F', Phi, Phi', Psi, Psi', A, A', B].
    """

    F, dF, Phi, dPhi, Psi, dPsi, A, dA, B = y
    rr = max(r, 1.0e-30)

    geom_F = 1.0 - Phi - 2.0 * Psi + 2.0 * Psi**2 + 2.0 * Psi * Phi + (2.0 / 3.0) * Phi**2
    rest_F = (
        -geom_F * Phi * F
        + 3.0 * B * F
        + (1.0 - 2.0 * Psi - (4.0 / 3.0) * Phi) * (3.0 * F**3 / 16.0)
        + 0.5 * dF * (dPhi - dPsi + dA)
        - F * dF**2 / 16.0
        - (1.0 - 2.0 * Psi + 2.0 * Psi**2 + A - F**2 / 8.0) * (delta * F**3 / 8.0)
        - (1.0 - 2.0 * Psi + 1.5 * Phi) * (delta**2 * F**5 / 768.0)
        + delta**2 * F**3 * dF**2 / 1024.0
        + delta**3 * F**7 / 73728.0
    )
    d2F = -2.0 * rest_F - 2.0 * dF / rr

    geom_Psi = (
        1.0 - Phi - 2.0 * Psi + 2.0 * Psi**2 + 2.0 * Phi * Psi
        + Phi**2 + A - 3.0 * F**2 / 32.0
    )
    rest_Psi = (
        -0.5 * dPsi**2
        - geom_Psi * F**2 / 2.0
        - dF**2 / 4.0
        - (1.0 - 2.0 * Psi) * delta * F**4 / 32.0
        - 13.0 * eta * delta**2 * F**6 / 18432.0
    )
    d2Psi = -2.0 * dPsi / rr - rest_Psi

    geom_Phi = (
        1.0 - 4.0 * Phi - 2.0 * Psi + 2.0 * Psi**2 + 8.0 * Phi * Psi
        + 4.0 * Phi**2 + A + 3.0 * F**2 / 16.0
    )
    rest_Phi = (
        -geom_Phi * F**2 / 2.0
        + dPhi * (dPhi - dPsi)
        + (1.0 - 2.0 * Psi) * delta * F**4 / 16.0
        - eta * delta**2 * F**6 / 18432.0
    )
    d2Phi = -2.0 * dPhi / rr - rest_Phi

    rhs_A = 12.0 * B - 0.5 * F**2 * Phi - delta * F**4 / 64.0
    d2A = rhs_A - 2.0 * dA / rr
    dB = F**2 * dPhi / 16.0
    return [dF, d2F, dPhi, d2Phi, dPsi, d2Psi, dA, d2A, dB]


def taylor_coefficients(
    F0: float,
    Phi0: float,
    Psi0: float,
    A0: float,
    B0: float,
    delta: float,
    eta: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """Near-origin r^2 Taylor coefficients.

    The eta dependence enters the delta^2 F^6 terms in the Phi/Psi equations;
    the corresponding origin coefficients keep eta explicit.
    """

    F = F0
    P = Phi0
    Q = Psi0
    A = A0
    B = B0
    d = delta

    F2 = F * (
        9216.0 * A * F**2 * d - 221184.0 * B - F**6 * d**3
        + 144.0 * F**4 * P * d**2 - 192.0 * F**4 * Q * d**2
        + 96.0 * F**4 * d**2 - 1152.0 * F**4 * d
        + 18432.0 * F**2 * P + 18432.0 * F**2 * Q**2 * d
        - 18432.0 * F**2 * Q * d + 27648.0 * F**2 * Q
        + 9216.0 * F**2 * d - 13824.0 * F**2
        + 49152.0 * P**3 + 147456.0 * P**2 * Q - 73728.0 * P**2
        + 147456.0 * P * Q**2 - 147456.0 * P * Q + 73728.0 * P
    ) / 221184.0

    Phi2 = F**2 * (
        9216.0 * A + eta * F**4 * d**2 + 2304.0 * F**2 * Q * d
        - 1152.0 * F**2 * d + 1728.0 * F**2 + 36864.0 * P**2
        + 73728.0 * P * Q - 36864.0 * P + 18432.0 * Q**2
        - 18432.0 * Q + 9216.0
    ) / 110592.0

    Psi2 = F**2 * (
        9216.0 * A + 13.0 * eta * F**4 * d**2 - 1152.0 * F**2 * Q * d
        + 576.0 * F**2 * d - 864.0 * F**2 + 9216.0 * P**2
        + 18432.0 * P * Q - 9216.0 * P + 18432.0 * Q**2
        - 18432.0 * Q + 9216.0
    ) / 110592.0

    A2 = (768.0 * B - F**4 * d - 32.0 * F**2 * P) / 384.0
    B2 = F**2 * Phi2 / 16.0
    return F2, Phi2, Psi2, A2, B2


def initial_conditions(
    F0: float,
    Phi0: float,
    Psi0: float,
    A0: float,
    B0: float,
    delta: float,
    eta: float,
    radius: float,
) -> list[float]:
    """Initial data at a small positive radius."""

    F2, Phi2, Psi2, A2, B2 = taylor_coefficients(F0, Phi0, Psi0, A0, B0, delta, eta)
    return [
        F0 + F2 * radius**2,
        2.0 * F2 * radius,
        Phi0 + Phi2 * radius**2,
        2.0 * Phi2 * radius,
        Psi0 + Psi2 * radius**2,
        2.0 * Psi2 * radius,
        A0 + A2 * radius**2,
        2.0 * A2 * radius,
        B0 + B2 * radius**2,
    ]


def explosion_event(r, y, mu, delta, eta):
    return 50.0 - max(abs(y[0]), abs(y[2]), abs(y[4]), abs(y[6]), abs(y[8]))


explosion_event.terminal = True
explosion_event.direction = -1


def node_event(r, y, mu, delta, eta):
    return y[0]


node_event.terminal = True
node_event.direction = -1


def monotonicity_event(r, y, mu, delta, eta):
    return y[1]


monotonicity_event.terminal = True
monotonicity_event.direction = 1


def shooting_target(params, mu: float, config: SecondOrderConfig, r_max: float) -> np.ndarray:
    """Five boundary residuals for the second-order shooting problem."""

    F0, Phi0, Psi0, A0, B0 = params
    if F0 <= 0:
        return np.ones(5) * 1.0e6

    y0 = initial_conditions(F0, Phi0, Psi0, A0, B0, config.delta, config.eta, config.r_min)
    sol = solve_ivp(
        equations_second_order,
        (config.r_min, r_max),
        y0,
        args=(mu, config.delta, config.eta),
        method="Radau",
        rtol=config.rtol_ivp,
        atol=config.atol_ivp,
        dense_output=False,
        events=(explosion_event, node_event, monotonicity_event),
    )
    if (not sol.success) and sol.t[-1] < 0.5 * r_max:
        return np.ones(5) * 1.0e6

    F, dF, Phi, dPhi, Psi, dPsi, A, dA, B = sol.y[:, -1]
    r_end = sol.t[-1]
    omega = omega_from_mu(mu)
    if not np.isfinite(omega):
        return np.ones(5) * 1.0e6

    return np.array([
        F + (r_end * dF) / (1.0 + r_end * omega),
        Phi + r_end * dPhi - phi_infinity_from_mu(mu),
        Psi + r_end * dPsi,
        A + r_end * dA,
        B,
    ])


def mass_integrand(r: np.ndarray, y: np.ndarray, delta: float, eta: float) -> np.ndarray:
    """Dimensionless second-order mass integrand dM/dr."""

    F, dF, Phi, _, Psi, _, A, _, _ = y
    density = (
        F**2 * (
            1.0 - Phi - 2.5 * Psi + (25.0 / 8.0) * Psi**2
            + 2.5 * Phi * Psi + Phi**2 + A - 3.0 * F**2 / 32.0
        )
        + 0.5 * dF**2 * (1.0 - 0.5 * Psi)
        + (delta * F**4 / 16.0) * (1.0 - 2.5 * Psi)
        + 13.0 * eta * delta**2 * F**6 / 9216.0
    )
    return 4.0 * np.pi * r**2 * density


def compute_observables(sol, config: SecondOrderConfig) -> tuple[float, float, float]:
    """Compute M, R95, and compactness from the dense solution."""

    r_grid = np.linspace(sol.t[0], sol.t[-1], config.mass_grid_points)
    y_grid = sol.sol(r_grid)
    integrand = mass_integrand(r_grid, y_grid, config.delta, config.eta)
    cumulative_mass = cumulative_trapezoid(integrand, r_grid, initial=0.0)
    mass = cumulative_mass[-1]
    if not np.isfinite(mass) or mass <= 0:
        return np.nan, np.nan, np.nan
    radius_95 = np.interp(0.95 * mass, cumulative_mass, r_grid)
    compactness_95 = 0.95 * mass / max(radius_95, 1.0e-30)
    return mass, radius_95, compactness_95


def scalar_tail_fraction(sol, central_amplitude: float, config: SecondOrderConfig) -> float:
    """Return the largest relative scalar tail in the outer radial interval."""

    r_start = sol.t[0] + (1.0 - config.tail_check_fraction) * (sol.t[-1] - sol.t[0])
    r_grid = np.linspace(r_start, sol.t[-1], 500)
    F_tail = sol.sol(r_grid)[0]
    return float(np.max(np.abs(F_tail)) / max(abs(central_amplitude), 1.0e-30))


def validate_solution(
    sol,
    config: SecondOrderConfig,
    tail_fraction: float | None = None,
) -> tuple[bool, str]:
    """Ground-state and numerical-quality checks."""

    r_grid = np.linspace(sol.t[0], sol.t[-1], 3000)
    y_grid = sol.sol(r_grid)
    F = y_grid[0]
    dF = y_grid[1]
    Phi = y_grid[2]
    Psi = y_grid[4]
    A = y_grid[6]
    if tail_fraction is None:
        tail_fraction = scalar_tail_fraction(sol, F[0], config)
    mask = r_grid > 1.0e-3
    if np.min(F[mask]) < -1.0e-6:
        return False, "node detected"
    if np.max(dF[mask]) > 1.0e-5:
        return False, "profile is not monotonic"
    if tail_fraction > config.tail_fraction_tolerance:
        return False, "tail is too large"
    if max(np.max(np.abs(Phi)), np.max(np.abs(Psi)), np.max(np.abs(A))) > 1.0:
        return False, "metric potentials too large"
    return True, "OK"


def solve_one_mu(
    mu: float,
    config: SecondOrderConfig,
    seed: Iterable[float] | None = None,
    is_refined: bool = False,
) -> dict:
    """Solve one second-order EFT point."""

    started = time.perf_counter()
    least_squares_calls = 0
    ivp_calls = 0
    seed_array = np.array(
        list(seed) if seed is not None else [
            config.F0_seed,
            config.Phi0_seed,
            config.Psi0_seed,
            config.A0_seed,
            config.B0_seed,
        ],
        dtype=float,
    )
    run_id = make_run_id("second_order_eft", config.delta)

    for attempt in range(config.max_attempts):
        trial_seed = seed_array.copy()
        factor = config.seed_growth**attempt
        trial_seed[0] *= factor
        trial_seed[1] *= factor
        trial_seed[2] *= factor
        r_max, r_max_limit = radial_domain_for_mu(mu, config)

        while r_max <= r_max_limit:
            F0_seed = abs(trial_seed[0])
            lower = np.array([max(1.0e-4, 0.35 * F0_seed), -20.0, -20.0, -20.0, -20.0])
            upper = np.array([max(0.5, 2.50 * F0_seed), 20.0, 20.0, 20.0, 20.0])

            least_squares_calls += 1
            res = least_squares(
                shooting_target,
                trial_seed,
                args=(mu, config, r_max),
                bounds=(lower, upper),
                method="trf",
                ftol=config.ftol_ls,
                xtol=config.xtol_ls,
                gtol=config.gtol_ls,
                max_nfev=config.max_nfev_ls,
            )

            F0, Phi0, Psi0, A0, B0 = res.x
            y0 = initial_conditions(F0, Phi0, Psi0, A0, B0, config.delta, config.eta, config.r_min)
            ivp_calls += 1
            sol = solve_ivp(
                equations_second_order,
                (config.r_min, r_max),
                y0,
                args=(mu, config.delta, config.eta),
                method="Radau",
                rtol=config.rtol_ivp,
                atol=config.atol_ivp,
                dense_output=True,
                events=(explosion_event, node_event, monotonicity_event),
            )
            if not sol.success:
                break

            residual = shooting_target(res.x, mu, config, r_max)
            residual_norm = float(np.linalg.norm(residual))
            F_end = float(sol.y[0, -1])
            tail_fraction = scalar_tail_fraction(sol, F0, config)

            if config.verbose:
                print(
                    f"mu={mu:.6g} attempt={attempt + 1}/{config.max_attempts} "
                    f"r_max={r_max:.2f} residual={residual_norm:.3e} "
                    f"|F_end|={abs(F_end):.3e} tail_fraction={tail_fraction:.3e}"
                )

            if (
                residual_norm < config.residual_tolerance
                and abs(F_end) < config.tail_tolerance
                and tail_fraction < config.tail_fraction_tolerance
            ):
                mass, radius_95, compactness_95 = compute_observables(sol, config)
                valid, message = validate_solution(sol, config, tail_fraction)
                accepted = valid and np.isfinite(mass) and np.isfinite(radius_95)
                return {
                    "run_id": run_id,
                    "regime": "effective_second_order",
                    "model": "second_order",
                    "field_type": "real_scalar_envelope",
                    "potential": "quartic_attractive",
                    "self_interaction": config.delta,
                    "lambda_input": config.delta,
                    "lambda_convention": "delta = lambda M_Pl^2 / m^2",
                    "mu": mu,
                    "omega": omega_from_mu(mu),
                    "central_amplitude": F0,
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
                    "tail": abs(F_end),
                    "tail_fraction": tail_fraction,
                    "elapsed_seconds": time.perf_counter() - started,
                    "attempts_used": attempt + 1,
                    "least_squares_calls": least_squares_calls,
                    "ivp_calls": ivp_calls,
                    "message": message,
                    "F0": F0,
                    "Phi0": Phi0,
                    "Psi0": Psi0,
                    "A0": A0,
                    "B0": B0,
                    "solution": sol,
                }

            trial_seed = res.x
            r_max *= config.r_max_factor

    return {
        "run_id": run_id,
        "regime": "effective_second_order",
        "model": "second_order",
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


def autosave_scan(rows: list[dict], raw: list[dict], output_dir: str | Path) -> None:
    """Write partial second-order scan results after each completed mu point."""

    output_path = ensure_output_dir(output_dir)
    df = ordered_dataframe(rows)
    if not df.empty:
        df = mark_edge_max(df, "mu")
    df.to_csv(output_path / "sequence.csv", index=False)

    excluded = {"solution"}
    raw_rows = [{k: v for k, v in row.items() if k not in excluded} for row in raw]
    pd.DataFrame(raw_rows).to_csv(output_path / "attempt_log.csv", index=False)


def scan_mu(
    config: SecondOrderConfig,
    is_refined: bool = False,
    autosave_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Scan the second-order system over a linear mu grid."""

    rows = []
    raw = []
    seed = np.array([config.F0_seed, config.Phi0_seed, config.Psi0_seed, config.A0_seed, config.B0_seed])
    consecutive_failures = 0
    for mu in np.linspace(config.mu_min, config.mu_max, config.n_mu):
        result = solve_one_mu(mu, config, seed=seed, is_refined=is_refined)
        raw.append(result)
        if result.get("success"):
            rows.append({k: v for k, v in result.items() if k != "solution"})
            seed = np.array([result["F0"], result["Phi0"], result["Psi0"], result["A0"], result["B0"]])
            consecutive_failures = 0
        elif config.verbose:
            consecutive_failures += 1
            print(f"mu={mu:.6g} failed; keeping previous seed.")
        else:
            consecutive_failures += 1

        if autosave_dir is not None:
            autosave_scan(rows, raw, autosave_dir)
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
    return df, raw


def refine_around_maximum(
    coarse_df: pd.DataFrame,
    config: SecondOrderConfig,
    n_refined: int = 30,
    width_factor: float = 1.5,
    autosave_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Second pass around the coarse maximum-mass point."""

    if len(coarse_df) < 3:
        raise ValueError("Need at least three successful points to refine.")
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

    refined = SecondOrderConfig(**{**config.__dict__})
    refined.mu_min = max(center - half_width, np.finfo(float).eps)
    refined.mu_max = center + half_width
    refined.n_mu = n_refined
    refined.F0_seed = float(coarse_df.iloc[imax]["F0"])
    refined.Phi0_seed = float(coarse_df.iloc[imax]["Phi0"])
    refined.Psi0_seed = float(coarse_df.iloc[imax]["Psi0"])
    refined.A0_seed = float(coarse_df.iloc[imax]["A0"])
    refined.B0_seed = float(coarse_df.iloc[imax]["B0"])
    return scan_mu(refined, is_refined=True, autosave_dir=autosave_dir)
