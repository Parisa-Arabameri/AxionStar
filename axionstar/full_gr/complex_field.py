"""Full-GR complex-field boson-star solver.

The complex scalar is stationary, so the full-GR system reduces to a radial
boundary-value problem for the metric functions and scalar profile.  Compared
with the real-field oscillaton solver, this system is numerically simpler: no
Fourier time expansion is needed.

The internal dimensionless mass used by the equations is called ``M_tilde_GR``.
For comparison with the EFT output convention, the standard output table also
exposes ``mass = 8*pi*M_tilde_GR``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp

from axionstar.common import make_run_id, mark_edge_max, ordered_dataframe


@dataclass
class ComplexGRConfig:
    """Configuration for the stationary complex-field full-GR solver."""

    potential_type: str = "quartic"  # free, quartic, liouville, log
    lambda_conv: float = -12.0  # package convention: lambda = 2 Lambda_GR
    Lambda_GR: float | None = None  # if None, computed as lambda_conv/2
    f_tilde: float = 1.0
    phi0_min: float = 0.03
    phi0_max: float = 0.35
    n_phi0: int = 35
    r_max: float = 100.0
    n_mesh: int = 700
    omega_guess: float = 0.95
    tol: float = 1.0e-5
    max_nodes: int = 50000
    max_radius_fraction: float = 0.90
    max_compactness_95: float = 0.35
    max_tail_fraction: float = 1.0e-3
    tail_check_fraction: float = 0.10
    omega_max: float = 1.20
    verbose: bool = True

    @property
    def Lambda(self) -> float:
        return self.lambda_conv / 2.0 if self.Lambda_GR is None else self.Lambda_GR


def potential(phi: np.ndarray, potential_type: str, Lambda: float, f_tilde: float) -> tuple[np.ndarray, np.ndarray]:
    """Return U(phi) and dU/d(phi^2) for the selected potential."""

    x = phi**2
    if potential_type == "free":
        return x, np.ones_like(phi)
    if potential_type == "quartic":
        return x + 0.5 * Lambda * x**2, 1.0 + Lambda * x
    if potential_type == "liouville":
        f2 = f_tilde**2
        y = np.minimum(x / f2, 80.0)
        return f2 * (np.exp(y) - 1.0), np.exp(y)
    if potential_type == "log":
        f2 = f_tilde**2
        return f2 * np.log1p(x / f2), 1.0 / (1.0 + x / f2)
    raise ValueError("potential_type must be free, quartic, liouville, or log.")


def make_ode(config: ComplexGRConfig):
    """Create the EKG radial RHS for the chosen potential."""

    Lambda = config.Lambda

    def ode(r, y, p):
        omega = p[0]
        gamma = y[0]
        lam = y[1]
        phi = y[2]
        psi = y[3]
        rr = np.maximum(r, 1.0e-8)

        U, dU = potential(phi, config.potential_type, Lambda, config.f_tilde)
        exp_lam = np.exp(np.clip(lam, -80.0, 80.0))
        exp_minus_lam = np.exp(np.clip(-lam, -80.0, 80.0))
        exp_minus_gamma = np.exp(np.clip(-gamma, -80.0, 80.0))

        common = omega**2 * exp_minus_gamma * phi**2 + exp_minus_lam * psi**2
        gamma_p = (exp_lam - 1.0) / rr + rr * exp_lam * (-U + common)
        lambda_p = -(exp_lam - 1.0) / rr + rr * exp_lam * (U + common)
        phi_pp = (
            -(2.0 / rr + 0.5 * (gamma_p - lambda_p)) * psi
            + exp_lam * phi * (dU - exp_minus_gamma * omega**2)
        )
        m_p = 0.5 * rr**2 * (U + omega**2 * exp_minus_gamma * phi**2 + exp_minus_lam * psi**2)
        return np.vstack([gamma_p, lambda_p, psi, phi_pp, m_p])

    return ode


def make_bc(phi0: float):
    """Boundary conditions at the origin and outer boundary."""

    def bc(ya, yb, p):
        return np.array([
            ya[1],          # lambda(0) = 0
            ya[3],          # phi'(0) = 0
            ya[4],          # m(0) = 0
            ya[2] - phi0,   # phi(0) = phi0
            yb[2],          # phi(rmax) = 0
            yb[0] + yb[1],  # Schwarzschild asymptotic relation
        ])

    return bc


def initial_guess(r: np.ndarray, phi0: float, omega_guess: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooth nodeless initial guess."""

    R0 = 10.0 / np.sqrt(max(phi0, 1.0e-3))
    phi_guess = phi0 * np.exp(-r / R0)
    psi_guess = -phi_guess / R0
    gamma_guess = np.zeros_like(r)
    lambda_guess = np.zeros_like(r)
    m_guess = 0.5 * phi0**2 * r**3 / (1.0 + r**3)
    m_guess -= m_guess[0]
    y_guess = np.vstack([gamma_guess, lambda_guess, phi_guess, psi_guess, m_guess])
    return y_guess, np.array([omega_guess])


def count_nodes(profile: np.ndarray) -> int:
    """Count robust sign changes in a scalar profile."""

    profile = np.asarray(profile)
    tol = 1.0e-6 * max(1.0, np.max(np.abs(profile)))
    mask = np.abs(profile) > tol
    if np.sum(mask) < 3:
        return 0
    signs = np.sign(profile[mask])
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def solve_one(phi0: float, config: ComplexGRConfig, previous_solution=None) -> dict:
    """Solve one central-field value."""

    r_mesh = np.linspace(1.0e-5, config.r_max, config.n_mesh)
    if previous_solution is None:
        y_guess, p_guess = initial_guess(r_mesh, phi0, config.omega_guess)
    else:
        y_guess = previous_solution.sol(r_mesh)
        scale = phi0 / max(abs(previous_solution.y[2, 0]), 1.0e-30)
        y_guess[2] *= scale
        y_guess[3] *= scale
        p_guess = previous_solution.p.copy()

    sol = solve_bvp(
        make_ode(config),
        make_bc(phi0),
        r_mesh,
        y_guess,
        p=p_guess,
        tol=config.tol,
        max_nodes=config.max_nodes,
        verbose=0,
    )
    if not sol.success:
        return {"success": False, "accepted": False, "phi0": phi0, "message": sol.message}

    r_dense = np.linspace(1.0e-5, config.r_max, 3000)
    y = sol.sol(r_dense)
    gamma, lam, phi, psi, m_profile = y
    omega_raw = sol.p[0]

    # Rescale the time coordinate so gamma(infinity) -> 0.
    ln_a = -(gamma[-1] + lam[-1])
    gamma = gamma + ln_a
    omega = omega_raw * np.sqrt(np.exp(ln_a))

    m_profile = np.maximum.accumulate(m_profile)
    M_tilde_GR = m_profile[-1]
    R95 = np.interp(0.95 * M_tilde_GR, m_profile, r_dense)
    R99 = np.interp(0.99 * M_tilde_GR, m_profile, r_dense)
    C95_GR = 0.95 * M_tilde_GR / max(R95, 1.0e-30)
    nodes = count_nodes(phi)
    tail_start = int((1.0 - config.tail_check_fraction) * len(phi))
    tail = abs(phi[-1])
    tail_fraction = float(np.max(np.abs(phi[tail_start:])) / max(abs(phi0), 1.0e-30))

    accepted = (
        nodes == 0
        and np.isfinite(M_tilde_GR)
        and M_tilde_GR > 0
        and R95 < config.max_radius_fraction * config.r_max
        and 0.0 < omega < config.omega_max
        and C95_GR < config.max_compactness_95
        and tail_fraction < config.max_tail_fraction
    )

    mass_comparison = 8.0 * np.pi * M_tilde_GR
    return {
        "success": True,
        "accepted": bool(accepted),
        "phi0": phi0,
        "omega": omega,
        "omega_raw": omega_raw,
        "mu": 1.0 - omega,
        "M_tilde_GR": M_tilde_GR,
        "mass": mass_comparison,
        "R95_tilde_GR": R95,
        "R99_tilde_GR": R99,
        "radius_95": R95,
        "C95_GR": C95_GR,
        "compactness_95": C95_GR,
        "nodes": nodes,
        "tail": tail,
        "tail_fraction": tail_fraction,
        "r": r_dense,
        "gamma": gamma,
        "lambda_metric": lam,
        "phi": phi,
        "psi": psi,
        "m_profile": m_profile,
        "solution": sol,
        "message": "OK" if accepted else "failed validation",
    }


def phi0_grid(config: ComplexGRConfig) -> np.ndarray:
    return np.linspace(config.phi0_min, config.phi0_max, config.n_phi0)


def build_sequence(config: ComplexGRConfig) -> tuple[pd.DataFrame, list[dict]]:
    """Build a complex-field sequence using continuation."""

    run_id = make_run_id("full_gr_complex", config.lambda_conv)
    rows = []
    raw = []
    previous_solution = None
    for phi0 in phi0_grid(config):
        if config.verbose:
            print(f"Solving complex full-GR phi0={phi0:.6g}")
        result = solve_one(phi0, config, previous_solution=previous_solution)
        raw.append(result)
        if result.get("success"):
            if result.get("accepted"):
                previous_solution = result["solution"]
            rows.append({
                "run_id": run_id,
                "regime": "full_gr",
                "model": "complex_field",
                "field_type": "complex_scalar",
                "potential": config.potential_type,
                "self_interaction": config.lambda_conv,
                "lambda_input": config.lambda_conv,
                "lambda_convention": "lambda = 2 Lambda_GR; mass column is 8*pi*M_tilde_GR",
                "mu": result["mu"],
                "omega": result["omega"],
                "central_amplitude": result["phi0"],
                "mass": result["mass"],
                "radius_95": result["radius_95"],
                "compactness_95": result["compactness_95"],
                "radius_type": "dimensionless_R95",
                "success": result["success"],
                "accepted": result["accepted"],
                "is_refined": False,
                "r_max": config.r_max,
                "nodes": result["nodes"],
                "tail": result["tail"],
                "tail_fraction": result["tail_fraction"],
                "message": result["message"],
                "Lambda_GR": config.Lambda,
                "M_tilde_GR": result["M_tilde_GR"],
                "R95_tilde_GR": result["R95_tilde_GR"],
                "R99_tilde_GR": result["R99_tilde_GR"],
                "C95_GR": result["C95_GR"],
            })
    df = ordered_dataframe(rows)
    if not df.empty:
        df = mark_edge_max(df, "central_amplitude")
    return df, raw


def _accepted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return accepted sequence rows robustly for in-memory or CSV-like data."""

    if "accepted" not in df.columns:
        return df.copy()
    accepted = df["accepted"]
    if accepted.dtype == bool:
        mask = accepted
    else:
        mask = accepted.astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return df[mask].copy()


def scan_lambda_detailed(
    lambda_values: Iterable[float],
    base_config: ComplexGRConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[ComplexGRConfig, pd.DataFrame, list[dict]]]]:
    """Scan several couplings and retain both full sequences and maxima.

    Returns
    -------
    combined_sequence:
        All sequence rows for every requested lambda value.
    lambda_summary:
        One maximum-mass accepted row per lambda value (or a diagnostic row
        when no accepted solution was found).
    runs:
        Per-lambda ``(config, sequence, raw_results)`` tuples so callers can
        save individual run directories and profile arrays without re-solving.
    """

    sequence_frames: list[pd.DataFrame] = []
    max_rows: list[dict] = []
    runs: list[tuple[ComplexGRConfig, pd.DataFrame, list[dict]]] = []

    for lambda_conv in lambda_values:
        config = ComplexGRConfig(
            **{**base_config.__dict__, "lambda_conv": float(lambda_conv), "Lambda_GR": None}
        )
        df, raw = build_sequence(config)
        runs.append((config, df, raw))
        if not df.empty:
            sequence_frames.append(df)

        accepted = _accepted_rows(df)
        if accepted.empty:
            max_rows.append({
                "run_id": make_run_id("full_gr_complex", float(lambda_conv)),
                "regime": "full_gr",
                "model": "complex_field",
                "field_type": "complex_scalar",
                "potential": config.potential_type,
                "self_interaction": float(lambda_conv),
                "lambda_input": float(lambda_conv),
                "lambda_convention": "lambda = 2 Lambda_GR; mass column is 8*pi*M_tilde_GR",
                "success": False,
                "accepted": False,
                "message": "no accepted solutions",
                "Lambda_GR": config.Lambda,
            })
            continue

        imax = accepted["mass"].astype(float).idxmax()
        max_row = dict(accepted.loc[imax])
        max_row["is_lambda_max"] = True
        max_rows.append(max_row)

    if sequence_frames:
        combined = pd.concat(sequence_frames, ignore_index=True, sort=False)
    else:
        combined = ordered_dataframe([])
    summary = ordered_dataframe(max_rows)
    return combined, summary, runs


def scan_lambda(lambda_values: Iterable[float], base_config: ComplexGRConfig) -> pd.DataFrame:
    """Backward-compatible lambda scan returning one maximum row per lambda."""

    _, summary, _ = scan_lambda_detailed(lambda_values, base_config)
    return summary


def profiles_to_npz(
    raw_results: list[dict],
    output_path,
    *,
    selected_phi0: float | None = None,
) -> None:
    """Save accepted complex-GR profiles compactly.

    ``mass_profile`` is retained for backward compatibility and stores the
    internal ``M_tilde_GR(r)`` profile.  ``mass_profile_comparison`` stores
    ``8*pi*M_tilde_GR(r)``, matching the package's standard ``mass`` column.
    If ``selected_phi0`` is supplied, only the accepted profile closest to that
    central amplitude is saved.
    """

    profiles = [res for res in raw_results if res.get("success") and res.get("accepted")]
    if selected_phi0 is not None and profiles:
        closest = min(profiles, key=lambda res: abs(float(res["phi0"]) - float(selected_phi0)))
        profiles = [closest]

    np.savez_compressed(
        output_path,
        phi0=np.array([res["phi0"] for res in profiles], dtype=float),
        omega=np.array([res["omega"] for res in profiles], dtype=float),
        mu=np.array([res["mu"] for res in profiles], dtype=float),
        mass=np.array([res["mass"] for res in profiles], dtype=float),
        radius_95=np.array([res["radius_95"] for res in profiles], dtype=float),
        r=np.array([res["r"] for res in profiles], dtype=object),
        phi=np.array([res["phi"] for res in profiles], dtype=object),
        psi=np.array([res["psi"] for res in profiles], dtype=object),
        gamma=np.array([res["gamma"] for res in profiles], dtype=object),
        lambda_metric=np.array([res["lambda_metric"] for res in profiles], dtype=object),
        mass_profile=np.array([res["m_profile"] for res in profiles], dtype=object),
        mass_profile_tilde=np.array([res["m_profile"] for res in profiles], dtype=object),
        mass_profile_comparison=np.array(
            [8.0 * np.pi * np.asarray(res["m_profile"]) for res in profiles],
            dtype=object,
        ),
    )
