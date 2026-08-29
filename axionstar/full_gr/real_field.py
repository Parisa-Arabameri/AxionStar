"""Full-GR real-field oscillaton solver.

A real scalar field does not admit a strictly stationary scalar profile.
Instead the field and metric are expanded in Fourier harmonics of the
fundamental oscillation frequency.  Following the usual oscillaton
construction, the scalar profile uses odd cosine modes and the metric
functions use even modes.

The solver can run either a broad coarse scan in central amplitude or an
optional refined scan around the maximum found in a previous sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, solve_bvp

from axionstar.common import make_run_id, mark_edge_max, ordered_dataframe


@dataclass
class RealGRConfig:
    """Configuration for the real-field KGE oscillaton sequence."""

    lambda_tilde: float = -4.0
    phi0_min: float = 0.02
    phi0_max: float = 0.40
    n_phi0: int = 45
    max_metric_harmonic: int = 4
    n_mesh: int = 450
    n_theta: int = 128
    x_min: float = 1.0e-4
    x_max: float = 250.0
    tol: float = 1.0e-4
    max_nodes: int = 50000
    omega_guess: float = 0.95
    max_R95_fraction_of_box: float = 0.90
    max_C95: float = 0.45
    max_tail_fraction: float = 1.0e-3
    tail_check_fraction: float = 0.10
    max_higher_harmonic_ratio: float = 0.20
    max_bc_residual: float = 1.0e-3
    omega_min_allowed: float = 0.0
    omega_max_allowed: float = 1.05
    max_R_jump_factor: float = 1.8
    max_M_jump_factor: float = 1.25
    verbose: bool = True


class RealFieldContext:
    """Precomputed harmonic and theta arrays for one real-field run."""

    def __init__(self, config: RealGRConfig):
        self.config = config
        self.odd_modes = np.arange(1, config.max_metric_harmonic, 2)
        self.even_modes = np.arange(0, config.max_metric_harmonic + 1, 2)
        self.n_phi = len(self.odd_modes)
        self.n_even = len(self.even_modes)
        self.idx_phi = slice(0, self.n_phi)
        self.idx_dphi = slice(self.n_phi, 2 * self.n_phi)
        self.idx_A = slice(2 * self.n_phi, 2 * self.n_phi + self.n_even)
        self.idx_C = slice(2 * self.n_phi + self.n_even, 2 * self.n_phi + 2 * self.n_even)
        self.n_y = 2 * self.n_phi + 2 * self.n_even
        self.theta = np.linspace(0.0, 2.0 * np.pi, config.n_theta, endpoint=False)
        self.cos_odd = np.array([np.cos(n * self.theta) for n in self.odd_modes])
        self.cos_even = np.array([np.cos(n * self.theta) for n in self.even_modes])

    def project_cos(self, values: np.ndarray, modes: np.ndarray) -> np.ndarray:
        """Project an array sampled on theta onto cosine modes."""

        coeffs = []
        for n in modes:
            c = np.cos(n * self.theta)
            coeffs.append(np.mean(values) if n == 0 else 2.0 * np.mean(values * c))
        return np.array(coeffs)

    def potential(self, Phi: np.ndarray) -> np.ndarray:
        return 0.5 * Phi**2 + (self.config.lambda_tilde / 24.0) * Phi**4

    def d_potential(self, Phi: np.ndarray) -> np.ndarray:
        return Phi + (self.config.lambda_tilde / 6.0) * Phi**3


def phi0_grid(config: RealGRConfig) -> np.ndarray:
    """Default broad central-amplitude scan."""

    return np.linspace(config.phi0_min, config.phi0_max, config.n_phi0)


def refined_phi0_grid_from_csv(
    csv_path: str | Path,
    n_refined: int = 121,
    width_factor: float = 1.5,
    mass_col: str = "mass",
    phi_col: str = "central_amplitude",
) -> np.ndarray:
    """Build a refined phi0 grid around the maximum of a previous run."""

    old = pd.read_csv(csv_path)
    return refined_phi0_grid_from_dataframe(
        old,
        n_refined=n_refined,
        width_factor=width_factor,
        mass_col=mass_col,
        phi_col=phi_col,
    )


def refined_phi0_grid_from_dataframe(
    old: pd.DataFrame,
    n_refined: int = 121,
    width_factor: float = 1.5,
    mass_col: str = "mass",
    phi_col: str = "central_amplitude",
) -> np.ndarray:
    """Build a refined phi0 grid around the maximum of a coarse sequence."""

    old = old.copy()
    if "accepted" in old.columns:
        old = old[old["accepted"].astype(bool)]
    if mass_col not in old.columns and "M_metric" in old.columns:
        mass_col = "M_metric"
    if phi_col not in old.columns and "phi10" in old.columns:
        phi_col = "phi10"
    if len(old) < 2:
        raise ValueError("A refined grid needs at least two accepted coarse points.")
    if mass_col not in old.columns or phi_col not in old.columns:
        raise ValueError(f"Cannot refine without columns {mass_col!r} and {phi_col!r}.")

    old = old.sort_values(phi_col).reset_index(drop=True)
    imax = int(old[mass_col].idxmax())
    if imax == 0:
        left, right = old.loc[0, phi_col], old.loc[1, phi_col]
    elif imax == len(old) - 1:
        left, right = old.loc[len(old) - 2, phi_col], old.loc[len(old) - 1, phi_col]
    else:
        left, right = old.loc[imax - 1, phi_col], old.loc[imax + 1, phi_col]
    center = old.loc[imax, phi_col]
    half_width = 0.5 * abs(right - left) * width_factor
    return np.linspace(center - half_width, center + half_width, n_refined)


def accepted_phi0_values_from_dataframe(
    old: pd.DataFrame,
    phi_col: str = "central_amplitude",
) -> np.ndarray:
    """Return accepted central amplitudes from a previous sequence.

    These values are useful as continuation warmup points for refined scans.
    They are not automatically written as refined output rows.
    """

    old = old.copy()
    if "accepted" in old.columns:
        old = old[old["accepted"].astype(bool)]
    if phi_col not in old.columns and "phi10" in old.columns:
        phi_col = "phi10"
    if phi_col not in old.columns:
        raise ValueError(f"Cannot read central amplitudes without column {phi_col!r}.")
    vals = pd.to_numeric(old[phi_col], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return np.array([], dtype=float)
    return np.unique(np.sort(vals))


def warmup_phi0_values_from_csv(
    csv_path: str | Path,
    refined_phi0_values: np.ndarray,
    phi_col: str = "central_amplitude",
) -> np.ndarray:
    """Select coarse points to solve before a refined full-GR real scan.

    The warmup points are accepted points from the previous coarse branch up to
    the beginning of the refined interval.  They allow ``solve_bvp`` to reach
    the physical branch by continuation before the refined points are attempted.
    """

    old = pd.read_csv(csv_path)
    return warmup_phi0_values_from_dataframe(old, refined_phi0_values, phi_col=phi_col)


def warmup_phi0_values_from_dataframe(
    old: pd.DataFrame,
    refined_phi0_values: np.ndarray,
    phi_col: str = "central_amplitude",
) -> np.ndarray:
    """Select coarse continuation warmup points for a refined scan."""

    coarse_phi0 = accepted_phi0_values_from_dataframe(old, phi_col=phi_col)
    refined = np.asarray(refined_phi0_values, dtype=float)
    if coarse_phi0.size == 0 or refined.size == 0:
        return np.array([], dtype=float)
    start = float(np.min(refined))

    # Use all accepted coarse points up to the first refined point.  If the
    # refined interval starts between two coarse points, include the nearest
    # lower point as the final warmup solution.  This gives the BVP solver a
    # branch-consistent initial guess instead of an analytic low-amplitude one.
    warm = coarse_phi0[coarse_phi0 <= start]
    if warm.size == 0:
        warm = coarse_phi0[:1]
    return np.unique(np.sort(warm))


def kge_rhs(x_array: np.ndarray, y: np.ndarray, p: np.ndarray, ctx: RealFieldContext) -> np.ndarray:
    """Fourier-projected KGE radial RHS."""

    omega = p[0]
    dydx = np.zeros_like(y)
    for k, x in enumerate(x_array):
        xx = max(x, 1.0e-10)
        phi_modes = y[ctx.idx_phi, k]
        dphi_modes = y[ctx.idx_dphi, k]
        A_modes = y[ctx.idx_A, k]
        C_modes = y[ctx.idx_C, k]

        Phi = phi_modes @ ctx.cos_odd
        Phi_x = dphi_modes @ ctx.cos_odd
        Phi_t = np.zeros_like(ctx.theta)
        Phi_tt = np.zeros_like(ctx.theta)
        for i, n in enumerate(ctx.odd_modes):
            Phi_t += -n * omega * phi_modes[i] * np.sin(n * ctx.theta)
            Phi_tt += -(n * omega) ** 2 * phi_modes[i] * np.cos(n * ctx.theta)

        A = A_modes @ ctx.cos_even
        C = C_modes @ ctx.cos_even
        C_t = np.zeros_like(ctx.theta)
        for i, n in enumerate(ctx.even_modes):
            if n != 0:
                C_t += -n * omega * C_modes[i] * np.sin(n * ctx.theta)

        # Clipping protects the collocation iterations from numerical overflow.
        # Validation rejects configurations with nonphysical metric functions.
        A_safe = np.clip(A, 1.0e-10, 1.0e10)
        C_safe = np.clip(C, 1.0e-10, 1.0e10)
        U = ctx.potential(Phi)
        Up = ctx.d_potential(Phi)

        A_x_theta = (
            A_safe * xx / 2.0 * (C_safe * Phi_t**2 + Phi_x**2 + 2.0 * A_safe * U)
            + A_safe / xx * (1.0 - A_safe)
        )
        C_x_theta = 2.0 * C_safe / xx * (1.0 + A_safe * (xx**2 * U - 1.0))
        C_x_over_C = C_x_theta / C_safe
        Phi_xx_theta = (
            C_safe * Phi_tt
            + 0.5 * C_t * Phi_t
            - Phi_x * (2.0 / xx - 0.5 * C_x_over_C)
            + A_safe * Up
        )

        dydx[ctx.idx_phi, k] = dphi_modes
        dydx[ctx.idx_dphi, k] = ctx.project_cos(Phi_xx_theta, ctx.odd_modes)
        dydx[ctx.idx_A, k] = ctx.project_cos(A_x_theta, ctx.even_modes)
        dydx[ctx.idx_C, k] = ctx.project_cos(C_x_theta, ctx.even_modes)
    return dydx


def make_bc(phi0: float, ctx: RealFieldContext):
    """Boundary conditions for one central scalar amplitude."""

    def bc(ya, yb, p):
        res = [ya[ctx.idx_phi][0] - phi0]
        res.extend(ya[ctx.idx_dphi])
        A_origin = ya[ctx.idx_A]
        res.append(A_origin[0] - 1.0)
        res.extend(A_origin[1:])
        res.extend(yb[ctx.idx_phi])
        C_inf = yb[ctx.idx_C]
        res.append(C_inf[0] - 1.0)
        res.extend(C_inf[1:])
        return np.array(res)

    return bc


def initial_guess(x: np.ndarray, phi0: float, ctx: RealFieldContext, omega_guess: float):
    """Smooth low-amplitude initial guess."""

    y = np.zeros((ctx.n_y, len(x)))
    R0 = 12.0 / np.sqrt(max(phi0, 1.0e-3))
    phi1 = phi0 * np.exp(-x / R0)
    dphi1 = -phi1 / R0
    y[ctx.idx_phi.start, :] = phi1
    y[ctx.idx_dphi.start, :] = dphi1
    for i in range(1, ctx.n_phi):
        amp = 0.02**i
        y[ctx.idx_phi.start + i, :] = amp * phi1
        y[ctx.idx_dphi.start + i, :] = amp * dphi1
    y[ctx.idx_A.start, :] = 1.0 + 0.1 * phi0**2 * x**2 * np.exp(-2.0 * x / R0)
    y[ctx.idx_C.start, :] = 1.0
    return y, np.array([omega_guess])


def continuation_guess(x: np.ndarray, previous_solution, new_phi0: float, old_phi0: float, ctx: RealFieldContext):
    """Use a previous solution as the next collocation initial guess."""

    y_new = previous_solution.sol(x)
    scale = new_phi0 / old_phi0
    y_new[ctx.idx_phi, :] *= scale
    y_new[ctx.idx_dphi, :] *= scale
    return y_new, previous_solution.p.copy()


def count_nodes(profile: np.ndarray) -> int:
    profile = np.asarray(profile)
    tol = 1.0e-6 * max(1.0, np.max(np.abs(profile)))
    mask = np.abs(profile) > tol
    if np.sum(mask) < 3:
        return 0
    signs = np.sign(profile[mask])
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def reconstruct_A_C(y_eval: np.ndarray, theta_test: np.ndarray, ctx: RealFieldContext):
    A_modes = y_eval[ctx.idx_A, :]
    C_modes = y_eval[ctx.idx_C, :]
    A_theta = np.zeros((len(theta_test), y_eval.shape[1]))
    C_theta = np.zeros_like(A_theta)
    for i, n in enumerate(ctx.even_modes):
        c = np.cos(n * theta_test)[:, None]
        A_theta += c * A_modes[i, :][None, :]
        C_theta += c * C_modes[i, :][None, :]
    return A_theta, C_theta


def isotropic_radius_from_A0(x: np.ndarray, A0: np.ndarray) -> np.ndarray:
    """Convert areal radius to isotropic radius using the metric A0 mode."""

    A0_safe = np.clip(A0, 1.0e-12, 1.0e12)
    M_nonred = 0.5 * x[-1] * (1.0 - 1.0 / A0_safe[-1])
    disc = max(x[-1] ** 2 - 2.0 * M_nonred * x[-1], 1.0e-30)
    R_outer = 0.5 * (x[-1] - M_nonred + np.sqrt(disc))
    integrand = np.sqrt(A0_safe) / np.clip(x, 1.0e-12, None)
    I = cumulative_trapezoid(integrand, x, initial=0.0)
    lnR = np.log(R_outer) - (I[-1] - I)
    return np.exp(lnR)


def radius_at_fraction(M_profile: np.ndarray, R_profile: np.ndarray, frac: float = 0.95) -> float:
    target = frac * M_profile[-1]
    if target <= M_profile[0] or target >= M_profile[-1]:
        return np.nan
    return float(np.interp(target, M_profile, R_profile))


def density_mass_profile(x: np.ndarray, y: np.ndarray, omega: float, ctx: RealFieldContext):
    dm_nonred_dx_avg = np.zeros(len(x))
    for j, xx_raw in enumerate(x):
        xx = max(xx_raw, 1.0e-12)
        phi_modes = y[ctx.idx_phi, j]
        dphi_modes = y[ctx.idx_dphi, j]
        A_modes = y[ctx.idx_A, j]
        C_modes = y[ctx.idx_C, j]
        Phi = phi_modes @ ctx.cos_odd
        Phi_x = dphi_modes @ ctx.cos_odd
        Phi_t = np.zeros_like(ctx.theta)
        for i, n in enumerate(ctx.odd_modes):
            Phi_t += -n * omega * phi_modes[i] * np.sin(n * ctx.theta)
        A = np.clip(A_modes @ ctx.cos_even, 1.0e-12, 1.0e12)
        C = np.clip(C_modes @ ctx.cos_even, 1.0e-12, 1.0e12)
        U = ctx.potential(Phi)
        dm_nonred_dx = xx**2 / (4.0 * A) * (C * Phi_t**2 + Phi_x**2) + xx**2 / 2.0 * U
        dm_nonred_dx_avg[j] = np.mean(dm_nonred_dx)
    M_red = 8.0 * np.pi * cumulative_trapezoid(dm_nonred_dx_avg, x, initial=0.0)
    return np.maximum.accumulate(M_red), 8.0 * np.pi * dm_nonred_dx_avg


def extract_mass_radius(sol, config: RealGRConfig, ctx: RealFieldContext, n_eval: int = 2500):
    x = np.linspace(config.x_min, config.x_max, n_eval)
    y = sol.sol(x)
    omega = sol.p[0]
    A0 = y[ctx.idx_A, :][0, :]
    A0_safe = np.clip(A0, 1.0e-12, 1.0e12)
    M_nonred_metric = 0.5 * x * (1.0 - 1.0 / A0_safe)
    M_metric = np.maximum.accumulate(8.0 * np.pi * M_nonred_metric)
    M_total = M_metric[-1]
    if not np.isfinite(M_total) or M_total <= 0:
        return None
    R95_areal = radius_at_fraction(M_metric, x)
    R_iso = isotropic_radius_from_A0(x, A0)
    R95_iso = radius_at_fraction(M_metric, R_iso)
    M_density, dMdx_density = density_mass_profile(x, y, omega, ctx)
    R95_density_areal = radius_at_fraction(M_density, x) if M_density[-1] > 0 else np.nan
    out = {
        "x_areal": x,
        "R_iso": R_iso,
        "omega": omega,
        "A0": A0,
        "M_metric_profile": M_metric,
        "M_metric": M_total,
        "M_density_profile": M_density,
        "dMdx_density": dMdx_density,
        "M_density": M_density[-1],
        "R95_areal": R95_areal,
        "R95_iso": R95_iso,
        "R95_density_areal": R95_density_areal,
        "C95_areal": (0.95 * (M_total / (8.0 * np.pi))) / max(R95_areal, 1.0e-30),
    }
    for i, n in enumerate(ctx.odd_modes):
        out[f"phi{n}"] = y[ctx.idx_phi.start + i, :]
    for i, n in enumerate(ctx.even_modes):
        out[f"A{n}"] = y[ctx.idx_A.start + i, :]
        out[f"C{n}"] = y[ctx.idx_C.start + i, :]
    return out


def validate_solution(extracted: dict, sol, phi0: float, config: RealGRConfig, ctx: RealFieldContext):
    phi1 = extracted["phi1"]
    phi1_max = np.max(np.abs(phi1))
    tail_start = int((1.0 - config.tail_check_fraction) * len(phi1))
    tail_phi1 = np.max(np.abs(phi1[tail_start:])) / max(phi1_max, 1.0e-30)
    if "phi3" in extracted:
        phi3 = extracted["phi3"]
        tail_phi3 = np.max(np.abs(phi3[tail_start:])) / max(phi1_max, 1.0e-30)
        harmonic_ratio = np.max(np.abs(phi3)) / max(phi1_max, 1.0e-30)
    else:
        tail_phi3 = 0.0
        harmonic_ratio = 0.0

    bc_res = make_bc(phi0, ctx)(sol.y[:, 0], sol.y[:, -1], sol.p)
    x = extracted["x_areal"]
    y_eval = sol.sol(x)
    theta_test = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    A_theta, C_theta = reconstruct_A_C(y_eval, theta_test, ctx)

    checks = {
        "phi1_nodeless": count_nodes(phi1) == 0,
        "mass_positive_finite": np.isfinite(extracted["M_metric"]) and extracted["M_metric"] > 0,
        "R95_inside_box": np.isfinite(extracted["R95_areal"])
        and extracted["R95_areal"] < config.max_R95_fraction_of_box * config.x_max,
        "omega_reasonable": config.omega_min_allowed < extracted["omega"] < config.omega_max_allowed,
        "compactness_reasonable": np.isfinite(extracted["C95_areal"]) and extracted["C95_areal"] < config.max_C95,
        "scalar_tail_small": tail_phi1 < config.max_tail_fraction and tail_phi3 < config.max_tail_fraction,
        "A_positive_all_times": np.min(A_theta) > 0.0,
        "C_positive_all_times": np.min(C_theta) > 0.0,
        "higher_harmonic_not_dominant": harmonic_ratio < config.max_higher_harmonic_ratio,
        "bc_residual_small": np.max(np.abs(bc_res)) < config.max_bc_residual,
    }
    diagnostics = {
        "nodes_phi1": count_nodes(phi1),
        "tail_phi1": tail_phi1,
        "tail_phi3": tail_phi3,
        "phi3_over_phi1": harmonic_ratio,
        "A_min_all_times": np.min(A_theta),
        "C_min_all_times": np.min(C_theta),
        "max_bc_residual": np.max(np.abs(bc_res)),
    }
    diagnostics.update(checks)
    return all(checks.values()), diagnostics


def branch_continuity(extracted: dict, previous_good: dict | None, config: RealGRConfig):
    if previous_good is None:
        return True, {"R_jump_factor": 1.0, "M_jump_factor": 1.0, "branch_continuity_ok": True}
    R_now, M_now = extracted["R95_areal"], extracted["M_metric"]
    R_prev, M_prev = previous_good["R95_areal"], previous_good["M_metric"]
    if min(R_now, M_now, R_prev, M_prev) <= 0 or not np.all(np.isfinite([R_now, M_now, R_prev, M_prev])):
        return False, {"R_jump_factor": np.inf, "M_jump_factor": np.inf, "branch_continuity_ok": False}
    R_jump = max(R_now / R_prev, R_prev / R_now)
    M_jump = max(M_now / M_prev, M_prev / M_now)
    ok = R_jump < config.max_R_jump_factor and M_jump < config.max_M_jump_factor
    return ok, {"R_jump_factor": R_jump, "M_jump_factor": M_jump, "branch_continuity_ok": ok}


def solve_one_phi0(phi0: float, config: RealGRConfig, ctx: RealFieldContext, previous_solution=None, previous_phi0=None):
    x_mesh = np.linspace(config.x_min, config.x_max, config.n_mesh)
    if previous_solution is None:
        y_guess, p_guess = initial_guess(x_mesh, phi0, ctx, config.omega_guess)
    else:
        y_guess, p_guess = continuation_guess(x_mesh, previous_solution, phi0, previous_phi0, ctx)
    return solve_bvp(
        lambda x, y, p: kge_rhs(x, y, p, ctx),
        make_bc(phi0, ctx),
        x_mesh,
        y_guess,
        p=p_guess,
        tol=config.tol,
        max_nodes=config.max_nodes,
        verbose=0,
    )


def build_sequence(
    config: RealGRConfig,
    phi0_values: np.ndarray | None = None,
    is_refined: bool = False,
    warmup_phi0_values: np.ndarray | None = None,
):
    """Run a real-field full-GR sequence.

    ``warmup_phi0_values`` are solved first and used only for continuation.
    They are recorded in the attempt log with ``warmup_only=True`` but are not
    included in ``sequence.csv`` or ``profiles.csv``.  This is important for
    refined scans near the maximum, where starting directly from the analytic
    initial guess can converge to a wrong high-harmonic branch.
    """

    ctx = RealFieldContext(config)
    values = phi0_grid(config) if phi0_values is None else np.asarray(phi0_values, dtype=float)
    warmup_values = (
        np.asarray(warmup_phi0_values, dtype=float)
        if warmup_phi0_values is not None
        else np.array([], dtype=float)
    )
    run_id = make_run_id("full_gr_real", config.lambda_tilde)
    rows = []
    raw = []
    profile_rows = []
    previous_solution = None
    previous_phi0 = None
    previous_good = None

    scan_items = [(float(phi0), True) for phi0 in warmup_values]
    scan_items.extend((float(phi0), False) for phi0 in values)

    for phi0, warmup_only in scan_items:
        if config.verbose:
            tag = "warmup " if warmup_only else ""
            print(f"Solving real full-GR {tag}phi1(0)={phi0:.6g}")
        try:
            sol = solve_one_phi0(phi0, config, ctx, previous_solution, previous_phi0)
            if not sol.success:
                raw.append({"phi10": phi0, "success": False, "warmup_only": bool(warmup_only), "message": sol.message})
                continue
            extracted = extract_mass_radius(sol, config, ctx)
            if extracted is None:
                raw.append({"phi10": phi0, "success": True, "accepted": False, "warmup_only": bool(warmup_only), "message": "mass extraction failed"})
                continue
            accepted, diagnostics = validate_solution(extracted, sol, phi0, config, ctx)
            branch_ok, branch_diag = branch_continuity(extracted, previous_good, config)
            diagnostics.update(branch_diag)
            accepted = accepted and branch_ok
            if accepted:
                previous_good = extracted
                previous_solution = sol
                previous_phi0 = phi0

            if warmup_only:
                raw.append({
                    "phi10": phi0,
                    "success": True,
                    "accepted": bool(accepted),
                    "warmup_only": True,
                    **diagnostics,
                })
                continue

            row = {
                "run_id": run_id,
                "regime": "full_gr",
                "model": "real_field_oscillaton",
                "field_type": "real_scalar",
                "potential": "quartic",
                "self_interaction": config.lambda_tilde,
                "lambda_input": config.lambda_tilde,
                "lambda_convention": "lambda_tilde in U=Phi^2/2 + lambda_tilde Phi^4/24",
                "mu": 1.0 - extracted["omega"],
                "omega": extracted["omega"],
                "central_amplitude": phi0,
                "mass": extracted["M_metric"],
                "radius_95": extracted["R95_areal"],
                "compactness_95": extracted["C95_areal"],
                "radius_type": "areal_R95",
                "success": True,
                "accepted": bool(accepted),
                "is_refined": is_refined,
                "warmup_only": False,
                "r_max": config.x_max,
                "residual_norm": diagnostics["max_bc_residual"],
        "nodes": diagnostics["nodes_phi1"],
                "tail": abs(extracted["phi1"][-1]),
                "tail_fraction": diagnostics["tail_phi1"],
                "message": "OK" if accepted else "failed validation",
                "M_metric": extracted["M_metric"],
                "M_density": extracted["M_density"],
                "R95_areal": extracted["R95_areal"],
                "R95_iso": extracted["R95_iso"],
                "R95_density_areal": extracted["R95_density_areal"],
            }
            row.update(diagnostics)
            rows.append(row)
            raw.append({"phi10": phi0, "success": True, "accepted": bool(accepted), "warmup_only": False, **diagnostics})

            if accepted:
                for j in range(len(extracted["x_areal"])):
                    profile_rows.append({
                        "lambda_tilde": config.lambda_tilde,
                        "phi10": phi0,
                        "omega": extracted["omega"],
                        "x_areal": extracted["x_areal"][j],
                        "R_iso": extracted["R_iso"][j],
                        "M_metric_profile": extracted["M_metric_profile"][j],
                        "M_density_profile": extracted["M_density_profile"][j],
                        "phi1": extracted["phi1"][j],
                    })
        except Exception as exc:
            raw.append({"phi10": phi0, "success": False, "accepted": False, "warmup_only": bool(warmup_only), "message": repr(exc)})

    df = ordered_dataframe(rows)
    if not df.empty:
        df = mark_edge_max(df, "central_amplitude")
    return df, pd.DataFrame(raw), pd.DataFrame(profile_rows)
