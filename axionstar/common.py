"""Shared utilities and output conventions for the axion-star solvers.

Every solver writes the same leading output columns.  Regime-specific
diagnostics are still kept, but they appear after the common columns.  This
layout supports comparison plots across the nonrelativistic, EFT, and full-GR
calculations.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Mapping, Any
import json

import numpy as np
import pandas as pd


COMMON_COLUMNS = [
    "run_id",
    "regime",
    "model",
    "field_type",
    "potential",
    "self_interaction",
    "lambda_input",
    "lambda_convention",
    "mu",
    "omega",
    "central_amplitude",
    "mass",
    "radius_95",
    "compactness_95",
    "radius_type",
    "success",
    "accepted",
    "is_refined",
    "max_at_edge",
    "r_max",
    "residual_norm",
    "cost",
    "nodes",
    "tail",
    "tail_fraction",
    "message",
]


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create and return the output directory used by a run."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_id(regime: str, self_interaction: float | None = None) -> str:
    """Make a compact deterministic run label for file names."""

    if self_interaction is None:
        return regime.replace(" ", "_")
    value = f"{self_interaction:+.6g}".replace("+", "p").replace("-", "m")
    value = value.replace(".", "p")
    return f"{regime}_{value}".replace(" ", "_")


def ordered_dataframe(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame with common columns first.

    Missing common columns are inserted with NaN.  Any extra diagnostics are
    kept after the common columns in their original DataFrame order.
    """

    df = pd.DataFrame(list(rows))
    for col in COMMON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    extra_cols = [col for col in df.columns if col not in COMMON_COLUMNS]
    return df[COMMON_COLUMNS + extra_cols]


def finite_max_row(df: pd.DataFrame, mass_col: str = "mass") -> pd.Series:
    """Return the row with maximum finite mass."""

    if df.empty:
        raise ValueError("Cannot find maximum of an empty DataFrame.")
    mask = np.isfinite(df[mass_col].to_numpy(dtype=float))
    if not np.any(mask):
        raise ValueError(f"No finite values in column {mass_col!r}.")
    return df.loc[df.loc[mask, mass_col].idxmax()]


def save_config(config: Any, output_dir: str | Path, filename: str = "config.json") -> Path:
    """Save a dataclass or mapping configuration as JSON."""

    output_path = ensure_output_dir(output_dir) / filename
    if is_dataclass(config):
        data = asdict(config)
    elif isinstance(config, Mapping):
        data = dict(config)
    else:
        data = {"repr": repr(config)}
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return output_path


def save_results(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "sequence.csv",
) -> Path:
    """Save a result table as CSV."""

    output_path = ensure_output_dir(output_dir) / filename
    df.to_csv(output_path, index=False)
    return output_path


def save_max_summary(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "max_summary.csv",
) -> Path:
    """Save the maximum-mass row using the shared output convention."""

    output_path = ensure_output_dir(output_dir) / filename
    max_row = finite_max_row(df)
    pd.DataFrame([max_row]).to_csv(output_path, index=False)
    return output_path


def mark_edge_max(df: pd.DataFrame, parameter_col: str, mass_col: str = "mass") -> pd.DataFrame:
    """Mark whether the maximum mass occurs at the edge of the scanned range."""

    out = df.copy()
    if out.empty or parameter_col not in out.columns:
        out["max_at_edge"] = np.nan
        return out

    imax = int(np.nanargmax(out[mass_col].to_numpy(dtype=float)))
    out["max_at_edge"] = False
    if imax == 0 or imax == len(out) - 1:
        out.loc[out.index[imax], "max_at_edge"] = True
    return out
