"""Numerical solvers for axion-star equilibrium sequences.

The package is organized by approximation regime:

* ``axionstar.effective.first_order`` solves the Schrodinger-Poisson
  system and its leading relativistic EFT correction.
* ``axionstar.effective.second_order`` solves the second-order EFT system.
* ``axionstar.full_gr.real_field`` solves the real-field KGE oscillaton
  system with a Fourier decomposition.
* ``axionstar.full_gr.complex_field`` solves the stationary complex-field
  boson-star system.

All solvers return pandas DataFrames with a shared set of leading columns so
that mass-radius curves from different regimes can be compared directly.
"""

__all__ = ["common"]

