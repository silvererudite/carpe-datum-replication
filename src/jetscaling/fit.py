"""The parametric scaling-law fit. Follows Hoffmann et al. (2203.15556) appendix D.

STATUS: contract only. Implement in build step 6 (docs/orientation.md 6.5).
This is the module where a replication actually succeeds or fails; the training runs
are just data collection for it.

Model
-----
    L(N, D) = L_inf + A / N^alpha + B / D^beta

Fit it in the exponentiated parameterisation so every parameter is unconstrained and
the optimiser cannot walk into a negative A, B or L_inf:

    L_hat(N, D) = exp(e) + exp(a - alpha * log N) + exp(b - beta * log D)
    theta = (e, a, b, alpha, beta)

Objective: Huber loss, delta = 1e-3, on LOG-space residuals:

    obj(theta) = sum_i Huber_delta( log L_hat(N_i, D_i) - log L_i )

Two independent reasons for the log and the Huber. Log: the residuals we care about
are fractional, and a raw-space fit lets the largest losses dominate. Huber: with 16
points, one anomalous run (a diverged seed, a mis-shuffled subsample) would otherwise
drag the exponents on its own.

Optimisation: L-BFGS from a GRID of initialisations, keep the best final objective,
and REPORT the sensitivity. Hoffmann's grid, which we adopt:

    a, b     in {0, 5, 10, 15, 20, 25}
    alpha, beta in {0, 0.5, 1.0, 1.5, 2.0}
    e        in {-1, -0.5, 0, 0.5, 1}
                                        -> 4500 starts, seconds on 16 points

Why the grid and why reporting sensitivity is mandatory: Besiroglu et al.
(2404.10102) re-fit Hoffmann's own approach-3 data and found the published
uncertainties implausibly tight and the fit sensitive to exactly these choices. A
replication that reports a single (alpha, beta) with no init spread has not
replicated the method, only the number.

Uncertainty: bootstrap over the 16 grid points, >= 1000 resamples, percentile CIs on
alpha, beta, L_inf and on the derived quantities (alpha - beta), (alpha / beta).
With 16 points these CIs are wide -- that is the honest answer, not a bug to tune away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HUBER_DELTA = 1e-3
INIT_GRID = {
    "a": (0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    "b": (0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    "alpha": (0.0, 0.5, 1.0, 1.5, 2.0),
    "beta": (0.0, 0.5, 1.0, 1.5, 2.0),
    "e": (-1.0, -0.5, 0.0, 0.5, 1.0),
}


@dataclass(frozen=True)
class FitResult:
    l_inf: float
    a_coef: float  # A, in linear space
    b_coef: float  # B, in linear space
    alpha: float
    beta: float
    objective: float
    n_inits: int
    n_inits_within_tol: int  # how many starts reached the best objective (tol 1e-6)
    alpha_spread_top_decile: float  # max-min alpha over the best 10% of starts
    beta_spread_top_decile: float
    max_abs_rel_residual: float  # max |L_hat - L| / L over the fitted points


def predict(theta: np.ndarray, log_n: np.ndarray, log_d: np.ndarray) -> np.ndarray:
    """L_hat for theta = (e, a, b, alpha, beta). Vectorised, no loops."""
    raise NotImplementedError("build step 6")


def objective(theta: np.ndarray, log_n: np.ndarray, log_d: np.ndarray, log_l: np.ndarray) -> float:
    """Huber(delta=HUBER_DELTA) on log residuals. Analytic gradient is worth writing:
    L-BFGS with numerical gradients over 4500 starts is slow and less reliable."""
    raise NotImplementedError("build step 6")


def fit_scaling_law(n: np.ndarray, d: np.ndarray, loss: np.ndarray) -> FitResult:
    """Multi-start L-BFGS. Populates the init-sensitivity fields -- they are reported."""
    raise NotImplementedError("build step 6")


def bootstrap(
    n: np.ndarray, d: np.ndarray, loss: np.ndarray, n_resamples: int = 1000, seed: int = 0
) -> dict:
    """Percentile CIs. Returns {param: (lo, median, hi)} including alpha-beta, alpha/beta.

    Resample the grid POINTS with replacement (that is the unit of observation here,
    not the jets). Record how many resamples failed to converge and report it.
    """
    raise NotImplementedError("build step 6")


def fit_exponential_alternative(n: np.ndarray, d: np.ndarray, loss: np.ndarray) -> dict:
    """The falsification target: L = L_inf + A*exp(-N/n0) + B*exp(-D/d0).

    Same objective, same multi-start protocol, same number of free parameters (5), so
    the leave-one-out comparison against the power law is apples-to-apples.
    """
    raise NotImplementedError("build step 6")


def leave_one_out_compare(n: np.ndarray, d: np.ndarray, loss: np.ndarray) -> dict:
    """Refit both forms 16 times, each omitting one grid point; score on the held-out
    point. Returns per-point held-out log residuals for both forms plus the sign test.

    Pre-registered criterion (docs/preregistration.md C2): the power law wins if it
    has the smaller mean absolute held-out log residual AND is better on >= 12 of 16
    points (one-sided sign test, p < 0.05).
    """
    raise NotImplementedError("build step 6")


def refit_excluding_largest_n(n: np.ndarray, d: np.ndarray, loss: np.ndarray) -> dict:
    """Extrapolation check: drop the largest-N column, refit, predict that column.

    A scaling law whose only job is to interpolate the points it was fitted on is not
    a scaling law. Report predicted vs observed loss for each held-out cell.
    """
    raise NotImplementedError("build step 6")
