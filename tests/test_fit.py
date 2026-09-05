"""The fitter must recover known exponents from synthetic data before it is trusted
on real losses. On 16 real points there is no way to notice a fitter that is lying.

Build step 6: implement src/jetscaling/fit.py, then delete the skip guard below and
make these pass. Written first, on purpose -- red is the correct state today.
"""

from __future__ import annotations

import numpy as np
import pytest

from jetscaling import fit as fitmod

# Chosen so that BOTH terms vary comparably across the grid (~0.18 in L each).
# A synthetic test where one term is negligible cannot detect a fitter that
# ignores that axis -- which is exactly the failure mode you need to catch.
TRUE = {"l_inf": 0.30, "a_coef": 80.0, "alpha": 0.55, "b_coef": 2.0, "beta": 0.12}

# The real grid, so the synthetic test has the same (few points, narrow range)
# handicap as the real fit.
N_GRID = np.array([5.6e4, 1.9e5, 7.9e5, 3.1e6])
D_GRID = np.array([5.0e4, 1.5e5, 5.0e5, 1.2e6])


def synthetic_grid(noise_frac: float = 0.0, seed: int = 0):
    """(N, D, L) on the 4x4 grid from TRUE, with optional multiplicative noise.

    noise_frac is the std of a lognormal multiplying L -- 0.01 is a realistic stand-in
    for seed-to-seed variation in a converged run.
    """
    rng = np.random.default_rng(seed)
    n, d = np.meshgrid(N_GRID, D_GRID, indexing="ij")
    n, d = n.ravel(), d.ravel()
    loss = TRUE["l_inf"] + TRUE["a_coef"] * n ** -TRUE["alpha"] + TRUE["b_coef"] * d ** -TRUE["beta"]
    if noise_frac:
        loss = loss * np.exp(rng.normal(0.0, noise_frac, size=loss.shape))
    return n, d, loss


def _requires_fit():
    try:
        fitmod.predict(np.zeros(5), np.zeros(1), np.zeros(1))
    except NotImplementedError:
        pytest.skip("jetscaling.fit not implemented yet (build step 6)")
    except Exception:
        pass  # any other error means it IS implemented; let the real test report it


def test_synthetic_grid_is_well_posed():
    """Scaffold self-check: the synthetic surface must actually vary on this grid.

    If the loss range across the grid were comparable to realistic noise, the test
    below could not discriminate anything -- and neither could the real sweep.
    """
    _n, _d, loss = synthetic_grid()
    assert loss.shape == (16,)
    assert np.all(np.isfinite(loss))
    assert loss.max() - loss.min() > 0.05, "synthetic grid too flat to be a real test"
    assert loss.min() > TRUE["l_inf"], "loss must sit above the floor everywhere"


def test_recovers_exponents_noiseless():
    _requires_fit()
    n, d, loss = synthetic_grid(noise_frac=0.0)
    res = fitmod.fit_scaling_law(n, d, loss)
    assert res.alpha == pytest.approx(TRUE["alpha"], rel=0.05)
    assert res.beta == pytest.approx(TRUE["beta"], rel=0.05)
    assert res.l_inf == pytest.approx(TRUE["l_inf"], rel=0.05)


def test_recovers_exponents_within_bootstrap_ci():
    """The pre-registered version of the check: truth inside the 95% CI, with noise."""
    _requires_fit()
    n, d, loss = synthetic_grid(noise_frac=0.01, seed=7)
    ci = fitmod.bootstrap(n, d, loss, n_resamples=1000, seed=0)
    for name in ("alpha", "beta", "l_inf"):
        lo, _median, hi = ci[name]
        assert lo <= TRUE[name] <= hi, f"{name}: truth {TRUE[name]} outside CI [{lo}, {hi}]"


def test_reports_init_sensitivity():
    """Besiroglu et al. 2404.10102: a fit reported without its init spread is not a
    replication of the method. The FitResult must carry those fields, populated."""
    _requires_fit()
    n, d, loss = synthetic_grid(noise_frac=0.01, seed=3)
    res = fitmod.fit_scaling_law(n, d, loss)
    assert res.n_inits > 100, "multi-start grid too small"
    assert 0 < res.n_inits_within_tol <= res.n_inits
    assert res.alpha_spread_top_decile >= 0.0
    assert res.max_abs_rel_residual < 0.05


def test_power_law_beats_exponential_on_power_law_data():
    """Sanity direction check for C2: on data generated FROM a power law, the
    leave-one-out comparison must pick the power law. If it does not, the comparison
    itself is broken and C2 cannot be evaluated."""
    _requires_fit()
    n, d, loss = synthetic_grid(noise_frac=0.01, seed=11)
    out = fitmod.leave_one_out_compare(n, d, loss)
    assert out["power_law_wins"] >= 12, out
