"""Extended cost model C = 6ND + kD + mN. Pure analytics -- no GPU, no data.

STATUS: contract only. Implement in build step 5 (docs/orientation.md 6.6).
Build this while the sweep is running; it needs nothing from the sweep except, on
Day 2, our own fitted constants to overlay on the reference ones.

The standard compute-optimal analysis assumes the only cost is training FLOPs,
C = 6ND. In experimental HEP that is the wrong budget. Two extra terms:

    k * D   a per-training-example cost. Simulated jets are not free: producing one
            costs CPU-hours of generation, detector simulation and reconstruction.
    m * N   a per-parameter cost. A larger tagger costs more to *run*, in a trigger
            or in reconstruction, every time it is called -- often far more, summed
            over inference, than it cost to train.

Given a loss surface L(N, D) = L_inf + A/N^alpha + B/D^beta, the compute-optimal
allocation at budget C is

    (N*, D*) = argmin L(N, D)  s.t.  6ND + kD + mN = C

Pre-registered qualitative predictions (docs/preregistration.md C3):
    C3a  m > 0 pushes N* DOWN at fixed C (parameters got more expensive).
    C3b  k > 0 pushes D* DOWN and N* UP at fixed C (data got more expensive).
    C3c  both reduce the achievable loss at fixed C -- the frontier moves up.
These are checked numerically at >= 3 budgets across 1e17..1e21 FLOPs; the sign of
the shift is the claim, not its size.

Reference constants (from CLAUDE.md, attributed to ATL-SOFT-PUB-2026-002):
    L_inf = 0.619, alpha = 0.677, beta = 0.077
Notebook 02 uses these; on Day 2 notebook 03 overlays our own fitted constants.
"""

from __future__ import annotations

REFERENCE_CONSTANTS = {"l_inf": 0.619, "alpha": 0.677, "beta": 0.077, "a_coef": None, "b_coef": None}
# a_coef / b_coef are NOT given in CLAUDE.md. Until they are read off the note itself,
# notebook 02 must either (a) fit A and B so the reference curve passes through a stated
# reference point, or (b) plot only shape-invariant quantities (the N*/D* ratio and its
# deformation), which do not depend on A and B. Record which one you did.


def total_cost(n: float, d: float, k: float = 0.0, m: float = 0.0) -> float:
    """C = 6ND + kD + mN. Units: FLOPs, with k and m expressed in FLOP-equivalents."""
    return 6.0 * n * d + k * d + m * n


def loss(n: float, d: float, consts: dict) -> float:
    """L = L_inf + A/N^alpha + B/D^beta."""
    raise NotImplementedError("build step 5")


def compute_optimal(budget: float, consts: dict, k: float = 0.0, m: float = 0.0) -> tuple[float, float]:
    """argmin L(N, D) subject to total_cost(N, D, k, m) == budget.

    With k = m = 0 the constraint is a hyperbola and a Lagrange condition gives a
    closed form for N*/D*; with k or m nonzero, solve the 1-D problem numerically
    (parameterise by N, get D from the constraint, minimise). Do both and check they
    agree at k = m = 0 -- that is the free correctness test for this module.
    """
    raise NotImplementedError("build step 5")


def isoflop_curve(budget: float, consts: dict, k: float = 0.0, m: float = 0.0, n_points: int = 200):
    """-> (N_grid, L(N, D(N))) along the constraint surface, for the isoFLOP plots."""
    raise NotImplementedError("build step 5")


def frontier(budgets, consts: dict, k: float = 0.0, m: float = 0.0):
    """-> (budgets, N*, D*, L*) for the compute-optimal frontier plot."""
    raise NotImplementedError("build step 5")
