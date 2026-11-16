"""Shared direct-collocation helpers (CasADi): multi-phase OCP, jerk input, free phase times.

Phase 1 of the roadmap. Replaces the paper's GPOPS-II transcription.

Trapezoidal rule: x[k+1] - x[k] = h/2 * (f(x[k], u[k]) + f(x[k+1], u[k+1])).
One defect row per state per interval -> nx * N equality rows per phase.
"""

from __future__ import annotations

import casadi as ca


def trapezoid(opti: ca.Opti, f, X, U, h) -> None:
    """Add trapezoidal defect constraints. X: nx x (N+1), U: nu x (N+1), h: step (may be a variable)."""
    N = X.shape[1] - 1
    F = [f(X[:, k], U[:, k]) for k in range(N + 1)]
    for k in range(N):
        opti.subject_to(X[:, k + 1] - X[:, k] == h / 2 * (F[k] + F[k + 1]))


def trapz_cost(L, h):
    """Trapezoidal quadrature of a list of running-cost samples L[0..N]."""
    return h / 2 * (L[0] + L[-1]) + h * ca.sum1(ca.vertcat(*L[1:-1]))


def solve(opti: ca.Opti, verbose: bool = False):
    """IPOPT with quiet output; returns (sol, stats). Raises if IPOPT fails."""
    opts = {"print_time": False} if not verbose else {}
    opti.solver("ipopt", opts, {"print_level": 5 if verbose else 0, "sb": "yes"})
    sol = opti.solve()
    return sol, opti.stats()
