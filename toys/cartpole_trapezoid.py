"""Toy 1: cart-pole swing-up, trapezoidal collocation, force input.

Problem (Kelly 2017, "An Introduction to Trajectory Optimization", Sec. 6):
  state x = [q1 cart pos, q2 pole angle (0 = down), dq1, dq2], input u = cart force
  min  int u^2 dt     s.t. dynamics, |u| <= u_max, |q1| <= d_max,
  q(0) = [0, 0], q(T) = [d, pi], velocities zero at both ends, T fixed.

    python toys/cartpole_trapezoid.py
"""

import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_jump.collocation import solve, trapezoid, trapz_cost  # noqa: E402

M1, M2, L, G = 1.0, 0.3, 0.5, 9.81      # cart kg, pole kg, pole m, m/s^2
D, T, N = 1.0, 2.0, 40                   # target cart pos m, duration s, intervals
U_MAX, D_MAX = 20.0, 2.0


def f(x, u):
    """Cart-pole dynamics (point-mass pole on a massless rod)."""
    q2, dq2 = x[1], x[3]
    s, c = ca.sin(q2), ca.cos(q2)
    den = M1 + M2 * s**2
    ddq1 = (L * M2 * s * dq2**2 + u + M2 * G * c * s) / den
    ddq2 = -(L * M2 * c * s * dq2**2 + u * c + (M1 + M2) * G * s) / (L * den)
    return ca.vertcat(x[2], x[3], ddq1, ddq2)


def build_and_solve():
    opti = ca.Opti()
    X = opti.variable(4, N + 1)
    U = opti.variable(1, N + 1)
    h = T / N

    trapezoid(opti, f, X, U, h)
    opti.subject_to(X[:, 0] == [0, 0, 0, 0])
    opti.subject_to(X[:, N] == [D, np.pi, 0, 0])
    opti.subject_to(opti.bounded(-U_MAX, U, U_MAX))
    opti.subject_to(opti.bounded(-D_MAX, X[0, :], D_MAX))
    opti.minimize(trapz_cost([U[0, k] ** 2 for k in range(N + 1)], h))

    # Guess: linear interpolation between the boundary states, zero force
    s = np.linspace(0, 1, N + 1)
    opti.set_initial(X[0, :], D * s)
    opti.set_initial(X[1, :], np.pi * s)

    t0 = time.perf_counter()
    sol, stats = solve(opti)
    return sol.value(X), sol.value(U), stats, time.perf_counter() - t0


def check_dynamics(Xs, Us, sub=50):
    """Re-simulate with RK4 on a fine grid, input linearly interpolated (what trapezoid assumes).
    Returns the end-state error: small = the collocated trajectory is physically consistent."""
    h = T / N / sub
    x = Xs[:, 0].copy()
    F =ca.Function("f", [xs := ca.SX.sym("x", 4), us := ca.SX.sym("u")], [f(xs, us)])
    for k in range(N):
        for j in range(sub):
            u0 = Us[k] + (Us[k + 1] - Us[k]) * j / sub
            u1 = Us[k] + (Us[k + 1] - Us[k]) * (j + 1) / sub
            um = 0.5 * (u0 + u1)
            k1 = F(x, u0).full().ravel()
            k2 = F(x + h / 2 * k1, um).full().ravel()
            k3 = F(x + h / 2 * k2, um).full().ravel()
            k4 = F(x + h * k3, u1).full().ravel()
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.abs(x - Xs[:, -1])


if __name__ == "__main__":
    Xs, Us, stats, wall = build_and_solve()
    err = check_dynamics(Xs, Us)
    print(f"[cartpole/trapezoid] {stats['return_status']} | iters {stats['iter_count']} | "
          f"{wall * 1e3:.0f} ms | max|u| {np.abs(Us).max():.2f} N | "
          f"open-loop RK4 end error: q1 {err[0]*1e3:.1f} mm, q2 {np.degrees(err[1]):.2f} deg")
    assert stats["success"]
    assert abs(Xs[1, -1] - np.pi) < 1e-6 and np.abs(Us).max() <= U_MAX + 1e-6
