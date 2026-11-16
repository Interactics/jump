"""Toy 2: cart-pole swing-up with JERK as the input (the paper's choice in Stages 1 and 3).

The cart is position-controlled, as the paper's joints are: its acceleration a becomes a state and
the input is jerk j = da/dt. The pole only feels the cart acceleration:
    ddq2 = -(a cos q2 + g sin q2) / l
state x = [q1, q2, dq1, dq2, a], input j.  min int (a^2 + W_J j^2) dt.
Why jerk: a (and hence force/torque) becomes continuous and starts/ends at zero, which a
position-playback controller can actually track.

    python toys/cartpole_jerk.py
"""

import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_jump.collocation import solve, trapezoid, trapz_cost  # noqa: E402

L, G = 0.5, 9.81
D, T, N = 1.0, 2.0, 40
A_MAX, D_MAX, W_J = 30.0, 2.0, 1e-3       # cart accel m/s^2, track m, jerk weight


def f(x, j):
    a, q2 = x[4], x[1]
    ddq2 = -(a * ca.cos(q2) + G * ca.sin(q2)) / L
    return ca.vertcat(x[2], x[3], a, ddq2, j)


def build_and_solve(n=N):
    opti = ca.Opti()
    X = opti.variable(5, n + 1)
    J = opti.variable(1, n + 1)
    h = T / n

    trapezoid(opti, f, X, J, h)
    opti.subject_to(X[:, 0] == [0, 0, 0, 0, 0])        # a(0) = 0: start from rest, no force step
    opti.subject_to(X[:, n] == [D, np.pi, 0, 0, 0])
    opti.subject_to(opti.bounded(-A_MAX, X[4, :], A_MAX))
    opti.subject_to(opti.bounded(-D_MAX, X[0, :], D_MAX))
    opti.minimize(trapz_cost([X[4, k] ** 2 + W_J * J[0, k] ** 2 for k in range(n + 1)], h))

    s = np.linspace(0, 1, n + 1)
    opti.set_initial(X[0, :], D * s)
    opti.set_initial(X[1, :], np.pi * s)

    t0 = time.perf_counter()
    sol, stats = solve(opti)
    return sol.value(X), sol.value(J), stats, time.perf_counter() - t0


if __name__ == "__main__":
    Xs, Js, stats, wall = build_and_solve()
    a = Xs[4]
    print(f"[cartpole/jerk] {stats['return_status']} | iters {stats['iter_count']} | {wall * 1e3:.0f} ms | "
          f"max|a| {np.abs(a).max():.1f} m/s^2 | max|j| {np.abs(Js).max():.0f} m/s^3 | "
          f"a(0), a(T) = {a[0]:.1e}, {a[-1]:.1e}")
    assert stats["success"]
    assert abs(Xs[1, -1] - np.pi) < 1e-6 and abs(a[0]) < 1e-9 and abs(a[-1]) < 1e-9
