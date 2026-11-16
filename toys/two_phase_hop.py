"""Toy 3 (Phase 1 gate): point-mass forward hop, stance push-off + ballistic flight, free phase times.

Sagittal point mass (H1-2 mass), foot fixed at the origin during stance. Forces in body weights (BW).
  Stance  x = [x, z, vx, vz, Fx, Fz], input dF (force rate = jerk-like input, so GRF is continuous
          and can go to 0 at take-off).   acc = g * (Fx, Fz - 1)
  Flight  x = [x, z, vx, vz], ballistic.   acc = (0, -g)
Decision variables also include T1 (stance) and T2 (flight); each phase's step h = T / N.
  min  int_stance (|F - (0,1)|^2 + W_DF |dF|^2) dt
  s.t. Fz in [0, FZ_MAX], |Fx| <= MU Fz, leg length <= L_MAX, z >= Z_MIN,
       start at rest in a crouch (F = mg), F(T1) = 0, state continuity at take-off,
       land at x0 + D with z = Z_LAND and vz <= 0.

    python toys/two_phase_hop.py           # nominal guess + 20 random poor guesses
"""

import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_jump.collocation import trapezoid, trapz_cost  # noqa: E402

G = 9.81
N1, N2 = 30, 20
X0, Z0 = -0.03, 0.75                # crouched CoM relative to the foot [m]
D, Z_LAND = 0.4, 0.90               # forward jump distance, CoM height at touchdown [m]
L_MAX, Z_MIN = 0.95, 0.65           # max leg length, min CoM height in stance [m]
FZ_MAX, MU, W_DF = 3.0, 0.7, 1e-3   # BW, friction coeff, force-rate weight


def f_stance(x, dF):
    return ca.vertcat(x[2], x[3], G * x[4], G * (x[5] - 1), dF)


def f_flight(x, _u):
    return ca.vertcat(x[2], x[3], 0, -G)


def build():
    o = ca.Opti()
    X1, U1 = o.variable(6, N1 + 1), o.variable(2, N1 + 1)
    X2 = o.variable(4, N2 + 1)
    T1, T2 = o.variable(), o.variable()
    rows = {}

    def add(name, c):
        n0 = o.ng
        o.subject_to(c)
        rows[name] = rows.get(name, 0) + o.ng - n0

    trapezoid(o, f_stance, X1, U1, T1 / N1); rows["stance defects (6 x N1)"] = o.ng
    n0 = o.ng
    trapezoid(o, f_flight, X2, ca.DM.zeros(1, N2 + 1), T2 / N2)
    rows["flight defects (4 x N2)"] = o.ng - n0

    add("initial state (6)", X1[:, 0] == [X0, Z0, 0, 0, 0, 1])
    add("take-off GRF = 0 (2)", X1[4:6, N1] == 0)
    add("continuity (4)", X2[:, 0] == X1[0:4, N1])
    add("landing x, z (2)", X2[0:2, N2] == [X0 + D, Z_LAND])
    add("landing vz <= 0 (1)", X2[3, N2] <= 0)
    for k in range(N1 + 1):
        x, z, Fx, Fz = X1[0, k], X1[1, k], X1[4, k], X1[5, k]
        add("Fz bounds (N1+1)", o.bounded(0, Fz, FZ_MAX))
        add("friction cone (2 x (N1+1))", ca.vertcat(Fx - MU * Fz, -Fx - MU * Fz) <= 0)
        add("leg length (N1+1)", x**2 + z**2 <= L_MAX**2)
        add("z >= Z_MIN (N1+1)", z >= Z_MIN)
    add("phase times (2)", o.bounded(0.1, T1, 1.5))
    add("phase times (2)", o.bounded(0.05, T2, 1.0))

    L = [ca.sumsqr(X1[4:6, k] - ca.DM([0, 1])) + W_DF * ca.sumsqr(U1[:, k]) for k in range(N1 + 1)]
    o.minimize(trapz_cost(L, T1 / N1))
    return o, (X1, U1, X2, T1, T2), rows


def solve(guess: str, rng=None):
    o, (X1, U1, X2, T1, T2), rows = build()
    if guess == "nominal":   # crouch held, then a straight-line flight
        o.set_initial(X1[0:2, :], np.tile([[X0], [Z0]], N1 + 1))
        o.set_initial(X1[5, :], 1.0)
        o.set_initial(X2[0, :], np.linspace(X0, X0 + D, N2 + 1))
        o.set_initial(X2[1, :], Z_LAND)
        o.set_initial(T1, 0.5); o.set_initial(T2, 0.3)
    else:                    # poor: random states/inputs around zero, random phase times
        for V in (X1, U1, X2):
            o.set_initial(V, rng.normal(0, 1, V.shape))
        o.set_initial(T1, rng.uniform(0.1, 1.5)); o.set_initial(T2, rng.uniform(0.05, 1.0))
    o.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes", "max_iter": 3000})
    t0 = time.perf_counter()
    try:
        sol = o.solve()
        ok, val = True, sol.value
    except RuntimeError:
        ok, val = False, o.debug.value
    st = o.stats()
    return dict(ok=ok, iters=st["iter_count"], ms=(time.perf_counter() - t0) * 1e3,
                T1=val(T1), T2=val(T2), v_to=val(X1[2:4, N1]), Fmax=val(X1[5, :]).max(),
                cost=val(o.f), ng=o.ng, nx=o.nx, rows=rows)


if __name__ == "__main__":
    r = solve("nominal")
    print(f"NLP: {r['nx']} variables, {r['ng']} constraint rows")
    for k, v in r["rows"].items():
        print(f"  {v:4d}  {k}")
    assert sum(r["rows"].values()) == r["ng"], "row tally does not match the NLP"
    print(f"[nominal] ok={r['ok']} iters {r['iters']} {r['ms']:.0f} ms | T1 {r['T1']:.3f} s, T2 {r['T2']:.3f} s | "
          f"take-off v ({r['v_to'][0]:.2f}, {r['v_to'][1]:.2f}) m/s | peak Fz {r['Fmax']:.2f} BW | cost {r['cost']:.4f}")

    rng = np.random.default_rng(0)
    runs = [solve("poor", rng) for _ in range(20)]
    good = [x for x in runs if x["ok"]]
    same = [x for x in good if abs(x["cost"] - r["cost"]) < 1e-4 * max(1, r["cost"])]
    print(f"[poor x20] converged {len(good)}/20, same optimum as nominal {len(same)}/20 | "
          f"iters median {np.median([x['iters'] for x in good]):.0f} (max {max(x['iters'] for x in good)}) | "
          f"time median {np.median([x['ms'] for x in good]):.0f} ms")
    assert r["ok"] and len(same) >= 18
