"""Stage 1: SRMP trajectory optimization (paper Sec. IV). Launch NLP + flight NLP.

Launch (Eqs. 9-23): state = pos/vel/acc of the 4 SRMP DOF, input = jerk, free duration t_sl.
Flight (Eqs. 24-28): state = (theta, rho, drho, ddrho), input = dddrho, fixed duration t_f = 2 vz / g.

Choices the paper leaves open (see CLAUDE.md, Phase 2):
  * Eq. 22 is used multiplied by Fz = m (az + g), plus Fz >= 0: the fraction is 0/0 at take-off.
  * rho^I of Eq. 26 is never given a value -> 0 here, so rho^R is the whole inertia.
  * Initial state: at rest in a crouch, barbell aligned with the leg (theta_pl = 0).

Two modes:
  * sequential (the paper): launch first, then flight with the take-off state fixed (Eq. 28).
  * joint (ours): both phases in one NLP, so the launch knows which L the flight needs.

    python -m humanoid_jump.stages.stage1_srmp            # h = 0.10 m, l = 0.40 m, both modes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

from humanoid_jump.collocation import trapezoid
from humanoid_jump.models.srmp import COP_RANGE, G, MASS, PHI_RANGE, RHO_RANGE, q_of

ROOT = Path(__file__).resolve().parents[2]
N_L, N_F = 40, 30

# Table I (launch) and Table II (flight)
W_U = np.array([1e4, 1e2, 1e3, 1e4])
W_TH, W_V, W_L, W_LD, W_PL = 1e5, np.array([1e3, 1e6]), 5e2, 1e3, 1e8
W_T = 5e7                     # Table I says 5e5; with that t_sl is not determined (3.4 s unbounded). 5e7 -> 0.84 s
W_UF, W_IF = 1.0, 0.1

# Crouch a = 0.8 rad, arms down (scripts/srmp_numbers.py): CoM (0.048, 0.725) m, rho 7.08 kg m^2
P0, RHO0 = (0.048, 0.725), 7.08
TH_POS = (-0.3, 0.3)          # Eq. 20 bound on the barbell's absolute angle at take-off [rad]
L_MAX = 40.0                  # Eq. 23 bound on |L| [kg m^2/s]
SCALE = 1e-4                  # objective scaling for IPOPT (weights go up to 1e8)


def _ipopt(o, max_iter=3000):
    o.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes", "max_iter": max_iter})
    t0 = time.perf_counter()
    try:
        val, ok = o.solve().value, True
    except RuntimeError:
        val, ok = o.debug.value, False
    return val, ok, o.stats()["iter_count"], (time.perf_counter() - t0) * 1e3


def _flight_block(o, L, x0, t_f, th_tar):
    """Flight phase in Opti ``o``. x0 = (theta, rho, drho, ddrho) at take-off; L, x0 may be symbolic."""
    X, U = o.variable(4, N_F + 1), o.variable(1, N_F + 1)
    trapezoid(o, lambda x, u: ca.vertcat(L / x[1], x[2], x[3], u), X, U, t_f / N_F)     # Eq. 26, rho^I = 0
    o.subject_to(X[:, 0] == x0)                                                         # Eq. 28
    o.subject_to(X[0, N_F] + x0[0] == th_tar)                                           # Eq. 27
    o.subject_to(o.bounded(RHO_RANGE[0], X[1, :], RHO_RANGE[1]))
    return X, W_IF * ca.sumsqr(X[3, :]) + W_UF * ca.sumsqr(U)                           # Eq. 25


def _flight_out(Xs, L, t_f, ok=True, iters=0, ms=0.0):
    th0 = Xs[0, 0]
    reach = sorted(th0 + L * t_f / np.array(RHO_RANGE))        # landing angles reachable with constant rho
    return dict(ok=ok, iters=iters, ms=ms, t=np.linspace(0, t_f, N_F + 1), X=Xs, L=L, th0=th0,
                th_end=Xs[0, -1], reach=reach, omega=L / Xs[1])


def launch(h_jump: float, l_jump: float, joint: bool = False, th_tar: float = 0.0, t_max: float = 2.0,
           land_offset: float | None = None) -> dict:
    """land_offset = how far the foot lands ahead of the CoM [m]. None = the paper: mirror the take-off (Eq. 17)."""
    vz_exp = np.sqrt(2 * G * h_jump)
    t_f = 2 * vz_exp / G
    o = ca.Opti()
    X, U, T = o.variable(12, N_L + 1), o.variable(4, N_L + 1), o.variable()
    trapezoid(o, lambda x, u: ca.vertcat(x[4:12], u), X, U, T / N_L)
    Q = [q_of(X[:, k]) for k in range(N_L + 1)]

    th0, phi0, r0 = np.arctan2(P0[0], P0[1]), np.hypot(*P0), np.sqrt(RHO0 / MASS)
    x0 = np.r_[th0, phi0, 0.0, r0, np.zeros(8)]
    o.subject_to(X[:, 0] == x0)                       # at rest: CoP = CoM x (Sec. IV-A5)
    o.subject_to(o.bounded(0.15, T, t_max))
    o.subject_to(o.bounded(PHI_RANGE[0], X[1, :], PHI_RANGE[1]))

    for k in range(N_L + 1):
        p, a, Ld = Q[k]["p"], Q[k]["a"], Q[k]["Ld"]
        fz = MASS * (a[1] + G)
        mom = MASS * p[0] * (a[1] + G) - MASS * p[1] * a[0] - Ld      # = X_CoP * Fz   (Eq. 21)
        o.subject_to(fz >= 0)
        o.subject_to(COP_RANGE[0] * fz <= mom)                         # Eq. 22 times Fz
        o.subject_to(mom <= COP_RANGE[1] * fz)
        o.subject_to(o.bounded(RHO_RANGE[0], Q[k]["rho"], RHO_RANGE[1]))   # Eq. 23 (and 19 at k = N)
        o.subject_to(o.bounded(-L_MAX, Q[k]["L"], L_MAX))

    e = Q[N_L]
    if land_offset is None:
        o.subject_to(2 * e["p"][0] + e["v"][0] * t_f == l_jump)        # Eq. 17
    else:                                                              # Eq. 17 with a chosen landing offset
        o.subject_to(e["p"][0] + e["v"][0] * t_f + land_offset == l_jump)
        th_tar = X[0, N_L] - ca.atan2(land_offset, e["p"][1])          # landing leg angle = -atan(offset / height)
    o.subject_to(e["v"][1] == vz_exp)                                  # Eq. 18
    o.subject_to(e["a"] == ca.vertcat(0, -G))                          # Eq. 18: GRF = 0  <=>  F_l = tau_l - tau_pl = 0
    o.subject_to(e["Ld"] == 0)                                         # Eq. 19
    post = X[0, N_L] + X[2, N_L]
    o.subject_to(o.bounded(TH_POS[0], post, TH_POS[1]))                # Eq. 20

    J = W_TH * post**2 + W_L * e["L"] ** 2 + W_T * T**2                # Eqs. 12, 14, 16
    for k in range(N_L + 1):                                           # Eqs. 11, 12, 14, 15 (sums, as in the paper)
        J += ca.dot(W_U, U[:, k] ** 2) + ca.dot(W_V, Q[k]["v"] ** 2) + W_LD * Q[k]["Ld"] ** 2 + W_PL * X[2, k] ** 2
    if joint:
        XF, JF = _flight_block(o, e["L"], ca.vertcat(X[0, N_L], e["rho"], e["drho"], e["ddrho"]), t_f, th_tar)
        J += JF
        o.set_initial(XF[1, :], RHO0)
    o.minimize(SCALE * J)

    o.set_initial(X, np.tile(x0[:, None], N_L + 1))
    o.set_initial(X[1, :], np.linspace(phi0, 0.9, N_L + 1))
    o.set_initial(T, 0.5)
    val, ok, iters, ms = _ipopt(o)

    Xs, Ts = val(X), val(T)
    keys = ("p", "v", "a", "rho", "drho", "ddrho", "L", "Ld", "F")
    out = {n: np.array([np.ravel(val(Q[k][n])) for k in range(N_L + 1)]).squeeze() for n in keys}
    fz = MASS * (out["a"][:, 1] + G)
    mom = MASS * out["p"][:, 0] * (out["a"][:, 1] + G) - MASS * out["p"][:, 1] * out["a"][:, 0] - out["Ld"]
    cop = np.where(fz > 1.0, mom / np.maximum(fz, 1.0), np.nan)        # undefined once the foot unloads
    return dict(ok=ok, iters=iters, ms=ms, t=np.linspace(0, Ts, N_L + 1), T=Ts, X=Xs, U=val(U), fz=fz,
                fx=MASS * out["a"][:, 0], cop=cop, t_f=t_f, vz_exp=vz_exp, h_jump=h_jump, l_jump=l_jump, land_offset=land_offset,
                flight=_flight_out(val(XF), out["L"][-1], t_f) if joint else None, **out)


def flight(la: dict, th_tar: float = 0.0) -> dict:
    """Sequential mode: L is fixed by the launch; shape rho so the body turns to th_tar - theta_init."""
    o = ca.Opti()
    x0 = [la["X"][0, -1], la["rho"][-1], la["drho"][-1], la["ddrho"][-1]]
    X, J = _flight_block(o, la["L"][-1], x0, la["t_f"], th_tar)
    o.minimize(J)
    o.set_initial(X[1, :], la["rho"][-1])
    val, ok, iters, ms = _ipopt(o)
    return _flight_out(val(X), la["L"][-1], la["t_f"], ok, iters, ms)


def plot(la: dict, fl: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    t, tf = la["t"], la["T"] + fl["t"]
    ax[0, 0].plot(t, la["p"][:, 0], label="x"); ax[0, 0].plot(t, la["p"][:, 1], label="z")
    ax[0, 0].set_title("CoM position [m]"); ax[0, 0].legend()
    ax[0, 1].plot(t, la["fz"] / (MASS * G), label="Fz"); ax[0, 1].plot(t, la["fx"] / (MASS * G), label="Fx")
    ax[0, 1].set_title("GRF [body weight]"); ax[0, 1].legend()
    ax[0, 2].plot(t, la["cop"], label="CoP"); ax[0, 2].plot(t, la["p"][:, 0], "--", label="CoM x")
    for c in COP_RANGE:
        ax[0, 2].axhline(c, color="k", lw=0.8, ls=":")
    ax[0, 2].set_title("CoP x [m] (dotted: limits)"); ax[0, 2].legend()
    ax[1, 0].plot(t, la["L"]); ax[1, 0].plot(tf, np.full_like(tf, fl["L"]))
    ax[1, 0].set_title("CAM L [kg m^2/s]")
    ax[1, 1].plot(t, la["rho"]); ax[1, 1].plot(tf, fl["X"][1])
    for c in RHO_RANGE:
        ax[1, 1].axhline(c, color="k", lw=0.8, ls=":")
    ax[1, 1].set_title("inertia rho [kg m^2]")
    ax[1, 2].plot(t, np.degrees(la["X"][0]), label="leg angle"); ax[1, 2].plot(tf, np.degrees(fl["X"][0]), label="flight angle")
    ax[1, 2].plot(t, np.degrees(la["X"][0] + la["X"][2]), "--", label="barbell abs.")
    ax[1, 2].set_title("angles [deg]"); ax[1, 2].legend()
    for a in ax.ravel():
        a.axvline(la["T"], color="gray", lw=0.8); a.set_xlabel("t [s]"); a.grid(alpha=0.3)
    fig.tight_layout(); path.parent.mkdir(exist_ok=True); fig.savefig(path, dpi=130)


def report(tag: str, la: dict, fl: dict, l_jump: float) -> None:
    e, on = -1, la["fz"] > 50
    print(f"[{tag}] launch ok={la['ok']} iters {la['iters']} {la['ms']:.0f} ms | t_sl {la['T']:.3f} s, t_f {la['t_f']:.3f} s")
    print(f"  take-off: CoM ({la['p'][e, 0]:.3f}, {la['p'][e, 1]:.3f}) m, v ({la['v'][e, 0]:.3f}, {la['v'][e, 1]:.3f}) m/s "
          f"(vz target {la['vz_exp']:.3f}) | L {la['L'][e]:.2f} kg m^2/s | rho {la['rho'][e]:.2f} | "
          f"2 px + vx t_f = {2 * la['p'][e, 0] + la['v'][e, 0] * la['t_f']:.3f} m (target {l_jump})")
    print(f"  CoP in [{np.nanmin(la['cop']):.3f}, {np.nanmax(la['cop']):.3f}] m (limits {COP_RANGE}) | "
          f"peak Fz {la['fz'].max() / (MASS * G):.2f} BW | max |Fx/Fz| {np.abs(la['fx'][on] / la['fz'][on]).max():.2f} | "
          f"max |theta_pl| {np.degrees(np.abs(la['X'][2]).max()):.2f} deg | L range [{la['L'].min():.1f}, {la['L'].max():.1f}]")
    print(f"  flight ok={fl['ok']} iters {fl['iters']} | angle {np.degrees(fl['th0']):.2f} -> {np.degrees(fl['th_end']):.2f} deg "
          f"(target {np.degrees(-fl['th0']):.2f}; reachable by rho alone: [{np.degrees(fl['reach'][0]):.2f}, "
          f"{np.degrees(fl['reach'][1]):.2f}]) | rho {fl['X'][1].min():.2f}..{fl['X'][1].max():.2f} | "
          f"omega {np.degrees(fl['omega'].min()):.0f}..{np.degrees(fl['omega'].max()):.0f} deg/s")


if __name__ == "__main__":
    h_jump, l_jump = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.10, 0.40)

    seq = launch(h_jump, l_jump)                      # the paper's order
    report("sequential", seq, flight(seq), l_jump)

    la = launch(h_jump, l_jump, joint=True)           # ours
    fl = la["flight"]
    report("joint", la, fl, l_jump)
    plot(la, fl, ROOT / "results" / "stage1_srmp.png")

    # Phase 2 gate
    assert la["ok"]
    assert np.nanmin(la["cop"]) >= COP_RANGE[0] - 1e-4 and np.nanmax(la["cop"]) <= COP_RANGE[1] + 1e-4   # CoP in foot
    assert abs(la["v"][-1, 1] - la["vz_exp"]) < 1e-6                                                     # take-off vz
    assert abs(fl["th_end"] + fl["th0"]) < 1e-6                                                          # landing angle
    assert np.ptp(fl["X"][1]) > 0.05 or fl["reach"][0] <= -fl["th0"] <= fl["reach"][1]                   # inertia used
    print("Phase 2 gate: PASS")
