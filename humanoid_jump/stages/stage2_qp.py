"""Stage 2: per-timestep QP that maps the Stage 1 references to joint space (paper Sec. V). Initial guess only.

No iterations: at each time step one small QP gives the joint accelerations (and, in launch, the ground wrench),
which are integrated forward. The result is the starting point for Stage 3.

  Launch (Eqs. 32-33): variables (ddq, wrench).  Hard: floating-base dynamics, sole fixed, friction, CoP, torque limits.
                       Soft: CoM acceleration, CAM rate and (weakly) inertia follow the SRMP references.
  Flight (Eqs. 38-39): variables ddq.            Hard: floating-base dynamics with no wrench, torque limits.
                       Soft: sole position relative to the CoM moves to the landing offset, sole flat; inertia.

Differences from the paper:
  * The paper's QP runs on the full 3D model with a hip-symmetry constraint. Here it runs on the 7-DOF symmetric
    model, which is that model with the symmetry built in, and is what Stage 3 uses.
  * Flight foot path: a quintic in time from the take-off offset to the landing offset (zero end velocity), not
    the SRMP leg angle, which is still turning at touch-down while Stage 3 wants the joints at rest (Eq. 56).

    python -m humanoid_jump.stages.stage2_qp        # Stage 1 -> Stage 2 -> Stage 3, compared with the naive guess
"""

from __future__ import annotations

import sys
import time

import numpy as np
import osqp
import scipy.sparse as sp

from humanoid_jump.models.srmp import COP_RANGE
from humanoid_jump.models.symmetric import G, MASS, NQ, Q_MAX, Q_MIN, TAU_MAX, crouch, linear_terms

BW = MASS * G
MU = 0.7
N_SUB = 4                                   # QP steps per Stage 3 node
W = dict(com=1e2, cam=1e1, rho=1e-2, rel=1e2, damp=1e-2, lam=1e-4, pitch=10.0)
KP_PITCH, KD_PITCH = 100.0, 20.0            # weak "pelvis upright" task: resolves the redundancy, keeps the leg off full extension
DDQ_MAX, T_LIM = 300.0, 0.05                # acceleration box [rad/s^2]; joint-limit look-ahead [s]
Q_LO = np.maximum(Q_MIN, [-3.14, 0.35, -0.897, -3.14])   # knee kept away from the straight-leg singularity
KP_COM, KD_COM, K_L, KP_RHO, KD_RHO = 100.0, 20.0, 20.0, 25.0, 10.0
KP_REL, KD_REL = 400.0, 40.0                # flight foot task, critically damped at 20 rad/s
W_FOOT = 30.0                               # Baumgarte stabilisation of the sole constraint [rad/s]
REF_KEYS = ("p", "v", "a", "L", "Ld", "rho", "drho", "ddrho")


def _qp(rows, eq, ineq, n):
    """min sum w |A x - b|^2  s.t.  A_eq x = b_eq,  lo <= A_in x <= hi.  Returns x, or None if OSQP fails."""
    P, q = 1e-9 * np.eye(n), np.zeros(n)
    for w, A, b in rows:
        P += w * A.T @ A
        q -= w * A.T @ b
    A = np.vstack([eq[0], ineq[0]])
    prob = osqp.OSQP()
    prob.setup(sp.csc_matrix(np.triu(P)), q, sp.csc_matrix(A), np.r_[eq[1], ineq[1]], np.r_[eq[1], ineq[2]],
               verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=4000, polish=False)
    res = prob.solve()                  # on an infeasible QP OSQP returns garbage in x: check the status
    return res.x if res.info.status in ("solved", "solved inaccurate") else None


def _box(q, dq):
    """Bounds on ddq: acceleration box, tightened so that q + dq T + ddq T^2 / 2 stays inside the joint range."""
    lo, hi = np.full(NQ, -DDQ_MAX), np.full(NQ, DDQ_MAX)
    lo[3:] = np.clip(2 * (Q_LO - q[3:] - dq[3:] * T_LIM) / T_LIM**2, -DDQ_MAX, DDQ_MAX)
    hi[3:] = np.clip(2 * (Q_MAX - q[3:] - dq[3:] * T_LIM) / T_LIM**2, -DDQ_MAX, DDQ_MAX)
    return lo, np.maximum(hi, lo)


def _pitch_row(q, dq):
    return np.eye(NQ)[2:3], np.r_[-KP_PITCH * q[2] - KD_PITCH * dq[2]]


def _terms(q, dq):
    return {k: np.array(v) if v.shape[1] > 1 else np.ravel(v) for k, v in linear_terms(q=q, dq=dq).items()}


def _launch_step(m, q, dq, r):
    """One launch QP. x = [ddq (7), wrench in body weights (3)]. r = references at this instant."""
    n, Z = NQ + 3, np.zeros
    pad = lambda J: np.c_[J, Z((J.shape[0], 3))]                          # noqa: E731
    dyn = np.c_[m["J_tau"], -BW * m["J_ddfoot"].T]                        # needed generalized force minus J^T wrench
    eq = (np.vstack([dyn[0:3], pad(m["J_ddfoot"])]),
          np.r_[-m["tau0"][0:3], -m["ddfoot0"] - 2 * W_FOOT * m["dfoot"] - W_FOOT**2 * m["foot"]])
    lam = lambda c: np.r_[Z(NQ), c]                                       # noqa: E731
    A_in = np.vstack([lam([0, 1, 0]), lam([1, -MU, 0]), lam([-1, -MU, 0]),
                      lam([0, COP_RANGE[0], 1]), lam([0, -COP_RANGE[1], -1]), dyn[3:7]])
    blo, bhi = _box(q, dq)
    A_in = np.vstack([A_in, pad(np.eye(NQ))])
    lo = np.r_[0, [-np.inf] * 4, -TAU_MAX - m["tau0"][3:7], blo]
    hi = np.r_[np.inf, [0] * 4, TAU_MAX - m["tau0"][3:7], bhi]
    Jp, bp = _pitch_row(q, dq)
    a_des = r["a"] + KP_COM * (r["p"] - m["p"]) + KD_COM * (r["v"] - m["v"])
    Ld_des = r["Ld"] + K_L * (r["L"] - m["L"])
    rho_dd = r["ddrho"] + KP_RHO * (r["rho"] - m["rho"]) + KD_RHO * (r["drho"] - m["drho"])
    rows = [(W["com"], pad(m["J_a"]), a_des - m["a0"]), (W["cam"], pad(m["J_Ld"]), np.r_[Ld_des - m["Ld0"]]),
            (W["rho"], pad(m["J_ddrho"]), np.r_[rho_dd - m["ddrho0"]]),
            (W["damp"], pad(np.eye(NQ)[3:]), -2.0 * dq[3:]), (W["lam"], np.c_[Z((3, NQ)), np.eye(3)], Z(3)),
            (W["pitch"], pad(Jp), bp)]
    x = _qp(rows, eq, (A_in, lo, hi), n)
    ok = x is not None
    if x is None:                       # ponytail: keep only the acceleration box; Stage 3 enforces the rest anyway
        x = _qp(rows, eq, (A_in[-NQ:], lo[-NQ:], hi[-NQ:]), n)
    if x is None:
        x = np.zeros(n)
    return x[:NQ], x[NQ:], ok


def _flight_step(m, q, dq, rel_d, drel_d, r):
    """One flight QP. x = ddq (7)."""
    eq = (m["J_tau"][0:3], -m["tau0"][0:3])
    blo, bhi = _box(q, dq)
    blo[:3], bhi[:3] = -np.inf, np.inf                                    # the base is in free fall
    ineq = (np.vstack([m["J_tau"][3:7], np.eye(NQ)]), np.r_[-TAU_MAX - m["tau0"][3:7], blo],
            np.r_[TAU_MAX - m["tau0"][3:7], bhi])
    rel_dd = KP_REL * (rel_d - m["rel"]) + KD_REL * (drel_d - m["drel"])
    rho_dd = r["ddrho"] + KP_RHO * (r["rho"] - m["rho"]) + KD_RHO * (r["drho"] - m["drho"])
    rows = [(W["rel"], m["J_ddrel"], rel_dd - m["ddrel0"]), (W["rho"], m["J_ddrho"], np.r_[rho_dd - m["ddrho0"]]),
            (W["damp"], np.eye(NQ)[3:], -2.0 * dq[3:])]
    x = _qp(rows, eq, ineq, NQ)
    if x is None:
        x = _qp(rows, eq, (np.zeros((1, NQ)), [-1.0], [1.0]), NQ)
    return x if x is not None else np.zeros(NQ)


def solve(ref: dict) -> dict:
    """ref = Stage 1 result (joint mode). Returns a Stage 3 guess dict(X, U, Lam, XF, UF) plus a few numbers."""
    t0 = time.perf_counter()
    NL, NF = len(ref["t"]) - 1, len(ref["flight"]["t"]) - 1
    fl = ref["flight"]
    at = lambda t, tt, Y: np.array([np.interp(t, tt, y) for y in np.atleast_2d(np.asarray(Y).T)]).squeeze()  # noqa: E731

    q, dq = crouch(0.8), np.zeros(NQ)                                    # ---------------- launch
    X, Lam, n_fail = np.zeros((3 * NQ, NL + 1)), np.zeros((3, NL + 1)), 0
    h = ref["T"] / NL / N_SUB
    for i in range(NL * N_SUB + 1):
        r = {k: at(i * h, ref["t"], ref[k]) for k in REF_KEYS}
        m = _terms(q, dq)
        ddq, lam, ok = _launch_step(m, q, dq, r)
        n_fail += not ok
        if i % N_SUB == 0:
            X[:, i // N_SUB], Lam[:, i // N_SUB] = np.r_[q, dq, ddq], lam
        if i < NL * N_SUB:
            dq = dq + h * ddq                                            # semi-implicit Euler
            q = q + h * dq
    v_to = m["v"].copy()

    rel0 = m["rel"].copy()                                               # ---------------- flight
    d_land = ref.get("land_offset")
    rel_end = np.array([-rel0[0] if d_land is None else d_land, rel0[1], 0.0])   # same height: sole meets the floor
    XF = np.zeros((3 * NQ, NF + 1))
    h, t_f = ref["t_f"] / NF / N_SUB, ref["t_f"]
    fref = dict(rho=fl["X"][1], drho=fl["X"][2], ddrho=fl["X"][3])
    for i in range(NF * N_SUB + 1):
        s = min(i * h / t_f, 1.0)
        blend, dblend = 10 * s**3 - 15 * s**4 + 6 * s**5, (30 * s**2 - 60 * s**3 + 30 * s**4) / t_f
        r = {k: at(i * h, fl["t"], fref[k]) for k in fref}
        m = _terms(q, dq)
        ddq = _flight_step(m, q, dq, rel0 + blend * (rel_end - rel0), dblend * (rel_end - rel0), r)
        if i % N_SUB == 0:
            XF[:, i // N_SUB] = np.r_[q, dq, ddq]
        if i < NF * N_SUB:
            dq = dq + h * ddq
            q = q + h * dq
    Lam[:, -1] = 0.0
    XF[:, 0] = X[:, -1]                                                  # phase link (Eq. 55)
    U = np.gradient(X[2 * NQ:], ref["t"], axis=1)
    UF = np.gradient(XF[2 * NQ:], fl["t"], axis=1)
    return dict(X=X, U=U, Lam=Lam, XF=XF, UF=UF, ms=(time.perf_counter() - t0) * 1e3, v_to=v_to,
                rel_end=m["rel"], rel_target=rel_end, foot_end=m["foot"], n_fail=n_fail)


if __name__ == "__main__":
    from humanoid_jump.stages import stage1_srmp as s1
    from humanoid_jump.stages import stage3_nlp as s3

    h_jump, l_jump = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.10, 0.40)
    ref = s1.launch(h_jump, l_jump, joint=True, land_offset=s3.LAND_OFFSET)
    g = solve(ref)
    print(f"[stage1] iters {ref['iters']} {ref['ms'] / 1e3:.2f} s")
    print(f"[stage2] {g['ms'] / 1e3:.2f} s for {(len(ref['t']) + len(ref['flight']['t']) - 2) * N_SUB} QPs | "
          f"take-off v ({g['v_to'][0]:.3f}, {g['v_to'][1]:.3f}) m/s (ref {ref['v'][-1, 0]:.3f}, {ref['v'][-1, 1]:.3f}) | "
          f"landing foot - CoM x {g['rel_end'][0]:.3f} m (target {g['rel_target'][0]:.3f}), "
          f"sole z {g['foot_end'][1]:.3f} m, pitch {np.degrees(g['foot_end'][2]):.1f} deg")
    warm = s3.solve(ref, guess=g)
    print("--- Stage 3 from the Stage 2 guess"); s3.report(warm)
    naive = s3.solve(ref)
    print("--- Stage 3 from the naive guess"); s3.report(naive)
    tot = lambda r, extra=0.0: (ref["ms"] + extra + r["ms"]) / 1e3      # noqa: E731
    print(f"\nTotal: naive {tot(naive):.1f} s ({naive['iters']} iters) | with Stage 2 {tot(warm, g['ms']):.1f} s "
          f"({warm['iters']} iters) | speed-up x{tot(naive) / tot(warm, g['ms']):.1f}")
    assert warm["ok"]
