"""Stage 3: whole-body NLP on the 7-DOF symmetric model (paper Sec. VI). Launch + flight in one NLP.

State = (q, dq, ddq) of the 7 DOF (Eq. 42), input = jerk (+ ground wrench in launch, Eq. 43).
References (CoM velocity, inertia rho, CAM rate) come from Stage 1, node by node, same durations.

Differences from the paper (see CLAUDE.md, Phase 3):
  * Flat foot during launch (sole fixed, CoP inside the sole) instead of "toe fixed + heel may lift":
    the paper applies a ground torque at the toe even when the heel is up, which a point contact cannot give.
    ponytail: no heel lift; add a toe-contact phase if flat-footed take-off limits the jump.
  * Joint torque limits, friction and CoP limits are constraints (the paper lists none of them).
  * Launch and flight are one NLP (the paper solves them one after the other).
  * Landing: sole flat on the floor at the end of flight, at x = l, foot LAND_OFFSET ahead of the CoM, and
    Stage 1's distance equation uses the same offset. (The paper's Eq. 56 mirrors the initial offset, which
    contradicts Eq. 17; mirroring the take-off as Eq. 17 assumes lands too far back on the heels.)
  * Flight acceleration cost (Eq. 54) covers the joints only: the base is in free fall, penalising it
    only adds a constant -g term that hurts conditioning.

    python -m humanoid_jump.stages.stage3_nlp              # h = 0.10 m, l = 0.40 m, Stage 2 guess (naive if that fails)
    python -m humanoid_jump.stages.stage3_nlp --naive      # naive initial guess only (the Phase 3 baseline)
    python -m humanoid_jump.stages.stage3_nlp 0.2 0.5 --box   # onto a 0.10 m box (not in the paper) -> stage3_box.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

from humanoid_jump.collocation import trapezoid
from humanoid_jump.models.srmp import COP_RANGE
from humanoid_jump.models.symmetric import G, MASS, NQ, Q_MAX, Q_MIN, TAU_MAX, crouch, q_of
from humanoid_jump.stages import stage1_srmp as s1

ROOT = Path(__file__).resolve().parents[2]
MU = 0.7
# IPOPT options for starting from a good guess: by default IPOPT pushes the start point away from the bounds
# (bound_push 1e-2) and begins with a large barrier (mu_init 0.1), which throws a warm start away.
WARM = dict(mu_init=1e-3, bound_push=1e-6, bound_frac=1e-6, slack_bound_push=1e-6, slack_bound_frac=1e-6)
# How far the foot lands ahead of the CoM [m]. The take-off mirror (0.134 m here) puts the capture point at the
# heel: in MuJoCo the feet rock back 13-15 deg after touch-down. Sweep 0.09-0.12: 0.09 lands with the feet flat.
LAND_OFFSET = 0.09
BW = MASS * G
# Table III (launch): W_rho, W_drho, W_v, W_ddq, W_u.  Table IV (flight): W_rho, W_drho, W_ddq, W_u, W_ankle
WL = dict(rho=1e4, drho=0.1, v=np.array([1e7, 1e6]), ddq=0.2, u=1.0)
# Table IV is (0.1, 1e8, 2e5, 1, 1e8). With those IPOPT does not converge on H1-2 (objective ~2e11, 3000 iters):
# the 1e8 terms cannot be driven to zero (rho differs from the SRMP's, the ankle must move to land flat).
# Same ordering kept (drho, ankle >> ddq >> u), range compressed.
WF = dict(rho=0.1, drho=1e3, ddq=1.0, u=1e-3, ankle=1e3)
SCALE = 1e-6
# NOT part of the paper, off by default (the pipeline follows the paper). Kept only as a recorded finding:
# variable scaling (jerk in launch, flight; q, dq, ddq scaled by 1, 10, 100) took Stage 3 from 7/11 to 11/11 targets
# in about 18 iters. Opt in with solve(..., xscale=XSCALE); scripts/bench_warmstart.py uses it for the comparison.
XSCALE = (1e3, 1e4)
# Box landing, NOT in the paper (user extension, off unless Stage 1 got box=...). BOX = (height, front edge x) [m];
# the edge sits 6 cm behind the heel for the ankle landing at x = 0.50. Smooth step: 1 cm wide, 2 cm before the edge.
# With h 0.2: l 0.5 converges (445 iters); l 0.4 (edge 0.25) is infeasible: the toe passes the edge 6 cm after
# take-off, before the foot can rise 0.1 m; l 0.6 (edge 0.45) hits 1500 iters.
BOX, BOX_W, BOX_MARGIN = (0.10, 0.35), 0.01, 0.02
KEYS = ("p", "v", "a", "rho", "drho", "L", "Ld", "tau", "foot", "dfoot", "toe", "heel")


def _integrator(x, u):
    return ca.vertcat(x[NQ:3 * NQ], u)


def solve(ref: dict, hard_cam: bool = True, guess: dict | None = None, max_iter: int = 3000, verbose: int = 0,
          flight: bool = True, foot_level: str = "pos", skip: tuple = (), ipopt: dict | None = None,
          xscale: tuple | None = None) -> dict:
    """ref = Stage 1 result (joint mode). guess = dict(X, U, Lam, XF, UF) or None for the naive guess."""
    NL, NF = len(ref["t"]) - 1, len(ref["flight"]["t"]) - 1
    T, t_f = ref["T"], ref["t_f"]
    o = ca.Opti()
    rows = {}

    def add(name, c):
        if name in skip:
            return
        n0 = o.ng
        o.subject_to(c)
        rows[name] = rows.get(name, 0) + o.ng - n0

    X, U, Lam = o.variable(3 * NQ, NL + 1), o.variable(NQ, NL + 1), o.variable(3, NL + 1)
    XF, UF = o.variable(3 * NQ, NF + 1), o.variable(NQ, NF + 1)
    n0 = o.ng; trapezoid(o, _integrator, X, U, T / NL); rows["launch defects"] = o.ng - n0
    n0 = o.ng; trapezoid(o, _integrator, XF, UF, t_f / NF); rows["flight defects"] = o.ng - n0
    Q, QF = [q_of(X[:, k]) for k in range(NL + 1)], [q_of(XF[:, k]) for k in range(NF + 1)]

    q0 = crouch(0.8)
    x0 = np.r_[q0, np.zeros(2 * NQ)]
    add("initial state", X[:, 0] == x0)
    J = 0
    for k in range(NL + 1):                                                   # ---------------- launch
        m, lam = Q[k], Lam[:, k]                                              # lam in body weights (and BW m)
        res = m["tau"] / BW - m["Jf"].T @ lam                                 # Eq. 49 (all 7 rows), scaled by 1 / BW
        add("launch base dynamics", res[0:3] == 0)
        add("launch torque limits", o.bounded(-TAU_MAX / BW, res[3:7], TAU_MAX / BW))
        add("joint ranges", o.bounded(Q_MIN, X[3:7, k], Q_MAX))
        if k > 0:                                                             # Eq. 50, flat foot
            add("sole fixed", (m["foot"] if foot_level == "pos" else m["dfoot"]) == 0)
        add("pelvis pitch", o.bounded(-1.0, X[2, k], 1.0))
        if k < NL:        # at k = NL the wrench is fixed to 0: adding these there makes the constraints degenerate
            add("Fz >= 0", lam[1] >= 0)
            add("friction", ca.vertcat(lam[0] - MU * lam[1], -lam[0] - MU * lam[1]) <= 0)
            add("CoP in sole", ca.vertcat(COP_RANGE[0] * lam[1] + lam[2], -lam[2] - COP_RANGE[1] * lam[1]) <= 0)
        if hard_cam and 0 < k < NL:
            add("CAM rate = reference", m["Ld"] == ref["Ld"][k])              # Eq. 47
        J += (WL["rho"] * (m["rho"] - ref["rho"][k]) ** 2 + WL["drho"] * (m["drho"] - ref["drho"][k]) ** 2
              + ca.dot(WL["v"], (m["v"] - ref["v"][k]) ** 2)                  # Eq. 44
              + WL["ddq"] * ca.sumsqr(X[2 * NQ:, k]) + WL["u"] * (ca.sumsqr(U[:, k]) + ca.sumsqr(BW * lam)))   # Eq. 45
    add("take-off CoM velocity", Q[NL]["v"] == ref["v"][-1])                  # Eq. 51
    add("take-off wrench = 0", Lam[:, NL] == 0)

    fl = ref["flight"]
    drho_f = fl["X"][2]
    box = ref.get("box")                                                      # (height, front edge x) or None
    if flight:
        add("phase link", XF[:, 0] == X[:, NL])                                   # Eq. 55
        for k in range(NF + 1):                                                   # ---------------- flight
            m = QF[k]
            if k > 0:     # k = 0 is the take-off node, already constrained in the launch
                add("flight base dynamics", m["tau"][0:3] / BW == 0)
            add("pelvis pitch", o.bounded(-1.0, XF[2, k], 1.0))
            add("flight torque limits", o.bounded(-TAU_MAX / BW, m["tau"][3:7] / BW, TAU_MAX / BW))
            add("joint ranges", o.bounded(Q_MIN, XF[3:7, k], Q_MAX))
            if k < NF and not box:
                add("foot above floor", ca.vertcat(m["toe"][1], m["heel"][1]) >= 0)
            elif k < NF:      # smooth step at the box edge instead of a collision check, as in dai2014.py
                for p in (m["toe"], m["heel"]):
                    add("foot above floor / box", p[1] >= box[0] * 0.5 * (1 + ca.tanh((p[0] - box[1] + BOX_MARGIN) / BOX_W)))
            J += (WF["rho"] * (m["rho"] - fl["X"][1][k]) ** 2 + WF["drho"] * (m["drho"] - drho_f[k]) ** 2   # Eq. 53
                  + WF["ddq"] * ca.sumsqr(XF[2 * NQ + 3:, k]) + WF["u"] * ca.sumsqr(UF[:, k])
                  + WF["ankle"] * XF[NQ + 5, k] ** 2)                             # Eq. 54
        e = QF[NF]
        add("landing: joints at rest", XF[NQ + 3:2 * NQ, NF] == 0)                # Eq. 56
        add("landing: joints at rest", XF[2 * NQ + 3:, NF] == 0)
        # Eq. 56 mirrors the *initial* CoM offset, but Eq. 17 (Stage 1) assumes the *take-off* offset is mirrored;
        # with Eq. 56 as written the jump is about 9 cm short. Mirror the take-off, and fix the distance itself.
        d_land = ref.get("land_offset")                                        # None = mirror the take-off
        add("landing: foot ahead of CoM",
            e["foot"][0] - e["p"][0] == (Q[NL]["p"][0] - Q[NL]["foot"][0] if d_land is None else d_land))
        add("landing: jump distance", e["foot"][0] == ref["l_jump"])
        add("landing: sole flat on floor", e["foot"][1:3] == ca.vertcat(box[0] if box else 0.0, 0.0))
    o.minimize(SCALE * J)
    if xscale:    # typical magnitudes: q ~1, dq ~10, ddq ~100, jerk ~1e3 (launch) / 1e4 (flight)
        sx = np.r_[np.ones(NQ), 10 * np.ones(NQ), 100 * np.ones(NQ)]
        o.set_linear_scale(X, np.tile(sx[:, None], NL + 1)); o.set_linear_scale(XF, np.tile(sx[:, None], NF + 1))
        o.set_linear_scale(U, xscale[0] * np.ones((NQ, NL + 1))); o.set_linear_scale(UF, xscale[1] * np.ones((NQ, NF + 1)))

    if guess is None:                                                         # naive: hold the crouch
        o.set_initial(X, np.tile(x0[:, None], NL + 1))
        o.set_initial(XF, np.tile(x0[:, None], NF + 1))
        o.set_initial(Lam[1, :], 1.0)
    else:
        for V, n in ((X, "X"), (U, "U"), (Lam, "Lam"), (XF, "XF"), (UF, "UF")):
            o.set_initial(V, guess[n])

    o.solver("ipopt", {"print_time": False, "expand": True}, {"print_level": verbose, "sb": "yes", "max_iter": max_iter, **(ipopt or {})})
    t0 = time.perf_counter()
    try:
        val, ok = o.solve().value, True
    except RuntimeError:
        val, ok = o.debug.value, False
    ms = (time.perf_counter() - t0) * 1e3

    def traj(Qs):
        return {n: np.array([np.ravel(val(m[n])) for m in Qs]).squeeze() for n in KEYS}

    la, fo = traj(Q), traj(QF)
    Xs, Ls = val(X), val(Lam) * BW                                            # back to N and N m
    tau_l = np.array([np.ravel(val(Q[k]["tau"] - BW * Q[k]["Jf"].T @ Lam[:, k]))[3:] for k in range(NL + 1)])
    cop = np.where(Ls[1] > 1.0, -Ls[2] / np.maximum(Ls[1], 1.0), np.nan)
    return dict(ok=ok, status=o.stats()["return_status"], iters=o.stats()["iter_count"], ms=ms, nx=o.nx, ng=o.ng,
                cost=float(val(o.f)), rows=rows, T=T, t_f=t_f,
                t=ref["t"], tF=T + fl["t"], X=Xs, U=val(U), Lam=Ls, XF=val(XF), UF=val(UF),
                la=la, fl=fo, tau_l=tau_l, tau_f=fo["tau"][:, 3:], cop=cop, ref=ref)


def save(r: dict, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    np.savez(path, t=r["t"], tF=r["tF"], X=r["X"], XF=r["XF"], Lam=r["Lam"], tau_l=r["tau_l"], tau_f=r["tau_f"])


def report(r: dict) -> None:
    ref, la, fl = r["ref"], r["la"], r["fl"]
    print(f"[stage3] ok={r['ok']} ({r['status']}) iters {r['iters']} {r['ms'] / 1e3:.2f} s | {r['nx']} variables, {r['ng']} constraint rows")
    tmax = np.abs(np.vstack([r["tau_l"], r["tau_f"]])).max(axis=0)
    print("  peak torque / limit (L+R sum): " + ", ".join(
        f"{n} {t:.0f}/{m:.0f}" for n, t, m in zip(("hip", "knee", "ankle", "shoulder"), tmax, TAU_MAX)))
    print(f"  launch: CoP in [{np.nanmin(r['cop']):.3f}, {np.nanmax(r['cop']):.3f}] m (limits {COP_RANGE}) | "
          f"peak Fz {r['Lam'][1].max() / (MASS * G):.2f} BW | sole drift: max |v| {np.abs(la['dfoot']).max():.1e}")
    print(f"  take-off: v ({la['v'][-1, 0]:.3f}, {la['v'][-1, 1]:.3f}) m/s (ref {ref['v'][-1, 0]:.3f}, {ref['v'][-1, 1]:.3f}) | "
          f"L {la['L'][-1]:.2f} (ref {ref['L'][-1]:.2f}) | rho {la['rho'][-1]:.2f} (ref {ref['rho'][-1]:.2f})")
    print(f"  tracking RMS: launch v {np.sqrt(np.mean((la['v'] - ref['v']) ** 2)):.3f} m/s, "
          f"rho {np.sqrt(np.mean((la['rho'] - ref['rho']) ** 2)):.2f} kg m^2 | "
          f"flight rho {np.sqrt(np.mean((fl['rho'] - ref['flight']['X'][1]) ** 2)):.2f} kg m^2 | "
          f"flight L drift {np.ptp(fl['L']):.1e}")
    print(f"  landing: foot - CoM x = {fl['foot'][-1, 0] - fl['p'][-1, 0]:.3f} m (take-off offset {la['p'][-1, 0] - la['foot'][-1, 0]:.3f}) | "
          f"foot x {fl['foot'][-1, 0]:.3f} m = jump distance (target {ref['l_jump']}) | sole z, pitch {fl['foot'][-1, 1]:.1e}, {fl['foot'][-1, 2]:.1e} | "
          f"pelvis pitch {np.degrees(r['XF'][2, -1]):.1f} deg")


def plot(r: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref, la, fl, t, tF = r["ref"], r["la"], r["fl"], r["t"], r["tF"]
    fig, ax = plt.subplots(2, 4, figsize=(17, 7))
    names = ("hip", "knee", "ankle", "shoulder")
    for i, n in enumerate(names):
        ax[0, 0].plot(np.r_[t, tF], np.degrees(np.r_[r["X"][3 + i], r["XF"][3 + i]]), label=n)
    ax[0, 0].plot(np.r_[t, tF], np.degrees(np.r_[r["X"][2], r["XF"][2]]), "k--", label="pelvis pitch")
    ax[0, 0].set_title("angles [deg]"); ax[0, 0].legend(fontsize=7)
    for i, n in enumerate(names):
        ax[0, 1].plot(np.r_[t, tF], np.r_[r["tau_l"][:, i], r["tau_f"][:, i]] / TAU_MAX[i], label=n)
    ax[0, 1].axhline(1, color="k", ls=":"); ax[0, 1].axhline(-1, color="k", ls=":")
    ax[0, 1].set_title("torque / limit"); ax[0, 1].legend(fontsize=7)
    ax[0, 2].plot(t, r["Lam"][1] / (MASS * G), label="Fz"); ax[0, 2].plot(t, r["Lam"][0] / (MASS * G), label="Fx")
    ax[0, 2].plot(t, ref["fz"] / (MASS * G), "--", label="Fz Stage 1"); ax[0, 2].set_title("GRF [BW]"); ax[0, 2].legend(fontsize=7)
    ax[0, 3].plot(t, r["cop"], label="CoP"); ax[0, 3].plot(t, ref["cop"], "--", label="Stage 1")
    for c in COP_RANGE:
        ax[0, 3].axhline(c, color="k", ls=":")
    ax[0, 3].set_title("CoP x [m]"); ax[0, 3].legend(fontsize=7)
    for j, (lab, c) in enumerate((("vx", 0), ("vz", 1))):
        ax[1, 0].plot(np.r_[t, tF], np.r_[la["v"][:, c], fl["v"][:, c]], label=lab)
        ax[1, 0].plot(t, ref["v"][:, c], "--", label=lab + " ref")
    ax[1, 0].set_title("CoM velocity [m/s]"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].plot(np.r_[t, tF], np.r_[la["L"], fl["L"]], label="L"); ax[1, 1].plot(t, ref["L"], "--", label="ref")
    ax[1, 1].set_title("CAM L [kg m^2/s]"); ax[1, 1].legend(fontsize=7)
    ax[1, 2].plot(np.r_[t, tF], np.r_[la["rho"], fl["rho"]], label="rho")
    ax[1, 2].plot(np.r_[t, tF], np.r_[ref["rho"], ref["flight"]["X"][1]], "--", label="ref")
    ax[1, 2].set_title("inertia rho [kg m^2]"); ax[1, 2].legend(fontsize=7)
    ax[1, 3].plot(np.r_[la["p"][:, 0], fl["p"][:, 0]], np.r_[la["p"][:, 1], fl["p"][:, 1]], label="CoM")
    ax[1, 3].plot(fl["toe"][:, 0], fl["toe"][:, 1], label="toe"); ax[1, 3].plot(fl["heel"][:, 0], fl["heel"][:, 1], label="heel")
    ax[1, 3].set_title("paths in x-z [m]"); ax[1, 3].legend(fontsize=7); ax[1, 3].set_xlabel("x [m]")
    for a in ax.ravel()[:-1]:
        a.axvline(r["T"], color="gray", lw=0.8); a.set_xlabel("t [s]")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout(); path.parent.mkdir(exist_ok=True); fig.savefig(path, dpi=120)


if __name__ == "__main__":
    from humanoid_jump.stages import stage2_qp as s2

    nums = [float(a) for a in sys.argv[1:] if not a.startswith("--")]
    h_jump, l_jump = nums if len(nums) == 2 else (0.10, 0.40)
    box = BOX if "--box" in sys.argv else None                                # e.g. 0.2 0.5 --box -> stage3_box.npz
    ref = s1.launch(h_jump, l_jump, joint=True, land_offset=LAND_OFFSET, box=box)
    assert ref["ok"]
    print(f"[stage1] iters {ref['iters']} {ref['ms'] / 1e3:.2f} s | t_sl {ref['T']:.3f} s, t_f {ref['t_f']:.3f} s")
    total, r = ref["ms"], None
    if "--naive" not in sys.argv:
        g = s2.solve(ref)
        r = solve(ref, guess=g, ipopt=WARM, max_iter=1500)
        total += g["ms"] + r["ms"]
        print(f"[stage2] {g['ms'] / 1e3:.2f} s | Stage 3 from the Stage 2 guess: {r['status']}, {r['iters']} iters, {r['ms'] / 1e3:.1f} s")
    if r is None or not r["ok"]:
        r = solve(ref)
        total += r["ms"]
        print(f"Stage 3 from the naive guess: {r['status']}, {r['iters']} iters, {r['ms'] / 1e3:.1f} s")
    report(r)
    print(f"Total optimization time: {total / 1e3:.1f} s")
    for k, v in r["rows"].items():
        print(f"  {v:5d}  {k}")
    name = "stage3_box" if box else "stage3_nlp"
    plot(r, ROOT / "results" / f"{name}.png")
    save(r, ROOT / "results" / f"{name}.npz")
    assert r["ok"], "Stage 3 did not converge"
    print("Phase 3 gate: converged")
