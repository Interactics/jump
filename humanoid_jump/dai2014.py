"""Dai, Valenzuela, Tedrake 2014: whole-body planning with centroidal dynamics and full kinematics, on H1-2.

One NLP (Sec. II-D) on the full 3D model (models/full.py, 27 joints). Decision variables at every knot k
(Eq. 6): q, v, CoM r, r_dot, r_ddot, contact points c_j, contact forces F_j, centroidal angular momentum h, h_dot,
and the interval durations dt[k]. Constraints (Eq. 7):
  7a  m r_ddot = sum F_j + m g             7e  h[k] - h[k-1] = h_dot[k] dt[k]
  7b  h_dot = sum (c_j - r) x F_j           7f  r[k] - r[k-1] = (r_dot[k] + r_dot[k-1]) / 2 dt[k]
  7c  h = A(q) v                            7g  r_dot[k] - r_dot[k-1] = r_ddot[k] dt[k]
  7d  q[k] - q[k-1] = v[k] dt[k]            7h  r = com(q),  7i  c_j = p_j(q),  7j  c_j on the floor when in contact
  7k  joint ranges, joint velocity limits, friction cone.
Contact schedule fixed as in Sec. III-A: heels take off, toes take off, toes touch, heels touch (4 corners per foot).

Choices the paper leaves open (G) or that H1-2 / the tools force (R), see CLAUDE.md "Dai 2014":
  * IPOPT, not SNOPT (not in this CasADi build).                                                      (R)
  * Point contacts, so tau_j = 0; friction cone as a pyramid |Fx|, |Fy| <= mu Fz, mu = 0.7 (as Stage 3).  (G)
  * Task, same as the T-ASE pipeline: take-off CoM vz = sqrt(2 g h) at the last toe-stance knot (T-ASE Eq. 18),
    toes touch down at x = l. Start at rest in crouch(0.8); end at rest (v, r_dot, r_ddot, h, h_dot = 0).      (G)
  * Cost weights, q_nom = the start posture, dt multiplying the whole running cost (as Eq. 10), dt bounds.   (G)
  * Non-contact corners stay above the floor (c_z >= 0) instead of GJK collision avoidance (Eqs. 3-5).    (G)
  * No joint torque limits in the NLP by default (Sec. II-A assumes sufficient torque); torques are checked
    afterwards. torque_limits=True / --torque-limits adds |tau_joint| <= tau_max at every knot, with tau from
    inverse dynamics (a[k] = (v[k] - v[k-1]) / dt[k]). Off by default: it is our deviation, needed on H1-2.  (R)
  * Naive initial guess: hold the crouch (the paper gives no guess; same as the T-ASE Phase 3 baseline).

    python -m humanoid_jump.dai2014 [h l]                    # default h = 0.10 m, l = 0.40 m -> results/dai2014.npz, .png
    python -m humanoid_jump.dai2014 [h l] --torque-limits    # -> results/dai2014_tau.npz, .png
    python -m humanoid_jump.dai2014 h l y                    # sideways target y (e.g. diagonal) -> results/dai2014_y<y>.npz
    python -m humanoid_jump.dai2014 h l --box                # onto a box (BOX: 0.10 m high, front edge at x 0.25 m) -> results/dai2014_box.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

from humanoid_jump.models.full import CORNERS, G, HEEL, MASS, NQ, Q_MAX, Q_MIN, TAU_MAX, TOE, V_MAX, crouch, \
    joint_tau, joint_torques, kin

ROOT = Path(__file__).resolve().parents[1]
BW = MASS * G
MU = 0.7
NC = len(CORNERS)
ALL = list(range(NC))
# (name, active contact points, knots, nominal duration [s]); the flight duration is set from h
PHASES = [("stance", ALL, 20, 0.80), ("toe", TOE, 6, 0.10), ("flight", [], 12, None), ("toe land", TOE, 4, 0.05),
          ("land", ALL, 12, 0.50)]
DT_RANGE = (0.25, 2.5)                       # dt[k] bounds as a multiple of the phase's nominal dt
W = dict(q=1.0, rpy=10.0, v=1e-2, a=1e-2, F=1.0)
PITCH_MAX = 1.2                              # keep the rpy chart away from its singularity at pitch = +-pi/2


def schedule(h_jump: float):
    """Per knot: phase index and active contact set; per phase: nominal duration."""
    t_f = 2 * np.sqrt(2 * G * h_jump) / G
    ph, act, dur = [], [], []
    for i, (_, a, n, d) in enumerate(PHASES):
        ph += [i] * n
        act += [a] * n
        dur.append(t_f if d is None else d)
    return np.array(ph), act, np.array(dur)


def solve(h_jump: float = 0.10, l_jump: float = 0.40, max_iter: int = 3000, verbose: int = 0, ipopt: dict | None = None,
          torque_limits: bool = False, y_jump: float = 0.0, box: tuple | None = None, flat: bool = False) -> dict:
    ph, act, dur = schedule(h_jump)
    if flat:    # ponytail: NOT in the paper, comparison only; push-off on all 8 corners (no toe-only phase), landing unchanged
        act = [ALL if p == 1 else a for p, a in zip(ph, act)]
    N = len(ph)
    n_ph = np.bincount(ph)
    dt_nom = dur[ph] / n_ph[ph]                                 # nominal dt of the interval ending at knot k
    k_to = int(np.where(ph == 1)[0][-1])                        # last toe-stance knot = take-off
    k_td = int(np.where(ph == 3)[0][0])                         # first toe-landing knot = touch-down
    k_hs = int(np.where(ph == 4)[0][0])                         # heel strike

    o = ca.Opti()
    q, v = o.variable(NQ, N), o.variable(NQ, N)
    r, rd, rdd = o.variable(3, N), o.variable(3, N), o.variable(3, N)
    h, hd = o.variable(3, N), o.variable(3, N)
    c = o.variable(3 * NC, N)                                   # contact point positions (Eq. 6 lists them as variables)
    dtv = o.variable(1, N - 1)
    dt = ca.horzcat(dtv[0], dtv)                                # dt[k] = duration of (k-1, k]; dt[0] := dt[1] (cost of knot 0)
    Fv = {k: o.variable(3, len(act[k])) for k in range(N) if act[k]}   # forces in body weights, active points only
    rows = {}

    def add(name, e):
        n0 = o.ng
        o.subject_to(e)
        rows[name] = rows.get(name, 0) + o.ng - n0

    def F(k):                                                   # 3 x NC, zero for inactive points
        cols = [ca.MX.zeros(3, 1)] * NC
        for i, j in enumerate(act[k]):
            cols[j] = Fv[k][:, i]
        return ca.horzcat(*cols)

    box_h, box_x = box if box else (0.0, np.inf)                # box: (height, front edge x); landing on its top
    com, hq, cq = kin.map(N)(q, v)                              # 3 x N, 3 x N, 3 x (NC N)
    gvec = np.array([0.0, 0.0, -G])
    J = 0
    for k in range(N):
        Ck = ca.reshape(c[:, k], 3, NC)
        Fk = F(k)
        add("7a linear momentum", rdd[:, k] - G * ca.sum2(Fk) - gvec == 0)            # m r_ddot = BW sum F + m g, / m
        add("7b angular momentum", hd[:, k] / BW - sum(ca.cross(Ck[:, j] - r[:, k], Fk[:, j]) for j in act[k]) == 0
            if act[k] else hd[:, k] == 0)
        add("7c h = A(q) v", h[:, k] - hq[:, k] == 0)
        add("7h r = com(q)", r[:, k] - com[:, k] == 0)
        add("7i c = p(q)", c[:, k] - ca.vec(cq[:, NC * k:NC * (k + 1)]) == 0)
        for j in range(NC):
            if j in act[k]:
                add("7j on the floor", Ck[2, j] == (box_h if k >= k_td else 0.0))
                if k > 0 and j in act[k - 1]:
                    add("7j no sliding", Ck[0:2, j] - ca.reshape(c[:, k - 1], 3, NC)[0:2, j] == 0)
            else:
                add("above the floor", Ck[2, j] >= 0)
                if box:                                         # clear the box edge (smooth step instead of GJK, Eqs. 3-5)
                    hb = box_h + (BOX_CLEAR if ph[k] == 2 else 0.0)  # extra clearance in flight only
                    add("above the box", Ck[2, j] >= hb * 0.5 * (1 + ca.tanh((Ck[0, j] - box_x + BOX_MARGIN) / BOX_W)))
        if act[k]:
            f = Fv[k]
            add("7k friction cone", f[2, :] >= 0)
            add("7k friction cone", ca.vec(ca.vertcat(f[0, :] - MU * f[2, :], -f[0, :] - MU * f[2, :],
                                                      f[1, :] - MU * f[2, :], -f[1, :] - MU * f[2, :])) <= 0)
        add("7k joint ranges", o.bounded(Q_MIN, q[6:, k], Q_MAX))
        add("7k joint velocities", o.bounded(-V_MAX, v[6:, k], V_MAX))
        add("base pitch (rpy chart)", o.bounded(-PITCH_MAX, q[4, k], PITCH_MAX))
        if k > 0:
            add("7d q integration", q[:, k] - q[:, k - 1] - v[:, k] * dt[k] == 0)
            add("7e h integration", h[:, k] - h[:, k - 1] - hd[:, k] * dt[k] == 0)
            add("7f r integration", r[:, k] - r[:, k - 1] - (rd[:, k] + rd[:, k - 1]) / 2 * dt[k] == 0)
            add("7g r_dot integration", rd[:, k] - rd[:, k - 1] - rdd[:, k] * dt[k] == 0)
            add("dt bounds", o.bounded(DT_RANGE[0] * dt_nom[k], dt[k], DT_RANGE[1] * dt_nom[k]))
        run = (W["q"] * ca.sumsqr(q[6:, k] - QNOM[6:]) + W["rpy"] * ca.sumsqr(q[3:6, k]) + W["v"] * ca.sumsqr(v[:, k])
               + W["a"] * ca.sumsqr(rdd[:, k]) + (W["F"] * ca.sumsqr(Fv[k]) if act[k] else 0))
        J += run * dt[k]                                                           # Eq. 6 / Eq. 10

    if torque_limits:                                           # our deviation (R), off by default
        acc = ca.horzcat(ca.MX.zeros(NQ, 1), (v[:, 1:] - v[:, :-1]) / ca.repmat(dtv, NQ, 1))
        tau = joint_tau.map(N)(q, v, acc, BW * ca.horzcat(*[F(k) for k in range(N)]))
        add("joint torque limits (ours)", o.bounded(-1, ca.vec(tau / np.tile(TAU_MAX[:, None], (1, N))), 1))
    C0 = np.array(kin(QNOM, np.zeros(NQ))[2])                   # start contact points (3 x NC)
    add("start at rest", ca.vertcat(q[:, 0] - QNOM, v[:, 0], rd[:, 0], rdd[:, 0], h[:, 0]) == 0)
    add("take-off CoM vz", rd[2, k_to] == np.sqrt(2 * G * h_jump))
    Ctd = ca.reshape(c[:, k_td], 3, NC)
    add("toes touch down at (x, y) = (l, y_jump)", ca.vec(Ctd[0:2, TOE] - (C0[0:2, TOE] + np.array([[l_jump], [y_jump]]))) == 0)
    add("end at rest", ca.vertcat(v[:, N - 1], rd[:, N - 1], rdd[:, N - 1], h[:, N - 1], hd[:, N - 1]) == 0)
    o.minimize(J)

    # naive guess: hold the crouch
    o.set_initial(q, np.tile(QNOM[:, None], N))
    o.set_initial(r, np.tile(np.array(kin(QNOM, np.zeros(NQ))[0]), (1, N)))
    o.set_initial(c, np.tile(C0.T.reshape(-1, 1), (1, N)))
    o.set_initial(dtv, dt_nom[None, 1:])
    if box:                                                     # landing knots: the same crouch, moved onto the box
        shift = np.array([l_jump, y_jump, box_h])
        o.set_initial(q[0:3, k_td:], np.tile((QNOM[0:3] + shift)[:, None], N - k_td))
        o.set_initial(r[:, k_td:], np.tile(np.array(kin(QNOM, np.zeros(NQ))[0]) + shift[:, None], (1, N - k_td)))
        o.set_initial(c[:, k_td:], np.tile((C0 + shift[:, None]).T.reshape(-1, 1), (1, N - k_td)))
    for k, f in Fv.items():
        o.set_initial(f[2, :], 1.0 / f.shape[1])
    o.solver("ipopt", {"print_time": False, "expand": True},
             {"print_level": verbose, "sb": "yes", "max_iter": max_iter, **(ipopt or {})})
    t0 = time.perf_counter()
    try:
        val, ok = o.solve().value, True
    except RuntimeError:
        val, ok = o.debug.value, False
    ms = (time.perf_counter() - t0) * 1e3

    dts = np.r_[0.0, np.ravel(val(dtv))]
    out = dict(ok=ok, torque_limits=torque_limits, status=o.stats()["return_status"], iters=o.stats()["iter_count"], ms=ms, nx=o.nx, ng=o.ng, rows=rows,
               cost=float(val(o.f)), h_jump=h_jump, l_jump=l_jump, y_jump=y_jump, box=np.array(box or []), phase=ph, k_to=k_to, k_td=k_td, k_hs=k_hs,
               t=np.cumsum(dts), q=val(q), v=val(v), r=val(r), rd=val(rd), rdd=val(rdd), h=val(h), c=val(c),
               F=np.stack([np.array(val(F(k))) * BW if act[k] else np.zeros((3, NC)) for k in range(N)]))
    out.update(_check(out))
    return out


QNOM = crouch(0.8)
BOX = (0.10, 0.25)          # --box: height, front edge x [m]. Earlier: (0.20, 0.25) with l 0.55 and margin 0.08 stood;
                            # (0.10, 0.35) with l 0.50 (the [T] box) fell
BOX_MARGIN, BOX_W, BOX_CLEAR = 0.04, 0.01, 0.05   # feet up 4 cm before the edge (8 cm: the toe starts only 7.6 cm
                                                  # from a 0.25 edge), step width, 5 cm above the top in flight
# (2 cm / 0 cm clipped the edge in MuJoCo: the executed jump is lower than planned)


def _check(s: dict) -> dict:
    """Post-hoc: joint torques by inverse dynamics (Sec. II-A) and how well the discrete plan satisfies the full
    dynamics (rows 0-5 of the RNEA residual, which Eqs. 7a-7c only enforce through r and h)."""
    t, q, v, F = s["t"], s["q"], s["v"], s["F"]
    N = len(t)
    a = np.zeros_like(v)
    a[:, 1:] = np.diff(v, axis=1) / np.diff(t)                  # backward Euler, consistent with Eq. 7d
    tau = np.array([joint_torques(q[:, k], v[:, k], a[:, k], F[k].T) for k in range(N)])
    com_mid = np.array([np.ravel(kin(0.5 * (q[:, k] + q[:, k - 1]), np.zeros(NQ))[0]) for k in range(1, N)])
    r_mid = 0.5 * (s["r"][:, 1:] + s["r"][:, :-1]).T
    return dict(a=a, tau=tau[:, 6:], base_residual=np.abs(tau[1:, :6]).max(axis=0),
                com_gap_mid=np.abs(com_mid - r_mid).max())


def report(s: dict) -> None:
    t, ph = s["t"], s["phase"]
    names = [p[0] for p in PHASES]
    print(f"[dai2014{' + torque limits' if s['torque_limits'] else ''}] ok={s['ok']} ({s['status']}) iters {s['iters']} {s['ms'] / 1e3:.2f} s | {s['nx']} variables, "
          f"{s['ng']} constraint rows | cost {s['cost']:.3f}")
    print("  phase durations: " + ", ".join(f"{n} {t[np.where(ph == i)[0][-1]] - (t[np.where(ph == i)[0][0] - 1] if i else 0):.3f} s"
                                         for i, n in enumerate(names)))
    k_to, k_td = s["k_to"], s["k_td"]
    print(f"  take-off: CoM ({s['r'][0, k_to]:.3f}, {s['r'][2, k_to]:.3f}) m, v ({s['rd'][0, k_to]:.3f}, {s['rd'][2, k_to]:.3f}) m/s, "
          f"L_y {s['h'][1, k_to]:.2f} kg m^2/s | apex CoM z {s['r'][2].max():.3f} m | "
          f"touch-down toe x {np.mean(s['c'].reshape(NC, 3, -1)[TOE, 0, k_td]):.3f} m (target {s['l_jump'] + 0.174:.3f})")
    print(f"  peak vertical force {s['F'][:, 2, :].sum(axis=1).max() / BW:.2f} BW (launch), "
          f"{s['F'][k_td:, 2, :].sum(axis=1).max() / BW:.2f} BW (landing)")
    from humanoid_jump.models.full import JOINT_NAMES
    peak = np.abs(s["tau"]).max(axis=0) / TAU_MAX
    over = [f"{JOINT_NAMES[i].replace('_joint', '')} {peak[i]:.2f}" for i in np.argsort(-peak)[:6]]
    print(f"  peak |tau| / limit (not constrained, Sec. II-A): {', '.join(over)} | joints over the limit: {int((peak > 1).sum())}")
    vpeak = np.abs(s["v"][6:]).max(axis=1) / V_MAX
    print(f"  peak |v| / limit {vpeak.max():.2f} | full-dynamics residual on the base rows (should be 0): "
          f"force {s['base_residual'][:3].max():.1f} N, torque {s['base_residual'][3:].max():.1f} N m | "
          f"com(q) vs r at mid-interval: {1e3 * s['com_gap_mid']:.1f} mm")


def save(s: dict, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    np.savez(path, **{k: s[k] for k in ("t", "q", "v", "a", "r", "rd", "h", "c", "F", "tau", "phase", "k_to", "k_td", "k_hs",
                                         "h_jump", "l_jump", "y_jump", "box")})


def plot(s: dict, path: Path) -> None:
    """Same panels as the T-ASE Stage 3 plot (stages/stage3_nlp.plot), for side-by-side slides. Left-side joints;
    gray line = take-off, dashed = touch-down. CoP and GRF over the launch (as Stage 3); no reference curves ([D] has none)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pinocchio as pin

    from humanoid_jump.models.full import DATA, JOINT_NAMES, MODEL
    from humanoid_jump.sim.robot import FOOT_X
    t, k_to, k_td = s["t"], int(s["k_to"]), int(s["k_td"])
    cc = s["c"].reshape(NC, 3, -1)
    L = slice(0, k_to + 1)                                       # launch
    fig, ax = plt.subplots(2, 4, figsize=(17, 7))
    for n, lab in (("left_hip_pitch", "hip"), ("left_knee", "knee"), ("left_ankle_pitch", "ankle"), ("left_shoulder_pitch", "shoulder")):
        i = JOINT_NAMES.index(n + "_joint")
        ax[0, 0].plot(t, np.degrees(s["q"][6 + i]), label=lab)
        ax[0, 1].plot(t, s["tau"][:, i] / TAU_MAX[i], label=lab)
    ax[0, 0].plot(t, np.degrees(s["q"][4]), "k--", label="pelvis pitch")
    ax[0, 0].set_title("angles [deg]"); ax[0, 0].legend(fontsize=7)
    ax[0, 1].axhline(1, color="k", ls=":"); ax[0, 1].axhline(-1, color="k", ls=":")
    ax[0, 1].set_title("torque / limit (not constrained, Sec. II-A)"); ax[0, 1].legend(fontsize=7)
    Fz, Fx = s["F"][:, 2, :].sum(axis=1), s["F"][:, 0, :].sum(axis=1)
    ax[0, 2].plot(t[L], Fz[L] / BW, label="Fz"); ax[0, 2].plot(t[L], Fx[L] / BW, label="Fx")
    ax[0, 2].plot(t[L], s["F"][L, 2][:, TOE].sum(axis=1) / BW, "--", label="Fz toes")
    ax[0, 2].set_title("GRF [BW]"); ax[0, 2].legend(fontsize=7)
    ankle_x = cc[HEEL, 0, 0].mean() - FOOT_X[0]                 # start ankle x (feet do not slide in launch)
    cop = (cc[:, 0, L].T * s["F"][L, 2]).sum(axis=1) / np.maximum(Fz[L], 1e-9) - ankle_x
    ax[0, 3].plot(t[L], cop, label="CoP")
    for c in FOOT_X:
        ax[0, 3].axhline(c, color="k", ls=":")
    ax[0, 3].set_title("CoP x [m] (from the ankle; dotted: sole edges)"); ax[0, 3].legend(fontsize=7)
    ax[1, 0].plot(t, s["rd"][0], label="vx"); ax[1, 0].plot(t, s["rd"][2], label="vz")
    ax[1, 0].set_title("CoM velocity [m/s]"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].plot(t, s["h"][1], label="L"); ax[1, 1].set_title("CAM L [kg m^2/s]"); ax[1, 1].legend(fontsize=7)
    rho = []
    for k in range(len(t)):
        pin.ccrba(MODEL, DATA, s["q"][:, k], np.zeros(MODEL.nv))
        rho.append(DATA.Ig.inertia[1, 1])                        # centroidal pitch inertia
    ax[1, 2].plot(t, rho, label="rho"); ax[1, 2].set_title("inertia rho [kg m^2]"); ax[1, 2].legend(fontsize=7)
    ax[1, 3].plot(s["r"][0], s["r"][2], label="CoM")
    ax[1, 3].plot(cc[TOE[:2], 0].mean(axis=0), cc[TOE[:2], 2].mean(axis=0), label="toe")
    ax[1, 3].plot(cc[HEEL[:2], 0].mean(axis=0), cc[HEEL[:2], 2].mean(axis=0), label="heel")
    ax[1, 3].set_title("paths in x-z [m]"); ax[1, 3].legend(fontsize=7); ax[1, 3].set_xlabel("x [m]")
    for a in ax.ravel()[:-1]:
        a.axvline(t[k_to], color="gray", lw=0.8); a.set_xlabel("t [s]")
        if a not in (ax[0, 2], ax[0, 3]):
            a.axvline(t[k_td], color="gray", lw=0.8, ls="--")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout(); path.parent.mkdir(exist_ok=True); fig.savefig(path, dpi=120)


if __name__ == "__main__":
    nums = [float(a) for a in sys.argv[1:] if not a.startswith("--")]
    h_jump, l_jump, y_jump = (nums + [0.0])[:3] if len(nums) >= 2 else (0.10, 0.40, 0.0)
    tl = "--torque-limits" in sys.argv
    bx = BOX if "--box" in sys.argv else None
    fl = "--flat" in sys.argv
    s = solve(h_jump, l_jump, verbose=5 if "--verbose" in sys.argv else 0, torque_limits=tl, y_jump=y_jump, box=bx, flat=fl)
    report(s)
    for k, n in s["rows"].items():
        print(f"  {n:5d}  {k}")
    tag = ("dai2014_tau" if tl else "dai2014") + (f"_y{y_jump}" if y_jump else "") + ("_box" if bx else "") + ("_flat" if fl else "")
    plot(s, ROOT / "results" / f"{tag}.png")
    save(s, ROOT / "results" / f"{tag}.npz")
    assert s["ok"], "did not converge"
