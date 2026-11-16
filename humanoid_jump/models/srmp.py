"""SRMP model (4 DOF), paper Sec. III-A. Used by Stage 1.

    q = [th_l leg angle from vertical (+ = forward), phi leg length (support point -> CoM),
         th_pl barbell angle relative to the leg, r barbell radius (the paper's varphi)]
    F = [tau_l, F_phi, tau_pl, F_r]                                    (Eq. 3)

Massless leg, support point fixed at the origin, two point masses m/2 at CoM +- r.
Pitch is positive about +y (x forward, z up), so a forward lean and a forward spin are positive.
The paper never writes M and H of Eq. 2; here they come from the Lagrangian via CasADi.

    python -m humanoid_jump.models.srmp      # self-check: energy conservation + free fall at F = 0
"""

from __future__ import annotations

import casadi as ca
import numpy as np

G = 9.81
# H1-2 handless, from scripts/srmp_numbers.py (support point = ankle projected on the floor)
MASS = 66.98
PHI_RANGE = (0.60, 0.93)       # CoM height: deep crouch (a = 1.0) .. standing, arms down [m]
RHO_RANGE = (6.1, 13.6)        # pitch inertia: deep crouch, arms down .. stretched, arms up [kg m^2]
COP_RANGE = (-0.066, 0.154)    # sole x in [-0.086, 0.174] minus a 2 cm margin [m]


def _build(m: float = MASS):
    q, dq, ddq = ca.SX.sym("q", 4), ca.SX.sym("dq", 4), ca.SX.sym("ddq", 4)
    th, phi, r = q[0], q[1], q[3]
    w = dq[0] + dq[2]                                    # barbell absolute angular velocity

    T = 0.5 * m * (dq[1] ** 2 + phi**2 * dq[0] ** 2) + 0.5 * m * (dq[3] ** 2 + r**2 * w**2)
    V = m * G * phi * ca.cos(th)
    Lag = T - V
    mom = ca.jacobian(Lag, dq).T
    M = ca.jacobian(mom, dq)
    H = ca.jacobian(mom, q) @ dq - ca.jacobian(Lag, q).T
    F = M @ ddq + H                                      # Eq. 2

    def ddt(e):                                          # time derivative of e(q, dq)
        return ca.jacobian(e, q) @ dq + ca.jacobian(e, dq) @ ddq

    p = ca.vertcat(phi * ca.sin(th), phi * ca.cos(th))   # CoM (x, z)
    v = ca.jacobian(p, q) @ dq
    a = ddt(v)
    rho = m * r**2                                       # Eq. 4 with m = 2 m_p
    L = rho * w                                          # Eq. 13
    Ld = ddt(L)

    out = dict(p=p, v=v, a=a, rho=rho, drho=ddt(rho), ddrho=2 * m * (dq[3] ** 2 + r * ddq[3]), L=L, Ld=Ld, F=F, E=T + V)
    quant = ca.Function("srmp", [q, dq, ddq], list(out.values()), ["q", "dq", "ddq"], list(out))
    Fin = ca.SX.sym("F", 4)
    fwd = ca.Function("srmp_fwd", [q, dq, Fin], [ca.solve(M, Fin - H)])
    return quant, fwd


quantities, forward_dynamics = _build()


def q_of(x):
    """Split a (pos, vel, acc) state vector (Eq. 9) and evaluate all model quantities."""
    return quantities(q=x[0:4], dq=x[4:8], ddq=x[8:12])


if __name__ == "__main__":
    # Passive motion (F = 0): the leg tips over, the barbell spins and flies outward. Energy must stay.
    q, dq = np.array([0.1, 0.8, 0.0, 0.33]), np.array([0.3, 0.5, 1.0, 0.0])
    z4 = np.zeros(4)
    E0 = float(quantities(q=q, dq=dq, ddq=z4)["E"])
    h = 1e-4
    f = lambda y: np.concatenate([y[4:], forward_dynamics(y[:4], y[4:], z4).full().ravel()])  # noqa: E731
    y = np.concatenate([q, dq])
    for _ in range(3000):                                # 0.3 s, RK4
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    E1 = float(quantities(q=y[:4], dq=y[4:], ddq=z4)["E"])
    dd = forward_dynamics(q, dq, z4).full().ravel()
    o = quantities(q=q, dq=dq, ddq=dd)
    print(f"[srmp] energy {E0:.4f} -> {E1:.4f} J (rel. drift {abs(E1 - E0) / E0:.1e}) | "
          f"F = 0: CoM acc {o['a'].full().ravel()} m/s^2, Ldot {float(o['Ld']):.1e}")
    assert abs(E1 - E0) / E0 < 1e-8
    assert np.allclose(o["a"].full().ravel(), [0, -G], atol=1e-9) and abs(float(o["Ld"])) < 1e-9
