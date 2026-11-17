"""7-DOF symmetric planar model of H1-2 (paper Sec. III-C). Used by Stage 3.

    q = [x, z, pitch (pelvis), hip_pitch, knee, ankle_pitch, shoulder_pitch]

Built from the URDF with Pinocchio: planar root joint (PX, PZ, RY), all other joints locked except the
four pitch joints per side, and left = right enforced by a constant matrix S (11 -> 7 velocities).
So masses and inertias are the real ones; nothing is merged by hand. Joint torques are left + right sums.
Pitch is positive about +y (same convention as the SRMP).

    python -m humanoid_jump.models.symmetric     # self-check against the MuJoCo model
"""

from __future__ import annotations

import os
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from humanoid_jump.sim.robot import ARMS, FOOT_X, SOLE_Z

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "third_party/unitree_ros/robots/h1_2_description/h1_2_handless.urdf"
JOINTS = ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch")
NQ = 7
G = 9.81
# Sum of left + right actuator limits [N m] and joint ranges [rad] (MJCF ctrlrange / range)
TAU_MAX = np.array([400.0, 600.0, 120.0, 80.0])
Q_MIN = np.array([-3.14, -0.12, -0.897, -3.14])
Q_MAX = np.array([2.5, 2.19, 0.524, 1.57])


def _load():
    root = pin.JointModelComposite()
    for j in (pin.JointModelPX(), pin.JointModelPZ(), pin.JointModelRY()):
        root.addJoint(j)
    fd = os.dup(2)                      # urdfdom prints a harmless visual-material error per link
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    try:
        full = pin.buildModelFromUrdf(str(URDF), root)
    finally:
        os.dup2(fd, 2)
    keep = [f"{s}_{j}_joint" for s in ("left", "right") for j in JOINTS]
    q0 = pin.neutral(full)
    for n, v in ARMS.items():           # locked joints keep the arm posture used in the simulator
        q0[full.joints[full.getJointId(n)].idx_q] = v
    lock = [full.getJointId(n) for n in full.names[2:] if n not in keep]
    model = pin.buildReducedModel(full, lock, q0)
    S = np.zeros((model.nv, NQ))
    S[0:3, 0:3] = np.eye(3)
    for i, j in enumerate(JOINTS):
        for s in ("left", "right"):
            S[model.joints[model.getJointId(f"{s}_{j}_joint")].idx_v, 3 + i] = 1.0
    return model, S


MODEL, S = _load()
MASS = pin.computeTotalMass(MODEL)


def _build():
    cm = cpin.Model(MODEL)
    cd = cm.createData()
    q, dq, ddq = ca.SX.sym("q", NQ), ca.SX.sym("dq", NQ), ca.SX.sym("ddq", NQ)
    Q, V, A = S @ q, S @ dq, S @ ddq

    def ddt(e):                          # time derivative of e(q, dq)
        return ca.jacobian(e, q) @ dq + ca.jacobian(e, dq) @ ddq

    tau = S.T @ cpin.rnea(cm, cd, Q, V, A)                       # generalized forces needed for (q, dq, ddq)
    cpin.computeCentroidalMomentumTimeVariation(cm, cd, Q, V, A)
    L, Ld = cd.hg.angular[1], cd.dhg.angular[1]
    c = cpin.centerOfMass(cm, cd, Q)
    p = ca.vertcat(c[0], c[2])
    v = ca.jacobian(p, q) @ dq
    a = ddt(v)
    cpin.ccrba(cm, cd, Q, V)
    rho = cd.Ig.inertia[1, 1]                                    # centroidal pitch inertia

    cpin.framesForwardKinematics(cm, cd, Q)
    M = cd.oMf[cm.getFrameId("left_ankle_roll_link")]
    R, t = M.rotation, M.translation

    def sole(x):                                                 # point on the sole, x along the foot
        w = t + R @ ca.vertcat(x, 0, SOLE_Z)
        return ca.vertcat(w[0], w[2])

    foot = ca.vertcat(sole(0.0), ca.atan2(R[0, 2], R[2, 2]))     # (x, z, pitch) of the sole point under the ankle
    Jf = ca.jacobian(foot, q)                                    # lambda = (Fx, Fz, My about that point)
    out = dict(p=p, v=v, a=a, rho=rho, drho=ca.jacobian(rho, q) @ dq, L=L, Ld=Ld, tau=tau, foot=foot, Jf=Jf,
               dfoot=Jf @ dq, toe=sole(FOOT_X[1]), heel=sole(FOOT_X[0]))
    quant = ca.Function("sym7", [q, dq, ddq], list(out.values()), ["q", "dq", "ddq"], list(out))

    # Everything Stage 2's QP needs is affine in ddq: e = J(q, dq) ddq + e0(q, dq)
    def affine(e):
        return ca.jacobian(e, ddq), ca.substitute(e, ddq, ca.SX.zeros(NQ))

    rel = foot - ca.vertcat(p, 0)                                # sole point relative to the CoM (x, z, pitch)
    drel = ca.jacobian(rel, q) @ dq
    lin = dict(p=p, v=v, L=L, rho=rho, drho=out["drho"], foot=foot, dfoot=out["dfoot"], rel=rel, drel=drel)
    for name, e in (("tau", tau), ("a", a), ("Ld", Ld), ("ddrho", ddt(out["drho"])), ("ddfoot", ddt(out["dfoot"])),
                    ("ddrel", ddt(drel))):
        lin["J_" + name], lin[name + "0"] = affine(e)
    linear = ca.Function("sym7_lin", [q, dq], list(lin.values()), ["q", "dq"], list(lin))
    return quant, linear


quantities, linear_terms = _build()


def q_of(x):
    """Split a (pos, vel, acc) state vector (Eq. 42) and evaluate all model quantities."""
    return quantities(q=x[0:NQ], dq=x[NQ:2 * NQ], ddq=x[2 * NQ:3 * NQ])


def crouch(a: float) -> np.ndarray:
    """Flat-footed symmetric crouch (hip = -a, knee = 2a, ankle = -a), sole on the floor, ankle at x = 0."""
    q = np.array([0.0, 0.0, 0.0, -a, 2 * a, -a, 0.0])
    f = quantities(q=q, dq=np.zeros(NQ), ddq=np.zeros(NQ))["foot"].full().ravel()
    q[0], q[1] = -f[0], -f[1]
    return q


if __name__ == "__main__":
    import mujoco

    from humanoid_jump.sim.robot import Robot, build_model

    r = Robot(build_model())
    m, d = r.m, r.d
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
    for i, j in enumerate(JOINTS):       # hard-coded limits must match the MJCF
        k = jid(f"left_{j}_joint")
        act = r.act_names.index(f"left_{j}_joint")
        assert np.isclose(2 * r.tau_max[act], TAU_MAX[i]), (j, r.tau_max[act])
        assert np.allclose(m.jnt_range[k], [Q_MIN[i], Q_MAX[i]], atol=5e-3), (j, m.jnt_range[k])

    rng, err = np.random.default_rng(0), np.zeros(5)
    for _ in range(20):
        q = np.r_[rng.uniform(-0.3, 0.3), rng.uniform(0.7, 1.0), rng.uniform(-0.5, 0.5),
                  rng.uniform(-1.2, 0.2), rng.uniform(0.1, 1.8), rng.uniform(-0.8, 0.4), rng.uniform(-2.0, 1.0)]
        dq = rng.uniform(-2, 2, NQ)
        o = {k: np.ravel(v) for k, v in quantities(q=q, dq=dq, ddq=np.zeros(NQ)).items()}

        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.qpos0
        d.qpos[0:3] = [q[0], 0, q[1]]
        d.qpos[3:7] = [np.cos(q[2] / 2), 0, np.sin(q[2] / 2), 0]
        Rb = np.array([[np.cos(q[2]), 0, np.sin(q[2])], [0, 1, 0], [-np.sin(q[2]), 0, np.cos(q[2])]])
        d.qvel[0:3] = [dq[0], 0, dq[1]]                       # free joint: linear world, angular local
        d.qvel[3:6] = [0, dq[2], 0]
        for n, v in ARMS.items():
            d.qpos[r.qadr[n]] = v
        for i, j in enumerate(JOINTS):
            for sd in ("left", "right"):
                d.qpos[r.qadr[f"{sd}_{j}_joint"]] = q[3 + i]
                d.qvel[m.jnt_dofadr[jid(f"{sd}_{j}_joint")]] = dq[3 + i]
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        c = d.subtree_com[1]
        rho = sum((d.ximat[b].reshape(3, 3) @ np.diag(m.body_inertia[b]) @ d.ximat[b].reshape(3, 3).T)[1, 1]
                  + m.body_mass[b] * ((d.xipos[b] - c)[0] ** 2 + (d.xipos[b] - c)[2] ** 2) for b in range(1, m.nbody))
        bias = d.qfrc_bias
        b7 = np.r_[bias[0], bias[2], bias[4],
                   [sum(bias[m.jnt_dofadr[jid(f"{sd}_{j}_joint")]] for sd in ("left", "right")) for j in JOINTS]]
        sole_mj = r.sole_pos("left") - d.xmat[r.foot_body["left"]].reshape(3, 3) @ [0.5 * (FOOT_X[0] + FOOT_X[1]), 0, 0]
        err = np.maximum(err, [np.abs(o["p"] - c[[0, 2]]).max(), abs(o["L"][0] - d.subtree_angmom[1][1]),
                               abs(o["rho"][0] - rho), np.abs(o["tau"] - b7).max(),
                               np.abs(o["foot"][:2] - sole_mj[[0, 2]]).max()])
    print(f"[symmetric] mass {MASS:.3f} kg (MuJoCo {r.mass:.3f}) | max error over 20 random states: CoM {err[0]:.1e} m, "
          f"L {err[1]:.1e} kg m^2/s, rho {err[2]:.1e} kg m^2, bias forces {err[3]:.1e} N(m), sole point {err[4]:.1e} m")
    q0 = crouch(0.8)
    o = quantities(q=q0, dq=np.zeros(NQ), ddq=np.zeros(NQ))
    print(f"  crouch(0.8): CoM {np.ravel(o['p'])} m, rho {float(o['rho']):.2f} kg m^2 (Stage 1 start: (0.048, 0.725), 7.08)")
    assert abs(MASS - r.mass) < 1e-6 and err[0] < 1e-6 and err[1] < 1e-5 and err[2] < 1e-5 and err[3] < 1e-4 and err[4] < 1e-6
