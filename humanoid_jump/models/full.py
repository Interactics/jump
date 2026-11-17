"""Full 3D H1-2 model (all 27 joints) via Pinocchio, for the Dai 2014 planner (`humanoid_jump/dai2014.py`).

Floating base as roll-pitch-yaw: root = PX, PY, PZ, RZ, RY, RX (R = Rz Ry Rx), so q, v are both in R^{n+6}
(n = 27, nq = nv = 33) and q_dot = v exactly, as Dai's Eq. 7d (q[k] - q[k-1] = v[k] dt[k]) assumes.

Contact points: the 4 corners of each sole box (sim/robot.py), Dai Sec. III-A ("contact points to the four corners
of each foot"). Order per foot: heel-right, heel-left, toe-right, toe-left; left foot first.

    python -m humanoid_jump.models.full      # self-check against the MuJoCo model
"""

from __future__ import annotations

import os
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from humanoid_jump.sim.robot import ARMS, FEET, FOOT_X, FOOT_Y, SOLE_Z, leg_pose

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "third_party/unitree_ros/robots/h1_2_description/h1_2_handless.urdf"
G = 9.81
CORNERS = [(s, x, y) for s in FEET for x in FOOT_X for y in FOOT_Y]      # 8 points
HEEL = [i for i, (_, x, _) in enumerate(CORNERS) if x == FOOT_X[0]]
TOE = [i for i, (_, x, _) in enumerate(CORNERS) if x == FOOT_X[1]]


def _load():
    root = pin.JointModelComposite()
    for j in (pin.JointModelPX(), pin.JointModelPY(), pin.JointModelPZ(),
              pin.JointModelRZ(), pin.JointModelRY(), pin.JointModelRX()):
        root.addJoint(j)
    fd = os.dup(2)                      # urdfdom prints a harmless visual-material error per link
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    try:
        return pin.buildModelFromUrdf(str(URDF), root)
    finally:
        os.dup2(fd, 2)


MODEL = _load()
DATA = MODEL.createData()
NQ = MODEL.nq                           # 33 = nv
MASS = pin.computeTotalMass(MODEL)
JOINT_NAMES = list(MODEL.names[2:])     # the 27 actuated joints, in q order (q[6:])
Q_MIN, Q_MAX = MODEL.lowerPositionLimit[6:].copy(), MODEL.upperPositionLimit[6:].copy()
V_MAX = MODEL.velocityLimit[6:].copy()
TAU_MAX = MODEL.effortLimit[6:].copy()
FOOT_FRAME = {s: MODEL.getFrameId(f"{s}_ankle_roll_link") for s in FEET}


def _build():
    cm = cpin.Model(MODEL)
    cd = cm.createData()
    q, v = ca.SX.sym("q", NQ), ca.SX.sym("v", NQ)
    com = cpin.centerOfMass(cm, cd, q)
    cpin.computeCentroidalMomentum(cm, cd, q, v)
    h = cd.hg.angular                                            # centroidal angular momentum = A(q) v  (Eq. 2)
    cpin.framesForwardKinematics(cm, cd, q)
    pts = []
    for s, x, y in CORNERS:
        M = cd.oMf[FOOT_FRAME[s]]
        pts.append(M.translation + M.rotation @ ca.vertcat(x, y, SOLE_Z))
    C = ca.horzcat(*pts)
    kin = ca.Function("full_kin", [q, v], [com, h, C], ["q", "v"], ["com", "h", "c"])

    # Joint torques for the optional torque limits: tau = RNEA(q, v, a) - sum J_j^T F_j (joint rows only).
    # With the rpy root q_dot = v, so J_j = d c_j / d q.
    a, F = ca.SX.sym("a", NQ), ca.SX.sym("F", 3, len(CORNERS))
    tau = cpin.rnea(cm, cd, q, v, a)
    for j in range(len(CORNERS)):
        tau -= ca.jacobian(C[:, j], q).T @ F[:, j]
    return kin, ca.Function("full_tau", [q, v, a, F], [tau[6:]], ["q", "v", "a", "F"], ["tau"])


kin, joint_tau = _build()


def corner_jacobians(q: np.ndarray) -> list[np.ndarray]:
    """3 x nv world-frame Jacobian of each contact point (numeric)."""
    pin.computeJointJacobians(MODEL, DATA, q)
    pin.updateFramePlacements(MODEL, DATA)
    out = []
    for s, x, y in CORNERS:
        f = FOOT_FRAME[s]
        J6 = pin.getFrameJacobian(MODEL, DATA, f, pin.LOCAL_WORLD_ALIGNED)
        r = DATA.oMf[f].rotation @ np.array([x, y, SOLE_Z])
        out.append(J6[:3] - pin.skew(r) @ J6[3:])
    return out


def joint_torques(q, v, a, forces) -> np.ndarray:
    """Inverse dynamics: tau = RNEA(q, v, a) - sum J_j^T F_j. Returns all nv rows (rows 0-5 should be ~0)."""
    tau = pin.rnea(MODEL, DATA, q, v, a).copy()
    for J, F in zip(corner_jacobians(q), forces):
        tau -= J.T @ F
    return tau


def crouch(a: float) -> np.ndarray:
    """Flat-footed crouch (hip = -a, knee = 2a, ankle = -a), arms as in the simulator, soles on the floor,
    ankle at x = 0 (same posture and placement as models/symmetric.crouch)."""
    q = np.zeros(NQ)
    for n, val in {**leg_pose(a), **ARMS}.items():
        q[6 + JOINT_NAMES.index(n)] = val
    c = np.array(kin(q, np.zeros(NQ))[2])
    heel_x = c[0, HEEL].mean()
    q[0] = -(heel_x - FOOT_X[0])                    # sole point under the ankle (x = 0 in the foot frame) at x = 0
    q[2] = -c[2].mean()
    return q


def to_mujoco(q: np.ndarray, m, qadr: dict) -> np.ndarray:
    """MuJoCo qpos for configuration q (free joint quaternion is w, x, y, z)."""
    qpos = m.qpos0.copy()
    R = pin.rpy.rpyToMatrix(q[5], q[4], q[3])      # roll, pitch, yaw -> Rz(yaw) Ry(pitch) Rx(roll)
    x, y, z, w = pin.Quaternion(R).coeffs()
    qpos[0:3], qpos[3:7] = q[0:3], [w, x, y, z]
    for i, n in enumerate(JOINT_NAMES):
        qpos[qadr[n]] = q[6 + i]
    return qpos


if __name__ == "__main__":
    import mujoco

    from humanoid_jump.sim.robot import Robot, build_model

    r = Robot(build_model())
    m, d = r.m, r.d
    assert sorted(JOINT_NAMES) == sorted(r.act_names)
    rng, err = np.random.default_rng(0), np.zeros(4)
    for _ in range(20):
        q = np.r_[rng.uniform(-0.3, 0.3, 2), rng.uniform(0.8, 1.0), rng.uniform(-0.4, 0.4, 3),
                  rng.uniform(np.maximum(Q_MIN, -1.0), np.minimum(Q_MAX, 1.0))]
        v = np.r_[np.zeros(6), rng.uniform(-2, 2, NQ - 6)]          # base velocity 0: no frame-convention mismatch
        com, h, c = (np.array(o) for o in kin(q, v))
        mujoco.mj_resetData(m, d)
        d.qpos[:] = to_mujoco(q, m, r.qadr)
        for i, n in enumerate(JOINT_NAMES):
            d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v[6 + i]
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        c_mj = np.array([d.xpos[r.foot_body[s]] + d.xmat[r.foot_body[s]].reshape(3, 3) @ [x, y, SOLE_Z] for s, x, y in CORNERS]).T
        tau = pin.rnea(MODEL, DATA, q, v, np.zeros(NQ))              # bias forces, joint rows vs MuJoCo
        g_mj = np.array([d.qfrc_bias[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in JOINT_NAMES])
        err = np.maximum(err, [np.abs(com.ravel() - d.subtree_com[1]).max(), np.abs(h.ravel() - d.subtree_angmom[1]).max(),
                               np.abs(c - c_mj).max(), np.abs(tau[6:] - g_mj).max()])
    q0 = crouch(0.8)
    com0, _, c0 = (np.array(o) for o in kin(q0, np.zeros(NQ)))
    print(f"[full] nq = nv = {NQ}, mass {MASS:.3f} kg (MuJoCo {r.mass:.3f}) | max error over 20 random states: "
          f"CoM {err[0]:.1e} m, CAM {err[1]:.1e} kg m^2/s, sole corners {err[2]:.1e} m, bias torques {err[3]:.1e} N m")
    print(f"  crouch(0.8): CoM ({com0[0, 0]:.3f}, {com0[1, 0]:.3f}, {com0[2, 0]:.3f}) m (7-DOF model: (0.048, 0.725)), "
          f"corner z {np.abs(c0[2]).max():.1e} m")
    assert abs(MASS - r.mass) < 1e-6 and err.max() < 1e-5 and np.abs(c0[2]).max() < 1e-9
    print("full model self-check: PASS")
