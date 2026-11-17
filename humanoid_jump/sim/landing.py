"""Landing controller: thin wrapper around the sister project's T-RO re-implementation
(../../h1_2_cop_cam_jump, used read-only): CoM PD -> wrench QP (friction, CoP in sole) -> IACVM foot-torque
correction -> acceleration QP + inverse dynamics. Only the glue of its landing branch is repeated here.

One gain differs from the sister project: the CoM PD in x is (20, 25) instead of (60, 18). The CoM touches down
behind the feet with forward velocity; a stiff pull towards the foot centre drives the CoP to the heel and tips
the feet back. Softer stiffness and more damping lets the forward momentum carry the CoM over the feet.

Second difference: IACVM also acts on foot roll, with the opposite sign and gain 0.7 (iacvm_roll = -0.7).
- Why: the sister project uses pitch only (--iacvm_pitch_only 1). Then, after harder landings (h 0.2 l 0.4/0.5,
  h 0.15 l 0.5, box), both feet keep rocking in roll (±12 deg std, ankle roll τ ±20 N·m, sole Fz 0.5-2.3 BW) and it
  never decays. After h 0.1 l 0.2 the feet stay rolled 30 deg.
- Sweep over 7 landings plus 3 model errors (trunk +5 kg, mass +10 %, friction 0.5): every gain from -0.3 to -1
  stops the rocking, and the feet end flat.
- -0.7 is the only gain that loses no case that stood before. Each other gain loses one marginal case:
  -0.3, -0.8 and -1 lose h 0.2 l 0.6; -0.5 loses trunk +5 kg. Those two landings flip with small changes.
- +1 leaves the feet at ±20 deg or falls. Flipping pitch to -1 makes every case fall.
- The sign and the gain come from these tests, not from a derivation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

from humanoid_jump.sim.robot import ARMS, FEET, Robot, leg_pose

SISTER = Path(__file__).resolve().parents[4] / "h1_2_cop_cam_jump"
G = 9.81


class _MujocoCompat:
    """The sister project targets MuJoCo 4 (mj_fullM(m, d, M)); this env has 3.9 (mj_fullM(m, M, d.qM))."""

    def __getattr__(self, name):
        return getattr(mujoco, name)

    @staticmethod
    def mj_fullM(m, d, M):
        try:
            mujoco.mj_fullM(m, d, M)
        except TypeError:
            mujoco.mj_fullM(m, M, d.qM)


class LandingController:
    def __init__(self, robot: Robot, z_land: float = 0.78, a_land: float = 0.6, kp=(20, 60, 80), kd=(25, 18, 22),
                 k_L: float = 10.0, k_ori: float = 60.0, i_body: float = 8.0, kp_post: float = 100.0,
                 kp_ori: float = 100.0, x_offset: float = 0.0, iacvm_roll: float = -0.7):
        sys.path.insert(0, str(SISTER))
        import h1_2_jump.cdc as cdc
        from h1_2_jump.cdc import AccelQP, WrenchQP
        from h1_2_jump.controllers import IACVM, CoMPD
        from h1_2_jump.model import Robot as SisterRobot

        cdc.mujoco = _MujocoCompat()                         # runtime only; nothing on disk changes
        self.r = SisterRobot(robot.m)
        self.r.d = robot.d                                   # same simulation state
        self.pd, self.iacvm = CoMPD(kp=kp, kd=kd), IACVM(use_accel=False)
        self.wqp, self.aqp = WrenchQP(mu=0.8), AccelQP(self.r, w_base_ori=2.0)
        for i, n in enumerate(self.r.jnt_names):             # arms may move to help with momentum
            if "shoulder" in n or "elbow" in n:
                self.aqp.w[self.r.act_dof[i]] = 2.0
            elif "wrist" in n:
                self.aqp.w[self.r.act_dof[i]] = 5.0
        q = robot.m.qpos0.copy()
        for n, v in {**leg_pose(a_land), **ARMS}.items():
            q[robot.qadr[n]] = v
        self.q_nom = q[7:]
        self.z_land, self.x_offset, self.iacvm_roll = z_land, x_offset, iacvm_roll
        self.k_L, self.k_ori, self.i_body, self.kp_post, self.kp_ori = k_L, k_ori, i_body, kp_post, kp_ori
        self.c_ref = None

    def start(self) -> None:
        mid = 0.5 * (self.r.sole_pos("left") + self.r.sole_pos("right"))
        self.c_ref = np.array([mid[0] + self.x_offset, mid[1], self.z_land])
        self.iacvm.reset()

    def torque(self, dt: float) -> np.ndarray:
        r, d = self.r, self.r.d
        c, v, L = r.centroidal_state()
        a = self.pd(self.c_ref, np.zeros(3), c, v)
        a[2] = np.clip(a[2], -G + 0.5, 3 * G)
        F_des = r.mass * (a + [0, 0, G])
        rpy = r.body_rpy(r.pelvis)
        Ldot = -self.k_L * L
        Ldot[:2] += self.i_body * (-self.k_ori * rpy[:2])
        W = self.wqp.solve(r, c, F_des, Ldot)
        for k, s in enumerate(FEET):                         # IACVM on foot pitch (and roll if iacvm_roll)
            w6 = np.zeros(6)
            mujoco.mj_objectVelocity(r.m, d, mujoco.mjtObj.mjOBJ_BODY, r.foot_body[s], w6, 1)
            dT = self.iacvm(r.body_rpy(r.foot_body[s])[:2], w6[:2], dt)
            W[k, 3:6] += r.foot_R(s) @ np.array([dT[0] * self.iacvm_roll, dT[1], 0.0])
        qdd = np.zeros(r.nv)
        qdd[6:] = self.kp_post * (self.q_nom - d.qpos[7:]) - 2 * np.sqrt(self.kp_post) * d.qvel[6:]
        qdd[3:6] = -self.kp_ori * np.array([rpy[0], rpy[1], 0.0]) - 2 * np.sqrt(self.kp_ori) * d.qvel[3:6]
        return self.aqp.solve(qdd, W)[1]
