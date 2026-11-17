"""H1-2 in MuJoCo: model building, spawning in a standing pose, and a joint PD hold.

Model fixes (same as h1_2_cop_cam_jump/model.py):
  * Foot collision mesh -> thin sole box, so CoP and contact points are well defined
  * No damping/armature/friction on the free joint (the <default> class adds them otherwise)
  * No self-collision; only the soles touch the floor
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = ROOT / "assets" / "h1_2" / "scene.xml"

# Foot geometry in the ankle-roll-link frame (measured from the mesh)
FOOT_X = (-0.086, 0.174)
FOOT_Y = (-0.043, 0.043)
SOLE_Z = -0.045
SOLE_CENTER = np.array([0.5 * (FOOT_X[0] + FOOT_X[1]), 0.0, SOLE_Z])
FEET = ("left", "right")

# Arm posture held during the whole jump (elbows bent so the arms clear the torso)
ARMS = {
    "left_shoulder_roll_joint": 0.15,
    "right_shoulder_roll_joint": -0.15,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
}


def leg_pose(a: float) -> dict[str, float]:
    """Symmetric flexion: hip_pitch = -a, knee = 2a, ankle = -a keeps the foot parallel to the pelvis."""
    q = {}
    for s in FEET:
        q[f"{s}_hip_pitch_joint"] = -a
        q[f"{s}_knee_joint"] = 2 * a
        q[f"{s}_ankle_pitch_joint"] = -a
    return q


def build_model(scene: Path | str = DEFAULT_SCENE, timestep: float = 0.0005, box: tuple | None = None) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(Path(scene).resolve()))
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    hx, hy = 0.5 * (FOOT_X[1] - FOOT_X[0]), 0.5 * (FOOT_Y[1] - FOOT_Y[0])
    for side in FEET:
        body = spec.body(f"{side}_ankle_roll_link")
        for g in body.geoms:
            if g.contype:
                g.contype = g.conaffinity = 0
        sole = body.add_geom()
        sole.name = f"{side}_sole"
        sole.type = mujoco.mjtGeom.mjGEOM_BOX
        sole.size = [hx, hy, 0.01]
        sole.pos = [SOLE_CENTER[0], 0.0, SOLE_Z + 0.01]
        sole.rgba = [0.8, 0.3, 0.2, 0.0]  # invisible; the mesh is drawn
        sole.friction = [1.0, 0.02, 0.001]
        sole.condim = 6

    fj = spec.joint("floating_base_joint")
    fj.damping = [0.0, 0.0, 0.0]
    fj.armature = 0.0
    fj.frictionloss = 0.0

    for g in spec.geoms:
        if g.name not in ("floor", "left_sole", "right_sole") and g.contype:
            g.contype = g.conaffinity = 0

    if box:  # (height, front edge x): a 0.75 m deep, 1 m wide box to jump onto (Dai 2014 baseline)
        g = spec.worldbody.add_geom()
        g.name, g.type = "box", mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.375, 0.5, box[0] / 2]
        g.pos = [box[1] + 0.375, 0.0, box[0] / 2]
        g.rgba = [0.72, 0.55, 0.36, 1.0]
        g.friction = [1.0, 0.02, 0.001]

    for tex in spec.textures:  # light sky so the dark robot is visible
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.rgb1 = [0.85, 0.87, 0.9]
            tex.rgb2 = [0.6, 0.65, 0.72]
            tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    return spec.compile()


class Robot:
    """Ids, spawning and a joint-space PD hold."""

    def __init__(self, model: mujoco.MjModel):
        m = self.m = model
        self.d = mujoco.MjData(m)
        self.mass = float(m.body_subtreemass[1])
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.foot_body = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in FEET}

        # All actuators are direct torque motors on hinge joints
        self.act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        self.act_qpos = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.tau_max = m.actuator_ctrlrange[:, 1].copy()
        self.qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.act_names}

        # PD gains, tuned for a static stand (tested: CoM drift < 1 cm over 10 s at a = 0.2).
        # Ankle kp must beat the inverted-pendulum stiffness m*g*h (about 600 Nm/rad);
        # with weaker legs (e.g. 300) the knees fold and the robot falls backward.
        self.kp = np.array([1000.0 if any(k in n for k in ("hip", "knee", "torso")) else
                            600.0 if "ankle" in n else 100.0 for n in self.act_names])
        self.kd = 2.0 * np.sqrt(self.kp) * 0.8
        self.q_target = np.zeros(m.nu)

    # ---------------------------------------------------------------- spawning
    def sole_pos(self, side: str) -> np.ndarray:
        b = self.foot_body[side]
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ SOLE_CENTER

    def spawn(self, joints: dict[str, float], clearance: float = 0.002) -> None:
        """Set the joint posture and put both soles flat on the floor (``clearance`` above it)."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.qpos0
        d.qpos[0:3] = [0.0, 0.0, 1.0]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for n, v in joints.items():
            d.qpos[self.qadr[n]] = v
        mujoco.mj_forward(m, d)

        d.qpos[2] -= min(self.sole_pos(s)[2] for s in FEET) - clearance
        mujoco.mj_forward(m, d)
        self.q_target = d.qpos[self.act_qpos].copy()

    # ---------------------------------------------------------------- control
    def pd_torque(self) -> np.ndarray:
        q = self.d.qpos[self.act_qpos]
        dq = self.d.qvel[self.act_dof]
        tau = self.kp * (self.q_target - q) - self.kd * dq
        return np.clip(tau, -self.tau_max, self.tau_max)

    def step(self) -> None:
        self.d.ctrl[:] = self.pd_torque()
        mujoco.mj_step(self.m, self.d)

    # ---------------------------------------------------------------- state
    def com_offset_x(self) -> float:
        """CoM x minus the sole centre x (positive = CoM ahead of the foot centre)."""
        cx = 0.5 * (self.sole_pos("left")[0] + self.sole_pos("right")[0])
        return float(self.d.subtree_com[1][0] - cx)

    def com(self) -> np.ndarray:
        return self.d.subtree_com[1].copy()

    def base_rpy(self) -> np.ndarray:
        w, x, y, z = self.d.qpos[3:7]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw])
