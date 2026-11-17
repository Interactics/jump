"""MuJoCo execution of a full-body (27-joint) plan from humanoid_jump/dai2014.py.

Same execution as the T-ASE runner (sim/mujoco_runner.py, #21): joint PD on the planned (q, v) plus the planned
joint torque as feedforward, now on all 27 joints. Dai's plan contains the landing (toes, then heels touch down,
end at rest), so by default the whole plan is played and its last posture held with its static torques.
--landing switches to the landing controller at touch-down instead, as in the T-ASE pipeline.
--unlimited runs it on H1-2 with the motor torque limits removed (x100): Dai's assumption of sufficient torque
(Sec. II-A), on the robot side. Without it the real H1-2 limits apply.

    python -m humanoid_jump.sim.play_full [path] [--landing] [--unlimited]    # default results/dai2014.npz
    mjpython -m humanoid_jump.sim.play_full --view
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from humanoid_jump.models.full import JOINT_NAMES, to_mujoco
from humanoid_jump.sim.landing import LandingController
from humanoid_jump.sim.mujoco_runner import TD_FORCE, plot, summarize
from humanoid_jump.sim.robot import FEET, SOLE_Z, Robot, build_model

ROOT = Path(__file__).resolve().parents[2]
UNLIMITED = 100.0          # torque-limit scale for --unlimited (x10 already gave 0 % saturation at h 0.10, l 0.40)


def unlimited_model(box: tuple | None = None) -> mujoco.MjModel:
    """H1-2 with the actuator ctrlrange and the joint actuatorfrcrange (both limit torque in the MJCF) x UNLIMITED."""
    m = build_model(box=box)
    m.actuator_ctrlrange[:] *= UNLIMITED
    m.jnt_actfrcrange[:] *= UNLIMITED
    return m


def load(path: Path = ROOT / "results" / "dai2014.npz") -> dict:
    d = dict(np.load(path))
    t, k_to, k_td = d["t"], int(d["k_to"]), int(d["k_td"])
    # keys used by mujoco_runner.summarize / plot; q rows = (x, z, pelvis pitch) as in the 7-DOF plans
    return dict(t=t, T=float(t[k_to]), t_td=float(t[k_td]), k_td=k_td, l=float(d["l_jump"]), qf=d["q"], vf=d["v"],
                tau=d["tau"], p=d["r"][[0, 2]].T, v=d["rd"][[0, 2]].T, L=d["h"][1], q=d["q"][[0, 2, 4]])


def until_touchdown(tr: dict) -> dict:
    """Plan cut at touch-down, so that summarize() compares lift-off, apex and touch-down with the right times."""
    k = tr["k_td"] + 1
    return {**tr, "t": tr["t"][:k], "p": tr["p"][:k], "v": tr["v"][:k], "L": tr["L"][:k], "q": tr["q"][:, :k]}


def play(tr: dict, ff: bool = True, landing: bool = False, hold: float = 3.0, model=None, on_step=None, frames_at=()) -> dict:
    r = Robot(model if model is not None else build_model())
    m, d = r.m, r.d
    act = [r.act_names.index(n) for n in JOINT_NAMES]           # plan joint i -> actuator act[i]
    soles = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_sole") for s in FEET}
    t_end = tr["t"][-1]

    def at(t):
        k = min(int(np.searchsorted(tr["t"], t)), len(tr["t"]) - 1)        # v[k] is the velocity on (t[k-1], t[k]] (Eq. 7d)
        q = np.array([np.interp(t, tr["t"], y) for y in tr["qf"][6:]])
        tau = np.array([np.interp(t, tr["t"], y) for y in tr["tau"].T])     # after the end: the final static torques
        return q, tr["vf"][6:, k] * (t <= t_end), tau

    mujoco.mj_resetData(m, d)
    d.qpos[:] = to_mujoco(tr["qf"][:, 0], m, r.qadr)
    mujoco.mj_forward(m, d)
    q_ref, dq_ref, tau_ff = np.zeros(m.nu), np.zeros(m.nu), np.zeros(m.nu)

    renderer, cam, frames = None, None, []
    if frames_at:
        renderer = mujoco.Renderer(m, 480, 360)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 90, -8, 3.4
    frames_at = list(frames_at)

    lc = LandingController(r) if landing else None
    landed, fz, tau_land = False, 0.0, np.zeros(m.nu)
    log = {k: [] for k in ("t", "p", "v", "L", "pitch", "fz", "foot_x", "foot_pitch", "sat", "landed", "tau_frac")}
    for step in range(int((t_end + hold) / m.opt.timestep)):
        if lc is not None and not landed and d.time > tr["T"] + 0.1 and fz > TD_FORCE:
            landed = True
            lc.start()
        if landed:
            if step % 2 == 0:
                tau_land = lc.torque(2 * m.opt.timestep)
            tau = tau_land
        else:
            qd, dqd, taud = at(d.time)
            q_ref[act], dq_ref[act], tau_ff[act] = qd, dqd, taud * ff
            tau = tau_ff + r.kp * (q_ref - d.qpos[r.act_qpos]) + r.kd * (dq_ref - d.qvel[r.act_dof])
        d.ctrl[:] = np.clip(tau, -r.tau_max, r.tau_max)
        mujoco.mj_step(m, d)
        mujoco.mj_subtreeVel(m, d)

        fz, f6 = 0.0, np.zeros(6)
        for c in range(d.ncon):
            if d.contact[c].geom1 in soles or d.contact[c].geom2 in soles:
                mujoco.mj_contactForce(m, d, c, f6)
                fz += f6[0]
        Rf = d.xmat[r.foot_body["left"]].reshape(3, 3)
        log["t"].append(d.time); log["p"].append(d.subtree_com[1][[0, 2]].copy())
        log["v"].append(d.subtree_linvel[1][[0, 2]].copy()); log["L"].append(d.subtree_angmom[1][1])
        log["pitch"].append(r.base_rpy()[1]); log["fz"].append(fz)
        log["foot_x"].append((d.xpos[r.foot_body["left"]] + Rf @ [0, 0, SOLE_Z])[0])
        log["foot_pitch"].append(np.arctan2(Rf[0, 2], Rf[2, 2]))
        log["sat"].append(bool(np.any(np.abs(tau) > r.tau_max)))
        log["landed"].append(landed); log["tau_frac"].append(np.abs(d.ctrl / r.tau_max))
        if frames_at and d.time >= frames_at[0]:
            frames_at.pop(0)
            cam.lookat[:] = [d.subtree_com[1][0], 0, 0.75]
            renderer.update_scene(d, cam)
            frames.append(renderer.render().copy())
        if on_step is not None:
            on_step(d)
    out = {k: np.array(v) for k, v in log.items()}
    out["frames"], out["pelvis_z"], out["has_landing"] = frames, float(d.qpos[2]), landing
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tr = load(Path(args[0]) if args else ROOT / "results" / "dai2014.npz")
    land, unl = "--landing" in sys.argv, "--unlimited" in sys.argv
    if "--view" in sys.argv:
        import mujoco.viewer

        r0 = Robot(build_model())
        with mujoco.viewer.launch_passive(r0.m, r0.d) as viewer:
            def show(d, slow=4.0):
                r0.d.qpos[:] = d.qpos
                mujoco.mj_forward(r0.m, r0.d)
                if int(d.time / r0.m.opt.timestep) % 20 == 0:
                    viewer.sync(); time.sleep(0.01 * slow)
            while viewer.is_running():
                play(tr, landing=land, on_step=show)
        sys.exit(0)
    tag = ("Dai 2014 plan, PD + feedforward" + (", landing controller" if land else ", planned landing")
           + (", unlimited torque" if unl else ", H1-2 torque limits"))
    shots = list(np.r_[np.linspace(0.0, tr["t"][-1], 7), tr["t"][-1] + np.array([0.5, 2.9])])
    lg = play(tr, landing=land, frames_at=shots, model=unlimited_model() if unl else None)
    res = summarize(until_touchdown(tr), lg, tag)
    sat = lg["sat"][lg["t"] <= tr["t"][-1]]
    print(f"  torque saturated (clipped) in {100 * sat.mean():.0f} % of the plan's steps | "
          f"success (stands, touch-down within 10 cm): {bool(res.get('upright') and res['td'] and abs(res['td'][1] - tr['l']) < 0.10)}")
    plot(tr, {tag: lg}, ROOT / "results" / f"playback_{Path(args[0]).stem if args else 'dai2014'}{'_landing' if land else ''}{'_unlimited' if unl else ''}.png")
