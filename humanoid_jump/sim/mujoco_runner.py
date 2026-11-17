"""MuJoCo execution of the Stage 3 trajectory: launch + flight playback, then the landing controller. Phase 5.

Launch and flight: joint PD on the planned (q, dq) plus the planned joint torque as feedforward, on the four
pitch joints per side; all other joints hold their spawn angle.
Landing (from touch-down, detected from the ground force): sim/landing.py, the sister project's controller.

    python -m humanoid_jump.sim.mujoco_runner              # headless: numbers + results/playback.png
    python -m humanoid_jump.sim.mujoco_runner --no-ff      # PD only (the paper's position playback)
    python -m humanoid_jump.sim.mujoco_runner --no-landing # hold the last posture instead of the landing controller
    mjpython -m humanoid_jump.sim.mujoco_runner --view     # viewer, quarter speed, loops
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from humanoid_jump.models.symmetric import JOINTS, NQ, quantities
from humanoid_jump.sim.landing import LandingController
from humanoid_jump.sim.robot import ARMS, FEET, SOLE_Z, Robot, build_model

ROOT = Path(__file__).resolve().parents[2]
G = 9.81


def load(path: Path = ROOT / "results" / "stage3_nlp.npz") -> dict:
    """Planned trajectory on one time grid, plus the planned CoM / CAM for comparison."""
    d = np.load(path)
    t = np.r_[d["t"], d["tF"][1:]]
    X = np.c_[d["X"], d["XF"][:, 1:]]
    tau = np.r_[d["tau_l"], d["tau_f"][1:]]
    o = [quantities(q=X[:NQ, k], dq=X[NQ:2 * NQ, k], ddq=X[2 * NQ:, k]) for k in range(len(t))]
    return dict(t=t, T=float(d["t"][-1]), q=X[:NQ], dq=X[NQ:2 * NQ], tau=tau,
                p=np.array([np.ravel(m["p"]) for m in o]), v=np.array([np.ravel(m["v"]) for m in o]),
                L=np.array([float(m["L"]) for m in o]))


TD_FORCE = 80.0     # touch-down detection: total vertical ground force [N]


def play(tr: dict, ff: bool = True, hold: float = 2.0, on_step=None, frames_at=(), landing: bool = True,
         model: mujoco.MjModel | None = None, **land_kw) -> dict:
    r = Robot(model if model is not None else build_model())             # model: e.g. a perturbed one (Phase 6)
    m, d = r.m, r.d
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
    act = {n: i for i, n in enumerate(r.act_names)}
    idx = [[act[f"{s}_{j}_joint"] for s in FEET] for j in JOINTS]       # actuator ids per planned joint
    soles = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_sole") for s in FEET}

    def at(t):                                                           # planned q, dq, tau at time t (held after the end)
        f = lambda Y: np.array([np.interp(t, tr["t"], y) for y in Y])    # noqa: E731
        return f(tr["q"]), f(tr["dq"]) * (t <= tr["t"][-1]), f(tr["tau"].T) * (t <= tr["t"][-1])

    q0, dq0, _ = at(0.0)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.qpos0
    d.qpos[0:3] = [q0[0], 0, q0[1]]
    d.qpos[3:7] = [np.cos(q0[2] / 2), 0, np.sin(q0[2] / 2), 0]
    for n, v in ARMS.items():
        d.qpos[r.qadr[n]] = v
    for i, j in enumerate(JOINTS):
        for s in FEET:
            d.qpos[r.qadr[f"{s}_{j}_joint"]] = q0[3 + i]
    mujoco.mj_forward(m, d)
    q_ref, dq_ref, tau_ff = d.qpos[r.act_qpos].copy(), np.zeros(m.nu), np.zeros(m.nu)

    renderer, cam, frames = None, None, []
    if frames_at:
        renderer = mujoco.Renderer(m, 480, 360)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 90, -8, 3.4

    lc = LandingController(r, **land_kw) if landing else None
    landed, fz, tau_land = False, 0.0, np.zeros(m.nu)
    log = {k: [] for k in ("t", "p", "v", "L", "pitch", "fz", "foot_x", "foot_pitch", "sat", "landed", "tau_frac")}
    frames_at = list(frames_at)
    n_steps = int((tr["t"][-1] + hold) / m.opt.timestep)
    for step in range(n_steps):
        if lc is not None and not landed and d.time > tr["T"] + 0.1 and fz > TD_FORCE:
            landed = True
            lc.start()
        if landed:
            if step % 2 == 0:                                                # 1 kHz, as in the sister project
                tau_land = lc.torque(2 * m.opt.timestep)
            tau = tau_land
        else:
            qd, dqd, taud = at(d.time)
            for i in range(len(JOINTS)):
                q_ref[idx[i]], dq_ref[idx[i]], tau_ff[idx[i]] = qd[3 + i], dqd[3 + i], taud[i] / 2 * ff
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
        log["sat"].append(np.any(np.abs(tau[np.ravel(idx)]) > r.tau_max[np.ravel(idx)]))
        log["landed"].append(landed); log["tau_frac"].append(np.abs(d.ctrl[np.ravel(idx)] / r.tau_max[np.ravel(idx)]))
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


def summarize(tr: dict, lg: dict, tag: str) -> dict:
    t, T, bw = lg["t"], tr["T"], 66.984 * G
    k_to = np.searchsorted(t, T)
    air = (lg["fz"] < 1.0) & (t > 0.3)
    t_off = t[np.argmax(air)] if air.any() else np.nan
    after = np.where((lg["fz"] > 1.0) & (t > t_off + 0.05))[0] if air.any() else []
    k_td = after[0] if len(after) else None
    v_plan = np.array([np.interp(T, tr["t"], tr["v"][:, i]) for i in (0, 1)])
    print(f"[playback/{tag}] lift-off at {t_off:.3f} s (plan {T:.3f}) | at the planned take-off time: "
          f"v ({lg['v'][k_to, 0]:.3f}, {lg['v'][k_to, 1]:.3f}) m/s (plan {v_plan[0]:.3f}, {v_plan[1]:.3f}) | "
          f"L {lg['L'][k_to]:.2f} (plan {np.interp(T, tr['t'], tr['L']):.2f}) | "
          f"pelvis pitch {np.degrees(lg['pitch'][k_to]):.1f} deg (plan {np.degrees(np.interp(T, tr['t'], tr['q'][2])):.1f})")
    apex_plan = np.interp(T, tr["t"], tr["p"][:, 1]) + v_plan[1] ** 2 / (2 * G)
    res = dict(t_off=t_off, v_to=lg["v"][k_to], apex=lg["p"][:, 1].max(), td=None)
    line = f"  launch peak Fz {lg['fz'][:k_to].max() / bw:.2f} BW | CoM apex {res['apex']:.3f} m (plan {apex_plan:.3f}) | torque saturated in {100 * lg['sat'][:k_to].mean():.0f} % of launch steps"
    if k_td is not None:
        res["td"] = (t[k_td], lg["foot_x"][k_td])
        res["impact"] = lg["fz"][k_td:].max() / bw
        line += (f"\n  touch-down at {t[k_td]:.3f} s (plan {tr['t'][-1]:.3f}) | ankle x {lg['foot_x'][k_td]:.3f} m (plan {tr.get('l', 0.4):.3f}) | "
                 f"foot pitch {np.degrees(lg['foot_pitch'][k_td]):.1f} deg, pelvis pitch {np.degrees(lg['pitch'][k_td]):.1f} deg | "
                 f"impact peak Fz {lg['fz'][k_td:].max() / bw:.1f} BW | "
                 f"\n  {t[-1] - t[k_td]:.2f} s after touch-down ({'landing controller' if lg['has_landing'] else 'PD hold'}): "
                 f"pelvis z {lg['pelvis_z']:.2f} m, pitch {np.degrees(lg['pitch'][-1]):.1f} deg, "
                 f"CoM speed {np.linalg.norm(lg['v'][-1]):.3f} m/s, ankle x {lg['foot_x'][-1]:.3f} m "
                 f"(moved {1e3 * (lg['foot_x'][-1] - lg['foot_x'][k_td]):+.0f} mm), CoM x - ankle x {lg['p'][-1, 0] - lg['foot_x'][-1]:+.3f} m")
        # CoM over the foot too: a stumble can end at standing height with one foot 0.6 m ahead (Phase 6, h 0.05 l 0.4)
        # Mean CoM speed over the last 0.5 s, not the last sample: some landings keep a residual sway (peaks 0.09 m/s)
        # while the feet do not move (Phase 6, h 0.2 l 0.4).
        sway = np.linalg.norm(lg["v"][t > t[-1] - 0.5], axis=1).mean()
        res["upright"] = bool(lg["pelvis_z"] > 0.6 and abs(lg["pitch"][-1]) < 0.3 and sway < 0.05
                              and abs(lg["p"][-1, 0] - lg["foot_x"][-1]) < 0.1)
        line += f" -> {'STANDING' if res['upright'] else 'FELL'}"
    else:
        line += "\n  no flight phase detected" if not air.any() else "\n  no touch-down within the simulated time"
    print(line)
    return res


def plot(tr: dict, runs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 5)
    panels = (("CoM z [m]", lambda g: g["p"][:, 1], tr["p"][:, 1]), ("CoM vx [m/s]", lambda g: g["v"][:, 0], tr["v"][:, 0]),
              ("CoM vz [m/s]", lambda g: g["v"][:, 1], tr["v"][:, 1]), ("CAM L [kg m^2/s]", lambda g: g["L"], tr["L"]),
              ("pelvis pitch [deg]", lambda g: np.degrees(g["pitch"]), np.degrees(tr["q"][2])))
    for i, (title, f, plan) in enumerate(panels):
        a = fig.add_subplot(gs[0, i])
        for tag, lg in runs.items():
            a.plot(lg["t"], f(lg), label=f"sim, {tag}")
        a.plot(tr["t"], plan, "k--", label="plan")
        a.axvline(tr["T"], color="gray", lw=0.8); a.axvline(tr["t"][-1], color="gray", lw=0.8)
        a.set_xlim(0, lg["t"][-1]); a.set_title(title); a.set_xlabel("t [s]"); a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=7)
    lg = next(iter(runs.values()))
    if lg["frames"]:
        a = fig.add_subplot(gs[1, :])
        a.imshow(np.hstack(lg["frames"])); a.axis("off")
        a.set_title(f"simulation snapshots ({next(iter(runs))})")
    fig.tight_layout(); path.parent.mkdir(exist_ok=True); fig.savefig(path, dpi=110)


if __name__ == "__main__":
    tr = load()
    if "--view" in sys.argv:
        import mujoco.viewer

        r0 = Robot(build_model())
        with mujoco.viewer.launch_passive(r0.m, r0.d) as viewer:
            def show(d, slow=4.0):
                r0.d.qpos[:] = d.qpos
                mujoco.mj_forward(r0.m, r0.d)
                if int(d.time / r0.m.opt.timestep) % 20 == 0:          # 100 fps of simulated time
                    viewer.sync(); time.sleep(0.01 * slow)
            while viewer.is_running():
                play(tr, ff="--no-ff" not in sys.argv, landing="--no-landing" not in sys.argv, on_step=show)
        sys.exit(0)

    shots = list(np.r_[np.linspace(0.0, tr["t"][-1], 6), tr["t"][-1] + np.array([0.15, 0.5, 1.9])])
    ff, land = "--no-ff" not in sys.argv, "--no-landing" not in sys.argv
    tag = ("PD + torque feedforward" if ff else "PD only") + (", landing controller" if land else ", PD hold")
    runs = {tag: play(tr, ff=ff, landing=land, frames_at=shots)}
    res = {tag: summarize(tr, lg, tag) for tag, lg in runs.items()}
    plot(tr, runs, ROOT / "results" / "playback.png")
    if land:
        assert res[tag].get("upright"), "the robot did not end up standing"
        print("Phase 5 gate: full jump without falling")
