"""Phase 6: evaluation of the paper's pipeline. One table + one figure.

1. Targets: a grid of (h, l). For each: Stage 1, Stage 2, Stage 3 from the naive guess (baseline) and from the
   Stage 2 guess (the paper's pipeline, WARM options); the pipeline result (naive as fallback) is run in MuJoCo.
   -> optimization success rate, speed-up vs baseline, simulation success rate, landing distance error.
2. Model errors: the nominal plan (h 0.10, l 0.40) run on perturbed MuJoCo models (mass, trunk payload, motor
   strength, friction). The plan and the launch/flight playback do not know about the error.
   ponytail: the landing controller reads the perturbed MuJoCo model, i.e. it knows the error (best case for the
   landing); a nominal controller model would need a second MjModel inside sim/landing.py.

    python scripts/evaluate.py             # targets (resumes from results/eval/targets.json, about 40 min) + model errors
    python scripts/evaluate.py --report    # table + figure from the saved results, no solving
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT = Path(__file__).resolve().parents[1] / "results" / "eval"
H, L = (0.05, 0.10, 0.15, 0.20), (0.2, 0.3, 0.4, 0.5, 0.6)
NOMINAL = (0.10, 0.40)
MAX_ITER = 1500
TOL = 0.10        # [m] a jump succeeds if the robot ends standing and touches down within TOL of the target


def success(s, l):
    return bool(s and s["upright"] and s["td_x"] is not None and abs(s["td_x"] - l) < TOL)


def perturbations():
    def scale_mass(s):
        def f(m):
            m.body_mass[:] *= s; m.body_inertia[:] *= s
        return f

    def payload(kg):
        def f(m):
            import mujoco
            m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")] += kg
        return f

    def motors(s):
        def f(m):
            m.actuator_gear[:, 0] *= s
        return f

    def friction(mu):
        def f(m):
            m.geom_friction[:, 0] = mu
        return f

    return {"nominal": lambda m: None, "mass -10%": scale_mass(0.9), "mass +10%": scale_mass(1.1),
            "trunk +5 kg": payload(5.0), "motors -10%": motors(0.9), "motors -20%": motors(0.8),
            "friction 0.5": friction(0.5), "friction 0.3": friction(0.3)}


def sim(path, l, model=None, tag=""):
    from humanoid_jump.sim import mujoco_runner as mr
    tr = mr.load(path); tr["l"] = l
    # 3 s after the end of the plan: with 2 s, h 0.2 l 0.3 was still settling (0.054 m/s > 0.05) and counted as a fall
    res = mr.summarize(tr, mr.play(tr, model=model, hold=3.0), tag)
    td = res.get("td")
    return dict(upright=bool(res.get("upright", False)), td_x=None if td is None else float(td[1]),
                impact=res.get("impact"), v_to=[float(v) for v in res["v_to"]], t_off=float(res["t_off"]),
                apex=float(res["apex"]), plan_apex=float(np.interp(tr["T"], tr["t"], tr["p"][:, 1])
                                                         + np.interp(tr["T"], tr["t"], tr["v"][:, 1]) ** 2 / (2 * 9.81)))


def run_targets():
    from humanoid_jump.stages import stage1_srmp as s1
    from humanoid_jump.stages import stage2_qp as s2
    from humanoid_jump.stages import stage3_nlp as s3

    f = OUT / "targets.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    done = {(r["h"], r["l"]) for r in rows}
    for h in H:
        for l in L:
            if (h, l) in done:
                continue
            ref = s1.launch(h, l, joint=True, land_offset=s3.LAND_OFFSET)
            row = dict(h=h, l=l, s1_ok=bool(ref["ok"]), s1=ref["ms"] / 1e3)
            if ref["ok"]:
                g = s2.solve(ref)
                row["s2"] = g["ms"] / 1e3
                out = {}
                for k, guess, opts in (("naive", None, None), ("stage2", g, s3.WARM)):
                    out[k] = s3.solve(ref, guess=guess, ipopt=opts, max_iter=MAX_ITER)
                    row[k] = dict(ok=bool(out[k]["ok"]), status=out[k]["status"], iters=int(out[k]["iters"]), s=out[k]["ms"] / 1e3)
                best = "stage2" if row["stage2"]["ok"] else "naive" if row["naive"]["ok"] else None
                row["used"] = best
                if best:
                    path = OUT / f"traj_h{h}_l{l}.npz"
                    s3.save(out[best], path)
                    row["sim"] = sim(path, l, tag=f"h {h} l {l}")
            print(row, flush=True)
            rows.append(row)
            OUT.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(rows, indent=1))
    return rows


def run_model_errors():
    import mujoco

    from humanoid_jump.sim.robot import build_model

    path = OUT / f"traj_h{NOMINAL[0]}_l{NOMINAL[1]}.npz"
    rows = []
    for name, edit in perturbations().items():
        m = build_model()
        edit(m)
        mujoco.mj_setConst(m, mujoco.MjData(m))   # else body_subtreemass stays nominal and the logged CoM is wrong
        rows.append(dict(name=name, **sim(path, NOMINAL[1], model=m, tag=name)))
        print(rows[-1], flush=True)
    (OUT / "model_errors.json").write_text(json.dumps(rows, indent=1))
    return rows


def report(rows, errs):
    ok1 = [r for r in rows if r["s1_ok"]]
    n = len(rows)
    print(f"\n### Targets: h {H} m x l {L} m = {n}, landing offset 0.09 m, max {MAX_ITER} iters\n")
    print("| h | l | Stage 1 | Stage 3 naive | Stage 3 from Stage 2 | MuJoCo | touch-down x | impact |")
    print("|---|---|---|---|---|---|---|---|")
    cell = lambda c: f"{c['iters']} it, {c['s']:.1f} s" if c["ok"] else f"FAIL ({c['iters']} it)"  # noqa: E731
    for r in rows:
        if not r["s1_ok"]:
            print(f"| {r['h']} | {r['l']} | FAIL | | | | | |"); continue
        s = r.get("sim")
        sim_s = "-" if s is None else ("SUCCESS" if success(s, r["l"]) else "stands, off target" if s["upright"] else "FELL")
        td = "-" if not s or s["td_x"] is None else f"{s['td_x']:.3f} m"
        imp = "-" if not s or s["impact"] is None else f"{s['impact']:.1f} BW"
        print(f"| {r['h']} | {r['l']} | {r['s1']:.2f} s | {cell(r['naive'])} | {cell(r['stage2'])} | {sim_s} | {td} | {imp} |")
    nv = [r for r in ok1 if r["naive"]["ok"]]; wm = [r for r in ok1 if r["stage2"]["ok"]]
    pipe = [r for r in ok1 if r.get("used")]; stand = [r for r in pipe if success(r["sim"], r["l"])]
    both = [r for r in ok1 if r["naive"]["ok"] and r["stage2"]["ok"]]
    print(f"\nStage 1 success {len(ok1)}/{n} | Stage 3 naive {len(nv)}/{len(ok1)} | Stage 3 from Stage 2 {len(wm)}/{len(ok1)} | "
          f"either (pipeline with fallback) {len(pipe)}/{len(ok1)} | MuJoCo success (stands, within {100 * TOL:.0f} cm) "
          f"{len(stand)}/{len(pipe)} -> end to end {len(stand)}/{n}")
    med = lambda xs: float(np.median(xs)) if xs else float("nan")  # noqa: E731
    print(f"median Stage 3 time over its successes: naive {med([r['naive']['s'] for r in nv]):.1f} s, "
          f"from Stage 2 {med([r['stage2']['s'] for r in wm]):.1f} s")
    if both:
        sp = [r["naive"]["s"] / (r["stage2"]["s"] + r["s2"]) for r in both]
        print(f"speed-up (incl. Stage 2) on {len(both)} common successes: median x{np.median(sp):.1f}, range x{min(sp):.1f}-x{max(sp):.1f}")
    err = [r["sim"]["td_x"] - r["l"] for r in pipe if r["sim"]["td_x"] is not None]
    if err:
        print(f"touch-down x - target: median {1e3 * np.median(err):+.0f} mm, range {1e3 * min(err):+.0f}..{1e3 * max(err):+.0f} mm")

    if errs:
        print(f"\n### Model errors, plan h {NOMINAL[0]} m, l {NOMINAL[1]} m\n")
        print("| error | MuJoCo | touch-down x | impact | take-off v |")
        print("|---|---|---|---|---|")
        for e in errs:
            td = "-" if e["td_x"] is None else f"{e['td_x']:.3f} m"
            imp = "-" if e["impact"] is None else f"{e['impact']:.1f} BW"
            print(f"| {e['name']} | {'STANDING' if e['upright'] else 'FELL'} | {td} | {imp} | ({e['v_to'][0]:.2f}, {e['v_to'][1]:.2f}) m/s |")
    plot(rows, errs, OUT / "evaluation.png")


def plot(rows, errs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    for r in rows:                                                       # (a) outcome map
        if not r["s1_ok"]:
            c, lab = "0.6", "Stage 1 failed"
        elif not r.get("used"):
            c, lab = "tab:red", "Stage 3 failed"
        elif not r["sim"]["upright"]:
            c, lab = "tab:orange", "optimized, fell"
        elif not success(r["sim"], r["l"]):
            c, lab = "gold", "stands, off target"
        else:
            c, lab = "tab:green", "success"
        ax[0].scatter(r["l"], r["h"], s=420, c=c, marker="s", label=lab)
        if r.get("used"):
            ax[0].annotate("S2" if r["used"] == "stage2" else "N", (r["l"], r["h"]), ha="center", va="center", fontsize=8, color="w")
    hd, lb = ax[0].get_legend_handles_labels(); u = dict(zip(lb, hd))
    ax[0].legend(u.values(), u.keys(), fontsize=7, loc="upper center", ncol=2, markerscale=0.4)
    ax[0].set(xlabel="target l [m]", ylabel="target h [m]", title="(a) outcome (S2 = Stage 2 guess, N = naive fallback)",
              xlim=(0.13, 0.67), ylim=(0.02, 0.28))

    ok1 = [r for r in rows if r["s1_ok"]]                                # (b) time naive vs Stage 2
    cap = 1.05 * max([r[k]["s"] for r in ok1 for k in ("naive", "stage2")] or [1])
    for r in ok1:
        x = r["naive"]["s"] if r["naive"]["ok"] else cap
        y = r["stage2"]["s"] + r["s2"] if r["stage2"]["ok"] else cap
        ax[1].scatter(x, y, c="k" if r["naive"]["ok"] and r["stage2"]["ok"] else "tab:red", s=25)
    ax[1].plot([0, cap], [0, cap], "k:", lw=0.8)
    ax[1].axhline(cap, color="tab:red", lw=0.5); ax[1].axvline(cap, color="tab:red", lw=0.5)
    ax[1].set(xlabel="Stage 3, naive guess [s]", ylabel="Stage 2 + Stage 3 from it [s]",
              title="(b) solve time (red line = failed)", xlim=(0, cap * 1.03), ylim=(0, cap * 1.03))

    pipe = [r for r in rows if r.get("used") and r["sim"]["td_x"] is not None]   # (c) landing distance
    for r in pipe:
        ax[2].scatter(r["l"], r["sim"]["td_x"], c="tab:green" if success(r["sim"], r["l"]) else "tab:orange", s=25)
    ax[2].plot([0.15, 0.65], [0.15, 0.65], "k:", lw=0.8)
    ax[2].fill_between([0.15, 0.65], [0.15 - TOL, 0.65 - TOL], [0.15 + TOL, 0.65 + TOL], color="tab:green", alpha=0.08)
    ax[2].set(xlabel="target l [m]", ylabel="touch-down ankle x in MuJoCo [m]", title="(c) landing distance (green = success)")

    if errs:                                                             # (d) model errors
        y = np.arange(len(errs))
        ax[3].barh(y, [1e3 * ((e["td_x"] or np.nan) - NOMINAL[1]) for e in errs],
                   color=["tab:green" if e["upright"] else "tab:orange" for e in errs])
        ax[3].set_yticks(y, [e["name"] for e in errs]); ax[3].invert_yaxis()
        ax[3].set(xlabel="touch-down x - target [mm]", title=f"(d) model errors, plan h {NOMINAL[0]}, l {NOMINAL[1]}")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"figure: {path}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--resim" in sys.argv:            # rerun only the MuJoCo part from the saved trajectories (no solving)
        f = OUT / "targets.json"
        rows = json.loads(f.read_text())
        for r in rows:
            if r.get("used"):
                r["sim"] = sim(OUT / f"traj_h{r['h']}_l{r['l']}.npz", r["l"], tag=f"h {r['h']} l {r['l']}")
        f.write_text(json.dumps(rows, indent=1))
        report(rows, run_model_errors())
        sys.exit(0)
    if "--report" in sys.argv:
        e = OUT / "model_errors.json"
        report(json.loads((OUT / "targets.json").read_text()), json.loads(e.read_text()) if e.exists() else [])
        sys.exit(0)
    rows = run_targets()
    errs = run_model_errors()
    report(rows, errs)
