"""Dai 2014 on the Phase 6 target grid, compared with the T-ASE pipeline (results/eval/targets.json).

Per target: solve humanoid_jump/dai2014.py (naive guess, max 1500 iters as in Phase 6), then MuJoCo twice:
the planned landing (the paper: the plan ends at rest) and the landing controller from touch-down (as T-ASE).
Success = same criterion as Phase 6 (stands, touch-down within 10 cm of the target).
--unlimited re-runs MuJoCo on the saved plans with the motor torque limits removed (Dai's assumption of sufficient
torque, Sec. II-A), without solving again; stored as sim_unl / sim_lc_unl.

    python scripts/eval_dai2014.py                     # resumes from results/eval_dai2014/targets.json (about 1 h)
    python scripts/eval_dai2014.py --torque-limits     # same with our torque-limit deviation -> results/eval_dai2014_tau/
    python scripts/eval_dai2014.py --unlimited          # MuJoCo again on the saved plans, unlimited torque (about 15 min)
    python scripts/eval_dai2014.py [--torque-limits] --report    # table only
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import H, L, MAX_ITER, success  # noqa: E402  (same grid, iteration cap and success criterion)

ROOT = Path(__file__).resolve().parents[1]
TL = "--torque-limits" in sys.argv
OUT = ROOT / "results" / ("eval_dai2014_tau" if TL else "eval_dai2014")


def sim(path, landing, unlimited=False):
    from humanoid_jump.sim import play_full as pf
    from humanoid_jump.sim.mujoco_runner import summarize
    tr = pf.load(path)
    lg = pf.play(tr, landing=landing, model=pf.unlimited_model() if unlimited else None)
    res = summarize(pf.until_touchdown(tr), lg, f"{path.stem} {'landing ctrl' if landing else 'planned landing'}{', unlimited' if unlimited else ''}")
    td = res.get("td")
    return dict(upright=bool(res.get("upright", False)), td_x=None if td is None else float(td[1]),
                v_to=[float(v) for v in res["v_to"]], sat=float(lg["sat"][lg["t"] <= tr["t"][-1]].mean()))


def run():
    from humanoid_jump import dai2014 as dai
    f = OUT / "targets.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    done = {(r["h"], r["l"]) for r in rows}
    OUT.mkdir(parents=True, exist_ok=True)
    for h in H:
        for l in L:
            if (h, l) in done:
                continue
            s = dai.solve(h, l, max_iter=MAX_ITER, torque_limits=TL)
            row = dict(h=h, l=l, ok=bool(s["ok"]), status=s["status"], iters=int(s["iters"]), s=s["ms"] / 1e3,
                       tau_peak=float(np.max(np.abs(s["tau"]) / dai.TAU_MAX)))
            if s["ok"]:
                path = OUT / f"traj_h{h}_l{l}.npz"
                dai.save(s, path)
                row["sim"], row["sim_lc"] = sim(path, False), sim(path, True)
            print(row, flush=True)
            rows.append(row)
            f.write_text(json.dumps(rows, indent=1))
    return rows


def resim_unlimited():
    f = OUT / "targets.json"
    rows = json.loads(f.read_text())
    for r in rows:
        if r["ok"]:
            path = OUT / f"traj_h{r['h']}_l{r['l']}.npz"
            r["sim_unl"], r["sim_lc_unl"] = sim(path, False, True), sim(path, True, True)
            f.write_text(json.dumps(rows, indent=1))
    return rows


def report(rows):
    tase = {(r["h"], r["l"]): r for r in json.loads((ROOT / "results" / "eval" / "targets.json").read_text())}

    def tase_time(r):            # Stage 1 + Stage 2 + the Stage 3 solve that was used (Phase 6 definition)
        return None if not r.get("used") else r["s1"] + r["s2"] + r[r["used"]]["s"]

    print("| h | l | T-ASE time | T-ASE MuJoCo | Dai time (iters) | Dai peak tau/limit | Dai, H1-2 limits | Dai, unlimited torque | Dai, unlimited + landing ctrl |")
    print("|---|---|---|---|---|---|---|---|---|")
    cell = lambda s, l: "-" if s is None else ("SUCCESS" if success(s, l) else "stands, off target" if s["upright"] else "FELL") \
        + ("" if s is None or s["td_x"] is None else f" ({s['td_x']:.2f} m)")  # noqa: E731
    for r in rows:
        t = tase[(r["h"], r["l"])]
        tt = tase_time(t)
        tase_s = "FAIL" if tt is None else f"{tt:.1f} s"
        dai_s = f"{r['s']:.1f} s ({r['iters']})" if r["ok"] else f"FAIL ({r['iters']} it)"
        print(f"| {r['h']} | {r['l']} | {tase_s} | {cell(t.get('sim'), r['l'])} | {dai_s} | "
              f"{r['tau_peak']:.2f} | {cell(r.get('sim'), r['l'])} | {cell(r.get('sim_unl'), r['l'])} | {cell(r.get('sim_lc_unl'), r['l'])} |")
    n = len(rows)
    ok = [r for r in rows if r["ok"]]
    t_ok = [tase[(r["h"], r["l"])] for r in rows if tase_time(tase[(r["h"], r["l"])]) is not None]
    med = lambda x: float(np.median(x)) if x else float("nan")  # noqa: E731
    print(f"\nOptimization success: T-ASE {len(t_ok)}/{n}, Dai {len(ok)}/{n} | median time over successes: "
          f"T-ASE {med([tase_time(t) for t in t_ok]):.1f} s, Dai {med([r['s'] for r in ok]):.1f} s "
          f"(median iters {med([r['iters'] for r in ok]):.0f})")
    print(f"End to end (MuJoCo success): T-ASE {sum(success(t.get('sim'), t['l']) for t in tase.values() if (t['h'], t['l']) in {(r['h'], r['l']) for r in rows})}/{n}, "
          f"Dai with H1-2 limits {sum(success(r.get('sim'), r['l']) for r in rows)}/{n} (landing ctrl {sum(success(r.get('sim_lc'), r['l']) for r in rows)}/{n}), "
          f"Dai with unlimited torque {sum(success(r.get('sim_unl'), r['l']) for r in rows)}/{n} (landing ctrl {sum(success(r.get('sim_lc_unl'), r['l']) for r in rows)}/{n}) | "
          f"Dai plans over a torque limit: {sum(r['tau_peak'] > 1 for r in ok)}/{len(ok)}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(json.loads((OUT / "targets.json").read_text()))
    else:
        report(resim_unlimited() if "--unlimited" in sys.argv else run())
