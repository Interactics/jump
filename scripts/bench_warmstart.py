"""Phase 4 gate: does the Stage 2 guess make Stage 3 faster and more reliable? And does variable scaling?

For each jump target: Stage 1, then Stage 3 from {naive guess, Stage 2 guess} x {default IPOPT, warm-start IPOPT},
unscaled (the Phase 4 runs) and with Stage 3's variable scaling (s3.XSCALE).
Results go to results/warmstart.json. Runs already in the file are kept; only missing ones are solved.

    python scripts/bench_warmstart.py            # all missing runs (unscaled ones take about 20-30 min)
    python scripts/bench_warmstart.py --table
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT = Path(__file__).resolve().parents[1] / "results" / "warmstart.json"
MAX_ITER = 1500
# (h, l, landing offset); None = mirror the take-off
CASES = [(0.10, 0.40, d) for d in (None, 0.12, 0.11, 0.10, 0.09, 0.07, 0.048)] + \
        [(0.05, 0.30, 0.09), (0.10, 0.30, 0.09), (0.15, 0.50, 0.09), (0.20, 0.60, 0.09)]
RUNS = ("naive/default", "naive/warm-opts", "stage2/default", "stage2/warm-opts")      # unscaled (xscale=None)
SCALED = ("naive/scaled", "stage2/scaled", "stage2/scaled+warm-opts")                   # Stage 3 default scaling


def table(rows):
    runs = [k for k in RUNS + SCALED if all(k in r for r in rows)]
    print("| h | l | offset | " + " | ".join(runs) + " | Stage 1 | Stage 2 |")
    print("|---|---|---|" + "---|" * (len(runs) + 2))
    for r in rows:
        cells = [f"{c['iters']} it, {c['s']:.1f} s" if c["ok"] else f"FAIL ({c['iters']} it, {c['s']:.1f} s)" for c in (r[k] for k in runs)]
        print(f"| {r['h']} | {r['l']} | {r['d']} | " + " | ".join(cells) + f" | {r['s1']:.2f} s | {r['s2']:.2f} s |")
    for k in runs:
        ok = [r[k] for r in rows if r[k]["ok"]]
        print(f"{k:24s} success {len(ok)}/{len(rows)} | median over successes: {np.median([c['iters'] for c in ok]):.0f} it, "
              f"{np.median([c['s'] for c in ok]):.1f} s | total time incl. failures {sum(r[k]['s'] for r in rows):.0f} s")
    both = [r for r in rows if r["naive/default"]["ok"] and r["stage2/warm-opts"]["ok"]]
    if both:
        sp = [(r["naive/default"]["s"]) / (r["stage2/warm-opts"]["s"] + r["s2"]) for r in both]
        print(f"speed-up of Stage 3, stage2/warm-opts (incl. Stage 2 time) vs naive/default, {len(both)} common successes: "
              f"median x{np.median(sp):.1f}, range x{min(sp):.1f}-x{max(sp):.1f}")


if __name__ == "__main__":
    old = json.loads(OUT.read_text()) if OUT.exists() else []
    if "--table" in sys.argv:
        table(old)
        sys.exit(0)
    from humanoid_jump.stages import stage1_srmp as s1
    from humanoid_jump.stages import stage2_qp as s2
    from humanoid_jump.stages import stage3_nlp as s3

    rows = []
    for h, l, d in CASES:
        row = next((r for r in old if (r["h"], r["l"], r["d"]) == (h, l, d)), None) or dict(h=h, l=l, d=d)
        ref = s1.launch(h, l, joint=True, land_offset=d)
        row.update(s1=ref["ms"] / 1e3, s1_ok=bool(ref["ok"]))
        if not ref["ok"]:
            print(h, l, d, "Stage 1 failed", flush=True)
            continue
        g = s2.solve(ref)
        row.update(s2=g["ms"] / 1e3, s2_fail=int(g["n_fail"]))
        for name, guess, opts, xs in (("naive/default", None, None, None), ("naive/warm-opts", None, s3.WARM, None),
                                      ("stage2/default", g, None, None), ("stage2/warm-opts", g, s3.WARM, None),
                                      ("naive/scaled", None, None, s3.XSCALE), ("stage2/scaled", g, None, s3.XSCALE),
                                      ("stage2/scaled+warm-opts", g, s3.WARM, s3.XSCALE)):
            if name in row:
                continue
            r = s3.solve(ref, guess=guess, ipopt=opts, max_iter=MAX_ITER, xscale=xs)
            row[name] = dict(ok=bool(r["ok"]), status=r["status"], iters=int(r["iters"]), s=r["ms"] / 1e3, cost=r["cost"])
            print(h, l, d, name, row[name], flush=True)
        rows.append(row)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(rows + [r for r in old if (r["h"], r["l"], r["d"]) not in {(x["h"], x["l"], x["d"]) for x in rows}], indent=1))
    table(rows)
