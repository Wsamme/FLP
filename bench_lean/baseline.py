"""Baselines on bench_lean.

classic : our two-stage UA-FLP solver (bench.methods.ours) on the classic view of the instance -
          an empty rectangle of the building's size, department areas and aspect limits, weights =
          material (trips x equipment rate) + people. Aisles, docks, exits, X-gaps and fixed zones are ignored.
          The result is then checked by bench_lean.evaluate, which shows what a classic solver misses.
ref     : bench_lean.reference (aisle-aware packing + local search), which respects every rule.
rows    : bench_lean.rowsolver (rows along aisles; SA over rows/order, SOCP geometry), which respects every rule.

Usage: python -m bench_lean.baseline METHOD INSTANCE SEED SECONDS OUTDIR
"""
import json, os, sys, time
import numpy as np
from .evaluate import load, evaluate, weights


def classic_problem(inst):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bench.common import Problem
    from flp.instance import Instance
    D = inst["departments"]; n = len(D); B = inst["building"]
    areas = np.array([d["area"] for d in D], float)
    alpha = np.array([d["aspect_max"] for d in D], float)
    Wm, Wp = weights(inst); Wt = Wm + Wp
    pairs = [[i + 1, j + 1, float(Wt[i, j])] for i in range(n) for j in range(i + 1, n) if Wt[i, j] > 0]
    I = Instance(inst["name"], n, float(B["W"]), float(B["H"]), areas, Wt, "Rmax", float(alpha.max()),
                 dict(name=inst["name"], n=n, W=B["W"], H=B["H"], areas=areas.tolist(), weights=pairs))
    free = B["W"] * B["H"] - areas.sum()
    k = int(min(30, max(5, round(free / np.median(areas)))))
    areas_d = np.r_[areas, np.full(k, free / k)]; alpha_d = np.r_[alpha, np.zeros(k)]
    Wt_d = np.pad(Wt, ((0, k), (0, k)))
    pen = 0.05 * np.triu(Wt, 1).sum() * (B["W"] + B["H"])
    return Problem(I, n, float(B["W"]), float(B["H"]), areas, Wt, alpha, n + k, areas_d, Wt_d, alpha_d, pen)


def run(method, name, seed, T):
    inst = load(name); t = time.time()
    if method == "classic":
        from bench.methods import ours
        prob = classic_problem(inst)
        out = ours.solve(prob, seed, T, log=lambda *a: None)
        rects = out["rects"]
    elif method == "ref":
        from .reference import solve
        out = solve(inst, seed, T)
        rects = None if out is None else out["rects"]
    elif method == "rows":
        from .rowsolver import solve
        out = solve(inst, seed, T)
        rects = None if out is None else out["rects"]
    else:
        raise ValueError(method)
    wall = time.time() - t
    if rects is None:
        return dict(method=method, instance=name, seed=seed, wall=wall, rects=None, eval=None)
    ev = evaluate(inst, rects)
    ev.pop("io", None)
    return dict(method=method, instance=name, seed=seed, wall=wall, rects=[list(map(float, r)) for r in rects], eval=ev)


if __name__ == "__main__":
    m, name, seed, T, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    os.makedirs(out, exist_ok=True)
    rec = run(m, name, seed, T)
    json.dump(rec, open(os.path.join(out, f"{m}_{name}_s{seed}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    e = rec["eval"]
    print(m, name, seed, "none" if e is None else f"feasible={e['feasible']} viol={e['n_violations']} path={e['path_total']:.1f} centroid={e['centroid_material'] + e['centroid_people']:.1f}", flush=True)
