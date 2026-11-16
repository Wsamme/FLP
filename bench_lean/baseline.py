"""Baselines on bench_lean.

classic : our two-stage UA-FLP solver (bench.methods.ours) on the classic view of the instance -
          an empty rectangle of the building's size, department areas and aspect limits, weights =
          material (trips x equipment rate) + people. Aisles, docks, exits, X-gaps and fixed zones are ignored.
          The result is then checked by bench_lean.evaluate, which shows what a classic solver misses.
ref     : bench_lean.reference (aisle-aware packing + local search), which respects every rule.
rows    : bench_lean.rowsolver (rows along aisles; SA over rows/order, SOCP geometry), which respects every rule.
legal   : classic for half the time, then its layout is mapped onto the rows (rowsolver.Model.from_rects) and the
          row solver runs only until the first feasible layout ("ignore the real elements, repair afterwards").

Usage: python -m bench_lean.baseline METHOD INSTANCE SEED SECONDS OUTDIR [TAG] [key=value ...]
TAG names the configuration in the result file (default METHOD); key=value pairs are passed to the solver.
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


def _commit():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def _val(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return {"true": True, "false": False}.get(v.lower(), v)


def run(method, name, seed, T, opts=None, tag=None):
    opts = opts or {}
    inst = load(name); t = time.time(); stats = {}
    if method == "classic":
        from bench.methods import ours
        prob = classic_problem(inst)
        out = ours.solve(prob, seed, T, log=lambda *a: None)
        rects = out["rects"]
    elif method == "ref":
        from .reference import solve
        out = solve(inst, seed, T, **opts)
        rects = None if out is None else out["rects"]
        if out: stats = dict(iterations=out["iterations"], accepted=out["accepted"])
    elif method == "legal":
        from bench.methods import ours
        from .rowsolver import solve
        t1 = time.time()
        cl = ours.solve(classic_problem(inst), seed, T * 0.5, log=lambda *a: None)
        left = T - (time.time() - t1)
        out = solve(inst, seed, max(5.0, left), init_rects=cl["rects"], stop_feasible=True, **opts)
        rects = None if out is None else out["rects"]
        if out: stats = {k: v for k, v in out.items() if k not in ("rects", "eval")}
    elif method == "rows":
        from .rowsolver import solve
        out = solve(inst, seed, T, **opts)
        rects = None if out is None else out["rects"]
        if out: stats = {k: v for k, v in out.items() if k not in ("rects", "eval")}
    else:
        raise ValueError(method)
    wall = time.time() - t
    meta = dict(method=tag or method, solver=method, options=opts, budget=T, commit=_commit(), instance=name, seed=seed, wall=wall)
    if rects is None:
        return dict(meta, rects=None, eval=None, stats=stats)
    ev = evaluate(inst, rects)
    ev.pop("io", None)
    return dict(meta, rects=[list(map(float, r)) for r in rects], eval=ev, stats=stats)


if __name__ == "__main__":
    m, name, seed, T, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    rest = sys.argv[6:]
    tag = rest.pop(0) if rest and "=" not in rest[0] else m
    opts = {k: _val(v) for k, v in (a.split("=", 1) for a in rest)}
    os.makedirs(out, exist_ok=True)
    rec = run(m, name, seed, T, opts, tag)
    conv = lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))
    path = os.path.join(out, f"{tag}_{name}_s{seed}.json")
    with open(path + ".tmp", "w", encoding="utf-8") as fh:        # write then rename: no half-written results
        json.dump(rec, fh, ensure_ascii=False, default=conv)
    os.replace(path + ".tmp", path)
    e = rec["eval"]
    print(tag, name, seed, f"wall={rec['wall']:.1f}", "none" if e is None else f"feasible={e['feasible']} viol={e['n_violations']} path={e['path_total']:.1f} centroid={e['centroid_material'] + e['centroid_people']:.1f}", flush=True)
