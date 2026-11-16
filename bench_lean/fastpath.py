"""Fast copy of the path-cost part of bench_lean.evaluate, for use inside solvers.

Same I/O candidates (edge midpoints whose outside point is free), same grid and shortest paths, and the same
first-improvement I/O search (departments in order, candidates in order, accept when the total drops by > 1e-9),
but the distances are held in one matrix and each candidate change is priced by a vectorised delta instead of a
full recount. evaluate() stays the official checker; test_fastpath() checks that both give the same cost.

path_cost(inst, rects) -> (path_total, path_material, path_people, unreachable_pairs)
"""
import numpy as np
from scipy.sparse.csgraph import dijkstra
from .evaluate import grid, weights

_W = {}


def _weights(inst):
    k = (inst["name"], len(inst["material_flows"]), len(inst["people_flows"]))
    if k not in _W:
        Wm, Wp = weights(inst); _W[k] = (Wm, Wp, Wm + Wp)
    return _W[k]


def path_cost(inst, rects):
    B = inst["building"]; W, H = B["W"], B["H"]; n = len(inst["departments"])
    R = [list(map(float, r)) for r in rects]
    zones = [f["rect"] for f in inst["fixed"] if f.get("dept") is None]
    obst = R + zones
    Ra = np.array(obst)
    x0, y0, x1, y1 = Ra[:, 0] + 1e-7, Ra[:, 1] + 1e-7, Ra[:, 0] + Ra[:, 2] - 1e-7, Ra[:, 1] + Ra[:, 3] - 1e-7
    cand = []
    for i in range(n):
        x, y, w, h = R[i]
        mids = [((x + w / 2, y), (0, -0.5)), ((x + w / 2, y + h), (0, 0.5)), ((x, y + h / 2), (-0.5, 0)), ((x + w, y + h / 2), (0.5, 0))]
        ok = []
        for m, o in mids:
            px, py = m[0] + o[0], m[1] + o[1]
            if px < 0 or py < 0 or px > W or py > H: continue
            if ((px > x0) & (px < x1) & (py > y0) & (py < y1)).any(): continue
            ok.append(m)
        if not ok: ok = [m for m, o in mids]
        cand.append(ok)
    pts = [m for c in cand for m in c]
    G, node, _ = grid(obst, W, H, pts)
    nodes = np.array([node(p) for p in pts])
    src, inv = np.unique(nodes, return_inverse=True)
    dist = dijkstra(G, directed=False, indices=src)
    D = dist[inv][:, nodes]                          # candidate x candidate
    start = np.cumsum([0] + [len(c) for c in cand])[:-1]
    Wm, Wp, Wt = _weights(inst)
    BIG = 10.0 * (W + H)
    Df = np.where(np.isfinite(D), D, BIG)
    ch = start.copy()                                # chosen candidate index (global) per department
    Wpos = np.where(Wt > 0, Wt, 0.0); np.fill_diagonal(Wpos, 0.0)
    imp = True
    while imp:
        imp = False
        for i in range(n):
            cnt = len(cand[i])
            if cnt == 1: continue
            for k in range(cnt):
                g = start[i] + k
                if g == ch[i]: continue
                # change of the total when department i switches from ch[i] to g
                dlt = float(Wpos[i] @ (Df[g, ch] - Df[ch[i], ch]))
                if dlt < -1e-9:
                    ch[i] = g; imp = True
    iu = np.triu_indices(n, 1)
    mask = Wt[iu] > 0
    a, b = iu[0][mask], iu[1][mask]
    d = D[ch[a], ch[b]]
    unr = int((~np.isfinite(d)).sum())
    pm = float((Wm[a, b] * np.where(np.isfinite(d), d, 0.0)).sum())
    pp = float((Wp[a, b] * np.where(np.isfinite(d), d, 0.0)).sum())
    if unr: pm = pp = float("inf")
    return pm + pp, pm, pp, unr


def test_fastpath(names=None, per=20, seed=0):
    """Compare path_cost with evaluate on stored layouts and random perturbations of them."""
    import os, json, time
    from .evaluate import load, evaluate
    here = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(seed); worst = 0.0; t_e = t_f = 0.0; k = 0
    d = os.path.join(here, "results_rows")
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json") or f.startswith("summary"): continue
        r = json.load(open(os.path.join(d, f), encoding="utf-8"))
        if names and r["instance"] not in names: continue
        inst = load(r["instance"])
        for _ in range(per):
            rects = [list(x) for x in r["rects"]]
            for x in rects:                       # jitter positions a little (layout need not stay feasible)
                x[0] += float(rng.normal(0, 0.3)); x[1] += float(rng.normal(0, 0.3))
            t = time.time(); e = evaluate(inst, rects)["path_total"]; t_e += time.time() - t
            t = time.time(); p = path_cost(inst, rects)[0]; t_f += time.time() - t
            if np.isfinite(e) or np.isfinite(p):
                worst = max(worst, abs(e - p) / max(1.0, abs(e))); k += 1
    return dict(checked=k, max_rel_diff=worst, ms_evaluate=1000 * t_e / max(k, 1), ms_fast=1000 * t_f / max(k, 1))


if __name__ == "__main__":
    print(test_fastpath(per=3))
