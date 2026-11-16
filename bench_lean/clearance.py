"""Path cost with a minimum walkway width (robustness check for bench_lean.evaluate; C3 in 改进规划.md).

bench_lean.evaluate (v0.1) lets a path run along the zero-width seam between two touching departments and through
gaps of any width (a 0.2 m slit counts as a walkway). This module recomputes the same cost with a clear width c
(default: the instance's pedestrian aisle width, 1.4 m):
  * every department and fixed zone is grown by c/2 on each side (clipped to the building) for routing,
    so a path keeps c/2 from every obstacle, i.e. it needs a passage at least c wide;
  * each I/O point is the edge midpoint moved out by c/2; the c/2 spur at both ends is added to the distance;
  * an edge is a candidate I/O only if its moved-out point lies in walkable space.
Same I/O search and weights as evaluate. It does not change any feasibility verdict; evaluate stays the official
checker. Pairs that cannot be connected under the clear width are counted, and their cost is left out.

clear_cost(inst, rects, clear=None) -> dict(path_material, path_people, path_total, unreachable_pairs, no_io_depts)
"""
import numpy as np
from scipy.sparse.csgraph import dijkstra
from .evaluate import grid, weights


def clear_cost(inst, rects, clear=None):
    B = inst["building"]; W, H = B["W"], B["H"]; n = len(inst["departments"])
    c = inst["rules"]["pedestrian_aisle"] if clear is None else clear; h = c / 2
    R = [list(map(float, r)) for r in rects]
    zones = [f["rect"] for f in inst["fixed"] if f.get("dept") is None]
    grow = lambda r: [max(0.0, r[0] - h), max(0.0, r[1] - h), min(W, r[0] + r[2] + h) - max(0.0, r[0] - h),
                      min(H, r[1] + r[3] + h) - max(0.0, r[1] - h)]
    obst = [grow(r) for r in R] + [grow(z) for z in zones]
    Ra = np.array(obst)

    def walkable(px, py):
        if px < -1e-9 or py < -1e-9 or px > W + 1e-9 or py > H + 1e-9: return False
        return not bool(((px > Ra[:, 0] + 1e-7) & (px < Ra[:, 0] + Ra[:, 2] - 1e-7) &
                         (py > Ra[:, 1] + 1e-7) & (py < Ra[:, 1] + Ra[:, 3] - 1e-7)).any())
    cand, no_io = [], []
    for i in range(n):
        x, y, w, hh = R[i]
        pts = [(x + w / 2, y - h), (x + w / 2, y + hh + h), (x - h, y + hh / 2), (x + w + h, y + hh / 2)]
        ok = [p for p in pts if walkable(*p)]
        if not ok: no_io.append(i)
        cand.append(ok)
    Wm, Wp = weights(inst); Wt = Wm + Wp
    allp = [p for cs in cand for p in cs]
    G, node, _ = grid(obst, W, H, allp)
    src = sorted({node(p) for p in allp})
    dist = dijkstra(G, directed=False, indices=src) if src else np.zeros((0, 0))
    row = {s: k for k, s in enumerate(src)}
    BIG = 10.0 * (W + H)
    iu = np.triu_indices(n, 1)
    pairs = [(i, j) for i, j in zip(*iu) if Wt[i, j] > 0 and cand[i] and cand[j]]

    def dd(i, a, j, b):
        return dist[row[node(cand[i][a])], node(cand[j][b])] + c
    def ddf(i, a, j, b):
        v = dd(i, a, j, b); return v if np.isfinite(v) else BIG
    ch = [0] * n
    def total():
        return sum(Wt[i, j] * ddf(i, ch[i], j, ch[j]) for i, j in pairs)
    best = total(); imp = True
    while imp:
        imp = False
        for i in range(n):
            for k in range(len(cand[i])):
                if k == ch[i]: continue
                old = ch[i]; ch[i] = k; v = total()
                if v < best - 1e-9: best = v; imp = True
                else: ch[i] = old
    pm = pp = 0.0; unr = 0
    for i, j in pairs:
        d = dd(i, ch[i], j, ch[j])
        if not np.isfinite(d): unr += 1; continue
        pm += Wm[i, j] * d; pp += Wp[i, j] * d
    unr += sum(1 for i, j in zip(*iu) if Wt[i, j] > 0 and (not cand[i] or not cand[j]))
    return dict(clear=c, path_material=float(pm), path_people=float(pp), path_total=float(pm + pp),
                unreachable_pairs=int(unr), no_io_depts=no_io)
