"""Construction heuristic in the CORELAP spirit + flexible-bay placement (deterministic; seed ignored).

Order (CORELAP-style, Lee & Moore 1967, here on the flow/weight matrix instead of REL codes):
  1. first department = largest total closeness TCR_i = sum_j w_ij (ties: lower index);
  2. each next department = unplaced one with the largest sum of weights to the already ordered ones;
     ties: larger TCR, then lower index.
Placement (flexible bay structure, Tong 1991): vertical bays left to right, each bay of width
(bay area) / H, its departments stacked from top to bottom in order, each of height area / bay width.
For every number of bays k = 1..n the ordered list is split into contiguous bays by a greedy rule:
bay b (b = 1..k-1) is closed as soon as the cumulative area of all departments so far is
>= b/k of the total department area (each bay holds at least one department, and enough departments
are left for the remaining bays). The best layout with every department within its aspect limit
(strict) is returned; if none is feasible, the one with the least total aspect excess is returned
(meta.feasible = False).
Both bay orientations are tried (our addition): vertical bays as above, and horizontal bays (the same
construction on the transposed floor, mirrored back over the diagonal: bays are rows from the bottom,
departments placed right to left). With vertical bays only, AB20 has no feasible layout (best total aspect
excess 0.82), which is why the transposed variant was added; meta.orientation says which one won.
Our choices (not from a paper): the weight-sum ordering as CORELAP substitute, the greedy split rule,
the two orientations, bays packed from x = 0 (with empty floor area, as in SC30, the free strip is at the end).
"""
import numpy as np


def corelap_order(Wt):
    n = len(Wt); tcr = Wt.sum(1)
    first = min(range(n), key=lambda i: (-tcr[i], i))
    order = [first]; left = set(range(n)) - {first}
    while left:
        nxt = min(left, key=lambda i: (-Wt[i, order].sum(), -tcr[i], i))
        order.append(nxt); left.remove(nxt)
    return order


def bay_layout(prob, order, k):
    a = prob.areas; tot = a[order].sum(); n = len(order)
    bays, cur, cum = [], [], 0.0
    for p, d in enumerate(order):
        cur.append(d); cum += a[d]
        b = len(bays) + 1                     # index of the bay being filled
        remaining = n - p - 1; bays_left = k - b
        if b < k and (cum >= b / k * tot - 1e-12 or remaining == bays_left):
            bays.append(cur); cur = []
    if cur: bays.append(cur)
    rects = [None] * prob.n; x = 0.0
    for bay in bays:
        bw = a[bay].sum() / prob.H; y = prob.H
        for d in bay:
            h = a[d] / bw; y -= h
            rects[d] = [x, max(y, 0.0), bw, h]
        x += bw
    return rects, len(bays)


def cost_excess(prob, rects):
    R = np.array(rects); c = R[:, :2] + R[:, 2:] / 2
    D = np.abs(c[:, None, 0] - c[None, :, 0]) + np.abs(c[:, None, 1] - c[None, :, 1])
    cost = float(np.triu(prob.Wt * D, 1).sum())
    r = np.maximum(R[:, 2] / R[:, 3], R[:, 3] / R[:, 2])
    ex = float(np.where(prob.alpha > 0, np.maximum(r - prob.alpha, 0), 0).sum())
    return cost, ex


def solve(prob, seed, time_limit, log):
    import copy, time
    t0 = time.time()
    order = corelap_order(prob.Wt)
    tp = copy.copy(prob); tp.W, tp.H = prob.H, prob.W
    best = None
    for orient, P in (("vertical", prob), ("horizontal", tp)):
        for k in range(1, prob.n + 1):
            rects, nb = bay_layout(P, order, k)
            if orient == "horizontal":
                rects = [[y, x, h, w] for x, y, w, h in rects]
            c, ex = cost_excess(prob, rects)
            key = (ex > 0, ex, c)
            if best is None or key < best[0]:
                best = (key, rects, k, nb, c, ex, orient)
    key, rects, k, nb, c, ex, orient = best
    return {"rects": rects, "meta": dict(order=[int(i) + 1 for i in order], orientation=orient, k=k, bays=nb,
                                         own_cost=c, excess=ex, feasible=ex == 0,
                                         trace=[[round(time.time() - t0, 4), c if ex == 0 else None]])}
