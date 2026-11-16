"""Aisle-aware reference solver: proves each instance has a feasible layout and gives a first cost to beat.

1. Free floor = building minus fixed aisles and fixed zones, cut into rectangular regions.
2. Dock departments go first, against their dock doors; paint goes to a region on the exterior wall.
3. The other departments are packed region by region in process order: each region is filled with columns
   of 1-4 departments that share the column width and the full region height (aspect and min side checked).
4. Local search: swap two departments in the order (or move one), re-pack, keep the change if the layout
   stays feasible and the path cost (bench_lean.evaluate) drops.
"""
import time, math
import numpy as np
from .evaluate import evaluate

ORDER_TYPES = ["dock", "store", "cell", "monument", "line", "area", "support"]


def subtract(region, hole):
    x, y, w, h = region; hx, hy, hw, hh = hole
    ix0, iy0 = max(x, hx), max(y, hy); ix1, iy1 = min(x + w, hx + hw), min(y + h, hy + hh)
    if ix1 <= ix0 + 1e-9 or iy1 <= iy0 + 1e-9: return [region]
    out = []
    if ix0 > x + 1e-9: out.append([x, y, ix0 - x, h])
    if ix1 < x + w - 1e-9: out.append([ix1, y, x + w - ix1, h])
    if iy0 > y + 1e-9: out.append([ix0, y, ix1 - ix0, iy0 - y])
    if iy1 < y + h - 1e-9: out.append([ix0, iy1, ix1 - ix0, y + h - iy1])
    return [r for r in out if r[2] > 0.5 and r[3] > 0.5]


def regions(inst):
    B = inst["building"]; R = [[0.0, 0.0, B["W"], B["H"]]]
    holes = [a["rect"] for a in inst["aisles"]] + [f["rect"] for f in inst["fixed"]]
    for hole in holes:
        R = [q for r in R for q in subtract(r, hole)]
    return R


def ok_shape(d, w, h):
    return min(w, h) >= d["min_side"] - 1e-9 and max(w, h) / min(w, h) <= d["aspect_max"] + 1e-9


def place_dock(inst, d, regs):
    """Put a dock department against its wall, covering all its doors, inside one free region."""
    B = inst["building"]; dk = next(k for k in inst["docks"] if k["kind"] == d["dock"])
    cs = [door["center"] for door in dk["doors"]]; wall = dk["doors"][0]["wall"]; half = dk["doors"][0]["width"] / 2
    lo, hi = min(cs) - half - 1.0, max(cs) + half + 1.0
    along = (lambda r: (r[1], r[1] + r[3])) if wall in "WE" else (lambda r: (r[0], r[0] + r[2]))
    depth_of = (lambda r: r[2]) if wall in "WE" else (lambda r: r[3])
    on_wall = {"W": lambda r: r[0] < 1e-6, "E": lambda r: r[0] + r[2] > B["W"] - 1e-6,
               "S": lambda r: r[1] < 1e-6, "N": lambda r: r[1] + r[3] > B["H"] - 1e-6}[wall]
    reg = next((r for r in regs if on_wall(r) and along(r)[0] <= lo + 1e-6 and along(r)[1] >= hi - 1e-6), None)
    if reg is None: return None
    a0, a1 = along(reg); dmax = depth_of(reg)
    span = hi - lo
    depth = d["area"] / span
    if depth > dmax or depth / span > d["aspect_max"]:
        span = min(a1 - a0, max(span, d["area"] / dmax, math.sqrt(d["area"] / d["aspect_max"])))
        depth = d["area"] / span
    if span / depth > d["aspect_max"]:
        depth = span / d["aspect_max"]
    if depth > dmax + 1e-9 or depth < d["min_side"] - 1e-9: return None
    start = min(max(a0, (lo + hi) / 2 - span / 2), a1 - span)
    start = min(start, lo); start = max(start, hi - span, a0)
    if wall == "W": return [0.0, start, depth, span]
    if wall == "E": return [B["W"] - depth, start, depth, span]
    if wall == "S": return [start, 0.0, span, depth]
    return [start, B["H"] - depth, span, depth]


def pack(inst, order, paint_region=None):
    D = inst["departments"]; n = len(D)
    rects = [None] * n
    regs = regions(inst)
    for i, d in enumerate(D):
        if d.get("fixed_rect"): rects[i] = list(d["fixed_rect"])
    for i, d in enumerate(D):
        if d.get("dock"):
            rects[i] = place_dock(inst, d, regs)
            if rects[i] is None: return None
            regs = [q for r in regs for q in subtract(r, rects[i])]
    pi = next((i for i, d in enumerate(D) if d.get("exterior_wall")), None)
    if pi is not None:
        B = inst["building"]
        cands = [r for r in regs if r[0] < 1e-6 or r[1] < 1e-6 or r[0] + r[2] > B["W"] - 1e-6 or r[1] + r[3] > B["H"] - 1e-6]
        cands = [r for r in cands if r[2] * r[3] >= D[pi]["area"] * 1.2]
        if not cands: return None
        r = cands[(paint_region or 0) % len(cands)]
        h = r[3]; w = D[pi]["area"] / h
        if not ok_shape(D[pi], w, h):
            w = max(D[pi]["min_side"], math.sqrt(D[pi]["area"] / D[pi]["aspect_max"])); h = D[pi]["area"] / w
            if h > r[3] or not ok_shape(D[pi], w, h): return None
        # put it at the end of the region that touches the outer wall
        x = r[0] if r[0] < 1e-6 or r[0] + r[2] < inst["building"]["W"] - 1e-6 else r[0] + r[2] - w
        y = r[1] if r[1] < 1e-6 else r[1] + r[3] - h
        rects[pi] = [x, y, w, h]
        regs = [q for rr in regs for q in subtract(rr, rects[pi])]
    queue = [i for i in order if rects[i] is None]
    H = inst["building"]["H"]
    aisles = [a["rect"] for a in inst["aisles"]]

    def touches(y):        # does the horizontal line y lie on the edge of an aisle?
        return any(abs(y - a[1]) < 1e-6 or abs(y - (a[1] + a[3])) < 1e-6 for a in aisles)
    regs.sort(key=lambda r: (r[1] > H / 2, r[0] if r[1] < H / 2 else -r[0]))
    for (rx, ry, rw, rh) in regs:
        x = rx
        lo_open, hi_open = touches(ry), touches(ry + rh)
        both = lo_open and hi_open
        k = 0
        while k < len(queue):
            i = queue[k]; d = D[i]; A = d["area"]
            placed = None
            # (a) full region height, touches both region edges
            wc = A / rh
            if ok_shape(d, wc, rh) and x + wc <= rx + rw + 1e-9:
                placed = [x, ry, wc, rh]; adv = wc
            # (b) two departments stacked, only when both region edges are aisles
            elif both and k + 1 < len(queue):
                j = queue[k + 1]; A2 = A + D[j]["area"]; wc = A2 / rh
                h1, h2 = A / wc, D[j]["area"] / wc
                if ok_shape(d, wc, h1) and ok_shape(D[j], wc, h2) and x + wc <= rx + rw + 1e-9:
                    rects[i] = [x, ry, wc, h1]; rects[j] = [x, ry + h1, wc, h2]
                    x += wc; queue.pop(k + 1); queue.pop(k); continue
            # (c) shorter than the region, anchored on the aisle side
            if placed is None:
                hc = min(rh, max(d["min_side"], math.sqrt(A)))
                wc = A / hc
                if ok_shape(d, wc, hc) and x + wc <= rx + rw + 1e-9 and (lo_open or hi_open):
                    yy = ry if lo_open else ry + rh - hc
                    placed = [x, yy, wc, hc]; adv = wc
            if placed is None:
                k += 1; continue                 # try the next department in this region
            rects[i] = placed; x += adv; queue.pop(k)
    if queue: return None
    return rects


def default_order(inst):
    D = inst["departments"]
    return sorted(range(len(D)), key=lambda i: (ORDER_TYPES.index(D[i]["type"]) if D[i]["type"] in ORDER_TYPES else 9, i))


def solve(inst, seed=0, time_limit=60.0, log=None):
    rng = np.random.default_rng(seed)
    order = default_order(inst)
    best = None
    for pr in range(8):                       # try paint regions until a feasible start is found
        for attempt in range(200):
            rects = pack(inst, order, pr)
            if rects is not None:
                ev = evaluate(inst, rects)
                if ev["feasible"]:
                    best = (ev["path_total"], order[:], pr, rects, ev); break
            order = order[:]; i, j = rng.integers(len(order), size=2); order[i], order[j] = order[j], order[i]
        if best: break
    if best is None: return None
    t0 = time.time(); it = acc = 0
    cur = best
    while time.time() - t0 < time_limit:
        it += 1
        o = cur[1][:]
        if rng.random() < 0.6:
            i, j = rng.integers(len(o), size=2); o[i], o[j] = o[j], o[i]
        else:
            i, j = rng.integers(len(o), size=2); x = o.pop(i); o.insert(j, x)
        pr = cur[2] if rng.random() < 0.9 else int(rng.integers(8))
        rects = pack(inst, o, pr)
        if rects is None: continue
        ev = evaluate(inst, rects)
        if ev["feasible"] and ev["path_total"] < cur[0] - 1e-9:
            cur = (ev["path_total"], o, pr, rects, ev); acc += 1
    return dict(rects=cur[3], eval=cur[4], iterations=it, accepted=acc)
