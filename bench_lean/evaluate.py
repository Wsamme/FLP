"""Independent checker and cost accounting for bench_lean instances.

evaluate(inst, rects) -> dict
  rects: one [x, y, w, h] (lower-left corner, metres) per department, in department order.
  Departments with a fixed_rect must be placed exactly there.

Hard checks (each reported with the departments involved):
  area, aspect / min side, inside building, department overlap, overlap with fixed aisles and fixed zones,
  fixed department moved, dock departments touching their dock doors, paint on an exterior wall,
  X-separation gap, fire egress (straight-line distance from every corner to the nearest exit),
  I/O reachable (at least one edge opens to free floor or an aisle).
Cost (only meaningful when feasible):
  each department gets one I/O point at an edge midpoint that opens to free floor; the edge is chosen by
  local search on the total cost. Distance = shortest rectilinear path that does not enter any department
  or fixed zone (aisles and free floor are walkable).
  material = sum trips * equipment rate * path ; people = sum trips * people rate * path.
  centroid = same weights with centroid Manhattan distance (the classic UA-FLP measure), for comparison.
"""
import json, os
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-6


def load(name):
    return json.load(open(os.path.join(HERE, "instances", name + ".json"), encoding="utf-8"))


def r4(v):
    return round(float(v), 4)


def _overlap(a, b):
    dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, dx) * max(0.0, dy)


def _gap(a, b):
    dx = max(0.0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return float(np.hypot(dx, dy))


def weights(inst):
    n = len(inst["departments"])
    Wm = np.zeros((n, n)); Wp = np.zeros((n, n))
    for i, j, t, eq, _ in inst["material_flows"]:
        c = t * inst["handling"][eq]; Wm[i, j] += c; Wm[j, i] += c
    for i, j, t, _ in inst["people_flows"]:
        c = t * inst["people_rate"]; Wp[i, j] += c; Wp[j, i] += c
    return Wm, Wp


def grid(obst, W, H, pts):
    """Rectilinear grid over all obstacle edges and query points; moves that pass through an obstacle interior are cut."""
    xs = sorted({r4(v) for x, y, w, h in obst for v in (x, x + w)} | {0.0, r4(W)} | {r4(p[0]) for p in pts})
    ys = sorted({r4(v) for x, y, w, h in obst for v in (y, y + h)} | {0.0, r4(H)} | {r4(p[1]) for p in pts})
    xs = np.array(xs); ys = np.array(ys); R = np.array(obst) if len(obst) else np.zeros((0, 4))
    x0, y0, x1, y1 = R[:, 0], R[:, 1], R[:, 0] + R[:, 2], R[:, 1] + R[:, 3]

    def inside(px, py):
        px = px[:, None]; py = py[:, None]
        return ((px > x0 + 1e-7) & (px < x1 - 1e-7) & (py > y0 + 1e-7) & (py < y1 - 1e-7)).any(1)
    nx, ny = len(xs), len(ys)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    valid = ~inside(gx.ravel(), gy.ravel()).reshape(nx, ny)
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [], [], []
    mx = ((xs[:-1] + xs[1:]) / 2)[:, None].repeat(ny, 1); my = ys[None, :].repeat(nx - 1, 0)
    ok = valid[:-1, :] & valid[1:, :] & ~inside(mx.ravel(), my.ravel()).reshape(nx - 1, ny)
    a, b = np.nonzero(ok); rows += list(idx[a, b]); cols += list(idx[a + 1, b]); vals += list(xs[a + 1] - xs[a])
    mx = xs[:, None].repeat(ny - 1, 1); my = ((ys[:-1] + ys[1:]) / 2)[None, :].repeat(nx, 0)
    ok = valid[:, :-1] & valid[:, 1:] & ~inside(mx.ravel(), my.ravel()).reshape(nx, ny - 1)
    a, b = np.nonzero(ok); rows += list(idx[a, b]); cols += list(idx[a, b + 1]); vals += list(ys[b + 1] - ys[b])
    G = coo_matrix((np.maximum(np.array(vals, float), 1e-12), (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()
    xi = {v: k for k, v in enumerate(xs.tolist())}; yi = {v: k for k, v in enumerate(ys.tolist())}
    node = lambda p: int(idx[xi[r4(p[0])], yi[r4(p[1])]])
    return G, node, (xs, ys, idx)


def evaluate(inst, rects, with_paths=False):
    B = inst["building"]; W, H = B["W"], B["H"]; D = inst["departments"]; n = len(D)
    R = [list(map(float, r)) for r in rects]
    v = []                     # violations: (check, depts, detail)

    def bad(check, ids, detail):
        v.append(dict(check=check, depts=list(ids), detail=detail))
    for i, d in enumerate(D):
        x, y, w, h = R[i]
        if w * h < d["area"] * (1 - 1e-3): bad("面积", [i], f"{w*h:.1f} < {d['area']}")
        if min(w, h) < d["min_side"] - 1e-6: bad("最短边", [i], f"{min(w, h):.2f} < {d['min_side']}")
        if max(w, h) / max(min(w, h), 1e-9) > d["aspect_max"] * (1 + 1e-6): bad("长宽比", [i], f"{max(w, h)/min(w, h):.2f} > {d['aspect_max']}")
        if x < -TOL or y < -TOL or x + w > W + TOL or y + h > H + TOL: bad("超出厂房", [i], "")
        if d.get("fixed_rect") and max(abs(a - b) for a, b in zip(R[i], d["fixed_rect"])) > 1e-3: bad("固定部门被移动", [i], "")
    for i in range(n):
        for j in range(i + 1, n):
            o = _overlap(R[i], R[j])
            if o > 1e-4: bad("部门重叠", [i, j], f"{o:.2f} m²")
    for a in inst["aisles"]:
        for i in range(n):
            o = _overlap(R[i], a["rect"])
            if o > 1e-4: bad("占用固定通道", [i], f"{a['id']} {o:.1f} m²")
    zones = [f for f in inst["fixed"] if f.get("dept") is None]
    for f in zones:
        for i in range(n):
            o = _overlap(R[i], f["rect"])
            if o > 1e-4: bad("占用固定区域", [i], f"{f['name']} {o:.1f} m²")
    # docks
    for dk in inst["docks"]:
        i = next(k for k, d in enumerate(D) if d.get("dock") == dk["kind"])
        x, y, w, h = R[i]
        for door in dk["doors"]:
            c, half = door["center"], door["width"] / 2
            wall = door["wall"]
            touch = {"W": abs(x) < 1e-3, "E": abs(x + w - W) < 1e-3, "S": abs(y) < 1e-3, "N": abs(y + h - H) < 1e-3}[wall]
            span = (y, y + h) if wall in "WE" else (x, x + w)
            if not (touch and span[0] <= c - half + 1e-3 and span[1] >= c + half - 1e-3):
                bad("未接装卸门", [i], f"{dk['kind']} {wall} 墙 {c}"); break
    for i, d in enumerate(D):
        if d.get("exterior_wall"):
            x, y, w, h = R[i]
            if not (x < 1e-3 or y < 1e-3 or x + w > W - 1e-3 or y + h > H - 1e-3): bad("未靠外墙", [i], "")
    for rel in inst["relations"]:
        if rel[2] == "X":
            g = _gap(R[rel[0]], R[rel[1]])
            if g < rel[4] - 1e-6: bad("X 级隔离", [rel[0], rel[1]], f"净距 {g:.1f} < {rel[4]}")
    ex = np.array([[e["x"], e["y"]] for e in inst["exits"]])
    egress = []
    for i in range(n):
        x, y, w, h = R[i]
        cs = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        dmax = float(np.max(np.min(np.linalg.norm(cs[:, None, :] - ex[None, :, :], axis=2), axis=1)))
        egress.append(dmax)
        if dmax > inst["rules"]["egress_max"] + 1e-6: bad("疏散距离", [i], f"{dmax:.1f} > {inst['rules']['egress_max']}")
    # I/O candidates: edge midpoints that open to free floor or an aisle
    obst = [r for r in R] + [f["rect"] for f in zones]
    Ra = np.array(obst)

    def blocked(px, py):
        if px < 0 or py < 0 or px > W or py > H: return True
        return bool(((px > Ra[:, 0] + 1e-7) & (px < Ra[:, 0] + Ra[:, 2] - 1e-7) & (py > Ra[:, 1] + 1e-7) & (py < Ra[:, 1] + Ra[:, 3] - 1e-7)).any())
    cand, valid_io = [], []
    for i in range(n):
        x, y, w, h = R[i]
        mids = [((x + w / 2, y), (0, -0.5), "S"), ((x + w / 2, y + h), (0, 0.5), "N"), ((x, y + h / 2), (-0.5, 0), "W"), ((x + w, y + h / 2), (0.5, 0), "E")]
        ok = [(m, s) for m, o, s in mids if not blocked(m[0] + o[0], m[1] + o[1])]
        if not ok: bad("出入口不通", [i], "四条边外侧都被挡住"); ok = [(m, s) for m, o, s in mids]
        cand.append(ok); valid_io.append(len(ok))
    feasible = len(v) == 0
    Wm, Wp = weights(inst)
    Wt = Wm + Wp
    C = np.array([[x + w / 2, y + h / 2] for x, y, w, h in R])
    iu = np.triu_indices(n, 1)
    dcen = np.abs(C[iu[0]] - C[iu[1]]).sum(1)
    cen_mat, cen_ppl = float((Wm[iu] * dcen).sum()), float((Wp[iu] * dcen).sum())
    out = dict(feasible=feasible, violations=v, n_violations=len(v),
               checks={k: sum(1 for x in v if x["check"] == k) for k in
                       ["面积", "最短边", "长宽比", "超出厂房", "部门重叠", "占用固定通道", "占用固定区域", "固定部门被移动",
                        "未接装卸门", "未靠外墙", "X 级隔离", "疏散距离", "出入口不通", "物流不连通"]},
               centroid_material=cen_mat, centroid_people=cen_ppl, egress_max=max(egress))
    # path cost (computed even when infeasible, then only indicative)
    pts = [m for c in cand for m, s in c]
    G, node, _ = grid(obst, W, H, pts)
    src = sorted({node(p) for p in pts})
    dist, pred = dijkstra(G, directed=False, indices=src, return_predecessors=True)
    row = {s: k for k, s in enumerate(src)}

    def dd(i, a, j, b):
        return dist[row[node(cand[i][a][0])], node(cand[j][b][0])]
    BIG = 10.0 * (W + H)                                    # stands in for an unreachable pair during the I/O search
    def ddf(i, a, j, b):
        x = dd(i, a, j, b); return x if np.isfinite(x) else BIG
    pairs = [(i, j) for i, j in zip(*iu) if Wt[i, j] > 0]
    ch = [0] * n

    def total(c):
        return sum(Wt[i, j] * ddf(i, c[i], j, c[j]) for i, j in pairs)
    best = total(ch); imp = True
    while imp:
        imp = False
        for i in range(n):
            for k in range(len(cand[i])):
                if k == ch[i]: continue
                old = ch[i]; ch[i] = k; c = total(ch)
                if c < best - 1e-9: best = c; imp = True
                else: ch[i] = old
    pm = sum(Wm[i, j] * dd(i, ch[i], j, ch[j]) for i, j in pairs if Wm[i, j] > 0)
    pp = sum(Wp[i, j] * dd(i, ch[i], j, ch[j]) for i, j in pairs if Wp[i, j] > 0)
    unr = [(int(i), int(j)) for i, j in pairs if not np.isfinite(dd(i, ch[i], j, ch[j]))]
    unreachable = len(unr)
    if unr:
        bad("物流不连通", sorted({k for p in unr for k in p}), f"{len(unr)} 对物流找不到路径")
        out.update(feasible=False, violations=v, n_violations=len(v))
        out["checks"]["物流不连通"] = 1
    out.update(path_material=float(pm), path_people=float(pp), path_total=float(pm + pp), unreachable_pairs=unreachable,
               io=[[*cand[i][ch[i]][0], cand[i][ch[i]][1]] for i in range(n)])
    if with_paths:
        xs, ys, idx = _
        def path(i, a, j, b):
            s, t = node(cand[i][a][0]), node(cand[j][b][0]); P = pred[row[s]]; seq = [t]
            while seq[-1] != s and P[seq[-1]] >= 0: seq.append(P[seq[-1]])
            out_ = []
            for k in reversed(seq):
                ix, iy = divmod(k, len(ys)); out_.append([r4(xs[ix]), r4(ys[iy])])
            return out_
        out["paths"] = [dict(i=int(i), j=int(j), w=float(Wt[i, j]), d=float(dd(i, ch[i], j, ch[j])), path=path(i, ch[i], j, ch[j])) for i, j in pairs]
    return out
