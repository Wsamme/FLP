"""Row-based two-stage solver for bench_lean (discrete part by SA, continuous geometry by SOCP).

Representation (rows along aisles):
  The fixed aisles cut the free floor into regions (reference.regions). Every region edge that lies on a
  horizontal aisle carries one row; a region with aisles on both sides carries two rows that share the
  region depth through one split variable s. Each department belongs to one row and sits against that
  row's aisle edge, so its I/O point is always on an aisle and every flow is connected.
  A fixed department (brownfield office) keeps its rectangle; a 2 m keep-clear corridor from its edge
  midpoint to the nearest horizontal aisle is cut out of the free floor, and that point is its I/O.

SOCP (rows and order fixed), per department x, w, d (along the aisle, width, depth):
  w * d >= A (rotated cone), aspect and min side (linear), order inside the row x_{k+1} >= x_k + w_k,
  row inside its region, d <= s / d <= depth - s for two rows sharing a region,
  dock departments: S/N doors -> full region depth, x covers the doors; W/E doors -> first/last in the row,
  touching the wall, depth covers the doors. Paint: touches an outer wall (row end on the W/E wall, or full
  depth when the row's far side is the outer wall). Same-row X pairs: x_j >= x_i + w_i + gap.
  Objective = sum_ij weight_ij * distance along the aisle network between I/O points
  (I/O = midpoint of the aisle-side edge). Same aisle: |cx_i - cx_j| + |dy|; different horizontal aisles:
  |cx_i - c| + |cx_j - c| + |dy| through vertical aisle c. Both convex.
  X pairs in different rows and fire egress are checked after the SOCP; failing layouts are rejected.

SA moves: swap two departments; move one department to another row/position; swap neighbours in a row.
  W/E dock departments stay at their row end, S/N dock departments stay in their row.
Search uses the SOCP objective; new best layouts are re-scored with bench_lean.evaluate (true path cost,
all checks) and the best feasible one under the true cost is returned.
"""
import time, math
import numpy as np
import scipy.sparse as sp
import clarabel
from .evaluate import evaluate, weights, _gap
from .reference import regions as base_regions, subtract, default_order

E = 1e-6
CORRIDOR = 2.0


class Model:
    def __init__(self, inst):
        self.inst = inst
        B = inst["building"]; self.W, self.H = float(B["W"]), float(B["H"])
        self.D = D = inst["departments"]; self.n = n = len(D)
        Wm, Wp = weights(inst); self.Wt = Wm + Wp
        self.pairs = [(i, j, float(self.Wt[i, j])) for i in range(n) for j in range(i + 1, n) if self.Wt[i, j] > 0]
        self.ha = [a["rect"] for a in inst["aisles"] if a["rect"][2] >= a["rect"][3]]
        self.conn = [a["rect"][0] + a["rect"][2] / 2 for a in inst["aisles"] if a["rect"][2] < a["rect"][3]]
        self.ms = [d["min_side"] + 1e-4 for d in D]
        self.am = [d["aspect_max"] * (1 - 1e-5) for d in D]
        self.A = [d["area"] * (1 + 1e-4) for d in D]
        self.mindepth = [max(self.ms[i], math.sqrt(self.A[i] / self.am[i])) for i in range(n)]
        # fixed departments
        self.fixed = {}
        corridors = []
        for i, d in enumerate(D):
            if d.get("fixed_rect"):
                f = self._fixed_io(d["fixed_rect"]); self.fixed[i] = f
                if f["corridor"]: corridors.append(f["corridor"])
        regs = base_regions(inst)
        for c in corridors:
            regs = [q for r in regs for q in subtract(r, c)]
        self.rows, self.groups = [], []
        for r in regs:
            self._add_rows(r)
        self.mate = [None] * len(self.rows)
        for ri, r in enumerate(self.rows):
            for rj, q in enumerate(self.rows):
                if rj != ri and r["group"] is not None and q["group"] == r["group"]: self.mate[ri] = rj
        # X pairs
        self.xpairs = {}
        for rel in inst["relations"]:
            if rel[2] == "X":
                self.xpairs[(rel[0], rel[1])] = rel[4]; self.xpairs[(rel[1], rel[0])] = rel[4]
        self.exits = np.array([[e["x"], e["y"]] for e in inst["exits"]])
        self.egress = inst["rules"]["egress_max"]
        # docks and paint
        self.dock = {}
        for i, d in enumerate(D):
            if d.get("dock"):
                spec = self._dock_row(i, d)
                if spec is None: raise RuntimeError(f"no row for dock {d['name']}")
                self.dock[i] = spec
        self.paint = [i for i, d in enumerate(D) if d.get("exterior_wall")]
        self.movers = [i for i in range(n) if i not in self.fixed and i not in self.dock]
        self.placed = [i for i in range(n) if i not in self.fixed]
        self.penalty = 2.0 * sum(w for _, _, w in self.pairs)      # cost of one metre of shortfall

    # ---------- geometry set-up ----------
    def _aisle_at(self, y, x0, x1, top_edge):
        for k, a in enumerate(self.ha):
            yy = a[1] + a[3] if top_edge else a[1]
            if abs(yy - y) < E and a[0] < x1 - E and a[0] + a[2] > x0 + E:
                return k
        return None

    def _fixed_io(self, rect):
        x, y, w, h = rect; mx = x + w / 2
        best = None
        for k, a in enumerate(self.ha):
            if not (a[0] <= mx <= a[0] + a[2]): continue
            if a[1] + a[3] <= y + E:
                L = y - (a[1] + a[3]); cand = (L, k, a[1] + a[3], [mx - CORRIDOR / 2, a[1] + a[3], CORRIDOR, L])
            elif a[1] >= y + h - E:
                L = a[1] - (y + h); cand = (L, k, a[1], [mx - CORRIDOR / 2, y + h, CORRIDOR, L])
            else:
                continue
            if best is None or cand[0] < best[0]: best = cand
        if best is None:
            return dict(rect=rect, x=mx, y=y + h / 2, aisle=None, extra=0.0, corridor=None)
        L, k, ya, cor = best
        return dict(rect=rect, x=mx, y=ya, aisle=k, extra=L, corridor=cor if L > E else None)

    def _add_rows(self, r):
        rx, ry, rw, rh = r
        if rh < 4.0 - E or rw < 4.0 - E: return
        kb = self._aisle_at(ry, rx, rx + rw, top_edge=True)
        kt = self._aisle_at(ry + rh, rx, rx + rw, top_edge=False)
        g = None
        if kb is not None and kt is not None:
            g = len(self.groups); self.groups.append(rh)
        base = dict(x0=rx, x1=rx + rw, ry=ry, rh=rh, group=g, wallW=rx < E, wallE=rx + rw > self.W - E)
        if kb is not None:
            self.rows.append(dict(base, side="B", y=ry, aisle=kb, far_wall=(g is None and ry + rh > self.H - E)))
        if kt is not None:
            self.rows.append(dict(base, side="T", y=ry + rh, aisle=kt, far_wall=(g is None and ry < E)))

    def _dock_row(self, i, d):
        dk = next(k for k in self.inst["docks"] if k["kind"] == d["dock"])
        cs = [door["center"] for door in dk["doors"]]; half = dk["doors"][0]["width"] / 2
        lo, hi = min(cs) - half - 1e-3, max(cs) + half + 1e-3
        wall = dk["doors"][0]["wall"]
        best = None
        for ri, R in enumerate(self.rows):
            if wall in "SN":
                ok = (R["ry"] < E if wall == "S" else R["ry"] + R["rh"] > self.H - E) and R["group"] is None
                if ok and R["x0"] <= lo + E and R["x1"] >= hi - E:
                    return dict(row=ri, mode=wall, lo=lo, hi=hi, dreq=R["rh"])
            else:
                ok = R["wallW"] if wall == "W" else R["wallE"]
                if not ok or R["ry"] > lo + E or R["ry"] + R["rh"] < hi - E: continue
                dreq = hi - R["ry"] if R["side"] == "B" else R["ry"] + R["rh"] - lo
                if dreq <= R["rh"] + E and (best is None or dreq < best["dreq"]):
                    best = dict(row=ri, mode=wall, lo=lo, hi=hi, dreq=max(dreq, 0.0))
        return best

    # ---------- state helpers ----------
    def normalize(self, state):
        for i, s in self.dock.items():
            seq = state[s["row"]]
            if s["mode"] == "W" and seq[0] != i: seq.remove(i); seq.insert(0, i)
            if s["mode"] == "E" and seq[-1] != i: seq.remove(i); seq.append(i)
        return state

    def overflow(self, state):
        """Necessary-condition check of the rows: excess width (m) + 10 x missing depth (m); 0 = may be feasible."""
        R = self.rows
        need = [max((self.mindepth[i] for i in seq), default=0.0) for seq in state]
        ov = 0.0
        for ri, seq in enumerate(state):
            if not seq: continue
            r = R[ri]; cap = r["rh"]
            if r["group"] is not None:
                other = self.mate[ri]
                if other is not None: cap -= need[other]
            if cap < need[ri] - E:
                ov += 10.0 * (need[ri] - cap); cap = need[ri]
            tot = 0.0
            for i in seq:
                if i in self.dock and self.dock[i]["mode"] in "SN":
                    tot += max(self.A[i] / r["rh"], self.dock[i]["hi"] - self.dock[i]["lo"], self.ms[i])
                else:
                    tot += max(self.ms[i], self.A[i] / cap, math.sqrt(self.A[i] / self.am[i]))
            ov += max(0.0, tot - (r["x1"] - r["x0"]))
        return ov

    def capacity_ok(self, state):
        return self.overflow(state) <= E

    # ---------- SOCP ----------
    def socp(self, state):
        R = self.rows
        pos = {}
        for ri, seq in enumerate(state):
            for k, i in enumerate(seq): pos[i] = (ri, k)
        vi = {}; nv = 0
        for i in self.placed:
            vi[i] = nv; nv += 3
        used = {}
        for ri, seq in enumerate(state):
            g = R[ri]["group"]
            if g is not None and seq: used.setdefault(g, []).append(ri)
        gvar = {}
        for g, rs in used.items():
            if len(rs) == 2: gvar[g] = nv; nv += 1
        ineq_r, ineq_c, ineq_v, ineq_b = [], [], [], []
        eq_r, eq_c, eq_v, eq_b = [], [], [], []
        cnt = [0, 0]

        def le(coefs, b):            # sum coef * z <= b
            k = cnt[0]; cnt[0] += 1
            for c, v in coefs: ineq_r.append(k); ineq_c.append(c); ineq_v.append(v)
            ineq_b.append(b)

        def eq(coefs, b):
            k = cnt[1]; cnt[1] += 1
            for c, v in coefs: eq_r.append(k); eq_c.append(c); eq_v.append(v)
            eq_b.append(b)
        for ri, seq in enumerate(state):
            r = R[ri]; m = len(seq)
            for k, i in enumerate(seq):
                x, w, d = vi[i], vi[i] + 1, vi[i] + 2
                ms, a = self.ms[i], self.am[i]
                le([(w, -1)], -ms); le([(d, -1)], -ms)
                le([(w, 1), (d, -a)], 0.0); le([(d, 1), (w, -a)], 0.0)
                g = r["group"]
                if g in gvar:
                    if r["side"] == "B": le([(d, 1), (gvar[g], -1)], 0.0)
                    else: le([(d, 1), (gvar[g], 1)], r["rh"])
                else:
                    le([(d, 1)], r["rh"])
                if k == 0: le([(x, -1)], -r["x0"])
                else:
                    p = seq[k - 1]; gap = self.xpairs.get((p, i), 0.0)
                    le([(vi[p], 1), (vi[p] + 1, 1), (x, -1)], -gap)
                if k == m - 1: le([(x, 1), (w, 1)], r["x1"])
                for kk in range(k + 2, m):                 # X pairs further apart in the same row
                    j = seq[kk]; gap = self.xpairs.get((i, j))
                    if gap: le([(x, 1), (w, 1), (vi[j], -1)], -gap)
                if i in self.dock:
                    s = self.dock[i]
                    if s["mode"] in "SN":
                        le([(d, -1)], -r["rh"]); le([(x, 1)], s["lo"]); le([(x, -1), (w, -1)], -s["hi"])
                    elif s["mode"] == "W":
                        if k != 0: return None
                        eq([(x, 1)], r["x0"]); le([(d, -1)], -s["dreq"])
                    else:
                        if k != m - 1: return None
                        eq([(x, 1), (w, 1)], r["x1"]); le([(d, -1)], -s["dreq"])
                if i in self.paint:
                    if k == 0 and r["wallW"]: eq([(x, 1)], r["x0"])
                    elif k == m - 1 and r["wallE"]: eq([(x, 1), (w, 1)], r["x1"])
                    elif r["far_wall"]: le([(d, -1)], -r["rh"])
                    else: return None
        # objective along the aisle network
        q_extra = {}; const = 0.0

        def newvar():
            nonlocal nv
            nv += 1; return nv - 1

        def info(i):
            if i in self.fixed:
                f = self.fixed[i]; return None, f["x"], f["aisle"], f["y"], f["extra"]
            ri, _ = pos[i]; r = R[ri]
            return vi[i], 0.0, r["aisle"], r["y"], 0.0
        uvar = {}
        for i, j, wt in self.pairs:
            vi_i, ci, ai, yi, ei = info(i); vj, cj, aj, yj, ej = info(j)
            const += wt * (abs(yi - yj) + ei + ej)
            if vi_i is None and vj is None:
                if ai == aj or not self.conn: const += wt * abs(ci - cj)
                else: const += wt * min(abs(ci - c) + abs(cj - c) for c in self.conn)
                continue
            if ai == aj or ai is None or aj is None or not self.conn:
                t = newvar(); q_extra[t] = q_extra.get(t, 0.0) + wt
                lhs = []
                if vi_i is not None: lhs += [(vi_i, 1), (vi_i + 1, 0.5)]
                if vj is not None: lhs += [(vj, -1), (vj + 1, -0.5)]
                cst = ci - cj
                le(lhs + [(t, -1)], -cst)
                le([(c, -v) for c, v in lhs] + [(t, -1)], cst)
            else:
                ri_c = (ci if vi_i is None else (R[pos[i][0]]["x0"] + R[pos[i][0]]["x1"]) / 2)
                rj_c = (cj if vj is None else (R[pos[j][0]]["x0"] + R[pos[j][0]]["x1"]) / 2)
                c = min(self.conn, key=lambda cc: abs(ri_c - cc) + abs(rj_c - cc))
                for vv, cc0 in ((vi_i, ci), (vj, cj)):
                    if vv is None: const += wt * abs(cc0 - c); continue
                    key = (vv, c)
                    if key not in uvar:
                        u = newvar(); uvar[key] = u
                        le([(vv, 1), (vv + 1, 0.5), (u, -1)], c)
                        le([(vv, -1), (vv + 1, -0.5), (u, -1)], -c)
                    q_extra[uvar[key]] = q_extra.get(uvar[key], 0.0) + wt
        q = np.zeros(nv)
        for k, v in q_extra.items(): q[k] += v
        # SOC rows: s = b - A z, s = (w + d, w - d, 2 sqrt(A))
        soc_r, soc_c, soc_v, soc_b = [], [], [], []
        for t, i in enumerate(self.placed):
            w, d = vi[i] + 1, vi[i] + 2; b0 = 3 * t
            soc_r += [b0, b0, b0 + 1, b0 + 1]; soc_c += [w, d, w, d]; soc_v += [-1, -1, -1, 1]
            soc_b += [0.0, 0.0, 2 * math.sqrt(self.A[i])]
        ne, ni, ns = cnt[1], cnt[0], 3 * len(self.placed)
        rows = np.r_[np.array(eq_r, int), ne + np.array(ineq_r, int), ne + ni + np.array(soc_r, int)]
        cols = np.r_[np.array(eq_c, int), np.array(ineq_c, int), np.array(soc_c, int)]
        vals = np.r_[np.array(eq_v, float), np.array(ineq_v, float), np.array(soc_v, float)]
        Amat = sp.csc_matrix((vals, (rows, cols)), shape=(ne + ni + ns, nv))
        b = np.r_[np.array(eq_b, float), np.array(ineq_b, float), np.array(soc_b, float)]
        cones = ([clarabel.ZeroConeT(ne)] if ne else []) + [clarabel.NonnegativeConeT(ni)] + \
                [clarabel.SecondOrderConeT(3)] * len(self.placed)
        st = clarabel.DefaultSettings(); st.verbose = False
        P = sp.csc_matrix((nv, nv))
        sol = clarabel.DefaultSolver(P, q, Amat, b, cones, st).solve()
        if str(sol.status) not in ("Solved", "AlmostSolved", "SolverStatus.Solved"):
            return None
        z = np.array(sol.x)
        rects = [None] * self.n
        for i, f in self.fixed.items(): rects[i] = list(map(float, f["rect"]))
        for ri, seq in enumerate(state):
            r = R[ri]
            for i in seq:
                x, w, d = z[vi[i]], z[vi[i] + 1], z[vi[i] + 2]
                y = r["ry"] if r["side"] == "B" else r["ry"] + r["rh"] - d
                rects[i] = [float(x), float(y), float(w), float(d)]
        return float(sol.obj_val) + const, rects

    def shortfall(self, rects):
        """X gaps between rows and fire egress, checked after the SOCP: total shortfall in metres (0 = ok)."""
        v = 0.0
        for (i, j), gap in self.xpairs.items():
            if i < j: v += max(0.0, gap - _gap(rects[i], rects[j]))
        for x, y, w, h in rects:
            cs = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            v += max(0.0, float(np.max(np.min(np.linalg.norm(cs[:, None, :] - self.exits[None], axis=2), axis=1))) - self.egress)
        return v

    def score(self, state):
        """(search cost, rects, ok) or None when the rows cannot hold the departments.
        search cost = SOCP objective + penalty x shortfall; ok = no shortfall."""
        if not self.capacity_ok(state): return None
        out = self.socp(state)
        if out is None: return None
        v = self.shortfall(out[1])
        return out[0] + self.penalty * v, out[1], v <= 1e-6

    # ---------- initial solution ----------
    def initial(self, rng, tries=20000):
        R = self.rows; state = [[] for _ in R]
        for i, s in self.dock.items(): state[s["row"]].append(i)
        def ctr(ri): return np.array([(R[ri]["x0"] + R[ri]["x1"]) / 2, R[ri]["ry"] + R[ri]["rh"] / 2])
        recv = next((s["row"] for i, s in self.dock.items() if self.D[i]["dock"] == "receiving"), 0)
        ship = next((s["row"] for i, s in self.dock.items() if self.D[i]["dock"] == "shipping"), 0)
        key = lambda ri: np.abs(ctr(ri) - ctr(recv)).sum() - np.abs(ctr(ri) - ctr(ship)).sum()
        order_rows = sorted(range(len(R)), key=key)
        used = [0.0] * len(R)
        for ri, seq in enumerate(state):
            for i in seq: used[ri] += self.A[i] / R[ri]["rh"]
        def est(i, ri):
            cap = R[ri]["rh"] / (2 if R[ri]["group"] is not None else 1)
            return max(self.ms[i], self.A[i] / max(cap, 1e-6), math.sqrt(self.A[i] / self.am[i])) + 0.5
        # paint first, on a row where it can touch the outer wall
        for i in self.paint:
            ok = [ri for ri in range(len(R)) if R[ri]["wallW"] or R[ri]["wallE"] or R[ri]["far_wall"]]
            ri = min(ok, key=lambda q: used[q] / (R[q]["x1"] - R[q]["x0"]))
            if R[ri]["wallW"]: state[ri].insert(0, i)
            else: state[ri].append(i)
            used[ri] += est(i, ri)
        for i in [i for i in default_order(self.inst) if i in self.movers and i not in self.paint]:
            for ri in order_rows:
                if used[ri] + est(i, ri) <= R[ri]["x1"] - R[ri]["x0"]:
                    seq = state[ri]
                    k = len(seq)
                    if seq and seq[-1] in self.dock and self.dock[seq[-1]]["mode"] == "E": k -= 1
                    if seq and seq[-1] in self.paint and R[ri]["wallE"] and not R[ri]["wallW"]: k = min(k, len(seq) - 1)
                    seq.insert(k, i); used[ri] += est(i, ri); break
            else:
                ri = min(range(len(R)), key=lambda q: used[q] / (R[q]["x1"] - R[q]["x0"]))
                state[ri].insert(len(state[ri]) // 2, i); used[ri] += est(i, ri)
        self.normalize(state)
        ov = self.overflow(state); out = self.score(state) if ov <= E else None
        k = 0
        while out is None and k < tries:           # repair: descend on row overflow, then need a SOCP solution
            k += 1
            cand = self.neighbor(state, rng)
            o2 = self.overflow(cand)
            if o2 <= ov + E:
                state, ov = cand, o2
                if ov <= E: out = self.score(state)
        return state, out

    def neighbor(self, state, rng):
        s = [seq[:] for seq in state]
        u = rng.random()
        mv = self.movers
        if u < 0.4 and len(mv) > 1:
            i, j = rng.choice(mv, 2, replace=False)
            (ri, ki), (rj, kj) = self._find(s, i), self._find(s, j)
            s[ri][ki], s[rj][kj] = j, i
        elif u < 0.8:
            i = mv[rng.integers(len(mv))]; ri, ki = self._find(s, i); s[ri].pop(ki)
            rj = int(rng.integers(len(s))); s[rj].insert(int(rng.integers(len(s[rj]) + 1)), i)
        else:
            nz = [ri for ri, seq in enumerate(s) if len(seq) > 1]
            if nz:
                ri = nz[rng.integers(len(nz))]; k = int(rng.integers(len(s[ri]) - 1))
                s[ri][k], s[ri][k + 1] = s[ri][k + 1], s[ri][k]
        return self.normalize(s)

    @staticmethod
    def _find(s, i):
        for ri, seq in enumerate(s):
            if i in seq: return ri, seq.index(i)
        raise KeyError(i)


def solve(inst, seed=0, time_limit=60.0, log=None):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    M = Model(inst)
    state, out = M.initial(rng)
    if out is None: return None
    cur_state, cur = state, out[0]
    best = None                                  # best feasible under the search cost: (cost, state, rects)
    true_best = None
    def true_eval(rects):
        nonlocal true_best
        ev = evaluate(inst, rects)
        if ev["feasible"] and (true_best is None or ev["path_total"] < true_best[0] - 1e-9):
            true_best = (ev["path_total"], rects, ev)
        return ev
    pending = False
    if out[2]:
        best = (out[0], state, out[1]); true_eval(out[1])
    ds = []
    for _ in range(30):
        o = M.score(M.neighbor(cur_state, rng))
        if o is not None and o[0] > cur: ds.append(min(o[0] - cur, 0.2 * cur))
    T0 = (np.median(ds) if ds else 0.01 * cur) / -math.log(0.3); Tend = T0 * 1e-3
    it = feas = acc = 0; last_true = time.time()
    t_start = time.time(); budget = max(1.0, time_limit - (t_start - t0) - 1.0)
    while True:
        el = time.time() - t_start
        if el >= budget: break
        T = T0 * (Tend / T0) ** (el / budget)
        it += 1
        cand = M.neighbor(cur_state, rng)
        o = M.score(cand)
        if o is None: continue
        feas += 1
        dlt = o[0] - cur
        if dlt <= 0 or rng.random() < math.exp(-dlt / T):
            cur_state, cur = cand, o[0]; acc += 1
            if o[2] and (best is None or cur < best[0] - 1e-9):
                best = (cur, cand, o[1]); pending = True
        if pending and time.time() - last_true > 0.5:
            true_eval(best[2]); last_true = time.time(); pending = False
    if pending: true_eval(best[2])
    if true_best is None: return None
    if log: log(f"it={it} socp_ok={feas} acc={acc} proxy={best[0]:.1f} true={true_best[0]:.1f}")
    return dict(rects=true_best[1], eval=true_best[2], iterations=it, socp_feasible=feas, accepted=acc,
                proxy_best=best[0], rows=[[int(i) for i in seq] for seq in best[1]])
