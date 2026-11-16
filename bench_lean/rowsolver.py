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

Option pairs=True (B1 in 改进规划.md): the two rows of a region are coupled pair by pair instead of by one split.
  Each region keeps two interleavings (a sequence pair restricted to two rows) of its bottom and top departments;
  bottom i / top j are "i left of j", "j left of i" or stacked (d_i + d_j <= depth), so a deep department only
  limits the departments facing it. X pairs inside the region are handled exactly (horizontal or vertical gap).
  The all-stacked interleaving is the single-split model, so this only adds layouts.
Option elite=K: keep the K best distinct layouts under the search cost and re-score them all with evaluate.
Option polish=f (C1): spend the last fraction f of the time on a descent scored by the true path cost.

Elastic SOCP (phase-1 idea): every non-empty row has a slack sigma >= 0 that may relax its end, depth and
  dock-door constraints at price sig_pen per metre. The SOCP therefore always solves and sum(sigma) grades how
  far a state is from fitting; the SA minimises cost + penalties and only states with sigma = 0 and no
  shortfall count as feasible. This replaces the old repair loop, which could wander for a minute on tight
  instances (LA-L-U-BF-s21: departments fill 79 % of the usable row area).

SA moves: swap two departments; move one department to another row/position; swap neighbours in a row;
  with pairs=True also swap two neighbouring entries of an interleaving.
  W/E dock departments stay at their row end, S/N dock departments stay in their row.
Search uses the SOCP objective; new best layouts are re-scored with bench_lean.evaluate (true path cost,
all checks) and the best feasible one under the true cost is returned.
"""
import time, math
import numpy as np
import scipy.sparse as sp
import clarabel
from .evaluate import evaluate, weights, _gap
from .fastpath import path_cost
from .reference import regions as base_regions, subtract, default_order

E = 1e-6
CORRIDOR = 2.0


class St(list):
    """Rows (list of department lists) plus, for pairs=True, the interleavings of two-row regions."""
    inter = None

    def copy(self):
        s = St(seq[:] for seq in self)
        s.inter = {g: (a[:], b[:]) for g, (a, b) in (self.inter or {}).items()}
        return s


class Model:
    def __init__(self, inst, pairs=False, near=0.0):
        self.near = float(near)              # D3: probability that a move puts a department next to a flow partner
        self.pairmode = bool(pairs)          # option B1 (not the list of flow pairs below)
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
        self.grows = {}                 # group -> (bottom-side row, top-side row)
        for ri, r in enumerate(self.rows):
            if r["group"] is not None and r["side"] == "B": self.grows[r["group"]] = (ri, self.mate[ri])
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
        self.penalty = 20.0 * sum(w for _, _, w in self.pairs)     # cost of one metre of shortfall
        self.step = 0.02                                             # plus 2 % of the objective while any shortfall remains
        self.sig_pen = 20.0 * sum(w for _, _, w in self.pairs)     # cost of one metre of row slack

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
        if self.pairmode: self.sync(state)
        return state

    def sync(self, state):
        """Rebuild each interleaving so that it holds exactly the region's departments in row order.
        The B/T pattern of the old interleaving is kept; a new department enters stacked with the other row."""
        if state.inter is None: state.inter = {}
        for g, (rb, rt) in self.grows.items():
            Bs, Ts = state[rb], state[rt]; sb, st = set(Bs), set(Ts)
            old = state.inter.get(g, ([], []))
            new = []
            for which in (0, 1):
                seq = [i for i in old[which] if i in sb or i in st]
                have = set(seq)
                for i in Bs + Ts:
                    if i in have: continue
                    # plus: top first, bottom last; minus: bottom first, top last  -> stacked with the other row
                    if (i in sb) == (which == 0): seq.append(i)
                    else: seq.insert(0, i)
                lab = ["B" if i in sb else "T" for i in seq]
                ib, it = iter(Bs), iter(Ts)
                new.append([next(ib) if l == "B" else next(it) for l in lab])
            state.inter[g] = (new[0], new[1])

    def overflow(self, state):
        """Necessary-condition check of the rows: excess width (m) + 10 x missing depth (m); 0 = may be feasible."""
        R = self.rows
        need = [max((self.mindepth[i] for i in seq), default=0.0) for seq in state]
        ov = 0.0
        for ri, seq in enumerate(state):
            if not seq: continue
            r = R[ri]; cap = r["rh"]
            if r["group"] is not None and not self.pairmode:
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
            if len(rs) == 2 and not self.pairmode: gvar[g] = nv; nv += 1
        sig = {}
        for ri, seq in enumerate(state):
            if seq: sig[ri] = nv; nv += 1
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
        for ri in sig: le([(sig[ri], -1)], 0.0)
        for ri, seq in enumerate(state):
            r = R[ri]; m = len(seq); sg = sig.get(ri)
            for k, i in enumerate(seq):
                x, w, d = vi[i], vi[i] + 1, vi[i] + 2
                ms, a = self.ms[i], self.am[i]
                le([(w, -1)], -ms); le([(d, -1)], -ms)
                le([(w, 1), (d, -a)], 0.0); le([(d, 1), (w, -a)], 0.0)
                g = r["group"]
                if g in gvar:
                    if r["side"] == "B": le([(d, 1), (gvar[g], -1), (sg, -1)], 0.0)
                    else: le([(d, 1), (gvar[g], 1), (sg, -1)], r["rh"])
                else:
                    le([(d, 1), (sg, -1)], r["rh"])
                if k == 0: le([(x, -1)], -r["x0"])
                else:
                    p = seq[k - 1]; gap = self.xpairs.get((p, i), 0.0)
                    le([(vi[p], 1), (vi[p] + 1, 1), (x, -1)], -gap)
                if k == m - 1: le([(x, 1), (w, 1), (sg, -1)], r["x1"])
                for kk in range(k + 2, m):                 # X pairs further apart in the same row
                    j = seq[kk]; gap = self.xpairs.get((i, j))
                    if gap: le([(x, 1), (w, 1), (vi[j], -1)], -gap)
                if i in self.dock:
                    s = self.dock[i]
                    if s["mode"] in "SN":
                        le([(d, -1)], -r["rh"]); le([(x, 1), (sg, -1)], s["lo"]); le([(x, -1), (w, -1), (sg, -1)], -s["hi"])
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
        if self.pairmode:
            for g, (rb, rt) in self.grows.items():
                if not state[rb] or not state[rt]: continue
                plus, minus = state.inter[g]
                pp = {i: k for k, i in enumerate(plus)}; pm = {i: k for k, i in enumerate(minus)}
                rh = R[rb]["rh"]
                for i in state[rb]:
                    for j in state[rt]:
                        gap = self.xpairs.get((i, j), 0.0)
                        a, b = pp[i] < pp[j], pm[i] < pm[j]
                        if a and b: le([(vi[i], 1), (vi[i] + 1, 1), (vi[j], -1)], -gap)
                        elif not a and not b: le([(vi[j], 1), (vi[j] + 1, 1), (vi[i], -1)], -gap)
                        else: le([(vi[i] + 2, 1), (vi[j] + 2, 1), (sig[rb], -1)], rh - gap)
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
        # pairs on the same aisle -> one t >= |cx_i - cx_j| each; pairs on different aisles -> u >= |cx - c| per
        # (department, connector), shared. Rows are built as arrays (same rows as one le() per row).
        P_i, P_j, P_w, P_c = [], [], [], []
        uw = {}
        for i, j, wt in self.pairs:
            vi_i, ci, ai, yi, ei = info(i); vj, cj, aj, yj, ej = info(j)
            const += wt * (abs(yi - yj) + ei + ej)
            if vi_i is None and vj is None:
                if ai == aj or not self.conn: const += wt * abs(ci - cj)
                else: const += wt * min(abs(ci - c) + abs(cj - c) for c in self.conn)
                continue
            if ai == aj or ai is None or aj is None or not self.conn:
                P_i.append(-1 if vi_i is None else vi_i); P_j.append(-1 if vj is None else vj)
                P_w.append(wt); P_c.append(ci - cj)
            else:
                ri_c = (ci if vi_i is None else (R[pos[i][0]]["x0"] + R[pos[i][0]]["x1"]) / 2)
                rj_c = (cj if vj is None else (R[pos[j][0]]["x0"] + R[pos[j][0]]["x1"]) / 2)
                c = min(self.conn, key=lambda cc: abs(ri_c - cc) + abs(rj_c - cc))
                for vv, cc0 in ((vi_i, ci), (vj, cj)):
                    if vv is None: const += wt * abs(cc0 - c); continue
                    uw[(vv, c)] = uw.get((vv, c), 0.0) + wt
        blk_r, blk_c, blk_v, blk_b = [], [], [], []
        m = len(P_w)
        if m:
            PI, PJ, PC = np.array(P_i), np.array(P_j), np.array(P_c, float)
            t0 = nv; nv += m
            k = np.arange(m); ra, rb_ = cnt[0] + 2 * k, cnt[0] + 2 * k + 1
            rr = [ra, rb_]; cc = [t0 + k, t0 + k]; vv_ = [-np.ones(m), -np.ones(m)]
            for idx, sgn in ((PI, 1.0), (PJ, -1.0)):
                mk = idx >= 0
                for off, coef in ((0, 1.0), (1, 0.5)):
                    rr += [ra[mk], rb_[mk]]; cc += [idx[mk] + off, idx[mk] + off]
                    vv_ += [np.full(mk.sum(), sgn * coef), np.full(mk.sum(), -sgn * coef)]
            blk_r += rr; blk_c += cc; blk_v += vv_
            bb = np.empty(2 * m); bb[0::2] = -PC; bb[1::2] = PC; blk_b.append(bb)
            cnt[0] += 2 * m
        mu = len(uw)
        if mu:
            keys = list(uw); UV = np.array([kk[0] for kk in keys]); UC = np.array([kk[1] for kk in keys], float)
            u0 = nv; nv += mu
            k = np.arange(mu); ra, rb_ = cnt[0] + 2 * k, cnt[0] + 2 * k + 1
            blk_r += [ra, ra, ra, rb_, rb_, rb_]; blk_c += [UV, UV + 1, u0 + k, UV, UV + 1, u0 + k]
            blk_v += [np.ones(mu), np.full(mu, 0.5), -np.ones(mu), -np.ones(mu), np.full(mu, -0.5), -np.ones(mu)]
            bb = np.empty(2 * mu); bb[0::2] = UC; bb[1::2] = -UC; blk_b.append(bb)
            cnt[0] += 2 * mu
        q = np.zeros(nv)
        if m: q[t0:t0 + m] = P_w
        if mu: q[u0:u0 + mu] = [uw[kk] for kk in keys]
        for ri, c in sig.items(): q[c] += self.sig_pen
        # SOC rows: s = b - A z, s = (w + d, w - d, 2 sqrt(A))
        soc_r, soc_c, soc_v, soc_b = [], [], [], []
        for t, i in enumerate(self.placed):
            w, d = vi[i] + 1, vi[i] + 2; b0 = 3 * t
            soc_r += [b0, b0, b0 + 1, b0 + 1]; soc_c += [w, d, w, d]; soc_v += [-1, -1, -1, 1]
            soc_b += [0.0, 0.0, 2 * math.sqrt(self.A[i])]
        ne, ni, ns = cnt[1], cnt[0], 3 * len(self.placed)
        rows = np.concatenate([np.array(eq_r, int), ne + np.array(ineq_r, int)] + [ne + np.asarray(x, int) for x in blk_r]
                              + [ne + ni + np.array(soc_r, int)])
        cols = np.concatenate([np.array(eq_c, int), np.array(ineq_c, int)] + [np.asarray(x, int) for x in blk_c]
                              + [np.array(soc_c, int)])
        vals = np.concatenate([np.array(eq_v, float), np.array(ineq_v, float)] + [np.asarray(x, float) for x in blk_v]
                              + [np.array(soc_v, float)])
        Amat = sp.csc_matrix((vals, (rows, cols)), shape=(ne + ni + ns, nv))
        b = np.concatenate([np.array(eq_b, float), np.array(ineq_b, float)] + blk_b + [np.array(soc_b, float)])
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
        sg_tot = float(sum(max(0.0, z[c]) for c in sig.values()))
        obj = float(sol.obj_val) + const
        return obj, rects, sg_tot, obj - self.sig_pen * sg_tot

    def shortfall(self, rects):
        """X gaps between rows and fire egress, checked after the SOCP: total shortfall in metres (0 = ok)."""
        v = 0.0
        for (i, j), gap in self.xpairs.items():
            if i < j: v += max(0.0, gap - _gap(rects[i], rects[j]))
        Rr = np.asarray(rects, float)
        cs = np.stack([Rr[:, :2], Rr[:, :2] + Rr[:, 2:4] * [1, 0], Rr[:, :2] + Rr[:, 2:4] * [0, 1], Rr[:, :2] + Rr[:, 2:4]], 1)
        dmin = np.sqrt(((cs[:, :, None, :] - self.exits[None, None]) ** 2).sum(-1)).min(2)    # corner -> nearest exit
        v += float(np.maximum(dmin.max(1) - self.egress, 0.0).sum())
        return v

    def score(self, state, prefilter=False):
        """(search cost, rects, ok, shortfall, row slack, objective without penalties), or None when the paint
        department sits where it cannot touch an outer wall (or, with prefilter, when the rows cannot hold the
        departments). search cost = SOCP objective (incl. slack price) + penalty x shortfall;
        ok = no row slack and no shortfall."""
        if prefilter and not self.capacity_ok(state): return None
        out = self.socp(state)
        if out is None: return None
        v = self.shortfall(out[1])
        short = v > 1e-6
        cost = out[0] + self.penalty * v + (self.step * out[3] if short else 0.0)
        return cost, out[1], not short and out[2] <= 1e-6, v, out[2], out[3]

    # ---------- initial solution ----------
    def initial(self, rng, deadline=None, tries=20000):
        R = self.rows; state = St([] for _ in R); state.inter = {}
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
        out = self.score(state)                    # elastic: only None if paint cannot reach a wall
        k = 0
        while out is None and k < tries and (deadline is None or time.time() < deadline):
            k += 1
            state = self.neighbor(state, rng); out = self.score(state)
        return state, out

    def from_rects(self, rects):
        """Rows state that follows a given layout (e.g. a classic solution that ignores aisles): each department
        goes to the row whose band contains its centre (nearest row otherwise), ordered by centre x."""
        R = self.rows; state = St([] for _ in R); state.inter = {}
        for i, s_ in self.dock.items(): state[s_["row"]].append(i)
        def band(r):                     # the half of the region a row may use
            if r["group"] is None: return r["ry"], r["ry"] + r["rh"]
            mid = r["ry"] + r["rh"] / 2
            return (r["ry"], mid) if r["side"] == "B" else (mid, r["ry"] + r["rh"])
        place = []
        for i in self.movers:
            x, y, w, h = rects[i]; cx, cy = x + w / 2, y + h / 2
            def dist(ri):
                r = R[ri]; y0, y1 = band(r)
                dx = max(0.0, r["x0"] - cx, cx - r["x1"]); dy = max(0.0, y0 - cy, cy - y1)
                return dx + dy
            ri = min(range(len(R)), key=dist)
            if i in self.paint and not (R[ri]["wallW"] or R[ri]["wallE"] or R[ri]["far_wall"]):
                ri = min((q for q in range(len(R)) if R[q]["wallW"] or R[q]["wallE"] or R[q]["far_wall"]), key=dist)
            place.append((ri, cx, i))
        for ri, cx, i in sorted(place):
            state[ri].append(i)
        for ri, seq in enumerate(state):            # order by centre x, docks stay where normalize puts them
            cxs = {i: (rects[i][0] + rects[i][2] / 2 if i in self.movers else (R[ri]["x0"] + R[ri]["x1"]) / 2) for i in seq}
            if i in self.dock and self.dock[i]["mode"] in "SN":
                pass
            seq.sort(key=lambda i: cxs[i] if i not in self.dock else
                     (self.dock[i]["lo"] + self.dock[i]["hi"]) / 2 if self.dock[i]["mode"] in "SN" else
                     (-1e9 if self.dock[i]["mode"] == "W" else 1e9))
        for i in self.paint:                        # paint must sit where it can touch an outer wall
            ri = next(q for q, seq in enumerate(state) if i in seq); r = R[ri]; seq = state[ri]
            if r["far_wall"]: continue
            seq.remove(i)
            if r["wallW"]: seq.insert(1 if seq and seq[0] in self.dock and self.dock[seq[0]]["mode"] == "W" else 0, i)
            else: seq.insert(len(seq) - (1 if seq and seq[-1] in self.dock and self.dock[seq[-1]]["mode"] == "E" else 0), i)
        return self.normalize(state)

    def neighbor(self, state, rng):
        s = state.copy()
        if self.pairmode and rng.random() < 0.25:
            gs = [g for g, (rb, rt) in self.grows.items() if s[rb] and s[rt]]
            if gs:
                g = gs[rng.integers(len(gs))]; seq = s.inter[g][int(rng.integers(2))]
                rb = self.grows[g][0]; sb = set(s[rb])
                ks = [k for k in range(len(seq) - 1) if (seq[k] in sb) != (seq[k + 1] in sb)]
                if ks:
                    k = ks[rng.integers(len(ks))]; seq[k], seq[k + 1] = seq[k + 1], seq[k]
                    return s
        u = rng.random()
        mv = self.movers
        if u < 0.4 and len(mv) > 1:
            i, j = rng.choice(mv, 2, replace=False)
            (ri, ki), (rj, kj) = self._find(s, i), self._find(s, j)
            s[ri][ki], s[rj][kj] = j, i
        elif u < 0.8:
            i = mv[rng.integers(len(mv))]; ri, ki = self._find(s, i); s[ri].pop(ki)
            wi = self.Wt[i].copy(); wi[i] = 0.0
            if self.near > 0 and wi.sum() > 0 and rng.random() < self.near:
                j = int(rng.choice(self.n, p=wi / wi.sum()))            # a flow partner, by weight
                if j in self.fixed:
                    rj = int(rng.integers(len(s))); s[rj].insert(int(rng.integers(len(s[rj]) + 1)), i)
                else:
                    rj, kj = self._find(s, j); s[rj].insert(kj + int(rng.integers(2)), i)
            else:
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


def solve(inst, seed=0, time_limit=60.0, log=None, trace=None, pairs=False, elite=0, polish=0.0, accept="proxy",
          check_every=0.5, init_rects=None, stop_feasible=False, near=0.0):
    """trace: optional callable(proxy_cost, rects, ok, elapsed) called on every accepted move (diagnostics).
    pairs: pairwise coupling of two-row regions (B1).
    elite: keep the K best distinct layouts under the search cost and re-score them by true path cost at the end.
    polish: fraction of the time for a final descent that accepts only true-cost improvements (C1).
    accept: "proxy" = SA accepts on the SOCP objective; "true" = SA accepts on the true path cost
            (+ the same shortfall penalty); the SOCP still sets the geometry of every candidate.
    check_every: seconds between true-cost checks of a new best layout (accept="proxy").
    init_rects: start from this layout (Model.from_rects) instead of the greedy start.
    stop_feasible: return the first feasible layout found (used to legalise a classic solution).
    near: probability that a relocation puts the department next to a flow partner (chosen by weight).
    True path costs come from fastpath.path_cost (same numbers as evaluate, about 5x faster); the returned layout
    is the cheapest candidate that evaluate confirms as feasible."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    M = Model(inst, pairs=bool(pairs), near=near)
    if init_rects is not None:
        state = M.from_rects(init_rects); out = M.score(state)
        if out is None: state, out = M.initial(rng, deadline=t0 + 0.3 * time_limit)
    else:
        state, out = M.initial(rng, deadline=t0 + 0.3 * time_limit)
    if out is None: return None
    best = None                                  # best feasible under the search cost: (cost, state, rects)
    cands = []                                   # (true path cost, rects, state), cheapest first, at most 8
    pool = {}                                    # elite: rounded search cost -> (cost, state, rects)
    n_true = 0

    def true_cost(rects, st=None, record=True):
        nonlocal n_true
        pc = path_cost(inst, rects)[0]; n_true += 1
        if record and np.isfinite(pc) and (len(cands) < 8 or pc < cands[-1][0] - 1e-9):
            if all(abs(pc - c[0]) > 1e-6 for c in cands):
                cands.append((pc, rects, st)); cands.sort(key=lambda v: v[0]); del cands[8:]
        return pc

    def remember(cost, st, rects):
        if elite <= 0: return
        key = round(cost, 3)
        if key in pool: return
        if len(pool) < elite: pool[key] = (cost, st, rects); return
        worst = max(pool)
        if key < worst: del pool[worst]; pool[key] = (cost, st, rects)
    use_true = accept == "true"

    def scost(o, st):
        # accept=true prices feasible states by true path cost; states with row slack or shortfall have
        # overlapping or illegal rectangles, so they keep the proxy price (where the penalties dominate)
        if not use_true or not o[2]: return o[0]
        return true_cost(o[1], st)
    cur_state, cur = state, scost(out, state)
    cur_ok, cur_raw = out[2], out[5]
    pending = False
    if out[2]:
        best = (cur, state, out[1]); remember(cur, state, out[1])
        if not use_true: true_cost(out[1], state)
    ds = []
    for _ in range(30):
        nb = M.neighbor(cur_state, rng); o = M.score(nb)
        if o is None: continue
        if o[5] > cur_raw: ds.append(min(o[5] - cur_raw, 0.2 * cur_raw))
    T0 = (np.median(ds) if ds else 0.01 * cur_raw) / -math.log(0.3); Tend = T0 * 1e-3
    it = feas = acc = 0; last_true = time.time()
    t_start = time.time(); total = max(1.0, time_limit - (t_start - t0) - 1.0)
    budget = total * (1.0 - polish)
    while True:
        el = time.time() - t_start
        if el >= budget: break
        T = T0 * (Tend / T0) ** (el / budget)
        it += 1
        cand = M.neighbor(cur_state, rng)
        o = M.score(cand, prefilter=cur_ok)
        if o is None: continue
        feas += 1
        c = scost(o, cand)
        dlt = c - cur
        if dlt <= 0 or rng.random() < math.exp(-dlt / T):
            cur_state, cur, cur_ok = cand, c, o[2]; acc += 1
            if trace: trace(o[0], o[1], o[2], time.time() - t0)
            if o[2]:
                remember(cur, cand, o[1])
                if best is None or cur < best[0] - 1e-9:
                    best = (cur, cand, o[1]); pending = True
                if stop_feasible: break
        if pending and not use_true and time.time() - last_true > check_every:
            true_cost(best[2], best[1]); last_true = time.time(); pending = False
    if stop_feasible and out[2] and not cands: true_cost(out[1], state)
    if pending and not use_true: true_cost(best[2], best[1])
    for c, st, rects in sorted(pool.values(), key=lambda v: v[0]):
        if time.time() - t_start > total: break
        true_cost(rects, st)
    pol_it = pol_acc = 0
    if polish > 0 and cands:
        st, tb = cands[0][2], cands[0][0]
        while time.time() - t_start < total:
            pol_it += 1
            cand = M.neighbor(st, rng)
            o = M.score(cand, prefilter=True)
            if o is None or not o[2]: continue
            pc = true_cost(o[1], cand)
            if pc < tb - 1e-9: st, tb = cand, pc; pol_acc += 1
    chosen = None
    for pc, rects, st in cands:                  # official check, cheapest first
        ev = evaluate(inst, rects)
        if ev["feasible"]: chosen = (pc, rects, ev, st); break
    if chosen is None: return None
    if log: log(f"it={it} socp_ok={feas} acc={acc} search_best={best[0] if best else float('nan'):.1f} "
                f"true={chosen[0]:.1f} true_evals={n_true} polish={pol_acc}/{pol_it}")
    st = chosen[3]
    return dict(rects=chosen[1], eval=chosen[2], iterations=it, socp_feasible=feas, accepted=acc,
                search_best=best[0] if best else None, true_evals=n_true, polish_iterations=pol_it,
                polish_accepted=pol_acc, rows=[[int(i) for i in seq] for seq in st],
                inter={str(g): [[int(i) for i in a], [int(i) for i in b]] for g, (a, b) in (st.inter or {}).items()} if pairs else None)
