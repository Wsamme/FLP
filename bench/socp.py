"""Stage 2: exact continuous geometry for fixed pairwise relations (convex SOCP, Clarabel).

Given, for every department pair, one relation "i left of j" or "i below j", solve

  min  sum_{i<j, w_ij>0} w_ij (dx_ij + dy_ij)
  s.t. dx_ij >= +-(x_i - x_j),  dy_ij >= +-(y_i - y_j)                  (rectilinear distance)
       x_i + w_i/2 <= x_j - w_j/2            for i left of j             (no overlap)
       y_i + h_i/2 <= y_j - h_j/2            for i below j
       w_i/2 <= x_i <= W - w_i/2,  h_i/2 <= y_i <= H - h_i/2             (inside the floor)
       h_i <= a' w_i,  w_i <= a' h_i,  a' = alpha (1 - margin)           (aspect)
       w_i h_i >= A_i',  i.e. ||(w_i - h_i, 2 sqrt(A_i'))||_2 <= w_i + h_i (rotated cone, exact area)

with (x, y) the department centre.  Relations implied by transitivity (i left of k left of j) are
dropped: they are redundant because widths are positive.  The problem is built directly as sparse
matrices for clarabel.DefaultSolver (no modelling layer).

Strictness.  Floors with free area (O7, SC30, Du62): areas are solved with margin A(1+2e-7), aspect with
alpha(1-1e-7); afterwards each rectangle is shrunk about its centre by 5e-8 (relative) to remove solver
round-off, and the result must satisfy every constraint EXACTLY (no tolerance: w h >= A, aspect <= alpha,
inside the floor, no positive overlap).  Floors that are 100 % full (vC10a, AB20) admit no margin: there
the solution must satisfy every constraint to 1e-7 (relative area / absolute length), 10x tighter than the
checker flp.evaluate (1e-6).
"""
import time
import numpy as np
import scipy.sparse as sp
import clarabel


def sp_relations(gp, gm, reduce=True):
    """Sequence pair (gp, gm), both permutations of 0..n-1 (department ids in order).
    i left of j  <=> i before j in gp and in gm;  i below j <=> i after j in gp, before j in gm.
    Returns boolean matrices L (L[i,j]: i left of j) and B (B[i,j]: i below j), transitively reduced."""
    n = len(gp)
    pp = np.empty(n, int); pm = np.empty(n, int)
    pp[gp] = np.arange(n); pm[gm] = np.arange(n)
    bp = pp[:, None] < pp[None, :]; bm = pm[:, None] < pm[None, :]
    L = bp & bm
    B = (~bp) & bm
    np.fill_diagonal(B, False)
    if reduce:
        L, B = reduce_relations(L, B)
    return L, B


def reduce_relations(L, B):
    """Transitive reduction: i left of j is implied when i left of k and k left of j (widths are positive)."""
    Li = L.astype(np.int32); Bi = B.astype(np.int32)
    return L & ~((Li @ Li) > 0), B & ~((Bi @ Bi) > 0)


def rects_relations(R, tol=1e-6):
    """Geometric relations of a feasible layout: for each pair the clearest separation (largest gap)."""
    R = np.asarray(R, float); n = len(R)
    x0, y0, x1, y1 = R[:, 0], R[:, 1], R[:, 0] + R[:, 2], R[:, 1] + R[:, 3]
    gx = x0[None, :] - x1[:, None]      # gx[i,j] >= 0: i left of j
    gy = y0[None, :] - y1[:, None]
    L = np.zeros((n, n), bool); B = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            g = [gx[i, j], gx[j, i], gy[i, j], gy[j, i]]
            k = int(np.argmax(g))
            if g[k] < -tol:
                raise ValueError(f"departments {i} and {j} overlap")
            if k == 0: L[i, j] = True
            elif k == 1: L[j, i] = True
            elif k == 2: B[i, j] = True
            else: B[j, i] = True
    return L, B


def rects_to_sp(R, tol=1e-6):
    """A sequence pair consistent with a non-overlapping layout (cf. Murata et al. 1996).
    gp is a topological order of 'i left of j OR i above j', gm of 'i left of j OR i below j', using all
    geometric relations (with tolerance tol).  For any pair the SP then implies left/right or above/below,
    and that relation holds in the layout, so Stage 2 on this SP is feasible and can only improve it."""
    R = np.asarray(R, float); n = len(R)
    x0, y0, x1, y1 = R[:, 0], R[:, 1], R[:, 0] + R[:, 2], R[:, 1] + R[:, 3]
    Lg = x1[:, None] <= x0[None, :] + tol
    Bg = y1[:, None] <= y0[None, :] + tol
    np.fill_diagonal(Lg, False); np.fill_diagonal(Bg, False)
    sep = Lg | Lg.T | Bg | Bg.T; np.fill_diagonal(sep, True)
    if not sep.all():
        raise ValueError("overlapping layout")
    cx = x0 + R[:, 2] / 2; cy = y0 + R[:, 3] / 2
    out = []
    for M, key in ((Lg | Bg.T, cx - cy), (Lg | Bg, cx + cy)):
        M = M & ~M.T            # diagonal pairs: this sequence is free, the other one decides
        indeg = M.sum(0).astype(int); order = []
        avail = [i for i in range(n) if indeg[i] == 0]
        while avail:
            i = min(avail, key=lambda k: key[k]); avail.remove(i); order.append(i)
            for j in np.nonzero(M[i])[0]:
                indeg[j] -= 1
                if indeg[j] == 0: avail.append(int(j))
        if len(order) < n:
            return None
        out.append(np.array(order))
    gp, gm = out
    # sanity: every implied relation holds geometrically
    L, B = sp_relations(gp, gm, reduce=False)
    assert not (L & ~Lg).any() and not (B & ~Bg).any()
    return gp, gm


class Stage2:
    def __init__(self, W, H, areas, alpha, Wt, area_margin=None, aspect_margin=1e-7, tol=1e-9,
                 strict_tol=1e-7, inst=None):
        self.W, self.H = float(W), float(H)
        self.A = np.asarray(areas, float); self.n = n = len(self.A)
        self.alpha = np.asarray(alpha, float)
        self.Wt = np.asarray(Wt, float)
        full = self.A.sum() >= self.W * self.H * (1 - 1e-6)
        if area_margin is None:
            area_margin = 0.0 if full else 2e-7     # a full floor leaves no room for any margin
        self.full = full
        # with free floor area, every rectangle is shrunk about its centre by this relative amount after the
        # solve: removes solver round-off overlaps (~1e-9) while keeping area >= A (1 + ~1e-7)
        self.shrink = 0.0 if full else 5e-8
        if not full: strict_tol = 0.0             # then every constraint must hold with no tolerance at all
        self.Ap = self.A * (1 + area_margin)
        self.alp = self.alpha * (1 - aspect_margin)
        self.strict_tol = strict_tol; self.inst = inst
        iu = np.triu_indices(n, 1)
        m = self.Wt[iu] > 0
        self.pi, self.pj = iu[0][m], iu[1][m]; self.pw = self.Wt[iu][m]
        self.P = len(self.pw)
        st = clarabel.DefaultSettings()
        st.verbose = False
        st.tol_gap_abs = tol; st.tol_gap_rel = tol; st.tol_feas = tol
        st.tol_infeas_abs = 1e-10; st.tol_infeas_rel = 1e-10; st.tol_ktratio = 1e-8
        st.max_iter = 200
        self.settings = st
        self._build_static()
        self.nsolve = 0; self.tsolve = 0.0; self.last_iter = 0

    # variables: x 0..n-1, y n..2n-1 (centres), w 2n.., h 3n.., then one auxiliary d per distance term whose
    # sign is not known from the relations (|x_i - x_j| <= d)
    def _build_static(self):
        n = self.n
        X, Y, Wv, Hv = 0, n, 2 * n, 3 * n
        r, c, v, b = [], [], [], []
        row = 0
        i = np.arange(n)
        # floor: -x + w/2 <= 0 ; x + w/2 <= W ; same for y   (Clarabel form: s = b - A z >= 0)
        for (O, S, lim) in ((X, Wv, self.W), (Y, Hv, self.H)):
            rows = row + i; r += [rows, rows]; c += [O + i, S + i]; v += [np.full(n, -1.0), np.full(n, 0.5)]
            b.append(np.zeros(n)); row += n
            rows = row + i; r += [rows, rows]; c += [O + i, S + i]; v += [np.full(n, 1.0), np.full(n, 0.5)]
            b.append(np.full(n, lim)); row += n
        # aspect: w - a' h <= 0 ; h - a' w <= 0  (only where alpha > 0)
        ia = i[self.alpha > 0]
        for (U, V) in ((Wv, Hv), (Hv, Wv)):
            rows = row + np.arange(len(ia)); r += [rows, rows]; c += [U + ia, V + ia]
            v += [np.ones(len(ia)), -self.alp[ia]]; b.append(np.zeros(len(ia))); row += len(ia)
        self.static = (np.concatenate(r), np.concatenate(c), np.concatenate(v), np.concatenate(b), row)
        # SOC rows: s = (w+h, w-h, 2 sqrt(A)) ; s = b - A z  -> A rows: -(w+h), -(w-h), 0
        rr, cc, vv = [], [], []
        for t in range(n):
            base = 3 * t
            rr += [base, base, base + 1, base + 1]; cc += [Wv + t, Hv + t, Wv + t, Hv + t]; vv += [-1, -1, -1, 1]
        self.soc = (np.array(rr), np.array(cc), np.array(vv, float),
                    np.column_stack([np.zeros(n), np.zeros(n), 2 * np.sqrt(self.Ap)]).ravel())

    def solve(self, L, B, Lfull=None, Bfull=None):
        """L, B: boolean n x n relation matrices (L[i,j]: i left of j; B[i,j]: i below j), or lists of pairs.
        Lfull/Bfull (optional, e.g. the unreduced SP relations): every relation known to hold; for a flow pair
        with a known order the distance term is linear (x_j - x_i), which removes its auxiliary variable.
        Returns dict(status, cost, rects)."""
        n = self.n
        if not isinstance(L, np.ndarray):
            Lm = np.zeros((n, n), bool); Bm = np.zeros((n, n), bool)
            for a, b_ in L: Lm[a, b_] = True
            for a, b_ in B: Bm[a, b_] = True
            L, B = Lm, Bm
        if Lfull is None: Lfull, Bfull = L, B
        X, Y, Wv, Hv = 0, n, 2 * n, 3 * n
        pi, pj, pw = self.pi, self.pj, self.pw
        q = np.zeros(4 * n)
        # sign of each term: +1 if coordinate of i < coordinate of j is implied, -1 if >, 0 unknown
        sx = Lfull[pi, pj].astype(int) - Lfull[pj, pi].astype(int)
        sy = Bfull[pi, pj].astype(int) - Bfull[pj, pi].astype(int)
        for (O, sg) in ((X, sx), (Y, sy)):
            np.add.at(q, O + pj, pw * sg); np.add.at(q, O + pi, -pw * sg)
        ux = np.nonzero(sx == 0)[0]; uy = np.nonzero(sy == 0)[0]
        K = len(ux) + len(uy); nv = 4 * n + K
        r0, c0, v0, b0, m0 = self.static
        rr, cc, vv = [r0], [c0], [v0]
        row = m0
        # aux rows: -d + (c_i - c_j) <= 0 and -d - (c_i - c_j) <= 0
        dcol = 4 * n
        for (O, u) in ((X, ux), (Y, uy)):
            k = len(u); dk = dcol + np.arange(k)
            for sgn in (1.0, -1.0):
                rows = row + np.arange(k)
                rr += [rows, rows, rows]; cc += [dk, O + pi[u], O + pj[u]]
                vv += [np.full(k, -1.0), np.full(k, sgn), np.full(k, -sgn)]; row += k
            dcol += k
        q = np.concatenate([q, pw[ux], pw[uy]])
        li, lj = np.nonzero(L); bi, bj = np.nonzero(B)
        nl, nb = len(li), len(bi)
        rl = row + np.arange(nl); rb = row + nl + np.arange(nb)
        # x_i + w_i/2 - x_j + w_j/2 <= 0 ; same in y
        rr += [rl, rl, rl, rl, rb, rb, rb, rb]
        cc += [X + li, X + lj, Wv + li, Wv + lj, Y + bi, Y + bj, Hv + bi, Hv + bj]
        vv += [np.ones(nl), -np.ones(nl), np.full(nl, .5), np.full(nl, .5),
               np.ones(nb), -np.ones(nb), np.full(nb, .5), np.full(nb, .5)]
        m1 = row + nl + nb
        rs, cs, vs, bs = self.soc
        rr.append(m1 + rs); cc.append(cs); vv.append(vs)
        b = np.concatenate([b0, np.zeros(m1 - m0), bs])
        A = sp.csc_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(m1 + 3 * n, nv))
        cones = [clarabel.NonnegativeConeT(m1)] + [clarabel.SecondOrderConeT(3)] * n
        t0 = time.perf_counter()
        s = clarabel.DefaultSolver(sp.csc_matrix((nv, nv)), q, A, b, cones, self.settings)
        sol = s.solve()
        self.tsolve += time.perf_counter() - t0; self.nsolve += 1
        st = str(sol.status)
        self.last_iter = sol.iterations
        if not st.endswith("Solved"):          # "Solved" or "AlmostSolved"; the strict check below decides
            return dict(status=st, cost=np.inf, rects=None)
        z = np.asarray(sol.x)
        x, y, w, h = z[:n], z[n:2 * n], z[2 * n:3 * n], z[3 * n:4 * n]
        if self.shrink:
            w = w * (1 - self.shrink); h = h * (1 - self.shrink)
        R = np.column_stack([x - w / 2, y - h / 2, w, h])
        ok, why = self.strict_ok(R)
        if not ok:
            return dict(status=st + "|strict:" + why, cost=np.inf, rects=R)
        return dict(status=st, cost=self.cost(R), rects=R)

    def cost(self, R):
        c = R[:, :2] + R[:, 2:] / 2
        return float(np.sum(self.pw * (np.abs(c[self.pi, 0] - c[self.pj, 0]) + np.abs(c[self.pi, 1] - c[self.pj, 1]))))

    def strict_ok(self, R):
        """Every constraint to strict_tol: area >= A(1-tol), aspect <= alpha, inside floor and no overlap
        to tol (absolute, length units)."""
        t = self.strict_tol
        x, y, w, h = R.T
        if np.any(w <= 0) or np.any(h <= 0): return False, "size"
        if np.any(w * h < self.A * (1 - t)): return False, "area"
        if t == 0 and np.any(w * h < self.A): return False, "area"
        asp = np.maximum(w / h, h / w)
        if np.any((self.alpha > 0) & (asp > self.alpha)): return False, "aspect"
        if np.any(x < -t) or np.any(y < -t) or np.any(x + w > self.W + t) or np.any(y + h > self.H + t):
            return False, "outside"
        ox = np.minimum((x + w)[:, None], (x + w)[None, :]) - np.maximum(x[:, None], x[None, :])
        oy = np.minimum((y + h)[:, None], (y + h)[None, :]) - np.maximum(y[:, None], y[None, :])
        ov = (ox > t) & (oy > t) if t > 0 else (ox > 0) & (oy > 0); np.fill_diagonal(ov, False)
        if ov.any(): return False, "overlap"
        return True, ""

    def solve_sp(self, gp, gm):
        Lf, Bf = sp_relations(gp, gm, reduce=False)
        L, B = reduce_relations(Lf, Bf)
        return self.solve(L, B, Lf, Bf)


def from_problem(prob, **kw):
    return Stage2(prob.W, prob.H, prob.areas, prob.alpha, prob.Wt, inst=prob.inst, **kw)
