"""AV16: the two-stage optimisation framework of Anjos & Vieira (2016), Optimization Letters 10:1379-1392.
Re-implemented from the paper text (Sec. 2-4).  No code was published; open points are marked [ours].

Stage 1 (Sec. 2, model (10)-(17)): departments are rectangles centred at (x_i, y_i) (floor centre at the
origin), variables x, y, w, h.  Objective
    sum_{i<j} c_ij D2_ij + K * sum_{i<j} (T2_ij / D2_ij - 1),
    D2_ij = (x_i - x_j)^2 + (y_i - y_j)^2,  T2_ij = ((w_i + w_j)^2 + (h_i + h_j)^2) / 4,  K = a * sum c_ij,
s.t. w_i h_i >= A_i, beta w_i >= h_i, beta h_i >= w_i, and containment with at most 10 % of a
department's width / height outside the floor (Sec. 2, last paragraph).
Start (Sec. 4): w_i = 2 sqrt(A_i beta), h_i = 2 sqrt(A_i / beta), centres on a circle of radius 100,
x_i = 100 cos(2 pi (i-1)/N), y_i = 100 sin(2 pi (i-1)/N).
[ours] Solver: the paper uses SNOPT (SQP, local optimum); here scipy SLSQP (also SQP) with analytic
gradients.  The circle radius 100 is the paper's value; it is kept as given.

Relative positions (Sec. 3, rule for (29)): for each pair, if |dx| > |dy| in the Stage-1 solution the
pair is separated horizontally in its Stage-1 order, otherwise vertically.

Stage 2 (Sec. 3, model (19)-(29)): exact convex model with those separations, w_i h_i >= A_i, aspect and
containment -- the same model as our Stage 2, so bench/socp.py (Clarabel SOCP, exact area via the rotated
cone) is used unchanged.  The paper uses CPLEX.

Parameter a (Sec. 4.1): drawn uniformly from [0.01, 2.0] for each trial; the framework is run for as many
values as the time budget allows (the paper: 20 to 2000 values) and the best layout is returned.
meta['trace'] = [[seconds, best feasible cost], ...].
"""
import time
import numpy as np
from scipy.optimize import minimize
from bench.socp import from_problem, reduce_relations

MAXIT_SCALED = 2000
OUT = 0.1          # at most 10 % of a department's width / height outside the floor in Stage 1


class Stage1:
    def __init__(self, prob, scaled=False):
        self.n = n = prob.n
        # [ours] optional numerical scaling for SLSQP: lengths in units of L = max(W, H), objective divided by
        # sum c_ij L^2.  The model is the same (positions scale by 1/L, the objective by a constant).  Used only
        # when the unscaled solve stops on a numerical failure (Du62: exit 8 after ~5 iterations), because SLSQP's
        # path, hence the local optimum it reaches, depends on the scaling.
        self.L = L = max(float(prob.W), float(prob.H)) if scaled else 1.0
        self.A = np.asarray(prob.areas, float) / L ** 2
        self.beta = np.where(prob.alpha > 0, prob.alpha, 1e3)
        self.W, self.H = float(prob.W) / L, float(prob.H) / L
        iu = np.triu_indices(n, 1)
        self.pi, self.pj = iu
        self.csum = np.asarray(prob.Wt, float)[iu].sum()
        self.cs = self.csum if scaled else 1.0
        self.c = np.asarray(prob.Wt, float)[iu] / self.cs
        # linear constraints  G z <= g  (containment, aspect)
        rows, g = [], []
        I = np.eye(n)
        Z = np.zeros((n, n))
        # x + (0.5 - OUT) w <= W/2 ;  -x + (0.5 - OUT) w <= W/2 ; same for y with h
        k = 0.5 - OUT
        rows += [np.hstack([I, Z, k * I, Z]), np.hstack([-I, Z, k * I, Z]),
                 np.hstack([Z, I, Z, k * I]), np.hstack([Z, -I, Z, k * I])]
        g += [np.full(n, self.W / 2)] * 2 + [np.full(n, self.H / 2)] * 2
        # h - beta w <= 0 ; w - beta h <= 0
        B = np.diag(self.beta)
        rows += [np.hstack([Z, Z, -B, I]), np.hstack([Z, Z, I, -B])]
        g += [np.zeros(n), np.zeros(n)]
        self.G = np.vstack(rows); self.g = np.concatenate(g)

    def start(self):
        n = self.n
        t = 2 * np.pi * np.arange(n) / n
        R = 100 / self.L                         # the paper's radius 100, in scaled units
        return np.concatenate([R * np.cos(t), R * np.sin(t), 2 * np.sqrt(self.A * self.beta),
                               2 * np.sqrt(self.A / self.beta)])

    def solve(self, a, maxiter=500):
        n, pi, pj, c = self.n, self.pi, self.pj, self.c
        K = a * self.csum / (self.cs * self.L ** 2)   # Eq. 18, K = a sum c_ij (objective divided by cs L^2)

        def fg(z):
            x, y, w, h = z[:n], z[n:2 * n], z[2 * n:3 * n], z[3 * n:]
            dx = x[pi] - x[pj]; dy = y[pi] - y[pj]
            D2 = np.maximum(dx * dx + dy * dy, 1e-12)
            sw = w[pi] + w[pj]; sh = h[pi] + h[pj]
            T2 = (sw * sw + sh * sh) / 4
            f = np.sum(c * D2) + K * np.sum(T2 / D2 - 1)
            gD = c - K * T2 / (D2 * D2)          # d f / d D2
            gT = K / D2                          # d f / d T2
            gx = np.zeros(n); gy = np.zeros(n); gw = np.zeros(n); gh = np.zeros(n)
            np.add.at(gx, pi, 2 * dx * gD); np.add.at(gx, pj, -2 * dx * gD)
            np.add.at(gy, pi, 2 * dy * gD); np.add.at(gy, pj, -2 * dy * gD)
            np.add.at(gw, pi, gT * sw / 2); np.add.at(gw, pj, gT * sw / 2)
            np.add.at(gh, pi, gT * sh / 2); np.add.at(gh, pj, gT * sh / 2)
            return f, np.concatenate([gx, gy, gw, gh])

        G, g, A = self.G, self.g, self.A
        cons = [dict(type="ineq", fun=lambda z: g - G @ z, jac=lambda z: -G),
                dict(type="ineq", fun=lambda z: z[2 * n:3 * n] * z[3 * n:] - A,
                     jac=lambda z: np.hstack([np.zeros((n, 2 * n)), np.diag(z[3 * n:]), np.diag(z[2 * n:3 * n])]))]
        bounds = [(None, None)] * (2 * n) + [(1e-6, None)] * (2 * n)
        r = minimize(fg, self.start(), jac=True, method="SLSQP", constraints=cons, bounds=bounds,
                     options=dict(maxiter=maxiter, ftol=1e-9))
        return r.x, r


def relations(z, n):
    x, y = z[:n], z[n:2 * n]
    dx = x[None, :] - x[:, None]; dy = y[None, :] - y[:, None]      # dx[i,j] = x_j - x_i
    horiz = np.abs(dx) > np.abs(dy)
    L = horiz & (dx > 0)                 # i left of j
    B = (~horiz) & (dy > 0)              # i below j  (ties: vertical, order by dy > 0)
    tie = (~horiz) & (dy == 0)
    B |= tie & np.triu(np.ones((n, n), bool), 1)
    np.fill_diagonal(L, False); np.fill_diagonal(B, False)
    return L, B


def solve(prob, seed, time_limit, log=print):
    t0 = time.time(); rng = np.random.default_rng(seed)
    S1 = Stage1(prob); S1s = None; S2 = from_problem(prob)
    best = np.inf; best_R = None; trace = []; trials = 0; feas = 0; s1_fail = 0; t1 = t2 = 0.0; rescaled = 0
    alphas = []
    while time.time() - t0 < time_limit:
        a = float(rng.uniform(0.01, 2.0))
        ta = time.time()
        z, r = S1.solve(a)
        if r.status == 8:                        # numerical failure of SLSQP: retry on the scaled model
            S1s = S1s or Stage1(prob, scaled=True)
            z, r = S1s.solve(a, maxiter=MAXIT_SCALED); rescaled += 1
        t1 += time.time() - ta; trials += 1
        if not np.all(np.isfinite(z)):
            s1_fail += 1; continue
        Lf, Bf = relations(z, prob.n)
        L, B = reduce_relations(Lf, Bf)
        tb = time.time()
        out = S2.solve(L, B, Lf, Bf)
        t2 += time.time() - tb
        if out["rects"] is not None and np.isfinite(out["cost"]):
            feas += 1
            if out["cost"] < best - 1e-9:
                best = out["cost"]; best_R = out["rects"]; alphas.append(a)
                trace.append([round(time.time() - t0, 4), float(best)])
    trace.append([round(time.time() - t0, 4), float(best) if np.isfinite(best) else None])
    meta = dict(trace=trace, trials=trials, rescaled=rescaled, stage2_feasible=feas, stage1_failed=s1_fail,
                t_stage1=round(t1, 2), t_stage2=round(t2, 2), best_alpha=alphas[-1] if alphas else None,
                own_cost=float(best), feasible_found=bool(np.isfinite(best)))
    return {"rects": None if best_R is None else np.asarray(best_R).tolist(), "meta": meta}
