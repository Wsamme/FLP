"""Coral reefs optimization (CRO) with slicing tree structure, after Jeong & Chae (2026),
J. Eng. Res. 14, 1016-1028, Sec. "Coral reefs optimization" and Algorithm 1.

Parameters from the paper (Table 5): rho0 = 0.35, Fb = 0.23, Fa = 0.27, Fd = 0.38, Pd = 0.1;
grids 4x4 / 7x7 / 10x10; at most 30,000 iterations; stop after 1000 iterations without improvement;
iterated local search stops after 30 failures; larvae try 3 cells; levy flight when > 80 % of corals
are identical (applied to the top 40 %, bottom 50 % removed, best 10 % kept).

Points the paper does not specify are marked ASSUMPTION and listed in README.md.
"""
import numpy as np
from . import sts

P = dict(rho0=0.35, Fb=0.23, Fa=0.27, Fd=0.38, Pd=0.1, max_iter=30000, stall=1000,
         ils_fail=30, settle_tries=3, levy_same=0.8, same_by='fitness', bud_tries=3, pen_scale=1.0, nodup=False, penalty='static')


class Coral:
    __slots__ = ('seq', 'cuts', 'codes', 'fit', 'cost', 'ex', 'ninf')

    def __init__(self, seq, cuts, codes):
        self.seq, self.cuts, self.codes = seq, cuts, codes

    def copy(self):
        c = Coral(self.seq.copy(), self.cuts.copy(), self.codes.copy())
        c.fit, c.cost, c.ex, c.ninf = self.fit, self.cost, self.ex, self.ninf
        return c

    def key(self):
        return self.seq.tobytes() + self.cuts.tobytes() + self.codes.tobytes()


class CRO:
    def __init__(self, areas, W, H, Wt, alpha, grid, seed, params=None):
        self.p = dict(P, **(params or {}))
        self.areas = np.asarray(areas, float); self.W, self.H = float(W), float(H)
        self.Wt = np.asarray(Wt, float); self.alpha = np.asarray(alpha, float)
        self.n = len(areas); self.N = grid * grid
        self.rng = np.random.default_rng(seed); sts.seed(int(seed))
        # ASSUMPTION: penalty for aspect-ratio excess; pen = sum of all pair weights x (W + H),
        # an upper bound on the cost, so a total excess of 1 outweighs any cost difference.
        self.pen = float(np.triu(self.Wt, 1).sum() * (self.W + self.H)) * self.p['pen_scale']
        # OPTION 'adaptive': Tate & Smith (1995) penalty used in earlier slicing-tree work,
        # fitness = cost + (V_feas - V_all) * ninf^3, V_feas / V_all = best feasible / best unpenalised cost so far
        self.D = 0.0
        if self.p['penalty'] == 'adaptive':
            self.D = self.pen; self.pen = 0.0
        self.v_feas = np.inf; self.v_all = np.inf
        self.nev = 0

    # ---- helpers
    def evaluate(self, c):
        c.fit, c.cost, c.ex, c.ninf = sts.evaluate(c.seq, c.cuts, c.codes, self.areas, self.W, self.H,
                                                   self.Wt, self.alpha, self.pen, self.D)
        self.nev += 1
        self.v_all = min(self.v_all, c.cost)
        if c.ninf == 0: self.v_feas = min(self.v_feas, c.cost)
        return c

    def refit(self, c):
        c.fit = c.cost + self.pen * c.ex + self.D * c.ninf ** 3
        return c

    def update_D(self, reef):
        if self.p['penalty'] != 'adaptive':
            return
        if np.isfinite(self.v_feas):
            self.D = max(self.v_feas - self.v_all, 1e-6)
        for c in reef:
            if c is not None: self.refit(c)

    def random_coral(self):
        n, r = self.n, self.rng
        return self.evaluate(Coral(r.permutation(n), r.permutation(n - 1), r.integers(0, 4, n - 1)))

    def ils(self, c):
        c.fit, ne = sts.ils(c.seq, c.cuts, c.codes, c.fit, self.areas, self.W, self.H, self.Wt,
                            self.alpha, self.pen, self.p['ils_fail'], self.D)
        self.nev += ne
        return self.evaluate(c)

    # ---- crossover (broadcast spawning): one method per pair, applied to all three vectors
    def _pmx(self, a, b):
        m = len(a); i, j = sorted(self.rng.choice(m + 1, 2, replace=False))
        child = -np.ones(m, int); child[i:j] = a[i:j]
        for k in range(i, j):
            v = b[k]
            if v in child[i:j]:
                continue
            pos = k
            while i <= pos < j:
                pos = int(np.where(b == a[pos])[0][0])
            child[pos] = v
        mask = child < 0; child[mask] = b[mask]
        return child

    def _ox(self, a, b):
        m = len(a); i, j = sorted(self.rng.choice(m + 1, 2, replace=False))
        child = -np.ones(m, int); child[i:j] = a[i:j]
        rest = [v for v in np.r_[b[j:], b[:j]] if v not in set(a[i:j])]
        idx = [k for k in np.r_[np.arange(j, m), np.arange(0, j)] if child[k] < 0]
        child[idx] = rest
        return child

    def _half_perm(self, a, b):
        # ASSUMPTION: "half crossover" on a permutation = first half of a, rest in b's order
        h = len(a) // 2; first = list(a[:h]); s = set(first)
        return np.array(first + [v for v in b if v not in s])

    def crossover(self, x, y):
        m = self.rng.integers(3)
        perm = [self._pmx, self._half_perm, self._ox][m]
        seq = perm(x.seq, y.seq); cuts = perm(x.cuts, y.cuts)
        # ASSUMPTION: cut codes are not permutations; PMX/OX are applied as two-point crossover,
        # half crossover as first half from x, second half from y
        h = len(x.codes) // 2
        if m == 1:
            codes = np.r_[x.codes[:h], y.codes[h:]]
        else:
            i, j = sorted(self.rng.choice(len(x.codes) + 1, 2, replace=False))
            codes = y.codes.copy(); codes[i:j] = x.codes[i:j]
        return self.evaluate(Coral(seq.astype(np.int64), cuts.astype(np.int64), codes.astype(np.int64)))

    # ---- mutation (brooding): swap, inversion, partial shuffle, random cut codes; one per coral
    def mutate(self, c):
        c = c.copy(); r = self.rng; m = r.integers(4)
        for v in (c.seq, c.cuts, c.codes):
            L = len(v); i, j = sorted(r.choice(L, 2, replace=False))
            if m == 0:
                v[i], v[j] = v[j], v[i]
            elif m == 1:
                v[i:j + 1] = v[i:j + 1][::-1].copy()
            elif m == 2:
                seg = v[i:j + 1].copy(); r.shuffle(seg); v[i:j + 1] = seg
            elif v is c.codes:            # random cut codes between two points (codes only)
                v[i:j + 1] = r.integers(0, 4, j - i + 1)
            else:                         # ASSUMPTION: for permutations this method acts as swap
                v[i], v[j] = v[j], v[i]
        return self.evaluate(c)

    # ---- levy flight (Fig. 9): values 0.1*k plus a levy step, re-sorted
    def _levy(self, size, beta=1.5):
        # ASSUMPTION: Mantegna's algorithm, beta = 1.5, step scaled by 0.1
        from math import gamma, sin, pi
        s = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = self.rng.normal(0, s, size); v = self.rng.normal(0, 1, size)
        return 0.1 * u / np.abs(v) ** (1 / beta)

    def levy(self, c):
        c = c.copy()
        for name in ('seq', 'cuts'):
            v = getattr(c, name); vals = 0.1 * np.arange(1, len(v) + 1) + self._levy(len(v))
            setattr(c, name, v[np.argsort(vals)].copy())
        jump = np.rint(self._levy(len(c.codes)) * 10).astype(np.int64)
        c.codes = (c.codes + jump) % 4
        return self.evaluate(c)

    # ---- reef operations
    def settle(self, reef, child, tries):
        if self.p['nodup']:
            # OPTION (not in the paper): reject a larva whose objective equals an existing coral's
            for c in reef:
                if c is not None and abs(c.fit - child.fit) < 1e-6:
                    return False
        cells = self.rng.permutation(self.N)[:tries]
        for k in cells:
            if reef[k] is None or child.fit < reef[k].fit:
                reef[k] = child; return True
        return False

    def run(self, log_every=0):
        p, r, N = self.p, self.rng, self.N
        reef = [None] * N
        for k in r.choice(N, int(round(N * p['rho0'])), replace=False):
            reef[k] = self.random_coral()
        best = min((c for c in reef if c is not None), key=lambda c: c.fit).copy()
        best_feas = best.copy() if best.ex == 0 else None
        stall, it = 0, 0
        hist = []
        while it < p['max_iter'] and stall < p['stall']:
            it += 1
            self.update_D(reef)
            if self.p['penalty'] == 'adaptive': self.refit(best)
            occ = [k for k in range(N) if reef[k] is not None]
            r.shuffle(occ); nb = int(round(len(occ) * p['Fb']))
            spawners, brooders = occ[:nb], occ[nb:]
            larvae = []
            # broadcast spawning: tournament pairs, each coral used once
            pool = list(spawners)
            while len(pool) >= 2:
                cand = r.choice(len(pool), min(4, len(pool)), replace=False)
                cand = sorted(cand, key=lambda q: reef[pool[q]].fit)
                a, b = pool[cand[0]], pool[cand[1]]
                larvae.append(self.crossover(reef[a], reef[b]))
                pool = [q for q in pool if q not in (a, b)]
            # brooding
            for k in brooders:
                larvae.append(self.mutate(reef[k]))
            # larvae setting with iterated local search
            for ch in larvae:
                self.ils(ch); self.settle(reef, ch, p['settle_tries'])
            # budding: clone the best Fa fraction, improve, implant
            occ = sorted([k for k in range(N) if reef[k] is not None], key=lambda k: reef[k].fit)
            for k in occ[:max(1, int(round(len(occ) * p['Fa'])))]:
                cl = self.ils(reef[k].copy()); self.settle(reef, cl, p['bud_tries'] if p['bud_tries'] else N)
            # depredation
            if r.random() < p['Pd']:
                occ = sorted([k for k in range(N) if reef[k] is not None], key=lambda k: reef[k].fit)
                for k in occ[::-1][:int(round(len(occ) * p['Fd']))]:
                    if len(occ) > 2: reef[k] = None
            # levy flight when the reef has converged
            occ = sorted([k for k in range(N) if reef[k] is not None], key=lambda k: reef[k].fit)
            # ASSUMPTION: "same coral" judged by objective value (default) or by representation
            keys = [round(reef[k].fit, 6) if p['same_by'] == 'fitness' else reef[k].key() for k in occ]
            if keys and max(keys.count(x) for x in set(keys)) > p['levy_same'] * len(keys):
                self.levy_count = getattr(self, 'levy_count', 0) + 1
                top = [reef[k] for k in occ[:max(1, int(round(0.4 * len(occ))))]]
                keep = max(1, int(round(0.1 * len(occ))))
                for k in occ[::-1][:int(round(0.5 * len(occ)))]:
                    if k not in occ[:keep]: reef[k] = None
                for c in top:
                    self.settle(reef, self.levy(c), N)
            # bookkeeping
            cur = min((reef[k] for k in range(N) if reef[k] is not None), key=lambda c: c.fit)
            improved = False
            if cur.fit < best.fit - 1e-9:
                best = cur.copy(); improved = True
            for k in range(N):
                c = reef[k]
                if c is not None and c.ex == 0 and (best_feas is None or c.cost < best_feas.cost - 1e-9):
                    best_feas = c.copy(); improved = True
            stall = 0 if improved else stall + 1
            if log_every and it % log_every == 0:
                hist.append((it, self.nev, best_feas.cost if best_feas else None))
        return dict(best_feasible=best_feas, best=best, iterations=it, evaluations=self.nev, history=hist, levy=getattr(self, 'levy_count', 0))
