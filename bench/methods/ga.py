"""Genetic algorithm on the slicing-tree (STS) encoding, without local search.

Population 50 of uniformly random genomes; fitness = cost + prob.pen * aspect excess (static penalty).
Each generation: 2 elites copied; 48 children, each from two parents chosen by binary tournament
(tournament size 2), crossover always applied (rate 1.0, our choice): PMX on seq, PMX on cuts, one-point
crossover on codes; then with probability 0.2 one mutation, the same move set as sa.py (swap two in seq /
swap two in cuts / change one code, 1/3 each). Generational replacement, until the time limit.
Returns the best strictly feasible layout among all evaluated genomes.
meta['trace'] = [[seconds, best feasible cost], ...] at each improvement plus a final point.
"""
import time
import numpy as np
from baselines.cro_sts import sts
from bench.methods._sts import View, Best, apply_move, random_move, new_code

POP, ELITE, PMUT = 50, 2, 0.2


def pmx(a, b, rng):
    m = len(a)
    if m < 2: return a.copy()
    i, j = sorted(rng.choice(m + 1, 2, replace=False))
    child = -np.ones(m, np.int64); child[i:j] = a[i:j]
    pos_b = np.empty(m, np.int64); pos_b[b] = np.arange(m)
    seg = set(a[i:j].tolist())
    for k in range(i, j):
        v = b[k]
        if v in seg: continue
        pos = k
        while i <= pos < j:
            pos = pos_b[a[pos]]
        child[pos] = v
    mask = child < 0; child[mask] = b[mask]
    return child


def solve(prob, seed, time_limit, log):
    t0 = time.time(); V = View(prob); rng = np.random.default_rng(seed); sts.seed(int(seed))
    best = Best(t0); nev = 0
    pop = []
    for _ in range(POP):
        g = V.random(rng); f, c, e, ni = V.evaluate(*g); nev += 1
        best.offer(c, ni, *g, t=time.time() - t0); pop.append((f, g))
    gen = 0
    while time.time() - t0 < time_limit:
        gen += 1
        pop.sort(key=lambda p: p[0])
        fits = np.array([p[0] for p in pop])
        new = pop[:ELITE]
        while len(new) < POP:
            par = []
            for _ in range(2):
                i, j = rng.integers(POP, size=2); par.append(pop[i if fits[i] <= fits[j] else j][1])
            (sa_, ca, ka), (sb, cb, kb) = par
            seq = pmx(sa_, sb, rng); cuts = pmx(ca, cb, rng)
            p = rng.integers(1, len(ka)) if len(ka) > 1 else 0
            codes = np.r_[ka[:p], kb[p:]].astype(np.int64)
            if rng.random() < PMUT:
                kind, i, j = random_move(V.n)
                if kind == 2: j = new_code(codes[i])
                apply_move(seq, cuts, codes, kind, i, j)
            f, c, e, ni = V.evaluate(seq, cuts, codes); nev += 1
            best.offer(c, ni, seq, cuts, codes, t=time.time() - t0)
            new.append((f, (seq, cuts, codes)))
            if time.time() - t0 >= time_limit: break
        pop = new + pop[len(new):] if len(new) < POP else new
    meta = dict(trace=best.close(), generations=gen, evaluations=nev, own_cost=best.cost, t_best=best.t, feasible_found=best.g is not None)
    if best.g is None:
        pop.sort(key=lambda p: p[0])
        return {"rects": V.rects(*pop[0][1]), "meta": meta}
    return {"rects": V.rects(*best.g), "meta": meta}
