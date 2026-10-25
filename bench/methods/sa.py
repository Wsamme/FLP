"""Simulated annealing on the slicing-tree (STS) encoding.

Moves (probability 1/3 each): swap two positions of seq; swap two positions of cuts; set one cut code to
a different value (0-3). Fitness = cost + prob.pen * aspect excess (static penalty).
Metropolis acceptance. Initial temperature: from a uniformly random start, 200 random moves are evaluated;
T0 = -median(positive deltas) / ln(0.75), i.e. a median worsening move is accepted with p = 0.75.
Cooling by elapsed time: T(t) = T0 * (1e-4)^(t / time_limit), updated every chunk of moves
(chunk = 2000 moves, our choice), so T reaches T0*1e-4 at the time limit.
Restart rule (our choice): if the best fitness has not improved for 10 % of the time limit, the current
solution is reset to the best-fitness genome (temperature schedule continues).
Returns the best strictly feasible layout among all evaluated genomes.
meta['trace'] = [[seconds, best feasible cost], ...]: one point after each chunk (2000 moves) in which the
best feasible cost improved (time = end of that chunk), plus a final point.
"""
import math, time
import numpy as np
from numba import njit
from baselines.cro_sts import sts
from bench.methods._sts import View, Best, apply_move, random_move, new_code

CHUNK = 2000


@njit(cache=True)
def chunk(seq, cuts, codes, f, T, nm, areas, W, H, Wt, alpha, pen, bseq, bcuts, bcodes, bfit, fseq, fcuts, fcodes, fcost):
    """nm Metropolis moves at temperature T. Updates current (seq,cuts,codes) in place, the best-fitness
    genome (b*) and best feasible genome (f*). Returns (f, bfit, fcost, accepted)."""
    n = seq.shape[0]; acc = 0
    for _ in range(nm):
        kind, i, j = random_move(n)
        if kind == 2:
            old = codes[i]; j = new_code(old)
        elif kind == 0:
            old = 0
        else:
            old = 0
        apply_move(seq, cuts, codes, kind, i, j)
        nf, c, e, ni = sts.evaluate(seq, cuts, codes, areas, W, H, Wt, alpha, pen, 0.0)
        d = nf - f
        if d <= 0 or np.random.random() < math.exp(-d / T):
            f = nf; acc += 1
            if ni == 0 and c < fcost:
                fcost = c; fseq[:] = seq; fcuts[:] = cuts; fcodes[:] = codes
            if f < bfit:
                bfit = f; bseq[:] = seq; bcuts[:] = cuts; bcodes[:] = codes
        else:
            if kind == 2:
                codes[i] = old
            else:
                apply_move(seq, cuts, codes, kind, i, j)
    return f, bfit, fcost, acc


def solve(prob, seed, time_limit, log):
    t0 = time.time(); V = View(prob); rng = np.random.default_rng(seed); sts.seed(int(seed))
    seq, cuts, codes = V.random(rng)
    f, c, e, ni = V.evaluate(seq, cuts, codes)
    # T0 from 200 random moves around the start
    deltas = []
    for _ in range(200):
        s2, c2, k2 = seq.copy(), cuts.copy(), codes.copy()
        kind, i, j = random_move(V.n)
        if kind == 2: j = new_code(k2[i])
        apply_move(s2, c2, k2, kind, i, j)
        d = V.evaluate(s2, c2, k2)[0] - f
        if d > 0: deltas.append(d)
    T0 = -float(np.median(deltas)) / math.log(0.75) if deltas else 1.0
    bseq, bcuts, bcodes, bfit = seq.copy(), cuts.copy(), codes.copy(), f
    fseq, fcuts, fcodes = seq.copy(), cuts.copy(), codes.copy()
    fcost = c if ni == 0 else np.inf
    nev = 201; acc = 0; restarts = 0; t_imp = 0.0; t_fbest = 0.0 if ni == 0 else None
    trace = [[round(time.time() - t0, 4), float(fcost)]] if ni == 0 else []
    while True:
        t = time.time() - t0
        if t >= time_limit: break
        T = T0 * (1e-4) ** (t / time_limit)
        ob, ofc = bfit, fcost
        f, bfit, fcost, a = chunk(seq, cuts, codes, f, T, CHUNK, V.areas, V.W, V.H, V.Wt, V.alpha, V.pen,
                                  bseq, bcuts, bcodes, bfit, fseq, fcuts, fcodes, fcost)
        nev += CHUNK; acc += a
        t = time.time() - t0
        if fcost < ofc:
            t_fbest = t; trace.append([round(t, 4), float(fcost)])
        if bfit < ob - 1e-9: t_imp = t
        elif t - t_imp > 0.1 * time_limit:
            seq[:], cuts[:], codes[:] = bseq, bcuts, bcodes; f = bfit; restarts += 1; t_imp = t
    trace.append([round(time.time() - t0, 4), float(fcost) if np.isfinite(fcost) else None])
    meta = dict(trace=trace, T0=T0, evaluations=nev, accept_rate=acc / max(nev - 201, 1), restarts=restarts,
                own_cost=float(fcost), t_best=t_fbest, feasible_found=bool(np.isfinite(fcost)))
    if not np.isfinite(fcost):
        return {"rects": V.rects(bseq, bcuts, bcodes), "meta": meta}
    return {"rects": V.rects(fseq, fcuts, fcodes), "meta": meta}
