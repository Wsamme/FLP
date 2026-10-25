"""CRAFT-style pairwise exchange, adapted to the slicing-tree (STS) encoding.

CRAFT (Armour & Buffa 1963; Buffa, Armour & Vollmann 1964) improves a grid layout by steepest-descent
exchanges of departments that are adjacent or of equal area. Adaptation here (our choice): the layout is
an STS genome (seq, cuts, codes, see baselines/cro_sts/sts.py); the neighbourhood is every pairwise swap
of two departments in seq, with cuts and codes fixed (any pair, no adjacency/equal-area condition,
since the slicing decoder re-derives the shapes). Steepest descent on fitness = cost + prob.pen * aspect
excess: apply the best improving swap until none improves, then restart from a new uniformly random
genome; repeat until the time limit. Every evaluated feasible genome is a candidate result
(best strictly feasible layout is returned). meta['trace'] = [[seconds, best feasible cost], ...] at each
improvement plus a final point.
"""
import time
import numpy as np
from numba import njit
from baselines.cro_sts import sts
from bench.methods._sts import View, Best


@njit(cache=True)
def scan(seq, cuts, codes, areas, W, H, Wt, alpha, pen):
    """Evaluate all seq swaps. Returns (best fit, bi, bj, best feasible cost, fi, fj, n evals)."""
    n = seq.shape[0]
    bf = np.inf; bi = -1; bj = -1; fc = np.inf; fi = -1; fj = -1; ne = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            t = seq[i]; seq[i] = seq[j]; seq[j] = t
            f, c, e, ni = sts.evaluate(seq, cuts, codes, areas, W, H, Wt, alpha, pen, 0.0); ne += 1
            if f < bf:
                bf = f; bi = i; bj = j
            if ni == 0 and c < fc:
                fc = c; fi = i; fj = j
            t = seq[i]; seq[i] = seq[j]; seq[j] = t
    return bf, bi, bj, fc, fi, fj, ne


def solve(prob, seed, time_limit, log):
    t0 = time.time(); V = View(prob); rng = np.random.default_rng(seed)
    best = Best(t0); nev = 0; starts = 0; steps = 0
    while time.time() - t0 < time_limit:
        seq, cuts, codes = V.random(rng); starts += 1
        f, c, e, ni = V.evaluate(seq, cuts, codes); nev += 1
        best.offer(c, ni, seq, cuts, codes, time.time() - t0)
        while time.time() - t0 < time_limit:
            bf, bi, bj, fc, fi, fj, ne = scan(seq, cuts, codes, V.areas, V.W, V.H, V.Wt, V.alpha, V.pen)
            nev += ne
            if fi >= 0 and fc < best.cost - 1e-9:
                s2 = seq.copy(); s2[fi], s2[fj] = s2[fj], s2[fi]
                best.offer(fc, 0, s2, cuts, codes, time.time() - t0)
            if bf < f - 1e-9:
                seq[bi], seq[bj] = seq[bj], seq[bi]; f = bf; steps += 1
            else:
                break
    meta = dict(trace=best.close(), evaluations=nev, restarts=starts, improving_steps=steps, own_cost=best.cost, t_best=best.t)
    if best.g is None:
        return {"rects": V.rects(seq, cuts, codes), "meta": dict(meta, feasible_found=False)}
    return {"rects": V.rects(*best.g), "meta": dict(meta, feasible_found=True)}
