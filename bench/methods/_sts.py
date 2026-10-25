"""Shared helpers for the STS-based baselines (swap, sa, ga, cro).

All STS searches work on the dummy view of the problem (prob.areas_d, prob.Wt_d, prob.alpha_d,
prob.n_d); for instances without empty area this equals the plain problem. Fitness =
cost + prob.pen * total aspect excess (static penalty). Only strictly feasible genomes
(no department above its aspect limit) are kept as results; the first prob.n decoded
rectangles are returned.
"""
import time
import numpy as np
from numba import njit
from baselines.cro_sts import sts


class View:
    def __init__(self, prob):
        self.prob = prob
        self.n = int(prob.n_d)
        self.areas = np.ascontiguousarray(prob.areas_d, dtype=np.float64)
        self.Wt = np.ascontiguousarray(prob.Wt_d, dtype=np.float64)
        self.alpha = np.ascontiguousarray(prob.alpha_d, dtype=np.float64)
        self.W, self.H, self.pen = float(prob.W), float(prob.H), float(prob.pen)

    def evaluate(self, seq, cuts, codes):
        return sts.evaluate(seq, cuts, codes, self.areas, self.W, self.H, self.Wt, self.alpha, self.pen, 0.0)

    def random(self, rng):
        n = self.n
        return (rng.permutation(n).astype(np.int64), rng.permutation(n - 1).astype(np.int64),
                rng.integers(0, 4, n - 1).astype(np.int64))

    def rects(self, seq, cuts, codes):
        R = sts.decode(seq, cuts, codes, self.areas, self.W, self.H)
        return R[:self.prob.n].tolist()


class Best:
    """Best strictly feasible genome seen so far, with an anytime trace
    [[seconds since start, best feasible cost], ...] (one point per improvement; close() adds a final point)."""
    def __init__(self, t0=None):
        self.cost = np.inf; self.g = None; self.t = None; self.n_upd = 0; self.trace = []
        self.t0 = time.time() if t0 is None else t0

    def offer(self, cost, ninf, seq, cuts, codes, t=None):
        if ninf == 0 and cost < self.cost - 1e-9:
            if t is None: t = time.time() - self.t0
            self.cost = float(cost); self.g = (seq.copy(), cuts.copy(), codes.copy()); self.t = t; self.n_upd += 1
            self.trace.append([round(float(t), 4), float(cost)])
            return True
        return False

    def close(self):
        """Final trace point (time of the end of the run, best feasible cost or None)."""
        self.trace.append([round(time.time() - self.t0, 4), float(self.cost) if np.isfinite(self.cost) else None])
        return self.trace


@njit(cache=True)
def apply_move(seq, cuts, codes, kind, i, j):
    """kind 0: swap seq[i], seq[j]; 1: swap cuts[i], cuts[j]; 2: codes[i] = j (a code value)."""
    if kind == 0:
        t = seq[i]; seq[i] = seq[j]; seq[j] = t
    elif kind == 1:
        t = cuts[i]; cuts[i] = cuts[j]; cuts[j] = t
    else:
        codes[i] = j


@njit(cache=True)
def random_move(n):
    """One SA/GA move, each kind with probability 1/3; returns (kind, i, j). For kind 2, j is the new code
    (different from the old one is ensured by the caller via random_code)."""
    kind = np.random.randint(3)
    if kind == 0:
        i = np.random.randint(n); j = np.random.randint(n - 1)
        if j >= i: j += 1
    elif kind == 1:
        i = np.random.randint(n - 1); j = np.random.randint(n - 2) if n > 2 else 0
        if n > 2 and j >= i: j += 1
    else:
        i = np.random.randint(n - 1); j = 0
    return kind, i, j


@njit(cache=True)
def new_code(old):
    c = np.random.randint(3)
    if c >= old: c += 1
    return c
