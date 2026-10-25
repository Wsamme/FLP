"""Adapter: CRO + STS (our reproduction of Jeong & Chae 2026, baselines/cro_sts/cro.py) under the common
wall-clock budget.

Settings as in our reproduction runs: grid 7x7, nodup=True, static penalty with pen_scale = 0.05, i.e.
CRO.pen = 0.05 * sum_{i<j} w_ij * (W + H) = prob.pen (checked by an assertion). Dummy view for SC30
(17 blocks, no flow, no shape limit). Note: the earlier full-length SC30 runs used the adaptive
(Tate-Smith) penalty; here every STS method uses the same static penalty prob.pen.
Budget (our choice): the paper's stop rules (30,000 iterations / 1,000 stalled iterations) are disabled
and the run stops at the wall-clock limit, like the other stochastic methods. The time check is done in a
subclass's evaluate(); when the time is up it raises an internal exception that ends run(). The subclass
also records the best strictly feasible genome among all evaluated corals (a superset of what run() tracks).
cro.py itself is not modified.
meta['trace'] = [[seconds, best feasible cost], ...] at each improvement plus a final point.
"""
import time
import numpy as np
from baselines.cro_sts.cro import CRO
from bench.methods._sts import View, Best


class _TimeUp(Exception):
    pass


class TimedCRO(CRO):
    def setup(self, deadline, t0):
        self.deadline, self.t0, self.bestf = deadline, t0, Best(t0)

    def evaluate(self, c):
        c = super().evaluate(c)
        self.bestf.offer(c.cost, c.ninf, c.seq, c.cuts, c.codes, time.time() - self.t0)
        if time.time() >= self.deadline:
            raise _TimeUp
        return c


def solve(prob, seed, time_limit, log):
    t0 = time.time(); V = View(prob)
    cro = TimedCRO(V.areas, V.W, V.H, V.Wt, V.alpha, 7, seed,
                   dict(pen_scale=0.05, nodup=True, penalty='static', max_iter=10 ** 9, stall=10 ** 9))
    assert abs(cro.pen - prob.pen) <= 1e-9 * max(1.0, prob.pen), (cro.pen, prob.pen)
    cro.setup(t0 + time_limit, t0)
    res = None
    try:
        res = cro.run()
    except _TimeUp:
        pass
    b = cro.bestf
    meta = dict(trace=b.close(), evaluations=cro.nev, levy=getattr(cro, 'levy_count', 0), own_cost=b.cost, t_best=b.t,
                feasible_found=b.g is not None, finished_run=res is not None)
    if b.g is None:
        return {"rects": None, "meta": meta}
    return {"rects": V.rects(*b.g), "meta": meta}
