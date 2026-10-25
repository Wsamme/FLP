"""Core of our method ("ours"): STS simulated annealing (Stage 1) + sequence-pair search with an exact
SOCP geometry (Stage 2).  Method wrappers: bench/methods/ours.py, ours_s1.py, ours_s2.py.

Stage 1  (slicing tree, fast, approximate shapes handled by penalty)
  Representation: the STS genome of Kang & Chae 2017 / Jeong & Chae 2026 (department sequence, cut order,
  cut codes); decoded by baselines.cro_sts.sts.decode (area-proportional cuts, exact areas, full floor).
  Search: simulated annealing, moves = swap two departments, insert one department elsewhere, swap two cut
  positions, set one cut code at random, swap two cut codes.  Fitness = cost + pen * (total aspect excess),
  pen = 0.05 * sum(w) * (W + H) (the static penalty of our CRO baseline).  Only aspect-feasible layouts are
  recorded as results.  Instances with empty floor area use the same dummy blocks as the CRO baseline.

  Cycles: 10,000 n iterations, T from 0.05 c to 1e-5 c (c = cost of the cycle's start), geometric; each cycle
  starts from the best genome, every 4th from a random one.  The 4 best distinct cycle results form the elite
  pool handed to Stage 2.

Stage 2  (sequence pair + SOCP, exact)
  Representation: sequence pair (gp, gm) (Murata et al. 1996); every pair of departments gets a left/below
  relation, non-slicing layouts included.  Evaluation: bench.socp.Stage2 (convex SOCP with exact areas).
  Start: each elite layout is converted to an SP consistent with its geometry (bench.socp.rects_to_sp), so the
  first SOCP is never worse than the elite.  Search: simulated annealing over sequence pairs, infeasible SOCP =
  rejected move; T from 1e-2 c0 to 2e-6 c0 (geometric in elapsed time); moves see SPSearch.neighbour.
  SOCP results are cached by the SP's relation matrix.
"""
import time, math
import numpy as np
from numba import njit
from baselines.cro_sts import sts
from bench.socp import from_problem, rects_to_sp, sp_relations, reduce_relations


class Trace:
    """Anytime trace: [[seconds since start, best feasible cost], ...], appended only on improvement."""

    def __init__(self, t0=None):
        self.t0 = time.time() if t0 is None else t0; self.pts = []; self.best = np.inf

    def add(self, c):
        if c < self.best - 1e-9:
            self.best = float(c); self.pts.append([round(time.time() - self.t0, 3), float(c)])

    def final(self):
        if self.pts and np.isfinite(self.best):
            self.pts.append([round(time.time() - self.t0, 3), self.best])
        return self.pts


# ----------------------------------------------------------------------------------------------- Stage 1
@njit(cache=True)
def _sts_sa(seq, cuts, codes, areas, W, H, Wt, alpha, pen, n_iter, T0, T1, nreal):
    n = seq.shape[0]
    f, c, e, ni = sts.evaluate(seq, cuts, codes, areas, W, H, Wt, alpha, pen, 0.0)
    bseq = seq.copy(); bcuts = cuts.copy(); bcodes = codes.copy()
    bcost = c if ni == 0 else 1e300
    tmp = np.empty(n, np.int64)
    for it in range(n_iter):
        T = T0 * (T1 / T0) ** (it / n_iter)
        r = np.random.randint(5)
        i = 0; j = 0; old = 0
        if r == 0:
            i = np.random.randint(n); j = np.random.randint(n)
            seq[i], seq[j] = seq[j], seq[i]
        elif r == 1:
            i = np.random.randint(n); j = np.random.randint(n)
            tmp[:] = seq
            v = seq[i]
            if i < j:
                seq[i:j] = tmp[i + 1:j + 1]
            elif j < i:
                seq[j + 1:i + 1] = tmp[j:i]
            seq[j] = v
        elif r == 2:
            i = np.random.randint(n - 1); j = np.random.randint(n - 1)
            cuts[i], cuts[j] = cuts[j], cuts[i]
        elif r == 3:
            i = np.random.randint(n - 1); old = codes[i]
            codes[i] = (old + 1 + np.random.randint(3)) % 4
        else:
            i = np.random.randint(n - 1); j = np.random.randint(n - 1)
            codes[i], codes[j] = codes[j], codes[i]
        f2, c2, e2, ni2 = sts.evaluate(seq, cuts, codes, areas, W, H, Wt, alpha, pen, 0.0)
        if f2 <= f or np.random.random() < math.exp(-(f2 - f) / T):
            f = f2
            if ni2 == 0 and c2 < bcost - 1e-12:
                bcost = c2; bseq[:] = seq; bcuts[:] = cuts; bcodes[:] = codes
        else:
            if r == 0:
                seq[i], seq[j] = seq[j], seq[i]
            elif r == 1:
                seq[:] = tmp
            elif r == 2:
                cuts[i], cuts[j] = cuts[j], cuts[i]
            elif r == 3:
                codes[i] = old
            else:
                codes[i], codes[j] = codes[j], codes[i]
    return bseq, bcuts, bcodes, bcost, f


def sts_stage(prob, seed, time_limit, log=print, n_iter=None, T0rel=0.05, T1rel=1e-5, init=None, n_elite=4, trace=None):
    """Iterated SA on the STS genome until time_limit.  Each cycle starts from the best feasible genome
    found so far (first cycle: random or `init`).  Returns dict(seq, cuts, codes, cost, rects, cycles, evals)."""
    rng = np.random.default_rng(seed); sts.seed(int(seed) + 12345)
    areas, Wt, alpha, n = prob.areas_d, prob.Wt_d, prob.alpha_d, prob.n_d
    pen = float(prob.pen)
    if n_iter is None:
        n_iter = int(10000 * n)                 # ASSUMPTION (tuned on AB20): cycle length
    t0 = time.time(); best = None; cycles = 0; evals = 0; elite = []
    while time.time() - t0 < time_limit or best is None:
        if best is None or cycles % 4 == 3:     # every 4th cycle restarts from scratch (diversification)
            if init is not None and best is None:     # continuation: start like a non-random cycle
                seq, cuts, codes = [np.array(a, np.int64) for a in init]; T0 = T0rel / 4
            else:
                seq = rng.permutation(n).astype(np.int64); cuts = rng.permutation(n - 1).astype(np.int64)
                codes = rng.integers(0, 4, n - 1).astype(np.int64); T0 = T0rel
        else:
            seq, cuts, codes = best[0].copy(), best[1].copy(), best[2].copy(); T0 = T0rel / 4
        f, c, _, _ = sts.evaluate(seq, cuts, codes, areas, prob.W, prob.H, Wt, alpha, pen, 0.0)
        scale = c if c > 0 else 1.0
        bs, bc, bd, bcost, _ = _sts_sa(seq, cuts, codes, areas, prob.W, prob.H, Wt, alpha, pen,
                                       n_iter, T0 * scale, T1rel * scale, prob.n)
        cycles += 1; evals += n_iter
        if bcost < 1e299:
            # elite pool: the K best distinct (by cost) cycle results, starting points for Stage 2
            if all(abs(bcost - e[3]) > 1e-7 * bcost for e in elite):
                elite.append((bs.copy(), bc.copy(), bd.copy(), bcost))
                elite.sort(key=lambda e: e[3]); del elite[n_elite:]
        if bcost < 1e299 and (best is None or bcost < best[3] - 1e-9):
            best = (bs.copy(), bc.copy(), bd.copy(), bcost)
            if trace is not None: trace.add(bcost)      # Stage-1 points are recorded at cycle ends
            log(f"  [S1] cycle {cycles} t={time.time()-t0:.1f}s cost {bcost:.4f}")
        if best is None and time.time() - t0 > 3 * time_limit + 60:
            break
    if best is None:
        return None
    R = sts.decode(best[0], best[1], best[2], areas, prob.W, prob.H)[:prob.n]
    el = [(e[3], sts.decode(e[0], e[1], e[2], areas, prob.W, prob.H)[:prob.n]) for e in elite]
    return dict(seq=best[0], cuts=best[1], codes=best[2], cost=best[3], rects=R, cycles=cycles, evals=evals,
                elite=el)


# ----------------------------------------------------------------------------------------------- Stage 2
class SPSearch:
    """Simulated annealing over sequence pairs; every candidate is evaluated exactly by the Stage-2 SOCP."""

    def __init__(self, prob, seed, log=print, trace=None, p_near=0.8, **s2kw):
        self.prob = prob; self.s2 = from_problem(prob, **s2kw); self.n = prob.n
        self.trace = trace; self.p_near = p_near
        self.rng = np.random.default_rng(seed); self.log = log
        self.cache = {}; self.nev = 0; self.ninf = 0

    def evaluate(self, gp, gm):
        Lf, Bf = sp_relations(gp, gm, reduce=False)
        key = Lf.tobytes() + Bf.tobytes()        # SPs with the same relations share one solve
        if key in self.cache:
            return self.cache[key]
        L, B = reduce_relations(Lf, Bf)
        o = self.s2.solve(L, B, Lf, Bf); self.nev += 1
        if o["rects"] is None or not np.isfinite(o["cost"]):
            self.ninf += 1
            out = (np.inf, None)
        else:
            out = (o["cost"], o["rects"])
        if len(self.cache) > 200000: self.cache.clear()
        self.cache[key] = out
        return out

    def neighbour(self, gp, gm, near):
        """Moves (choices not from a paper): pick department a; b is one of its k nearest departments
        (centroid distance in the current layout) with prob. 0.8, otherwise random.
          0 swap a,b in gp      1 swap a,b in gm      2 swap a,b in both (exchange positions)
          3 move a next to b in gp and gm (insert before/after, at random in each sequence)"""
        n = self.n; r = self.rng
        a = int(r.integers(n))
        if r.random() < self.p_near:
            b = int(near[a][r.integers(len(near[a]))])
        else:
            b = int(r.integers(n - 1)); b += b >= a
        t = int(r.integers(4))
        gp = gp.copy(); gm = gm.copy()
        if t in (0, 2):
            ia, ib = np.nonzero(gp == a)[0][0], np.nonzero(gp == b)[0][0]; gp[ia], gp[ib] = b, a
        if t in (1, 2):
            ia, ib = np.nonzero(gm == a)[0][0], np.nonzero(gm == b)[0][0]; gm[ia], gm[ib] = b, a
        if t == 3:
            out = []
            for g in (gp, gm):
                h = [int(v) for v in g if v != a]
                k = h.index(b) + int(r.integers(2))
                h.insert(k, a); out.append(np.array(h))
            gp, gm = out
        return gp, gm

    @staticmethod
    def near_lists(R, k):
        c = R[:, :2] + R[:, 2:] / 2
        d = np.abs(c[:, None, 0] - c[None, :, 0]) + np.abs(c[:, None, 1] - c[None, :, 1])
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, 1)[:, :k]

    def run(self, gp, gm, time_limit, T0rel=1e-2, T1rel=2e-6, k_near=6, restart_after=None):
        """Time-based geometric cooling from T0rel*c0 to T1rel*c0; reheats (from the best SP) when the
        schedule ends before the time limit.  Returns best (cost, rects, gp, gm)."""
        t0 = time.time()
        c, R = self.evaluate(gp, gm)
        if not np.isfinite(c):
            raise RuntimeError("start sequence pair infeasible in Stage 2")
        best = (c, R, gp.copy(), gm.copy()); cur = (c, R, gp, gm)
        if self.trace is not None: self.trace.add(c)
        c0 = c; near = self.near_lists(R, k_near)
        # a cycle lasts at most `cyc` seconds; the first cycle may use the whole budget
        nsteps = 0; tcyc0 = time.time(); cyc = time_limit if restart_after is None else restart_after
        nacc = 0
        while True:
            el = time.time() - t0
            if el >= time_limit: break
            frac = min(1.0, (time.time() - tcyc0) / min(cyc, time_limit - (tcyc0 - t0) + 1e-9))
            if frac >= 1.0:          # reheat from the best
                cur = best; tcyc0 = time.time(); frac = 0.0
                near = self.near_lists(best[1], k_near)
            T = c0 * T0rel * (T1rel / T0rel) ** frac
            gp2, gm2 = self.neighbour(cur[2], cur[3], near)
            c2, R2 = self.evaluate(gp2, gm2); nsteps += 1
            if not np.isfinite(c2): continue
            if c2 <= cur[0] or self.rng.random() < math.exp(-(c2 - cur[0]) / T):
                cur = (c2, R2, gp2, gm2); nacc += 1
                if nacc % 50 == 0: near = self.near_lists(R2, k_near)
                if c2 < best[0] - 1e-9:
                    best = (c2, R2, gp2.copy(), gm2.copy())
                    if self.trace is not None: self.trace.add(c2)
                    self.log(f"  [S2] t={time.time()-t0:.1f}s ev={self.nev} cost {c2:.4f}")
        self.log(f"  [S2] done: {nsteps} steps, {self.nev} SOCP solves ({self.ninf} infeasible), "
                 f"{self.s2.tsolve / max(1, self.s2.nsolve) * 1e3:.1f} ms/solve")
        return best


# ----------------------------------------------------------------------------------------------- full method
N_MIN = 3000      # Stage-2 search only if the remaining time affords at least this many SOCP solves (our choice)


def random_feasible_sp(S, time_limit, log=print):
    """Random sequence pairs until one is feasible in Stage 2 (used by the no-Stage-1 ablation)."""
    t0 = time.time(); k = 0
    while time.time() - t0 < time_limit:
        gp = S.rng.permutation(S.n); gm = S.rng.permutation(S.n); k += 1
        c, R = S.evaluate(gp, gm)
        if np.isfinite(c):
            log(f"  random SP feasible after {k} tries ({time.time()-t0:.1f}s), cost {c:.4f}")
            return gp, gm, c, R, k
    log(f"  no feasible random SP in {k} tries")
    return None, None, np.inf, None, k


def run_ours(prob, seed, time_limit, log=print, s1frac=0.5, n_elite=4, stage1=True, p_near=0.8, T0=1e-2,
             budget_rule=True, warm=()):
    """The full method and its ablations.  Self-contained from the seed: nothing external is read unless
    `warm` (a list of layout files, OFF by default; ours.py only sets it from the OURS_WARM variable)."""
    import json
    from flp.evaluate import evaluate
    t0 = time.time(); trace = Trace(t0)
    S = SPSearch(prob, seed, log, trace=trace, p_near=p_near)
    starts = []; s1 = None
    if not stage1:
        gp, gm, c, R, k = random_feasible_sp(S, time_limit, log)
        meta0 = dict(random_sp_tries=k)
        if R is None:
            return {"rects": None, "meta": dict(meta0, error="no feasible random sequence pair", trace=trace.final())}
        starts = [("random_sp", c, R)]
    else:
        meta0 = {}
        s1 = sts_stage(prob, seed, s1frac * time_limit, log, n_elite=n_elite, trace=trace)
        if s1 is not None and budget_rule:
            q = S.s2; tq = time.time()
            o = q.solve_sp(*rects_to_sp(np.asarray(s1["rects"])[:prob.n])); tq = time.time() - tq
            rest = time_limit - (time.time() - t0)
            if rest / max(tq, 1e-4) < N_MIN:
                log(f"  SOCP {tq*1e3:.0f} ms: {rest/tq:.0f} solves < {N_MIN} -> Stage 1 continues, Stage 2 = polish only")
                s1b = sts_stage(prob, seed + 1000, max(rest - 3 * tq - 0.5, 0.0), log,
                                init=(s1["seq"], s1["cuts"], s1["codes"]), n_elite=n_elite, trace=trace)
                if s1b is not None and s1b["cost"] < s1["cost"]:
                    s1 = s1b
                R = np.asarray(s1["rects"])[:prob.n]
                o = q.solve_sp(*rects_to_sp(R))
                if o["rects"] is not None and np.isfinite(o["cost"]) and o["cost"] <= s1["cost"]:
                    R = o["rects"]; trace.add(o["cost"])
                return {"rects": np.asarray(R).tolist(), "meta": dict(stage1_cost=s1["cost"], stage2="polish only",
                        stage2_cost=o["cost"], socp_ms=tq * 1e3, trace=trace.final())}
        if s1 is not None:
            starts += [(f"stage1_elite{k}", c, np.asarray(R)[:prob.n]) for k, (c, R) in enumerate(s1["elite"])]
    for f in warm:
        R = np.asarray(json.load(open(f))["rects"], float)[:prob.n]
        ev = evaluate(prob.inst, [tuple(r) for r in R])
        if ev["feasible"]:
            starts.append((f, ev["cost"], R))
    if not starts:
        return {"rects": None, "meta": {"error": "no feasible start", "trace": trace.final()}}
    starts.sort(key=lambda s: s[1]); starts = starts[:max(n_elite, 1)]
    best = None; hist = []
    for k, (name, c_start, R) in enumerate(starts):
        rest = time_limit - (time.time() - t0)
        if rest <= 0.5 and best is not None: break
        tl = rest / (len(starts) - k)
        log(f"  Stage 2 from {name} (cost {c_start:.4f}), {tl:.0f} s")
        gp, gm = rects_to_sp(R)
        c, Rb, gp, gm = S.run(gp, gm, max(tl, 0.5), T0rel=T0)
        hist.append((name, c_start, c))
        if best is None or c < best[0]:
            best = (c, Rb, gp, gm, name, c_start)
    c, Rb, gp, gm, name, c_start = best
    meta = dict(meta0, stage1_cost=None if s1 is None else s1["cost"], stage1_evals=None if s1 is None else s1["evals"],
                start=name, start_cost=c_start, stage2_cost=c, stage2_runs=hist, socp_solves=S.nev,
                socp_infeasible=S.ninf, socp_ms=1e3 * S.s2.tsolve / max(1, S.s2.nsolve), gp=gp.tolist(),
                gm=gm.tolist(), trace=trace.final())
    return {"rects": np.asarray(Rb).tolist(), "meta": meta}
