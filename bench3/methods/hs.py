"""HS: harmony search on the slicing tree, Kang & Chae (2017), Expert Systems with Applications 79:269-281.
Re-implemented from the paper text (Sec. 4-5); no code was published.  Open points are marked [ours].

Representation (Sec. 4.1): cut point sequence (n-1), cut code sequence (n-1, values 0-3 incl. inverse cuts),
department sequence (n); decoded by baselines/cro_sts/sts.py (the same STS decoder used for the CRO
baseline of Jeong & Chae 2026, from the same group).  Free floor area: dummy departments (Sec. 5.3), i.e.
the dummy view of bench.common (SC30: 17 blocks).

Penalty (Sec. 4.3): OFV x u^v, u = 1.2, v = number of departments violating the aspect limit.

Improvisation (Sec. 4.2), one new harmony per iteration, value by value:
  memory consideration (prob. HMCR): the value at that position is taken from a randomly chosen harmony
  of the HM; random selection (1 - HMCR): a random value.  Duplicates in the department or cut point
  sequence are repaired [ours: each repeated value, scanning left to right, is replaced by the first missing
  value in the order of the harmony it was copied from -- the paper's Fig. 4 example replaces the later
  duplicate with a value from that harmony].
  pitch adjustment (prob. PAR): one of four rules at random -- swap two departments, swap two cut points,
  insert a department at another position, insert a cut point at another position.
  pitch re-adjustment: every cut code is re-drawn from {0,1,2,3} with probability PRAR.
Update: the new harmony replaces the worst one in the HM if its penalised OFV is better.
Parameters (Table 2, tuned by irace per instance): HMS, HMCR, PAR, PRAR below.
[ours] Stop: the protocol time budget (the paper stops after stopImp non-improving iterations).
Returns the best strictly feasible layout seen.  meta['trace'] = [[seconds, best feasible cost], ...].
"""
import time
import numpy as np
from numba import njit
from baselines.cro_sts import sts

PARAMS = {"O7": (15, 0.89, 0.30, 0.15), "vC10a": (18, 0.92, 0.59, 0.28), "AB20": (17, 0.96, 0.52, 0.10),
          "SC30": (23, 0.94, 0.20, 0.05), "Du62": (49, 0.95, 0.27, 0.03)}
U = 1.2
CHUNK = 500


@njit(cache=True)
def seed_nb(s):
    np.random.seed(s)


@njit(cache=True)
def repair_perm(v, src):
    """Make v a permutation: repeated values (later occurrences) are replaced by the missing values in the
    order in which they appear in src."""
    m = v.shape[0]
    seen = np.zeros(m, np.bool_)
    dup = np.zeros(m, np.bool_)
    for k in range(m):
        if seen[v[k]]:
            dup[k] = True
        else:
            seen[v[k]] = True
    p = 0
    for k in range(m):
        if dup[k]:
            while seen[src[p]]:
                p += 1
            v[k] = src[p]; seen[src[p]] = True
    return v


@njit(cache=True)
def improvise(HS, HC, HK, hmcr, par, prar, seq, cuts, codes):
    hms, n = HS.shape
    # department sequence
    src = np.random.randint(hms)
    for k in range(n):
        if np.random.random() < hmcr:
            seq[k] = HS[np.random.randint(hms), k]
        else:
            seq[k] = np.random.randint(n)
    repair_perm(seq, HS[src])
    # cut point sequence
    src = np.random.randint(hms)
    for k in range(n - 1):
        if np.random.random() < hmcr:
            cuts[k] = HC[np.random.randint(hms), k]
        else:
            cuts[k] = np.random.randint(n - 1)
    repair_perm(cuts, HC[src])
    # cut codes
    for k in range(n - 1):
        if np.random.random() < hmcr:
            codes[k] = HK[np.random.randint(hms), k]
        else:
            codes[k] = np.random.randint(4)
    # pitch adjustment: one of four rules
    if np.random.random() < par:
        r = np.random.randint(4)
        v = seq if (r == 0 or r == 2) else cuts
        m = v.shape[0]
        if m >= 2:
            i = np.random.randint(m); j = np.random.randint(m - 1)
            if j >= i:
                j += 1
            if r < 2:                          # swap
                t = v[i]; v[i] = v[j]; v[j] = t
            else:                              # insert v[i] at position j
                t = v[i]
                if i < j:
                    for k in range(i, j):
                        v[k] = v[k + 1]
                else:
                    for k in range(i, j, -1):
                        v[k] = v[k - 1]
                v[j] = t
    # pitch re-adjustment of the cut codes
    for k in range(n - 1):
        if np.random.random() < prar:
            codes[k] = np.random.randint(4)


@njit(cache=True)
def run_chunk(HS, HC, HK, HF, hmcr, par, prar, nit, areas, W, H, Wt, alpha, bs, bc, bk, best, seq, cuts, codes):
    """nit improvisations.  HF: penalised OFV of each harmony.  best[0] = best feasible cost."""
    for _ in range(nit):
        improvise(HS, HC, HK, hmcr, par, prar, seq, cuts, codes)
        nf, c, e, ni = sts.evaluate(seq, cuts, codes, areas, W, H, Wt, alpha, 0.0, 0.0)
        f = c * U ** ni
        if ni == 0 and c < best[0]:
            best[0] = c; bs[:] = seq; bc[:] = cuts; bk[:] = codes
        w = 0
        for p in range(1, HF.shape[0]):
            if HF[p] > HF[w]:
                w = p
        if f < HF[w]:
            HS[w] = seq; HC[w] = cuts; HK[w] = codes; HF[w] = f


def solve(prob, seed, time_limit, log=print):
    t0 = time.time(); seed_nb(int(seed)); rng = np.random.default_rng(seed)
    hms, hmcr, par, prar = PARAMS[prob.inst.name] if prob.inst.name in PARAMS else (prob.n, 0.9, 0.3, 0.1)
    n = int(prob.n_d)
    areas = np.ascontiguousarray(prob.areas_d, float); Wt = np.ascontiguousarray(prob.Wt_d, float)
    alpha = np.ascontiguousarray(prob.alpha_d, float); W, H = float(prob.W), float(prob.H)
    HS = np.array([rng.permutation(n) for _ in range(hms)], np.int64)
    HC = np.array([rng.permutation(n - 1) for _ in range(hms)], np.int64)
    HK = rng.integers(0, 4, (hms, n - 1)).astype(np.int64)
    HF = np.empty(hms); best = np.array([np.inf])
    bs, bc, bk = HS[0].copy(), HC[0].copy(), HK[0].copy()
    for p in range(hms):
        _, c, e, ni = sts.evaluate(HS[p], HC[p], HK[p], areas, W, H, Wt, alpha, 0.0, 0.0)
        HF[p] = c * U ** ni
        if ni == 0 and c < best[0]:
            best[0] = c; bs[:] = HS[p]; bc[:] = HC[p]; bk[:] = HK[p]
    seq = np.zeros(n, np.int64); cuts = np.zeros(n - 1, np.int64); codes = np.zeros(n - 1, np.int64)
    trace = [[round(time.time() - t0, 4), float(best[0])]] if np.isfinite(best[0]) else []
    it = 0
    while time.time() - t0 < time_limit:
        old = best[0]
        run_chunk(HS, HC, HK, HF, hmcr, par, prar, CHUNK, areas, W, H, Wt, alpha, bs, bc, bk, best, seq, cuts, codes)
        it += CHUNK
        if best[0] < old:
            trace.append([round(time.time() - t0, 4), float(best[0])])
    trace.append([round(time.time() - t0, 4), float(best[0]) if np.isfinite(best[0]) else None])
    meta = dict(trace=trace, improvisations=it, params=dict(HMS=hms, HMCR=hmcr, PAR=par, PRAR=prar),
                own_cost=float(best[0]), feasible_found=bool(np.isfinite(best[0])))
    if not np.isfinite(best[0]):
        p = int(np.argmin(HF)); bs, bc, bk = HS[p], HC[p], HK[p]
    R = sts.decode(bs, bc, bk, areas, W, H)
    return {"rects": R[:prob.n].tolist(), "meta": meta}
