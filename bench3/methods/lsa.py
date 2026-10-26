"""LSA: learning-based simulated annealing on the flexible bay structure (FBS).
Re-implementation of Lin, Shen, Wu & Zhong (2024), Soft Computing 28:5667-5682, from the paper text
(Sec. 3-5, Algorithms 2-3, Eqs. 9-10).  No code was published; points the paper leaves open are marked
[ours] below.

Representation (Sec. 3.2): a bay list = facility sequence `seq` + bay-end flags `brk` (brk[k] = 1 if seq[k]
is the last facility of its bay; brk[n-1] = 1).  Each bay spans the full floor length L; its width is
(sum of its areas) / L; facilities are stacked in sequence order, each with its exact area.
[ours] Orientation: the paper does not say whether bays run along W or H.  Every solution is decoded in
both orientations and the better one (lower penalised fitness) is used -- a choice that can only help LSA
(on a square floor, Du62, the two orientations are mirror images, so only one is decoded).
[ours] Empty floor area: the paper does not say.  As for every slicing/bay baseline in this benchmark,
the dummy view of the problem is used (SC30: the free area split into 17 zero-flow, shape-free blocks,
bench.common.DUMMIES); bays then fill the floor.  Only the n real departments are returned.  (O7's free
area is 0.02 %, left as a strip.)

Fitness (Eq. 9, improved penalty, k = 1):  f = sum f_ij d_ij + D_inf^k * V_worst, D_inf = number of
facilities violating the aspect limit, V_worst = worst cost among the feasible solutions seen so far.

Initial solution (Algorithm 3): facilities sorted by area (ascending), random two-point swap, then bays
are formed greedily: from the current index, the longest run of the remaining facilities that forms a
feasible bay [ours: the paper's loop removes the last facility until the bay is feasible]; if no single
facility fits, restart with a new swap.

Operators (Sec. 4.2): facility swap, bay swap, bay flip, infeasible-solution repair.
  bay flip: pick a facility; last in its bay -> detached into a new bay of its own; in the middle -> the
            facilities after it form a new bay.  [ours] If it is already alone in its bay, it is merged
            into the next bay (otherwise bays could only ever split).
  repair:   pick a bay [ours: an infeasible bay if there is one]; too wide (flat facilities) -> split at a
            random facility; too narrow (tall facilities) -> merged into a random other bay.
Operator choice: epsilon-greedy on Q values; Q of every operator starts at 1/f(s) at each outer iteration;
after use, Q <- (Q * times + 1/f(ns)) / (times + 1) (Eq. 10).  With probability EPS the operator is drawn
uniformly from all four (the repair operator is only reachable this way).  [ours] EPS = 0.1.
Acceptance: Metropolis.  After an accepted move, greedy two-stage local search (Sec. 4.3): (1) swap
adjacent bays one after another, keep a swap if f improves; (2) inside each bay swap adjacent facilities,
same rule.
Temperature (Sec. 4.3): 100 neighbours of the initial solution, |f(s) - f(ns)| sorted ascending, the 90th
value is t0.  Cooling t <- 0.99 t per outer iteration, G = 1000 outer iterations (Sec. 5).
[ours] Protocol time budget: the paper runs G x M = 1000 x 100n moves; here the outer-iteration index is
tied to elapsed time, g = floor(1000 t / time_limit), so the full cooling schedule (t0 -> 0.99^1000 t0)
is completed exactly at the time limit, as for every other method in the benchmark.

Returns the best feasible layout found (gBest).  meta['trace'] = [[seconds, best feasible cost], ...].
"""
import math, time
import numpy as np
from numba import njit

G = 1000
COOL = 0.99
EPS = 0.1
CHUNK = 256
# Readings of two points the paper leaves open (environment LSA_VARIANT = "<reward><vworst>", default "00"):
#   reward 0: Eq. 10 as printed, reward 1/f(ns);  1: relative improvement (f(s) - f(ns)) / f(s)
#             ("a reward Q is assigned based on the degree of improvement it achieves", Sec. 4.2)
#   vworst 0: worst cost among all feasible solutions seen;  1: among those of the current outer iteration
import os as _os
_V = _os.environ.get("LSA_VARIANT", "00")
REWARD, VWORST = int(_V[0]), int(_V[1])


@njit(cache=True)
def seed_nb(s):
    np.random.seed(s)


@njit(cache=True)
def decode(seq, brk, A, W, H, alpha, o, cx, cy, wv, hv):
    """Fill centres / sizes for orientation o (0: bays are columns spanning H; 1: rows spanning W).
    Returns the number of aspect violations."""
    n = seq.shape[0]
    L = H if o == 0 else W
    pos = 0.0; start = 0; ninf = 0
    for k in range(n):
        if brk[k] == 1:
            S = 0.0
            for p in range(start, k + 1):
                S += A[seq[p]]
            bw = S / L
            off = 0.0
            for p in range(start, k + 1):
                d = seq[p]; ln = A[d] / bw
                if o == 0:
                    wv[d] = bw; hv[d] = ln; cx[d] = pos + bw / 2; cy[d] = off + ln / 2
                else:
                    hv[d] = bw; wv[d] = ln; cy[d] = pos + bw / 2; cx[d] = off + ln / 2
                off += ln
                if alpha[d] > 0:
                    r = bw / ln if bw > ln else ln / bw
                    if r > alpha[d] + 1e-9:
                        ninf += 1
            pos += bw; start = k + 1
    return ninf


@njit(cache=True)
def pair_cost(cx, cy, pi, pj, pw):
    c = 0.0
    for t in range(pi.shape[0]):
        c += pw[t] * (abs(cx[pi[t]] - cx[pj[t]]) + abs(cy[pi[t]] - cy[pj[t]]))
    return c


@njit(cache=True)
def fitness(seq, brk, A, W, H, alpha, pi, pj, pw, vworst, buf):
    """Best of the two orientations.  Returns (f, cost, ninf, orientation)."""
    cx, cy, wv, hv = buf[0], buf[1], buf[2], buf[3]
    bf = np.inf; bc = 0.0; bn = 0; bo = 0
    no = 1 if W == H else 2                       # square floor: both orientations are the same layout family
    for o in range(no):
        ni = decode(seq, brk, A, W, H, alpha, o, cx, cy, wv, hv)
        c = pair_cost(cx, cy, pi, pj, pw)
        f = c + ni * vworst
        if f < bf:
            bf = f; bc = c; bn = ni; bo = o
    return bf, bc, bn, bo


@njit(cache=True)
def bays(brk):
    """Start and end (inclusive) positions of each bay."""
    n = brk.shape[0]
    nb = 0
    for k in range(n):
        nb += brk[k]
    st = np.empty(nb, np.int64); en = np.empty(nb, np.int64)
    b = 0; s = 0
    for k in range(n):
        if brk[k] == 1:
            st[b] = s; en[b] = k; b += 1; s = k + 1
    return st, en


@njit(cache=True)
def rebuild(seq, brk, order, st, en, out_seq, out_brk):
    """Write the bays in the given order into out arrays."""
    p = 0
    for b in order:
        for k in range(st[b], en[b] + 1):
            out_seq[p] = seq[k]; out_brk[p] = 0; p += 1
        out_brk[p - 1] = 1


@njit(cache=True)
def op_fswap(s, b):
    n = s.shape[0]
    i = np.random.randint(n); j = np.random.randint(n - 1)
    if j >= i:
        j += 1
    t = s[i]; s[i] = s[j]; s[j] = t


@njit(cache=True)
def op_bswap(s, b, tmp_s, tmp_b):
    st, en = bays(b)
    nb = st.shape[0]
    if nb < 2:
        op_fswap(s, b); return
    i = np.random.randint(nb); j = np.random.randint(nb - 1)
    if j >= i:
        j += 1
    order = np.arange(nb)
    order[i] = j; order[j] = i
    rebuild(s, b, order, st, en, tmp_s, tmp_b)
    s[:] = tmp_s; b[:] = tmp_b


@njit(cache=True)
def op_flip(s, b):
    n = s.shape[0]
    p = np.random.randint(n)
    if b[p] == 0:
        b[p] = 1                                   # facilities after p form a new bay
    else:
        alone = (p == 0) or (b[p - 1] == 1)
        if not alone:
            b[p - 1] = 1                           # detach p into a bay of its own
        elif p < n - 1:
            b[p] = 0                               # [ours] alone: merge into the next bay


@njit(cache=True)
def op_repair(s, b, A, W, H, alpha, o, buf, tmp_s, tmp_b):
    cx, cy, wv, hv = buf[0], buf[1], buf[2], buf[3]
    decode(s, b, A, W, H, alpha, o, cx, cy, wv, hv)
    st, en = bays(b)
    nb = st.shape[0]
    # bay state: +1 too wide (a facility is flat across the bay), -1 too narrow (a facility is tall), 0 ok
    state = np.zeros(nb, np.int64)
    for q in range(nb):
        for k in range(st[q], en[q] + 1):
            d = s[k]
            if alpha[d] > 0:
                bw = wv[d] if o == 0 else hv[d]
                ln = hv[d] if o == 0 else wv[d]
                if bw / ln > alpha[d] + 1e-9:
                    state[q] = 1
                elif ln / bw > alpha[d] + 1e-9 and state[q] == 0:
                    state[q] = -1
    nbad = 0
    for q in range(nb):
        if state[q] != 0:
            nbad += 1
    if nbad > 0:
        r = np.random.randint(nbad); q = -1
        for t in range(nb):
            if state[t] != 0:
                if r == 0:
                    q = t; break
                r -= 1
    else:
        q = np.random.randint(nb)
    if state[q] == 1 and en[q] > st[q]:
        k = st[q] + np.random.randint(en[q] - st[q])
        b[k] = 1                                   # split
    elif state[q] == -1 and nb > 1:
        t = np.random.randint(nb - 1)
        if t >= q:
            t += 1
        # move bay q to the end of bay t
        p = 0
        for u in range(nb):
            if u == q:
                continue
            for k in range(st[u], en[u] + 1):
                tmp_s[p] = s[k]; tmp_b[p] = 0; p += 1
            if u == t:
                for k in range(st[q], en[q] + 1):
                    tmp_s[p] = s[k]; tmp_b[p] = 0; p += 1
            tmp_b[p - 1] = 1
        s[:] = tmp_s; b[:] = tmp_b


@njit(cache=True)
def apply_op(op, s, b, A, W, H, alpha, o, buf, tmp_s, tmp_b):
    if op == 0:
        op_fswap(s, b)
    elif op == 1:
        op_bswap(s, b, tmp_s, tmp_b)
    elif op == 2:
        op_flip(s, b)
    else:
        op_repair(s, b, A, W, H, alpha, o, buf, tmp_s, tmp_b)


@njit(cache=True)
def local_search(s, b, f, A, W, H, alpha, pi, pj, pw, vworst, buf, tmp_s, tmp_b, ts, tb):
    """Greedy two-stage local search (Sec. 4.3).  Returns (f, cost, ninf, o, evaluations)."""
    ne = 0
    ff, c, ni, o = fitness(s, b, A, W, H, alpha, pi, pj, pw, vworst, buf)
    # stage 1: adjacent bays
    st, en = bays(b)
    nb = st.shape[0]
    for q in range(nb - 1):
        st, en = bays(b)
        order = np.arange(st.shape[0])
        order[q] = q + 1; order[q + 1] = q
        rebuild(s, b, order, st, en, tmp_s, tmp_b)
        f2, c2, n2, o2 = fitness(tmp_s, tmp_b, A, W, H, alpha, pi, pj, pw, vworst, buf); ne += 1
        if f2 < ff:
            s[:] = tmp_s; b[:] = tmp_b; ff = f2; c = c2; ni = n2; o = o2
    # stage 2: adjacent facilities inside each bay
    st, en = bays(b)
    for q in range(st.shape[0]):
        for k in range(st[q], en[q]):
            t = s[k]; s[k] = s[k + 1]; s[k + 1] = t
            f2, c2, n2, o2 = fitness(s, b, A, W, H, alpha, pi, pj, pw, vworst, buf); ne += 1
            if f2 < ff:
                ff = f2; c = c2; ni = n2; o = o2
            else:
                t = s[k]; s[k] = s[k + 1]; s[k + 1] = t
    return ff, c, ni, o, ne


@njit(cache=True)
def chunk(s, b, f, c, ni, o, T, nm, Q, times, eps, A, W, H, alpha, pi, pj, pw, vw,
          gs, gb, gstate, buf, tmp_s, tmp_b, ts, tb, ns, nbk, stats, rew):
    """nm SA moves at temperature T (inner loop of Algorithm 2).  gstate = [gBest cost, gBest orientation].
    vw[0] = V_worst (updated in place).  stats = [evaluations, accepted, op counts 0-3]."""
    for _ in range(nm):
        if np.random.random() < eps:
            op = np.random.randint(4)
        else:
            op = 0
            for k in range(1, 3):
                if Q[k] > Q[op] or (Q[k] == Q[op] and np.random.random() < 0.5):
                    op = k
        ns[:] = s; nbk[:] = b
        apply_op(op, ns, nbk, A, W, H, alpha, o, buf, tmp_s, tmp_b)
        f2, c2, n2, o2 = fitness(ns, nbk, A, W, H, alpha, pi, pj, pw, vw[0], buf)
        stats[0] += 1; stats[2 + op] += 1
        if n2 == 0:
            if c2 > vw[0]:
                vw[0] = c2
            if c2 < gstate[0]:
                gstate[0] = c2; gstate[1] = o2; gs[:] = ns; gb[:] = nbk
        r = 1.0 / f2 if rew == 0 else (f - f2) / f
        Q[op] = (Q[op] * times[op] + r) / (times[op] + 1); times[op] += 1
        d = f2 - f
        if d < 0 or np.random.random() < math.exp(-abs(d) / T):
            s[:] = ns; b[:] = nbk
            f, c, ni, o, ne = local_search(s, b, f2, A, W, H, alpha, pi, pj, pw, vw[0], buf, tmp_s, tmp_b, ts, tb)
            stats[0] += ne; stats[1] += 1
            if ni == 0:
                if c > vw[0]:
                    vw[0] = c
                if c < gstate[0]:
                    gstate[0] = c; gstate[1] = o; gs[:] = s; gb[:] = b
    return f, c, ni, o


@njit(cache=True)
def bay_ok(seq, start, stop, A, L, alpha):
    S = 0.0
    for p in range(start, stop):
        S += A[seq[p]]
    bw = S / L
    for p in range(start, stop):
        d = seq[p]
        if alpha[d] > 0:
            ln = A[d] / bw
            r = bw / ln if bw > ln else ln / bw
            if r > alpha[d] + 1e-9:
                return False
    return True


@njit(cache=True)
def greedy_init(fp, A, W, H, alpha, o, max_tries, out_s, out_b):
    """Algorithm 3.  Returns True if a feasible bay list was built (in out_s, out_b)."""
    n = fp.shape[0]
    L = H if o == 0 else W
    for _ in range(max_tries):
        fl = fp.copy()
        i = np.random.randint(n); j = np.random.randint(n - 1)
        if j >= i:
            j += 1
        t = fl[i]; fl[i] = fl[j]; fl[j] = t
        idx = 0; ok = True
        while idx < n and ok:
            ln = n - idx
            while ln > 0 and not bay_ok(fl, idx, idx + ln, A, L, alpha):
                ln -= 1
            if ln == 0:
                ok = False
            else:
                for p in range(idx, idx + ln):
                    out_s[p] = fl[p]; out_b[p] = 0
                out_b[idx + ln - 1] = 1
                idx += ln
        if ok:
            return True
    return False


def solve(prob, seed, time_limit, log=print):
    t0w = time.time()
    seed_nb(int(seed)); rng = np.random.default_rng(seed)
    n = int(prob.n_d); A = np.ascontiguousarray(prob.areas_d, float); al = np.ascontiguousarray(prob.alpha_d, float)
    W, H = float(prob.W), float(prob.H); Wt = np.asarray(prob.Wt_d, float)
    iu = np.triu_indices(n, 1); m = Wt[iu] > 0
    pi = np.ascontiguousarray(iu[0][m]).astype(np.int64); pj = np.ascontiguousarray(iu[1][m]).astype(np.int64)
    pw = np.ascontiguousarray(Wt[iu][m], float)
    buf = np.zeros((4, n)); tmp_s = np.zeros(n, np.int64); tmp_b = np.zeros(n, np.int64)
    ts = np.zeros(n, np.int64); tb = np.zeros(n, np.int64); ns = np.zeros(n, np.int64); nbk = np.zeros(n, np.int64)

    # --- Algorithm 3: greedy initial solution (orientation 0 first, then 1) ---
    fp = np.argsort(A, kind="stable").astype(np.int64)
    s = np.zeros(n, np.int64); b = np.zeros(n, np.int64)
    init = "greedy"
    if not greedy_init(fp, A, W, H, al, 0, 2000, s, b) and not greedy_init(fp, A, W, H, al, 1, 2000, s, b):
        init = "random"                            # [ours] fallback: one bay per facility, random order
        s[:] = rng.permutation(n); b[:] = 1
    f, c, ni, o = fitness(s, b, A, W, H, al, pi, pj, pw, 0.0, buf)
    vw = np.array([c if ni == 0 else 0.0])
    if ni > 0:                                     # no feasible solution yet: penalty from the cost scale
        vw[0] = c
    f, c, ni, o = fitness(s, b, A, W, H, al, pi, pj, pw, vw[0], buf)
    gs = s.copy(); gb = b.copy(); gstate = np.array([c if ni == 0 else np.inf, float(o)])

    # --- initial temperature: 100 neighbours, 90th smallest |delta| ---
    ds = []
    for _ in range(100):
        ns[:] = s; nbk[:] = b
        apply_op(int(rng.integers(4)), ns, nbk, A, W, H, al, o, buf, tmp_s, tmp_b)
        ds.append(abs(fitness(ns, nbk, A, W, H, al, pi, pj, pw, vw[0], buf)[0] - f))
    ds.sort(); T0 = ds[89] if ds[89] > 0 else max(max(ds), 1e-9)

    Q = np.zeros(4); times = np.zeros(4); stats = np.zeros(6, np.int64)
    g_last = -1; trace = []
    if np.isfinite(gstate[0]):
        trace.append([round(time.time() - t0w, 4), float(gstate[0])])
    while True:
        t = time.time() - t0w
        if t >= time_limit:
            break
        g = min(int(G * t / time_limit), G - 1)
        if g != g_last:                            # new outer iteration: reset the operator values
            Q[:] = 1.0 / f if REWARD == 0 else 0.0; times[:] = 0; g_last = g
            if VWORST == 1 and ni == 0:            # V_worst over the current outer iteration only
                vw[0] = c
        T = T0 * COOL ** g
        old = gstate[0]
        f, c, ni, o = chunk(s, b, f, c, ni, o, T, CHUNK, Q, times, EPS, A, W, H, al, pi, pj, pw, vw,
                            gs, gb, gstate, buf, tmp_s, tmp_b, ts, tb, ns, nbk, stats, REWARD)
        if gstate[0] < old:
            trace.append([round(time.time() - t0w, 4), float(gstate[0])])
    trace.append([round(time.time() - t0w, 4), float(gstate[0]) if np.isfinite(gstate[0]) else None])
    meta = dict(trace=trace, T0=float(T0), init=init, evaluations=int(stats[0]), accepted=int(stats[1]),
                op_counts=[int(x) for x in stats[2:]], own_cost=float(gstate[0]), V_worst=float(vw[0]),
                outer_iterations_reached=int(g_last + 1), orientation=int(gstate[1]), variant=_V,
                feasible_found=bool(np.isfinite(gstate[0])))
    if not np.isfinite(gstate[0]):
        gs, gb, oo = s, b, o
    else:
        oo = int(gstate[1])
    cx, cy, wv, hv = (np.zeros(n) for _ in range(4))
    decode(gs, gb, A, W, H, al, oo, cx, cy, wv, hv)
    rects = np.column_stack([cx - wv / 2, cy - hv / 2, wv, hv])[:prob.n]
    return {"rects": rects.tolist(), "meta": meta}
