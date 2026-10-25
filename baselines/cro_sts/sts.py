"""Slicing tree structure (STS) decoding and evaluation, numba-accelerated.

Representation (Jeong & Chae 2026, Sec. "Layout representation"; Kang & Chae 2017):
  seq   : department sequence, permutation of 0..n-1
  cuts  : cut point order, permutation of 0..n-2; cuts[k] = branch b cut at step k
          (branch b lies between seq[b] and seq[b+1])
  codes : cut code for step k, 0 horizontal (left part above), 1 vertical (left part left),
          2 inverse horizontal (left part below), 3 inverse vertical (left part right)
Each cut splits the current sub-rectangle in proportion to the areas of the two parts.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def decode(seq, cuts, codes, areas, W, H):
    n = seq.shape[0]
    # segments: lo, hi (exclusive) in sequence positions, and rectangle
    lo = np.empty(n, np.int64); hi = np.empty(n, np.int64)
    sx = np.empty(n); sy = np.empty(n); sw = np.empty(n); sh = np.empty(n)
    owner = np.zeros(n, np.int64)
    lo[0] = 0; hi[0] = n; sx[0] = 0.0; sy[0] = 0.0; sw[0] = W; sh[0] = H
    nseg = 1
    pref = np.empty(n + 1); pref[0] = 0.0
    for p in range(n):
        pref[p + 1] = pref[p] + areas[seq[p]]
    for k in range(n - 1):
        b = cuts[k]; s = owner[b]
        l0 = lo[s]; h0 = hi[s]
        aL = pref[b + 1] - pref[l0]; aR = pref[h0] - pref[b + 1]
        fL = aL / (aL + aR)
        x = sx[s]; y = sy[s]; w = sw[s]; h = sh[s]
        c = codes[k]
        if c == 0:      # horizontal: left above, right below
            hl = h * fL
            Lx, Ly, Lw, Lh = x, y + h - hl, w, hl
            Rx, Ry, Rw, Rh = x, y, w, h - hl
        elif c == 1:    # vertical: left on the left
            wl = w * fL
            Lx, Ly, Lw, Lh = x, y, wl, h
            Rx, Ry, Rw, Rh = x + wl, y, w - wl, h
        elif c == 2:    # inverse horizontal: left below
            hl = h * fL
            Lx, Ly, Lw, Lh = x, y, w, hl
            Rx, Ry, Rw, Rh = x, y + hl, w, h - hl
        else:           # inverse vertical: left on the right
            wl = w * fL
            Lx, Ly, Lw, Lh = x + w - wl, y, wl, h
            Rx, Ry, Rw, Rh = x, y, w - wl, h
        # left part keeps segment s, right part becomes new segment
        hi[s] = b + 1; sx[s] = Lx; sy[s] = Ly; sw[s] = Lw; sh[s] = Lh
        t = nseg; nseg += 1
        lo[t] = b + 1; hi[t] = h0; sx[t] = Rx; sy[t] = Ry; sw[t] = Rw; sh[t] = Rh
        for p in range(b + 1, h0):
            owner[p] = t
    R = np.empty((n, 4))
    for s in range(nseg):
        d = seq[lo[s]]
        R[d, 0] = sx[s]; R[d, 1] = sy[s]; R[d, 2] = sw[s]; R[d, 3] = sh[s]
    return R


@njit(cache=True)
def evaluate(seq, cuts, codes, areas, W, H, Wt, alpha, pen, D=0.0):
    """Returns (penalised fitness, cost, total aspect excess, number of infeasible departments).
    Wt: symmetric pair weights, cost = sum_{i<j}.
    fitness = cost + pen * excess + D * ninf**3  (static penalty: D = 0; adaptive Tate-Smith penalty: pen = 0)."""
    R = decode(seq, cuts, codes, areas, W, H)
    n = R.shape[0]
    cost = 0.0
    for i in range(n):
        cxi = R[i, 0] + R[i, 2] / 2; cyi = R[i, 1] + R[i, 3] / 2
        for j in range(i + 1, n):
            wij = Wt[i, j]
            if wij != 0.0:
                cost += wij * (abs(cxi - R[j, 0] - R[j, 2] / 2) + abs(cyi - R[j, 1] - R[j, 3] / 2))
    ex = 0.0; ninf = 0
    for i in range(n):
        a = R[i, 2]; b = R[i, 3]
        r = a / b if a > b else b / a
        if alpha[i] > 0 and r > alpha[i]:
            ex += r - alpha[i]; ninf += 1
    return cost + pen * ex + D * ninf ** 3, cost, ex, ninf


@njit(cache=True)
def ils(seq, cuts, codes, fit, areas, W, H, Wt, alpha, pen, max_fail, D=0.0):
    """Iterated local search of the paper: pick one representation at random, apply a swap
    mutation, keep it if better; stop after max_fail consecutive failures."""
    n = seq.shape[0]; fail = 0; nev = 0
    while fail < max_fail:
        which = np.random.randint(3)
        if which == 0:
            i = np.random.randint(n); j = np.random.randint(n)
            seq[i], seq[j] = seq[j], seq[i]
        elif which == 1:
            i = np.random.randint(n - 1); j = np.random.randint(n - 1)
            cuts[i], cuts[j] = cuts[j], cuts[i]
        else:
            i = np.random.randint(n - 1); j = np.random.randint(n - 1)
            codes[i], codes[j] = codes[j], codes[i]
        f, c, e, ni = evaluate(seq, cuts, codes, areas, W, H, Wt, alpha, pen, D); nev += 1
        if f < fit - 1e-12:
            fit = f; fail = 0
        else:
            if which == 0:
                seq[i], seq[j] = seq[j], seq[i]
            elif which == 1:
                cuts[i], cuts[j] = cuts[j], cuts[i]
            else:
                codes[i], codes[j] = codes[j], codes[i]
            fail += 1
    return fit, nev


@njit(cache=True)
def seed(s):
    np.random.seed(s)
