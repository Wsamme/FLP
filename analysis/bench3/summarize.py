"""Experiment 1b summary: re-implemented literature methods vs ours, same machine, same protocol.
Reads results/bench3/*.json (bench3/run.py records; cost = independent checker flp.evaluate).
Writes analysis/bench3/summary.json and prints tables.

Per method x instance: feasible runs, best, mean, median, worst, std, mean wall.
Ours vs each method: two-sided Mann-Whitney U on the feasible costs, Holm correction over the methods
compared on that instance (same rule as Experiment 1: a tie = all costs equal counts as p = 1 and is included;
fewer than two feasible runs = n/a, not included), effect size A12 = P(ours < other) + 0.5 P(equal);
A12 > 0.5 means ours is better.
    python analysis/bench3/summarize.py [--dir results/bench3]
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import mannwhitneyu

INST = ["O7", "vC10a", "AB20", "SC30", "Du62"]
REF = {"O7": 131.6531381, "vC10a": 18522.79, "AB20": 5021.17, "SC30": 3343.47, "Du62": 3635307.30}
LABEL = {"ours": "Ours", "lsa": "LSA (Lin 2024)", "av16": "Two-stage (Anjos & Vieira 2016)",
         "hs": "HS (Kang & Chae 2017)", "cro": "CRO (Jeong & Chae 2026)"}


def load(d):
    R = defaultdict(list)
    for fn in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(fn))
        R[(r["method"], r["instance"])].append(r)
    return R


def holm(raw):
    order = sorted(raw, key=lambda i: raw[i]); m = len(raw); adj = {}; run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (m - r) * raw[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results/bench3")
    a = ap.parse_args()
    R = load(a.dir)
    methods = [m for m in ["ours", "lsa", "av16", "hs", "cro"] if any((m, k) in R for k in INST)]
    S = {}
    for m in methods:
        for k in INST:
            rs = R.get((m, k), [])
            c = np.array(sorted(r["cost"] for r in rs if r["feasible"]))
            S[(m, k)] = dict(runs=len(rs), feas=int(len(c)), costs=c.tolist(),
                             best=float(c.min()) if len(c) else None, mean=float(c.mean()) if len(c) else None,
                             median=float(np.median(c)) if len(c) else None, worst=float(c.max()) if len(c) else None,
                             std=float(c.std(ddof=1)) if len(c) > 1 else None,
                             wall=float(np.mean([r["wall"] for r in rs])) if rs else None)
    stats = {}
    for k in INST:
        a_ = np.array(S[("ours", k)]["costs"]) if ("ours", k) in S else np.array([])
        ps = {}
        for m in methods:
            if m == "ours":
                continue
            b = np.array(S[(m, k)]["costs"])
            if len(a_) < 2 or len(b) < 2:
                ps[m] = ("n/a", None, None); continue
            if np.allclose(np.r_[a_, b], a_[0], rtol=1e-6):
                ps[m] = ("tie", 1.0, 0.5); continue
            p = mannwhitneyu(a_, b, alternative="two-sided").pvalue
            A12 = float(np.mean([(x < y) + 0.5 * (x == y) for x in a_ for y in b]))
            ps[m] = ("p", float(p), A12)
        adj = holm({m: v[1] for m, v in ps.items() if v[0] != "n/a"})
        stats[k] = {m: dict(kind=v[0], p=v[1], p_holm=adj.get(m), A12=v[2]) for m, v in ps.items()}
    out = dict(summary={f"{m}|{k}": v for (m, k), v in S.items()}, stats=stats, ref=REF, methods=methods)
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary.json"), "w"), indent=1)
    for k in INST:
        print(f"\n{k}  (reference {REF[k]})")
        print(f"{'method':34s} {'feas':>5s} {'best':>14s} {'mean':>14s} {'gap_best%':>9s} {'gap_mean%':>9s}  {'Holm p':>8s} {'A12':>5s}")
        for m in methods:
            s = S[(m, k)]
            st = stats[k].get(m, {})
            gb = 100 * (s["best"] - REF[k]) / REF[k] if s["best"] is not None else float("nan")
            gm = 100 * (s["mean"] - REF[k]) / REF[k] if s["mean"] is not None else float("nan")
            ph = "" if not st else (st["kind"] if st["kind"] != "p" else f"{st['p_holm']:.2g}")
            a12 = "" if not st or st["A12"] is None else f"{st['A12']:.2f}"
            fmt = (lambda v: f"{v:14.2f}" if v is not None else f"{'-':>14s}")
            print(f"{LABEL.get(m, m):34s} {s['feas']:>2d}/{s['runs']:<2d} {fmt(s['best'])} {fmt(s['mean'])} {gb:9.3f} {gm:9.3f}  {ph:>8s} {a12:>5s}")


if __name__ == "__main__":
    main()
