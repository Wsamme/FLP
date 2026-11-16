"""A6: how well the rows search cost (SOCP objective on the aisle network) tracks the true path cost.

Runs rowsolver.solve, keeps every k-th accepted feasible layout (up to N), then scores each with
bench_lean.evaluate. Reports Spearman rank correlation and the ratio true / proxy per instance.
Usage: python -m bench_lean.diag_proxy OUT.json SECONDS SEED NAME [NAME ...]
"""
import sys, json, time
import numpy as np
from scipy.stats import spearmanr
from .evaluate import load, evaluate
from .rowsolver import solve

N = 300


def run(name, T, seed):
    inst = load(name); keep = []
    def tr(proxy, rects, ok, el):
        if ok: keep.append((proxy, rects, el))
    solve(inst, seed, T, trace=tr)
    step = max(1, len(keep) // N)
    sample = keep[::step][:N]
    px, tc = [], []
    for proxy, rects, el in sample:
        ev = evaluate(inst, rects)
        if ev["feasible"]: px.append(proxy); tc.append(ev["path_total"])
    px, tc = np.array(px), np.array(tc)
    rho = float(spearmanr(px, tc).statistic) if len(px) > 2 else None
    top = np.argsort(px)[: max(3, len(px) // 10)]                 # best 10 % by proxy
    rho_top = float(spearmanr(px[top], tc[top]).statistic) if len(top) > 2 else None
    return dict(instance=name, accepted_feasible=len(keep), sampled=len(px), spearman=rho, spearman_best10=rho_top,
                ratio_mean=float((tc / px).mean()), ratio_sd=float((tc / px).std()),
                best_proxy_true=float(tc[np.argmin(px)]), best_true=float(tc.min()))


if __name__ == "__main__":
    out, T, seed, names = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    res = []
    for n in names:
        r = run(n, T, seed); res.append(r); print(json.dumps(r), flush=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
