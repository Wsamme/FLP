"""Compare two methods on bench_lean results (feasibility and true path cost from bench_lean.evaluate).

Usage: python -m bench_lean.compare RESULTDIR [A=rows] [B=ref]
Prints a markdown table and writes RESULTDIR/summary_{A}_vs_{B}.json.

Per instance: feasible runs, mean / best path cost of feasible runs, change of the mean (A vs B),
exact two-sided Mann-Whitney U on the feasible runs (with 3 seeds per method the smallest possible p is 0.1).
Pooled: Wilcoxon signed-rank on the 11 per-instance mean costs (A vs B), exact two-sided.
"""
import os, sys, json
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


def load(d, m):
    out = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith(m + "_") and f.endswith(".json")): continue
        r = json.load(open(os.path.join(d, f), encoding="utf-8"))
        out.setdefault(r["instance"], []).append(r)
    return out


def main():
    d = sys.argv[1]; A = sys.argv[2] if len(sys.argv) > 2 else "rows"; B = sys.argv[3] if len(sys.argv) > 3 else "ref"
    ra, rb = load(d, A), load(d, B)
    rows = []; ma, mb = [], []
    print(f"| 算例 | {A} 可行 | {B} 可行 | {A} 平均 | {B} 平均 | 变化 | {A} 最好 | {B} 最好 | MW p |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in sorted(set(ra) | set(rb)):
        fa = [r["eval"]["path_total"] for r in ra.get(name, []) if r["eval"] and r["eval"]["feasible"]]
        fb = [r["eval"]["path_total"] for r in rb.get(name, []) if r["eval"] and r["eval"]["feasible"]]
        na, nb = len(ra.get(name, [])), len(rb.get(name, []))
        p = float(mannwhitneyu(fa, fb, alternative="two-sided", method="exact").pvalue) if fa and fb else None
        row = dict(instance=name, feasible_a=len(fa), runs_a=na, feasible_b=len(fb), runs_b=nb,
                   mean_a=float(np.mean(fa)) if fa else None, mean_b=float(np.mean(fb)) if fb else None,
                   best_a=float(min(fa)) if fa else None, best_b=float(min(fb)) if fb else None, mw_p=p)
        row["change"] = row["mean_a"] / row["mean_b"] - 1 if fa and fb else None
        rows.append(row)
        if fa and fb: ma.append(row["mean_a"]); mb.append(row["mean_b"])
        f = lambda v: "-" if v is None else f"{v:,.1f}"
        chg = "-" if row["change"] is None else f"{100 * row['change']:+.1f}%"
        ps = "-" if p is None else f"{p:.2f}"
        print(f"| {name} | {len(fa)}/{na} | {len(fb)}/{nb} | {f(row['mean_a'])} | {f(row['mean_b'])} | {chg} | "
              f"{f(row['best_a'])} | {f(row['best_b'])} | {ps} |")
    pooled = None
    if len(ma) >= 2:
        w = wilcoxon(ma, mb, alternative="two-sided", method="exact")
        ratio = np.array(ma) / np.array(mb)
        pooled = dict(n=len(ma), wilcoxon_p=float(w.pvalue), wins=int((ratio < 1).sum()),
                      geo_mean_ratio=float(np.exp(np.log(ratio).mean())), min_change=float(ratio.min() - 1), max_change=float(ratio.max() - 1))
        print(f"\n合并（{len(ma)} 个算例的平均成本，Wilcoxon 符号秩，精确双侧）：p = {w.pvalue:.4g}；"
              f"{A} 更低 {pooled['wins']}/{len(ma)}；几何平均变化 {100 * (pooled['geo_mean_ratio'] - 1):+.1f}%"
              f"（{100 * pooled['min_change']:+.1f}% 到 {100 * pooled['max_change']:+.1f}%）")
    json.dump(dict(A=A, B=B, per_instance=rows, pooled=pooled), open(os.path.join(d, f"summary_{A}_vs_{B}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
