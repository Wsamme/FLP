"""Compare two configurations on bench_lean results (feasibility and true path cost from bench_lean.evaluate).

Usage: python -m bench_lean.compare A B DIR [DIR ...] [--clear] [--out FILE]
  A, B : configuration tags (the "method" field of the result files), e.g. rows reft
  DIR  : one or more result directories (results are matched by instance and seed)
  --clear : also compare the path cost with a 1.4 m minimum walkway (bench_lean.clearance)
Prints a markdown table; writes a JSON summary (default DIR/summary_{A}_vs_{B}.json).

Per instance: feasible runs, mean / best path cost of feasible runs, change of the mean (A vs B), exact two-sided
Mann-Whitney U. Pooled: Wilcoxon signed-rank on the per-instance mean costs, and on all seed-paired runs
(pairs where both are feasible).
"""
import os, json, argparse
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


def load(dirs, m):
    out = {}
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json") or f.startswith("summary"): continue
            try:
                r = json.load(open(os.path.join(d, f), encoding="utf-8"))
            except ValueError:
                print(f"(skipped unreadable {f})"); continue
            if r.get("method") == m: out.setdefault(r["instance"], {})[r["seed"]] = r
    return out


def cost(r, clear):
    if not r["eval"] or not r["eval"]["feasible"]: return None
    if not clear: return r["eval"]["path_total"]
    from .evaluate import load as li
    from .clearance import clear_cost
    c = clear_cost(li(r["instance"]), r["rects"])
    return c["path_total"] if c["unreachable_pairs"] == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A"); ap.add_argument("B"); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--clear", action="store_true"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    ra, rb = load(a.dirs, a.A), load(a.dirs, a.B)
    rows = []; ma, mb = [], []; pa, pb = [], []
    unit = "（1.4 m 净宽路径成本）" if a.clear else "（真实路径成本）"
    print(f"{a.A} vs {a.B} {unit}\n")
    print(f"| 算例 | {a.A} 可行 | {a.B} 可行 | {a.A} 平均 | {a.B} 平均 | 变化 | {a.A} 最好 | {a.B} 最好 | MW p |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in sorted(set(ra) | set(rb)):
        A_, B_ = ra.get(name, {}), rb.get(name, {})
        ca = {s: cost(r, a.clear) for s, r in A_.items()}; cb = {s: cost(r, a.clear) for s, r in B_.items()}
        fa = [v for v in ca.values() if v is not None]; fb = [v for v in cb.values() if v is not None]
        for s in sorted(set(ca) & set(cb)):
            if ca[s] is not None and cb[s] is not None: pa.append(ca[s]); pb.append(cb[s])
        p = float(mannwhitneyu(fa, fb, alternative="two-sided", method="exact").pvalue) if fa and fb else None
        row = dict(instance=name, feasible_a=len(fa), runs_a=len(A_), feasible_b=len(fb), runs_b=len(B_),
                   mean_a=float(np.mean(fa)) if fa else None, mean_b=float(np.mean(fb)) if fb else None,
                   best_a=float(min(fa)) if fa else None, best_b=float(min(fb)) if fb else None, mw_p=p,
                   wall_max_a=max((r["wall"] for r in A_.values()), default=None),
                   wall_max_b=max((r["wall"] for r in B_.values()), default=None))
        row["change"] = row["mean_a"] / row["mean_b"] - 1 if fa and fb else None
        rows.append(row)
        if fa and fb: ma.append(row["mean_a"]); mb.append(row["mean_b"])
        f = lambda v: "-" if v is None else f"{v:,.1f}"
        chg = "-" if row["change"] is None else f"{100 * row['change']:+.1f}%"
        ps = "-" if p is None else f"{p:.3g}"
        print(f"| {name} | {len(fa)}/{len(A_)} | {len(fb)}/{len(B_)} | {f(row['mean_a'])} | {f(row['mean_b'])} | {chg} | "
              f"{f(row['best_a'])} | {f(row['best_b'])} | {ps} |")
    pooled = None
    if len(ma) >= 2:
        w = wilcoxon(ma, mb, alternative="two-sided")
        ratio = np.array(ma) / np.array(mb)
        pr = np.array(pa) / np.array(pb)
        wp = wilcoxon(pa, pb, alternative="two-sided") if len(pa) >= 2 else None
        pooled = dict(n_instances=len(ma), wilcoxon_p_instances=float(w.pvalue), wins_instances=int((ratio < 1).sum()),
                      geo_mean_ratio=float(np.exp(np.log(ratio).mean())), min_change=float(ratio.min() - 1),
                      max_change=float(ratio.max() - 1), n_pairs=len(pa), wins_pairs=int((pr < 1).sum()),
                      wilcoxon_p_pairs=float(wp.pvalue) if wp else None, geo_mean_ratio_pairs=float(np.exp(np.log(pr).mean())))
        print(f"\n按算例平均（{len(ma)} 个，Wilcoxon 符号秩）：p = {w.pvalue:.3g}；{a.A} 更低 {pooled['wins_instances']}/{len(ma)}；"
              f"几何平均变化 {100 * (pooled['geo_mean_ratio'] - 1):+.2f}%（{100 * pooled['min_change']:+.1f}% 到 {100 * pooled['max_change']:+.1f}%）")
        if wp:
            print(f"按种子配对（{len(pa)} 对）：p = {wp.pvalue:.3g}；{a.A} 更低 {pooled['wins_pairs']}/{len(pa)}；"
                  f"几何平均变化 {100 * (pooled['geo_mean_ratio_pairs'] - 1):+.2f}%")
    walls_a = [r["wall_max_a"] for r in rows if r["wall_max_a"]]; walls_b = [r["wall_max_b"] for r in rows if r["wall_max_b"]]
    if walls_a and walls_b: print(f"最长墙钟：{a.A} {max(walls_a):.1f} s，{a.B} {max(walls_b):.1f} s")
    out = a.out or os.path.join(a.dirs[0], f"summary_{a.A}_vs_{a.B}{'_clear' if a.clear else ''}.json")
    json.dump(dict(A=a.A, B=a.B, dirs=a.dirs, clear=a.clear, per_instance=rows, pooled=pooled),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
