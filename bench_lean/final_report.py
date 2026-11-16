"""Summary of the final bench_lean experiment (test set): tables as markdown + JSON, layout drawings as SVG.

Usage: python -m bench_lean.final_report RESULTDIR OUTDIR
Reads the runs written by phase_final.sh (tags rowsC, reft, legal, classic, rowsC_300, reft_300) and writes
OUTDIR/final_summary.md, OUTDIR/final_summary.json and OUTDIR/layout_{instance}.svg (best rowsC and best reft).
Every number comes from the result files: costs are bench_lean.evaluate's path_total stored in each run;
the 1.4 m walkway cost is recomputed with bench_lean.clearance.
"""
import os, sys, json
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
from .evaluate import load as load_inst
from .clearance import clear_cost

TAGS = ["rowsF", "reft", "legal", "classic", "rowsF_300", "reft_300", "rowsE_post", "rowsC", "rowsC2"]


def read(d):
    runs = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json") or f.startswith("summary") or f.startswith("final"): continue
        try:
            r = json.load(open(os.path.join(d, f), encoding="utf-8"))
        except ValueError:
            continue
        runs.setdefault(r["method"], {}).setdefault(r["instance"], {})[r["seed"]] = r
    return runs


def feas_cost(r):
    return r["eval"]["path_total"] if r.get("eval") and r["eval"]["feasible"] else None


def compare(runs, A, B, cost=feas_cost):
    rows, ma, mb, pa, pb = [], [], [], [], []
    for n in sorted(set(runs.get(A, {})) | set(runs.get(B, {}))):
        ra, rb = runs.get(A, {}).get(n, {}), runs.get(B, {}).get(n, {})
        ca = {s: cost(r) for s, r in ra.items()}; cb = {s: cost(r) for s, r in rb.items()}
        fa = [v for v in ca.values() if v is not None]; fb = [v for v in cb.values() if v is not None]
        for s in set(ca) & set(cb):
            if ca[s] is not None and cb[s] is not None: pa.append(ca[s]); pb.append(cb[s])
        p = float(mannwhitneyu(fa, fb, alternative="two-sided").pvalue) if len(fa) > 1 and len(fb) > 1 else None
        row = dict(instance=n, n_a=len(ra), feas_a=len(fa), n_b=len(rb), feas_b=len(fb),
                   mean_a=float(np.mean(fa)) if fa else None, sd_a=float(np.std(fa, ddof=1)) if len(fa) > 1 else None,
                   mean_b=float(np.mean(fb)) if fb else None, sd_b=float(np.std(fb, ddof=1)) if len(fb) > 1 else None,
                   best_a=min(fa) if fa else None, best_b=min(fb) if fb else None, mw_p=p,
                   wall_a=max((r["wall"] for r in ra.values()), default=None),
                   wall_b=max((r["wall"] for r in rb.values()), default=None))
        row["change"] = row["mean_a"] / row["mean_b"] - 1 if fa and fb else None
        rows.append(row)
        if fa and fb: ma.append(row["mean_a"]); mb.append(row["mean_b"])
    k = len(rows)
    ps = [r["mw_p"] for r in rows]
    order = sorted(range(k), key=lambda i: (ps[i] is None, ps[i] or 1))
    adj = [None] * k; run = 0.0                                   # Holm over the instances
    for rank, i in enumerate(order):
        if ps[i] is None: continue
        run = max(run, min(1.0, (k - rank) * ps[i])); adj[i] = run
    for r, a in zip(rows, adj): r["mw_p_holm"] = a
    pooled = None
    if len(ma) > 1:
        ratio = np.array(ma) / np.array(mb); pr = np.array(pa) / np.array(pb)
        pooled = dict(n_instances=len(ma), wilcoxon_instances=float(wilcoxon(ma, mb).pvalue), wins_instances=int((ratio < 1).sum()),
                      geo_change=float(np.exp(np.log(ratio).mean()) - 1), min_change=float(ratio.min() - 1), max_change=float(ratio.max() - 1),
                      n_pairs=len(pa), wins_pairs=int((pr < 1).sum()), wilcoxon_pairs=float(wilcoxon(pa, pb).pvalue),
                      geo_change_pairs=float(np.exp(np.log(pr).mean()) - 1))
    return dict(A=A, B=B, rows=rows, pooled=pooled)


def md_table(c, la, lb):
    f = lambda v: "-" if v is None else f"{v:,.0f}"
    out = [f"| 算例 | {la} 可行 | {lb} 可行 | {la} 平均 ± 标准差 | {lb} 平均 ± 标准差 | 变化 | {la} 最好 | {lb} 最好 | Holm 校正 p |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in c["rows"]:
        sa = f"{f(r['mean_a'])} ± {f(r['sd_a'])}" if r["mean_a"] is not None else "-"
        sb = f"{f(r['mean_b'])} ± {f(r['sd_b'])}" if r["mean_b"] is not None else "-"
        ch = "-" if r["change"] is None else f"{100 * r['change']:+.1f} %"
        p = "-" if r["mw_p_holm"] is None else f"{r['mw_p_holm']:.2g}"
        out.append(f"| {r['instance']} | {r['feas_a']}/{r['n_a']} | {r['feas_b']}/{r['n_b']} | {sa} | {sb} | {ch} | {f(r['best_a'])} | {f(r['best_b'])} | {p} |")
    p = c["pooled"]
    if p:
        out.append("")
        out.append(f"- 按算例平均（{p['n_instances']} 个）：{la} 更低 {p['wins_instances']}/{p['n_instances']}，几何平均变化 {100 * p['geo_change']:+.1f} %"
                   f"（{100 * p['min_change']:+.1f} % 到 {100 * p['max_change']:+.1f} %），Wilcoxon p = {p['wilcoxon_instances']:.2g}")
        out.append(f"- 按种子配对（{p['n_pairs']} 对，两边都可行）：{la} 更低 {p['wins_pairs']}/{p['n_pairs']}，几何平均变化 {100 * p['geo_change_pairs']:+.1f} %，"
                   f"Wilcoxon p = {p['wilcoxon_pairs']:.2g}")
    walls_a = [r["wall_a"] for r in c["rows"] if r["wall_a"]]; walls_b = [r["wall_b"] for r in c["rows"] if r["wall_b"]]
    if walls_a and walls_b: out.append(f"- 最长墙钟：{la} {max(walls_a):.1f} 秒，{lb} {max(walls_b):.1f} 秒")
    return "\n".join(out)


def svg(inst, rects, title, ev, k=6.0):
    B = inst["building"]; W, H = B["W"], B["H"]
    Y = lambda y, h=0: (H - y - h) * k
    col = {"dock": "#f6a", "store": "#8cf", "cell": "#fc6", "monument": "#f96", "line": "#9d9", "area": "#cce", "support": "#ddb"}
    el = [f'<g><text x="0" y="-8" font-size="12">{title}</text>',
          f'<rect x="0" y="0" width="{W * k}" height="{H * k}" fill="none" stroke="black" stroke-width="1.5"/>']
    for a in inst["aisles"]:
        x, y, w, h = a["rect"]; el.append(f'<rect x="{x * k}" y="{Y(y, h)}" width="{w * k}" height="{h * k}" fill="#e3e3e3"/>')
    for f in inst["fixed"]:
        x, y, w, h = f["rect"]; el.append(f'<rect x="{x * k}" y="{Y(y, h)}" width="{w * k}" height="{h * k}" fill="#999" opacity=".5"/>')
    for i, (x, y, w, h) in enumerate(rects):
        d = inst["departments"][i]
        el.append(f'<rect x="{x * k}" y="{Y(y, h)}" width="{w * k}" height="{h * k}" fill="{col.get(d["type"], "#eee")}" stroke="#333" stroke-width=".6"/>')
        el.append(f'<text x="{(x + w / 2) * k}" y="{Y(y + h / 2) + 3}" font-size="7" text-anchor="middle">{d["name"]}</text>')
    for x, y, _ in ev.get("io", []): el.append(f'<circle cx="{x * k}" cy="{Y(y)}" r="2"/>')
    for e in inst["exits"]: el.append(f'<circle cx="{e["x"] * k}" cy="{Y(e["y"])}" r="3.5" fill="#2a2"/>')
    el.append("</g>")
    return "\n".join(el), W * k, H * k


def main():
    d, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    runs = read(d)
    res = {}
    md = ["# bench_lean 最终实验（测试集）", "",
          f"结果目录：`{os.path.relpath(d)}`。成本是 `evaluate` 的真实路径成本（物流 + 人流）。只统计可行的运行；不可行或没有返回布局的运行计入“可行”一列的分母。", ""]
    md.append("## 可行率")
    md.append("")
    md.append("| 方法 | 预算 | 可行 / 运行 | 没有返回布局 | 最长墙钟 |")
    md.append("|---|---|---|---|---|")
    feas = {}
    for t in TAGS:
        allr = [r for n in runs.get(t, {}).values() for r in n.values()]
        if not allr: continue
        ok = sum(1 for r in allr if feas_cost(r) is not None); none = sum(1 for r in allr if not r.get("eval"))
        feas[t] = dict(runs=len(allr), feasible=ok, no_layout=none, wall_max=max(r["wall"] for r in allr), budget=allr[0]["budget"])
        md.append(f"| {t} | {allr[0]['budget']:.0f} 秒 | {ok}/{len(allr)} | {none} | {feas[t]['wall_max']:.1f} 秒 |")
    res["feasibility"] = feas
    if "classic" in runs:
        from collections import Counter
        cnt = Counter()
        for n in runs["classic"].values():
            for r in n.values():
                if r.get("eval"):
                    for kk, v in r["eval"]["checks"].items():
                        if v: cnt[kk] += 1
        md += ["", "经典求解器（忽略通道和规则）各类违规出现的运行次数：" + "，".join(f"{k} {v}" for k, v in cnt.most_common())]
        res["classic_violations"] = dict(cnt)
    for A, B, title in [("rowsF", "reft", "60 秒：rows（最终配置）与参考解法"), ("rowsF_300", "reft_300", "300 秒：rows 与参考解法"),
                        ("rowsF", "legal", "60 秒：rows 与“经典解 + 合法化”"), ("rowsF_300", "rowsF", "rows：300 秒与 60 秒"),
                        ("rowsE_post", "reft", "事后稳健性版本 rowsE（看过 rowsF 的测试集失败后才选，单独报告）与参考解法，60 秒"),
                        ("rowsE_post", "rowsF", "事后稳健性版本 rowsE 与最终配置 rowsF，60 秒")]:
        if A in runs and B in runs:
            c = compare(runs, A, B); res[f"{A}_vs_{B}"] = c
            md += ["", f"## {title}", "", md_table(c, A, B)]
    # robustness: 1.4 m walkway
    if "rowsF" in runs and "reft" in runs:
        cache = {}
        def ccost(r):
            if feas_cost(r) is None: return None
            key = (r["method"], r["instance"], r["seed"])
            if key not in cache:
                c = clear_cost(load_inst(r["instance"]), r["rects"])
                cache[key] = c["path_total"] if c["unreachable_pairs"] == 0 else None
            return cache[key]
        c = compare(runs, "rowsF", "reft", cost=ccost); res["clear_rowsF_vs_reft"] = c
        md += ["", "## 稳健性：通道净宽至少 1.4 m 时的路径成本（60 秒）", "",
               "`evaluate` 允许路径沿两个紧挨部门之间的零宽缝走；这里用 `clearance.py` 重算。", "", md_table(c, "rowsF", "reft")]
    # drawings
    if "rowsF" in runs and "reft" in runs:
        for n in sorted(runs["rowsF"]):
            inst = load_inst(n); parts = []; x0 = 0; hmax = 0
            for t in ("rowsF", "reft"):
                rs = [r for r in runs[t].get(n, {}).values() if feas_cost(r) is not None]
                if not rs: continue
                r = min(rs, key=feas_cost)
                from .evaluate import evaluate
                ev = evaluate(inst, r["rects"])
                g, w, h = svg(inst, r["rects"], f"{t} 种子 {r['seed']}：真实路径成本 {ev['path_total']:,.0f}", ev)
                parts.append(f'<g transform="translate({x0 + 10},{30})">{g}</g>'); x0 += w + 30; hmax = max(hmax, h)
            open(os.path.join(out, f"layout_{n}.svg"), "w", encoding="utf-8").write(
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{x0 + 10}" height="{hmax + 50}" font-family="sans-serif">'
                f'<rect width="100%" height="100%" fill="white"/>{"".join(parts)}</svg>')
    json.dump(res, open(os.path.join(out, "final_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(out, "final_summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
