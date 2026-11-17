"""Export the final experiment as small JSON tables for the results page.

Usage: python -m bench_lean.export_dash RESULTDIR OUTDIR
Reads the run files in RESULTDIR and RESULTDIR/report/final_summary.json (written by final_report.py) and writes
OUTDIR/{runs,comparisons,instances,violations,layouts}.json, each {"rows": [...], "meta": [...]}.
Costs are evaluate's path_total stored in each run; nothing is recomputed except the I/O points of the drawn layouts.
Percent changes are rounded to 0.1, p-values to 2 significant digits, mean costs to whole numbers.
"""
import os, sys, json
from .evaluate import load as load_inst, evaluate
from .final_report import read, feas_cost

METHODS = ["rowsF", "reft", "legal", "classic", "rowsF_300", "reft_300", "rowsE_post"]
COMPARISONS = [  # (key in final_summary.json, short id, label)
    ("rowsF_vs_reft", "f60", "rows 对参考解法，60 秒"),
    ("rowsF_300_vs_reft_300", "f300", "rows 对参考解法，300 秒"),
    ("rowsF_vs_legal", "legal", "rows 对“经典解 + 合法化”，60 秒"),
    ("clear_rowsF_vs_reft", "clear", "通道净宽 ≥ 1.4 m 重算，rows 对参考解法"),
    ("rowsF_300_vs_rowsF", "time", "rows 300 秒对 60 秒"),
    ("rowsE_post_vs_reft", "post_ref", "事后版本 rowsE 对参考解法（看过测试结果后才跑）"),
    ("rowsE_post_vs_rowsF", "post_f", "事后版本 rowsE 对 rows"),
]


def main():
    d, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    me = os.path.abspath(__file__)
    code = open(me, encoding="utf-8").read()
    meta = [{"label": "export_dash.py", "value": code, "type": "code", "lang": "python"},
            {"label": "Input", "value": "bench_lean/results_final/*.json, bench_lean/results_final/report/final_summary.json"}]
    runs = read(d)
    summ = json.load(open(os.path.join(d, "report", "final_summary.json"), encoding="utf-8"))

    rows = []
    for m in METHODS:
        for n, seeds in sorted(runs.get(m, {}).items()):
            for s, r in sorted(seeds.items()):
                c = feas_cost(r)
                rows.append(dict(method=m, instance=n, size=n.split("-")[1], seed=int(s), budget=int(r["budget"]),
                                 feasible=1 if c is not None else 0, cost=round(c, 1) if c is not None else None,
                                 wall=round(r["wall"], 2)))
    dump(out, "runs", rows, meta)

    comp, inst = [], []
    for key, cid, label in COMPARISONS:
        c = summ[key]; p = c["pooled"]
        comp.append(dict(id=cid, label=label, a=c["A"], b=c["B"], instances=p["n_instances"], wins_instances=p["wins_instances"],
                         change_pct=pct(p["geo_change"]), change_min_pct=pct(p["min_change"]), change_max_pct=pct(p["max_change"]),
                         p_instances=sig(p["wilcoxon_instances"]), pairs=p["n_pairs"], wins_pairs=p["wins_pairs"],
                         change_pairs_pct=pct(p["geo_change_pairs"]), p_pairs=sig(p["wilcoxon_pairs"])))
        for r in c["rows"]:
            inst.append(dict(comparison=cid, instance=r["instance"], change_pct=pct(r["change"]), mean_a=rnd(r["mean_a"]),
                             mean_b=rnd(r["mean_b"]), best_a=rnd(r["best_a"]), best_b=rnd(r["best_b"]), feas_a=r["feas_a"],
                             n_a=r["n_a"], feas_b=r["feas_b"], n_b=r["n_b"], p_holm=sig(r["mw_p_holm"])))
    dump(out, "comparisons", comp, meta)
    dump(out, "instances", inst, meta)
    dump(out, "violations", [dict(check=k, runs=v, of=220) for k, v in
                             sorted(summ["classic_violations"].items(), key=lambda kv: -kv[1])], meta)

    lay = []
    for n in sorted(runs["rowsF"]):
        I = load_inst(n); B = I["building"]
        for m in ("rowsF", "reft"):
            rs = [r for r in runs[m].get(n, {}).values() if feas_cost(r) is not None]
            if not rs: continue
            r = min(rs, key=feas_cost); ev = evaluate(I, r["rects"])
            base = dict(instance=n, method=m, seed=int(r["seed"]), cost=round(ev["path_total"], 1))
            lay.append(dict(base, kind="building", name="", type="", x=0, y=0, w=B["W"], h=B["H"]))
            for a in I["aisles"]:
                x, y, w, h = a["rect"]; lay.append(dict(base, kind="aisle", name=a["id"], type=a["type"], x=x, y=y, w=w, h=h))
            for f in I["fixed"]:
                x, y, w, h = f["rect"]; lay.append(dict(base, kind="fixed", name=f.get("name", ""), type="", x=x, y=y, w=w, h=h))
            for i, (x, y, w, h) in enumerate(r["rects"]):
                dd = I["departments"][i]
                lay.append(dict(base, kind="dept", name=dd["name"], type=dd["type"], x=round(x, 2), y=round(y, 2), w=round(w, 2), h=round(h, 2)))
            for e in I["exits"]:
                lay.append(dict(base, kind="exit", name="", type="", x=e["x"], y=e["y"], w=0, h=0))
            for x, y, *_ in ev.get("io", []):
                lay.append(dict(base, kind="io", name="", type="", x=round(x, 2), y=round(y, 2), w=0, h=0))
    dump(out, "layouts", lay, meta)


def pct(v):  # fraction -> percent, 1 decimal (as printed on the page)
    return None if v is None else round(100 * v, 1)


def sig(v):  # p-value to 2 significant digits
    return None if v is None else float(f"{v:.2g}")


def rnd(v):
    return None if v is None else round(v)


def dump(out, name, rows, meta):
    p = os.path.join(out, name + ".json")
    json.dump({"rows": rows, "meta": meta}, open(p, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(name, len(rows), os.path.getsize(p))


if __name__ == "__main__":
    main()
