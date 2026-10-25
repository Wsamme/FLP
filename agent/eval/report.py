"""Evaluate all runs against gold, write summary tables and black/white figures."""
import csv
import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.gold import load_gold, GRADES  # noqa: E402
from eval.eval_relations import evaluate  # noqa: E402

METRICS = ["exact", "within1", "kappa_linear", "kappa_unweighted", "nonU_recall", "nonU_precision", "X_recall"]
LABEL = {"exact": "完全一致率", "within1": "一级以内一致率", "kappa_linear": "κ 线性加权",
         "kappa_unweighted": "κ 不加权", "nonU_recall": "非U召回", "nonU_precision": "非U精确",
         "X_recall": "X召回"}
MODE_LABEL = {"allU": "全U基线", "oneshot": "一次性提示", "sc_off": "逐对+理由，自检关", "sc_on": "逐对+理由，自检开"}
ERR_TYPES = ["等级偏高", "等级偏低", "漏判X", "误判X", "有关系判成U", "U判成有关系"]

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "PingFang SC", "Microsoft YaHei",
                                   "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def fmt(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}"


def confusion_fig(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=200)
    mx = max(max(r) for r in cm) or 1
    ax.imshow([[v / mx for v in r] for r in cm], cmap="Greys", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            v = cm[i][j]
            ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=9,
                    color="white" if v / mx > 0.55 else "black")
    ax.set_xticks(range(6), list(GRADES))
    ax.set_yticks(range(6), list(GRADES))
    ax.set_xlabel("Agent 预测")
    ax.set_ylabel("书中标准 (gold)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def report(runs, cases, res_dir, cfg, root):
    os.makedirs(res_dir, exist_ok=True)
    rows, per_setting, fails = [], {}, {}
    for c in cases:
        g = load_gold(c)
        allu = {k: "U" for k in g["pairs"]}
        m = evaluate(g["pairs"], allu, g["exclude"])
        per_setting.setdefault((c, "allU"), []).append(m)
        rows.append({"case": c, "mode": "allU", "repeat": 0, **{k: m[k] for k in METRICS}})
    for meta, fin in runs:
        c, mode = meta["case"], meta["mode"]
        if c not in cases:
            continue
        if fin["failed"] or fin["pairs"] is None:
            fails.setdefault((c, mode), 0)
            fails[(c, mode)] += 1
            continue
        g = load_gold(c)
        m = evaluate(g["pairs"], fin["pairs"], g["exclude"])
        m["meta"] = meta
        per_setting.setdefault((c, mode), []).append(m)
        rows.append({"case": c, "mode": mode, "repeat": meta["repeat"], "run_id": meta["run_id"],
                     **{k: m[k] for k in METRICS}, "tokens_in": meta["input_tokens"],
                     "tokens_out": meta["output_tokens"], "seconds": meta["seconds"], "n_calls": meta["n_calls"]})

    with open(os.path.join(res_dir, "metrics_runs.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("case", "mode", "repeat"), k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    md = [f"# 关系推断结果汇总\n\n模型 `{cfg['llm']['model']}`，provider `{cfg['llm']['provider']}`，"
          f"temperature `{cfg['llm']['temperature']}`（null = 模型默认）。均值 ± 标准差（n = 成功运行次数）。\n"]
    order = ["allU", "oneshot", "sc_off", "sc_on"]
    for c in cases:
        g = load_gold(c)
        n_eval = len(g["pairs"]) - len(g["exclude"])
        md.append(f"\n## {c}（评估 {n_eval} 对" + ("，排除 " + ", ".join(f"{i}-{j}" for i, j in sorted(g["exclude"]))
                                            if g["exclude"] else "") + "）\n")
        md.append("| 设置 | n | 失败 | " + " | ".join(LABEL[k] for k in METRICS) + " |")
        md.append("|---|---|---|" + "---|" * len(METRICS))
        for mode in order:
            ms = per_setting.get((c, mode), [])
            if not ms and not fails.get((c, mode)):
                continue
            md.append(f"| {MODE_LABEL[mode]} | {len(ms)} | {fails.get((c, mode), 0)} | "
                      + " | ".join(fmt([x[k] for x in ms]) for k in METRICS) + " |")
        # error counts
        md.append("\n错误分类（每次运行的平均个数）\n")
        md.append("| 设置 | " + " | ".join(ERR_TYPES) + " |")
        md.append("|---|" + "---|" * len(ERR_TYPES))
        for mode in order[1:]:
            ms = per_setting.get((c, mode), [])
            if ms:
                md.append(f"| {MODE_LABEL[mode]} | " + " | ".join(
                    f"{st.mean([x['error_counts'].get(t, 0) for x in ms]):.1f}" for t in ERR_TYPES) + " |")
        # confusion matrices summed over repeats + examples from the first repeat
        for mode in order[1:]:
            ms = per_setting.get((c, mode), [])
            if not ms:
                continue
            cm = [[sum(x["confusion"][i][j] for x in ms) for j in range(6)] for i in range(6)]
            fn = f"confusion_{c}_{mode}.png"
            confusion_fig(cm, f"{c} · {MODE_LABEL[mode]}（{len(ms)} 次合计）", os.path.join(res_dir, fn))
            md.append(f"\n![{fn}]({fn})\n")
        for mode in ("sc_off", "sc_on"):
            ms = per_setting.get((c, mode), [])
            if not ms:
                continue
            md.append(f"\n错误示例（{MODE_LABEL[mode]}，第 1 次运行，每类最多 3 条）\n")
            md.append("| 类别 | 单位对 | gold | 预测 | 模型理由 |")
            md.append("|---|---|---|---|---|")
            names = g["names"]
            for t in ERR_TYPES:
                for e in ms[0]["errors"].get(t, [])[:3]:
                    i, j = map(int, e["pair"].split("-"))
                    md.append(f"| {t} | {i} {names[i-1]} – {j} {names[j-1]} | {e['gold']} | {e['pred']} | "
                              f"{(e['reason'] or '').replace('|', '/')} |")
    with open(os.path.join(res_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    json.dump({f"{c}|{m}": [{k: v for k, v in x.items() if k != "meta"} for x in ms]
               for (c, m), ms in per_setting.items()},
              open(os.path.join(res_dir, "metrics_full.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(md))
