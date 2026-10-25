"""Write the LaTeX tables (paper/tables/*.tex) and paper/numbers.tex (\\newcommand macros used in the text).

Usage:  python3 paper/make_tables.py
Every number in the experiments section comes from this script. Reads results/bench/*.json,
results/bench_polish/*.json, results/records/best_*.json, instances/uaflp/*.json. Runs no optimisation
(only the independent evaluator flp.evaluate on the five best layouts).
"""
import os, sys, json, re
import numpy as np
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402
import data as D   # noqa: E402

sys.path.insert(0, D.ROOT)
from flp.instance import load       # noqa: E402
from flp.evaluate import evaluate, violations   # noqa: E402

TAB = os.path.join(HERE, "tables")
os.makedirs(TAB, exist_ok=True)
RS = D.runs()

INAME = {"O7": "Oseven", "vC10a": "vCtena", "AB20": "ABtwenty", "SC30": "SCthirty", "Du62": "Dusixtytwo"}
MNAME = {"construct": "Construct", "swap": "Swap", "ga": "GA", "sa": "SA", "cro": "CRO", "mip": "MIP",
         "cpsat": "CPSAT", "ours": "Ours", "ours_s1": "OursSone", "ours_s2": "OursStwo",
         "ours_nos1": "OursNoSone", "ours_noelite": "OursNoElite", "ours_unif": "OursUnif",
         "minlp": "MINLP", "sa+polish": "SAPolish", "cro+polish": "CROPolish"}
MAIN = ["construct", "swap", "ga", "sa", "sa+polish", "cro", "cro+polish", "minlp", "ours"]
EXACT_RESTRICTED = ["mip", "cpsat"]
ABLM = ["ours", "ours_noelite", "ours_unif", "ours_s2", "ours_s1", "ours_nos1"]

macros = []


def M(name, value):
    assert name.isalpha(), name
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def c2(v, nd=2):
    """Cost with thousands separators for LaTeX."""
    return f"{v:,.{nd}f}".replace(",", "{,}")


def p2(v, nd=2):
    s = f"{v:.{nd}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    if float(s.replace("$-$", "-")) == 0:
        s = s.replace("$-$", "")
    return s


def digits(inst):
    return 2


# ------------------------------------------------------------------ statistics
ST = {(m, i): D.stats(RS, m, i, S.REF[i]) for m in MAIN + ABLM + EXACT_RESTRICTED for i in S.INSTANCES}

for (m, i), st in ST.items():
    if st["n"] == 0:
        continue
    k = MNAME[m] + INAME[i]
    M("bxNfeas" + k, st["nf"]); M("bxNruns" + k, st["n"])
    if st["nf"]:
        M("bxBest" + k, c2(st["best"])); M("bxMean" + k, c2(st["mean"])); M("bxStd" + k, c2(st["std"]))
        M("bxGapBest" + k, p2(st["gbest"])); M("bxGapMean" + k, p2(st["gmean"]))
        if i == "O7":
            M("bxGapMeanThree" + k, p2(st["gmean"], 3))

for i in S.INSTANCES:
    M("bxRef" + INAME[i], c2(S.REF[i])); M("bxBudget" + INAME[i], S.BUDGET[i])

# run counts
n_runs = len(RS)
n_layout = sum(1 for r in RS if r.get("rects"))
M("bxNrunfiles", n_runs); M("bxNlayouts", n_layout)
n_bench = sum(1 for r in RS if r["method"] in MAIN)
M("bxNrunsMain", n_bench)
M("bxNrunsAbl", sum(1 for r in RS if r["method"] in ("ours_s1", "ours_s2", "ours_nos1", "ours_noelite", "ours_unif")))

# over-runs (wall > budget); polish rows excluded (their wall is the baseline run; polish time is separate)
STOCH = ["swap", "ga", "sa", "cro"]
OURSF = ["ours", "ours_s1", "ours_s2", "ours_nos1", "ours_noelite", "ours_unif"]
for i in S.INSTANCES:
    T = S.BUDGET[i]
    ov = [r["wall"] - T for r in RS if r["instance"] == i and r["method"] in OURSF]
    M("bxOverOursMaxS" + INAME[i], f"{max(0.0, max(ov)):.2f}")
    M("bxOverOursMaxPct" + INAME[i], f"{max(0.0, 100 * max(ov) / T):.1f}")
ovb = [r["wall"] - S.BUDGET[r["instance"]] for r in RS if r["method"] in STOCH]
M("bxOverBaseMaxS", f"{max(0.0, max(ovb)):.2f}")
over = []
for r in RS:
    if r["method"] in ("mip", "cpsat", "construct", "minlp", "sa+polish", "cro+polish"):
        continue
    T = S.BUDGET[r["instance"]]
    if r["wall"] > T + 0.5:
        over.append(r)
M("bxNoverrun", len(over))
M("bxMaxOverrunPct", f"{max(100 * (r['wall'] - S.BUDGET[r['instance']]) / S.BUDGET[r['instance']] for r in over):.1f}"
  if over else "0")
M("bxMaxWallDu", f"{max(r['wall'] for r in RS if r['instance'] == 'Du62' and r['method'] not in ('mip', 'cpsat')):.1f}")
M("bxMaxWallExact", f"{max(r['wall'] for r in RS if r['method'] in ('mip', 'cpsat', 'minlp')):.1f}")

# polish rows: time and effect
for m in ["sa+polish", "cro+polish"]:
    pt = [r["polish_time"] for r in RS if r["method"] == m]
    M("bxPolishTimeMax" + MNAME[m], f"{max(pt):.2f}")
    for i in S.INSTANCES:
        R = D.by(RS, m, i)
        n_imp = sum(1 for r in R if r["cost"] < r["original_cost"] * (1 - 1e-8))
        M("bxPolishNimp" + MNAME[m] + INAME[i], n_imp)
        red = [100 * (r["original_cost"] - r["cost"]) / r["original_cost"] for r in R]
        md = np.median(red)
        M("bxPolishMedRed" + MNAME[m] + INAME[i], f"{md:.1f}" if md >= 1 else f"{md:.2g}")

# MINLP (exact convex model, SCIP, one 300 s run)
for i in S.INSTANCES:
    R = D.by(RS, "minlp", i)
    if not R:
        continue
    me = R[0]["meta"]
    M("bxMinlpStatus" + INAME[i], me.get("status", "?"))
    M("bxMinlpBound" + INAME[i], c2(me["bound"]))
    M("bxMinlpGap" + INAME[i], f"{me['gap']:.1f}" if me.get("gap") is not None else "--")
    M("bxMinlpNodes" + INAME[i], f"{me['nodes']:,}".replace(",", "{,}"))
    M("bxMinlpIncumbent" + INAME[i], c2(R[0]["cost"]) if R[0].get("feasible") else "none")
o7m = D.by(RS, "minlp", "O7")[0]
M("bxMinlpOsevenCostTen", f"{o7m['cost']:.7f}")
first_opt = min(t for t, p, d in o7m["meta"]["progress"] if p is not None and p <= S.REF["O7"] * (1 + 1e-8))
M("bxMinlpOsevenTfirstOpt", f"{first_opt:.0f}")
_rl0 = open(os.path.join(D.ROOT, "results", "bench", "RUNLOG.md")).read()
_mo = re.search(r"limit 3600 s:\s+OPTIMAL\s+([\d.]+)\s+at\s+(\d+) s", _rl0)
M("bxMinlpOsevenProofT", _mo.group(2)); M("bxMinlpOsevenProofVal", _mo.group(1))
M("bxRefOsevenFull", f"{S.REF['O7']:.7f}")

# which method has the lowest mean per instance (among heuristics with all runs feasible)
lowest = {}
for i in S.INSTANCES:
    cand = [(ST[(m, i)]["mean"], m) for m in ["swap", "ga", "sa", "sa+polish", "cro", "cro+polish", "ours"]
            if ST[(m, i)]["nf"] == ST[(m, i)]["n"]]
    best_mean = min(c[0] for c in cand)
    lowest[i] = [m for v, m in cand if round(v, 2) == round(best_mean, 2)]
M("bxNinstOursLowestMean", sum(1 for i in S.INSTANCES if "ours" in lowest[i]))

# medians of the gap
for (m, i), st in ST.items():
    if st["n"] and st["nf"]:
        M("bxGapMedian" + MNAME[m] + INAME[i], p2(100 * (st["median"] - S.REF[i]) / S.REF[i]))

# test-suite result recorded in results/bench/RUNLOG.md
import re as _re
_rl = open(os.path.join(D.ROOT, "results", "bench", "RUNLOG.md")).read()
_m = _re.findall(r"(\d+) passed, (\d+) xfailed", _rl)[-1]
M("bxTestsPassed", _m[0]); M("bxTestsXfail", _m[1])

# Mann-Whitney U tests with Holm correction (logic of review/stats.py): ours vs each baseline and vs ours_s1/s2
CMP = ["swap", "ga", "sa", "sa+polish", "cro", "cro+polish", "ours_s1", "ours_s2"]


def fmt_p(p):
    if p >= 0.01:
        return f"{p:.3f}" if p < 0.995 else "1"
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return f"${m:.1f}\\cdot10^{{{e}}}$"


STAT = {}
for i in S.INSTANCES:
    a = ST[("ours", i)]["costs"]
    rows = []
    for m in CMP:
        st = ST[(m, i)]
        if st["nf"] < 2:
            rows.append((m, None, None, None)); continue
        b = st["costs"]
        if np.ptp(np.r_[a, b]) < 1e-6 * abs(a.mean()):
            rows.append((m, 1.0, 0.5, "tie")); continue
        p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        A12 = (np.sum(a[:, None] < b[None, :] - 1e-9) + 0.5 * np.sum(np.abs(a[:, None] - b[None, :]) <= 1e-9)) / (len(a) * len(b))
        rows.append((m, p, A12, None))
    valid = [k for k, r in enumerate(rows) if r[1] is not None]
    order = sorted(valid, key=lambda k: rows[k][1]); kk = len(valid); run = 0.0; adj = {}
    for rank, k in enumerate(order):
        run = max(run, min(1.0, (kk - rank) * rows[k][1])); adj[k] = run
    for k, (m, p, A12, tag) in enumerate(rows):
        STAT[(m, i)] = dict(p=p, A12=A12, holm=adj.get(k), tag=tag)
        if p is not None:
            M("bxP" + MNAME[m] + INAME[i], fmt_p(p).strip("$") if tag is None else "1")
            M("bxPholm" + MNAME[m] + INAME[i], fmt_p(adj[k]).strip("$") if tag is None else "1")
            M("bxAtwelve" + MNAME[m] + INAME[i], f"{A12:.2f}")

# Stage-2 effect (in-run: stage1_cost vs final cost)
for m in ["ours", "ours_s2"]:
    for i in S.INSTANCES:
        P = [(r["meta"]["stage1_cost"], r["cost"]) for r in D.by(RS, m, i) if r.get("feasible")]
        red = [100 * (a - b) / a for a, b in P]
        k = MNAME[m] + INAME[i]
        M("bxStageTwoMedRed" + k, f"{np.median(red):.3g}")
        M("bxStageTwoMaxRed" + k, f"{np.max(red):.3g}")
        M("bxStageTwoNneg" + k, sum(1 for x in red if x < -1e-9))
# claim in the text: on vC10a and AB20 the single polish changes the cost by < 1e-5 %
for i in ["vC10a", "AB20"]:
    P = [(r["meta"]["stage1_cost"], r["cost"]) for r in D.by(RS, "ours_s2", i) if r.get("feasible")]
    assert max(abs(100 * (a - b) / a) for a, b in P) < 1e-5, i
# Stage-2 SOCP statistics (ours)
for i in S.INSTANCES:
    R = D.by(RS, "ours", i)
    po = sum(1 for r in R if r["meta"].get("stage2") == "polish only")
    M("bxPolishOnly" + INAME[i], po)
    sv = [r["meta"]["socp_solves"] for r in R if "socp_solves" in r["meta"]]
    inf = [r["meta"]["socp_infeasible"] / r["meta"]["socp_solves"] for r in R if r["meta"].get("socp_solves")]
    ms = [r["meta"]["socp_ms"] for r in R if "socp_ms" in r["meta"]]
    if sv:
        M("bxSocpSolves" + INAME[i], f"{np.mean(sv):,.0f}".replace(",", "{,}"))
        M("bxSocpInfPct" + INAME[i], f"{100 * np.mean(inf):.0f}")
    if ms:
        M("bxSocpMs" + INAME[i], f"{np.mean(ms):.1f}")

# identical results of ours / noelite / unif on Du62
same = all(abs(a["cost"] - b["cost"]) < 1e-6 and abs(a["cost"] - c["cost"]) < 1e-6
           for a, b, c in zip(D.by(RS, "ours", "Du62"), D.by(RS, "ours_noelite", "Du62"), D.by(RS, "ours_unif", "Du62")))
M("bxDuThreeIdentical", "yes" if same else "no")

# ------------------------------------------------------------------ best layouts (re-checked here)
BEST = {}
for i in S.INSTANCES:
    lay = D.best_layout(i); inst = load(i)
    ev = evaluate(inst, [tuple(r) for r in lay["rects"]])
    strict = len(violations(inst, [tuple(r) for r in lay["rects"]], eps=0.0)) == 0
    assert ev["feasible"], i
    assert abs(ev["cost"] - lay["cost"]) < 1e-6, (i, ev["cost"], lay["cost"])
    tl = max(int(x) for x in re.findall(r"(\d+) s\b", lay["run"]))
    BEST[i] = dict(cost=ev["cost"], strict=strict, tl=tl, src=lay.get("run_file"), run=lay["run"],
                   resid=lay.get("max_residual"))
    M("bxRecBest" + INAME[i], c2(ev["cost"]))
    M("bxRecStrict" + INAME[i], "yes" if strict else "no")
    M("bxRecTL" + INAME[i], f"{tl}")
M("bxRecMaxTL", f"{max(b['tl'] for b in BEST.values()):.0f}")
M("bxRecBestOsevenSeven", f"{BEST['O7']['cost']:.7f}")
M("bxRecOsevenRelGap", f"{(BEST['O7']['cost'] - S.REF['O7']) / S.REF['O7']:.1e}".replace("e-0", "\\cdot10^{-").replace("e-", "\\cdot10^{-") + "}")

# best protocol run over all methods (results/bench + bench_polish), per instance
PBEST = {}
for i in S.INSTANCES:
    c = [(r["cost"], r["method"], r["seed"]) for r in RS if r["instance"] == i and r.get("feasible")
         and r["method"] not in ("construct",)]
    cmin = min(c)[0]
    # methods within 1e-7 relative of the minimum count as ties (solver round-off, e.g. vC10a 18,522.78959)
    tied = sorted({m for v, m, sd in c if v <= cmin * (1 + 1e-7)})
    lab = {"ours_noelite": "Ours, elite pool 1", "ours_s2": "Ours, Stage 1 + polish"}
    name = lab.get(tied[0], S.LABEL.get(tied[0], tied[0])) if len(tied) == 1 else f"tie of {len(tied)} methods"
    PBEST[i] = (cmin, name, tied)
    M("bxProtBest" + INAME[i], c2(cmin, 4 if i == "O7" else 2)); M("bxProtBestMethod" + INAME[i], name)
    M("bxProtBestNtied" + INAME[i], len(tied))

# ------------------------------------------------------------------ published values (results/RECORDS.md)
# (instance, value, source text, bib key or None, area rule, comparability text)
PUB = [
    ("O7", 131.6531381, "MINLPLib \\texttt{o7}", "minlplib_o7", "exact", "Yes: proven optimum"),
    ("O7", 131.56, "G\\&R, T.~10", "goncalves2015", "outer approx.", "No: below the optimum"),
    ("O7", 131.6773, "Hou 2019\\tnote{a}", None, "unknown", "Above the optimum"),
    ("O7", 131.69, "J\\&C, T.~6", "jeong2026", "exact", "Yes; best slicing layout"),
    ("vC10a", 18522.79, "J\\&C, T.~6", "jeong2026", "exact", "Yes, $w_{9,10}=59.2$\\tnote{b}"),
    ("AB20", 4927.27, "J\\&C, T.~6", "jeong2026", "exact", "No: aspect 4.008 $>\\alpha$"),
    ("AB20", 5021.17, "G\\&R, T.~12", "goncalves2015", "outer approx.", "Cond.: flow version not recorded"),
    ("SC30", 3343.47, "J\\&C, T.~6", "jeong2026", "exact", "Yes, in our favour\\tnote{c}"),
    ("SC30", 3367.87, "G\\&R, T.~12", "goncalves2015", "outer approx.", "Cond.: flow version not recorded"),
    ("SC30", 3412.87, "Hou 2019\\tnote{a}", None, "unknown", "Cond.: rules not recorded"),
    ("Du62", 3635307.30, "K\\&C via J\\&C", "kang2017", "unknown", "Cond.: floor unconfirmed\\tnote{d}"),
    ("Du62", 3672799.99, "J\\&C, T.~6", "jeong2026", "exact", "Yes ($117.124$ square)"),
]


def write(name, text):
    open(os.path.join(TAB, name), "w").write(text)


# ------------------------------------------------------------------ tab_setup
rows = []
ORIG = {"O7": "meller1998", "vC10a": "vancamp1992", "AB20": "armour1963", "SC30": "liu2007", "Du62": "dunker2003"}
for i in S.INSTANCES:
    d = D.instance(i)
    fill = 100 * sum(d["areas"]) / (d["W"] * d["H"])
    M("bxFill" + INAME[i], f"{fill:.2f}")
    M("bxNpairs" + INAME[i], len(d["weights"]))
    M("bxFree" + INAME[i], f"{100 - fill:.2f}")
    W = f"{d['W']:g}"; H = f"{d['H']:g}"
    rows.append(f"{i} & \\cite{{{ORIG[i]}}} & {d['n']} & ${W}\\times{H}$ & {d['shape']['value']:g} & "
                f"{fill:.2f} & {len(d['weights'])} & {S.BUDGET[i]} \\\\")
write("tab_setup.tex", r"""\begin{table}[tbp]
\centering
\caption{Test instances and time budgets. $n$: number of departments; $\alpha$: maximum aspect ratio;
fill: total department area divided by floor area; pairs: department pairs with non-zero flow;
budget: wall-clock limit per run for every stochastic method.}
\label{tab:bx-setup}
\small
\begin{tabular}{llrcrrrr}
\toprule
Instance & Origin & $n$ & Floor $W\times H$ & $\alpha$ & Fill (\%) & Pairs & Budget (s)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ tab_main
lines = []
for i in S.INSTANCES:
    means = {m: ST[(m, i)]["mean"] for m in MAIN if ST[(m, i)]["nf"] and ST[(m, i)]["n"] > 1}
    bestm = min(means.values())
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{i}}} (reference {c2(S.REF[i], 4 if i == 'O7' else 2)}, budget {S.BUDGET[i]}~s)}}\\\\")
    for m in MAIN:
        st = ST[(m, i)]
        if st["n"] == 0:
            continue
        lab = S.LABEL[m]
        if not st["nf"]:
            lines.append(f"\\quad {lab} & {st['nf']}/{st['n']} & -- & -- & -- & -- \\\\")
            continue
        mean = c2(st["mean"]); std = c2(st["std"])
        ms = f"{mean} $\\pm$ {std}" if st["nf"] > 1 else f"{mean}"
        if m in means and round(st["mean"], 2) == round(bestm, 2):
            ms = f"\\textbf{{{mean}}} $\\pm$ {std}"
        nd = 4 if i == "O7" else 2
        if nd == 4:
            mean = f"{st['mean']:.4f}"; std = f"{st['std']:.4f}"
            ms = f"{mean} $\\pm$ {std}" if st["nf"] > 1 else f"{mean}"
            if m in means and round(st["mean"], 4) == round(bestm, 4):
                ms = f"\\textbf{{{mean}}} $\\pm$ {std}"
        lines.append(f"\\quad {lab} & {st['nf']}/{st['n']} & {c2(st['best'], nd)} & {ms} & {p2(st['gbest'], 3 if nd == 4 else 2)} & {p2(st['gmean'], 3 if nd == 4 else 2)} \\\\")
    if i != S.INSTANCES[-1]:
        lines.append("\\addlinespace")
write("tab_main.tex", r"""\begin{table}[tbp]
\centering
\caption{Results under the common protocol (10 seeds per stochastic method; one run for the deterministic
construction heuristic and one 300~s run for the exact convex MINLP). SA/CRO + polish: one Stage-2 SOCP solve
applied to the final layout of each SA/CRO run (polish time, at most \bxPolishTimeMaxSAPolish~s and
\bxPolishTimeMaxCROPolish~s, not included in the budget). Feas.: runs with a feasible layout / runs.
Costs are those of the independent evaluator, over feasible runs only; std is the sample standard deviation.
Gaps are $100\,(\text{cost}-\text{ref})/\text{ref}$ for the best run and for the mean; the O7 reference is the
proven optimum \bxRefOsevenFull. Bold: lowest mean among the methods with 10 runs.}
\label{tab:bx-main}
\footnotesize
\renewcommand{\arraystretch}{0.92}
\begin{tabular}{lrrrrr}
\toprule
Method & Feas. & Best & Mean $\pm$ std & Gap best (\%) & Gap mean (\%)\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ tab_ablation
lines = []
for i in ["O7", "vC10a", "AB20", "SC30", "Du62"]:
    ms = [m for m in ABLM if ST[(m, i)]["n"]]
    means = {m: ST[(m, i)]["mean"] for m in ms if ST[(m, i)]["nf"] == ST[(m, i)]["n"]}
    bestm = min(means.values())
    lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{i}}} (budget {S.BUDGET[i]}~s)}}\\\\")
    for m in ms:
        st = ST[(m, i)]
        lab = "Ours (full)" if m == "ours" else S.LABEL[m]
        if not st["nf"]:
            lines.append(f"\\quad {lab} & {st['nf']}/{st['n']} & -- & -- & -- \\\\")
            continue
        mean = c2(st["mean"]); std = c2(st["std"])
        mm = f"\\textbf{{{mean}}}" if round(st["mean"], 2) == round(bestm, 2) else mean
        lines.append(f"\\quad {lab} & {st['nf']}/{st['n']} & {c2(st['best'])} & {mm} $\\pm$ {std} & {p2(st['gmean'])} \\\\")
    if i != "Du62":
        lines.append("\\addlinespace")
write("tab_ablation.tex", r"""\begin{table}[tbp]
\centering
\caption{Ablation of the two-stage method (10 seeds, same budgets as \cref{tab:bx-main}). Stage 1 only: slicing-tree
annealing for the whole budget; Stage 1 + polish: one SOCP solve on the final Stage-1 layout; No Stage 1:
Stage 2 started from a random feasible sequence pair; Elite pool 1: Stage 2 from the best Stage-1 layout only;
Uniform partner: Stage-2 move partner chosen uniformly ($p_{\text{near}}=0$). The last three variants were run
on AB20, SC30 and Du62 only. Bold: lowest mean per instance among variants with all runs feasible.}
\label{tab:bx-ablation}
\small
\begin{tabular}{lrrrr}
\toprule
Variant & Feas. & Best & Mean $\pm$ std & Gap mean (\%)\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ tab_records
lines = []
prev = None
for inst, val, src, key, area, comp in PUB:
    if prev is not None and inst != prev:
        lines.append("\\addlinespace")
    first = inst != prev
    nd = 4 if inst == "O7" else 2
    ob = BEST[inst]["cost"]
    d_rec = 100 * (ob - val) / val
    cite = f"~\\cite{{{key}}}" if key else ""
    pc = f"{c2(PBEST[inst][0], nd)} ({PBEST[inst][1]})" if first else ""
    oc = f"{c2(ob, nd)}" if first else ""
    vtxt = c2(val, 7 if val == 131.6531381 else (4 if val == 131.6773 else 2))
    lines.append(f"{inst if first else ''} & {vtxt} & {src}{cite} & {area} & {comp} & {pc} & {oc} & {p2(d_rec, 3 if inst == 'O7' else 2)} \\\\")
    prev = inst
write("tab_records.tex", r"""\begin{table}[tbp]
\centering
\caption{Best published values and our best layouts. Area rule: \emph{exact} means every department meets its
area exactly; \emph{outer approx.} means the area constraint $w_ih_i\ge A_i$ is replaced by tangent cuts, so
areas can be slightly under-met and the value can lie below any exact-area layout. Comparable (Cond.: conditional):
whether the value can be compared with a strictly feasible layout on our instance file.
Best protocol run: lowest cost over all protocol runs of all methods (method in brackets).
Ours (record): best layout of our method from longer, separately configured runs (up to \bxRecMaxTL~s;
not protocol results), re-checked by the evaluator. $\Delta$: Ours (record) relative to the published value.
G\&R: Gon\c{c}alves and Resende; J\&C: Jeong and Chae; K\&C: Kang and Chae.}
\label{tab:bx-records}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{threeparttable}
\begin{tabular}{lrll>{\raggedright\arraybackslash}p{2.35cm}>{\raggedleft\arraybackslash}p{2.2cm}rr}
\toprule
Inst. & Published & Source & Area rule & Comparable & Best protocol run & Ours (record) & $\Delta$ (\%)\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item[a] Value as recorded in our literature notes; floor, aspect rule and area handling of this source were not
checked, and the full reference is not verified.
\item[b] Our instance file uses $w_{9,10}=59.2$, the value under which the published layout gives 18{,}522.79.
\item[c] The published layouts are slicing layouts that fill the free area with 17 dummy blocks; our model allows
free space anywhere and non-slicing layouts, so it is less restrictive.
\item[d] Secondary citation (Jeong and Chae; Kang13 data file); the floor is probably the $117.124$ square but this
is not confirmed from the original paper.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ------------------------------------------------------------------ tab_stats (Mann-Whitney U, Holm)
CMPLAB = {"swap": "Swap", "ga": "GA", "sa": "SA", "sa+polish": "SA + polish", "cro": "CRO",
          "cro+polish": "CRO + polish", "ours_s1": "Stage 1 only", "ours_s2": "Stage 1 + polish"}
lines = []
for m in CMP:
    cells = []
    for i in S.INSTANCES:
        d = STAT[(m, i)]
        if d["p"] is None:
            cells.append("n/a")
        elif d["tag"] == "tie":
            cells.append("tie")
        else:
            ph = fmt_p(d["holm"]); pr = fmt_p(d["p"])
            sig = d["holm"] < 0.05
            cells.append((f"\\textbf{{{ph}}}" if sig and not ph.startswith("$") else (f"$\\mathbf{{{ph.strip('$')}}}$" if sig else ph))
                         + f" ({d['A12']:.2f})")
    lines.append(CMPLAB[m] + " & " + " & ".join(cells) + " \\\\")
    if m == "cro+polish":
        lines.append("\\midrule")
write("tab_stats.tex", r"""\begin{table}[tbp]
\centering
\caption{Two-sided Mann--Whitney U tests of the full method against each other method (10 runs each, unpaired),
Holm-adjusted within each instance over the eight comparisons. Each cell: adjusted $p$-value and, in brackets,
the effect size $A_{12}=P(\text{ours}<\text{other})+\tfrac12P(\text{tie})$ (values above 0.5 favour our method).
Bold: adjusted $p<0.05$. tie: all 20 costs equal within $10^{-6}$ relative. n/a: fewer than two feasible runs.
The $p$-values use the exact distribution when there are no ties and the normal approximation otherwise.}
\label{tab:bx-stats}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Ours vs & O7 & vC10a & AB20 & SC30 & Du62\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ full-length CRO reproduction (bench/cro_reference.md)
import re
txt = open(os.path.join(D.ROOT, "bench", "cro_reference.md")).read()
sec = {k: txt.split(f"## {k}")[1].split("##")[0] for k in ["AB20", "SC30", "Du62"]}
for k, body in sec.items():
    row = [l for l in body.splitlines() if l.startswith("| 7x7")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    # Grid | Paper best | Our best | Gap | Paper mean | Our mean | Gap
    M("bxCroFullPaperBest" + INAME[k], c2(float(cells[1].replace(",", ""))))
    M("bxCroFullOurBest" + INAME[k], c2(float(cells[2].replace(",", ""))))
    M("bxCroFullPaperMean" + INAME[k], c2(float(cells[4].replace(",", ""))))
    M("bxCroFullOurMean" + INAME[k], c2(float(cells[5].replace(",", ""))))
ov = re.search(r"\| O7 \| ([\d.]+) \| ([\d.]+) \| (\d+) / (\d+)", txt)
M("bxCroFullOsevenHits", f"{ov.group(3)}/{ov.group(4)}")
vv = re.search(r"\| vC10a \(w9,10 = 59.2\) \| ([\d,.]+) \| ([\d,.]+) \| (\d+) / (\d+)", txt)
M("bxCroFullvCtenaHits", f"{vv.group(3)}/{vv.group(4)}")

# record deltas used in text (absolute values; the text says "below")
for inst, val, src, key, area, comp in PUB:
    pass
M("bxDeltaSCthirtyJC", p2(abs(100 * (BEST["SC30"]["cost"] - 3343.47) / 3343.47)))
M("bxDeltaABtwentyGR", p2(abs(100 * (BEST["AB20"]["cost"] - 5021.17) / 5021.17)))
M("bxDeltaABtwentyJC", p2(abs(100 * (BEST["AB20"]["cost"] - 4927.27) / 4927.27)))
M("bxDeltaDusixtytwoJC", p2(abs(100 * (BEST["Du62"]["cost"] - 3672799.99) / 3672799.99)))
M("bxDeltaDusixtytwoKC", p2(abs(100 * (BEST["Du62"]["cost"] - 3635307.30) / 3635307.30)))
M("bxDeltaDusixtytwoJCsa", p2(abs(100 * (ST[("sa", "Du62")]["best"] - 3672799.99) / 3672799.99)))
M("bxDeltaDusixtytwoKCsa", p2(abs(100 * (ST[("sa", "Du62")]["best"] - 3635307.30) / 3635307.30)))
assert BEST["SC30"]["cost"] < 3343.47 and BEST["AB20"]["cost"] < 4927.27 and BEST["Du62"]["cost"] < 3635307.30
assert ST[("sa", "Du62")]["best"] < 3635307.30 and ST[("sa", "AB20")]["best"] < 4927.27

open(os.path.join(HERE, "numbers.tex"), "w").write(
    "% Generated by paper/make_tables.py -- do not edit by hand.\n" + "\n".join(macros) + "\n")
print(f"{len(macros)} macros; lowest mean per instance: {lowest}; over-runs: {len(over)}")
for r in over:
    print("  over-run", r["method"], r["instance"], r["seed"], r["wall"])
print({i: (b["cost"], b["strict"], b["tl"], b["src"]) for i, b in BEST.items()})
