"""Regenerate every figure of the benchmark section (paper/figures/*.pdf and *.png).

Usage:  python3 paper/make_figures.py
Reads results/bench/*.json, results/ours/best_*.json and instances/uaflp/*.json. Runs no optimisation.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402
import data as D   # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
S.apply()
RS = D.runs()


def fmt_cost(v):
    return f"{v:,.2f}"


# ---------------------------------------------------------------- 1. gap box plots
def symlog_ticks(ax, lo, hi):
    cand = [-10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50, 100, 200]
    t = [c for c in cand if lo <= c <= hi]
    ax.yaxis.set_major_locator(FixedLocator(t))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def fig_gap_box():
    fig, axes = plt.subplots(1, 5, figsize=(S.COL2, 2.55))
    rng = np.random.default_rng(0)
    cols = S.METHODS + ["minlp"]
    for ax, inst in zip(axes, S.INSTANCES):
        lo, hi = 0.0, 0.0
        nolayout = []
        for k, m in enumerate(cols):
            st = D.stats(RS, m, inst, S.REF[inst])
            if not st["nf"]:
                nolayout.append((k, st))
                continue
            g = st["gaps"]
            lo, hi = min(lo, g.min()), max(hi, g.max())
            if len(g) >= 5:
                bp = ax.boxplot([g], positions=[k], widths=0.62, patch_artist=True, showfliers=False,
                                whis=(0, 100), medianprops=dict(color=S.INK, lw=1.0),
                                whiskerprops=dict(color=S.EDGE[m], lw=0.7), capprops=dict(color=S.EDGE[m], lw=0.7))
                for b in bp["boxes"]:
                    b.set(facecolor=S.COLOR[m] + "59", edgecolor=S.EDGE[m], lw=0.8, hatch=S.HATCH[m])
            # every run as a point (jittered), marker identifies the method in grayscale
            x = k + rng.uniform(-0.17, 0.17, len(g)) if len(g) > 1 else np.array([k])
            ax.scatter(x, g, s=9 if len(g) > 1 else 22, marker=S.MARKER[m], facecolor=S.COLOR[m],
                       edgecolor=S.EDGE[m], lw=0.5, zorder=3)
            if st["nf"] < st["n"]:
                txt = f"{st['nf']}/{st['n']} feasible"
                ax.annotate(txt, (k, g.max()), xytext=(-2, 5), textcoords="offset points", ha="left",
                            va="bottom", fontsize=7, color=S.INK2)
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
        ax.set_ylim(min(lo * 1.6, -0.6), hi * 2.2 + 2)
        symlog_ticks(ax, min(lo * 1.6, -0.6), hi * 2.2 + 2)
        ax.axhline(0, color=S.INK2, lw=0.7, ls="--", zorder=1)
        for k, st in nolayout:
            ax.text(k, 0.01, f"{st['nf']}/{st['n']}", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7, color=S.INK2)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([S.LABEL[m] for m in cols], rotation=60, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{inst}", pad=3)
    axes[0].set_ylabel("Gap to reference value (%)")
    fig.tight_layout(w_pad=0.6)
    S.save(fig, os.path.join(OUT, "fig_gap_box"))
    plt.close(fig)


# ---------------------------------------------------------------- 2. convergence
CONV_METHODS = ["swap", "ga", "sa", "cro", "ours"]


def step_values(trace, grid):
    t = np.array([p[0] for p in trace]); c = np.array([p[1] for p in trace])
    idx = np.searchsorted(t, grid, side="right") - 1
    v = np.where(idx >= 0, c[np.clip(idx, 0, None)], np.nan)
    return v


def fig_convergence():
    insts = ["AB20", "SC30", "Du62"]
    fig, axes = plt.subplots(1, 3, figsize=(S.COL2, 2.5))
    missing = {}
    minlp_info = {}
    for ax, inst in zip(axes, insts):
        T = S.BUDGET[inst]
        grid = np.logspace(np.log10(0.1), np.log10(T), 300)
        ref = S.REF[inst]
        ymax = -np.inf
        for m in CONV_METHODS:
            R = [r for r in D.by(RS, m, inst) if r.get("meta", {}).get("trace")]
            if len(R) < 10:
                missing.setdefault(inst, []).append(m)
                continue
            V = np.array([step_values(r["meta"]["trace"], grid) for r in R])
            nfin = np.isfinite(V).sum(0)
            ok = nfin >= (len(R) // 2 + 1)       # median defined once most seeds have a feasible layout
            Vg = 100 * (V - ref) / ref
            Vs = np.where(np.isfinite(Vg), Vg, np.inf)
            med = np.where(ok, np.median(Vs, 0), np.nan)
            full = nfin == len(R)
            q1 = np.where(full, np.percentile(np.where(full, Vs, 0), 25, axis=0), np.nan)
            q3 = np.where(full, np.percentile(np.where(full, Vs, 0), 75, axis=0), np.nan)
            ax.fill_between(grid, q1, q3, color=S.COLOR[m], alpha=0.18, lw=0)
            ax.plot(grid, med, color=S.COLOR[m], ls=S.LINESTYLE[m], lw=1.3, label=S.LABEL[m])
            # marker at the end of the run, so lines are distinguishable in grayscale
            ax.plot(grid[-1], med[-1], marker=S.MARKER[m], color=S.COLOR[m], mec=S.EDGE[m], mew=0.5, ms=5,
                    clip_on=False, zorder=4)
            sel = med[np.isfinite(med) & (grid >= 0.1 * T)]
            if len(sel):
                ymax = max(ymax, np.nanmax(sel))
        # exact MINLP: incumbent (primal) progress of its single run, if any
        for r in D.by(RS, "minlp", inst):
            pr = [(t, p) for t, p, d in r.get("meta", {}).get("progress", []) if p is not None]
            if pr:
                t = np.array([p[0] for p in pr]); g = 100 * (np.array([p[1] for p in pr]) - ref) / ref
                ax.step(t, g, where="post", color=S.COLOR["minlp"], lw=0.9, marker=S.MARKER["minlp"], ms=3)
                minlp_info[inst] = float(g[-1])
            else:
                minlp_info[inst] = None
        ax.axhline(0, color=S.INK, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(0.1, T * 1.12)
        ax.set_xlabel("Time (s, log scale)")
        lo = -2.5 if inst != "Du62" else -0.5
        top = {"AB20": 30, "SC30": 40, "Du62": 8}[inst]
        ax.set_ylim(lo, top)
        ax.set_title(f"{inst} (budget {T} s)", pad=3)
    axes[0].set_ylabel("Gap of best feasible cost (%)")
    h = [Line2D([], [], color=S.COLOR[m], ls=S.LINESTYLE[m], marker=S.MARKER[m], mec=S.EDGE[m], ms=4, lw=1.3,
                label=S.LABEL[m]) for m in CONV_METHODS if all(m not in v for v in missing.values())]
    h.append(Line2D([], [], color=S.INK, ls="--", lw=0.8, label="Reference value"))
    fig.legend(handles=h, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, 1.09), handlelength=2.6)
    fig.tight_layout(w_pad=0.8)
    S.save(fig, os.path.join(OUT, "fig_convergence"))
    plt.close(fig)
    return missing, minlp_info


# ---------------------------------------------------------------- 3. layouts
def draw_layout(ax, inst, lay, fs, kmax=40):
    W, H = inst["W"], inst["H"]
    rects = lay["rects"]
    cen = np.array([[x + w / 2, y + h / 2] for x, y, w, h in rects])
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec=S.INK, lw=1.2, zorder=5))
    for i, (x, y, w, h) in enumerate(rects):
        ax.add_patch(Rectangle((x, y), w, h, fc="#eef3fb", ec="#1c5cab", lw=0.6, zorder=1))
    # flows: thin lines for the largest flows (top 15 % of the pairs by weight, at least 5)
    ws = sorted(inst["weights"], key=lambda t: -t[2])
    k = min(kmax, max(5, int(round(0.15 * len(ws)))))
    top = ws[:k]; wmax = top[0][2]; wmin = top[-1][2]
    for i, j, w in top:
        a, b = cen[i - 1], cen[j - 1]
        lw = 0.4 + 1.0 * (w - wmin) / (wmax - wmin + 1e-12)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#b8481d", lw=lw, alpha=0.75, zorder=2, solid_capstyle="round")
    for i, (cx, cy) in enumerate(cen):
        ax.text(cx, cy, str(i + 1), ha="center", va="center", fontsize=fs, color=S.INK, zorder=6,
                bbox=dict(boxstyle="square,pad=0.05", fc="#eef3fb", ec="none", alpha=0.85))
    ax.set_xlim(-0.01 * W, 1.01 * W); ax.set_ylim(-0.01 * H, 1.01 * H)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return k


def fig_layouts():
    fig = plt.figure(figsize=(S.COL2, 5.8))
    # row 1: O7, vC10a, AB20 (same height); row 2: SC30 and Du62 (same width)
    g1 = fig.add_gridspec(1, 3, width_ratios=[8.54 / 13, 25 / 51, 1.5], left=0.0, right=1.0, top=0.97,
                          bottom=0.62, wspace=0.12)
    g2 = fig.add_gridspec(1, 2, left=0.0, right=1.0, top=0.565, bottom=0.0, wspace=0.06)
    pos = {"O7": g1[0, 0], "vC10a": g1[0, 1], "AB20": g1[0, 2], "SC30": g2[0, 0], "Du62": g2[0, 1]}
    info = {}
    for name in S.INSTANCES:
        inst = D.instance(name); lay = D.best_layout(name)
        ax = fig.add_subplot(pos[name])
        fs = 8 if name in ("O7", "vC10a", "AB20", "SC30") else 7
        k = draw_layout(ax, inst, lay, fs, kmax=25 if name == "Du62" else 40)
        cs = f"{lay['cost']:.4f}" if name == "O7" else fmt_cost(lay["cost"])
        ax.set_title(f"{name}: cost {cs}", pad=3, fontsize=8)
        info[name] = dict(cost=lay["cost"], nflows=k, W=inst["W"], H=inst["H"])
    S.save(fig, os.path.join(OUT, "fig_layouts"))
    plt.close(fig)
    return info


# ---------------------------------------------------------------- 4. ablation
def fig_ablation():
    insts = ["AB20", "SC30", "Du62"]
    fig, axes = plt.subplots(1, 3, figsize=(S.COL2, 2.3), sharey=True)
    ys = np.arange(len(S.ABL))[::-1]
    for ax, inst in zip(axes, insts):
        ref = S.REF[inst]
        for y, m in zip(ys, S.ABL):
            st = D.stats(RS, m, inst, ref)
            if not st["nf"]:
                ax.text(0.97, y, f"no feasible run ({st['nf']}/{st['n']})", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7, color=S.INK2)
                continue
            sd = 100 * st["std"] / ref
            ax.errorbar(st["gmean"], y, xerr=sd, fmt=S.ABL_MARKER[m], color=S.ABL_COLOR[m], ms=5, mew=0.6,
                        capsize=2, elinewidth=0.9, zorder=3)
            ax.plot(st["gbest"], y, marker="|", color=S.ABL_COLOR[m], ms=9, mew=1.4, zorder=3, ls="none")
        ax.axvline(0, color=S.INK, lw=0.8, ls="--")
        ax.set_title(inst, pad=3)
        ax.set_xlabel("Gap to reference (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([S.LABEL[m] if m != "ours" else "Ours (full)" for m in S.ABL])
    h = [Line2D([], [], marker="o", color=S.INK2, ls="none", ms=4, label="mean"),
         Line2D([], [], color=S.INK2, lw=0.9, label="± 1 std"),
         Line2D([], [], marker="|", color=S.INK2, ls="none", ms=8, mew=1.4, label="best of 10")]
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.06))
    fig.tight_layout(w_pad=0.8)
    S.save(fig, os.path.join(OUT, "fig_ablation"))
    plt.close(fig)


# ---------------------------------------------------------------- 5. Stage 2 effect
def stage2_pairs(method, inst):
    out = []
    for r in D.by(RS, method, inst):
        me = r.get("meta", {})
        if r.get("feasible") and me.get("stage1_cost") is not None:
            out.append((me["stage1_cost"], r["cost"]))
    return np.array(out)


def fig_stage2():
    fig, ax = plt.subplots(figsize=(S.COL1, 2.5))
    rng = np.random.default_rng(1)
    variants = [("ours_s2", "One SOCP polish (Stage 1 + polish)", "#6b6a66", "^"),
                ("ours", "Full Stage 2 (Ours)", S.COLOR["ours"], "o")]
    for k, inst in enumerate(S.INSTANCES):
        for j, (m, lab, col, mk) in enumerate(variants):
            P = stage2_pairs(m, inst)
            imp = 100 * (P[:, 0] - P[:, 1]) / P[:, 0]
            imp = np.maximum(imp, 0.0)
            x = k + (j - 0.5) * 0.36 + rng.uniform(-0.08, 0.08, len(imp))
            ax.scatter(x, imp, s=11, marker=mk, facecolor=col, edgecolor="white" if m == "ours" else col,
                       lw=0.3, zorder=3, label=lab if k == 0 else None)
            ax.plot([k + (j - 0.5) * 0.36 - 0.13, k + (j - 0.5) * 0.36 + 0.13], [np.median(imp)] * 2,
                    color=S.INK, lw=1.0, zorder=4)
    ax.set_yscale("symlog", linthresh=1e-3, linscale=0.5)
    ax.set_ylim(-1e-4, 20)
    ax.yaxis.set_major_locator(FixedLocator([0, 1e-3, 1e-2, 1e-1, 1, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: "0" if v == 0 else f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(S.INSTANCES))); ax.set_xticklabels(S.INSTANCES)
    ax.set_xlim(-0.6, len(S.INSTANCES) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Cost reduction by Stage 2 (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, handletextpad=0.2, borderaxespad=0.1)
    fig.tight_layout()
    S.save(fig, os.path.join(OUT, "fig_stage2"))
    plt.close(fig)


if __name__ == "__main__":
    fig_gap_box()
    miss, minfo = fig_convergence()
    print("convergence: MINLP final incumbent gap (%):", minfo)
    info = fig_layouts()
    fig_ablation()
    fig_stage2()
    print("convergence: methods without anytime traces:", miss)
    print("layouts:", info)
