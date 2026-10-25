"""Load benchmark results for the paper scripts (read-only; no optimisation is run)."""
import glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "results", "bench")
POLISH = os.path.join(ROOT, "results", "bench_polish")   # sa+polish, cro+polish (one SOCP on each baseline run)


def runs():
    """All per-run records in results/bench and results/bench_polish (one JSON per run)."""
    out = []
    for f in sorted(glob.glob(os.path.join(BENCH, "*_s*.json")) + glob.glob(os.path.join(POLISH, "*_s*.json"))):
        d = json.load(open(f))
        d["_file"] = os.path.relpath(f, ROOT)
        out.append(d)
    return out


def by(rs, method, inst):
    return sorted([r for r in rs if r["method"] == method and r["instance"] == inst], key=lambda r: r["seed"])


def stats(rs, method, inst, ref):
    R = by(rs, method, inst)
    c = np.array([r["cost"] for r in R if r.get("feasible") and r.get("cost") is not None], float)
    s = dict(n=len(R), nf=len(c), wall=float(np.mean([r["wall"] for r in R])) if R else np.nan,
             wall_max=float(np.max([r["wall"] for r in R])) if R else np.nan)
    if len(c):
        s.update(best=c.min(), mean=c.mean(), std=c.std(ddof=1) if len(c) > 1 else 0.0, median=np.median(c),
                 worst=c.max(), gbest=100 * (c.min() - ref) / ref, gmean=100 * (c.mean() - ref) / ref,
                 gaps=100 * (c - ref) / ref, costs=c)
    return s


def instance(name):
    return json.load(open(os.path.join(ROOT, "instances", "uaflp", name + ".json")))


def best_layout(name):
    return json.load(open(os.path.join(ROOT, "results", "records", f"best_{name}.json")))
