"""Ablation: Stage 1 (STS simulated annealing) for the whole budget, then ONE Stage-2 SOCP on the
relations of the Stage-1 layout (its sequence pair).  No search in Stage 2."""
import numpy as np
from bench.ours_core import sts_stage, Trace
from bench.socp import from_problem, rects_to_sp


def solve(prob, seed, time_limit, log=print):
    tr = Trace()
    q = from_problem(prob)
    # keep a small reserve for the polish (one SOCP; ~0.1 s on Du62)
    s1 = sts_stage(prob, seed, max(time_limit - 0.5, 0.0), log, trace=tr)
    if s1 is None:
        return {"rects": None, "meta": {"error": "no feasible layout", "trace": tr.final()}}
    R = np.asarray(s1["rects"])[:prob.n]
    o = q.solve_sp(*rects_to_sp(R))
    if o["rects"] is None or not np.isfinite(o["cost"]) or o["cost"] > s1["cost"]:
        log(f"  Stage 2 polish not used ({o['status']}), keeping Stage 1")
        return {"rects": R.tolist(), "meta": dict(stage1_cost=s1["cost"], polish=o["status"], trace=tr.final())}
    tr.add(o["cost"])
    log(f"  Stage 2 polish {s1['cost']:.4f} -> {o['cost']:.4f}")
    return {"rects": o["rects"].tolist(), "meta": dict(stage1_cost=s1["cost"], stage2_cost=o["cost"], evals=s1["evals"],
            trace=tr.final())}
