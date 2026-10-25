"""Ablation: Stage 1 only (STS simulated annealing of bench/ours_core.py), whole budget."""
from bench.ours_core import sts_stage, Trace


def solve(prob, seed, time_limit, log=print):
    tr = Trace()
    s1 = sts_stage(prob, seed, time_limit, log, trace=tr)
    if s1 is None:
        return {"rects": None, "meta": {"error": "no feasible layout", "trace": tr.final()}}
    return {"rects": s1["rects"].tolist(), "meta": dict(stage1_cost=s1["cost"], evals=s1["evals"], cycles=s1["cycles"],
            seq=s1["seq"].tolist(), cuts=s1["cuts"].tolist(), codes=s1["codes"].tolist(), trace=tr.final())}
