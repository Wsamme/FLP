"""Ablation: no Stage 1.  Stage 2 (SA over sequence pairs with the exact SOCP) starts from a random sequence
pair; random SPs are drawn until one is feasible.  Whole budget to Stage 2, no budget rule."""
from bench.ours_core import run_ours


def solve(prob, seed, time_limit, log=print):
    return run_ours(prob, seed, time_limit, log, stage1=False, n_elite=1, budget_rule=False)
