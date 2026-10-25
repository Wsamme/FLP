"""Ablation: elite pool of size 1 (only the best Stage-1 layout goes to Stage 2); otherwise as ours."""
from bench.ours_core import run_ours


def solve(prob, seed, time_limit, log=print):
    return run_ours(prob, seed, time_limit, log, n_elite=1)
