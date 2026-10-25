"""Ablation: Stage-2 moves pick the partner department uniformly at random (no nearest-neighbour bias,
p_near = 0 instead of 0.8); otherwise as ours."""
from bench.ours_core import run_ours


def solve(prob, seed, time_limit, log=print):
    return run_ours(prob, seed, time_limit, log, p_near=0.0)
