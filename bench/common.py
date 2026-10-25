"""Shared benchmark plumbing. Every method module in bench/methods/ exposes

    solve(prob: Problem, seed: int, time_limit: float, log) -> dict

returning at least {"rects": [[x, y, w, h], ...] (one per REAL department, lower-left corner),
"meta": {...}}.  The runner (bench/run.py) times the call, checks the layout with the
independent evaluator flp.evaluate (never the method's own cost), and writes one JSON per run.

Problem fields: inst (flp.instance.Instance), n, W, H, areas, Wt (symmetric weights, cost = sum_{i<j}),
alpha (per-department max aspect, 0 = unconstrained), and, for instances with empty floor area,
the STS view with dummy blocks (areas_d, Wt_d, alpha_d, n_d), as used for CRO in Jeong & Chae 2026.
"""
import json, os, sys, time
from dataclasses import dataclass, field
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flp.instance import load          # noqa: E402
from flp.evaluate import evaluate     # noqa: E402

INSTANCES = ["O7", "vC10a", "AB20", "SC30", "Du62"]
# wall-clock budget per run (seconds), the same for every stochastic method
BUDGET = {"O7": 20, "vC10a": 20, "AB20": 60, "SC30": 60, "Du62": 150}
DUMMIES = {"SC30": 17}   # empty area split into 17 unit blocks, as in the Kang13 data file


@dataclass
class Problem:
    inst: object
    n: int
    W: float
    H: float
    areas: np.ndarray
    Wt: np.ndarray
    alpha: np.ndarray
    n_d: int = 0
    areas_d: np.ndarray = None
    Wt_d: np.ndarray = None
    alpha_d: np.ndarray = None
    pen: float = 0.0          # static penalty per unit aspect excess: 0.05 * sum(w) * (W + H)
    meta: dict = field(default_factory=dict)


def problem(name):
    inst = load(name)
    areas = np.asarray(inst.areas, float); Wt = np.asarray(inst.w, float)
    alpha = np.full(inst.n, inst.shape_value) if inst.shape_type == "Rmax" else np.zeros(inst.n)
    k = DUMMIES.get(name, 0)
    if k:
        free = inst.W * inst.H - areas.sum()
        areas_d = np.r_[areas, np.full(k, free / k)]; alpha_d = np.r_[alpha, np.zeros(k)]
        Wt_d = np.pad(Wt, ((0, k), (0, k)))
    else:
        areas_d, alpha_d, Wt_d = areas.copy(), alpha.copy(), Wt.copy()
    pen = 0.05 * np.triu(Wt, 1).sum() * (inst.W + inst.H)
    return Problem(inst, inst.n, inst.W, inst.H, areas, Wt, alpha, len(areas_d), areas_d, Wt_d, alpha_d, pen)


def check(prob, rects):
    ev = evaluate(prob.inst, [tuple(map(float, r)) for r in rects])
    return ev


def save(rec, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{rec['method']}_{rec['instance']}_s{rec['seed']}.json")
    with open(fn, "w") as f:
        json.dump(rec, f)
    return fn
