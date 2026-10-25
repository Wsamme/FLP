"""Node ⑦ bridge: run the Experiment-1 solver ("ours", two-stage SA + SOCP) on a compiled instance dict.

The solver code is NOT modified. This module only rebuilds the same Problem object that
bench/common.py:problem() builds from instances/uaflp/<name>.json, but from an in-memory dict,
then calls bench.methods.ours.solve and checks the layout with the solver repo's independent
evaluator flp.evaluate (same as bench/run.py).

Limitation (checked 2026-10-25 in bench/socp.py and bench/ours_core.py): the solver has no
minimum-separation constraint, so the "min_separation" (X pairs) list is NOT enforced here.
X gaps are only measured afterwards by x_gaps().
"""
import os
import sys
import time

import numpy as np


def _import_solver(solver_dir):
    if solver_dir not in sys.path:
        sys.path.insert(0, solver_dir)
    from flp.instance import Instance          # noqa: E402
    from flp.evaluate import evaluate          # noqa: E402
    from bench.common import Problem           # noqa: E402
    from bench.methods import ours             # noqa: E402
    return Instance, evaluate, Problem, ours


def to_problem(inst_dict, solver_dir):
    Instance, _, Problem, _ = _import_solver(solver_dir)
    n = inst_dict["n"]
    w = np.zeros((n, n))
    for i, j, v in inst_dict["weights"]:
        w[i - 1, j - 1] = w[j - 1, i - 1] = v
    inst = Instance(inst_dict["name"], n, float(inst_dict["W"]), float(inst_dict["H"]),
                    np.array(inst_dict["areas"], float), w, inst_dict["shape"]["type"],
                    float(inst_dict["shape"]["value"]), inst_dict)
    # same as bench/common.py:problem() with no dummy blocks
    areas = np.asarray(inst.areas, float)
    alpha = np.full(n, inst.shape_value) if inst.shape_type == "Rmax" else np.zeros(n)
    pen = 0.05 * np.triu(w, 1).sum() * (inst.W + inst.H)
    return Problem(inst, n, inst.W, inst.H, areas, w, alpha, n, areas.copy(), w.copy(), alpha.copy(), pen)


def solve(inst_dict, solver_dir, seed=0, time_limit=30, log=print):
    _, evaluate, _, ours = _import_solver(solver_dir)
    prob = to_problem(inst_dict, solver_dir)
    t0 = time.time()
    out = ours.solve(prob, seed, time_limit, log)
    wall = time.time() - t0
    rects = [tuple(map(float, r)) for r in out["rects"]]
    ev = evaluate(prob.inst, rects)  # independent checker (node ⑨, geometry part)
    return {"rects": rects, "feasible": ev["feasible"], "violations": ev["violations"],
            "cost_own_weights": ev["cost"], "wall": round(wall, 2), "seed": seed, "time_limit": time_limit}


def cost_with(weights_matrix, rects):
    """Objective with any symmetric weight matrix (used to score every layout with GOLD weights)."""
    c = np.array([[x + w / 2, y + h / 2] for x, y, w, h in rects])
    d = np.abs(c[:, None, 0] - c[None, :, 0]) + np.abs(c[:, None, 1] - c[None, :, 1])
    return float(np.triu(np.asarray(weights_matrix) * d, 1).sum())


def x_gaps(rects, x_pairs):
    """Edge-to-edge clearance of each X pair: max(x-gap, y-gap); <= 0 means the rectangles touch."""
    out = []
    for i, j in x_pairs:
        xi, yi, wi, hi = rects[i - 1]
        xj, yj, wj, hj = rects[j - 1]
        gx = abs((xi + wi / 2) - (xj + wj / 2)) - (wi + wj) / 2
        gy = abs((yi + hi / 2) - (yj + hj / 2)) - (hi + hj) / 2
        out.append({"pair": f"{i}-{j}", "gap_x": round(gx, 2), "gap_y": round(gy, 2), "gap": round(max(gx, gy), 2)})
    return out
