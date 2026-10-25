"""Direct CP-SAT model of the UA-FLP on an integer grid (exact method on the grid, one run).

Grid scale s = 10 for O7, vC10a, AB20, SC30 and s = 1 for Du62 (sides ~117) — our choice, following the
task spec. Floor floor(W s) x floor(H s) cells. Integer w_i, h_i >= 1 with w_i h_i >= ceil(a_i s^2)
(AddMultiplicationEquality on an auxiliary product variable), aspect h_i <= alpha w_i and w_i <= alpha h_i
(alpha integer here; written as h*den <= num*w for a rational alpha), NoOverlap2D on interval variables,
objective sum_{i<j} round(100 w_ij) (|dx_ij| + |dy_ij|) on doubled centres (2x + w), with integer
absolute-value variables (AddAbsEquality). Real departments only (no dummies; empty area allowed).
Workers = min(8, os.cpu_count()), random seed = seed. Returned rectangles are divided by s; the
runner reports the evaluator's true cost.
Note: rounding the floor down while rounding areas up can make a fully packed instance infeasible on the
grid (e.g. O7: 85 x 130 = 11,050 cells < 11,100 needed; Du62 at s = 1: 117^2 = 13,689 < 13,718). meta
gives the cell balance; the model is still sent to the solver so the status is CP-SAT's own.
"""
import math, os, time
from fractions import Fraction
from ortools.sat.python import cp_model

SCALE = {"Du62": 1}


def solve(prob, seed, time_limit, log):
    name = prob.inst.name; sc = SCALE.get(name, 10)
    n = prob.n; GW, GH = int(math.floor(prob.W * sc + 1e-9)), int(math.floor(prob.H * sc + 1e-9))
    need = [int(math.ceil(a * sc * sc - 1e-9)) for a in prob.areas]
    m = cp_model.CpModel()
    x, y, w, h, xi, yi = [], [], [], [], [], []
    for i in range(n):
        wi = m.NewIntVar(1, GW, f'w{i}'); hi = m.NewIntVar(1, GH, f'h{i}')
        xs = m.NewIntVar(0, GW, f'x{i}'); ys = m.NewIntVar(0, GH, f'y{i}')
        xe = m.NewIntVar(0, GW, f'xe{i}'); ye = m.NewIntVar(0, GH, f'ye{i}')
        p = m.NewIntVar(need[i], GW * GH, f'p{i}'); m.AddMultiplicationEquality(p, [wi, hi])
        if prob.alpha[i] > 0:
            fr = Fraction(float(prob.alpha[i])).limit_denominator(1000)
            m.Add(hi * fr.denominator <= fr.numerator * wi); m.Add(wi * fr.denominator <= fr.numerator * hi)
        xi.append(m.NewIntervalVar(xs, wi, xe, f'ix{i}')); yi.append(m.NewIntervalVar(ys, hi, ye, f'iy{i}'))
        x.append(xs); y.append(ys); w.append(wi); h.append(hi)
    m.AddNoOverlap2D(xi, yi)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            wij = int(round(100 * prob.Wt[i, j]))
            if wij == 0: continue
            dx = m.NewIntVar(-2 * GW, 2 * GW, ''); dy = m.NewIntVar(-2 * GH, 2 * GH, '')
            m.Add(dx == 2 * x[i] + w[i] - 2 * x[j] - w[j]); m.Add(dy == 2 * y[i] + h[i] - 2 * y[j] - h[j])
            ax = m.NewIntVar(0, 2 * GW, ''); ay = m.NewIntVar(0, 2 * GH, '')
            m.AddAbsEquality(ax, dx); m.AddAbsEquality(ay, dy)
            terms.append(wij * (ax + ay))
    m.Minimize(sum(terms))
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = float(time_limit)
    sv.parameters.num_search_workers = min(8, os.cpu_count() or 1)
    sv.parameters.random_seed = int(seed)
    t0 = time.time(); st = sv.Solve(m); t = time.time() - t0
    unit = 100.0 * 2 * sc          # objective units: weights x100, doubled centres, grid scale
    meta = dict(status=sv.StatusName(st), solve_time=round(t, 1), scale=sc, grid=[GW, GH],
                cells_floor=GW * GH, cells_needed=sum(need), workers=sv.parameters.num_search_workers)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"rects": None, "meta": meta}
    meta.update(best_scaled=sv.ObjectiveValue() / unit, bound_scaled=sv.BestObjectiveBound() / unit,
                gap=100 * (sv.ObjectiveValue() - sv.BestObjectiveBound()) / sv.ObjectiveValue() if sv.ObjectiveValue() else 0.0)
    rects = [[sv.Value(x[i]) / sc, sv.Value(y[i]) / sc, sv.Value(w[i]) / sc, sv.Value(h[i]) / sc] for i in range(n)]
    return {"rects": rects, "meta": meta}
