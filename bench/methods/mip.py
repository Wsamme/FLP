"""Direct MIP for the UA-FLP, solved by SCIP through OR-Tools pywraplp (exact method, one run, seed ignored).

Model (our formulation, the classic shape-candidate / big-M disjunctive model, cf. Montreuil 1990 and
Meller, Narayanan & Vance 1999, but with discrete shapes instead of their area approximations):
  - each department i picks exactly one of K exact-area rectangles (binary z_ik): aspect ratios
    {1, 1.5, 2, 3, alpha} (those <= alpha), both orientations, kept only if they fit the floor;
    so area is exact and the aspect limit holds by construction;
  - centre (x_i, y_i) inside the floor (x_i >= w_i/2, x_i <= W - w_i/2, same for y);
  - for each pair 4 binaries left/right/below/above, at least one active, big-M = W (x) or H (y);
  - objective sum_{i<j} w_ij (dx_ij + dy_ij) with dx_ij >= |x_i - x_j| etc.
Real departments only (empty floor area is allowed, so no dummies for SC30). Time limit as given by
the runner (300 s for the exact runs). meta: status, best, bound, gap (%), model size.
Restricting shapes to K candidates makes this a heuristic for the continuous-shape problem: its bound is
the bound of the restricted model, not of the original UA-FLP.
"""
import math, time
from ortools.linear_solver import pywraplp


def solve(prob, seed, time_limit, log):
    n, W, H, A = prob.n, float(prob.W), float(prob.H), [float(a) for a in prob.areas]
    F = {(i, j): float(prob.Wt[i, j]) for i in range(n) for j in range(i + 1, n) if prob.Wt[i, j] != 0}
    s = pywraplp.Solver.CreateSolver('SCIP'); s.SetTimeLimit(int(time_limit * 1000))
    s.SetSolverSpecificParametersAsString(f"randomization/randomseedshift = {int(seed)}\n")
    shapes = []
    for i in range(n):
        al = prob.alpha[i] if prob.alpha[i] > 0 else 5.0
        ratios = sorted(set(r for r in [1, 1.5, 2, 3, al] if r <= al))
        c = []
        for r in ratios:
            w = math.sqrt(A[i] * r); h = A[i] / w
            for (ww, hh) in ([(w, h)] if r == 1 else [(w, h), (h, w)]):
                if ww <= W + 1e-9 and hh <= H + 1e-9: c.append((ww, hh))
        shapes.append(c)
    if any(not c for c in shapes):
        return {"rects": None, "meta": dict(status="no candidate shape fits the floor")}
    z = [[s.BoolVar(f'z{i}_{k}') for k in range(len(shapes[i]))] for i in range(n)]
    x = [s.NumVar(0, W, f'x{i}') for i in range(n)]; y = [s.NumVar(0, H, f'y{i}') for i in range(n)]
    w = [sum(z[i][k] * shapes[i][k][0] for k in range(len(shapes[i]))) for i in range(n)]
    h = [sum(z[i][k] * shapes[i][k][1] for k in range(len(shapes[i]))) for i in range(n)]
    for i in range(n):
        s.Add(sum(z[i]) == 1); s.Add(x[i] >= 0.5 * w[i]); s.Add(x[i] <= W - 0.5 * w[i])
        s.Add(y[i] >= 0.5 * h[i]); s.Add(y[i] <= H - 0.5 * h[i])
    for i in range(n):
        for j in range(i + 1, n):
            b = [s.BoolVar(f'b{i}_{j}_{q}') for q in range(4)]; s.Add(sum(b) >= 1)
            s.Add(x[i] + 0.5 * w[i] <= x[j] - 0.5 * w[j] + W * (1 - b[0]))
            s.Add(x[j] + 0.5 * w[j] <= x[i] - 0.5 * w[i] + W * (1 - b[1]))
            s.Add(y[i] + 0.5 * h[i] <= y[j] - 0.5 * h[j] + H * (1 - b[2]))
            s.Add(y[j] + 0.5 * h[j] <= y[i] - 0.5 * h[i] + H * (1 - b[3]))
    obj = []
    for (i, j), f in F.items():
        dx = s.NumVar(0, W, ''); dy = s.NumVar(0, H, '')
        s.Add(dx >= x[i] - x[j]); s.Add(dx >= x[j] - x[i]); s.Add(dy >= y[i] - y[j]); s.Add(dy >= y[j] - y[i])
        obj.append(f * (dx + dy))
    s.Minimize(sum(obj))
    t0 = time.time(); st = s.Solve(); t = time.time() - t0
    names = {s.OPTIMAL: 'optimal', s.FEASIBLE: 'feasible', s.INFEASIBLE: 'infeasible', s.NOT_SOLVED: 'no solution found'}
    meta = dict(status=names.get(st, str(st)), solve_time=round(t, 1), shape_vars=sum(len(c) for c in shapes),
                pair_vars=4 * n * (n - 1) // 2, constraints=s.NumConstraints(), variables=s.NumVariables(),
                fill=round(sum(A) / (W * H) * 100, 2))
    if st not in (s.OPTIMAL, s.FEASIBLE):
        return {"rects": None, "meta": meta}
    best, bound = s.Objective().Value(), s.Objective().BestBound()
    meta.update(best=best, bound=bound, gap=100 * (best - bound) / best if best else 0.0)
    rects = []
    for i in range(n):
        k = max(range(len(shapes[i])), key=lambda k: z[i][k].solution_value())
        ww, hh = shapes[i][k]
        rects.append([x[i].solution_value() - ww / 2, y[i].solution_value() - hh / 2, ww, hh])
    return {"rects": rects, "meta": meta}
