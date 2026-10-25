"""Run CRO+STS on one instance. Example: python -m baselines.cro_sts.run AB20 --grid 7 --seeds 0 1 2"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from flp.instance import load
from flp.evaluate import evaluate
from baselines.cro_sts.cro import CRO
from baselines.cro_sts import sts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('instance'); ap.add_argument('--grid', type=int, default=7)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--max-iter', type=int, default=30000); ap.add_argument('--stall', type=int, default=1000)
    ap.add_argument('--tol', type=float, default=0.0, help='relative tolerance on the aspect-ratio limit')
    ap.add_argument('--same-by', default='fitness'); ap.add_argument('--bud-tries', type=int, default=3); ap.add_argument('--pen-scale', type=float, default=1.0); ap.add_argument('--out', default='results/cro_sts'); ap.add_argument('--nodup', action='store_true'); ap.add_argument('--penalty', default='static', choices=['static', 'adaptive'])
    ap.add_argument('--dummies', type=int, default=0, help='fill empty floor area with this many dummy blocks (no flow, no shape limit), as in the Kang13 SC30 file')
    a = ap.parse_args()
    inst = load(a.instance)
    alpha = np.full(inst.n, inst.shape_value * (1 + a.tol)) if inst.shape_type == 'Rmax' else np.zeros(inst.n)
    areas, Wt = np.asarray(inst.areas, float), np.asarray(inst.w, float)
    if a.dummies:
        free = inst.W * inst.H - areas.sum(); n0 = inst.n; k = a.dummies
        areas = np.r_[areas, np.full(k, free / k)]; alpha = np.r_[alpha, np.zeros(k)]
        Wt = np.pad(Wt, ((0, k), (0, k)))
    os.makedirs(a.out, exist_ok=True)
    for s in a.seeds:
        t = time.time()
        res = CRO(areas, inst.W, inst.H, Wt, alpha, a.grid, s,
                  dict(max_iter=a.max_iter, stall=a.stall, same_by=a.same_by, bud_tries=a.bud_tries, pen_scale=a.pen_scale, nodup=a.nodup, penalty=a.penalty)).run()
        bf = res['best_feasible']
        rec = dict(instance=a.instance, grid=a.grid, seed=s, tol=a.tol, same_by=a.same_by, bud_tries=a.bud_tries, pen_scale=a.pen_scale, nodup=a.nodup, penalty=a.penalty, dummies=a.dummies, iterations=res['iterations'],
                   evaluations=res['evaluations'], levy=res['levy'], seconds=round(time.time() - t, 1))
        if bf is not None:
            R = sts.decode.py_func(bf.seq, bf.cuts, bf.codes, areas, inst.W, inst.H)[:inst.n]
            ev = evaluate(inst, [tuple(x) for x in R])
            rec.update(cost=bf.cost, check_cost=ev['cost'], feasible=ev['feasible'],
                       violations=[list(map(str, v)) for v in ev['violations']][:5],
                       seq=bf.seq.tolist(), cuts=bf.cuts.tolist(), codes=bf.codes.tolist(), rects=R.tolist())
        else:
            rec.update(cost=None)
        fn = os.path.join(a.out, f"{a.instance}_g{a.grid}_tol{a.tol}_s{s}.json")
        json.dump(rec, open(fn, 'w'))
        print(a.instance, 'grid', a.grid, 'seed', s, 'cost', rec.get('cost'), 'feasible', rec.get('feasible'),
              'iters', rec['iterations'], 'evals', rec['evaluations'], 'sec', rec['seconds'], flush=True)


if __name__ == '__main__':
    main()
