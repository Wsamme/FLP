"""Run one method on instances and seeds.
   python -m bench.run --method sa --instances O7 AB20 --seeds 0 1 2 [--time 20] [--out results/bench]
Each run is checked by the independent evaluator; the stored cost is the evaluator's cost."""
import argparse, importlib, time, traceback
from bench.common import problem, check, save, BUDGET, INSTANCES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--time", type=float, default=None, help="override the per-instance budget")
    ap.add_argument("--out", default="results/bench")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    mod = importlib.import_module(f"bench.methods.{a.method}")
    for name in a.instances:
        prob = problem(name)
        for s in a.seeds:
            tl = a.time if a.time is not None else BUDGET[name]
            t0 = time.time(); err = None
            try:
                out = mod.solve(prob, s, tl, print)
            except Exception:
                err = traceback.format_exc(); out = {"rects": None, "meta": {}}
            wall = time.time() - t0
            rec = dict(method=a.method + a.tag, instance=name, seed=s, time_limit=tl, wall=round(wall, 2),
                       meta=out.get("meta", {}), error=err)
            if out.get("rects") is not None:
                ev = check(prob, out["rects"])
                rec.update(cost=ev["cost"], feasible=ev["feasible"],
                           violations=[list(map(str, v)) for v in ev["violations"]][:10], rects=[list(map(float, r)) for r in out["rects"]])
            else:
                rec.update(cost=None, feasible=False, violations=["no layout"], rects=None)
            fn = save(rec, a.out)
            print(f"{rec['method']:12s} {name:6s} seed {s:2d} cost {rec['cost']} feasible {rec['feasible']} wall {wall:.1f}s -> {fn}", flush=True)


if __name__ == "__main__":
    main()
