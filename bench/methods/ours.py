"""Our method: STS simulated annealing (Stage 1) -> sequence pairs -> SA over sequence pairs with the exact
SOCP geometry of bench/socp.py (Stage 2).  See bench/ours_core.py for the design.

Stage 1 keeps an elite pool (the 4 best distinct cycle results).  Stage 2 refines each elite layout in turn
(best first), with an equal share of the remaining time, and returns the best layout found.

Budget rule: Stage 1 gets OURS_S1FRAC of the budget.  The time of one SOCP is then measured on the Stage-1
best; if the remaining time affords fewer than N_MIN = 3000 SOCP solves (Du62: ~0.1 s per solve), the Stage-2
search is skipped: Stage 1 continues from its best genome for the remaining time and Stage 2 only polishes the
final layout with one SOCP (as in ours_s2).  (Rule chosen from the Du62 runs, where 2,000-2,700 solves never
improved on the polish; N_MIN is our choice.)

Options (environment variables, for record runs; defaults are used in the benchmark):
  OURS_S1FRAC  share of the time budget for Stage 1 (default 0.5)
  OURS_ELITE   number of Stage-1 elite layouts refined in Stage 2 (default 4)
  OURS_WARM    JSON file(s, ':'-separated) with "rects" of a feasible layout; added to the start pool.
               OFF by default: without it the method reads no file and is self-contained from the seed.
               (Never used in any reported run.)
  OURS_T0      Stage-2 start temperature relative to the start cost (default 1e-2)
  OURS_FORCE_S2  if set, always run the Stage-2 search (ignore the budget rule)

meta['trace'] = [[seconds, best feasible cost], ...] (Stage-1 points at cycle ends, Stage-2 points at every
improvement, plus a final point).
"""
import os
from bench.ours_core import run_ours


def solve(prob, seed, time_limit, log=print):
    return run_ours(prob, seed, time_limit, log,
                    s1frac=float(os.environ.get("OURS_S1FRAC", 0.5)), n_elite=int(os.environ.get("OURS_ELITE", 4)),
                    T0=float(os.environ.get("OURS_T0", 1e-2)), budget_rule=os.environ.get("OURS_FORCE_S2") is None,
                    warm=[f for f in os.environ.get("OURS_WARM", "").split(":") if f])
