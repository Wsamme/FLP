# Experiment protocol (agreed 2026-10-05)

## Scope
UA-FLP only (QAP moves to an appendix at most). Static, single floor, rectangular departments,
objective = sum of pair weight x rectilinear centroid distance.

## Instance set (frozen, updated 2026-10-07)
The ME5001A report uses five instances, chosen so that each tests a different feature
(Jeong & Chae 2026, Tables 2–4).

| Instance | n | Floor W x H | Shape | Feature | File status |
|---|---|---|---|---|---|
| O7 | 7 | 8.54 x 13 | Rmax 4 | small; optimum reported | verified |
| vC10a | 10 | 25 x 51 | Rmax 5 | elongated floor (2:1) | verified (pair 9–10 = 59.2, see file note) |
| AB20 | 20 | 30 x 20 (report: 2 x 3 form) | Rmax 4 | asymmetric flow | verified |
| SC30 | 30 | 15 x 12 | Rmax 5 | 17 units of empty space | flows verified against Liu 2004 |
| Du62 | 62 | 117.124 x 117.124 (square) | Rmax 4 | largest; floor size open in the source, Jeong & Chae report results on this square | candidate |

Kept in `instances/uaflp/` but not used in the report: O9 (evaluator anchor, Konak et al. 2006
optimum 241.06 replayed in `tests/test_replay.py`), vC10s, SC35, Ba12.

Decision record: 2026-10-05 set (O9, vC10s, AB20, SC35; spares O7, Ba12) replaced on 2026-10-07
by the five instances above, to match the report. Du62 uses the 117.124 square, the floor on which Jeong & Chae report their Du62 results.

## Baselines (published methods, reproduced faithfully)
| Method | Paper | Acceptance |
|---|---|---|
| FBS MIP | Konak, Kulturel-Konak, Norman & Smith 2006, ORL | same optimal value where the paper proves optimality |
| BRKGA + LP | Gonçalves & Resende 2015, EJOR 246(1) | best of 10 runs within ±1 % of the paper |
| CRO + slicing tree | Jeong & Chae 2026, J. Eng. Research 14 | best and mean of 30 runs within ±1 % |

Our own self-written SA / GA / PPO from the exploratory phase are not used as baselines.

## Fairness rules
- Same hardware and wall-clock budget per method within a comparison; report hardware.
- Parameters as published; any tuning done on separate instances (irace) and reported.
- 10–30 independent runs; report best, median, worst; Friedman + post-hoc or Wilcoxon tests.
- All layouts re-checked by `flp.evaluate`; infeasible layouts are not reported as results.
