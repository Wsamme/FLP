# Experiment protocol (agreed 2026-10-05)

## Scope
UA-FLP only (QAP moves to an appendix at most). Static, single floor, rectangular departments,
objective = sum of pair weight x rectilinear centroid distance.

## Instance set (frozen)
| Role | Instance | Why |
|---|---|---|
| Anchor | O9 | Known optimal FBS value 241.06 (Konak et al. 2006) reproduced by our evaluator |
| Medium | vC10s | Minimum-side constraint; values reported by all three baselines |
| Hard | AB20 (Armour–Buffa weighted, Mendeley) | Asymmetric flows, 100 % fill, most literature values |
| Hard | SC35 (flows corrected from Qi Liu 2004) | Largest classic instance in the set (n = 35) |
| Spare | O7, Ba12 | Verified, used for sanity checks |

Excluded: O8 (incomplete flows, no original table), Ba14 (published optimum cannot be replayed on available data),
Du62 (not needed), gym-flp AB20 (missing cost matrix, shifted row).

"Hard" is defined by: larger n, 100 % fill (no slack), asymmetric flow, tight shape constraint.

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
