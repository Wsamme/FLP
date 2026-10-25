# CRO + STS baseline (reproduction of Jeong & Chae 2026)

Re-implementation of coral reefs optimization with a slicing tree structure:
Jeong & Chae, *J. Eng. Res.* 14 (2026) 1016–1028. The authors' code is not public (Java).
This is our own Python/numba version, written from the paper text.

```
python -m baselines.cro_sts.run AB20 --grid 7 --pen-scale 0.05 --nodup --seeds 0 1 2 --out results/cro_sts
python -m baselines.cro_sts.run SC30 --grid 7 --penalty adaptive --nodup --dummies 17 --seeds 0
```

Every final layout is re-checked with `flp.evaluate` (cost and feasibility); the JSON keeps
`cost` (optimizer) and `check_cost` (evaluator).

## Taken from the paper

| Item | Value |
|---|---|
| Encoding | department sequence, cut point order, cut codes 0–3 (H, V, inverse H, inverse V) |
| Split rule | each cut divides the sub-rectangle in proportion to the two parts' areas |
| ρ0, Fb, Fa, Fd, Pd | 0.35, 0.23, 0.27, 0.38, 0.1 |
| Reef | 4×4, 7×7, 10×10 |
| Stop | 30,000 iterations or 1,000 iterations without improvement |
| Local search | iterated local search, stop after 30 failed swaps |
| Larvae settling | 3 tries |
| Lévy flight | when > 80 % of corals are the same: top 40 % moved, bottom 50 % removed, best 10 % kept |
| Runs | 30 per instance and grid |

## Not specified in the paper (our choices, marked `ASSUMPTION` / `OPTION` in the code)

1. Shape penalty. Default: static, `pen_scale × Σw × (W+H)` per unit of aspect excess
   (`--pen-scale 0.05` for the reported runs). Option `--penalty adaptive`: Tate & Smith form
   `cost + (V_feas − V_all) × ninf³`.
2. Only strictly feasible layouts (aspect ≤ α, no tolerance) are reported.
3. "Same coral" for the Lévy trigger is judged by objective value.
4. Crossover on cut codes (not a permutation): two-point for PMX/OX, halves for half crossover.
   "Random cut codes" mutation acts as a swap on the two permutation vectors.
5. Lévy step: Mantegna, β = 1.5, scaled by 0.1.
6. Budding clones get 3 settling tries.
7. `--nodup`: a larva whose objective equals an existing coral's is not settled. Not in the paper.
   Without it the reef fills with copies of the best coral, Lévy flight fires about every second
   iteration, and AB20 means are 3–5 % worse (table below).
8. Empty floor area (SC30): `--dummies 17` adds 17 blocks of area 1, no flow, no shape limit,
   as in the Kang13 data file.

## Results (2026-10-07)

### O7 and vC10a (grid 7, 30 runs, `--pen-scale 0.05 --nodup`)

| Instance | Paper best | Our best | Runs at paper value | Our mean |
|---|---|---|---|---|
| O7 | 131.69 | 131.69 | 30 / 30 | 131.69 |
| vC10a (w₉,₁₀ = 59.2) | 18,522.79 | 18,522.79 | 22 / 30 | 18,657.77 |

### AB20 (30 runs per grid, `--pen-scale 0.05 --nodup`)

| Grid | Paper best (T6) | Our best | Gap | Paper mean (T7) | Our mean | Gap |
|---|---|---|---|---|---|---|
| 4×4 | 5105.64 | 5173.45 | +1.33 % | 5521.16 | 5530.36 | +0.17 % |
| 7×7 | 4927.27 | 5049.47 | +2.48 % | 5472.09 | 5425.39 | −0.85 % |
| 10×10 | 5068.28 | 5255.38 | +3.69 % | 5432.15 | 5442.10 | +0.18 % |

The paper's 4927.27 layout (Fig. 11a), written as an STS genome, decodes in `sts.decode` to
exactly 4927.27, with departments 10 and 14 at aspect 4.008 (> 4). Under the strict limit used
here that layout is not feasible.

Ablation, same settings without `--nodup`: means 5799.93 / 5684.47 / 5561.45, bests
5448.27 / 5204.69 / 5303.44. Grid 7 without `--nodup` and without early stop (30,000 iterations,
5 runs): mean 5389.91, best 5147.01.

### SC30 (`--penalty adaptive --nodup --dummies 17`, 30 runs per grid)

| Grid | Paper best (T6) | Our best | Gap | Paper mean (T7) | Our mean | Gap |
|---|---|---|---|---|---|---|
| 4×4 | 3738.75 | 3802.73 | +1.71 % | 4142.36 | 4574.89 | +10.44 % |
| 7×7 | 3343.47 | 3799.18 | +13.63 % | 3877.63 | 4331.87 | +11.71 % |

Not reproduced: means are 10–12 % higher. Things tried that did not close the gap: static
penalty, 10,000 iterations without early stop, an aspect tolerance of 1 % and 5 %, unconditional
budding (`--bud-tries 0`). With no shape limit the same code reaches about 1545, so the gap is in
searching under α = 5 with empty space.

### Du62 (117.124 square, `--pen-scale 0.05 --nodup`, 30 runs per grid)

The source (Dunker et al.) leaves the floor size open. Jeong & Chae varied one side from 90 to
117.124 and report their results on the 117.124 square, the floor in our instance file.

| Grid | Paper best (T6) | Our best | Gap | Paper mean (T7) | Our mean | Gap |
|---|---|---|---|---|---|---|
| 4×4 | 3,709,942.08 | 3,679,927.15 | −0.81 % | 3,771,603.24 | 3,713,357.80 | −1.54 % |
| 7×7 | 3,696,118.00 | 3,659,833.47 | −0.98 % | 3,741,596.01 | 3,698,739.27 | −1.15 % |

Grid 10×10 (paper best 3,672,799.99, mean 3,712,128.24) not run yet: about 7 min per run here.
