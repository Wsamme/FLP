#!/usr/bin/env bash
# Post-hoc robustness variant (chosen AFTER seeing rowsF's test-set failures, so reported separately):
# rowsE = ejection + 2 % step also for row slack. Already run on the tuning set (results_ablate2, 110/110 feasible,
# +0.5 % vs rowsF there, not significant). Test set, 60 s, 20 seeds.
PY=${PY:-python}; OUT=${OUT:-bench_lean/results_final}
$PY -X utf8 -m bench_lean.run_bench "$OUT" 60 12 --set test --seeds 0-19 \
  --cfg "rowsE_post:rows:elite=50,polish=0.25,eject=0.1,slack_step=true"
