#!/usr/bin/env bash
# Final experiment, second attempt (after the start-placement fix for the paint department).
# 1) tuning-set check of the fixed solver (compare with rows_s_c1 in results_ablate2)
# 2) test set 60 s, 20 seeds: rowsC2 + the baselines (runs already in OUT are skipped)
# 3) test set 300 s, 10 seeds: rowsC2_300 and reft_300
# The 85 runs tagged rowsC in results_final are from the solver before the fix (kept as a record).
PY=${PY:-python}; OUT=${OUT:-bench_lean/results_final}; TUNE_OUT=${TUNE_OUT:-bench_lean/results_ablate2}
OPTS="elite=50,polish=0.25"
$PY -X utf8 -m bench_lean.run_bench "$TUNE_OUT" 60 12 --set tune --seeds 0-9 --cfg "rowsC2:rows:$OPTS"
$PY -X utf8 -m bench_lean.run_bench "$OUT" 60 12 --set test --seeds 0-19 \
  --cfg "rowsC2:rows:$OPTS" --cfg reft:ref:count_init=1 --cfg legal:legal --cfg classic:classic
$PY -X utf8 -m bench_lean.run_bench "$OUT" 300 12 --set test --seeds 0-9 \
  --cfg "rowsC2_300:rows:$OPTS" --cfg reft_300:ref:count_init=1
