#!/usr/bin/env bash
# Final experiment of 改进规划.md §4 on the test set. Run from a frozen checkout.
# TAG / CFG_OPTS = the rows configuration chosen on the tuning set.
#   60 s : 20 seeds (0-19) for rows, ref with the start search timed (reft), classic + legalise, classic
#   300 s: 10 seeds (0-9) for rows and reft
PY=${PY:-python}; OUT=${OUT:-bench_lean/results_final}; TAG=${TAG:-rowsC}; CFG_OPTS=${CFG_OPTS:-elite=50,polish=0.25}
$PY -X utf8 -m bench_lean.run_bench "$OUT" 60 12 --set test --seeds 0-19 \
  --cfg "${TAG}:rows:${CFG_OPTS}" --cfg reft:ref:count_init=1 --cfg legal:legal --cfg classic:classic
$PY -X utf8 -m bench_lean.run_bench "$OUT" 300 12 --set test --seeds 0-9 \
  --cfg "${TAG}_300:rows:${CFG_OPTS}" --cfg reft_300:ref:count_init=1
