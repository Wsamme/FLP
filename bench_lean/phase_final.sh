#!/usr/bin/env bash
# Final experiment of 改进规划.md §4 on the test set (10 seeds). Run from a frozen checkout.
# TAG / CFG_OPTS = the rows configuration chosen on the tuning set.
PY=${PY:-python}; OUT=${OUT:-bench_lean/results_final}; TAG=${TAG:-rowsF}; CFG_OPTS=${CFG_OPTS:-}
$PY -X utf8 -m bench_lean.run_bench "$OUT" 60 12 --set test --seeds 0-9 \
  --cfg "${TAG}:rows:${CFG_OPTS}" --cfg reft:ref:count_init=1 --cfg legal:legal --cfg classic:classic
$PY -X utf8 -m bench_lean.run_bench "$OUT" 300 12 --set test --seeds 0-9 \
  --cfg "${TAG}_300:rows:${CFG_OPTS}" --cfg reft_300:ref:count_init=1
