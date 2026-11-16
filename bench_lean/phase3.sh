#!/usr/bin/env bash
# Phase 3: ejection move for large departments (eject=0.1), with and without the slack step, tuning set,
# 60 s, 10 seeds. Compare with rows_s_c1 (before the start fix) and rowsC2 (start fix + slack step), same dir.
PY=${PY:-python}; OUT=${OUT:-bench_lean/results_ablate2}
$PY -X utf8 -m bench_lean.run_bench "$OUT" 60 12 --set tune --seeds 0-9 \
  --cfg rowsE:rows:elite=50,polish=0.25,eject=0.1 \
  --cfg rowsEn:rows:elite=50,polish=0.25,eject=0.1,slack_step=false
