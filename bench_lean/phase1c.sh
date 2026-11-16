#!/usr/bin/env bash
# Phase 1c: B1 (pairwise two-row coupling) and C1 (elite + true-cost polish), 2 x 2, on the tuning set.
# Needed because before the fix in this commit the pairs option was always on (name clash in Model).
PY=${PY:-python}; OUT=${OUT:-bench_lean/results_ablate2}
$PY -X utf8 -m bench_lean.run_bench "$OUT" 60 12 --set tune --seeds 0-9 \
  --cfg rows_s:rows \
  --cfg rows_p:rows:pairs=1 \
  --cfg rows_s_c1:rows:elite=50,polish=0.25 \
  --cfg rows_p_c1:rows:pairs=1,elite=50,polish=0.25
