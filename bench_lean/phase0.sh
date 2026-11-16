#!/usr/bin/env bash
# Phase 0 of 改进规划.md: tuning-set baseline (A2-A4) and time curve (A7). Run from the repo root.
PY=${PY:-python}
$PY -X utf8 -m bench_lean.run_bench bench_lean/results_tune 60 12 --set tune --seeds 0-9 --cfg rows:rows --cfg reft:ref:count_init=1
for T in 10 300 600; do
  $PY -X utf8 -m bench_lean.run_bench bench_lean/results_tune_time $T 12 --set tune --seeds 0-4 --cfg rows_t$T:rows
done
