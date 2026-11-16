#!/usr/bin/env bash
# Phase 1a of 改进规划.md: ablation of the first improvements on the tuning set (60 s, seeds 0-9).
# Run from a frozen checkout (git worktree at a fixed commit); OUT is where results go.
PY=${PY:-python}; OUT=${OUT:-bench_lean/results_ablate}
$PY -X utf8 -m bench_lean.run_bench "$OUT" 60 12 --set tune --seeds 0-9 \
  --cfg rows0:rows \
  --cfg rows_e50:rows:elite=50 \
  --cfg rows_c1:rows:elite=50,polish=0.25 \
  --cfg rows_b1:rows:pairs=1 \
  --cfg rows_tacc:rows:accept=true \
  --cfg rows_b1e50:rows:pairs=1,elite=50
