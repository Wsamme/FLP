#!/usr/bin/env bash
# Phase 2: tune the chosen configuration (single split + elite 50 + polish) on the tuning set, 60 s, 10 seeds.
# Baseline for comparison: rows_s_c1 in results_ablate2 (same options, same seeds).
PY=${PY:-python}; OUT=${OUT:-bench_lean/results_phase2}
$PY -X utf8 -m bench_lean.run_bench "$OUT" 60 12 --set tune --seeds 0-9 \
  --cfg rowsC_near5:rows:elite=50,polish=0.25,near=0.5 \
  --cfg rowsC_pol10:rows:elite=50,polish=0.1 \
  --cfg rowsC_pol40:rows:elite=50,polish=0.4
