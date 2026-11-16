#!/usr/bin/env bash
# Phase 1b of 改进规划.md (A7): time curve of one rows configuration on the tuning set.
# CFG_OPTS = solver options of the configuration (e.g. "elite=50"), TAG = name prefix. Run from a frozen checkout.
PY=${PY:-python}; OUT=${OUT:-bench_lean/results_time}; TAG=${TAG:-rowsB}; CFG_OPTS=${CFG_OPTS:-}
$PY -X utf8 -m bench_lean.run_bench "$OUT" 10 12 --set tune --seeds 0-4 --cfg "${TAG}_t10:rows:${CFG_OPTS}"
$PY -X utf8 -m bench_lean.run_bench "$OUT" 300 12 --set tune --seeds 0-4 --cfg "${TAG}_t300:rows:${CFG_OPTS}"
$PY -X utf8 -m bench_lean.run_bench "$OUT" 600 12 --set tune --seeds 0-2 --cfg "${TAG}_t600:rows:${CFG_OPTS}"
