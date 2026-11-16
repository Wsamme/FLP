#!/usr/bin/env bash
# Final experiment, third attempt: rows with the ejection move (rowsF). Test set; runs already in OUT are skipped.
#   60 s : 20 seeds for rowsF, reft (ref, start search timed), legal (classic + legalise), classic
#   300 s: 10 seeds for rowsF_300, reft_300
# rowsC (85 runs, before the start fix) and rowsC2 (84 runs, start fix + slack step) in OUT are kept as a record.
PY=${PY:-python}; OUT=${OUT:-bench_lean/results_final}
OPTS="elite=50,polish=0.25,eject=0.1,slack_step=false"
$PY -X utf8 -m bench_lean.run_bench "$OUT" 60 12 --set test --seeds 0-19 \
  --cfg "rowsF:rows:$OPTS" --cfg reft:ref:count_init=1 --cfg legal:legal --cfg classic:classic
$PY -X utf8 -m bench_lean.run_bench "$OUT" 300 12 --set test --seeds 0-9 \
  --cfg "rowsF_300:rows:$OPTS" --cfg reft_300:ref:count_init=1
