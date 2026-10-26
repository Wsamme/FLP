#!/bin/bash
# Experiment 1b: re-implemented literature methods + ours, same protocol, same machine, 2 runs in parallel.
# usage: scripts/run_bench3.sh OUTDIR "method:pkg ..."
cd "$(dirname "$0")/.."
OUT=${1:-results/bench3}; METHODS=${2:-"ours:bench.methods lsa:bench3.methods av16:bench3.methods hs:bench3.methods cro:bench.methods"}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p "$OUT/logs"
jobs=()
for inst in Du62 SC30 AB20 vC10a O7; do          # longest first
  for mp in $METHODS; do jobs+=("${mp%%:*} ${mp##*:} $inst"); done
done
printf '%s\n' "${jobs[@]}" | xargs -P 2 -L 1 bash -c \
  'python -m bench3.run --method $0 --pkg $1 --instances $2 --seeds 0 1 2 3 4 5 6 7 8 9 --out '"$OUT"' > '"$OUT"'/logs/$0_$2.log 2>&1'
echo ALL_DONE
