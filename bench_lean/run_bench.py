"""Run baseline methods on all bench_lean instances in parallel (one process per run, single-threaded BLAS).

Usage: python -m bench_lean.run_bench OUTDIR SECONDS WORKERS METHOD [METHOD ...]
Writes OUTDIR/{method}_{instance}_s{seed}.json (same format as bench_lean.baseline) and OUTDIR/log.txt.
"""
import os, sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)


def instances():
    return sorted(f[:-5] for f in os.listdir(os.path.join(HERE, "instances")) if f.startswith("LA-") and f.endswith(".json"))


def main():
    out, T, workers, methods = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
    jobs = [(m, n, s) for s in SEEDS for n in instances() for m in methods]
    root = os.path.dirname(HERE)

    def run(job):
        m, n, s = job
        r = subprocess.run([sys.executable, "-m", "bench_lean.baseline", m, n, str(s), T, out],
                           cwd=root, env=env, capture_output=True, text=True, encoding="utf-8")
        return (r.stdout.strip() or ("ERROR " + " ".join(map(str, job)) + " " + r.stderr.strip()[-300:]))
    t0 = time.time()
    with open(os.path.join(out, "log.txt"), "a", encoding="utf-8") as log, ThreadPoolExecutor(workers) as ex:
        log.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} T={T}s workers={workers} methods={methods} host={os.environ.get('COMPUTERNAME', '')}\n")
        for line in ex.map(run, jobs):
            print(line, flush=True); log.write(line + "\n"); log.flush()
        log.write(f"# done in {time.time() - t0:.0f} s\n")


if __name__ == "__main__":
    main()
