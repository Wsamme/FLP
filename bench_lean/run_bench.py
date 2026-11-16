"""Run solver configurations on a bench_lean instance set in parallel (one process per run, single-threaded BLAS).

Usage:
  python -m bench_lean.run_bench OUTDIR SECONDS WORKERS --set test|tune --seeds 0-9
         --cfg TAG:METHOD[:key=value,key=value] [--cfg ...] [--only NAME,NAME]
Example: --cfg rows:rows --cfg reft:ref:count_init=1
Writes OUTDIR/{TAG}_{instance}_s{seed}.json (bench_lean.baseline format) and appends to OUTDIR/log.txt.
Existing result files are skipped, so an interrupted batch can be restarted.
"""
import os, sys, subprocess, time, argparse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def instances(which):
    sub = {"test": "instances", "tune": "instances_tune"}[which]
    return sorted(f[:-5] for f in os.listdir(os.path.join(HERE, sub)) if f.startswith("LA-") and f.endswith(".json"))


def seeds(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-"); out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("seconds"); ap.add_argument("workers", type=int)
    ap.add_argument("--set", default="test"); ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--cfg", action="append", required=True)
    ap.add_argument("--only", default="", help="comma-separated instance names (default: all of the set)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
    cfgs = []
    for c in a.cfg:
        parts = c.split(":", 2)
        opts = parts[2].split(",") if len(parts) > 2 and parts[2] else []
        cfgs.append((parts[0], parts[1], opts))
    names = [n for n in instances(a.set) if not a.only or n in a.only.split(",")]
    jobs = [(c, n, s) for s in seeds(a.seeds) for n in names for c in cfgs]
    root = os.path.dirname(HERE)

    def run(job):
        (tag, method, opts), n, s = job
        if os.path.exists(os.path.join(a.out, f"{tag}_{n}_s{s}.json")):
            return f"skip {tag} {n} {s} (exists)"
        r = subprocess.run([sys.executable, "-m", "bench_lean.baseline", method, n, str(s), a.seconds, a.out, tag, *opts],
                           cwd=root, env=env, capture_output=True, text=True, encoding="utf-8")
        return r.stdout.strip() or f"ERROR {tag} {n} {s} {r.stderr.strip()[-400:]}"
    t0 = time.time()
    with open(os.path.join(a.out, "log.txt"), "a", encoding="utf-8") as log, ThreadPoolExecutor(a.workers) as ex:
        log.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} T={a.seconds}s workers={a.workers} set={a.set} "
                  f"seeds={a.seeds} cfg={a.cfg} host={os.environ.get('COMPUTERNAME', '')}\n")
        for line in ex.map(run, jobs):
            print(line, flush=True); log.write(line + "\n"); log.flush()
        log.write(f"# done in {time.time() - t0:.0f} s\n")


if __name__ == "__main__":
    main()
