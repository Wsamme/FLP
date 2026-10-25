"""One command for the whole experiment:  python run_all.py

Order: gold check -> leak check on briefs -> relation runs (case x self_check x repeat, + one-shot control)
-> evaluation vs gold -> summary tables and black/white figures -> (fork14) compile + solver if configured.

Options:
  --first            one repeat per setting (the mid-term check run)
  --provider mock    offline pipeline test; numbers are meaningless and are written to runs_mock/, results_mock/
  --cases fork14     restrict cases
Every stage writes JSON to runs/<run_id>/ so any step can be re-run alone (OptiMUS-style staged archive).
"""
import argparse
import datetime as dt
import json
import os
import re
import statistics as st
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from agent.llm import LLM  # noqa: E402
from agent import relation_agent as ra  # noqa: E402
from agent.compile import compile_instance, weight_matrix  # noqa: E402


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def leak_check(case):
    """Brief must contain no grade letters as tokens and no decimal numbers (flow intensities)."""
    with open(os.path.join(ROOT, "data", f"{case}_brief.md"), encoding="utf-8") as f:
        t = f.read()
    probs = []
    if re.search(r"(?<![A-Za-z])[AEIOUX](?![A-Za-z])", t):
        probs.append("grade-like single letter found")
    if re.search(r"\d+\.\d+", t):
        probs.append("decimal number found")
    for bad in ("必须相邻", "不能靠近", "关系等级", "物流强度", "综合关系"):
        if bad in t:
            probs.append(f"forbidden phrase「{bad}」")
    return probs


def one_run(cfg, case, mode, rep, runs_dir):
    """mode: 'sc_off' | 'sc_on' | 'oneshot'"""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = f"{stamp}_{case}_{mode}_r{rep}"
    d = os.path.join(runs_dir, rid)
    llm = LLM(cfg["llm"])
    brief, zones = ra.load_inputs(case)
    exp = cfg["experiment"]
    dump(os.path.join(d, "state_1_prompt.json"),
         {"system": ra.SYSTEM_ONESHOT if mode == "oneshot" else ra.SYSTEM_TABLE,
          "user": ra.table_prompt(brief, zones)})
    t = ra.ask_table(llm, brief, zones, oneshot=(mode == "oneshot"), max_retries=exp["max_format_retries"],
                     save=lambda r: dump(os.path.join(d, "state_2_table.json"), r))
    final = t["pairs"]
    if final is not None and mode == "sc_on":
        sc = ra.self_check(llm, brief, zones, final, threshold=exp["confidence_threshold"],
                           workers=exp["selfcheck_workers"],
                           save=lambda r: dump(os.path.join(d, "state_3_selfcheck.json"), r))
        final = sc["pairs"]
    dump(os.path.join(d, "state_4_final.json"), {"failed": t["failed"], "pairs": final})
    meta = {"run_id": rid, "case": case, "mode": mode, "repeat": rep, "date": dt.datetime.now().isoformat(),
            "provider": cfg["llm"]["provider"], "model": cfg["llm"]["model"],
            "temperature": cfg["llm"]["temperature"], "format_retries": t["n_retries"], "failed": t["failed"],
            **llm.totals(), "calls": llm.calls}
    dump(os.path.join(d, "meta.json"), meta)
    print(f"  {rid}: failed={t['failed']} retries={t['n_retries']} calls={meta['n_calls']} "
          f"tok={meta['input_tokens']}/{meta['output_tokens']} {meta['seconds']}s")
    return d


def collect(runs_dir):
    out = []
    for rid in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, rid)
        if not os.path.exists(os.path.join(p, "meta.json")):
            continue
        meta = json.load(open(os.path.join(p, "meta.json"), encoding="utf-8"))
        fin = json.load(open(os.path.join(p, "state_4_final.json"), encoding="utf-8"))
        out.append((meta, fin))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--first", action="store_true")
    ap.add_argument("--provider")
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--skip-runs", action="store_true", help="only re-evaluate existing runs")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    if a.provider:
        cfg["llm"]["provider"] = a.provider
        if a.provider == "mock":
            cfg["llm"]["model"] = "mock"
    mock = cfg["llm"]["provider"] == "mock"
    runs_dir = os.path.join(ROOT, "runs_mock" if mock else "runs")
    res_dir = os.path.join(ROOT, "results_mock" if mock else "results")
    exp = cfg["experiment"]
    cases = a.cases or exp["cases"]

    # 1. gold check (must pass before anything is used)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "eval", "check_gold.py")], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode:
        sys.exit("gold check failed — stop")
    # 2. leak check
    for c in cases:
        probs = leak_check(c)
        if probs:
            sys.exit(f"{c} brief leak check failed: {probs}")
    if not mock and cfg["llm"]["model"] == "TO_CONFIRM":
        sys.exit("set llm.model in config.json first")

    # 3. runs
    if not a.skip_runs:
        reps = 1 if a.first else exp["repeats"]
        modes = [("sc_on" if s else "sc_off") for s in exp["self_check_settings"]]
        if exp.get("run_oneshot_control"):
            modes.append("oneshot")
        for c in cases:
            for m in modes:
                for rep in range(1, reps + 1):
                    one_run(cfg, c, m, rep, runs_dir)

    # 4. evaluate + report
    from eval.report import report
    report(collect(runs_dir), cases, res_dir, cfg, ROOT)


if __name__ == "__main__":
    main()
