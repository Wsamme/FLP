"""第 3 节核对：gold 表完整、对称、行合计等于书中"综合接近程度"。

Run:  python eval/check_gold.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.gold import load_gold, full_matrix, row_sums  # noqa: E402


def check(case):
    g = load_gold(case)
    n, pairs = g["n"], g["pairs"]
    report = {"case": case, "ok": True, "msgs": []}
    need = n * (n - 1) // 2
    if len(pairs) != need:
        report["ok"] = False
        report["msgs"].append(f"pairs {len(pairs)} != {need}")
    m = full_matrix(pairs, n)
    sym = all(m[a][b] == m[b][a] for a in range(n) for b in range(n))
    report["msgs"].append(f"symmetric: {sym}")
    report["ok"] &= sym
    sums = row_sums(pairs, n)
    report["row_sums"] = sums
    exp = g.get("expected_row_sums")
    if exp:
        bad = [(k + 1, sums[k], exp[k]) for k in range(n) if sums[k] != exp[k]]
        report["ok"] &= not bad
        report["msgs"].append("row sums match book" if not bad else f"row-sum mismatch (row, ours, book): {bad}")
    part = g.get("expected_row_sums_partial")
    if part:
        for r, v in part.items():
            report["msgs"].append(f"row {r}: ours {sums[int(r) - 1]}, book {v}")
        report["msgs"].append("full book row sums not on file -> only partial check")
    for note in g.get("notes", []):
        report["msgs"].append("NOTE " + note)
    counts = {}
    for gr in pairs.values():
        counts[gr] = counts.get(gr, 0) + 1
    report["grade_counts"] = dict(sorted(counts.items(), key=lambda kv: "AEIOUX".index(kv[0])))
    return report


if __name__ == "__main__":
    all_ok = True
    for case in ("fork14", "pump14"):
        r = check(case)
        print(f"== {case}: {'PASS' if r['ok'] else 'FAIL'}")
        print("   row sums:", r["row_sums"])
        print("   grade counts:", r["grade_counts"])
        for msg in r["msgs"]:
            print("  ", msg)
        all_ok &= r["ok"]
    sys.exit(0 if all_ok else 1)
