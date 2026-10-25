"""Gold relation tables (人工标准答案). Only eval/ may import this module.

Agent code (agent/*) must never read data/gold/ — see README §防泄题.
"""
import json
import os

GOLD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gold")

# Textbook closeness scores (方庆琯 SLP 综合接近程度): A4 E3 I2 O1 U0 X-1
SCORE = {"A": 4, "E": 3, "I": 2, "O": 1, "U": 0, "X": -1}
GRADES = "AEIOUX"


def parse_rows(rows_upper, n):
    """'2I 3I 4E ...' per row i  ->  dict {(i, j): grade} with i < j."""
    pairs = {}
    for i_str, text in rows_upper.items():
        i = int(i_str)
        for tok in text.split():
            j, g = int(tok[:-1]), tok[-1]
            if not (1 <= i < j <= n):
                raise ValueError(f"bad pair {i}-{j}")
            if g not in GRADES:
                raise ValueError(f"bad grade {g} at {i}-{j}")
            if (i, j) in pairs:
                raise ValueError(f"duplicate pair {i}-{j}")
            pairs[(i, j)] = g
    return pairs


def load_gold(case):
    with open(os.path.join(GOLD_DIR, f"{case}_gold.json"), encoding="utf-8") as f:
        d = json.load(f)
    d["pairs"] = parse_rows(d["rows_upper"], d["n"])
    d["exclude"] = {tuple(sorted(p)) for p in d.get("exclude_pairs", [])}
    return d


def full_matrix(pairs, n):
    """14x14 symmetric letter matrix, diagonal '-'."""
    m = [["-"] * n for _ in range(n)]
    for (i, j), g in pairs.items():
        m[i - 1][j - 1] = m[j - 1][i - 1] = g
    return m


def row_sums(pairs, n):
    """综合接近程度: sum of scores over each row of the full symmetric matrix."""
    s = [0] * n
    for (i, j), g in pairs.items():
        s[i - 1] += SCORE[g]
        s[j - 1] += SCORE[g]
    return s
