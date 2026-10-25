"""Node ⑪: compare a predicted relation table with the textbook gold table.

Metrics (task §6.3):
1 exact agreement; 2 within-one-level on A>E>I>O>U (X must match exactly);
3 Cohen's kappa — linear-weighted on the ordinal scale A,E,I,O,U,X (X placed after U, following the textbook
  score order A4 E3 I2 O1 U0 X-1; this placement is our assumption) and unweighted (X as a separate class);
4 non-U recall / precision; 5 X recall; 6 6x6 confusion matrix (rows = gold, cols = predicted);
7 error categories (OptiMUS-style error analysis).
"""
from eval.gold import GRADES

ORD = {g: k for k, g in enumerate("AEIOU")}  # rank for within-one-level / higher-lower


def confusion(gold, pred, keys):
    m = [[0] * 6 for _ in range(6)]
    for k in keys:
        m[GRADES.index(gold[k])][GRADES.index(pred[k])] += 1
    return m


def kappa(m, weighted):
    n = len(m)
    tot = sum(map(sum, m))
    if tot == 0:
        return float("nan")
    rows = [sum(r) for r in m]
    cols = [sum(m[i][j] for i in range(n)) for j in range(n)]
    def wt(i, j):  # disagreement weight
        return abs(i - j) / (n - 1) if weighted else float(i != j)
    obs = sum(wt(i, j) * m[i][j] for i in range(n) for j in range(n)) / tot
    exp = sum(wt(i, j) * rows[i] * cols[j] for i in range(n) for j in range(n)) / tot ** 2
    return 1 - obs / exp if exp else float("nan")


def error_type(g, p):
    if g == p:
        return None
    if g == "X":
        return "漏判X"
    if p == "X":
        return "误判X"
    if p == "U":
        return "有关系判成U"
    if g == "U":
        return "U判成有关系"
    return "等级偏高" if ORD[p] < ORD[g] else "等级偏低"


def evaluate(gold_pairs, pred, exclude=()):
    """gold_pairs: {(i,j): grade}; pred: {"i-j": {"grade", "reason"?}} or {(i,j): grade}."""
    def g_of(k):
        v = pred.get(k) if k in pred else pred.get(f"{k[0]}-{k[1]}")
        return v["grade"] if isinstance(v, dict) else v
    keys = [k for k in sorted(gold_pairs) if k not in set(exclude)]
    gold = {k: gold_pairs[k] for k in keys}
    p = {k: g_of(k) for k in keys}
    n = len(keys)
    exact = sum(gold[k] == p[k] for k in keys)
    def within1(a, b):
        if a == "X" or b == "X":
            return a == b
        return abs(ORD[a] - ORD[b]) <= 1
    w1 = sum(within1(gold[k], p[k]) for k in keys)
    gold_rel = [k for k in keys if gold[k] != "U"]
    pred_rel = [k for k in keys if p[k] != "U"]
    tp = sum(1 for k in gold_rel if p[k] != "U")
    gx = [k for k in keys if gold[k] == "X"]
    px = [k for k in keys if p[k] == "X"]
    cm = confusion(gold, p, keys)
    errors = {}
    for k in keys:
        t = error_type(gold[k], p[k])
        if t:
            reason = None
            v = pred.get(f"{k[0]}-{k[1]}")
            if isinstance(v, dict):
                reason = v.get("reason")
            errors.setdefault(t, []).append({"pair": f"{k[0]}-{k[1]}", "gold": gold[k], "pred": p[k],
                                              "reason": reason})
    return {
        "n_pairs": n,
        "exact": exact / n,
        "within1": w1 / n,
        "kappa_linear": kappa(cm, True),
        "kappa_unweighted": kappa(cm, False),
        "nonU_recall": tp / len(gold_rel) if gold_rel else float("nan"),
        "nonU_precision": tp / len(pred_rel) if pred_rel else float("nan"),
        "X_recall": sum(p[k] == "X" for k in gx) / len(gx) if gx else float("nan"),
        "X_precision": sum(gold[k] == "X" for k in px) / len(px) if px else float("nan"),
        "confusion": cm,
        "error_counts": {t: len(v) for t, v in errors.items()},
        "errors": errors,
    }
