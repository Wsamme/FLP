"""Evaluator validation: recompute objective values of layouts published in the literature."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from flp.instance import load
from flp.evaluate import evaluate
from flp import fbs


def test_konak2006_O9_optimal_fbs():
    # Konak et al. (2006), Table 3 p.10: O9 (alpha = 5) optimal FBS solution, objective 241.06
    inst = load("O9")
    inst.shape_value = 5
    r = evaluate(inst, fbs.decode(inst, fbs.parse("7-8|4-1-2|3-6-9-5")))
    assert r["feasible"], r["violations"]
    assert round(r["cost"], 2) == 241.06


def test_instance_area_and_floor():
    for name in ["O7", "O9", "Ba12", "vC10s", "SC35", "AB20", "vC10a", "SC30", "Du62"]:
        inst = load(name)
        assert inst.areas.sum() <= inst.W * inst.H + 1e-6
        assert np.allclose(inst.w, inst.w.T)


def test_sc35_corrections_applied():
    w = load("SC35").w
    assert w[4, 5] == 8.92 and w[5, 12] == 0 and w[5, 16] == 0.61 and w[14, 31] == 53.42 and w[15, 31] == 1.83


def test_ab20_report_form_matches():
    # The report's AB20 (2 x 3 form) column layout, scaled x10 and rotated into this 30 x 20 frame,
    # must give the same cost as computed in the report (10741.20).
    inst = load("AB20")
    a = inst.areas / 100.0
    cols = [range(0, 10), range(10, 16), range(16, 20)]
    rects = [None] * 20
    x = 0.0
    for col in cols:
        w = sum(a[i] for i in col) / 3.0
        y = 0.0
        for i in col:
            h = a[i] / w
            rects[i] = (y * 10, x * 10, h * 10, w * 10)   # (x, y) of 2x3 frame -> swapped, scaled
            y += h
        x += w
    r = evaluate(inst, rects)
    assert r["feasible"], r["violations"]
    assert round(r["cost"], 2) == 10741.20


def test_jeong2026_ab20_fig11a():
    # Jeong & Chae (2026), Fig. 11a: best AB20 layout, reported objective 4927.27 (2 x 3 form).
    # Slicing structure read from the figure; all sizes follow from the areas. Built in the 2 x 3 frame,
    # then scaled x10 and rotated into this 30 x 20 frame.
    inst = load("AB20")
    a = {i + 1: inst.areas[i] / 100.0 for i in range(20)}
    W, R = 2.0, {}
    def row(ids, y0, width=W):
        h = sum(a[i] for i in ids) / width; x = 0.0
        for i in ids:
            R[i] = (x, y0, a[i] / h, h); x += a[i] / h
        return h
    def col(ids, x0, y_top, width):
        y = y_top
        for i in ids:
            h = a[i] / width; R[i] = (x0, y - h, width, h); y -= h
    hA = (a[16] + a[11]) / W; yA = 3 - hA; row((16, 11), yA)
    hB = sum(a[i] for i in (17, 12, 9, 13, 15, 14, 10)) / W; yB = yA - hB
    wl = (a[17] + a[12]) / hB; col((17, 12), 0, yA, wl)
    wr = W - wl; hb = (a[9] + a[13] + a[15]) / wr; w9 = a[9] / hb
    R[9] = (wl, yA - hb, w9, hb); col((13, 15), wl + w9, yA, wr - w9); col((14, 10), wl, yA - hb, wr)
    hC = (a[19] + a[3]) / W; yC = yB - hC; row((19, 3), yC)
    hD = yC; w18 = a[18] / hD; R[18] = (0, 0, w18, hD)
    w51 = (a[5] + a[1]) / hD; col((5, 1), W - w51, hD, w51)
    wm = W - w18 - w51; ht = sum(a[i] for i in (4, 2, 7, 8, 6)) / wm
    w4, w278 = a[4] / ht, (a[2] + a[7] + a[8]) / ht
    R[4] = (w18, hD - ht, w4, ht); col((2, 7, 8), w18 + w4, hD, w278); R[6] = (w18 + w4 + w278, hD - ht, a[6] / ht, ht)
    R[20] = (w18, 0, wm, hD - ht)
    rects = [(R[i][1] * 10, R[i][0] * 10, R[i][3] * 10, R[i][2] * 10) for i in range(1, 21)]
    r = evaluate(inst, rects)
    assert round(r["cost"], 2) == 4927.27
    # the published layout exceeds Rmax = 4 slightly (4.008) for departments 10 and 14 only
    assert sorted(v[1] for v in r["violations"]) == [10, 14] and all(v[0] == "aspect" for v in r["violations"])


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("PASS", k)
