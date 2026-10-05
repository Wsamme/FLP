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
    for name in ["O7", "O9", "Ba12", "vC10s", "SC35", "AB20"]:
        inst = load(name)
        assert inst.areas.sum() <= inst.W * inst.H + 1e-6
        assert np.allclose(inst.w, inst.w.T)


def test_sc35_corrections_applied():
    w = load("SC35").w
    assert w[4, 5] == 8.92 and w[5, 12] == 0 and w[5, 16] == 0.61 and w[14, 31] == 53.42 and w[15, 31] == 1.83


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("PASS", k)
