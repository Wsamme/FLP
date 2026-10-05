"""Independent feasibility checker and cost function for UA-FLP layouts.

A layout is a list of rectangles (x, y, w, h), one per department, with (x, y) the lower-left corner.
"""
import numpy as np

EPS = 1e-6


def cost(inst, rects):
    c = np.array([[x + w / 2, y + h / 2] for x, y, w, h in rects])
    d = np.abs(c[:, None, 0] - c[None, :, 0]) + np.abs(c[:, None, 1] - c[None, :, 1])
    return float(np.triu(inst.w * d, 1).sum())


def violations(inst, rects, eps=EPS):
    v = []
    for i, (x, y, w, h) in enumerate(rects):
        if w * h < inst.areas[i] * (1 - eps):
            v.append(("area", i + 1, w * h, inst.areas[i]))
        if inst.shape_type == "Rmax" and max(w / h, h / w) > inst.shape_value + eps:
            v.append(("aspect", i + 1, max(w / h, h / w)))
        if inst.shape_type == "min_side" and min(w, h) < inst.shape_value - eps:
            v.append(("min_side", i + 1, min(w, h)))
        if x < -eps or y < -eps or x + w > inst.W + eps or y + h > inst.H + eps:
            v.append(("outside", i + 1))
        for j in range(i):
            a = rects[j]
            if min(x + w, a[0] + a[2]) - max(x, a[0]) > eps and min(y + h, a[1] + a[3]) - max(y, a[1]) > eps:
                v.append(("overlap", j + 1, i + 1))
    return v


def evaluate(inst, rects):
    v = violations(inst, rects)
    return {"cost": cost(inst, rects), "feasible": not v, "violations": v}
