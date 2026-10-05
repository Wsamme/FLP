"""Load frozen UA-FLP instances from instances/uaflp/*.json."""
import json, os
from dataclasses import dataclass
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instances", "uaflp")


@dataclass
class Instance:
    name: str
    n: int
    W: float              # floor width (x), as in Goncalves & Resende 2015 Table 8
    H: float              # floor height (y)
    areas: np.ndarray     # required area of each department
    w: np.ndarray         # symmetric pair weights; objective = sum_{i<j} w_ij * rectilinear centroid distance
    shape_type: str       # "Rmax" (max aspect ratio) or "min_side"
    shape_value: float
    meta: dict


def load(name: str) -> Instance:
    with open(os.path.join(ROOT, f"{name}.json")) as f:
        d = json.load(f)
    n = d["n"]
    w = np.zeros((n, n))
    for i, j, v in d["weights"]:
        w[i - 1, j - 1] = w[j - 1, i - 1] = v
    return Instance(d["name"], n, float(d["W"]), float(d["H"]), np.array(d["areas"], float), w,
                    d["shape"]["type"], float(d["shape"]["value"]), d)


def available():
    return sorted(f[:-5] for f in os.listdir(ROOT) if f.endswith(".json"))
