"""Node ⑥ compile: relation table + zone table -> solver instance JSON.

- Weights: textbook closeness scores A=4 E=3 I=2 O=1 U=0 replace the flow matrix in
  objective = sum_{i<j} w_ij * rectilinear centroid distance.
- X pairs: NOT a negative weight. They become hard minimum-separation constraints (distance value is an
  assumption, see README). Whether the solver can enforce them is checked separately — compile only writes them.
- Areas come straight from the zone table (OptiMUS: numbers bypass the LLM).

Output format follows the frozen instance files of Wsamme/FLP (instances/uaflp/*.json):
  {name, n, W, H, areas, weights: [[i, j, w], ...], shape: {type, value}, objective, provenance}
plus one extra key "min_separation": [[i, j, d], ...]. Adapter to benchmark_v1 is added once that code is seen.
"""
import json

WEIGHT = {"A": 4, "E": 3, "I": 2, "O": 1, "U": 0}  # textbook scores; X handled as constraint


def weight_matrix(pairs, n):
    """pairs: {"i-j": {"grade": ...}} -> n x n symmetric list of lists, X entries = 0."""
    w = [[0] * n for _ in range(n)]
    for k, v in pairs.items():
        i, j = map(int, k.split("-"))
        val = 0 if v["grade"] == "X" else WEIGHT[v["grade"]]
        w[i - 1][j - 1] = w[j - 1][i - 1] = val
    return w


def compile_instance(name, zones_file, pairs, floor, shape, x_min_dist, source_note):
    with open(zones_file, encoding="utf-8") as f:
        zt = json.load(f)
    zones = sorted(zt["zones"], key=lambda z: z["id"])
    n = len(zones)
    if any(z.get("area") is None for z in zones):
        raise ValueError(f"{name}: zone table has no areas — cannot compile (pump14 is relation-only)")
    w = weight_matrix(pairs, n)
    weights = [[i + 1, j + 1, w[i][j]] for i in range(n) for j in range(i + 1, n) if w[i][j] > 0]
    xsep = sorted([*map(int, k.split("-")), x_min_dist] for k, v in pairs.items() if v["grade"] == "X")
    return {
        "name": name,
        "n": n,
        "W": float(floor["W"]),
        "H": float(floor["H"]),
        "areas": [float(z["area"]) for z in zones],
        "weights": weights,
        "shape": shape,
        "min_separation": xsep,
        "objective": "sum_{i<j} w_ij * (|cx_i-cx_j| + |cy_i-cy_j|), c = rectangle centroid; w from A4 E3 I2 O1 U0",
        "provenance": {"zones": zt.get("source"), "relations": source_note,
                       "x_rule": "edge-to-edge clearance >= x_min_dist (assumption, README)"},
    }
