"""Flexible bay structure (FBS) decoding with the conventions of Konak et al. (2006).

Konak et al. assume the long side of the facility lies along the bay sequence, bays run across it,
each bay width = (sum of areas in the bay) / (length across the bay), empty space at the far end.
Departments inside a bay are listed top to bottom.
"""


def decode(inst, bays, bays_along="long"):
    """bays: list of lists of 1-based department ids. Returns rectangles in the instance's own (W, H) frame."""
    long_is_W = inst.W >= inst.H
    along_W = long_is_W if bays_along == "long" else (bays_along == "W")
    L = inst.W if along_W else inst.H        # length along the bay sequence
    T = inst.H if along_W else inst.W        # length across each bay
    rects = [None] * inst.n
    pos = 0.0
    for bay in bays:
        s = sum(inst.areas[i - 1] for i in bay)
        bw = s / T
        top = T
        for i in bay:
            h = inst.areas[i - 1] / s * T
            u, v, du, dv = pos, top - h, bw, h          # (along, across) frame
            rects[i - 1] = (u, v, du, dv) if along_W else (v, u, dv, du)
            top -= h
        pos += bw
    if pos > L + 1e-6:
        raise ValueError(f"bays exceed facility length: {pos:.4f} > {L}")
    return rects


def parse(code):
    """'7-8|4-1-2|3-6-9-5' -> [[7, 8], [4, 1, 2], [3, 6, 9, 5]]"""
    return [[int(t) for t in b.replace("−", "-").split("-")] for b in code.split("|")]
