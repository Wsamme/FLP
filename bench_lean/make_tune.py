"""Tuning instances for bench_lean: same tiers / flow patterns / brownfield cases as the test set, other seeds.

Test set (instances/) uses seeds 1-5 and is not used for any tuning. This writes instances_tune/ with seeds 11-21.
Usage: python -m bench_lean.make_tune
"""
import os, json
from .generate import build

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "instances_tune")
PLAN = [("S", "I", 11, False), ("S", "U", 12, False), ("S", "L", 13, False),
        ("M", "I", 14, False), ("M", "U", 15, False), ("M", "L", 16, False),
        ("L", "I", 17, False), ("L", "U", 18, False), ("L", "L", 19, False),
        ("M", "I", 20, True), ("L", "U", 21, True)]


def main():
    os.makedirs(OUT, exist_ok=True)
    for tier, pat, seed, bf in PLAN:
        inst = build(tier, pat, seed, brownfield=bf)
        json.dump(inst, open(os.path.join(OUT, inst["name"] + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(inst["name"], inst["summary"], inst["building"]["W"], "x", inst["building"]["H"])


if __name__ == "__main__":
    main()
