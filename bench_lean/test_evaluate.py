"""Sanity tests for bench_lean.evaluate: start from a feasible reference layout, break one rule at a time,
and check that exactly that rule is reported. Run: python -m bench_lean.test_evaluate"""
import copy
from .evaluate import load, evaluate
from .reference import solve


def main(name="LA-M-I-s1"):
    inst = load(name); D = inst["departments"]
    base = solve(inst, 0, 2)["rects"]
    assert evaluate(inst, base)["feasible"], "reference layout must be feasible"
    idx = {d["key"]: i for i, d in enumerate(D)}
    results = []

    def case(label, expect, mutate):
        R = copy.deepcopy(base); mutate(R)
        ev = evaluate(inst, R)
        hit = {k for k, v in ev["checks"].items() if v}
        results.append((label, expect, expect in hit, sorted(hit)))

    def shrink(R):
        i = idx["rm"]; R[i][2] *= 0.9
    case("面积缩小 10%", "面积", shrink)

    def overlap(R):
        a, b = idx["tool"], idx["qc"]; R[b][0], R[b][1] = R[a][0] + 0.5, R[a][1] + 0.5
    case("两个部门叠在一起", "部门重叠", overlap)

    def aisle(R):
        i = idx["qc"]; a = inst["aisles"][0]["rect"]; R[i][1] = a[1] - 1.0
    case("部门压到主通道", "占用固定通道", aisle)

    def dock(R):
        i = idx["recv"]; R[i][0] += 5.0; R[i][1] += 5.0
    case("收货区离开装卸门", "未接装卸门", dock)

    def egress(R):
        for e in inst["exits"]: e["x"], e["y"] = 0.0, 0.0
    inst_bak = copy.deepcopy(inst)
    R = copy.deepcopy(base); egress(R)
    ev = evaluate(inst, R); inst.clear(); inst.update(inst_bak)
    results.append(("只留一个角上的出口", "疏散距离", ev["checks"]["疏散距离"] > 0, sorted(k for k, v in ev["checks"].items() if v)))

    if "paint" in idx:
        def xsep(R):
            p, w = idx["paint"], idx["weld_0"]
            R[w][0] = R[p][0] + R[p][2] + 1.0 if R[p][0] + R[p][2] + 1.0 + R[w][2] <= inst["building"]["W"] else R[p][0] - R[w][2] - 1.0
            R[w][1] = R[p][1]
        case("焊接贴着喷漆", "X 级隔离", xsep)

    ok = all(r[2] for r in results)
    for label, expect, hit, got in results:
        print(("通过" if hit else "失败"), label, "-> 期望", expect, "实际", got)
    print("全部通过" if ok else "有失败")
    return ok


if __name__ == "__main__":
    main()
