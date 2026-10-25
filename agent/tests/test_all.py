"""Tests: gold check, format-validation edge cases, compiled weight matrix symmetry, leak guards."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.gold import load_gold, row_sums  # noqa: E402
from eval.eval_relations import evaluate, kappa  # noqa: E402
from agent.validate import validate_table, levenshtein  # noqa: E402
from agent.compile import weight_matrix, compile_instance  # noqa: E402

ZONES = {1: "原材料库", 2: "油料库", 3: "标准件外购件库", 4: "机加工车间"}


def table(rows):
    return "```json\n" + json.dumps({"pairs": rows}, ensure_ascii=False) + "\n```"


def full_rows(skip=()):
    ids = sorted(ZONES)
    return [{"i": i, "name_i": ZONES[i], "j": j, "name_j": ZONES[j], "grade": "U", "reason": "r", "confidence": 4}
            for a, i in enumerate(ids) for j in ids[a + 1:] if (i, j) not in skip]


# ---------- gold ----------
def test_fork14_gold_row_sums():
    g = load_gold("fork14")
    assert len(g["pairs"]) == 91
    assert row_sums(g["pairs"], 14) == [17, 7, 11, 18, 7, 3, 13, 21, 10, 4, 13, 7, 7, 14]


def test_pump14_gold_complete_and_46_excluded():
    g = load_gold("pump14")
    assert len(g["pairs"]) == 91 and (4, 6) in g["exclude"]
    assert row_sums(g["pairs"], 14)[3] == 12  # row 4 total in the book


# ---------- validation edge cases ----------
def test_valid_table_passes():
    pairs, errs, _ = validate_table(table(full_rows()), ZONES)
    assert errs == [] and len(pairs) == 6


def test_missing_pair():
    _, errs, _ = validate_table(table(full_rows(skip={(2, 3)})), ZONES)
    assert any("缺少 1 对" in e and "2-3" in e for e in errs)


def test_duplicate_pair():
    rows = full_rows()
    rows.append(dict(rows[0]))
    _, errs, _ = validate_table(table(rows), ZONES)
    assert any("重复" in e for e in errs)


def test_reversed_pair_is_swapped_not_duplicated():
    rows = full_rows(skip={(1, 2)})
    rows.append({"i": 2, "j": 1, "grade": "I", "reason": "r", "confidence": 3})
    pairs, errs, corr = validate_table(table(rows), ZONES)
    assert errs == [] and pairs[(1, 2)]["grade"] == "I" and any("swapped_to" in c for c in corr)


def test_wrong_name_is_corrected_by_edit_distance():
    rows = full_rows()
    rows[0]["name_j"] = "油料仓库"  # one extra char vs 油料库
    pairs, errs, corr = validate_table(table(rows), ZONES)
    assert errs == [] and corr[0]["to"] == "油料库" and corr[0]["edit_distance"] == 1


def test_name_id_conflict_is_an_error():
    rows = full_rows()
    rows[0]["name_j"] = "机加工车间"  # id 2 but name of 4
    _, errs, _ = validate_table(table(rows), ZONES)
    assert any("不一致" in e for e in errs)


def test_illegal_grade_and_bad_id():
    rows = full_rows()
    rows[0]["grade"] = "B"
    rows[1]["j"] = 99
    rows[1].pop("name_j")
    _, errs, _ = validate_table(table(rows), ZONES)
    assert any("不合法" in e for e in errs) and any("99" in e for e in errs)


def test_unparseable_reply():
    _, errs, _ = validate_table("sorry, here is the table: {broken", ZONES)
    assert errs and "JSON" in errs[0]


def test_missing_is_not_filled_with_U():
    pairs, errs, _ = validate_table(table(full_rows(skip={(1, 4)})), ZONES)
    assert errs and (1, 4) not in pairs


def test_levenshtein():
    assert levenshtein("油料库", "油料仓库") == 1 and levenshtein("", "ab") == 2


# ---------- compile ----------
def test_weight_matrix_symmetric_and_X_zero():
    g = load_gold("fork14")
    pairs = {f"{i}-{j}": {"grade": v} for (i, j), v in g["pairs"].items()}
    w = weight_matrix(pairs, 14)
    assert all(w[a][b] == w[b][a] for a in range(14) for b in range(14))
    assert w[3][4] == 4 and w[1][4] == 0  # 4-5 A -> 4; 2-5 X -> 0 (constraint, not weight)


def test_compile_instance_lists_X_pairs():
    g = load_gold("fork14")
    pairs = {f"{i}-{j}": {"grade": v} for (i, j), v in g["pairs"].items()}
    inst = compile_instance("fork14", os.path.join(ROOT, "data", "fork14_zones.json"), pairs,
                            {"W": 300, "H": 300}, {"type": "Rmax", "value": 5}, 10, "gold")
    assert sum(inst["areas"]) == 66464
    assert len(inst["min_separation"]) == 7
    assert all(i < j for i, j, _ in inst["weights"])


# ---------- metrics ----------
def test_all_U_baseline_kappa_zero_and_perfect_scores_one():
    g = load_gold("fork14")
    m = evaluate(g["pairs"], {k: "U" for k in g["pairs"]})
    assert abs(m["exact"] - 46 / 91) < 1e-12 and abs(m["kappa_unweighted"]) < 1e-12 and m["nonU_recall"] == 0
    m = evaluate(g["pairs"], dict(g["pairs"]))
    assert m["exact"] == 1 and abs(m["kappa_linear"] - 1) < 1e-12 and m["X_recall"] == 1


def test_kappa_textbook_example():
    # 2x2 example: [[20,5],[10,15]] -> po=0.7, pe=0.5 -> kappa=0.4
    cm = [[20, 5, 0, 0, 0, 0], [10, 15, 0, 0, 0, 0]] + [[0] * 6] * 4
    assert abs(kappa(cm, weighted=False) - 0.4) < 1e-12


# ---------- leak guards ----------
def test_agent_code_never_reads_gold():
    for fn in os.listdir(os.path.join(ROOT, "agent")):
        if fn.endswith(".py"):
            src = open(os.path.join(ROOT, "agent", fn), encoding="utf-8").read()
            code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
            assert "gold" not in re.sub(r'""".*?"""', "", code, flags=re.S).lower(), fn


def test_prompt_has_no_numbers_or_grades_from_data():
    from agent import relation_agent as ra
    for case in ("fork14", "pump14"):
        brief, zones = ra.load_inputs(case)
        p = ra.table_prompt(brief, zones)
        assert not re.search(r"\d+\.\d+", p)                 # no flow intensities
        assert not re.search(r"\b\d+\s*[×x]\s*\d+\b", p)     # no dimensions
        assert "66464" not in p and "17280" not in p          # no areas
