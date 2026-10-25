"""Node ⑤ format validation.

Sources:
- Holodeck: wrong object names are mapped to the nearest legal name by edit distance.
- Co-Layout: if the JSON cannot be parsed, ask again.
- OptiMUS: the concrete error message is sent back to the model.
A table that still fails after the allowed retries is recorded as FAILED — it is never silently filled with U.
"""
import json
import re

GRADES = set("AEIOUX")


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def nearest_name(name, legal):
    """legal: {id: name}. Returns (id, legal_name, distance)."""
    best = min(legal.items(), key=lambda kv: (levenshtein(name, kv[1]), kv[0]))
    return best[0], best[1], levenshtein(name, best[1])


def extract_json(text):
    """Take the first ```json block if any, else the outermost {...}. Raises ValueError."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    if not raw:
        raise ValueError("no JSON object found in the reply")
    return json.loads(raw)


def validate_table(text, zones, need_reason=True):
    """zones: {id: name}. Returns (pairs_dict, errors, corrections).

    pairs_dict: {(i, j): {"grade", "reason", "confidence"}} with i < j; only meaningful if errors == [].
    """
    errors, corrections = [], []
    try:
        d = extract_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        return {}, [f"JSON 解析失败: {e}"], corrections
    rows = d.get("pairs") if isinstance(d, dict) else None
    if not isinstance(rows, list):
        return {}, ['顶层必须是 {"pairs": [...]}'], corrections

    ids = sorted(zones)
    out = {}
    for k, r in enumerate(rows):
        if not isinstance(r, dict):
            errors.append(f"第 {k + 1} 条不是对象")
            continue
        end = []
        for side in ("i", "j"):
            zid, nm = r.get(side), r.get(f"name_{side}")
            if isinstance(nm, str) and nm and zones.get(zid) != nm:
                nid, legal, dist = nearest_name(nm, zones)
                if zid in zones and nid != zid:
                    errors.append(f"第 {k + 1} 条: 编号 {zid} 与名称「{nm}」不一致（最近的合法名称是 {nid}「{legal}」）")
                    zid = None
                else:
                    corrections.append({"row": k + 1, "given": nm, "to": legal, "id": nid, "edit_distance": dist})
                    zid = nid
            if zid not in zones:
                errors.append(f"第 {k + 1} 条: 编号 {r.get(side)} 不在区域表里（合法编号 {ids[0]}–{ids[-1]}）")
                zid = None
            end.append(zid)
        i, j = end
        if i is None or j is None:
            continue
        if i == j:
            errors.append(f"第 {k + 1} 条: i 与 j 相同 ({i})")
            continue
        if i > j:
            i, j = j, i
            corrections.append({"row": k + 1, "swapped_to": [i, j]})
        g = str(r.get("grade", "")).strip().upper()
        if g not in GRADES:
            errors.append(f"{i}-{j}: 等级「{r.get('grade')}」不合法，只能是 A/E/I/O/U/X")
            continue
        conf = r.get("confidence")
        if need_reason:
            if not isinstance(conf, int) or not 1 <= conf <= 5:
                errors.append(f"{i}-{j}: confidence 必须是 1–5 的整数")
                continue
            if not str(r.get("reason", "")).strip():
                errors.append(f"{i}-{j}: 缺少 reason")
                continue
        if (i, j) in out:
            errors.append(f"{i}-{j}: 重复出现")
            continue
        out[(i, j)] = {"grade": g, "reason": str(r.get("reason", "")), "confidence": conf}

    missing = [(i, j) for a, i in enumerate(ids) for j in ids[a + 1:] if (i, j) not in out]
    if missing:
        errors.append(f"缺少 {len(missing)} 对: " + ", ".join(f"{i}-{j}" for i, j in missing[:40])
                      + (" ..." if len(missing) > 40 else ""))
    return out, errors, corrections
