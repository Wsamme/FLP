"""Nodes ② (relation agent) and ③ (confidence self-check), plus the one-shot control prompt.

Design sources (do not change without asking the user):
- Grades A/E/I/O/U/X: Muther, Systematic Layout Planning (SLP).
- One line per pair, every pair listed: Holodeck.
- One agent does one job (relations only, no geometry): Co-Layout.
- Numbers (areas, sizes, flow intensities) never enter the prompt: OptiMUS.
- Per-item confidence and re-check of low-confidence items: OptiMUS (no ablation there -> we run on/off).

Leak guard: this module reads only data/<case>_brief.md and the id/name columns of data/<case>_zones.json.
It must never open data/gold/.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

from agent.validate import validate_table, extract_json, GRADES

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

GRADE_DEF = """等级定义（SLP 系统布置设计，Muther）：
A：绝对必要靠近
E：特别重要靠近
I：重要
O：一般
U：不重要（两者之间没有需要考虑的关系）
X：不希望靠近

常见判断依据（可同时考虑多条）：
1. 物料流动：工艺上前后衔接、物料搬运往来
2. 共用设备、场地或人员
3. 管理与服务联系：监督管理、辅助服务、人员往来
4. 安全、污染、噪声、振动：需要隔开时判 X
大多数单位对之间没有明显关系，判 U 是正常的。"""

SYSTEM_TABLE = f"""你是工厂布置中的作业单位相互关系分析员。你只做一件事：根据工厂说明，判断每一对作业单位之间的接近程度等级，并用一句话写出理由。不需要考虑面积、坐标或距离。

{GRADE_DEF}

输出要求：
- 只输出一个 ```json 代码块，格式为 {{"pairs": [{{"i": 1, "name_i": "...", "j": 2, "name_j": "...", "grade": "U", "reason": "一句话理由", "confidence": 4}}, ...]}}
- 每一对单位都要给出，只列 i<j，先按 i 再按 j 排序。
- reason：一句中文，说明依据。
- confidence：1–5 的整数，5 表示很确定，1 表示几乎是猜的。"""

SYSTEM_ONESHOT = f"""你是工厂布置中的作业单位相互关系分析员。根据工厂说明，直接给出每一对作业单位之间的接近程度等级。

{GRADE_DEF}

输出要求：只输出一个 ```json 代码块，格式为 {{"pairs": [{{"i": 1, "j": 2, "grade": "U"}}, ...]}}，每一对都要给出，只列 i<j。不要写理由。"""

SYSTEM_PAIR = f"""你是工厂布置中的作业单位相互关系分析员。现在只复核一对作业单位之间的接近程度等级。

{GRADE_DEF}

输出要求：只输出一个 ```json 代码块，格式为 {{"i": 1, "j": 2, "grade": "U", "reason": "一句话理由", "confidence": 4}}。"""


def load_inputs(case):
    """Node ①: the prompt gets ids + names + brief only. Areas/dimensions stay in the zones file for compile."""
    with open(os.path.join(DATA, f"{case}_brief.md"), encoding="utf-8") as f:
        brief = f.read()
    with open(os.path.join(DATA, f"{case}_zones.json"), encoding="utf-8") as f:
        zones = {z["id"]: z["name"] for z in json.load(f)["zones"]}  # OptiMUS: numbers do not enter the prompt
    return brief, zones


def zone_list(zones):
    return "\n".join(f"- [{i}] {n}" for i, n in sorted(zones.items()))


def table_prompt(brief, zones):
    n = len(zones)
    return (f"# 工厂说明\n\n{brief}\n\n# 作业单位清单（共 {n} 个）\n{zone_list(zones)}\n\n"
            f"请给出全部 {n * (n - 1) // 2} 对的等级。")


def feedback_message(errors):
    """OptiMUS: the concrete validation errors are sent back verbatim."""
    return ("你的输出没有通过格式检查，问题如下：\n- " + "\n- ".join(errors)
            + "\n请输出完整的修正后的 JSON（全部单位对）。")


def recheck_prompt(brief, zones, i, j, prev):
    return (f"# 工厂说明\n\n{brief}\n\n# 需要复核的一对\nPAIR [{i}] {zones[i]} 与 [{j}] {zones[j]}\n\n"
            f"你之前的判断：等级 {prev['grade']}，理由「{prev['reason']}」，置信度 {prev['confidence']}。\n"
            f"请独立重新判断这一对，确认或修改。")


def ask_table(llm, brief, zones, oneshot=False, max_retries=2, save=None):
    """Node ② (+ node ⑤ loop). Returns dict with pairs or failed=True."""
    system = SYSTEM_ONESHOT if oneshot else SYSTEM_TABLE
    messages = [{"role": "user", "content": table_prompt(brief, zones)}]
    attempts = []
    for attempt in range(max_retries + 1):
        reply = llm.chat(system, messages, tag=f"table_try{attempt}")
        pairs, errors, corr = validate_table(reply, zones, need_reason=not oneshot)
        attempts.append({"attempt": attempt, "errors": errors, "corrections": corr, "reply": reply})
        if not errors:
            break
        # OptiMUS: send the concrete error message back; Co-Layout: retry on parse failure
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": feedback_message(errors)}]
    result = {"system": system, "first_user": messages[0]["content"], "attempts": attempts,
              "failed": bool(errors), "n_retries": len(attempts) - 1,
              "pairs": None if errors else {f"{i}-{j}": v for (i, j), v in sorted(pairs.items())}}
    if save:
        save(result)
    return result


def _recheck_one(llm, brief, zones, i, j, prev):
    msgs = [{"role": "user", "content": recheck_prompt(brief, zones, i, j, prev)}]
    for attempt in range(2):
        reply = llm.chat(SYSTEM_PAIR, msgs, tag=f"recheck_{i}-{j}")
        try:
            d = extract_json(reply)
            g = str(d.get("grade", "")).strip().upper()
            c = d.get("confidence")
            if g in GRADES and isinstance(c, int) and 1 <= c <= 5:
                return {"grade": g, "reason": str(d.get("reason", "")), "confidence": c, "raw": reply}
            err = "grade 必须是 A/E/I/O/U/X 之一，confidence 必须是 1–5 的整数"
        except Exception as e:  # noqa: BLE001 - any parse failure is sent back once
            err = f"JSON 解析失败: {e}"
        msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": err + "，请重新输出。"}]
    return None  # keep the original judgement; logged as recheck_failed


def self_check(llm, brief, zones, pairs, threshold=3, workers=4, save=None):
    """Node ③: re-ask every pair with confidence <= threshold, one pair per call."""
    low = [k for k, v in pairs.items() if v["confidence"] is not None and v["confidence"] <= threshold]
    def job(k):
        i, j = map(int, k.split("-"))
        return k, _recheck_one(llm, brief, zones, i, j, pairs[k])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = dict(ex.map(job, low))
    new_pairs, log = {}, []
    for k, v in pairs.items():
        r = results.get(k)
        if k in results:
            log.append({"pair": k, "before": v, "after": r, "changed": bool(r and r["grade"] != v["grade"]),
                        "recheck_failed": r is None})
        new_pairs[k] = ({"grade": r["grade"], "reason": r["reason"], "confidence": r["confidence"],
                         "rechecked_from": v["grade"]} if r else v)
    out = {"threshold": threshold, "n_low": len(low), "n_changed": sum(x["changed"] for x in log),
           "log": log, "pairs": new_pairs}
    if save:
        save(out)
    return out
