"""Build notebooks/agent_pipeline.ipynb (cells below), then execute it with nbclient.
Run from 09_agent/:  python notebooks/build_notebook.py
"""
import os
import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
cells = []
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
C = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

M("""
# 需求理解 Agent · 逐格运行记录（fork14 端到端）
Requirement-understanding agent · cell-by-cell run (fork14 end to end)

**本次运行的模型位置用的是"关键词规则替身"，不是 LLM**（还没有 API key）。替身之后的每一格都是真实代码、真实计算；只有关系表本身来自替身，所以第 ⑪ 节的分数**不是 Agent 的实验结果**。
把第 0 格的 `PROVIDER` 改成 `"anthropic"`、设置环境变量 `ANTHROPIC_API_KEY` 后重跑，就是真实结果。

**The model slot in this run is a keyword-rule stand-in, not an LLM** (no API key yet). Every cell after it is real code with real computation. Only the relation table comes from the stand-in, so the scores in section ⑪ are **not agent results**. Set `PROVIDER = "anthropic"` and `ANTHROPIC_API_KEY`, then re-run for real results.

| 格 / Cells | 节点 / Node | 读 / Reads | 写 / Writes |
|---|---|---|---|
| 1–3 | 数据 | `data/fork14_zones.json`, `data/fork14_brief.md` | — |
| 4 | 标准答案（只给评估） | `data/gold/fork14_gold.json` | — |
| 5–6 | ① 数值隔离 + 防泄题 | brief, 区域名称 | `state_1_prompt.json` |
| 7–8 | ② 关系 Agent | 提示词 | `state_2_table.json` |
| 9–10 | ⑤ 格式校验 | 模型回复 | （记录在 state_2） |
| 11 | ③ 置信度自检 | 低置信度的对 | `state_3_selfcheck.json` |
| 12 | ⑥ 编译 | 关系表 + 面积表 | `instance_agent.json` |
| 13–16 | ⑦ 求解 + ⑨ 几何检查 | 算例 JSON | `layout_*.json`, 布局图 |
| 17–19 | ⑪ 评估 | 关系表 + gold | `metrics.json` |
| 20 | pump14（只测关系） | | |
| 21 | 本次运行写出的文件 | | `meta.json` |
""")

C("""
# 0 · 设置 / Setup
import json, os, sys, time, re, copy
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from IPython.display import display

ROOT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, ROOT)
from agent.llm import LLM
from agent import relation_agent as ra
from agent.validate import validate_table
from agent.compile import compile_instance, weight_matrix
from agent import solver_bridge as sb

PROVIDER = "keyword"          # 真实运行改成 "anthropic" / real run: "anthropic"
MODEL    = "keyword-rule"     # 例如 / e.g. "claude-sonnet-5-5"
SOLVER_DIR = os.environ.get("SOLVER_DIR", "/home/claude/solver_v2")   # 实验一求解器（work_v2, 10-11 版）
CASE = "fork14"
SEED, TIME_LIMIT = 0, 30      # 求解器种子、时限（秒）

# 假设待定（README 第 1–3 条）/ pending assumptions
FLOOR = {"W": 310, "H": 285}              # 厂区 310 × 285 m，填充率 75.2%（三个候选之一）
SHAPE = {"type": "Rmax", "value": 5}      # 长宽比 ≤ 5（书中最大的是办公楼 300/60 = 5）
X_MIN_DIST = None                         # X 最小净距 D：未定；本求解器也不支持此约束

RUN_DIR = os.path.join(ROOT, "runs_notebook", time.strftime("%Y%m%d-%H%M%S") + f"_{CASE}_{PROVIDER}")
os.makedirs(RUN_DIR, exist_ok=True)
def save(name, obj):
    with open(os.path.join(RUN_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
print("ROOT      ", ROOT)
print("RUN_DIR   ", os.path.relpath(RUN_DIR, ROOT))
print("PROVIDER  ", PROVIDER, "| MODEL", MODEL)
print("SOLVER_DIR", SOLVER_DIR, "| exists:", os.path.isdir(SOLVER_DIR))
""")

M("""
## 1 · 数据 / Data
面积表：只进编译 ⑥，不进提示词。  Zone table: goes to compile ⑥ only, never into the prompt.
""")
C("""
# 1 · 面积表 / zone table
zt = json.load(open(os.path.join(ROOT, "data", f"{CASE}_zones.json"), encoding="utf-8"))
zones_df = pd.DataFrame(zt["zones"]).set_index("id")
display(zones_df)
print("面积合计 / total area =", zones_df["area"].sum(), "m²   (书中给 66,464)")
print("厂区面积 / floor area =", FLOOR["W"] * FLOOR["H"], "m²   填充率 / fill =",
      f'{zones_df["area"].sum() / (FLOOR["W"] * FLOOR["H"]):.1%}')
""")
C("""
# 2 · Agent 能读到的东西 / what the agent may read
brief, zones = ra.load_inputs(CASE)      # 只取编号+名称，面积留在 zones_df
print("zones =", zones)
print("brief:", len(brief), "字 /", brief.count(chr(10)) + 1, "行")
""")
C("""
# 3 · brief 原文（你审核的就是这份） / the brief text under review
print(brief)
""")

M("""
### 标准答案（只给评估脚本用） / Answer key (evaluation only)
这一格读 `data/gold/`。Agent 代码 `agent/*.py` 不读它（有测试检查）。
This cell reads `data/gold/`. Agent code never does (a test checks this).
""")
C("""
# 4 · gold 14×14 + 行合计核对 / gold matrix + row-sum check
from eval.gold import load_gold, full_matrix, row_sums, SCORE
gold = load_gold(CASE)
G = pd.DataFrame(full_matrix(gold["pairs"], 14), index=range(1, 15), columns=range(1, 15))
G["合计"] = row_sums(gold["pairs"], 14)
G["书中"] = gold["expected_row_sums"]
display(G)
print("行合计全部一致 / all row sums match:", (G["合计"] == G["书中"]).all())
print("等级分布 / grade counts:", pd.Series(list(gold["pairs"].values())).value_counts().reindex(list("AEIOUX")).to_dict())
""")

M("""
## ① 数值隔离 + 防泄题 / Number isolation + leak checks
来源 OptiMUS：数值不进提示词。下面打印的就是发给模型的全部内容。
Source OptiMUS: numbers stay out of the prompt. Below is everything the model receives.
""")
C("""
# 5 · 实际发出的提示词 / the actual prompt
system = ra.SYSTEM_TABLE
user = ra.table_prompt(brief, zones)
save("state_1_prompt.json", {"system": system, "user": user})
print("=== SYSTEM (", len(system), "字) ===")
print(system)
print()
print("=== USER 末尾 / end of user message ===")
print(user[user.index("# 作业单位清单"):])
""")
C("""
# 6 · 防泄题自动检查 / automatic leak checks
checks = {
  "没有小数（物流强度）":        not re.search(r"\\d+\\.\\d+", user),
  "没有尺寸（如 72×36）":        not re.search(r"\\d+\\s*[×x]\\s*\\d+", user),
  "没有面积数":                  not any(str(a) in user for a in zones_df["area"]),
  "brief 里没有单独的等级字母":   not re.search(r"(?<![A-Za-z])[AEIOUX](?![A-Za-z])", brief),
  "没有 gold 里的'行 i: jG'写法": not re.search(r"\\b\\d{1,2}[AEIOUX]\\b", user),
}
display(pd.DataFrame({"检查 / check": checks.keys(), "通过 / pass": checks.values()}))
assert all(checks.values())
""")

M("""
## ② 关系整理 Agent / Relation agent
来源：SLP 等级（Muther）；每对一行（Holodeck）；一个 Agent 只做一件事（Co-Layout）。
本次 `PROVIDER = "keyword"`：规则是"brief 第 3 节同一行同时提到两个单位 → E（行里只说'往来'时置信度 3，否则 4）；其他 → U"。
This run's stand-in rule: two units named in the same line of section 3 → E (confidence 3 if the line only says "exchanges", else 4); otherwise U.
""")
C("""
# 7 · 调用模型 + ⑤ 校验循环 / call the model with the ⑤ validation loop
llm = LLM({"provider": PROVIDER, "model": MODEL, "temperature": None, "max_tokens": 16000})
t = ra.ask_table(llm, brief, zones, max_retries=2, save=lambda r: save("state_2_table.json", r))
print("尝试次数 / attempts:", len(t["attempts"]), "| failed:", t["failed"])
print("token / time:", llm.totals())
reply = t["attempts"][-1]["reply"]
print("\\n回复前 6 行 / first 6 lines of the reply:")
print("\\n".join(reply.splitlines()[:6]))
""")
C("""
# 8 · 91 对整理成表 / the 91 pairs as a table
pairs = t["pairs"]
P = pd.DataFrame([{"pair": k, "i": int(k.split("-")[0]), "j": int(k.split("-")[1]),
                   "单位 i": zones[int(k.split("-")[0])], "单位 j": zones[int(k.split("-")[1])], **v}
                  for k, v in pairs.items()])
print("共", len(P), "对；等级分布:", P["grade"].value_counts().reindex(list("AEIOUX"), fill_value=0).to_dict())
display(P[P["grade"] != "U"][["pair", "单位 i", "单位 j", "grade", "confidence", "reason"]])
""")

M("""
## ⑤ 格式校验 / Format validation
先看真实回复的校验结果，再人为制造 4 种错误，看校验器抓到什么、回传给模型的原话是什么。
First the real reply, then 4 planted errors: what the validator catches, and the exact message sent back.
""")
C("""
# 9 · 真实回复的校验结果 / validation of the real reply
_, errs, corr = validate_table(reply, zones)
print("errors:", errs)
print("corrections:", corr)
""")
C("""
# 10 · 人为制造 4 种错误 / plant 4 errors
d = json.loads(re.search(r"```json\\s*(\\{.*\\})\\s*```", reply, re.S).group(1))
rows = d["pairs"]
rows = [r for r in rows if (r["i"], r["j"]) != (2, 3)]                      # (a) 删掉 2-3
for r in rows:
    if (r["i"], r["j"]) == (1, 2): r["name_j"] = "油料仓库"                  # (b) 名称写错一个字
    if (r["i"], r["j"]) == (4, 5): r["grade"] = "B"                         # (c) 非法等级
rows.append(dict(next(r for r in rows if (r["i"], r["j"]) == (1, 4))))     # (d) 重复 1-4
bad = "```json\\n" + json.dumps({"pairs": rows}, ensure_ascii=False) + "\\n```"
_, errs, corr = validate_table(bad, zones)
print("纠正（不算错）/ auto-corrected:", corr)
print("\\n回传给模型的原话 / message sent back to the model:\\n")
print(ra.feedback_message(errs))
""")

M("""
## ③ 置信度自检 / Confidence self-check
来源 OptiMUS。confidence ≤ 3 的对，单独再问一次（只给这一对 + brief）。
Source OptiMUS. Pairs with confidence ≤ 3 are asked again, one at a time.
""")
C("""
# 11 · 自检 / self-check
low = P[P["confidence"] <= 3][["pair", "单位 i", "单位 j", "grade", "confidence", "reason"]]
print("需要复核的对 / pairs to re-check:", len(low))
display(low)
if len(low):
    k = low.iloc[0]["pair"]; i, j = map(int, k.split("-"))
    print("第一对的复核提示词（末尾）/ re-check prompt for the first pair (end):\\n")
    q = ra.recheck_prompt(brief, zones, i, j, pairs[k]); print(q[q.index("# 需要复核的一对"):])
sc = ra.self_check(llm, brief, zones, pairs, threshold=3, workers=4, save=lambda r: save("state_3_selfcheck.json", r))
print("\\n复核", sc["n_low"], "对，改了", sc["n_changed"], "对")
display(pd.DataFrame([{"pair": x["pair"], "前 / before": x["before"]["grade"],
                       "后 / after": x["after"]["grade"] if x["after"] else None, "改了": x["changed"]} for x in sc["log"]]))
final_pairs = sc["pairs"]
""")

M("""
## ⑥ 编译 / Compile
A=4 E=3 I=2 O=1 U=0 → 权重矩阵；X → `min_separation` 列表（不进权重）。面积直接取面积表。
Grades → weight matrix; X → a `min_separation` list (not a weight). Areas come straight from the zone table.
""")
C("""
# 12 · 编译成求解器算例 / compile into a solver instance
inst_agent = compile_instance("fork14_agent", os.path.join(ROOT, "data", "fork14_zones.json"), final_pairs,
                              FLOOR, SHAPE, X_MIN_DIST, f"{PROVIDER}:{MODEL}")
save("instance_agent.json", inst_agent)
W = pd.DataFrame(weight_matrix(final_pairs, 14), index=range(1, 15), columns=range(1, 15))
print("对称 / symmetric:", (W.values == W.values.T).all(), "| 非零权重对数:", len(inst_agent["weights"]))
display(W)
print("weights 前 6 条:", inst_agent["weights"][:6])
print("min_separation:", inst_agent["min_separation"])
print("areas:", inst_agent["areas"])
""")

M("""
## ⑦ 求解 + ⑨ 几何检查 / Solve + geometry check
求解器 = 实验一的"ours"（两阶段 SA + SOCP），代码来自 `~/FLP/results/bench_work/work_v2.tgz`（10-11 版），**一行未改**。
**这个求解器没有最小间距约束**，所以 X 对现在不受约束，只在事后量实际间距。
Solver = Experiment 1 "ours", unmodified (10-11 version). **It has no minimum-distance constraint**, so X pairs are not enforced; we only measure their gaps afterwards.

打分规则：两个布局都用 **gold 权重**算目标值。  Scoring: both layouts use **gold weights**.

求解器按时间预算停止，所以同一个种子重跑，目标值也可能略有不同；正式实验要多跑几个种子。
The solver stops on a time budget, so the same seed can give slightly different values on a re-run; the real experiment should use several seeds.
""")
C("""
# 13 · 用 Agent 的关系表求解 / solve with the agent's table
res_agent = sb.solve(inst_agent, SOLVER_DIR, seed=SEED, time_limit=TIME_LIMIT, log=lambda *a: None)
save("layout_agent.json", res_agent)
print("可行 / feasible:", res_agent["feasible"], "| 违规 / violations:", res_agent["violations"], "| 用时", res_agent["wall"], "s")
""")
C("""
# 14 · 参考上限：用 gold 关系表求解 / reference: solve with the gold table
gold_pairs = {f"{i}-{j}": {"grade": g} for (i, j), g in gold["pairs"].items()}
inst_gold = compile_instance("fork14_gold", os.path.join(ROOT, "data", "fork14_zones.json"), gold_pairs,
                             FLOOR, SHAPE, X_MIN_DIST, "gold")
res_gold = sb.solve(inst_gold, SOLVER_DIR, seed=SEED, time_limit=TIME_LIMIT, log=lambda *a: None)
save("layout_gold.json", res_gold)
print("可行 / feasible:", res_gold["feasible"], "| 违规 / violations:", res_gold["violations"], "| 用时", res_gold["wall"], "s")
""")
C("""
# 15 · 用 gold 权重打分 + X 对实际间距 / score with gold weights + X-pair gaps
Wg = np.array(weight_matrix(gold_pairs, 14))
x_pairs = [tuple(map(int, k.split("-"))) for k, v in gold_pairs.items() if v["grade"] == "X"]
score = pd.DataFrame({
  "关系表来源": [f"{PROVIDER}", "gold（上限参考）"],
  "可行": [res_agent["feasible"], res_gold["feasible"]],
  "gold 权重目标值": [round(sb.cost_with(Wg, res_agent["rects"]), 1), round(sb.cost_with(Wg, res_gold["rects"]), 1)],
})
score["比 gold 差"] = (score["gold 权重目标值"] / score["gold 权重目标值"].iloc[1] - 1).map("{:+.1%}".format)
display(score)
gx = pd.DataFrame(sb.x_gaps(res_agent["rects"], x_pairs)).rename(columns={"gap": f"间距（{PROVIDER}）"})
gx[ "间距（gold）"] = [g["gap"] for g in sb.x_gaps(res_gold["rects"], x_pairs)]
gx["单位"] = [f'{zones[i]} – {zones[j]}' for i, j in x_pairs]
display(gx[["pair", "单位", f"间距（{PROVIDER}）", "间距（gold）"]])
print("间距 ≤ 0 表示两块贴在一起 / gap ≤ 0 means touching")
""")
C("""
# 16 · 布局图 / layout plots
def plot(ax, rects, title):
    ax.add_patch(Rectangle((0, 0), FLOOR["W"], FLOOR["H"], fill=False, lw=1.5, ec="black"))
    for k, (x, y, w, h) in enumerate(rects, 1):
        ax.add_patch(Rectangle((x, y), w, h, fc="#eeeeee", ec="black", lw=0.8))
        lab = f"{k}\\n{zones[k][:4]}" if w >= 45 else f"{k}"   # 窄条只标编号 / narrow: number only
        ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=7)
    for i, j in x_pairs:
        a, b = rects[i - 1], rects[j - 1]
        ax.plot([a[0] + a[2] / 2, b[0] + b[2] / 2], [a[1] + a[3] / 2, b[1] + b[3] / 2], "k--", lw=0.8)
    ax.set_xlim(-5, FLOOR["W"] + 5); ax.set_ylim(-5, FLOOR["H"] + 5); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9); ax.set_xlabel("m"); ax.set_ylabel("m")
fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), dpi=110)
plot(axs[0], res_agent["rects"], f"关系表：{PROVIDER}   gold 权重目标值 {score.iloc[0, 2]}")
plot(axs[1], res_gold["rects"], f"关系表：gold   目标值 {score.iloc[1, 2]}")
fig.suptitle("虚线 = 书中 X 对（不希望靠近）；编号见第 1 格面积表", fontsize=9); fig.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "layouts.png")); plt.close(fig)
from IPython.display import Image
display(Image(filename=os.path.join(RUN_DIR, "layouts.png")))
""")

M("""
## ⑪ 评估 / Evaluation
逐对和 gold 比。再提醒一次：本次关系表来自关键词替身，下面的分数是**替身的分数**。
Pair-by-pair against gold. Again: these are **the stand-in's scores**, not the agent's.
""")
C("""
# 17 · 指标 / metrics
from eval.eval_relations import evaluate
KEYS = ["exact", "within1", "kappa_linear", "kappa_unweighted", "nonU_recall", "nonU_precision", "X_recall"]
m_run = evaluate(gold["pairs"], final_pairs, gold["exclude"])
m_allU = evaluate(gold["pairs"], {k: "U" for k in gold["pairs"]}, gold["exclude"])
M_ = pd.DataFrame([{k: m_allU[k] for k in KEYS}, {k: m_run[k] for k in KEYS}], index=["全 U 基线", f"{PROVIDER}（自检后）"]).round(3)
display(M_)
save("metrics.json", {k: v for k, v in m_run.items() if k != "errors"})
""")
C("""
# 18 · 混淆矩阵 / confusion matrix (rows = gold, cols = predicted)
cm = pd.DataFrame(m_run["confusion"], index=[f"gold {g}" for g in "AEIOUX"], columns=[f"预测 {g}" for g in "AEIOUX"])
display(cm)
print("对角线 = 判对的对数 / diagonal = correct pairs:", int(np.trace(cm.values)), "/", m_run["n_pairs"])
""")
C("""
# 19 · 错误分类 + 例子 / error types with examples
display(pd.DataFrame([m_run["error_counts"]]).reindex(columns=["等级偏高", "等级偏低", "漏判X", "误判X", "有关系判成U", "U判成有关系"], fill_value=0))
ex = [{"类别": t, "pair": e["pair"], "单位": " – ".join(zones[int(x)] for x in e["pair"].split("-")),
       "gold": e["gold"], "预测": e["pred"], "理由": e["reason"]}
      for t, es in m_run["errors"].items() for e in es[:2]]
display(pd.DataFrame(ex))
""")

M("""
## pump14（只测关系，没有面积） / pump14 (relations only)
同一套代码换算例。4–6 排除，按 90 对算。  Same code, other case; pair 4–6 excluded.
""")
C("""
# 20 · pump14
b2, z2 = ra.load_inputs("pump14")
t2 = ra.ask_table(LLM({"provider": PROVIDER, "model": MODEL}), b2, z2)
sc2 = ra.self_check(LLM({"provider": PROVIDER, "model": MODEL}), b2, z2, t2["pairs"])
g2 = load_gold("pump14")
r2 = evaluate(g2["pairs"], sc2["pairs"], g2["exclude"]); a2 = evaluate(g2["pairs"], {k: "U" for k in g2["pairs"]}, g2["exclude"])
display(pd.DataFrame([{k: a2[k] for k in KEYS}, {k: r2[k] for k in KEYS}], index=["全 U 基线", f"{PROVIDER}（自检后）"]).round(3))
""")
C("""
# 21 · 本次运行写出的文件 / files written by this run
save("meta.json", {"provider": PROVIDER, "model": MODEL, "case": CASE, "seed": SEED, "time_limit": TIME_LIMIT,
                   "floor": FLOOR, "shape": SHAPE, "x_min_dist": X_MIN_DIST, "solver_dir": SOLVER_DIR,
                   "llm_totals": llm.totals(), "date": time.strftime("%Y-%m-%d %H:%M:%S")})
for f in sorted(os.listdir(RUN_DIR)):
    print(f"{os.path.getsize(os.path.join(RUN_DIR, f)):>8,d} B  {f}")
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
path = os.path.join(HERE, "agent_pipeline.ipynb")
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
nbf.write(nb, path)
print("wrote", path)
