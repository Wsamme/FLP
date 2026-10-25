# agent/ — ME5001A 需求理解 Agent，第一阶段

Requirement-understanding agent for ME5001A, stage 1.

状态（2026-10-26）：代码、测试、notebook 都已跑通；**真实 LLM 还没跑**（没有 API key），notebook 里模型位置用的是关键词规则替身。
Status (2026-10-26): code, tests, and the notebook all run. **No real LLM run yet** (no API key); the notebook uses a keyword-rule stand-in for the model.

这个目录是新加的，不改动仓库里实验一的任何文件。
This folder is new. It does not change any Experiment 1 file in this repo.

## 先看这两个页面 / Start with these two pages

| 文件 / File | 内容 / What it is |
|---|---|
| `notebooks/agent_pipeline.html` | 运行记录：每一格代码 + 实际输出，按节点分节（fork14 端到端） / Run record: each code cell with its real output, by node |
| `docs/agent_method_walkthrough.html` | 设计讲解：每个节点为什么这样设计、出自哪篇论文，带算例计算 / Design walkthrough: why each node is designed this way, with worked numbers |

两个都是单文件 HTML，下载后用浏览器打开即可。
Both are single HTML files. Download and open in a browser.

## 运行 / How to run

```bash
pip install numpy pandas matplotlib numba clarabel scipy nbformat nbclient ipykernel markdown pytest
python -m pytest -q tests                  # 18 个测试 / 18 tests
python eval/check_gold.py                  # 标准答案核对 / answer-key check
export SOLVER_DIR=/path/to/experiment1     # 实验一求解器目录（含 bench/、flp/、baselines/）
python notebooks/build_notebook.py         # 重跑 notebook / re-run the notebook
python notebooks/render_html.py            # 生成 HTML / render the HTML
```

真实运行：notebook 第 0 格把 `PROVIDER` 改成 `"anthropic"`、`MODEL` 改成模型名，并设置环境变量 `ANTHROPIC_API_KEY`（不要写进代码）。
Real run: in cell 0 set `PROVIDER = "anthropic"` and the model name, and set `ANTHROPIC_API_KEY` as an environment variable (never in code).

求解器：用的是实验一的 "ours"（两阶段 SA + SOCP），来自 `results/bench_work/work_v2.tgz`（2026-10-11 版），**代码未改**。`agent/solver_bridge.py` 只是把编译出的算例转成求解器的 `Problem`。
Solver: Experiment 1 "ours" (two-stage SA + SOCP) from `work_v2.tgz` (2026-10-11), **unmodified**. `agent/solver_bridge.py` only converts the compiled instance into the solver's `Problem`.

## 目录 / Layout

```
agent/            ② 关系 Agent、③ 自检、⑤ 校验、⑥ 编译、⑦ 求解器接口、模型调用
eval/             标准答案读取与核对、⑪ 评估指标、汇总报告
data/             两份 brief、区域表；data/gold/ 只给 eval/ 读
notebooks/        notebook 源码、执行后的 .ipynb、HTML 页面
docs/             设计讲解页
runs_notebook/    本次 notebook 运行写出的全部中间文件
tests/            格式校验边界情况、权重矩阵对称、防泄题等
run_all.py        关系实验批量运行（2 算例 × 3 设置 × 5 次）
```

## 本次运行的结果 / Results of this run

`runs_notebook/20261025-035723_fork14_keyword/`，厂区 310 × 285 m，长宽比 ≤ 5，种子 0，每次 30 s。
Site 310 × 285 m, aspect ratio ≤ 5, seed 0, 30 s per solve.

| 关系表来源 / Relation table | 可行 / Feasible | 用书中权重算的目标值 / Objective (book weights) |
|---|---|---|
| 书中标准关系表 / book chart | 是，检查器零违规 | 6929.3 |
| 关键词替身 / keyword stand-in | 是，检查器零违规 | 9116.2 |

- 替身的关系分数（完全一致率 0.538，κ 0.233）**不是 Agent 结果**，只用来演示流程。
  The stand-in's relation scores (agreement 0.538, kappa 0.233) are **not agent results**; they only demonstrate the pipeline.
- 全 U 基线（真实）：fork14 一致率 0.505，pump14 0.733，κ 都是 0。
  All-U baseline (real): agreement 0.505 on fork14, 0.733 on pump14, kappa 0.
- 6929.3 是单个种子、30 秒的结果，不是证明过的最优。求解器按时间停，同一种子重跑结果也会变。
  6929.3 is one seed and 30 s, not a proven optimum. The solver stops on a time limit, so a re-run can differ.

## 已知问题 / Known issues

1. 求解器没有最小间距约束，X 对（不希望靠近）现在没被管住，只在求解后量实际间距。处理方式待定。
   The solver has no minimum-distance constraint, so X pairs are not enforced; gaps are only measured afterwards. Handling to be decided.
2. 长宽比 ≤ 5 会把车间压成细长条。形状规则待定。
   Aspect ratio ≤ 5 squeezes shops into long strips. Shape rule to be decided.
3. pump14 第 10 行（10–11、10–12）两个来源不一致，需对原书 p.168。
   pump14 row 10 (10–11, 10–12) differs between two transcripts; check the book, p.168.
4. 新版 Claude 模型传非默认 temperature 会报 400 错误，所以用默认采样，随机性靠重复次数来量。
   Newer Claude models return a 400 error for non-default temperature, so we use default sampling and measure randomness with repeats.

## 假设（待定项标【待定】）/ Assumptions

1. fork14 厂区边界：【待定】本次用 310 × 285 m（填充率 75.2%）。
2. 形状规则：【待定】本次用长宽比 ≤ 5（书中最大的是办公楼 300/60 = 5）。
3. X 级最小净距 D：【待定】。
4. 物流强度单位：书中未给出；本阶段不用物流数字，只用来写 brief 里的物料走向。
5. pump14 的 4–6：书中自相矛盾，评估排除，按 90 对算。
6. brief 第 4 节是书外常识（各 3 条），审核时可整节删除。
7. 加权 κ 的等级顺序 A, E, I, O, U, X（X 放在 U 后），线性权重，是我们的口径。
8. 提示词不放 few-shot 示例。

## 数据来源 / Data sources

- 尺寸：戢守峰《现代设施规划与物流分析》表 6-2（p.107）
- 物流强度、综合关系表：方庆琯《现代物流设施与规划》第 3 版，表 6-7（p.133–134）、表 6-16（p.141）；pump14 见 §6.6（p.160–168）
- 设计来源：OptiMUS（ICML 2024）、Holodeck（CVPR 2024）、Co-Layout（AAAI 2026）、Muther SLP；实验二参考 CANDO / ALPS-Bench（NeurIPS 2026）
