# FLP — reproducible benchmarking for the unequal-area facility layout problem (UA-FLP)

Part of the ME5001A project (NUS) on LLM-assisted factory layout. This repository holds the
**verified instance set**, an **independent evaluator**, and **faithful reproductions of published
baselines** that our own method (CP-SAT + agent planning layer) will be compared against.

## Principles
1. Every instance number has a traceable source (paper table + page, or dataset DOI). See `instances/MANIFEST.md`.
2. The evaluator is validated by recomputing layouts published in the literature (`tests/test_replay.py`).
3. A baseline counts as reproduced only after it reproduces the paper's own reported values
   (deterministic methods: same value; stochastic methods: best and mean within ±1%).
4. Results are compared only within the same problem setting (floor size, shape rule, objective, distance).

## Layout
| Path | Content |
|---|---|
| `instances/uaflp/` | frozen instances (JSON) |
| `instances/sources/` | original data files with licences |
| `instances/MANIFEST.md` | provenance + SHA-256 of every instance file |
| `flp/` | loader, evaluator, FBS decoder |
| `tests/` | evaluator validation against published solutions |
| `docs/` | experiment protocol and decisions |
| `baselines/` | reproductions of published methods (in progress) |

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q
```
