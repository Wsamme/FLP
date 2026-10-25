"""Model call wrapper. Model name / temperature come from config.json, never hard-coded.

Providers
- "anthropic": Claude Messages API over plain HTTPS (no SDK needed). Key from env ANTHROPIC_API_KEY.
- "openai_compatible": any /v1/chat/completions endpoint. Key from env LLM_API_KEY, URL from LLM_BASE_URL.
- "mock": offline stub for tests / dry runs. Returns syntactically valid but meaningless answers.
  It never sees gold data; its numbers must never be reported as results.

Note (checked 2026-10-21, platform.claude.com Sonnet 5.5 migration guide): Claude Sonnet 5.5 returns a
400 error for any non-default temperature/top_p/top_k. So temperature is only sent when config sets it.
"""
import json
import os
import random
import re
import time
import urllib.request


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, cfg):
        self.provider = cfg["provider"]
        self.model = cfg["model"]
        self.temperature = cfg.get("temperature")  # None -> do not send (model default)
        self.max_tokens = cfg.get("max_tokens", 16000)
        self.timeout = cfg.get("timeout_s", 600)
        self.mock_seed = cfg.get("mock_seed", 0)
        self.calls = []  # per-call log: tokens, seconds, tag

    # messages: list of {"role": "user"|"assistant", "content": str}
    def chat(self, system, messages, tag=""):
        t0 = time.time()
        if self.provider == "anthropic":
            text, usage = self._anthropic(system, messages)
        elif self.provider == "openai_compatible":
            text, usage = self._openai(system, messages)
        elif self.provider == "mock":
            text, usage = self._mock(system, messages)
        elif self.provider == "keyword":
            text, usage = self._keyword(system, messages)
        else:
            raise LLMError(f"unknown provider {self.provider}")
        self.calls.append({"tag": tag, "seconds": round(time.time() - t0, 3), **usage})
        return text

    def totals(self):
        return {
            "n_calls": len(self.calls),
            "input_tokens": sum(c.get("input_tokens", 0) for c in self.calls),
            "output_tokens": sum(c.get("output_tokens", 0) for c in self.calls),
            "seconds": round(sum(c["seconds"] for c in self.calls), 3),
        }

    # ---------------- providers ----------------
    def _post(self, url, headers, body):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        for attempt in range(4):  # transport retries only (429/5xx); content errors go to validate.py
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:500]
                if e.code in (429, 500, 502, 503, 529) and attempt < 3:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise LLMError(f"HTTP {e.code}: {msg}")
        raise LLMError("unreachable")

    def _anthropic(self, system, messages):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY not set")
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": messages}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        d = self._post("https://api.anthropic.com/v1/messages",
                       {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                       body)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage", {})
        return text, {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                      "stop_reason": d.get("stop_reason")}

    def _openai(self, system, messages):
        key, base = os.environ.get("LLM_API_KEY"), os.environ.get("LLM_BASE_URL")
        if not key or not base:
            raise LLMError("LLM_API_KEY / LLM_BASE_URL not set")
        body = {"model": self.model, "messages": [{"role": "system", "content": system}] + messages,
                "max_tokens": self.max_tokens}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        d = self._post(base.rstrip("/") + "/chat/completions",
                       {"Authorization": f"Bearer {key}", "content-type": "application/json"}, body)
        u = d.get("usage", {})
        return d["choices"][0]["message"]["content"], {
            "input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
            "stop_reason": d["choices"][0].get("finish_reason")}

    def _keyword(self, system, messages):
        """Stand-in, NOT an LLM: a fixed keyword rule that reads only the prompt (brief + unit list).
        Two units named in the same line of brief section 3 (process) -> E, conf 4
        (conf 3 if the line only says the two "have exchanges" 往来, i.e. direction/amount unclear).
        Two units named in the same line elsewhere in the brief -> O, conf 3 (so the self-check fires).
        Otherwise U, conf 5. A re-check call confirms the grade with conf 4.
        Used so the notebook shows every step with deterministic, explainable output before an API key exists."""
        first = messages[0]["content"]
        m = re.search(r"PAIR \[(\d+)\] (\S+) 与 \[(\d+)\] (\S+)", messages[-1]["content"])
        if m:
            g = re.search(r"等级 ([AEIOUX])", messages[-1]["content"]).group(1)
            out = {"i": int(m.group(1)), "j": int(m.group(3)), "grade": g,
                   "reason": "关键词规则：复核时维持原判断", "confidence": 4}
            return "```json\n" + json.dumps(out, ensure_ascii=False) + "\n```", {"input_tokens": 0, "output_tokens": 0}
        units = [(int(a), b) for a, b in re.findall(r"^- \[(\d+)\] (.+)$", first, re.M)]
        brief = first.split("# 作业单位清单")[0]
        section, hits = "", {}
        for line in brief.splitlines():
            if line.startswith("## "):
                section = line[3:5]
            names = [i for i, nm in units if nm in line]
            for a, i in enumerate(names):
                for j in names[a + 1:]:
                    key = (min(i, j), max(i, j))
                    grade = "E" if section.startswith("3") else "O"
                    conf = (3 if "往来" in line else 4) if grade == "E" else 3
                    if hits.get(key, ("U",))[0] != "E":
                        hits[key] = (grade, line.strip()[:40], conf)
        pairs = []
        for a, (i, ni) in enumerate(units):
            for j, nj in units[a + 1:]:
                g, src, conf = hits.get((i, j), ("U", None, 5))
                pairs.append({"i": i, "name_i": ni, "j": j, "name_j": nj, "grade": g,
                              "reason": f"关键词规则：同一行提到两者「{src}」" if src else "关键词规则：没有一行同时提到两者",
                              "confidence": conf})
        if "不要写理由" in system:
            pairs = [{"i": p["i"], "j": p["j"], "grade": p["grade"]} for p in pairs]
        return "```json\n" + json.dumps({"pairs": pairs}, ensure_ascii=False, indent=0) + "\n```", \
            {"input_tokens": 0, "output_tokens": 0}

    def _mock(self, system, messages):
        """Pipeline smoke test only: random grades, fixed seed. Reads nothing but the prompt."""
        rng = random.Random(self.mock_seed + len(self.calls))
        last = messages[-1]["content"]
        m = re.search(r"PAIR \[(\d+)\].*?\[(\d+)\]", last)
        if m:  # self-check call
            out = {"i": int(m.group(1)), "j": int(m.group(2)), "grade": rng.choice("AEIOUUUUX"),
                   "reason": "mock", "confidence": 4}
            return json.dumps(out, ensure_ascii=False), {"input_tokens": 0, "output_tokens": 0}
        ids = [int(x) for x in re.findall(r"^- \[(\d+)\] ", messages[0]["content"], re.M)]
        pairs = [{"i": i, "j": j, "grade": rng.choice("AEIOUUUUUUX"), "reason": "mock",
                  "confidence": rng.randint(2, 5)} for a, i in enumerate(ids) for j in ids[a + 1:]]
        if "不要写理由" in system:
            pairs = [{"i": p["i"], "j": p["j"], "grade": p["grade"]} for p in pairs]
        return "```json\n" + json.dumps({"pairs": pairs}, ensure_ascii=False) + "\n```", \
            {"input_tokens": 0, "output_tokens": 0}
