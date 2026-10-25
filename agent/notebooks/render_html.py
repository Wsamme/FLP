"""Render the executed notebook as a sectioned, notebook-style HTML page (black/white)."""
import html, os, re, sys
import markdown, nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbformat.read(os.path.join(HERE, "agent_pipeline.ipynb"), 4)

sections, cur = [], None
for c in nb.cells:
    if c.cell_type == "markdown" and (c.source.startswith("## ") or c.source.startswith("# ")):
        title = c.source.splitlines()[0].lstrip("# ").split(" / ")[0]
        if c.source.startswith("# "): title = "概述"
        cur = {"title": title, "cells": []}; sections.append(cur)
    cur["cells"].append(c)

def md(s):
    return markdown.markdown(s, extensions=["tables", "fenced_code"])

def out_html(o):
    t = o.output_type
    if t == "stream":
        return f'<pre class="out">{html.escape(o.text.rstrip())}</pre>'
    if t in ("execute_result", "display_data"):
        d = o.data
        if "image/png" in d:
            return f'<div class="img"><img alt="布局图" src="data:image/png;base64,{d["image/png"].strip()}"></div>'
        if "text/html" in d:
            h = re.sub(r'<style.*?</style>', '', d["text/html"], flags=re.S)
            h = h.replace('border="1" ', '')
            return f'<div class="df">{h}</div>'
        if "text/plain" in d:
            return f'<pre class="out">{html.escape(d["text/plain"])}</pre>'
    if t == "error":
        return f'<pre class="out err">{html.escape(o.ename + ": " + o.evalue)}</pre>'
    return ""

body, toc = [], []
for k, s in enumerate(sections):
    sid = f"n{k}"
    toc.append((sid, s["title"]))
    parts = []
    for c in s["cells"]:
        if c.cell_type == "markdown":
            parts.append(f'<div class="md">{md(c.source)}</div>')
        else:
            first = c.source.splitlines()[0]
            label = html.escape(first.lstrip("# ").strip())
            parts.append(
                f'<div class="cell"><div class="ch"><span class="in">In [{c.execution_count}]</span><span class="cl">{label}</span></div>'
                f'<pre class="code">{html.escape(c.source)}</pre>'
                + "".join(out_html(o) for o in c.outputs) + '</div>')
    body.append(f'<section class="step" id="{sid}" data-title="{html.escape(s["title"])}">' + "".join(parts) + '</section>')

tpl = open(os.path.join(HERE, "page_template.html"), encoding="utf-8").read()
page = tpl.replace("{{SECTIONS}}", "\n".join(body))
open(os.path.join(HERE, "agent_pipeline.html"), "w", encoding="utf-8").write(page)
print("sections:", [t for _, t in toc]); print("bytes:", len(page))
