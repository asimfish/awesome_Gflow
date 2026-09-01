#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draft notes for the 3 over-long docs (T12, O01, N061) with aggressive truncation."""
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/Users/liyufeng/Code/awesome_Gflow")
API = "http://127.0.0.1:18000/v1/chat/completions"
sys.path.insert(0, "/Users/liyufeng/Code/_gflow_ref/super_translate/.venv/lib/python3.13/site-packages")
import pymupdf

JOBS = {
    "T12": ("continuous_gflownets_theory", "ICML 2023", "paper"),
}

SYS_PAPER = (
    "你是 GFlowNet 领域的资深研究员，为中文读者写深度论文解读。禁用「值得注意的是」「综上所述」"
    "「总的来说」「本文」等套话，直接陈述；论断标注论文章节或公式号，个人判断标明「我的判断：」；"
    "公式用 $...$ 并逐个解释符号；「局限与批判」至少 4 条，每条 2-3 句具体展开；"
    "「与谁对话」用编号指代：T01=GFlowNet原始论文，T02=Foundations，T03=TB，T05=SubTB，"
    "T07=训练理解，T09=变分视角，T14=熵正则RL等价，T19=非无环理论，T36=离散非无环，O08=GFN隐式学OT；"
    "全文不少于 2500 汉字。"
)
SYS_MAP = (
    "你是 GFlowNet 与最优传输方向的资深研究员，为中文读者写「文献地图」而非逐节摘要。"
    "禁用套话，直接陈述；对每一章/节说明它讲什么、对 GFlowNet 研究有什么用、什么时候该读；"
    "个人判断标明「我的判断：」；给出明确的阅读路线（先读哪几节）；全文不少于 2000 汉字。"
)

TPL_PAPER = """# {pid} · {title}

> 发表：{venue} · 链接：{url}

## 一句话
## 问题与动机
## 方法核心
## 理论结果
## 实验与证据
## 局限与批判
## 与谁对话
## 对后续研究的启示"""

TPL_MAP = """# {pid} · {title}

> {venue} · 链接：{url}

## 一句话
## 这份材料的定位
## 章节地图
## 与 GFlowNet 研究的接口
## 阅读路线
## 局限与注意事项"""


def extract(pdf: Path, kind: str) -> str:
    doc = pymupdf.open(pdf)
    n = doc.page_count
    if kind == "paper":
        # 正文前 14 页足够覆盖方法与主定理
        pages = list(range(min(11, n)))
    else:
        # 讲义/论文：取目录页 + 每章开头采样
        step = max(1, n // 30)
        pages = list(range(0, min(12, n))) + list(range(12, n, step))
    seen, chunks = set(), []
    for i in pages:
        if i in seen:
            continue
        seen.add(i)
        chunks.append(f"[p{i+1}]\n" + doc.load_page(i).get_text())
    text = "\n".join(chunks)
    return text[:16000]


def gen(system, template, pid, title, url, venue, text):
    user = (
        f"编号 {pid}，标题《{title}》，{venue}，链接 {url}。\n"
        f"下面是节选正文（含页码标记）。请严格按模板写中文解读，保留标题结构：\n\n"
        f"{template.format(pid=pid, title=title, venue=venue, url=url)}\n\n===== 节选正文 =====\n{text}"
    )
    payload = {
        "model": "qwen2.5-32b-awq",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.3,
        "max_tokens": 6000,
    }
    req = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return json.load(opener.open(req, timeout=1800))["choices"][0]["message"]["content"]


def main():
    rows = {r["id"]: r for r in csv.DictReader(open(REPO / "papers/current_papers.tsv"), delimiter="\t")}
    en = {p.stem.split("_")[0]: p for p in (REPO / "pdfs/en").glob("*.pdf")}
    for pid, (slug, venue, kind) in JOBS.items():
        if list((REPO / "notes").glob(pid + "_*.md")):
            print("skip " + pid, flush=True)
            continue
        t0 = time.time()
        try:
            text = extract(en[pid], kind)
            sysmsg = SYS_PAPER if kind == "paper" else SYS_MAP
            tpl = TPL_PAPER if kind == "paper" else TPL_MAP
            md = gen(sysmsg, tpl, pid, rows[pid]["title"], rows[pid]["url"], venue, text)
            (REPO / "notes" / (pid + "_" + slug + ".md")).write_text(md)
            print("OK %s %ds %dchars (text=%d)" % (pid, time.time() - t0, len(md), len(text)), flush=True)
        except Exception as e:
            print("FAIL %s %s" % (pid, e), flush=True)
    print("LONG_DONE", flush=True)


if __name__ == "__main__":
    main()
