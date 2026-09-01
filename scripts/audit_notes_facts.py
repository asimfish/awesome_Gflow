#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抽取解读「实验与证据」节里的数字断言，回原文 PDF 核对是否出现。
只做「该数字在原文是否存在」的机械核对——存在不等于用法正确，但不存在几乎必然是幻觉。"""
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, "/Users/liyufeng/Code/_gflow_ref/super_translate/.venv/lib/python3.13/site-packages")
import pymupdf

REPO = Path("/Users/liyufeng/Code/awesome_Gflow")

# 提取形如 0.844 / 5.4× / 10.0% / 97.5% 的数值 token
NUM = re.compile(r"\d+\.\d+|\d+(?=\s*[×倍%])")
# 排除明显是编号/年份/引用的
SKIP = re.compile(r"^(19|20|21|22|23|24|25|26)\d{2}$")


def norm_variants(tok: str):
    """一个数字在 PDF 里可能的写法变体"""
    v = {tok}
    if "." in tok:
        a, b = tok.split(".", 1)
        v.add(f"{a}.{b}")
        if b.endswith("0"):
            v.add(f"{a}.{b.rstrip('0') or '0'}")     # 10.0 -> 10.
        v.add(tok.replace(".", ". "))                 # PDF 断字
    return v


def main():
    rows = []
    for md in sorted((REPO / "notes").glob("*.md")):
        pid = md.name.split("_")[0]
        pdfs = glob.glob(str(REPO / "pdfs/en" / f"{pid}_*.pdf"))
        if not pdfs:
            continue
        text = md.read_text(encoding="utf-8")
        m = re.search(r"## 实验与证据\n(.*?)(?=\n## )", text, re.S)
        if not m:
            continue
        # 去掉笔记自身的小节标题行，否则「### 4.2 复现性」里的编号会被当成引用原文的数字
        body = "\n".join(l for l in m.group(1).splitlines() if not l.lstrip().startswith("#"))
        toks = [t for t in NUM.findall(body) if not SKIP.match(t)]
        toks = list(dict.fromkeys(toks))
        if not toks:
            continue
        src = "".join(p.get_text() for p in pymupdf.open(pdfs[0]))
        src_flat = re.sub(r"\s+", "", src)
        found, missing = [], []
        for t in toks:
            hit = any(v in src or re.sub(r"\s+", "", v) in src_flat for v in norm_variants(t))
            (found if hit else missing).append(t)
        rows.append((pid, len(toks), len(found), missing))

    print(f"{'ID':<6}{'断言':>4}{'命中':>5}{'命中率':>8}  未在原文找到")
    print("-" * 78)
    tot = hit = 0
    for pid, n, f, miss in rows:
        tot += n
        hit += f
        rate = f"{f/n*100:.0f}%"
        print(f"{pid:<6}{n:>4}{f:>5}{rate:>8}  {', '.join(miss[:8]) if miss else '—'}")
    print("-" * 78)
    print(f"合计 {tot} 个数字断言，{hit} 个在原文中出现，整体命中率 {hit/tot*100:.1f}%")


if __name__ == "__main__":
    main()
