#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补录论文前的查重工具。精确标题匹配会漏判（题名差一个词就查不到），
所以同时做 arXiv/DOI 号命中 + 标题模糊相似度。

用法：
    python3 scripts/check_duplicate.py "论文标题" [arxiv或doi号]
    python3 scripts/check_duplicate.py --file candidates.txt   # 每行「标题<TAB>号」
"""
import csv
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

TSV = Path(__file__).resolve().parent.parent / "papers/current_papers.tsv"
DUP_THRESHOLD = 0.75


def load():
    with open(TSV, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def check(title: str, ident: str, rows) -> tuple[str, str]:
    ids_blob = " ".join(r["url"] for r in rows)
    if ident and ident in ids_blob:
        hit = next(r["id"] for r in rows if ident in r["url"])
        return "DUPLICATE", f"号命中 {hit}"
    nt = norm(title)
    best = max(rows, key=lambda r: SequenceMatcher(None, nt, norm(r["title"])).ratio())
    ratio = SequenceMatcher(None, nt, norm(best["title"])).ratio()
    if ratio >= DUP_THRESHOLD:
        return "LIKELY_DUP", f"sim={ratio:.2f} 疑似 {best['id']} 《{best['title'][:60]}》"
    return "NEW", f"sim={ratio:.2f} 最近似 {best['id']} 《{best['title'][:50]}》"


def main():
    rows = load()
    if len(sys.argv) >= 3 and sys.argv[1] == "--file":
        items = []
        for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            items.append((parts[0], parts[1] if len(parts) > 1 else ""))
    elif len(sys.argv) >= 2:
        items = [(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")]
    else:
        print(__doc__)
        return 1

    print(f"目录现有 {len(rows)} 篇；相似度阈值 {DUP_THRESHOLD}\n")
    for title, ident in items:
        verdict, detail = check(title, ident, rows)
        print(f"[{verdict:<10}] {title[:66]}")
        print(f"             {detail}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
