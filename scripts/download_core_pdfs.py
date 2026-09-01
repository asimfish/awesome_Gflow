#!/usr/bin/env python3
"""???? arXiv ????????? PDF ? pdfs/en/?

?? papers/core_papers_arxiv.json?id -> {arxiv_id, pdf_path, matched_title}
"""
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = json.load(open(ROOT / "papers/core_papers.json"))
OUT_DIR = ROOT / "pdfs/en"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = ROOT / "papers/core_papers_arxiv.json"
results = json.load(open(RESULT_PATH)) if RESULT_PATH.exists() else {}

NS = {"a": "http://www.w3.org/2005/Atom"}
UA = {"User-Agent": "awesome-gflow-builder/0.1 (mailto:research@example.com)"}

# ??????? arxiv ?????????
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\\\(|\\\)|\$", "", s)  # ?? LaTeX ????? f-TB ? \(f\)?
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def arxiv_search(title: str):
    q = urllib.parse.quote(f'ti:"{title}"')
    url = f"https://export.arxiv.org/api/query?search_query={q}&max_results=8"
    req = urllib.request.Request(url, headers=UA)
    with OPENER.open(req, timeout=30) as r:
        tree = ET.fromstring(r.read())
    want = norm(title)
    cands = []
    for e in tree.findall("a:entry", NS):
        t = e.find("a:title", NS).text or ""
        aid = (e.find("a:id", NS).text or "").rsplit("/abs/", 1)[-1]
        cands.append((norm(t), t.strip(), aid))
    # ???????????/????
    for nt, t, aid in cands:
        if nt == want:
            return aid, t
    for nt, t, aid in cands:
        if want in nt or nt in want:
            return aid, t
    return None, None


def arxiv_search_loose(title: str):
    """?????????????????????"""
    head = title.split(":")[0]
    if len(head.split()) < 3:
        return None, None
    q = urllib.parse.quote(f"all:{head}")
    url = f"https://export.arxiv.org/api/query?search_query={q}&max_results=10"
    req = urllib.request.Request(url, headers=UA)
    with OPENER.open(req, timeout=30) as r:
        tree = ET.fromstring(r.read())
    want = norm(head)
    for e in tree.findall("a:entry", NS):
        t = e.find("a:title", NS).text or ""
        aid = (e.find("a:id", NS).text or "").rsplit("/abs/", 1)[-1]
        if norm(t).startswith(want) or want in norm(t):
            return aid, t.strip()
    return None, None


def download_pdf(arxiv_id: str, dest: Path):
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    req = urllib.request.Request(url, headers=UA)
    with OPENER.open(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest.stat().st_size


for p in CORE:
    pid, title = p["id"], p["title"]
    if pid in results and results[pid].get("pdf_ok"):
        continue
    entry = {"title": title}
    try:
        aid, mt = arxiv_search(title)
        time.sleep(3)
        if not aid:
            aid, mt = arxiv_search_loose(title)
            time.sleep(3)
        if aid:
            slug = re.sub(r"[^a-z0-9]+", "_", norm(title))[:60].strip("_")
            dest = OUT_DIR / f"{pid}_{slug}.pdf"
            size = download_pdf(aid, dest)
            entry.update({
                "arxiv_id": aid,
                "matched_title": mt,
                "pdf_path": str(dest.relative_to(ROOT)),
                "pdf_bytes": size,
                "pdf_ok": size > 20000,
            })
            print(f"[OK] {pid} {aid} {size/1e6:.1f}MB")
        else:
            entry.update({"arxiv_id": None, "pdf_ok": False})
            print(f"[MISS] {pid} {title[:60]}")
    except Exception as ex:  # noqa: BLE001
        entry.update({"error": str(ex), "pdf_ok": False})
        print(f"[ERR] {pid} {ex}")
    results[pid] = entry
    json.dump(results, open(RESULT_PATH, "w"), ensure_ascii=False, indent=1)
    time.sleep(1)

ok = sum(1 for v in results.values() if v.get("pdf_ok"))
print(f"DONE: {ok}/{len(CORE)} PDFs")
