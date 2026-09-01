#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""翻译 pdfs/en_excerpt/*.pdf -> pdfs/zh/<name>_zh.pdf（超长参考材料的核心章节节选）。"""
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/Users/liyufeng/Code/awesome_Gflow")
ST = Path("/Users/liyufeng/Code/_gflow_ref/super_translate")
SRC = REPO / "pdfs/en_excerpt"
ZH = REPO / "pdfs/zh"
LOGS = REPO / "logs_translate"
LOGS.mkdir(exist_ok=True)

env = dict(os.environ, VLLM_KEY="dummy")


def run(src: Path) -> str:
    stem = src.stem
    out = ZH / f"{stem}_zh.pdf"
    if out.exists():
        return f"skip {stem}"
    log = LOGS / f"{stem}.log"
    cmd = [
        "uv", "run", "python", "-m", "pdf_zh_translator", "translate",
        str(src), str(out),
        "--api-mode", "openai-compatible",
        "--api-url", "http://127.0.0.1:18000/v1",
        "--model", "qwen2.5-32b-awq",
        "--api-key-env", "VLLM_KEY",
        "--preserve-graphics-text",
        "--timeout", "900",
        "--max-batch-chars", "800",
        "--cache-file", str(ZH / f"{stem}.cache.jsonl"),
    ]
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=ST, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=14400)
    ok = "OK" if (r.returncode == 0 and out.exists()) else f"FAIL rc={r.returncode}"
    return f"{ok} {stem} {int(time.time()-t0)}s"


def main():
    srcs = sorted(SRC.glob("*.pdf"))
    with ThreadPoolExecutor(max_workers=2) as ex:
        for msg in ex.map(run, srcs):
            print(msg, flush=True)
    print("EXCERPTS_DONE", flush=True)


if __name__ == "__main__":
    main()
