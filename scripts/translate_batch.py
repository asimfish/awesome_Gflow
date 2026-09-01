#!/usr/bin/env python3
"""批量翻译 pdfs/en/*.pdf -> pdfs/zh/<ID>_zh.pdf，3 路并行，可反复运行（增量）。"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/Users/liyufeng/Code/awesome_Gflow")
ST = Path("/Users/liyufeng/Code/_gflow_ref/super_translate")
EN = REPO / "pdfs/en"
ZH = REPO / "pdfs/zh"
LOGS = REPO / "logs_translate"
LOGS.mkdir(exist_ok=True)
WORKERS = int(os.environ.get("TRANSLATE_WORKERS", "3"))
TOTAL_EXPECTED = 35
IDLE_ROUNDS_MAX = 30  # 30 轮(每轮60s)无新文件则退出

env = dict(os.environ, VLLM_KEY="dummy")


def translate_one(en_pdf: Path) -> str:
    pid = en_pdf.stem.split("_")[0]  # T03 / N010 ...
    out = ZH / f"{pid}_zh.pdf"
    if out.exists():
        return f"skip {pid}"
    log = LOGS / f"{pid}.log"
    cmd = [
        "uv", "run", "python", "-m", "pdf_zh_translator", "translate",
        str(en_pdf), str(out),
        "--api-mode", "openai-compatible",
        "--api-url", "http://127.0.0.1:18000/v1",
        "--model", "qwen2.5-32b-awq",
        "--api-key-env", "VLLM_KEY",
        "--preserve-graphics-text",
        "--cache-file", str(ZH / f"{pid}.cache.jsonl"),
    ]
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=ST, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    dt = int(time.time() - t0)
    status = "OK" if (r.returncode == 0 and out.exists()) else f"FAIL rc={r.returncode}"
    return f"{status} {pid} {dt}s"


def done_ids():
    return {p.stem.replace("_zh", "") for p in ZH.glob("*_zh.pdf")}


def main():
    # 已有旧命名产物 T03_trajectory_balance_zh.pdf 归一化
    legacy = ZH / "T03_trajectory_balance_zh.pdf"
    if legacy.exists() and not (ZH / "T03_zh.pdf").exists():
        legacy.rename(ZH / "T03_zh.pdf")

    idle = 0
    processed: set[str] = set()
    while True:
        pending = []
        for en_pdf in sorted(EN.glob("*.pdf")):
            pid = en_pdf.stem.split("_")[0]
            if pid not in done_ids() and pid not in processed:
                pending.append(en_pdf)
        if pending:
            idle = 0
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for msg in ex.map(translate_one, pending):
                    print(msg, flush=True)
            processed |= {p.stem.split("_")[0] for p in pending}
        else:
            idle += 1
        n_done = len(done_ids())
        print(f"[round] done={n_done}/{TOTAL_EXPECTED} idle={idle}", flush=True)
        if n_done >= TOTAL_EXPECTED or idle >= IDLE_ROUNDS_MAX:
            break
        time.sleep(60)
    print("BATCH_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
