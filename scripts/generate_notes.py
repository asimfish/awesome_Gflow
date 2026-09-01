#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用本地 vLLM(qwen2.5-32b-awq) 从英文 PDF 生成中文深度解读初稿。增量：已有 notes/<ID>_*.md 的跳过。"""
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/Users/liyufeng/Code/awesome_Gflow")
API = "http://127.0.0.1:18000/v1/chat/completions"

PAPERS = {
    "T03": ("trajectory_balance_credit_assignment", "NeurIPS 2022"),
    "T05": ("subtrajectory_balance_partial_episodes", "ICML 2023"),
    "T07": ("understanding_improving_gfn_training", "ICML 2023"),
    "T12": ("continuous_gflownets_theory", "ICML 2023"),
    "T14": ("gfn_entropy_regularized_rl", "AISTATS 2024"),
    "T15": ("inference_as_control_multipath", "UAI 2024"),
    "T32": ("when_gfn_learn_right_distribution", "ICLR 2025 Spotlight"),
    "T36": ("revisiting_nonacyclic_gfn", "ICML 2025"),
    "T49": ("f_trajectory_balance", "ICML 2026"),
    "O01": ("ot_for_machine_learners", "课程讲义 2025"),
    "O07": ("shortest_paths_gflownets", "ICML 2026 SPIGM Workshop"),
    "O08": ("gfn_secretly_ot_plan", "ICML 2026 SPIGM Workshop"),
    "A01": ("gfn_scientific_discovery", "Digital Discovery 2023"),
    "N007": ("distributional_quantile_flows", "TMLR 2024"),
    "N011": ("generative_augmented_flow_networks", "ICLR 2023 notable-25%"),
    "N015": ("cflownets_continuous_control", "ICLR 2023"),
    "N018": ("trajectory_balance_asynchrony", "NeurIPS 2025"),
    "N019": ("flowrl_reward_distribution_llm", "ICLR 2026"),
    "N032": ("adjoint_matching_soc_finetuning", "ICLR 2025 Spotlight"),
    "N038": ("generalized_sb_on_graphs", "预印本 2026-02"),
    "N041": ("unsupervised_ot_unbalanced_graphs", "NeurIPS 2025"),
    "N061": ("deleu_phd_thesis_gfn", "博士论文 2025"),
    "N082": ("path_dependent_amortized_inference", "ICML 2026 Oral"),
    "N103": ("general_soft_operators_robust_rl", "ICLR 2026"),
}

TEMPLATE = """# {pid} · {title}

> 发表：{venue} · 链接：{url}

## 一句话
## 问题与动机
## 方法核心
## 理论结果
## 实验与证据
## 局限与批判
## 与谁对话
## 对后续研究的启示"""

SYSTEM = (
    "你是 GFlowNet 领域的资深研究员，为中文读者写深度论文解读。写作纪律："
    "禁用「值得注意的是」「综上所述」「总的来说」「本文」等套话，直接陈述；"
    "论断标注论文章节或公式号，个人判断要标明「我的判断：」；"
    "公式用 $...$ 或 $$...$$ 并逐个解释符号；"
    "「局限与批判」至少 3 条且要具体；"
    "「与谁对话」里用编号指代相关论文：T01=GFlowNet原始论文(NeurIPS2021)，T02=GFlowNet Foundations(JMLR2023)，"
    "T03=Trajectory Balance，T05=SubTB，T07=Towards Understanding GFN Training，T08=GFN与HVI比较，"
    "T09=变分视角，T12=连续GFN理论，T14=GFN=熵正则RL，T17=散度训练，T19=非无环理论，T32=何时学对分布，"
    "O08=GFN隐式学OT计划，N019=FlowRL，N043=Diffusion SB Matching；"
    "「对后续研究的启示」给 2-3 条可操作方向，不写空话。"
    "硬性要求：全文不少于 2500 汉字；「方法核心」至少 30 行，逐步推导并解释每个符号；"
    "「实验与证据」列出具体环境、基线与具体数字；「局限与批判」至少 4 条，每条 2-3 句具体展开；"
    "章节之间不留空段落占位。"
)


def pdf_text(pdf: Path) -> str:
    sys.path.insert(0, "/Users/liyufeng/Code/_gflow_ref/super_translate/.venv/lib/python3.13/site-packages")
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    doc = pymupdf.open(pdf)
    pages = [p.get_text() for p in doc]
    full = "\n".join(pages)
    if len(full) <= 26000:
        return full
    return full[:21000] + "\n...[中间省略]...\n" + full[-5000:]


def gen(pid, title, url, venue, text):
    user = (
        "论文编号 {}，标题《{}》，发表于 {}，链接 {}。\n".format(pid, title, venue, url)
        + "下面是论文正文（可能截断）。请严格按以下模板写中文深度解读，保留模板的标题结构：\n\n"
        + TEMPLATE.format(pid=pid, title=title, venue=venue, url=url)
        + "\n\n===== 论文正文 =====\n" + text
    )
    payload = {
        "model": "qwen2.5-32b-awq",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 6000,
    }
    req = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    resp = json.load(opener.open(req, timeout=1200))
    return resp["choices"][0]["message"]["content"]


def main():
    import csv
    rows = {r["id"]: r for r in csv.DictReader(open(REPO / "papers/current_papers.tsv"), delimiter="\t")}
    en_pdfs = {p.stem.split("_")[0]: p for p in (REPO / "pdfs/en").glob("*.pdf")}
    for pid, (slug, venue) in PAPERS.items():
        out = REPO / "notes" / "{}_{}.md".format(pid, slug)
        if list((REPO / "notes").glob(pid + "_*.md")):
            print("skip " + pid, flush=True)
            continue
        pdf = en_pdfs.get(pid)
        if not pdf:
            print("NOPDF " + pid, flush=True)
            continue
        t0 = time.time()
        try:
            text = pdf_text(pdf)
            md = gen(pid, rows[pid]["title"], rows[pid]["url"], venue, text)
            out.write_text(md)
            print("OK {} {}s {}chars".format(pid, int(time.time() - t0), len(md)), flush=True)
        except Exception as e:
            print("FAIL {} {}".format(pid, e), flush=True)
    print("NOTES_DONE", flush=True)


if __name__ == "__main__":
    main()
