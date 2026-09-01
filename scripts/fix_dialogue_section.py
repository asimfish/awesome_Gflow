#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rewrite the "与谁对话" section for notes where the model copied the id list verbatim."""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/Users/liyufeng/Code/awesome_Gflow")
API = "http://127.0.0.1:18000/v1/chat/completions"

CATALOG = {
    "T01": "GFlowNet 原始论文，DAG 上流匹配，NeurIPS 2021",
    "T02": "GFlowNet Foundations，统一数学框架，JMLR 2023",
    "T03": "Trajectory Balance，轨迹级恒等式，NeurIPS 2022",
    "T05": "SubTB，子轨迹平衡连接 DB 与 TB，ICML 2023",
    "T07": "Towards Understanding GFN Training，loss 与分布误差的鸿沟诊断，ICML 2023",
    "T08": "GFlowNets and Variational Inference，与层次变分推断的比较，ICLR 2023",
    "T09": "变分视角统一解释 TB，TMLR 2023",
    "T12": "连续 GFlowNet 理论，测度空间上的流守恒，ICML 2023",
    "T14": "GFN 等价于熵正则 RL，AISTATS 2024",
    "T15": "离散概率推断作为多路径环境中的控制，UAI 2024",
    "T17": "散度训练 GFlowNets，ICML 2026",
    "T19": "非无环 GFlowNet 理论，含环状态空间",
    "T32": "何时学对分布，loss 到对象分布误差，ICLR 2025 Spotlight",
    "T36": "离散环境下重访非无环 GFlowNet，ICML 2025",
    "T49": "f-Trajectory Balance，f-散度族训练目标，ICML 2026",
    "O01": "最优传输讲义（熵正则 OT、Sinkhorn、Schrödinger 桥）",
    "O07": "用 GFlowNet 学最短路，ICML 2026 SPIGM Workshop",
    "O08": "GFlowNet 隐式学到最优传输计划，ICML 2026 SPIGM Workshop",
    "A01": "GFlowNet 用于 AI 驱动的科学发现，Digital Discovery 2023",
    "N007": "分布式 GFlowNet 与分位数流，TMLR 2024",
    "N010": "改进扩散采样器的 off-policy 训练，NeurIPS 2024",
    "N011": "GAFN，生成增强流网络（内在奖励探索），ICLR 2023",
    "N015": "CFlowNets，连续控制，ICLR 2023",
    "N018": "TBA，异步轨迹平衡用于 LLM 后训练，NeurIPS 2025",
    "N019": "FlowRL，匹配奖励分布做 LLM 推理，ICLR 2026",
    "N029": "学习式扩散采样",
    "N032": "Adjoint Matching，用无记忆随机最优控制微调流/扩散模型，ICLR 2025 Spotlight",
    "N038": "图上广义 Schrödinger 桥，预印本 2026-02",
    "N041": "非平衡图间最优传输计划的无监督学习，NeurIPS 2025",
    "N043": "Diffusion Schrödinger Bridge Matching",
    "N046": "Categorical Schrödinger Bridge Matching",
    "N061": "Deleu 博士论文，GFlowNet 理论与结构学习应用",
    "N082": "路径依赖的离散摊销推断，ICML 2026 Oral",
    "N099": "torchgfn，PyTorch GFlowNet 库",
    "N103": "经由一般软算子与鲁棒 RL 的离散组合生成，ICLR 2026",
}

CATALOG_TEXT = "\n".join(f"{k}: {v}" for k, v in CATALOG.items())

SYSTEM = (
    "你是 GFlowNet 领域的资深研究员。任务：为一篇论文写「与谁对话」小节。"
    "要求：只挑 3-5 篇真正有实质关系的论文，每篇写 1-2 句说明是什么关系"
    "（继承/推广了什么、纠正了什么、被谁的结论限制、在同一问题上给出不同答案等）。"
    "禁止列举一长串编号而不说关系。禁止套话。只输出 markdown 列表条目，不要标题、不要前言。"
    "格式：- **编号**（论文简称）：关系说明。"
)


def gen_section(pid, title, brief):
    user = (
        f"目标论文：{pid} 《{title}》。\n"
        f"该论文的核心内容摘要（来自已写好的解读）：\n{brief}\n\n"
        f"可引用的目录论文清单：\n{CATALOG_TEXT}\n\n"
        f"请写出 {pid} 的「与谁对话」小节（3-5 条，不含 {pid} 自己）。"
    )
    payload = {
        "model": "qwen2.5-32b-awq",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0.4,
        "max_tokens": 1200,
    }
    req = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return json.load(opener.open(req, timeout=600))["choices"][0]["message"]["content"].strip()


def needs_fix(body: str) -> bool:
    """判定该节是否只是编号清单而没说清关系。"""
    items = [l.strip() for l in body.splitlines() if re.match(r"^\s*(?:[-*]|\d+\.)", l)]
    if len(items) < 3:
        return False
    avg = sum(len(i) for i in items) / len(items)
    # 平均条目短于 48 字符 → 只有「编号 + 论文简称」，没有关系说明
    return avg < 48


def main():
    fixed = 0
    for md in sorted((REPO / "notes").glob("*.md")):
        pid = md.name.split("_")[0]
        text = md.read_text()
        m = re.search(r"(## 与谁对话\n)(.*?)(?=\n## |\Z)", text, re.S)
        if not m:
            continue
        if not needs_fix(m.group(2)):
            continue
        title_m = re.match(r"# \S+ · (.+)", text)
        title = title_m.group(1) if title_m else pid
        # 摘要 = 一句话 + 问题与动机 + 方法核心前 800 字
        brief_parts = []
        for sec in ["一句话", "问题与动机", "方法核心"]:
            sm = re.search(r"## " + sec + r"\n(.*?)(?=\n## |\Z)", text, re.S)
            if sm:
                brief_parts.append(sm.group(1).strip()[:900])
        brief = "\n".join(brief_parts)
        try:
            new = gen_section(pid, title, brief)
            text = text[: m.start(2)] + new + "\n" + text[m.end(2):]
            md.write_text(text)
            fixed += 1
            print("FIXED %s (%d chars)" % (pid, len(new)), flush=True)
        except Exception as e:
            print("FAIL %s %s" % (pid, e), flush=True)
    print("DIALOGUE_DONE fixed=%d" % fixed, flush=True)


if __name__ == "__main__":
    main()
