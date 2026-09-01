#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按应用方向统计目录覆盖，找出 0 篇或明显偏少的方向。
用途：定期跑一次，覆盖为 0 的方向要么是该领域真没人做（本身是研究机会），
要么是检索盲区（如芯片布局：社区在做但 SOTA 论文标题不含 GFlowNet 关键词）。"""
import csv
import sys
from pathlib import Path

TSV = Path(__file__).resolve().parent.parent / "papers/current_papers.tsv"

DIRECTIONS = {
    "分子/药物": ["molecul", "drug", "ligand", "smiles", "protein", "peptide",
                "antibod", "rna", "dna", "biolog", "chem"],
    "材料/晶体": ["crystal", "material", "cataly", "alloy"],
    "LLM/推理": ["language model", "llm", "reasoning", "token", "gpt", "fine-tun"],
    "组合优化": ["combinatorial", "routing", "tsp", "scheduling", "sat", "knapsack",
              "graph coloring", "ant colony"],
    "因果/结构学习": ["causal", "structure learning", "bayesian network", "dag "],
    "芯片/EDA": ["chip", "floorplan", "circuit", "placement", "eda", "macro"],
    "NAS/超参搜索": ["architecture search", "nas", "hyperparameter", "autoML"],
    "主动学习/实验设计": ["active learning", "batch acquisition", "experimental design",
                    "bayesian optimization", "mutual information"],
    "推荐/检索": ["recommend", "retriev", "ranking"],
    "机器人/控制": ["robot", "control", "manipulat", "planning"],
    "金融/量化": ["financ", "portfolio", "alpha factor", "trading"],
    "安全/红队": ["red team", "safety", "adversarial", "attack", "jailbreak"],
    "视觉/图像": ["vision", "image", "video", "visual"],
}
THIN = 3  # 少于这个数视为覆盖薄


def main():
    with open(TSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"目录共 {len(rows)} 篇\n")
    print(f"{'方向':<20}{'篇数':>5}  编号（最多 10 个）")
    print("-" * 76)
    thin = []
    for name, pats in DIRECTIONS.items():
        hits = [r["id"] for r in rows
                if any(p in (r["title"] + " " + r["description"]).lower() for p in pats)]
        flag = "  ← 薄" if len(hits) < THIN else ""
        if len(hits) < THIN:
            thin.append((name, len(hits)))
        print(f"{name:<20}{len(hits):>5}  {' '.join(hits[:10])}{flag}")
    print("-" * 76)
    if thin:
        print("覆盖薄的方向（逐个判断是研究机会还是检索盲区）：")
        for n, c in thin:
            print(f"  · {n}：{c} 篇")
    else:
        print("所有方向覆盖均不少于阈值。")
    print("\n注意：关键词扫描会漏掉标题与简介都不含关键词的论文，"
          "0 篇不等于该方向不存在工作——芯片布局就是这样被漏掉的（SOTA 用 flow matching）。")


if __name__ == "__main__":
    sys.exit(main())
