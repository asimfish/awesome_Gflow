#!/usr/bin/env python3
"""从 surveys/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md 解析论文条目，生成 awesome 风格 README.md。
可重复运行：notes/ 与 pdfs/zh/ 中已存在的产物会自动挂上链接。"""
import re
from pathlib import Path

REPO = Path("/Users/liyufeng/Code/awesome_Gflow")
CATALOG = REPO / "surveys/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md"

# 目录章节 -> README 分区（顺序即展示顺序）
SECTION_MAP = [
    ("1.1", "foundations", "奠基、训练目标与信用分配 · Foundations & Training Objectives"),
    ("1.2", "connections", "VI、RL 与状态空间扩展 · Connections to VI / RL & Extensions"),
    ("11.5", "connections", None),
    ("11.6", "connections", None),
    ("11.1", "theory", "收敛性、泛化与表达能力 · Convergence, Generalization & Expressivity"),
    ("11.2", "objectives", "训练目标与损失设计 · Training Objectives & Loss Design"),
    ("1.3", "training", "训练、探索与效率 · Training, Exploration & Efficiency"),
    ("11.3", "training", None),
    ("11.4", "noncyclic", "连续、非无环与随机扩展 · Continuous, Non-acyclic & Stochastic"),
    ("1.4", "frontier", "2026 前沿方法 · Frontier Methods (2026)"),
    ("2", "ot", "GFlowNet × 最优传输 · GFlowNet × Optimal Transport"),
    ("11.8", "ot", None),
    ("11.9", "ot", None),
    ("11.7", "diffusion", "扩散采样器与随机最优控制 · Diffusion Samplers & SOC"),
    ("3.1", "molecules", "分子、蛋白与材料 · Molecules, Proteins & Materials"),
    ("11.10", "molecules", None),
    ("11.16", "molecules", None),
    ("3.2", "structure", "结构学习与组合优化 · Structure Learning & Combinatorial Optimization"),
    ("11.12", "structure", None),
    ("11.13", "structure", None),
    ("3.3", "llm", "LLM、推理与视觉 · LLMs, Reasoning & Vision"),
    ("11.14", "llm", None),
    ("11.17", "conditional", "多目标与条件生成 · Multi-objective & Conditional Generation"),
    ("11.18", "safety", "安全、红队与对齐 · Safety, Red-teaming & Alignment"),
    ("11.19", "other", "其他应用 · Other Applications"),
    ("11.20", "eco", "评测基准与软件 · Benchmarks & Software"),
    ("11.21", "supplement", "审查流水线补录 · Additional Curated Papers"),
    ("11.22", "supplement", None),
    ("11.23", "supplement", None),
]

GROUPS = [
    ("theory-block", "理论 Theory", ["foundations", "connections", "theory", "objectives", "training", "noncyclic", "frontier"]),
    ("cross-block", "交叉方向 Cross-cutting", ["ot", "diffusion"]),
    ("app-block", "应用 Applications", ["molecules", "structure", "llm", "conditional", "safety", "other"]),
    ("eco-block", "生态 Ecosystem", ["eco", "supplement"]),
]

ROW_RE = re.compile(r"^\|\s*\*\*([TOAN]\d+[a-z]?)\*\*\s*\|\s*\[(.*?)\]\((\S+?)\)\s*(?:·|\\u00b7)?\s*(.*?)\s*\|\s*(.*?)\s*\|")
SEC_RE = re.compile(r"^###?\s+(\d+(?:\.\d+)?)\s+(.*)")


def parse_catalog():
    """返回 {section_num: [(id,title,url,status,desc), ...]}"""
    sections: dict[str, list] = {}
    cur = None
    for line in CATALOG.read_text().splitlines():
        m = SEC_RE.match(line)
        if m:
            cur = m.group(1)
            continue
        m = ROW_RE.match(line)
        if m and cur:
            pid, title, url, status, desc = m.groups()
            # 简介截到第一个句号，README 里保持一句话
            short = re.split(r"(?<=[。])", desc)[0]
            sections.setdefault(cur, []).append((pid, title.strip(), url, status.strip(), short.strip()))
    return sections


def artifact_links(pid: str) -> str:
    parts = []
    notes = list((REPO / "notes").glob(f"{pid}_*.md"))
    if notes:
        parts.append(f"[📝 深度解读](notes/{notes[0].name})")
    zh = REPO / "pdfs/zh" / f"{pid}_zh.pdf"
    if zh.exists():
        parts.append(f"[🇨🇳 中文PDF](pdfs/zh/{zh.name})")
    en = list((REPO / "pdfs/en").glob(f"{pid}_*.pdf"))
    if en:
        parts.append(f"[📄 英文PDF](pdfs/en/{en[0].name})")
    return (" " + " · ".join(parts)) if parts else ""


def anchor(text: str) -> str:
    a = re.sub(r"[^\w\u4e00-\u9fff\- ]", "", text).strip().lower().replace(" ", "-")
    return a


def main():
    sections = parse_catalog()

    # 组装 README 分区 -> 条目
    readme_secs: list[tuple[str, str, list]] = []  # (key,label,papers)
    key_index: dict[str, int] = {}
    for sec_num, key, label in SECTION_MAP:
        papers = sections.get(sec_num, [])
        if key not in key_index:
            key_index[key] = len(readme_secs)
            readme_secs.append((key, label or key, []))
        readme_secs[key_index[key]][2].extend(papers)

    total = sum(len(p) for _, _, p in readme_secs)
    n_notes = len(list((REPO / "notes").glob("*.md")))
    n_zh = len(list((REPO / "pdfs/zh").glob("*_zh.pdf")))

    out = []
    out.append("# Awesome GFlowNets [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)\n")
    out.append(
        "精选的 GFlowNet（Generative Flow Networks）论文、代码、课程与深度解读清单。"
        f"收录论文 **{total}** 篇（检索截止 2026-08-25），其中 **35** 篇核心论文配有中文深度解读"
        f"（已完成 {n_notes} 篇）与保版式中文翻译 PDF（已完成 {n_zh} 篇，由 "
        "[SuperTranslate](https://github.com/asimfish/super_translate) + Qwen2.5-32B 生成）。\n")
    out.append(
        "A curated list of GFlowNet papers, code, courses and in-depth Chinese notes. "
        "Legend: 📝 深度解读 in-depth note · 🇨🇳 中文PDF Chinese translation · 📄 英文PDF original PDF.\n")

    # 目录
    out.append("## 目录 Contents\n")
    out.append("- [综述与资源 Surveys & Resources](#综述与资源-surveys--resources)")
    for gkey, glabel, keys in GROUPS:
        out.append(f"- **{glabel}**")
        for key, label, papers in readme_secs:
            if key in keys and papers:
                out.append(f"  - [{label}](#{anchor(label)}) ({len(papers)})")
    out.append("- [趋势洞察 Trends & Insights](#趋势洞察-trends--insights)")
    out.append("- [课程与教程 Courses & Tutorials](#课程与教程-courses--tutorials)")
    out.append("- [代码库 Codebases](#代码库-codebases)")
    out.append("")

    # 综述
    out.append("## 综述与资源 Surveys & Resources\n")
    out.append("本仓库自带的系统性中文调研文档：\n")
    for f, desc in [
        ("GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md", "206 篇论文的完整目录：分类、简介、优先级与阅读路线"),
        ("GFLOWNET_THEORY_GUIDE_CN.md", "GFlowNet 理论指南：从流匹配到各训练目标的推导与对比"),
        ("GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md", "GFlowNet × 最优传输的潜力分析与四个候选研究课题"),
    ]:
        p = REPO / "surveys" / f
        if p.exists():
            out.append(f"- [{f.replace('_CN.md','').replace('_',' ').title()}](surveys/{f}) — {desc}")
    for extra in sorted((REPO / "surveys").glob("*.md")):
        if extra.name not in {"GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md", "GFLOWNET_THEORY_GUIDE_CN.md", "GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md"}:
            out.append(f"- [{extra.stem.replace('_',' ')}](surveys/{extra.name})")
    out.append("")

    # 论文分区
    for gkey, glabel, keys in GROUPS:
        for key, label, papers in readme_secs:
            if key not in keys or not papers:
                continue
            out.append(f"## {label}\n")
            for pid, title, url, status, desc in papers:
                links = artifact_links(pid)
                out.append(f"- `{pid}` [{title}]({url}) · *{status}*{links}  \n  {desc}")
            out.append("")

    # 趋势
    out.append("## 趋势洞察 Trends & Insights\n")
    ins = sorted((REPO / "insights").glob("*.md"))
    if ins:
        for p in ins:
            out.append(f"- [{p.stem.replace('_',' ')}](insights/{p.name})")
    else:
        out.append("- （生成中）")
    out.append("")

    # 课程与代码：直接指向目录文档锚点
    out.append("## 课程与教程 Courses & Tutorials\n")
    out.append("见 [资源目录 §6 课程、教程与博客](surveys/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#6-课程教程与博客)，"
               "含 Mila IFT6167、Edward Hu 教程、torchgfn tutorial 等。\n")
    out.append("## 代码库 Codebases\n")
    out.append("| 库 | 说明 |")
    out.append("|---|---|")
    out.append("| [GFNOrg/torchgfn](https://github.com/GFNOrg/torchgfn) | PyTorch GFlowNet 库，官方维护，含 tutorials |")
    out.append("| [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet) | Recursion 的分子生成 GFlowNet 实现 |")
    out.append("| [GFNOrg/gfn-lm-tuning](https://github.com/GFNOrg/gfn-lm-tuning) | GFlowNet 微调 LLM 参考实现 |")
    out.append("| [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet) | 面向科学发现的 GFlowNet 框架（Crystal-GFN 等） |")
    out.append("")

    out.append("## 报告 Reports\n")
    out.append("- [HTML 汇报（PPT 风格）](slides/index.html)")
    out.append("- [Beamer PDF 报告](slides/awesome_gflownets_report.pdf)")
    out.append("")
    out.append("## 贡献 Contributing\n")
    out.append("欢迎 PR：新论文按 `编号 | 标题链接 | venue | 一句话简介` 追加到对应分区；"
               "深度解读放入 `notes/`，命名 `<ID>_<slug>.md`。\n")
    out.append("## License\n\n[CC0](LICENSE)\n")

    (REPO / "README.md").write_text("\n".join(out))
    print(f"README written: {total} papers, {n_notes} notes, {n_zh} zh pdfs")


if __name__ == "__main__":
    main()
