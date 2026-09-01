#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为超长参考材料（N061 博士论文 259 页、O01 讲义 480 页）抽取核心章节节选 PDF。
全译成本与价值不匹配（见 DELIVERY.md），改为翻译与 GFlowNet 最相关的章节。"""
import sys
from pathlib import Path

sys.path.insert(0, "/Users/liyufeng/Code/_gflow_ref/super_translate/.venv/lib/python3.13/site-packages")
import pymupdf

REPO = Path("/Users/liyufeng/Code/awesome_Gflow")
EN = REPO / "pdfs/en"
OUT = REPO / "pdfs/en_excerpt"
OUT.mkdir(exist_ok=True)

# (输出名, [(起页, 止页, 章节说明)])  页码为大纲中的 1-based 页码
JOBS = {
    "N061": ("N061_excerpt_intro_and_gfn_background", [
        (20, 38, "Introduction + Chapter 1 Background"),
        (39, 58, "Part I. Generative Flow Networks 开篇"),
    ]),
    "O01": ("O01_excerpt_kantorovich_and_dynamic_ot", [
        (50, 71, "Kantorovich Relaxation（O08 的 OT 等价性所依赖的形式）"),
        (316, 342, "Dynamic Optimal Transport（Benamou-Brenier 动态形式）"),
    ]),
    # N032 的 p23-30 是扩散生成图像密集的 Additional Figures，MuPDF 在该区间的对象树
    # 遍历会陷入长时间 CPU 密集（实测 20 分钟无进展），跳过后正文与数学附录可正常翻译。
    "N032": ("N032_excerpt_main_and_soc_appendix", [
        (1, 22, "正文全部（方法、memoryless schedule、Adjoint Matching）"),
        (37, 44, "附录 C/D：随机最优控制等价于连续时间最大熵 RL + memoryless 证明"),
    ]),
}


def main():
    for pid, (outname, ranges) in JOBS.items():
        src = list(EN.glob(pid + "_*.pdf"))[0]
        doc = pymupdf.open(src)
        out = pymupdf.open()
        total = 0
        for a, b, _desc in ranges:
            a0, b0 = a - 1, min(b - 1, doc.page_count - 1)
            out.insert_pdf(doc, from_page=a0, to_page=b0)
            total += b0 - a0 + 1
        dst = OUT / (outname + ".pdf")
        out.save(dst, garbage=4, deflate=True)
        secs = "; ".join(f"p{a}-{b} {d}" for a, b, d in ranges)
        print(f"{pid}: {total} 页 -> {dst.name} [{secs}]", flush=True)


if __name__ == "__main__":
    main()
