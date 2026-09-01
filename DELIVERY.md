# 交付说明 Delivery Notes

本文件记录这批交付物是**怎么产生的**、**哪些地方不可全信**，以便复核与增量维护。

## 1. 交付清单

| 产物 | 位置 | 数量 | 生成方式 |
|---|---|---|---|
| 论文分类目录 | `README.md`, `papers/current_papers.tsv` | 206 篇 | 人工调研（检索截止 2026-08-25），README 由 `scripts/build_readme.py` 从目录 markdown 生成 |
| 核心论文清单 | `papers/core_papers.json` | 35 篇 | 从目录中标记为 P0 精读的条目抽取 |
| 中文深度解读 | `notes/*.md` | 35 篇 | 本地 vLLM（Qwen2.5-32B-AWQ）读 PDF 全文生成初稿，固定 8 节模板 |
| 英文原文 PDF | `pdfs/en/` | 35 篇 | `scripts/download_core_pdfs.py`（arXiv API 标题检索；T32 从 ICLR proceedings 直取） |
| 中文翻译 PDF | `pdfs/zh/` | 见 README 计数 | SuperTranslate 保版式引擎 + 同一 vLLM 后端 |
| 趋势报告 | `insights/` | 3 份 | web 检索 + GitHub/PyPI API 实测，每条附来源链接与日期 |
| HTML 汇报 | `slides/index.html` | 13 页 | 手写，键盘 ←/→ 翻页 |
| Beamer PDF 汇报 | `slides/awesome_gflownets_report.pdf` | 15 页 | XeLaTeX，源码 `slides/awesome_gflownets_report.tex` |

## 2. 已知局限（重要）

### 2.1 解读笔记是机器初稿

`notes/` 下的解读由 32B 模型从论文全文生成，**没有逐篇人工校对**。已做的质量控制：

- 提示词强制 8 节模板、禁用套话、要求标注章节/公式号、「局限与批判」至少 3-4 条。
- 「与谁对话」一节曾出现模型把编号清单原样抄回的问题，已用 `scripts/fix_dialogue_section.py` 对 11 篇单独重生成，改为只列 3-5 篇并说明关系。
- 三篇超长文档（T12 论文附录长、O01 讲义 200+ 页、N061 博士论文）超出 16K 上下文，改用 `scripts/generate_notes_long.py` 按页采样抽取，O01 与 N061 写成「文献地图」而非逐节解读。

**使用建议**：把笔记当作「快速定位与判断是否值得读原文」的索引，涉及具体定理条件与实验数字时回查 `pdfs/en/` 原文。T12 的笔记明显短于其他篇（约 40 行），是抽取受限所致。

### 2.2 翻译 PDF 的保真边界

SuperTranslate 冻结公式、表格、算法伪代码与引用标记后按原坐标回填，不重排版面。观察到的问题：

- 少量页面会残留英文段落（引擎日志报 `translated body block still looks like English after retry`）。
- 双栏论文的版式识别置信度在 0.7 左右，个别块位置可能偏移。
- 字体子集化告警（`feat/meta/morx NOT subset`）不影响阅读。

翻译由本地 32B 模型完成，专业术语准确度低于人工；**引用论文观点时以英文原文为准**。

### 2.2.1 中文翻译的覆盖：32 篇全文 + 2 篇节选 + 1 篇未完成

| 编号 | 材料 | 页数 | 中文 PDF 状态 | 原因 |
|---|---|---|---|---|
| N061 | Deleu 博士论文《Generative Flow Networks: Theory and Applications to Structure Learning》 | 259 | 节选 39 页：Introduction + Chapter 1 Background + Part I 开篇 | 页数 |
| O01 | 《Optimal Transport for Machine Learners》讲义 | 480 | 节选 49 页：Kantorovich Relaxation（O08 的 OT 等价性所依赖的形式）+ Dynamic Optimal Transport（Benamou-Brenier） | 页数 |
| N032 | Adjoint Matching（ICLR 2025 Spotlight） | 55 | **无中文 PDF** | 见下 |

N061 与 O01 是**页数原因**：单卡 32B 的实测速度约 31 页 / 40 分钟，全译需 5-10 小时 GPU 时间，而它们的作用是「查阅特定概念」而非「通读」——解读笔记也相应写成「文献地图」而非逐节解读。英文节选在 `pdfs/en_excerpt/`，抽页范围写在 `scripts/make_excerpts.py` 的 `JOBS` 里，改范围重跑即可。

**N032 的中文翻译未完成**，两个独立故障叠加，投入约 3.5 小时后停止：

1. *MuPDF 层*：该文件 p23-30 是扩散模型生成图像密集的 Additional Figures，MuPDF 在这段的对象树遍历陷入纯 CPU 密集（实测占满一核 20 分钟无输出，`sample` 采样显示栈全在 `fz_push_try` / `pdf_name_eq` / `fz_free` 的 malloc-free 循环）。用 pymupdf 重存清理对象树（2837 → 2634 个 xref 对象）只减轻不根治；抽掉这 8 页做成 30 页节选后，MuPDF 层不再卡（CPU 时间从 20 分钟降到 5 秒）。
2. *模型层*：节选版换成纯网络等待后，仍有某个文本块让 32B 模型陷入无限重复生成——vLLM 侧持续显示 `Running: 1 reqs` 且 60+ tokens/s，客户端 `--timeout 180` 未能中断（服务端在持续产出，socket 读超时不触发），缓存文件 21 分钟零增长。

**N032 的可用产物**：英文全文 `pdfs/en/N032_*.pdf`、英文节选 `pdfs/en_excerpt/N032_excerpt_main_and_soc_appendix.pdf`、中文深度解读 `notes/N032_adjoint_matching_soc_finetuning.md`。若要重试，建议换一个更大的模型或改用商业 API（问题出在模型的重复生成而非流水线），命令模板见 `scripts/translate_excerpts.py`。

节选 PDF 的位置与命名：英文节选在 `pdfs/en_excerpt/`，中文译文在 `pdfs/zh/` 且文件名带 `_excerpt_` 标记，例如 `O01_excerpt_kantorovich_and_dynamic_ot_zh.pdf`。抽页范围写在 `scripts/make_excerpts.py` 的 `JOBS` 里，改范围后重跑该脚本即可。两篇的**英文全文仍在 `pdfs/en/`**，需要其他章节时直接查原文或调整抽页范围重译。

### 2.3 趋势报告的时效性

三份报告都是 **2026-09-01 单日快照**：

- `trends_methods.md`、`trends_neighbors.md`：web 检索，结论依赖当时可检索到的论文。
- `trends_applications.md`：GitHub / PyPI API 当日直读，star 与 issue 数会漂移；该报告自己给出了复核建议——优先看 `pushed_at` 与提交分布而非 star 数。

报告中明确写「未检索到」的条目，指的是当日检索不到，不等于不存在。

### 2.4 目录本身的覆盖边界

206 篇的检索截止 **2026-08-25**，之后的论文不在目录内。趋势报告里引用的目录外新论文（RapTB、Stable GFlowNets、DTB、alpha-GFN 等）尚未并入 `papers/current_papers.tsv`，需要按 `CONTRIBUTING.md` 的流程补录。

## 3. 复现环境

翻译与解读都需要一个 OpenAI 兼容端点，默认读 `http://127.0.0.1:18000/v1`，模型名 `qwen2.5-32b-awq`。本次使用的部署：

```bash
vllm serve <path>/Qwen2.5-32B-Instruct-AWQ \
  --served-model-name qwen2.5-32b-awq \
  --port 18000 --host 127.0.0.1 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 --max-num-seqs 16
```

单卡 A100 80G，AWQ 量化。踩过的坑：vLLM 0.15.1 与 FastAPI 0.141 / Starlette 0.52 不兼容，所有请求返回 500（`'_IncludedRouter' object has no attribute 'path'`），需降到 `fastapi==0.128.0` + `starlette<0.50`。

翻译脚本的两个关键参数（长文档必须调）：`--timeout 900`、`--max-batch-chars 800`。默认 60 秒超时在 32B 上会大量失败。

## 4. 脚本用途

| 脚本 | 作用 | 增量行为 |
|---|---|---|
| `download_core_pdfs.py` | arXiv 标题检索 → 下载英文 PDF | 已下载的跳过，进度记在 `papers/core_papers_arxiv.json` |
| `translate_batch.py` | 并行调 SuperTranslate 生成中文 PDF | 已有 `<ID>_zh.pdf` 的跳过；并发数由 `TRANSLATE_WORKERS` 控制 |
| `generate_notes.py` | PDF 全文 → 解读初稿 | 已有 `notes/<ID>_*.md` 的跳过 |
| `generate_notes_long.py` | 超长文档的按页采样版本 | 同上 |
| `fix_dialogue_section.py` | 检测并重写「与谁对话」节 | 只处理条目多且平均长度短的（抄清单特征） |
| `build_readme.py` | 目录 markdown → README | 每次全量重建，自动挂上已存在的产物链接 |
