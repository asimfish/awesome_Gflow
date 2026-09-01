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
