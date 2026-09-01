# 贡献指南 Contributing

## 添加论文

论文目录的 source of truth 是 `surveys/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md`，`README.md` 由脚本从它生成。

**第 0 步：先查重。** 精确标题匹配会漏判——目录里的 A19 题名是《Let the Flows Tell: Solving Graph Combinatorial **Problems**...》，而原论文是《...Combinatorial **Optimization** Problems...》，差一个词就查不到。用脚本查：

```bash
python3 scripts/check_duplicate.py "论文标题" 2604.23658   # 第二个参数是 arXiv/DOI 号，可省
python3 scripts/check_duplicate.py --file candidates.txt   # 批量，每行「标题<TAB>号」
```

判定为 `LIKELY_DUP` 时不要直接补录，先人工确认是同一篇还是同名不同工作。

1. 在目录文档对应的 `###` 小节里追加一行表格：

   ```
   | **<编号>** | [论文标题](链接) · <venue> | 一句话简介（说清它解决什么、代价是什么） | <优先级> |
   ```

   编号规则：`T` 理论 / `O` 最优传输 / `A` 应用综述 / `N` 后续补录，按该系列末号 +1。

2. 同步更新机器可读目录 `papers/current_papers.tsv`（制表符分隔，字段 `id title url status description`）。

3. 重新生成 README：

   ```bash
   python3 scripts/build_readme.py
   ```

## 添加深度解读

放入 `notes/`，命名 `<编号>_<英文小写下划线短名>.md`，按现有笔记的 8 节模板写：

一句话 / 问题与动机 / 方法核心 / 理论结果 / 实验与证据 / 局限与批判 / 与谁对话 / 对后续研究的启示

写作要求：

- 论断要给论文章节号或公式号；个人判断显式标注「我的判断：」。
- 公式用 `$...$`，符号逐个解释。
- 「局限与批判」至少 3 条且具体，不写「未来工作可以进一步探索」这类空话。
- 「与谁对话」只列 3-5 篇真正有实质关系的论文，每篇说清是什么关系。

## 添加翻译 PDF

```bash
python3 scripts/translate_batch.py     # 增量，已有产物自动跳过
```

需要先有一个 OpenAI 兼容的推理端点（默认读 `http://127.0.0.1:18000/v1`，见脚本内 `--api-url`）。

## 趋势报告

放入 `insights/`。每条论断必须给来源链接与检索日期；检索不到就写「未检索到」，不要凭印象补。

## 维护脚本

| 脚本 | 什么时候跑 |
|---|---|
| `check_duplicate.py` | 补录论文前，必跑 |
| `build_readme.py` | 改了目录文档或加了新产物之后 |
| `audit_notes_facts.py` | 加了新解读之后，核对数字断言是否都能在原文找到 |
| `audit_coverage.py` | 定期跑，找覆盖为 0 或偏薄的应用方向 |

`audit_coverage.py` 报出 0 篇的方向要逐个判断是**研究机会**还是**检索盲区**。芯片/EDA 曾是 0 篇，但 `torchgfn` 内置 `chip_design` 环境说明社区在做——查下去发现该方向 SOTA 用的是 flow matching（标题不含 GFlowNet 关键词），属于盲区。

新增目录章节（如 `### 11.27`）后，记得在 `build_readme.py` 的 `SECTION_MAP` 里注册，否则新论文不会出现在 README。
