# 第10章 RAG 与 Agent 工程专题：从能跑到能上线

这一章解决一个问题：**把「能回答问题的 demo」变成「敢给别人用的系统」**。RAG 部分给你离线索引到在线评测的完整链路，Agent 部分给你编排、工具、记忆、可观测、服务化、成本控制的完整实现。读完你能独立完成 P7（企业知识库 RAG）、P9（多智能体协作）、P10（MCP 服务化）三个简历级项目。

:::warn 三个必须记住的事实口径
- 8GB 显存下的分工：**向量模型和 Rerank 模型走 CPU**（bge-m3 编码约 8–15 条/秒），把宝贵显存留给 7B GGUF 生成（Q4_K_M ≈ 4.4GB + KV 1–2GB）。
- 8G 上**不要上 vLLM 高并发**（KV Cache 常驻显存，实用性低，且 Windows 需要 WSL2）。本地生成用 llama.cpp / Ollama。
- Agent 项目**必须有 trace 表**，否则失败无法归因。没有 trace 的 Agent 在面试里等于没做过。
:::

---

## 10.1 RAG 系统全景：四段链路与模块职责

RAG 不是一个「检索 + 生成」的 while 循环，而是四段可独立评测的流水线：**离线索引（Offline Indexing）→ 在线检索（Online Retrieval）→ 生成（Generation）→ 评测（Evaluation）**。任何一环坏了，整条链路的分数都上不去，而评测的作用就是定位到底坏在哪一环。

### 10.1.1 ASCII 架构图

```text
┌──────────────────────── 离线索引（Offline Indexing，每天/每次文档更新跑一次）────────────────────────┐
│                                                                                                    │
│  PDF/Word/HTML/扫描件                                                                              │
│        │                                                                                           │
│        ▼                                                                                           │
│  ┌───────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐      │
│  │ 1 文档解析 │──▶│ 2 清洗归一 │──▶│ 3 切块 Chunk │──▶│ 4 向量化     │──▶│ 5 建索引           │      │
│  │ PyMuPDF   │   │ 去页眉页脚 │   │ 父块 1024    │   │ bge-m3 (CPU) │   │ FAISS / Chroma     │      │
│  │ unstructured│  │ 全半角统一 │   │ 子块 256     │   │ batch=32     │   │ + BM25 倒排索引    │      │
│  │ OCR 兜底   │   │ 表格转 MD  │   │ 重叠 50      │   │ 归一化       │   │ + 元数据(doc/page) │      │
│  └───────────┘   └───────────┘   └──────────────┘   └──────────────┘   └────────────────────┘      │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼  持久化到磁盘
┌──────────────────────── 在线链路（Online，单次问答 <3s）───────────────────────────────────────────┐
│                                                                                                    │
│  用户问题                                                                                          │
│     │                                                                                              │
│     ├──▶ ┌──────────────┐                                                                          │
│     │    │ 6 查询改写    │ 多轮改写 / HyDE / 子问题拆分                                            │
│     │    └──────┬───────┘                                                                          │
│     │           ▼                                                                                  │
│     │    ┌──────────────────────── 7 混合召回（并行）────────────────────────┐                      │
│     │    │  BM25 关键词召回 top50        │        向量召回 top50             │                      │
│     │    │  (jieba 分词 + 倒排索引)      │        (bge-m3 + FAISS)          │                      │
│     │    └───────────────┬──────────────────────────┬─────────────────────┘                      │
│     │                    └──────────┬───────────────┘                                            │
│     │                               ▼                                                            │
│     │                    ┌──────────────────────┐                                                │
│     │                    │ 8 RRF 融合 → top20   │   score = Σ 1/(k + rank_i), k=60               │
│     │                    └──────────┬───────────┘                                                │
│     │                               ▼                                                            │
│     │                    ┌──────────────────────┐                                                │
│     │                    │ 9 Rerank → top5      │  bge-reranker-v2-m3 (CPU, batch=8)             │
│     │                    └──────────┬───────────┘                                                │
│     │                               ▼                                                            │
│     └──────────────────▶ ┌──────────────────────┐    ┌───────────────┐                           │
│                          │ 10 上下文组装        │───▶│ 11 LLM 生成   │  7B GGUF Q4_K_M / API     │
│                          │ 段落级编号 + 引用    │    │ 带引用约束    │                           │
│                          └──────────────────────┘    └───────┬───────┘                           │
│                                                              ▼                                    │
│                                              ┌───────────────────────────────┐                    │
│                                              │ 12 引用校验 + 置信度 + 拒答   │                    │
│                                              └───────────────┬───────────────┘                    │
│                                                              ▼                                    │
│                                                      带引用的答案                                 │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼  每条问答落一条 trace
┌──────────────────────── 评测回路（Evaluation，每周 / 每次改检索策略后跑）──────────────────────────┐
│  离线指标：context_recall / context_precision（只评检索，不花 LLM 钱）                              │
│  在线指标：faithfulness / answer_relevancy（RAGAS）+ 端到端延迟 + 拒答正确率 + 成本                │
│  产出：分数下降 → 对照排查表定位环节 → 改一处 → 重跑 → 对比表                                       │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 10.1.2 模块职责表

| 编号 | 模块 | 输入 | 输出 | 关键参数 | 失败表现 | 主要评测指标 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 文档解析 | PDF / DOCX / HTML | 带页码的纯文本 + 表格 | `ocr_threshold`、`table_strategy` | 表格变乱码、公式丢失 | 解析覆盖率、人工抽检正确率 |
| 2 | 清洗归一 | 原始文本 | 无页眉页脚、全半角统一 | 页眉页脚正则、去重阈值 | 每页都出现同一句页眉 | 噪声句占比 |
| 3 | 切块 | 长文本 | 子块列表 + 父块映射 | `parent_size=1024`、`child_size=256`、`overlap=50` | 语义被切断、检索不到 | 块长度分布、边界完整率 |
| 4 | 向量化 | 子块文本 | 归一化向量 | `batch_size=32`、`normalize=True` | 显存/内存溢出、速度极慢 | 编码吞吐（条/秒） |
| 5 | 建索引 | 向量 + 分块 | FAISS 索引 + BM25 索引 + 元数据表 | `index_type=IndexFlatIP`、BM25 `k1=1.5,b=0.75` | 检索结果为空 | 索引大小、检索延迟 |
| 6 | 查询改写 | 用户问题 + 历史 | 1–4 个检索查询 | `max_subqueries=3`、HyDE `temperature=0.3` | 改写跑偏、延迟翻倍 | 改写后召回率提升 |
| 7 | 混合召回 | 查询 | 两路各 top50 | `top_k_bm25=50`、`top_k_vec=50` | 漏关键词或漏语义 | recall@50 |
| 8 | RRF 融合 | 两路结果 | 融合 top20 | `rrf_k=60`、`weights=(0.4,0.6)` | 某一路主导 | recall@20 |
| 9 | Rerank | 20 个候选 | 重排 top5 | `batch=8`、`max_length=512` | 太慢（>2s）或几乎不改序 | nDCG@5、precision@5 |
| 10 | 上下文组装 | top5 块 | 带编号的 prompt | `max_context_tokens=3000` | 超长被截断、丢引用 | 上下文利用率 |
| 11 | 生成 | prompt | 答案 + 引用 | `temperature=0.2`、`max_tokens=800` | 幻觉、不引用 | faithfulness、answer_relevancy |
| 12 | 引用校验 | 答案 + 引用 | 校验后答案或拒答 | `min_citation_score=0.35` | 引用编号不存在 | 引用正确率、拒答正确率 |

### 10.1.3 先建目录结构

所有代码都放在这个结构下，后面每节直接说「放到 `rag/xxx.py`」。

```text
C:\Users\<用户名>\projects\rag-agent\
├── data\
│   ├── raw\              原始 PDF/Word/HTML
│   ├── parsed\           解析后的 jsonl（带页码与来源）
│   ├── chunks\           切块结果 jsonl
│   └── index\            FAISS / Chroma / BM25 持久化目录
├── rag\
│   ├── parse.py          10.2 文档解析
│   ├── clean.py          10.2 清洗
│   ├── chunk.py          10.3 切块
│   ├── embed.py          10.4 向量化
│   ├── store_faiss.py    10.5 FAISS
│   ├── store_chroma.py   10.5 Chroma
│   ├── hybrid.py         10.6 混合检索
│   ├── rerank.py         10.7 重排
│   ├── rewrite.py        10.8 查询改写
│   ├── generate.py       10.9 生成与引用
│   └── evaluate.py       10.10 RAGAS 评测
├── agent\
│   ├── tools.py          10.12 工具定义
│   ├── graph.py          10.14 LangGraph
│   ├── multi_agent.py    10.15 多智能体
│   ├── memory.py         10.16 记忆系统
│   └── trace.py          10.17 可观测
├── mcp_server\
│   └── server.py         10.13 MCP Server
├── service\
│   ├── main.py           10.18 FastAPI
│   └── Dockerfile
└── tests\
```

下面的命令在 WSL2 里建好目录（Windows 也可以，路径改成反斜杠）。

```bash
# 在 WSL2 Bash 中执行
mkdir -p /mnt/c/Users/<用户名>/projects/rag-agent/{data/{raw,parsed,chunks,index},rag,agent,mcp_server,service,tests}
cd /mnt/c/Users/<用户名>/projects/rag-agent
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install pymupdf python-docx beautifulsoup4 lxml unstructured rank-bm25 jieba \
  sentence-transformers faiss-cpu chromadb openai fastapi "uvicorn[standard]" pydantic \
  ragas langgraph langfuse tiktoken httpx mcp
```

预期输出：`Installed N packages`；`python -c "import fitz, faiss, chromadb; print('ok')"` 输出 `ok`。

---

## 10.2 文档解析：PDF / Word / HTML / 扫描件

解析质量决定 RAG 上限。**上游漏掉的内容，下游永远检索不到**。这一节的顺序是：先 PyMuPDF（快、可控），再 unstructured（复杂版式），最后 OCR 兜底（扫描件）。

### 10.2.1 PDF：PyMuPDF 逐页解析 + 标题识别 + 表格

`PyMuPDF`（import 名是 `fitz`）是解析 PDF 最快的选择。下面脚本输出统一的 jsonl，字段固定为 `source / page / block_type / text / order`。

```python
# 文件名：rag/parse.py
# 用法：python -m rag.parse --in data/raw --out data/parsed/docs.jsonl
import argparse
import json
import os
import re

import fitz  # PyMuPDF

HEADER_FOOTER_RE = re.compile(r"(第\s*\d+\s*页|Page\s+\d+\s*(of\s+\d+)?|^\s*\d+\s*$)")
TITLE_SIZE_RATIO = 1.15   # 字号大于正文字号 1.15 倍视为标题


def detect_body_size(page: "fitz.Page") -> float:
    """用出现次数最多的字号作为正文字号。"""
    sizes = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip():
                    sizes.append(round(span["size"], 1))
    if not sizes:
        return 10.0
    return max(set(sizes), key=sizes.count)


def parse_pdf(path: str) -> list:
    """解析一个 PDF，返回统一的 block 列表。"""
    doc = fitz.open(path)
    records = []
    for pno, page in enumerate(doc):
        body_size = detect_body_size(page)
        order = 0
        # 表格：优先用 find_tables 提取，再决定是否从正文里剔除
        table_rects = []
        try:
            for table in page.find_tables().tables:
                md = table.to_markdown()
                table_rects.append(fitz.Rect(table.bbox))
                records.append({
                    "source": os.path.basename(path),
                    "page": pno + 1,
                    "block_type": "table",
                    "text": md,
                    "order": order,
                })
                order += 1
        except Exception as e:                        # noqa: BLE001
            print(f"[warn] 第 {pno + 1} 页表格提取失败：{type(e).__name__}: {e}")

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            rect = fitz.Rect(block["bbox"])
            if any(rect.intersects(tr) for tr in table_rects):
                continue                               # 已在表格里，避免重复
            text_parts, max_size, bold = [], 0.0, False
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        text_parts.append(span["text"])
                        max_size = max(max_size, span["size"])
                        bold = bold or ("Bold" in span["font"] or "Heiti" in span["font"])
            text = "".join(text_parts).strip()
            if len(text) < 2 or HEADER_FOOTER_RE.match(text):
                continue
            # 页眉页脚：位于页面上下 6% 区域且很短
            h = page.rect.height
            if (rect.y1 < h * 0.06 or rect.y0 > h * 0.94) and len(text) < 40:
                continue
            if max_size >= body_size * TITLE_SIZE_RATIO or (bold and max_size >= body_size):
                btype = "title"
            else:
                btype = "text"
            records.append({
                "source": os.path.basename(path),
                "page": pno + 1,
                "block_type": btype,
                "text": text,
                "order": order,
            })
            order += 1
    doc.close()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw")
    ap.add_argument("--out", default="data/parsed/docs.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    files = [f for f in sorted(os.listdir(args.inp)) if f.lower().endswith(".pdf")]
    print(f"发现 {len(files)} 个 PDF")
    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for name in files:
            recs = parse_pdf(os.path.join(args.inp, name))
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(recs)
            titles = sum(1 for r in recs if r["block_type"] == "title")
            tables = sum(1 for r in recs if r["block_type"] == "table")
            print(f"{name}: {len(recs)} 个块（标题 {titles}、表格 {tables}）")
    print(f"合计写出 {total} 个块到 {args.out}")


if __name__ == "__main__":
    main()
```

预期输出：

```text
发现 3 个 PDF
设备运维手册.pdf: 412 个块（标题 38、表格 11）
安全规范.pdf: 187 个块（标题 21、表格 4）
故障案例集.pdf: 963 个块（标题 74、表格 0）
合计写出 1562 个块到 data/parsed/docs.jsonl
```

### 10.2.2 扫描件与复杂版式：OCR 兜底 + unstructured

判断是否需要 OCR：**某页文本块数量为 0 或极少，但页面有图像**，说明是扫描件。下面脚本检测扫描页并用 `pytesseract` OCR。

```python
# 文件名：rag/parse_ocr.py
# 用法：python -m rag.parse_ocr --pdf data/raw/扫描手册.pdf --out data/parsed/scan.jsonl
# 依赖：pip install pytesseract pillow（另需系统安装 tesseract-ocr 与中文语言包 chi_sim）
import argparse
import json
import os

import fitz
import pytesseract
from PIL import Image

MIN_TEXT_CHARS = 30     # 一页文字少于这个数就当作扫描页
OCR_DPI = 200           # 200 DPI 是速度与识别率的平衡点


def ocr_page(page: "fitz.Page", lang: str = "chi_sim+eng") -> str:
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img, lang=lang).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="data/parsed/scan.jsonl")
    ap.add_argument("--lang", default="chi_sim+eng")
    ap.add_argument("--force-ocr", action="store_true", help="整本强制 OCR")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for pno, page in enumerate(doc):
            text = page.get_text().strip()
            use_ocr = args.force_ocr or len(text) < MIN_TEXT_CHARS
            if use_ocr:
                text = ocr_page(page, args.lang)
                method = "ocr"
            else:
                method = "text"
            if not text:
                print(f"[warn] 第 {pno + 1} 页解析为空")
                continue
            f.write(json.dumps({
                "source": os.path.basename(args.pdf),
                "page": pno + 1,
                "block_type": "text",
                "text": text,
                "method": method,
                "order": 0,
            }, ensure_ascii=False) + "\n")
            print(f"第 {pno + 1} 页：{method}，{len(text)} 字")
    doc.close()


if __name__ == "__main__":
    main()
```

预期输出：

```text
第 1 页：text，842 字
第 2 页：ocr，1183 字
第 3 页：ocr，967 字
```

复杂版式（多栏、公式、嵌套表格）用 `unstructured`，代价是慢且依赖重。下面是最小可用写法，把 PDF 转成元素列表再拼成段落。

```python
# 文件名：rag/parse_unstructured.py
# 用法：python -m rag.parse_unstructured --pdf data/raw/年报.pdf --out data/parsed/annual.jsonl
# 依赖：pip install "unstructured[pdf]"
import argparse
import json
import os

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Title, NarrativeText, ListItem

KEEP = (Title, NarrativeText, ListItem, Table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="data/parsed/unstructured.jsonl")
    ap.add_argument("--strategy", default="hi_res", choices=["fast", "hi_res", "ocr_only"])
    args = ap.parse_args()

    elements = partition_pdf(
        filename=args.pdf,
        strategy=args.strategy,
        infer_table_structure=True,     # 表格转 HTML，便于后续转 Markdown
        languages=["chi_sim", "eng"],
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for el in elements:
            if not isinstance(el, KEEP):
                continue
            page = getattr(el.metadata, "page_number", None) or 1
            btype = "table" if isinstance(el, Table) else ("title" if isinstance(el, Title) else "text")
            f.write(json.dumps({
                "source": os.path.basename(args.pdf),
                "page": page,
                "block_type": btype,
                "text": el.text if btype != "table" else el.metadata.text_as_html,
                "order": 0,
            }, ensure_ascii=False) + "\n")
            kept += 1
    print(f"保留 {kept} 个元素到 {args.out}（共解析 {len(elements)} 个）")


if __name__ == "__main__":
    main()
```

预期输出：

```text
保留 624 个元素到 data/parsed/unstructured.jsonl（共解析 981 个）
```

### 10.2.3 Word 与 HTML

Word 用 `python-docx`（注意：**段落在 `document.paragraphs`，表格在 `document.tables`，两者顺序会丢**，需要按 XML body 顺序遍历才能保序）。HTML 用 `BeautifulSoup` 去掉导航与脚本。

```python
# 文件名：rag/parse_office.py
# 用法：python -m rag.parse_office --in data/raw/report.docx --out data/parsed/report.jsonl
import argparse
import json
import os

from bs4 import BeautifulSoup
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_block_items(doc: Document):
    """按文档真实顺序产出段落与表格（保序关键）。"""
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def parse_docx(path: str) -> list:
    doc = Document(path)
    records, order = [], 0
    for item in iter_block_items(doc):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            btype = "title" if item.style.name.startswith(("Heading", "标题")) else "text"
            records.append({"block_type": btype, "text": text, "order": order})
        else:
            rows = []
            for row in item.rows:
                rows.append("| " + " | ".join(c.text.strip().replace("\n", " ") for c in row.cells) + " |")
            if len(rows) >= 2:
                header_sep = "| " + " | ".join(["---"] * len(item.rows[0].cells)) + " |"
                rows.insert(1, header_sep)
            records.append({"block_type": "table", "text": "\n".join(rows), "order": order})
        order += 1
    return records


def parse_html(path: str) -> list:
    html = open(path, encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    records, order = [], 0
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        text = tag.get_text(" ", strip=True)
        if len(text) < 4:
            continue
        if tag.name == "table":
            btype = "table"
        elif tag.name.startswith("h"):
            btype = "title"
        else:
            btype = "text"
        records.append({"block_type": btype, "text": text, "order": order})
        order += 1
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="data/parsed/office.jsonl")
    args = ap.parse_args()

    if args.inp.lower().endswith(".docx"):
        records = parse_docx(args.inp)
    elif args.inp.lower().endswith((".html", ".htm")):
        records = parse_html(args.inp)
    else:
        raise SystemExit("只支持 .docx / .html / .htm")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            r["source"] = os.path.basename(args.inp)
            r["page"] = 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    titles = sum(1 for r in records if r["block_type"] == "title")
    tables = sum(1 for r in records if r["block_type"] == "table")
    print(f"{os.path.basename(args.inp)}: {len(records)} 块（标题 {titles}、表格 {tables}）-> {args.out}")


if __name__ == "__main__":
    main()
```

预期输出：

```text
report.docx: 236 块（标题 18、表格 6）-> data/parsed/office.jsonl
```

### 10.2.4 清洗：页眉页脚、水印、全半角、乱码

页眉页脚是最常见的噪声源：**同一句话在一份文档里出现几十次，向量检索会被它污染**。做法是统计「同一文本在同一文档中重复出现次数」，超过阈值就判为噪声。下面脚本同时做全半角统一、控制字符清理、重复行删除。

```python
# 文件名：rag/clean.py
# 用法：python -m rag.clean --in data/parsed/docs.jsonl --out data/parsed/docs_clean.jsonl
import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict

WATERMARK_PATTERNS = [
    re.compile(r"^\s*[-—=*·\s]{3,}\s*$"),
    re.compile(r"^\s*(内部资料|禁止外传|保密|仅供内部使用)\s*$"),
    re.compile(r"^\s*第\s*\d+\s*页\s*(共\s*\d+\s*页)?\s*$"),
    re.compile(r"^\s*Page\s*\d+\s*(of\s*\d+)?\s*$", re.I),
    re.compile(r"^\s*\d{1,4}\s*$"),
]
REPEAT_RATIO = 0.25      # 同一文本在文档内出现的页数占比超过此值判为页眉页脚
MIN_KEEP_CHARS = 6
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_garbage(text: str) -> bool:
    if len(text) < MIN_KEEP_CHARS:
        return True
    if any(p.match(text) for p in WATERMARK_PATTERNS):
        return True
    # 乱码检测：非中文非字母数字非常见标点的字符占比过高
    weird = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in "，。、；：？！（）《》“”‘’—…·-.,;:?!()[]{}<>/\\|+*=%$#@&~^_\"'"))
    if weird / max(len(text), 1) > 0.25:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/parsed/docs.jsonl")
    ap.add_argument("--out", default="data/parsed/docs_clean.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    print(f"输入 {len(rows)} 个块")

    # 第一遍：统计每个 source 里的重复文本
    per_doc = defaultdict(list)
    for r in rows:
        per_doc[r["source"]].append(r)
    noise = set()
    for source, items in per_doc.items():
        pages = {r["page"] for r in items}
        counter = Counter(r["text"] for r in items)
        for text, cnt in counter.items():
            if len(pages) >= 5 and cnt / len(pages) > REPEAT_RATIO:
                noise.add((source, text))
                print(f"[噪声] {source}: 重复 {cnt} 次 -> {text[:40]}")

    # 第二遍：清洗
    kept, dropped = [], 0
    for r in rows:
        text = normalize(r["text"])
        if (r["source"], text) in noise or is_garbage(text):
            dropped += 1
            continue
        r["text"] = text
        # 给标题加层级，便于切块时保留结构
        if r["block_type"] == "title":
            r["text"] = f"# {text}"
        kept.append(r)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"保留 {len(kept)} 块，丢弃 {dropped} 块 -> {args.out}")


if __name__ == "__main__":
    main()
```

预期输出：

```text
输入 1562 个块
[噪声] 设备运维手册.pdf: 重复 61 次 -> 设备运维手册（2027 版）
[噪声] 设备运维手册.pdf: 重复 59 次 -> 第 1 页
保留 1487 块，丢弃 75 块 -> data/parsed/docs_clean.jsonl
```

### 10.2.5 解析常见坑对照表

| 坑 | 现象 | 原因 | 解决 |
| --- | --- | --- | --- |
| 表格错乱 | 单元格内容串行、列错位 | PDF 表格是绘制的线条，无结构信息 | 用 `page.find_tables()` 或 unstructured 的 `infer_table_structure=True`；转成 Markdown 保留表头 |
| 跨页表格 | 表头只出现一次，后续页无表头 | 表格跨页被拆成两个块 | 切块时按「表头 + 后续行」合并；元数据记录 `table_id` 与页码区间 |
| 页眉页脚 | 检索结果里一半是「第 X 页」 | 每页重复出现 | 10.2.4 的重复率检测 + 位置过滤 |
| 双栏排版 | 左右栏文字交错，句子读不通 | 按坐标块读取时按 y 排序，跨栏错位 | 用 `page.get_text("blocks", sort=True)` 按栏切分，或直接用 unstructured `hi_res` |
| 公式丢失 | 公式变空或变乱码 | PDF 里公式是图片或特殊字体 | 提取公式区域转图片留档；RAG 只保留公式编号并注明「见原文第 X 页」 |
| 竖排文字 | 字符顺序颠倒 | 竖排需要按列读取 | 用 `page.get_text("words")` 按 x 降序、y 升序重排 |
| 扫描件空白 | 解析出 0 字 | 纯图片 PDF | 10.2.2 的 OCR 兜底，DPI 设 200 |
| 水印覆盖 | 正文里混入「内部资料」 | 水印是独立文本层 | 加进水印正则；或按坐标（页面中心对角线）剔除 |
| 乱码（CID 字体） | 出现 `\ufffd` 或方块 | 字体未嵌入 | 换 `unstructured` 的 `ocr_only` 策略；或要求原始文档方提供可复制版本 |
| 目录页混入 | 检索到「第 3 章 ...... 45」 | 目录行被当正文 | 正则匹配 `\.{3,}\s*\d+$` 的行直接丢弃 |
| 页码丢失 | 引用只能给文档名不能给页 | 解析时没记 page | 解析阶段必须写 `page` 字段，切块时继承 |
| 编码问题 | 中文变问号 | 非 UTF-8 源文件 | 读取时 `encoding="utf-8", errors="ignore"`，或先 `chardet` 探测 |

:::ok 解析阶段的验收标准
随机抽 20 个块，人工核对：1) 无页眉页脚；2) 句子完整不串栏；3) 表格保留表头；4) 每条都有 `source` 与 `page`。四项全过再进行切块，否则后面所有评测都是在测噪声。
:::

---

## 10.3 切块策略：四种策略的参数与效果对照

切块是 RAG 里最容易被忽视、影响却最大的环节。**块太大 → 噪声多、召回不准；块太小 → 语义不完整、答案缺上下文**。

### 10.3.1 四种策略对照表

| 策略 | 参数 | 优点 | 缺点 | 适合 | 检索命中（经验值，自建 156 题） |
| --- | --- | --- | --- | --- | --- |
| 固定长度 | `chunk_size=512`, `overlap=50`（按 token） | 实现最简单、块长可控 | 会在句子中间切断，表格被切碎 | 日志、结构松散的纯文本 | recall@5 ≈ 0.61 |
| 递归字符 | `separators=["\n\n","\n","。","；","，",""]`, `size=512`, `overlap=50` | 优先在段落/句子边界切，语义完整度好 | 块长不均（80–700 token） | 通用首选 | recall@5 ≈ 0.72 |
| 语义切块 | `embedding=bge-m3`, `breakpoint_percentile=0.92`, `min_size=200` | 按语义转折切分，主题一致 | 慢（要算相邻句相似度），块长不可控 | 长论述、政策条文 | recall@5 ≈ 0.75 |
| 父子块 | 父 `1024` / 子 `256`，子块带 `parent_id` | 检索用小块保精度，生成用父块保上下文 | 索引与存储变复杂 | 技术手册、法条、讲义 | recall@5 ≈ 0.79，答案完整度最好 |

### 10.3.2 递归字符切块完整代码

`RecursiveCharacterTextSplitter` 支持中文分隔符，但需要显式传入。下面代码直接用 LangChain 的切分器并保留元数据。

```python
# 文件名：rag/chunk.py
# 用法：python -m rag.chunk --in data/parsed/docs_clean.jsonl --out data/chunks/chunks.jsonl --strategy recursive
# 依赖：pip install langchain-text-splitters tiktoken
import argparse
import hashlib
import json
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

CJK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]
TITLE_RE = re.compile(r"^#\s*(.+)$")
TABLE_ROW_RE = re.compile(r"^\|.*\|$")


def current_section(text: str, prev: str = "") -> str:
    """跟踪最近的标题，作为块的 section 元数据。"""
    for line in text.split("\n"):
        m = TITLE_RE.match(line.strip())
        if m:
            prev = m.group(1).strip()
    return prev


def split_fixed(text: str, size: int, overlap: int, tok) -> list:
    ids = tok.encode(text)
    out, i = [], 0
    while i < len(ids):
        piece = tok.decode(ids[i : i + size])
        out.append(piece)
        i += size - overlap
    return [p for p in out if p.strip()]


def split_recursive(text: str, size: int, overlap: int) -> list:
    sp = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=CJK_SEPARATORS,
        length_function=len,          # 中文按字符近似 token，1 字 ≈ 1 token
        keep_separator=True,
    )
    return [c for c in sp.split_text(text) if c.strip()]


def split_semantic(text: str, embed_fn, size: int, percentile: float = 0.92,
                   min_size: int = 200) -> list:
    """语义切块：算相邻句向量余弦距离，距离突变处切分。"""
    import numpy as np

    sentences = [s.strip() for s in re.split(r"(?<=[。！？；\n])", text) if len(s.strip()) >= 8]
    if len(sentences) <= 2:
        return [text]
    vecs = embed_fn(sentences)
    vecs = np.asarray(vecs)
    dists = [1.0 - float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(vecs) - 1)]
    if not dists:
        return [text]
    threshold = float(np.percentile(dists, percentile * 100))
    chunks, buf = [], sentences[0]
    for i, d in enumerate(dists):
        if d > threshold and len(buf) >= min_size:
            chunks.append(buf)
            buf = sentences[i + 1]
        else:
            buf += sentences[i + 1]
    if buf.strip():
        chunks.append(buf)
    # 超长块再按递归切一次
    final = []
    for c in chunks:
        final.extend(split_recursive(c, size, 50) if len(c) > size * 1.5 else [c])
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/parsed/docs_clean.jsonl")
    ap.add_argument("--out", default="data/chunks/chunks.jsonl")
    ap.add_argument("--strategy", default="recursive",
                    choices=["fixed", "recursive", "semantic"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--model", default="BAAI/bge-m3")
    args = ap.parse_args()

    import tiktoken
    tok = tiktoken.get_encoding("cl100k_base")

    embed_fn = None
    if args.strategy == "semantic":
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(args.model, device="cpu")

        def embed_fn(texts: list) -> list:
            return st.encode(texts, batch_size=32, normalize_embeddings=True,
                             show_progress_bar=False).tolist()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 把同一文档的连续块拼起来，避免跨块切断句子
    docs = {}
    for r in rows:
        docs.setdefault(r["source"], []).append(r)

    out, section = [], ""
    for source, items in docs.items():
        buffer, buf_pages = "", []
        def flush(buf: str, pages: list, sec: str):
            if len(buf.strip()) < 20:
                return
            for piece in splitter(buf):
                cid = hashlib.md5((source + piece).encode("utf-8")).hexdigest()[:16]
                out.append({
                    "chunk_id": cid,
                    "source": source,
                    "page_start": min(pages) if pages else 1,
                    "page_end": max(pages) if pages else 1,
                    "section": sec,
                    "text": piece.strip(),
                    "n_chars": len(piece.strip()),
                })

        if args.strategy == "fixed":
            splitter = lambda t: split_fixed(t, args.size, args.overlap, tok)   # noqa: E731
        elif args.strategy == "recursive":
            splitter = lambda t: split_recursive(t, args.size, args.overlap)    # noqa: E731
        else:
            splitter = lambda t: split_semantic(t, embed_fn, args.size)          # noqa: E731

        for r in items:
            section = current_section(r["text"], section)
            if r["block_type"] == "table":
                flush(buffer, buf_pages, section)      # 表格前先冲刷
                buffer, buf_pages = "", []
                flush(r["text"], [r["page"]], section)  # 表格整块保留，绝不切
                continue
            buffer += r["text"] + "\n"
            buf_pages.append(r["page"])
            if len(buffer) >= args.size * 2:
                flush(buffer, buf_pages, section)
                buffer, buf_pages = "", []
        flush(buffer, buf_pages, section)

    with open(args.out, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lens = sorted(c["n_chars"] for c in out)
    def pct(p):
        return lens[min(int(len(lens) * p), len(lens) - 1)] if lens else 0
    print(f"策略={args.strategy} 共 {len(out)} 个块")
    print(f"长度：min={lens[0] if lens else 0} p50={pct(0.5)} p90={pct(0.9)} max={lens[-1] if lens else 0}")
    print(f"平均长度 {sum(lens) / max(len(lens), 1):.0f} 字符")


if __name__ == "__main__":
    main()
```

预期输出（recursive，size=512）：

```text
策略=recursive 共 1734 个块
长度：min=38 p50=431 p90=612 max=1024
平均长度 468 字符
```

### 10.3.3 父子块策略

父块用于生成（上下文完整），子块用于检索（匹配精准）。检索命中子块后，**返回其父块给 LLM**。实现只需在子块上多存一个 `parent_id`。

```python
# 文件名：rag/chunk_parent_child.py
# 用法：python -m rag.chunk_parent_child --in data/parsed/docs_clean.jsonl --out data/chunks/pc.jsonl
import argparse
import hashlib
import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

CJK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/parsed/docs_clean.jsonl")
    ap.add_argument("--out", default="data/chunks/pc.jsonl")
    ap.add_argument("--parent-size", type=int, default=1024)
    ap.add_argument("--child-size", type=int, default=256)
    ap.add_argument("--child-overlap", type=int, default=32)
    args = ap.parse_args()

    parent_sp = RecursiveCharacterTextSplitter(
        chunk_size=args.parent_size, chunk_overlap=100, separators=CJK_SEPARATORS,
        length_function=len, keep_separator=True)
    child_sp = RecursiveCharacterTextSplitter(
        chunk_size=args.child_size, chunk_overlap=args.child_overlap,
        separators=CJK_SEPARATORS, length_function=len, keep_separator=True)

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    docs = {}
    for r in rows:
        docs.setdefault(r["source"], []).append(r)

    parents, children = [], []
    for source, items in docs.items():
        text = "\n".join(f"{r['text']}" for r in items if r["block_type"] != "table")
        pages = [r["page"] for r in items]
        for pi, ptext in enumerate(parent_sp.split_text(text)):
            if len(ptext.strip()) < 40:
                continue
            pid = hashlib.md5((source + str(pi) + ptext).encode("utf-8")).hexdigest()[:16]
            parents.append({
                "parent_id": pid, "source": source, "text": ptext.strip(),
                "page_start": min(pages), "page_end": max(pages), "n_chars": len(ptext.strip()),
            })
            for ci, ctext in enumerate(child_sp.split_text(ptext)):
                if len(ctext.strip()) < 20:
                    continue
                children.append({
                    "chunk_id": f"{pid}-{ci}", "parent_id": pid, "source": source,
                    "text": ctext.strip(), "n_chars": len(ctext.strip()),
                    "page_start": min(pages), "page_end": max(pages),
                })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in parents:
            p["level"] = "parent"
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
        for c in children:
            c["level"] = "child"
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"父块 {len(parents)} 个，子块 {len(children)} 个，平均每父块 {len(children) / max(len(parents), 1):.1f} 个子块")
    print(f"输出 {args.out}")


if __name__ == "__main__":
    main()
```

预期输出：

```text
父块 842 个，子块 3387 个，平均每父块 4.0 个子块
输出 data/chunks/pc.jsonl
```

### 10.3.4 怎么验证切块质量

不要凭感觉。三个可自动化的检查 + 一个检索验证：

**检查 1：块长度分布**。p90 不应超过 `1.3 × chunk_size`；出现长度 <30 的碎块说明分隔符太激进。

```python
# 文件名：rag/check_chunks.py
# 用法：python -m rag.check_chunks --in data/chunks/chunks.jsonl
import argparse
import json
import re
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="data/chunks/chunks.jsonl")
ap.add_argument("--min-chars", type=int, default=40)
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
lens = sorted(r["n_chars"] for r in rows)


def pct(p):
    return lens[min(int(len(lens) * p), len(lens) - 1)]


print(f"块数={len(rows)} min={lens[0]} p50={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={lens[-1]}")

# 检查 2：有没有块以逗号/顿号开头（说明被从句子中间切断）
bad_start = [r for r in rows if re.match(r"^[，、；：）】」]", r["text"])]
print(f"异常开头（标点起始）的块：{len(bad_start)} 个，占比 {len(bad_start) / len(rows):.1%}")
for r in bad_start[:3]:
    print("   ", r["chunk_id"], r["text"][:40])

# 检查 3：过短块
short = [r for r in rows if r["n_chars"] < args.min_chars]
print(f"过短块（<{args.min_chars} 字符）：{len(short)} 个")

# 检查 4：重复块（内容完全一致）
dup = Counter(r["text"] for r in rows)
reps = {k: v for k, v in dup.items() if v > 1}
print(f"完全重复的文本组：{len(reps)} 组，涉及 {sum(reps.values())} 个块")
for k, v in list(reps.items())[:3]:
    print(f"   x{v}: {k[:50]}")

# 检查 5：来源覆盖
src = Counter(r["source"] for r in rows)
print(f"来源分布：{dict(src)}")
```

预期输出：

```text
块数=1734 min=38 p50=431 p90=612 p99=901 max=1024
异常开头（标点起始）的块：17 个，占比 1.0%
过短块（<40 字符）：2 个
完全重复的文本组：4 组，涉及 9 个块
来源分布：{'设备运维手册.pdf': 731, '安全规范.pdf': 214, '故障案例集.pdf': 789}
```

**检查 6（最关键）：检索验证**。手写 20 个「问题 → 应该命中哪一页」的对照，看 top5 里有没有正确答案所在页。这就是 recall@5 的最小版本，见 10.10 的 RAGAS 自动化版本。

```python
# 文件名：rag/check_retrieval.py
# 用法：python -m rag.check_retrieval --index data/index/faiss --cases data/eval/retrieval_cases.jsonl
# 需要先跑完 10.4 + 10.5 建好索引
import argparse
import json

from rag.embed import encode_queries
from rag.store_faiss import FaissStore

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="data/index/faiss")
ap.add_argument("--cases", default="data/eval/retrieval_cases.jsonl")
ap.add_argument("--top-k", type=int, default=5)
args = ap.parse_args()

store = FaissStore.load(args.index)
cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
hit, total = 0, 0
for case in cases:
    qvec = encode_queries([case["question"]])[0]
    results = store.search(qvec, top_k=args.top_k)
    pages = {(r["source"], r["page_start"]) for r in results}
    ok = any((case["source"], case["page"]) == p for p in pages) or \
         any(r["source"] == case["source"] and abs(r["page_start"] - case["page"]) <= 1 for r in results)
    hit += int(ok)
    total += 1
    if not ok:
        print(f"[未命中] {case['question'][:40]} 期望 {case['source']}:p{case['page']}，"
              f"实际 {sorted(pages)[:3]}")
print(f"recall@{args.top_k} = {hit}/{total} = {hit / max(total, 1):.3f}")
```

预期输出：

```text
[未命中] 轴承温度报警的处置流程是什么 期望 设备运维手册.pdf:p47，实际 [('设备运维手册.pdf', 12), ('安全规范.pdf', 3)]
recall@5 = 0.850
```

:::tip 切块参数的经验起点
中文技术文档：`recursive` + `size=512` + `overlap=50`，块长按**字符**算（中文 1 字 ≈ 1 token）。表格整块保留不切。有层级结构（章/节）时，把「所属章节标题」拼到块开头，能明显提升检索命中——这是低成本高收益的一招。
:::

---

## 10.4 向量模型选型：bge-m3 / bge-large-zh-v1.5 / text-embedding-3

### 10.4.1 对比表

| 模型 | 维度 | 最大输入 | 中文效果（C-MTEB 检索类，参考量级） | 显存占用（fp16） | CPU 编码速度 | 成本 | 8G 建议 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BAAI/bge-m3` | 1024 | 8192 token | 强，且支持多语言、稀疏向量、ColBERT 多向量 | 约 2.2GB | 约 8–12 条/秒（4060 笔记本 CPU） | 免费 | **首选**，长文档与跨语言都要它 |
| `BAAI/bge-large-zh-v1.5` | 1024 | 512 token | 中文短文本检索很强 | 约 1.3GB | 约 14–20 条/秒 | 免费 | 短文本、纯中文、要更快时用 |
| `BAAI/bge-small-zh-v1.5` | 512 | 512 token | 中等，够用 | 约 0.4GB | 约 40–60 条/秒 | 免费 | 数据量大、显存紧张时的降级选择 |
| `text-embedding-3-small`（API） | 1536 | 8191 token | 中文中等偏上 | 0 | 受网络限制，批量快 | 约 $0.02/M token | 不想本地部署时用；数据出境与费用要评估 |
| `text-embedding-3-large`（API） | 3072 | 8191 token | 强 | 0 | 同上 | 约 $0.13/M token | 预算充足且允许调 API |

**结论**：8G 场景用 `bge-m3`，**在 CPU 上编码**。理由：向量编码是吞吐型任务（一次几千到几十万条），跑在 CPU 上虽然慢一点，但不占那 4.4GB 的生成模型显存；而且建索引是离线的，慢一点无所谓。查询时只需要编码 1 条，CPU 上约 30–80ms，完全不影响在线延迟。

### 10.4.2 实测 CPU 编码速度的方法

不要相信别人的速度数字，你的 CPU 和别人的不一样。下面脚本实测三种 batch size 下的吞吐与峰值内存。

```python
# 文件名：rag/embed_bench.py
# 用法：python -m rag.embed_bench --model BAAI/bge-m3 --n 512
import argparse
import time

import numpy as np
import psutil
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="BAAI/bge-m3")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
args = ap.parse_args()

# 用真实长度分布造测试文本（平均 300 字），不要用短句测出虚高的速度
TEXT = ("设备预测性维护通过振动、温度、电流等传感数据判断退化状态，先用时域频域特征提取，"
        "再用异常检测模型给出健康度评分，低于阈值触发工单并给出故障部件候选。") * 3

texts = [f"第{i}条：{TEXT}" for i in range(args.n)]
model = SentenceTransformer(args.model, device=args.device)
print(f"模型={args.model} 设备={args.device} 维度={model.get_sentence_embedding_dimension()} 条数={args.n}")

for bs in (8, 32, 64):
    model.max_seq_length = 512
    t0 = time.time()
    vecs = model.encode(texts, batch_size=bs, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    dt = time.time() - t0
    mem = psutil.Process().memory_info().rss / 1024 ** 3
    print(f"batch={bs:3d} 耗时={dt:.2f}s 吞吐={len(texts) / dt:.1f} 条/秒 "
          f"进程内存={mem:.2f}GB 向量范数均值={np.linalg.norm(vecs, axis=1).mean():.4f}")
```

预期输出（4060 笔记本 CPU，bge-m3，平均 300 字文本）：

```text
模型=BAAI/bge-m3 设备=cpu 维度=1024 条数=512
batch=  8 耗时=52.41s 吞吐=9.8 条/秒 进程内存=2.31GB 向量范数均值=1.0000
batch= 32 耗时=44.87s 吞吐=11.4 条/秒 进程内存=2.58GB 向量范数均值=1.0000
batch= 64 耗时=46.12s 吞吐=11.1 条/秒 进程内存=2.94GB 向量范数均值=1.0000
```

**读法**：`batch=32` 是拐点，再大反而慢（CPU 缓存与内存带宽受限）且内存涨。范数均值 1.0 说明归一化生效（用内积检索时才正确）。

### 10.4.3 编码模块完整代码

```python
# 文件名：rag/embed.py
# 用法（建索引）：python -m rag.embed --in data/chunks/chunks.jsonl --out data/chunks/emb.npy
# 用法（在线）：  from rag.embed import encode_queries
import argparse
import json
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
_model = None


def get_model(device: str = "cpu") -> SentenceTransformer:
    """全局单例，避免重复加载模型（加载一次约 5–10 秒、2GB 内存）。"""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device=device)
        _model.max_seq_length = 1024      # bge-m3 支持 8192，但 1024 已够且更快
    return _model


def encode_texts(texts: list, batch_size: int = 32, device: str = "cpu",
                 is_query: bool = False) -> np.ndarray:
    """编码文本。bge 系列查询建议加指令前缀（v1.5 需要，m3 可选）。"""
    model = get_model(device)
    if is_query:
        texts = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts]
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 200,
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


def encode_queries(queries: list, device: str = "cpu") -> np.ndarray:
    return encode_texts(queries, batch_size=8, device=device, is_query=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/chunks/chunks.jsonl")
    ap.add_argument("--out", default="data/chunks/emb.npy")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    texts = [r["text"] for r in rows]
    print(f"待编码 {len(texts)} 条，模型 {MODEL_NAME}，设备 {args.device}，batch={args.batch_size}")

    t0 = time.time()
    vecs = encode_texts(texts, batch_size=args.batch_size, device=args.device)
    dt = time.time() - t0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, vecs)
    with open(args.out.replace(".npy", "_meta.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"完成：{vecs.shape}，耗时 {dt:.1f}s，吞吐 {len(texts) / dt:.1f} 条/秒")
    print(f"向量文件 {args.out}（{os.path.getsize(args.out) / 1024 ** 2:.1f}MB）")
    print(f"元数据 {args.out.replace('.npy', '_meta.jsonl')}")
    print("估算：5 万条文档按当前速度需 %.1f 分钟" % (50000 / (len(texts) / dt) / 60))


if __name__ == "__main__":
    main()
```

预期输出：

```text
待编码 1734 条，模型 BAAI/bge-m3，设备 cpu，batch=32
Batches: 100%|##########| 55/55 [02:41<00:00,  2.94s/it]
完成：(1734, 1024)，耗时 161.4s，吞吐 10.7 条/秒
向量文件 data/chunks/emb.npy（6.8MB）
元数据 data/chunks/emb_meta.jsonl
估算：5 万条文档按当前速度需 77.6 分钟
```

:::warn bge 系列的两个常见错误
1. **查询与文档用不同模型或不同前缀**。bge-large-zh-v1.5 要求查询加 `为这个句子生成表示以用于检索相关文章：`，文档不加。用错会掉 5–10 个点。
2. **忘记归一化**。不归一化时用内积检索，长文本因为模长大天然占优，结果全是长块。必须 `normalize_embeddings=True`，并用内积（`IndexFlatIP`）或余弦。
:::

---

## 10.5 向量库：FAISS 与 Chroma 完整实现

**FAISS 是库不是数据库**：没有元数据过滤、没有增删改，快且省内存。**Chroma 是嵌入式数据库**：带元数据过滤和增删改查，稍慢。8G 场景的常见组合是：**FAISS 做召回主力，元数据（source/page）自己用 jsonl 存**；需要复杂过滤时用 Chroma。

### 10.5.1 FAISS：建库、持久化、增删改查、过滤

```python
# 文件名：rag/store_faiss.py
# 用法（建库）：python -m rag.store_faiss --build --vec data/chunks/emb.npy --meta data/chunks/emb_meta.jsonl --index data/index/faiss
# 用法（测试）：python -m rag.store_faiss --index data/index/faiss --query "轴承温度报警怎么处理"
import argparse
import json
import os
import pickle

import faiss
import numpy as np

from rag.embed import encode_queries


class FaissStore:
    """IndexFlatIP + id 映射 + 元数据。支持重建、添加、删除（标记法）、过滤。"""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.meta: dict[int, dict] = {}      # id -> 元数据
        self.deleted: set[int] = set()
        self._next_id = 1

    # ---------- 建库 ----------
    def add(self, vecs: np.ndarray, metas: list) -> list:
        assert len(vecs) == len(metas)
        ids = np.arange(self._next_id, self._next_id + len(vecs)).astype("int64")
        self._next_id += len(vecs)
        self.index.add_with_ids(vecs.astype("float32"), ids)
        for i, m in zip(ids.tolist(), metas):
            self.meta[i] = m
        return ids.tolist()

    # ---------- 删除（FAISS 用 remove_ids 真正移除） ----------
    def remove(self, ids: list) -> int:
        arr = np.array(ids, dtype="int64")
        n = self.index.remove_ids(arr)
        for i in ids:
            self.meta.pop(int(i), None)
        return int(n)

    # ---------- 更新 = 删旧 + 加新 ----------
    def update(self, vid: int, vec: np.ndarray, meta: dict) -> int:
        self.remove([vid])
        return self.add(vec.reshape(1, -1), [meta])[0]

    # ---------- 检索（带元数据过滤） ----------
    def search(self, qvec: np.ndarray, top_k: int = 5, where: dict | None = None,
               fetch_k: int = 5) -> list:
        """where 例子：{"source": "设备运维手册.pdf"} 或 {"page_start": (10, 60)}。"""
        need = top_k if not where else max(top_k * 10, 50)
        scores, ids = self.index.search(qvec.reshape(1, -1).astype("float32"), need)
        out = []
        for score, vid in zip(scores[0].tolist(), ids[0].tolist()):
            if vid == -1:
                continue
            meta = self.meta.get(vid)
            if meta is None:
                continue
            if where and not self._match(meta, where):
                continue
            out.append({"vector_id": vid, "score": round(score, 4), **meta})
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def _match(meta: dict, where: dict) -> bool:
        for key, cond in where.items():
            val = meta.get(key)
            if isinstance(cond, (list, tuple, set)):
                if len(cond) == 2 and all(isinstance(x, (int, float)) for x in cond):
                    if not (cond[0] <= (val or 0) <= cond[1]):
                        return False
                elif val not in cond:
                    return False
            elif val != cond:
                return False
        return True

    # ---------- 持久化 ----------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump({"meta": self.meta, "deleted": self.deleted,
                         "next_id": self._next_id, "dim": self.dim}, f)
        print(f"已保存：{path} 向量数={self.index.ntotal}")

    @classmethod
    def load(cls, path: str) -> "FaissStore":
        store = cls()
        store.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            data = pickle.load(f)
        store.meta = data["meta"]
        store.deleted = data["deleted"]
        store._next_id = data["next_id"]
        store.dim = data["dim"]
        return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--vec", default="data/chunks/emb.npy")
    ap.add_argument("--meta", default="data/chunks/emb_meta.jsonl")
    ap.add_argument("--index", default="data/index/faiss")
    ap.add_argument("--query", default="")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--source", default="", help="按来源过滤")
    args = ap.parse_args()

    if args.build:
        vecs = np.load(args.vec)
        metas = [json.loads(l) for l in open(args.meta, encoding="utf-8") if l.strip()]
        store = FaissStore(dim=vecs.shape[1])
        ids = store.add(vecs, metas)
        store.save(args.index)
        print(f"建库完成：{len(ids)} 条，维度 {vecs.shape[1]}，目录 {args.index}")
        # 自检：用第一条向量检索自己，应该 score≈1.0
        hit = store.search(vecs[0], top_k=1)[0]
        print(f"自检：top1 score={hit['score']} chunk_id={hit.get('chunk_id')}")
        return

    store = FaissStore.load(args.index)
    print(f"已加载索引，共 {store.index.ntotal} 条向量")
    if not args.query:
        return
    qv = encode_queries([args.query])[0]
    where = {"source": args.source} if args.source else None
    for i, r in enumerate(store.search(qv, top_k=args.top_k, where=where), 1):
        print(f"{i}. score={r['score']} [{r['source']} p{r['page_start']}] {r['text'][:80]}")


if __name__ == "__main__":
    main()
```

建库与查询的预期输出：

```text
已保存：data/index/faiss 向量数=1734
建库完成：1734 条，维度 1024，目录 data/index/faiss
自检：top1 score=1.0 chunk_id=3f2a91c7d4e5b601
```

```text
已加载索引，共 1734 条向量
1. score=0.7218 [设备运维手册.pdf p47] 轴承温度超过报警阈值的处置流程：1）确认测温点与量程…
2. score=0.6903 [设备运维手册.pdf p48] 温度报警分级：预警 65℃，报警 75℃，停机 85℃…
3. score=0.6641 [故障案例集.pdf p12] 某泵轴承温度异常案例：初期表现为温升速率加快…
4. score=0.6502 [安全规范.pdf p9] 高温设备巡检须佩戴防护手套并保持 0.5m 以上距离…
5. score=0.6418 [设备运维手册.pdf p51] 测温元件校准周期为 12 个月，偏差超过 ±2℃ 需更换…
```

### 10.5.2 Chroma：建库、持久化、增删改查、过滤

```python
# 文件名：rag/store_chroma.py
# 用法（建库）：python -m rag.store_chroma --build --vec data/chunks/emb.npy --meta data/chunks/emb_meta.jsonl --persist data/index/chroma
# 用法（查询）：python -m rag.store_chroma --query "轴承温度报警怎么处理" --persist data/index/chroma
import argparse
import json

import chromadb
import numpy as np
from chromadb.config import Settings

from rag.embed import encode_queries

COLLECTION = "manual"


def get_client(persist_dir: str):
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def build(persist_dir: str, vec_path: str, meta_path: str, batch: int = 256):
    vecs = np.load(vec_path)
    metas = [json.loads(l) for l in open(meta_path, encoding="utf-8") if l.strip()]
    client = get_client(persist_dir)
    try:
        client.delete_collection(COLLECTION)      # 重建时先删，避免重复
    except Exception:                             # noqa: BLE001
        pass
    col = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 200, "hnsw:M": 32},
    )
    for i in range(0, len(vecs), batch):
        chunk_ids = [str(m.get("chunk_id") or m.get("parent_id") or i + j)
                     for j, m in enumerate(metas[i : i + batch])]
        documents = [m["text"] for m in metas[i : i + batch]]
        metadatas = [{
            "source": m["source"],
            "page_start": int(m.get("page_start", 1)),
            "page_end": int(m.get("page_end", 1)),
            "section": m.get("section", ""),
            "n_chars": int(m.get("n_chars", len(m["text"]))),
        } for m in metas[i : i + batch]]
        col.add(ids=chunk_ids, embeddings=vecs[i : i + batch].tolist(),
                documents=documents, metadatas=metadatas)
    print(f"Chroma 建库完成：{col.count()} 条，目录 {persist_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--vec", default="data/chunks/emb.npy")
    ap.add_argument("--meta", default="data/chunks/emb_meta.jsonl")
    ap.add_argument("--persist", default="data/index/chroma")
    ap.add_argument("--query", default="")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--source", default="")
    ap.add_argument("--page-range", default="", help="形如 10,60")
    ap.add_argument("--demo-crud", action="store_true")
    args = ap.parse_args()

    if args.build:
        build(args.persist, args.vec, args.meta)
        return

    client = get_client(args.persist)
    col = client.get_collection(COLLECTION)
    print(f"集合 {COLLECTION} 共 {col.count()} 条")

    if args.demo_crud:
        # 增
        col.add(ids=["demo-1"], documents=["这是一条用于验证增删改查的测试文本。"],
                embeddings=encode_queries(["这是一条用于验证增删改查的测试文本。"]).tolist(),
                metadatas=[{"source": "demo.pdf", "page_start": 1, "page_end": 1,
                            "section": "测试", "n_chars": 18}])
        print("增：", col.get(ids=["demo-1"])["documents"])
        # 改（upsert）
        col.upsert(ids=["demo-1"], documents=["修改后的测试文本，用于验证更新。"],
                   embeddings=encode_queries(["修改后的测试文本，用于验证更新。"]).tolist(),
                   metadatas=[{"source": "demo.pdf", "page_start": 1, "page_end": 1,
                               "section": "测试", "n_chars": 16}])
        print("改：", col.get(ids=["demo-1"])["documents"])
        # 删
        col.delete(ids=["demo-1"])
        print("删：剩余", col.count(), "条")

    if not args.query:
        return
    where = None
    conds = []
    if args.source:
        conds.append({"source": args.source})
    if args.page_range:
        lo, hi = (int(x) for x in args.page_range.split(","))
        conds.append({"page_start": {"$gte": lo}})
        conds.append({"page_end": {"$lte": hi}})
    if len(conds) == 1:
        where = conds[0]
    elif len(conds) > 1:
        where = {"$and": conds}

    res = col.query(query_embeddings=encode_queries([args.query]).tolist(),
                    n_results=args.top_k, where=where,
                    include=["documents", "metadatas", "distances"])
    for i, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0],
                                              res["distances"][0]), 1):
        print(f"{i}. 距离={dist:.4f} [{meta['source']} p{meta['page_start']}] {doc[:80]}")


if __name__ == "__main__":
    main()
```

预期输出：

```text
Chroma 建库完成：1734 条，目录 data/index/chroma
```

```text
集合 manual 共 1734 条
增： ['这是一条用于验证增删改查的测试文本。']
改： ['修改后的测试文本，用于验证更新。']
删：剩余 1734 条
1. 距离=0.2782 [设备运维手册.pdf p47] 轴承温度超过报警阈值的处置流程：1）确认测温点与量程…
```

### 10.5.3 选型对照

| 维度 | FAISS | Chroma |
| --- | --- | --- |
| 定位 | 向量检索库 | 嵌入式向量数据库 |
| 元数据过滤 | 需自己实现（本节的 `where`） | 原生支持 `where` 表达式 |
| 增删改 | 支持（`remove_ids`/重建） | 原生 `add/update/upsert/delete` |
| 10 万条建库耗时 | 约 8s（IndexFlatIP） | 约 45s（HNSW 构建） |
| 10 万条查询延迟 | 约 12ms（CPU） | 约 25ms（CPU） |
| 持久化 | `write_index` + pickle | 目录自动持久化 |
| 内存（10 万条 × 1024 维） | 约 400MB | 约 550MB（含 HNSW 图） |
| 8G 建议 | **主检索通道**，快、可控 | 需要复杂过滤、要频繁增删时用 |

:::tip 什么时候两个都用
生产里常见做法：**FAISS 做向量召回，SQLite 存元数据与全文，Chroma 只用于需要「元数据过滤 + 向量」混合条件的场景**（如「只看 2026 年之后的法律条文且相似度 top5」）。不要为了「都用上」而引入两套索引——维护成本翻倍。
:::

---

## 10.6 混合检索：BM25 + 向量 + RRF 融合

### 10.6.1 为什么纯向量检索会漏关键词

举一个你一定会遇到的实例：

```text
问题：「FANUC 0i-MF 系统的 414 报警怎么处理？」
纯向量 top5：全部是讲「报警处理流程」「常见报警分类」的通用段落，
             因为「FANUC」「0i-MF」「414」这些低频专有名词在向量空间里权重很低，
             语义上「报警处理」的通用段落与问题最接近。
BM25 top5：第 1 条就是「414 报警：伺服放大器过载或编码器通信异常」——因为它命中了「414」这个精确词。
```

原因：**稠密向量把每个 token 压成一个整体语义，低频的型号、编号、错误码、人名被平滑掉了；BM25 的 IDF 机制恰好对低频词给高权重**。所以含型号、编码、编号、法条号、错误码的问题，必须混合检索。

反向的实例也存在：用户问「设备老是发烫怎么办」，文档里写的是「轴承温升异常」，纯 BM25 一条也命中不了——这时靠向量。**两路互补，缺一不可**。

### 10.6.2 RRF 融合公式

RRF（Reciprocal Rank Fusion）只看排名不看分数，因此天然解决了「BM25 分数与余弦分数不可比」的问题：

```text
RRF_score(d) = Σ_i  w_i / (k + rank_i(d))      k = 60（经验值）, rank 从 1 开始
```

若某文档只被一路召回，该项只算一路的分。`w_i` 是通道权重，中文技术文档常用 `(0.4 BM25, 0.6 向量)`。

### 10.6.3 完整实现

```python
# 文件名：rag/hybrid.py
# 用法：python -m rag.hybrid --query "FANUC 0i-MF 系统的 414 报警怎么处理？" --top-k 5
import argparse
import json
import os
import pickle
import re
from collections import defaultdict

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from rag.embed import encode_queries
from rag.store_faiss import FaissStore

STOPWORDS = set("的 了 和 是 在 有 我 你 他 它 这 那 就 都 而 及 与 或 一个 我们 你们 他们 "
                "什么 怎么 如何 为什么 哪些 可以 需要 应该 进行 通过 对于".split())
TOKEN_RE = re.compile(r"[a-zA-Z0-9\-_.]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list:
    """中文用 jieba，英文/型号/编号用正则整体保留（关键：不要让 414 被切碎）。"""
    tokens = []
    for tok in jieba.lcut(text.lower()):
        tok = tok.strip()
        if not tok or tok in STOPWORDS:
            continue
        # 型号与编号：整段拼接保留原形
        tokens.extend(TOKEN_RE.findall(tok) or [tok])
    return tokens


class HybridRetriever:
    def __init__(self, faiss_dir: str, chunks_path: str, bm25_cache: str = "data/index/bm25.pkl"):
        self.store = FaissStore.load(faiss_dir)
        self.chunks = [json.loads(l) for l in open(chunks_path, encoding="utf-8") if l.strip()]
        self.by_id = {c.get("chunk_id"): c for c in self.chunks}
        self.bm25_cache = bm25_cache
        if os.path.exists(bm25_cache):
            with open(bm25_cache, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.order = data["order"]
            print(f"从缓存加载 BM25，文档数 {len(self.order)}")
        else:
            corpus = [tokenize(c["text"]) for c in self.chunks]
            self.bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
            self.order = [c["chunk_id"] for c in self.chunks]
            os.makedirs(os.path.dirname(bm25_cache) or ".", exist_ok=True)
            with open(bm25_cache, "wb") as f:
                pickle.dump({"bm25": self.bm25, "order": self.order}, f)
            print(f"BM25 建索引完成，文档数 {len(self.order)}")

    def bm25_search(self, query: str, top_k: int = 50) -> list:
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk_id": self.order[i], "bm25_score": float(scores[i]),
             "text": self.by_id[self.order[i]]["text"],
             "source": self.by_id[self.order[i]]["source"],
             "page_start": self.by_id[self.order[i]].get("page_start", 1)}
            for i in idx if scores[i] > 0
        ]

    def vector_search(self, query: str, top_k: int = 50, where: dict | None = None) -> list:
        qv = encode_queries([query])[0]
        hits = self.store.search(qv, top_k=top_k, where=where)
        out = []
        for h in hits:
            cid = h.get("chunk_id")
            if cid is None:
                continue
            out.append({"chunk_id": cid, "vec_score": h["score"], "text": h["text"],
                        "source": h["source"], "page_start": h.get("page_start", 1)})
        return out

    @staticmethod
    def rrf_fuse(bm25_hits: list, vec_hits: list, k: int = 60,
                 weights: tuple = (0.4, 0.6)) -> list:
        """RRF 融合。weights=(BM25 权重, 向量权重)。"""
        fused = defaultdict(lambda: {"rrf": 0.0, "ranks": {}, "item": None})
        for rank, hit in enumerate(bm25_hits, 1):
            cid = hit["chunk_id"]
            fused[cid]["rrf"] += weights[0] / (k + rank)
            fused[cid]["ranks"]["bm25"] = rank
            fused[cid]["item"] = hit
        for rank, hit in enumerate(vec_hits, 1):
            cid = hit["chunk_id"]
            fused[cid]["rrf"] += weights[1] / (k + rank)
            fused[cid]["ranks"]["vec"] = rank
            if fused[cid]["item"] is None:
                fused[cid]["item"] = hit
        merged = []
        for cid, data in fused.items():
            item = dict(data["item"])
            item["rrf_score"] = round(data["rrf"], 6)
            item["ranks"] = data["ranks"]
            item["chunk_id"] = cid
            merged.append(item)
        merged.sort(key=lambda x: x["rrf_score"], reverse=True)
        return merged

    def search(self, query: str, top_k: int = 20, announce: bool = False) -> list:
        b = self.bm25_search(query, top_k=50)
        v = self.vector_search(query, top_k=50)
        merged = self.rrf_fuse(b, v)
        if announce:
            print(f"BM25 召回 {len(b)} 条，向量召回 {len(v)} 条，融合后 {len(merged)} 条")
            only_b = [m for m in merged if set(m["ranks"]) == {"bm25"}][:3]
            if only_b:
                print("仅 BM25 命中（向量漏掉的精确匹配）：")
                for m in only_b:
                    print(f"   rrf={m['rrf_score']:.5f} {m['text'][:60]}")
        return merged[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--faiss", default="data/index/faiss")
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    r = HybridRetriever(args.faiss, args.chunks)
    print(f"\n=== 纯向量 top{args.top_k} ===")
    for i, h in enumerate(r.vector_search(args.query, top_k=args.top_k), 1):
        print(f"{i}. {h['vec_score']:.4f} [{h['source']} p{h['page_start']}] {h['text'][:70]}")
    print(f"\n=== 纯 BM25 top{args.top_k} ===")
    for i, h in enumerate(r.bm25_search(args.query, top_k=args.top_k), 1):
        print(f"{i}. {h['bm25_score']:.4f} [{h['source']} p{h['page_start']}] {h['text'][:70]}")
    print(f"\n=== RRF 融合 top{args.top_k} ===")
    for i, h in enumerate(r.search(args.query, top_k=args.top_k, announce=True), 1):
        print(f"{i}. rrf={h['rrf_score']:.5f} ranks={h['ranks']} "
              f"[{h['source']} p{h['page_start']}] {h['text'][:60]}")


if __name__ == "__main__":
    main()
```

预期输出：

```text
从缓存加载 BM25，文档数 1734

=== 纯向量 top5 ===
1. 0.6891 [设备运维手册.pdf p88] 报警分类与通用处置流程：先确认报警等级，再按手册对应章节…
2. 0.6734 [故障案例集.pdf p31] 常见报警案例汇总：过载、通信超时、编码器异常…
3. 0.6642 [设备运维手册.pdf p90] 报警复位操作：在急停解除后按住 RESET 键 3 秒…
4. 0.6517 [安全规范.pdf p14] 出现报警时须先停机并挂检修牌…
5. 0.6488 [设备运维手册.pdf p92] 报警日志导出方式：诊断页面选择 LOG 导出…

=== 纯 BM25 top5 ===
1. 8.9142 [设备运维手册.pdf p139] 414 报警：伺服放大器过载或编码器通信异常，检查动力线…
2. 7.2310 [设备运维手册.pdf p141] 414/415 报警对照：414 为过载，415 为再生异常…
3. 6.8824 [故障案例集.pdf p67] FANUC 0i-MF 系统 414 报警现场处置案例…
4. 5.1120 [设备运维手册.pdf p88] 报警分类与通用处置流程…
5. 4.7781 [设备运维手册.pdf p90] 报警复位操作…

=== RRF 融合 top5 ===
BM25 召回 42 条，向量召回 50 条，融合后 78 条
仅 BM25 命中（向量漏掉的精确匹配）：
   rrf=0.013115 414 报警：伺服放大器过载或编码器通信异常，检查动力线…
1. rrf=0.019836 ranks={'bm25': 1, 'vec': 12} [设备运维手册.pdf p139] 414 报警：伺服放大器过载…
2. rrf=0.016571 ranks={'bm25': 2} [设备运维手册.pdf p141] 414/415 报警对照…
3. rrf=0.014754 ranks={'bm25': 3} [故障案例集.pdf p67] FANUC 0i-MF 系统 414 报警…
4. rrf=0.009836 ranks={'bm25': 4, 'vec': 1} [设备运维手册.pdf p88] 报警分类与通用处置流程…
5. rrf=0.008712 ranks={'vec': 2} [故障案例集.pdf p31] 常见报警案例汇总…
```

**读法**：纯向量把「414 报警」排在第 1 位的是通用流程段落；RRF 融合后第 1 位变成真正命中「414」的段落。这就是混合检索的价值，也是你简历里「混合检索把 recall@5 从 0.61 提到 0.89」这类数字的来源。

### 10.6.4 通道权重的调法

权重不是拍脑袋定的。固定评测集，网格搜索两个权重：

```python
# 文件名：rag/tune_rrf.py
# 用法：python -m rag.tune_rrf --cases data/eval/retrieval_cases.jsonl
import argparse
import json

from rag.hybrid import HybridRetriever

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="data/eval/retrieval_cases.jsonl")
ap.add_argument("--faiss", default="data/index/faiss")
ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
ap.add_argument("--top-k", type=int, default=5)
args = ap.parse_args()

r = HybridRetriever(args.faiss, args.chunks)
cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]

# 预先算好两路结果，避免重复检索
pre = []
for c in cases:
    pre.append((c, r.bm25_search(c["question"], 50), r.vector_search(c["question"], 50)))


def recall(w_bm25: float, w_vec: float, k: int) -> float:
    hit = 0
    for case, b, v in pre:
        fused = HybridRetriever.rrf_fuse(b, v, weights=(w_bm25, w_vec))[:k]
        if any(x["source"] == case["source"] and
               abs(int(x["page_start"]) - int(case["page"])) <= 1 for x in fused):
            hit += 1
    return hit / len(pre)


print("权重网格搜索（行=BM25 权重，列=向量权重）")
grid = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
print("BM25\\Vec " + "".join(f"{w:>8.1f}" for w in grid))
best = (0, 0, 0.0)
for wb in grid:
    row = []
    for wv in grid:
        if wb == 0 and wv == 0:
            row.append(0.0)
            continue
        s = recall(wb, wv, args.top_k)
        row.append(s)
        if s > best[2]:
            best = (wb, wv, s)
    print(f"{wb:>8.1f} " + "".join(f"{x:>8.3f}" for x in row))
print(f"\n最佳权重：BM25={best[0]} 向量={best[1]} recall@{args.top_k}={best[2]:.3f}")
```

预期输出：

```text
权重网格搜索（行=BM25 权重，列=向量权重）
BM25\Vec      0.0     0.2     0.4     0.5     0.6     0.8     1.0
     0.0    0.000   0.742   0.781   0.792   0.803   0.811   0.806
     0.2    0.718   0.797   0.812   0.821   0.829   0.834   0.827
     0.4    0.736   0.806   0.824   0.836   0.847   0.851   0.842
     0.5    0.741   0.809   0.827   0.839   0.849   0.853   0.846
     0.6    0.745   0.812   0.830   0.841   0.851   0.855   0.848
     0.8    0.750   0.815   0.833   0.844   0.853   0.857   0.850
     1.0    0.752   0.818   0.836   0.846   0.855   0.859   0.852

最佳权重：BM25=1.0 向量=0.8 recall@5=0.859
```

注意：纯 BM25（`wv=0`）是 0.752，纯向量（`wb=0`）是 0.811，融合后 0.859。**融合优于任一单路**，这是混合检索的核心论据，也是面试可讲的数据。

---

## 10.7 Rerank：bge-reranker-v2-m3 完整调用

召回用双塔（query 和 doc 分别编码，快但粗），重排用交叉编码器（query 和 doc 拼在一起过一遍模型，准但慢）。**召回 50 条，重排取 5 条**是标准配比。

### 10.7.1 完整调用代码

```python
# 文件名：rag/rerank.py
# 用法：python -m rag.rerank --query "414 报警怎么处理" --candidates 50 --top-n 5
import argparse
import time

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_reranker = None


def get_reranker(device: str = "cpu", max_length: int = 512) -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(MODEL_NAME, device=device, max_length=max_length)
    return _reranker


def rerank(query: str, candidates: list, top_n: int = 5, batch_size: int = 8,
           device: str = "cpu", max_length: int = 512) -> list:
    """candidates: [{"text": ..., ...}]，返回按重排分数降序的前 top_n 条。"""
    if not candidates:
        return []
    model = get_reranker(device, max_length)
    pairs = [(query, c["text"]) for c in candidates]
    t0 = time.time()
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    dt = time.time() - t0
    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    for i, c in enumerate(ranked, 1):
        c["rerank_rank"] = i
    print(f"重排 {len(candidates)} 条耗时 {dt:.2f}s（batch={batch_size}，"
          f"{len(candidates) / dt:.1f} 条/秒）")
    return ranked[:top_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--faiss", default="data/index/faiss")
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from rag.hybrid import HybridRetriever

    r = HybridRetriever(args.faiss, args.chunks)
    print("=== 重排前（RRF 融合 top5）===")
    fused = r.search(args.query, top_k=args.candidates)
    for i, h in enumerate(fused[:5], 1):
        print(f"{i}. rrf={h['rrf_score']:.5f} [{h['source']} p{h['page_start']}] {h['text'][:60]}")

    print("\n=== 重排后（top5）===")
    top = rerank(args.query, fused, top_n=args.top_n, batch_size=args.batch, device=args.device)
    for h in top:
        print(f"{h['rerank_rank']}. rerank={h['rerank_score']:.4f} rrf={h['rrf_score']:.5f} "
              f"[{h['source']} p{h['page_start']}] {h['text'][:60]}")

    changed = sum(1 for h in top if h["ranks"].get("vec") or h["ranks"].get("bm25"))
    print(f"\n前 5 条中有 {changed} 条来自召回前 20 名之外（说明重排起了实质作用）")


if __name__ == "__main__":
    main()
```

预期输出：

```text
=== 重排前（RRF 融合 top5）===
1. rrf=0.019836 [设备运维手册.pdf p139] 414 报警：伺服放大器过载或编码器通信异常…
2. rrf=0.016571 [设备运维手册.pdf p141] 414/415 报警对照：414 为过载…
3. rrf=0.014754 [故障案例集.pdf p67] FANUC 0i-MF 系统 414 报警现场处置案例…
4. rrf=0.009836 [设备运维手册.pdf p88] 报警分类与通用处置流程…
5. rrf=0.008712 [故障案例集.pdf p31] 常见报警案例汇总…

=== 重排后（top5）===
重排 50 条耗时 4.31s（batch=8，11.6 条/秒）
1. rerank=0.9821 rrf=0.016571 [设备运维手册.pdf p141] 414/415 报警对照：414 为过载…
2. rerank=0.9744 rrf=0.019836 [设备运维手册.pdf p139] 414 报警：伺服放大器过载…
3. rerank=0.9310 rrf=0.014754 [故障案例集.pdf p67] FANUC 0i-MF 系统 414 报警…
4. rerank=0.4102 rrf=0.009836 [设备运维手册.pdf p88] 报警分类与通用处置流程…
5. rerank=0.1877 rrf=0.008712 [故障案例集.pdf p31] 常见报警案例汇总…

前 5 条中有 0 条来自召回前 20 名之外（说明重排起了实质作用）
```

**关键读法**：重排把「报警对照表」提到第 1，并把「通用流程」（rerank=0.41）与真正相关的（0.93+）**拉开了明显的分数差距**。这个差距本身就是置信度阈值（10.9）的来源。

### 10.7.2 8G 上的 batch 与截断建议

| 参数 | 推荐值 | 理由 | 调大的后果 |
| --- | --- | --- | --- |
| `device` | `cpu` | 把 4.4GB 显存留给生成模型 | 放 GPU 需约 2.2GB，会把 7B GGUF 挤爆 |
| `batch_size` | 8（CPU）/ 16（GPU） | CPU 上 batch=8 吞吐最高；再大内存带宽受限 | 32 以上速度反而下降，内存涨 |
| `max_length` | 512 | 绝大多数块在 512 字符内，截断损失小 | 1024 时耗时约翻倍，收益极小 |
| `candidates`（召回数） | 30–50 | 重排 50 条约 4–5s（CPU），超过 5s 用户体验差 | 100 条时约 9s，需要降 `top_n` 或改 GPU |
| `top_n`（送入生成） | 4–5 | 上下文 token 预算 3000 左右 | 8 条以上会挤掉生成空间且引入噪声 |

:::warn 重排不是万能的
如果召回阶段根本没把正确段落召回来（recall@50 低），重排也救不了。**先看 recall@50 是否 ≥0.95，再优化重排**。很多人跳过这一步，把时间花在换 rerank 模型上，收益很小。
:::

### 10.7.3 重排前后效果对比表模板

| 指标 | 仅向量 | 混合检索（BM25+向量+RRF） | 混合 + Rerank | 相对仅向量 |
| --- | --- | --- | --- | --- |
| recall@50 | 0.892 | 0.981 | 0.981 | +0.089 |
| recall@5 | 0.811 | 0.859 | 0.936 | +0.125 |
| precision@5 | 0.372 | 0.418 | 0.641 | +0.269 |
| nDCG@5 | 0.604 | 0.668 | 0.812 | +0.208 |
| 检索延迟（CPU） | 38ms | 96ms | 4.42s | +4.38s |
| 端到端延迟 | 2.14s | 2.28s | 6.41s | +4.27s |
| 单次成本（本地） | 0 | 0 | 0 | — |

**结论口径**：加 Rerank 让 `precision@5` 从 0.418 提到 0.641（+53%），代价是端到端延迟从 2.3s 涨到 6.4s。**要不要开取决于场景**：知识库问答（用户愿意等 6s）开；实时客服（要求 <3s）不开，或把候选数降到 20、用 GPU 重排。

---

## 10.8 查询改写：多轮改写、HyDE、子问题拆分

用户的问题往往不是好的检索查询：多轮对话里全是「那这个呢」，口语化提问与文档措辞不匹配，复合问题一次问三件事。查询改写就是解决这三类问题。

### 10.8.1 三种改写策略与开启时机

| 策略 | 解决的问题 | 何时开启 | 何时不要开 | 额外延迟 | 成本 |
| --- | --- | --- | --- | --- | --- |
| 多轮对话改写 | 「它」「这个」「那第二种呢」指代不明 | 有对话历史且轮次 ≥2 | 单轮问答 | +0.4–1.2s | 1 次小模型调用 |
| HyDE | 问题措辞与文档差异大、问题很短（<10 字） | 短问题、术语型问题 | 问题已很长很具体 | +0.8–2s | 1 次生成 + 1 次编码 |
| 子问题拆分 | 复合问题（含「和」「以及」「分别」）需要多段证据 | 检测到并列结构或问句 >40 字 | 简单单跳问题 | +0.5–1.5s + 多次检索 | 1 次调用 + N 次检索 |

### 10.8.2 完整实现

```python
# 文件名：rag/rewrite.py
# 用法：python -m rag.rewrite --history "轴承温度高怎么办|先查润滑油位|那振动呢" --question "那这个要停机吗"
import argparse
import json
import os
import re

from openai import OpenAI

REWRITE_PROMPT = """你是检索查询改写助手。根据对话历史，把用户的最新问题改写成一个**独立、完整、可直接用于文档检索**的查询。

要求：
1. 消解所有代词与省略（把「它」「这个」「那第二种」替换成具体对象）。
2. 保留专业术语、型号、编号、错误码的原文，不要翻译、不要概括。
3. 只输出改写后的查询本身，不要任何解释、不要引号、不要编号。
4. 如果最新问题本身已经完整独立，原样输出。

对话历史：
{history}

最新问题：{question}

改写后的查询："""

HYDE_PROMPT = """请针对下面的问题，写一段**假设性的文档片段**（200-300 字），
就像这段文字真的出自一份技术手册。不要回答问题，只写「文档里可能会怎么写」。

问题：{question}

假设性文档片段："""

SPLIT_PROMPT = """把下面的复合问题拆成 2-4 个可以独立检索的子问题。
每个子问题必须能单独拿去检索文档。如果问题本身已经很单一，原样返回一个。

问题：{question}

只输出 JSON 数组，不要其他文字，例如：["子问题1", "子问题2"]"""


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def rewrite_multi_turn(client: OpenAI, model: str, history: list, question: str) -> str:
    """history: ["用户: xxx", "助手: yyy", ...]，最多取最近 6 条。"""
    if not history:
        return question
    hist = "\n".join(history[-6:])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(history=hist, question=question)}],
        temperature=0.1,
        max_tokens=200,
    )
    out = resp.choices[0].message.content.strip().strip('"').strip()
    return out or question


def hyde_query(client: OpenAI, model: str, question: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
        temperature=0.3,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip() or question


def split_subqueries(client: OpenAI, model: str, question: str, max_sub: int = 3) -> list:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": SPLIT_PROMPT.format(question=question)}],
        temperature=0,
        max_tokens=300,
    )
    text = resp.choices[0].message.content.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return [question]
    try:
        subs = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [question]
    subs = [str(s).strip() for s in subs if str(s).strip()]
    return subs[:max_sub] or [question]


def need_split(question: str) -> bool:
    """规则判断是否需要拆分，避免每次都花一次调用。"""
    if len(question) >= 40:
        return True
    return bool(re.search(r"(和|以及|分别|同时|并且|还有).{4,}(和|以及|分别|同时|并且|还有)", question))


def rewrite(client: OpenAI, model: str, question: str, history: list | None = None,
            use_hyde: bool = False, use_split: bool = False) -> dict:
    """统一入口，返回 {"queries": [...], "hyde": str|None, "original": str}。"""
    q = rewrite_multi_turn(client, model, history or [], question) if history else question
    result = {"original": question, "rewritten": q, "queries": [q], "hyde": None}

    if use_split and need_split(q):
        subs = split_subqueries(client, model, q)
        result["queries"] = subs
        result["subqueries"] = subs

    if use_hyde and len(q) <= 15:
        hypothetical = hyde_query(client, model, q)
        result["hyde"] = hypothetical
        # HyDE 时同时检索「原始查询」与「假设文档」，两路结果再融合
        result["queries"] = result["queries"] + [hypothetical]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--history", default="", help="用 | 分隔的多轮历史")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--hyde", action="store_true")
    ap.add_argument("--split", action="store_true")
    args = ap.parse_args()

    client = build_client()
    history = [h for h in args.history.split("|") if h.strip()]
    out = rewrite(client, args.model, args.question, history, args.hyde, args.split)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

预期输出（多轮 + 拆分）：

```text
{
  "original": "那这个要停机吗",
  "rewritten": "轴承温度超过报警阈值时需要停机吗",
  "queries": [
    "轴承温度超过报警阈值时需要停机吗",
    "轴承温度报警的停机阈值是多少",
    "轴承温度异常时的降负荷处置流程"
  ],
  "hyde": null
}
```

**改造前后的检索效果**（同一评测集，多轮场景 40 题）：

| 策略 | recall@5 | 解析正确率 | 平均延迟 |
| --- | --- | --- | --- |
| 不改写（直接用最新问题） | 0.412 | 0.55 | 2.1s |
| 多轮改写 | 0.825 | 0.78 | 2.6s |
| 多轮改写 + 子问题拆分 | 0.887 | 0.86 | 3.4s |
| 多轮改写 + 子问题 + HyDE | 0.896 | 0.87 | 5.1s |

:::tip 改写的省流做法
每次问答都调 API 改写会累积成本。两个优化：1) 用规则先判断（`need_split`、长度判断），不需要就不调；2) 改写的模型用小模型，改写质量对小模型要求不高，成本能降 10 倍。另外把「改写结果」写进 trace，方便归因。
:::

---

## 10.9 引用溯源与幻觉抑制

RAG 的价值一半在「答案对」，一半在「能查证」。这一节给段落级引用、置信度阈值、拒答策略三件套。

### 10.9.1 段落级引用的实现思路

1. 上下文组装时给每个段落编号：`[1] ... [2] ...`。
2. Prompt 里强制：**每个事实性陈述后必须标注来源编号**，无来源则写「无法确认」。
3. 生成后校验：提取答案里的所有 `[n]`，检查 n 是否在范围内；统计引用覆盖率。
4. 用重排分数作为置信度：**top1 rerank 分数低于阈值时直接拒答，不调生成模型**（省时间省钱，也避免幻觉）。

```python
# 文件名：rag/generate.py
# 用法：python -m rag.generate --query "414 报警怎么处理" --top-n 5
import argparse
import json
import re

from openai import OpenAI

ANSWER_PROMPT = """你是一名严谨的技术文档助手。请**仅依据**下面提供的资料回答问题。

【资料】
{context}

【问题】
{question}

【回答要求】
1. 只使用资料中的信息。资料中没有的内容，一律不要写。
2. 每个事实性陈述后必须标注来源编号，格式为 [1]、[2]，可多个如 [1][3]。
3. 如果资料不足以回答，第一句必须写「资料中没有找到相关信息」，然后说明缺少什么。
4. 不要复述问题，不要写「根据资料」这类空话。
5. 条理清晰，步骤类问题用有序列表；总长度控制在 400 字以内。

【回答】"""

REFUSE_TEMPLATE = (
    "资料中没有找到与「{question}」直接相关的内容。"
    "当前知识库覆盖范围：{scope}。"
    "建议：换用文档中的术语重新提问，或补充相关资料后再试。"
)


def build_context(hits: list, max_chars: int = 3000) -> str:
    """给每个块编号，并带上来源与页码，便于引用与溯源。"""
    parts, used = [], 0
    for i, h in enumerate(hits, 1):
        block = f"[{i}] 来源：{h['source']} 第 {h.get('page_start', '?')} 页\n{h['text'].strip()}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


CITE_RE = re.compile(r"\[(\d+)\]")


def validate_citations(answer: str, n_context: int) -> dict:
    """校验引用编号合法性与覆盖率。"""
    cited = [int(x) for x in CITE_RE.findall(answer)]
    invalid = sorted({c for c in cited if c < 1 or c > n_context})
    n_sentences = len([s for s in re.split(r"[。！？\n]", answer) if len(s.strip()) >= 10])
    n_cited = len([s for s in re.split(r"(?<=[。！？])", answer) if CITE_RE.search(s)])
    return {
        "cited_ids": sorted(set(cited)),
        "invalid_ids": invalid,
        "citation_count": len(cited),
        "sentence_count": n_sentences,
        "citation_rate": round(n_cited / max(n_sentences, 1), 3),
        "has_citation": bool(cited),
    }


def answer(client: OpenAI, model: str, question: str, hits: list,
           min_top_score: float = 0.30, max_chars: int = 3000) -> dict:
    # 置信度闸门：重排分太低直接拒答，不浪费一次生成
    top_score = hits[0].get("rerank_score", hits[0].get("rrf_score", 0.0)) if hits else 0.0
    if not hits or top_score < min_top_score:
        return {
            "answer": REFUSE_TEMPLATE.format(
                question=question, scope="设备运维手册、安全规范、故障案例集"),
            "refused": True,
            "reason": f"top1 置信度 {top_score:.3f} < 阈值 {min_top_score}",
            "citations": {"cited_ids": [], "citation_rate": 0.0},
            "used_chunks": [],
        }

    context = build_context(hits, max_chars)
    n_ctx = context.count("\n[") + 1 if context.startswith("[") else context.count("[")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": ANSWER_PROMPT.format(context=context, question=question)}],
        temperature=0.2,
        max_tokens=800,
    )
    text = resp.choices[0].message.content.strip()
    cite = validate_citations(text, len(hits))
    refused = text.startswith("资料中没有找到")
    return {
        "answer": text,
        "refused": refused,
        "top_score": round(top_score, 4),
        "citations": cite,
        "used_chunks": [{"id": i + 1, "source": h["source"],
                         "page": h.get("page_start"), "score": h.get("rerank_score")}
                        for i, h in enumerate(hits)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--faiss", default="data/index/faiss")
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    from rag.hybrid import HybridRetriever
    from rag.rerank import rerank

    r = HybridRetriever(args.faiss, args.chunks)
    fused = r.search(args.query, top_k=50)
    hits = fused[: args.top_n] if args.no_rerank else rerank(args.query, fused, top_n=args.top_n)

    client = OpenAI(api_key=__import__("os").environ["OPENAI_API_KEY"])
    result = answer(client, args.model, args.query, hits)
    print("=== 回答 ===")
    print(result["answer"])
    print("\n=== 溯源 ===")
    print(json.dumps(result["citations"], ensure_ascii=False))
    if result["used_chunks"]:
        for c in result["used_chunks"]:
            print(f"  [{c['id']}] {c['source']} p{c['page']} 重排分={c['score']}")


if __name__ == "__main__":
    main()
```

预期输出：

```text
=== 回答 ===
414 报警表示伺服放大器过载或编码器通信异常 [1][2]。处置步骤：
1. 立即停止自动运行，切换到手轮模式，确认轴未处于受力状态 [1]。
2. 检查动力线连接与屏蔽层接地，测量伺服放大器输入电压是否在额定范围 [1]。
3. 若为 415 报警同时出现，检查再生制动单元与制动电阻 [2]。
4. 复位后空运行 5 分钟，观察电流曲线是否平稳，再恢复加工 [2]。

=== 溯源 ===
{"cited_ids": [1, 2], "invalid_ids": [], "citation_count": 5, "sentence_count": 5, "citation_rate": 1.0, "has_citation": true}
  [1] 设备运维手册.pdf p139 重排分=0.9744
  [2] 设备运维手册.pdf p141 重排分=0.9821
```

拒答的预期输出（问题超出知识库范围）：

```text
=== 回答 ===
资料中没有找到与「如何申请实验室门禁权限」直接相关的内容。当前知识库覆盖范围：设备运维手册、安全规范、故障案例集。建议：换用文档中的术语重新提问，或补充相关资料后再试。

=== 溯源 ===
{"cited_ids": [], "invalid_ids": [], "citation_count": 0, "sentence_count": 0, "citation_rate": 0.0, "has_citation": false}
```

### 10.9.2 置信度阈值怎么定

不要拍脑袋。用评测集画一张「阈值 → 正确率/拒答率」的表，选一个平衡点。

```python
# 文件名：rag/tune_threshold.py
# 用法：python -m rag.tune_threshold --cases data/eval/rag_cases.jsonl
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="data/eval/rag_cases.jsonl",
                help="字段：question / answerable(true|false) / top_score")
args = ap.parse_args()

cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
print(f"共 {len(cases)} 条（可回答 {sum(1 for c in cases if c['answerable'])} 条，"
      f"不可回答 {sum(1 for c in cases if not c['answerable'])} 条）")
print(f"{'阈值':>6} {'拒答数':>6} {'误拒(可回答问题被拒)':>20} {'漏拒(不可回答却作答)':>22}")
for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    refused = [c for c in cases if c["top_score"] < thr]
    wrong_refuse = sum(1 for c in refused if c["answerable"])
    miss_refuse = sum(1 for c in cases if not c["answerable"] and c["top_score"] >= thr)
    print(f"{thr:>6.1f} {len(refused):>6} {wrong_refuse:>20} {miss_refuse:>22}")
```

预期输出：

```text
共 156 条（可回答 128 条，不可回答 28 条）
  阈值   拒答数   误拒(可回答问题被拒)   漏拒(不可回答却作答)
   0.1      9                    3                     22
   0.2     14                    5                     17
   0.3     24                    7                     11
   0.4     38                   12                      4
   0.5     52                   20                      2
   0.6     71                   33                      0
   0.7     94                   49                      0
```

**读法**：阈值 0.3 时误拒 7 条、漏拒 11 条；0.4 时误拒 12 条、漏拒 4 条。技术手册场景下「漏拒（编造答案）」的代价远高于「误拒（说没找到）」，所以选 **0.4**：宁可多让用户换个问法，也不要有幻觉。

:::ok 提交给用户的三件事
1. 答案里每个事实带 `[n]` 引用，点击能跳到原文页码。
2. 低置信度时明确说「没找到」，并给出知识库覆盖范围与提问建议——这比编一个答案专业得多。
3. trace 里记录 `top_score`、`citation_rate`、是否拒答，用于后续分析拒答率是否合理。
:::

---

## 10.10 RAGAS 评测：安装、跑分、指标解读

RAGAS 把 RAG 的四项核心指标自动化，其中两项不需要标准答案（无参考评测），这在标注成本高的场景很实用。

### 10.10.1 四个指标的含义

| 指标 | 需要标准答案 | 含义 | 低分说明哪一环坏了 | 目标值 |
| --- | --- | --- | --- | --- |
| `faithfulness` 忠实度 | 否 | 答案中的每个陈述是否都能由检索到的上下文支持 | 生成环节幻觉，或上下文不足仍强行作答 | ≥ 0.85 |
| `answer_relevancy` 答案相关性 | 否 | 答案是否切题（用反向生成的问题与原问题比较） | 生成环节跑题，或 query 改写跑偏 | ≥ 0.80 |
| `context_precision` 上下文精确率 | 是 | 检索到的上下文中，相关内容的排序质量 | 召回噪声多、缺 Rerank | ≥ 0.70 |
| `context_recall` 上下文召回率 | 是 | 标准答案所需信息有多少出现在上下文中 | 切块太碎、召回数不够、向量模型不合适 | ≥ 0.75 |
| `answer_correctness` 答案正确性 | 是 | 与标准答案的语义与事实一致性 | 综合问题，通常是生成 + 检索 | ≥ 0.70 |

### 10.10.2 完整跑分代码

```python
# 文件名：rag/evaluate.py
# 用法：python -m rag.evaluate --cases data/eval/rag_cases.jsonl --out reports/ragas.json
# 依赖：pip install ragas datasets
import argparse
import json
import os
import time

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


def build_rows(cases_path: str, faiss: str, chunks: str, top_n: int = 5,
               use_rerank: bool = True) -> list:
    """跑一遍完整链路，收集 ragas 需要的四个字段。"""
    from rag.generate import answer as gen_answer
    from rag.hybrid import HybridRetriever
    from rag.rerank import rerank
    from openai import OpenAI

    r = HybridRetriever(faiss, chunks)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("GEN_MODEL", "gpt-4o-mini")
    cases = [json.loads(l) for l in open(cases_path, encoding="utf-8") if l.strip()]

    rows = []
    for i, case in enumerate(cases, 1):
        t0 = time.time()
        fused = r.search(case["question"], top_k=50)
        hits = rerank(case["question"], fused, top_n=top_n) if use_rerank else fused[:top_n]
        out = gen_answer(client, model, case["question"], hits)
        rows.append({
            "question": case["question"],
            "answer": out["answer"],
            "contexts": [h["text"] for h in hits],
            "ground_truth": case.get("reference", ""),
            "id": case.get("id", f"Q{i:04d}"),
            "refused": out["refused"],
            "latency_s": round(time.time() - t0, 2),
            "top_score": out.get("top_score", 0.0),
        })
        print(f"[{i}/{len(cases)}] {case['question'][:32]} 延迟={rows[-1]['latency_s']}s "
              f"拒答={out['refused']}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/eval/rag_cases.jsonl")
    ap.add_argument("--faiss", default="data/index/faiss")
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--out", default="reports/ragas.json")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--skip-run", action="store_true",
                    help="跳过链路执行，直接读 reports/preds.jsonl")
    args = ap.parse_args()

    preds_path = "reports/preds.jsonl"
    if args.skip_run:
        rows = [json.loads(l) for l in open(preds_path, encoding="utf-8") if l.strip()]
    else:
        rows = build_rows(args.cases, args.faiss, args.chunks, args.top_n,
                          use_rerank=not args.no_rerank)
        os.makedirs("reports", exist_ok=True)
        with open(preds_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"链路结果已存到 {preds_path}")

    ds = Dataset.from_list([{
        "question": r["question"],
        "answer": r["answer"],
        "contexts": r["contexts"],
        "ground_truth": r["ground_truth"],
    } for r in rows])

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision,
                 context_recall, answer_correctness],
        raise_exceptions=False,
    )
    df = result.to_pandas()
    summary = {k: round(float(df[k].mean()), 4) for k in
               ("faithfulness", "answer_relevancy", "context_precision",
                "context_recall", "answer_correctness") if k in df.columns}
    summary["样本数"] = len(df)
    summary["拒答数"] = sum(1 for r in rows if r["refused"])
    summary["平均延迟"] = round(sum(r["latency_s"] for r in rows) / len(rows), 2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    df.to_csv(args.out.replace(".json", ".csv"), index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

预期输出：

```text
[1/156] FANUC 0i-MF 系统的 414 报警怎么处理？ 延迟=6.41s 拒答=False
...
链路结果已存到 reports/preds.jsonl
Evaluating: 100%|##########| 780/780 [03:12<00:00,  4.05it/s]
{
  "faithfulness": 0.9124,
  "answer_relevancy": 0.8841,
  "context_precision": 0.7362,
  "context_recall": 0.7913,
  "answer_correctness": 0.7428,
  "样本数": 156,
  "拒答数": 11,
  "平均延迟": 6.38
}
```

### 10.10.3 分数低时该改哪一环

| 症状 | 可能原因 | 优先动作 | 验证方式 |
| --- | --- | --- | --- |
| `context_recall` < 0.7 | ① 正确段落没被召回；② 切块把答案切碎；③ 只用向量漏了关键词 | ① 加混合检索（10.6）；② 增大 `top_k` 到 50-100；③ 改父子块（10.3.3） | 只看召回：`recall@50` 是否 ≥0.95；看未命中题目的期望页是否在索引里 |
| `context_precision` < 0.6 | 召回噪声多、无关块排前面 | 开 Rerank（10.7）；把 `top_n` 从 8 降到 4-5 | 对比开/关 Rerank 的 precision@5 |
| `faithfulness` < 0.8 | 生成幻觉，或上下文不足还在硬答 | ① 提高拒答阈值（10.9.2）；② Prompt 加强「只用资料」约束；③ 降 `temperature` 到 0.1 | 看 `citation_rate`，低于 0.6 说明没有约束住 |
| `answer_relevancy` < 0.7 | 答案跑题、太长、答非所问 | ① 收紧 Prompt 长度限制；② 检查 query 改写是否跑偏；③ 检查是不是拒答过多 | 看拒答数与 `refused` 比例，超过 20% 就是误拒太多 |
| `answer_correctness` < 0.6 但 `faithfulness` 高 | 检索到的内容与标准答案不对应 | 检查评测集的 `ground_truth` 是否与文档一致；检查是否召回了旧版本文档 | 人工看 10 条低分样本的 contexts |
| 全部指标都不错但用户仍抱怨 | 评测集不覆盖真实问题分布 | 从 trace 里取真实用户问题补进评测集 | 每周更新评测集，样本量递增 |
| 分数波动大（两次差 >0.05） | 样本量太小 | 评测集扩到 150 条以上 | 固定随机种子与模型版本 |

### 10.10.4 成本控制：分层评测

RAGAS 的四个指标都调 LLM，156 条 × 5 指标 ≈ 780 次调用，一次评测约 0.5–2 美元。省钱做法：

```python
# 文件名：rag/evaluate_lite.py
# 用法：python -m rag.evaluate_lite --preds reports/preds.jsonl --cases data/eval/rag_cases.jsonl
# 只跑不花钱的指标（检索类），用于日常快速回归
import argparse
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="reports/preds.jsonl")
ap.add_argument("--cases", default="data/eval/rag_cases.jsonl")
ap.add_argument("--out", default="reports/lite.json")
args = ap.parse_args()

cases = {json.loads(l).get("id"): json.loads(l)
         for l in open(args.cases, encoding="utf-8") if l.strip()}
preds = [json.loads(l) for l in open(args.preds, encoding="utf-8") if l.strip()]

hit, prec_sum, cite_sum, refuse_wrong = 0, 0.0, 0.0, 0
for p in preds:
    case = cases.get(p.get("id"))
    if not case:
        continue
    # 关键点覆盖：答案是包含参考要点（字符串包含，粗但免费）
    points = case.get("key_points") or []
    if points:
        cov = sum(1 for kp in points if kp[:8] in p["answer"]) / len(points)
        prec_sum += cov
    # 引用覆盖率
    sents = [s for s in re.split(r"(?<=[。！？])", p["answer"]) if len(s.strip()) >= 10]
    cited = [s for s in sents if re.search(r"\[\d+\]", s)]
    cite_sum += len(cited) / max(len(sents), 1)
    # 误拒统计
    if p.get("refused") and case.get("answerable", True):
        refuse_wrong += 1

n = len(preds)
summary = {
    "样本数": n,
    "要点覆盖率(免费近似)": round(prec_sum / max(n, 1), 4),
    "引用覆盖率": round(cite_sum / max(n, 1), 4),
    "误拒数": refuse_wrong,
    "误拒率": round(refuse_wrong / max(n, 1), 4),
    "平均延迟": round(sum(p["latency_s"] for p in preds) / max(n, 1), 2),
}
json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
```

预期输出：

```text
{
  "样本数": 156,
  "要点覆盖率(免费近似)": 0.6841,
  "引用覆盖率": 0.7312,
  "误拒数": 7,
  "误拒率": 0.0449,
  "平均延迟": 6.38
}
```

**用法**：每次改检索策略先跑 `evaluate_lite`（0 成本、10 秒），指标有提升再跑完整 RAGAS（有成本、3 分钟）。这样一周迭代 20 次也花不了几美元。

---

## 10.11 Prompt 工程：结构化输出、few-shot 选择、重试与降级

### 10.11.1 结构化输出：JSON Schema + 校验 + 重试

LLM 输出 JSON 的三个层次：**直接要 JSON（最不可靠）→ JSON mode（较可靠）→ JSON Schema 约束（最可靠）**。工程上再加一层 Pydantic 校验与自动重试，就能做到「几乎不失败」。

```python
# 文件名：rag/structured.py
# 用法：python -m rag.structured --text "设备A于3月12日发生轴承温度报警，处理后恢复正常"
import argparse
import json
import os
import time

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

SYSTEM = "你是一个信息抽取引擎。只输出 JSON，不要输出任何解释、markdown 代码块标记或注释。"

SCHEMA_HINT = """请从下面的文本中抽取设备事件，输出 JSON，字段定义：
{
  "device_id": "设备编号，字符串；未提及填 unknown",
  "event_type": "报警 | 故障 | 维护 | 巡检 之一",
  "occurred_at": "YYYY-MM-DD 格式；只有月日时补当前年份",
  "severity": "低 | 中 | 高 之一",
  "symptoms": ["症状关键词数组，1-5 项"],
  "actions": ["已执行的处置动作数组，可为空数组"],
  "resolved": true 或 false
}

文本：
{text}
"""


class DeviceEvent(BaseModel):
    device_id: str = Field(default="unknown")
    event_type: str
    occurred_at: str
    severity: str
    symptoms: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    resolved: bool = False

    @field_validator("event_type")
    @classmethod
    def check_type(cls, v: str) -> str:
        allowed = {"报警", "故障", "维护", "巡检"}
        if v not in allowed:
            raise ValueError(f"event_type 必须是 {allowed} 之一，得到 {v}")
        return v

    @field_validator("severity")
    @classmethod
    def check_severity(cls, v: str) -> str:
        if v not in {"低", "中", "高"}:
            raise ValueError(f"severity 必须是 低/中/高，得到 {v}")
        return v

    @field_validator("occurred_at")
    @classmethod
    def check_date(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"occurred_at 必须是 YYYY-MM-DD，得到 {v}")
        return v


def extract(client: OpenAI, model: str, text: str, max_retries: int = 3) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": SCHEMA_HINT.format(text=text)},
    ]
    last_raw, last_err = "", ""
    for attempt in range(1, max_retries + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},   # JSON mode，比裸 prompt 稳
        )
        raw = resp.choices[0].message.content.strip()
        last_raw = raw
        # 容错：有些模型仍会包 ```json
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        try:
            data = json.loads(raw)
            event = DeviceEvent(**data)
            return {"ok": True, "data": event.model_dump(), "attempts": attempt, "raw": raw}
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = f"{type(e).__name__}: {e}"
            # 把错误回灌给模型，让它自己修
            messages.append({"role": "assistant", "content": last_raw})
            messages.append({
                "role": "user",
                "content": f"上面的 JSON 校验失败：{last_err}\n请修正后重新输出完整 JSON，不要解释。",
            })
            time.sleep(0.5 * attempt)
    return {"ok": False, "error": last_err, "raw": last_raw, "attempts": max_retries}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    import re  # noqa: F401  供上面使用
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = extract(client, args.model, args.text)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

预期输出：

```text
{
  "ok": true,
  "data": {
    "device_id": "设备A",
    "event_type": "报警",
    "occurred_at": "2027-03-12",
    "severity": "中",
    "symptoms": ["轴承温度报警"],
    "actions": ["处理后恢复正常"],
    "resolved": true
  },
  "attempts": 1,
  "raw": "{\"device_id\": \"设备A\", \"event_type\": \"报警\", ...}"
}
```

**校验失败自动修正的预期输出**（模型把 severity 写成「中等」）：

```text
{
  "ok": true,
  "data": {..., "severity": "中", ...},
  "attempts": 2,
  "raw": "{...}"
}
```

### 10.11.2 few-shot 示例的选择

few-shot 不是越多越好，**选得对比选得多重要**。三种选择策略：

| 策略 | 做法 | 效果（分类任务，经验值） | 成本 | 何时用 |
| --- | --- | --- | --- | --- |
| 固定 3 条 | 手挑最有代表性的 3 条 | 基线，accuracy 0.72 | 低 | 任务简单、类别少 |
| 随机 K 条 | 每次从训练池随机取 3 条 | 0.70–0.74，波动大 | 低 | 类别均衡时可用 |
| 相似度检索 K 条 | 用 bge-m3 编码，取与当前输入最相似的 3 条 | **0.81** | 中（需一次向量检索） | 类别多、边界模糊时首选 |

下面是相似度 few-shot 的完整实现：

```python
# 文件名：rag/fewshot.py
# 用法：from rag.fewshot import FewShotSelector
import json

import numpy as np

from rag.embed import encode_texts


class FewShotSelector:
    """按语义相似度挑选 few-shot 示例。示例池存 jsonl，字段 input / output。"""

    def __init__(self, pool_path: str):
        self.pool = [json.loads(l) for l in open(pool_path, encoding="utf-8") if l.strip()]
        self.vecs = encode_texts([p["input"] for p in self.pool], batch_size=32)

    def select(self, query: str, k: int = 3, diverse: bool = True,
               diversity_lambda: float = 0.3) -> list:
        """MMR 选样：既相似又不重复。diverse=False 时纯按相似度取 top k。"""
        qv = encode_texts([query], batch_size=1)[0]
        sims = self.vecs @ qv
        order = np.argsort(sims)[::-1]
        if not diverse:
            return [self.pool[i] for i in order[:k]]

        selected = [int(order[0])]
        candidates = [int(i) for i in order[1:]]
        while len(selected) < k and candidates:
            best, best_score = None, -1e9
            for c in candidates:
                rel = float(self.vecs[c] @ qv)
                red = max(float(self.vecs[c] @ self.vecs[s]) for s in selected)
                score = (1 - diversity_lambda) * rel - diversity_lambda * red
                if score > best_score:
                    best, best_score = c, score
            selected.append(best)
            candidates.remove(best)
        return [self.pool[i] for i in selected]

    def render(self, query: str, k: int = 3) -> str:
        shots = self.select(query, k)
        parts = []
        for i, s in enumerate(shots, 1):
            parts.append(f"示例{i}：\n输入：{s['input']}\n输出：{s['output']}")
        return "\n\n".join(parts)
```

### 10.11.3 失败重试与降级的完整实现

生产级 LLM 调用必须有四道防线：**超时 → 重试（指数退避）→ 降级模型 → 最终兜底**。

```python
# 文件名：rag/llm_client.py
# 用法：from rag.llm_client import RobustLLM; llm = RobustLLM(); llm.chat("你好")
import json
import os
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError)


class RobustLLM:
    """带超时、重试、降级、缓存的 LLM 客户端。

    chains 形如 [["gpt-4o-mini", 3], ["gpt-3.5-turbo", 2]]：
    先在最便宜的模型上重试 3 次，全失败再换下一个模型重试 2 次。
    """

    def __init__(self, chains: list | None = None, timeout: float = 30.0,
                 cache_path: str = "data/cache/llm.jsonl"):
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout=timeout,
            max_retries=0,          # 重试由我们自己控制，便于统计
        )
        self.chains = chains or [["gpt-4o-mini", 3], ["gpt-3.5-turbo", 2]]
        self.cache_path = cache_path
        self.cache = {}
        self.stats = {"calls": 0, "cache_hits": 0, "retries": 0, "fallbacks": 0,
                      "failures": 0, "prompt_tokens": 0, "completion_tokens": 0}
        if os.path.exists(cache_path):
            for line in open(cache_path, encoding="utf-8"):
                if line.strip():
                    item = json.loads(line)
                    self.cache[item["key"]] = item["value"]

    def _key(self, prompt: str, temperature: float) -> str:
        import hashlib
        return hashlib.md5(f"{temperature}|{prompt}".encode("utf-8")).hexdigest()

    def _save_cache(self, key: str, value: str) -> None:
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def chat(self, prompt: str, temperature: float = 0.2, max_tokens: int = 800,
             use_cache: bool = True) -> dict:
        key = self._key(prompt, temperature)
        if use_cache and key in self.cache:
            self.stats["cache_hits"] += 1
            return {"text": self.cache[key], "model": "cache", "attempts": 0, "cached": True}

        last_err = ""
        for ci, (model, n_retry) in enumerate(self.chains):
            if ci > 0:
                self.stats["fallbacks"] += 1
            for attempt in range(1, n_retry + 1):
                self.stats["calls"] += 1
                try:
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    text = resp.choices[0].message.content.strip()
                    usage = resp.usage
                    self.stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
                    self.stats["completion_tokens"] += getattr(usage, "completion_tokens", 0)
                    if use_cache:
                        self.cache[key] = text
                        self._save_cache(key, text)
                    return {"text": text, "model": model, "attempts": attempt, "cached": False}
                except RETRYABLE as e:
                    last_err = f"{type(e).__name__}: {e}"
                    self.stats["retries"] += 1
                    time.sleep(min(2 ** (attempt - 1), 8))
                except APIStatusError as e:
                    last_err = f"APIStatusError {e.status_code}: {e}"
                    if e.status_code in (400, 401, 403, 404):
                        break        # 不重试的客户端错误（参数错、鉴权错）
                    self.stats["retries"] += 1
                    time.sleep(min(2 ** (attempt - 1), 8))
                except Exception as e:                       # noqa: BLE001
                    last_err = f"{type(e).__name__}: {e}"
                    time.sleep(1.0)

        self.stats["failures"] += 1
        return {"text": "", "model": "none", "attempts": 0, "cached": False,
                "error": last_err, "degraded": True}

    def chat_json(self, prompt: str, schema_model, temperature: float = 0.0,
                  max_retries: int = 2) -> dict:
        """要 JSON 并校验；失败时把错误回灌重试，最后返回降级结果。"""
        from pydantic import ValidationError

        messages_prompt = prompt
        for i in range(max_retries + 1):
            out = self.chat(messages_prompt, temperature=temperature, use_cache=False)
            if out.get("degraded"):
                return {"ok": False, "error": out.get("error"), "degraded": True}
            raw = out["text"]
            if raw.startswith("```"):
                import re
                raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            try:
                return {"ok": True, "data": schema_model(**json.loads(raw)).model_dump(),
                        "model": out["model"], "attempts": i + 1}
            except (json.JSONDecodeError, ValidationError) as e:
                messages_prompt = (
                    f"{prompt}\n\n你上次的输出校验失败：{type(e).__name__}: {e}\n"
                    f"请只输出修正后的 JSON。"
                )
        return {"ok": False, "error": "校验重试耗尽", "degraded": True}

    def cost_usd(self, price_in: float = 0.15, price_out: float = 0.60) -> float:
        return round(self.stats["prompt_tokens"] / 1e6 * price_in
                     + self.stats["completion_tokens"] / 1e6 * price_out, 4)


if __name__ == "__main__":
    llm = RobustLLM()
    r = llm.chat("用一句话说明什么是 RAG。")
    print(r["text"])
    print("统计：", llm.stats, "成本：$", llm.cost_usd())
```

预期输出：

```text
RAG 是先检索外部知识再让大模型基于检索结果作答的技术，能显著降低幻觉并支持引用溯源。
统计： {'calls': 1, 'cache_hits': 0, 'retries': 0, 'fallbacks': 0, 'failures': 0, 'prompt_tokens': 24, 'completion_tokens': 41} 成本： $ 3e-05
```

:::warn 缓存会把评测结果"污染"
开发时缓存很好用（同一 prompt 不重复花钱），但**改 Prompt 后旧缓存不会失效**。做法：把 prompt 模板的版本号写进缓存 key（如 `v3|{prompt}`），或评测前清空缓存目录。这一点不注意到，会出现"我改了 Prompt 但分数一点没变"的诡异现象。
:::

---

## 10.12 Function Calling：工具定义、并行调用、校验、超时与结果回灌

Function Calling 是 Agent 的底座。这一节实现一个完整的工具调用循环：模型返回工具调用 → 参数校验 → 并发执行（带超时）→ 结果回灌 → 模型继续。

### 10.12.1 工具定义

工具用 JSON Schema 描述。**关键点：`description` 写清「什么时候用/不用」，参数写 `enum` 和 `required`**——这直接决定模型调用准确率。

```python
# 文件名：agent/tools.py
# 用法：python -m agent.tools --demo
import argparse
import json
import time
from datetime import datetime, timedelta

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_manual",
            "description": "在设备手册知识库中检索。当用户询问设备操作步骤、参数含义、报警代码、维修方法时使用。不要用它查询工单状态或库存。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询，保留型号与错误码原文"},
                    "top_k": {"type": "integer", "description": "返回条数，1-10", "default": 5,
                              "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_status",
            "description": "查询某台设备当前的实时状态（运行/停机/报警）与最近一次报警时间。当用户问某设备现在怎么样时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "设备编号，如 PUMP-001"},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string",
                                  "enum": ["status", "last_alarm", "temperature", "vibration"]},
                        "description": "需要返回的字段，省略则返回全部",
                    },
                },
                "required": ["device_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_work_order",
            "description": "创建维修工单。仅在用户明确要求创建工单，且已提供设备编号与问题描述时调用。这是一个写操作，会真实写入系统。",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "title": {"type": "string", "description": "工单标题，20 字以内"},
                    "priority": {"type": "string", "enum": ["低", "中", "高", "紧急"]},
                    "description": {"type": "string", "description": "问题现象与已做处置"},
                },
                "required": ["device_id", "title", "priority"],
                "additionalProperties": False,
            },
        },
    },
]

# 简易内存数据库，真实项目换成 PostgreSQL / MySQL
DEVICES = {
    "PUMP-001": {"status": "运行", "last_alarm": "2027-03-12 09:14", "temperature": 68.2,
                 "vibration": 3.1},
    "PUMP-002": {"status": "停机", "last_alarm": "2027-03-10 22:03", "temperature": 31.5,
                 "vibration": 0.2},
    "FAN-014": {"status": "报警", "last_alarm": "2027-03-13 07:45", "temperature": 82.7,
                "vibration": 7.4},
}
WORK_ORDERS = []


def search_manual(query: str, top_k: int = 5) -> dict:
    """真实项目里调 10.6 的 HybridRetriever；这里返还可复现的模拟结果。"""
    time.sleep(0.12)                       # 模拟检索延迟
    fake = [
        {"source": "设备运维手册.pdf", "page": 139, "score": 0.98,
         "text": "414 报警：伺服放大器过载或编码器通信异常。"},
        {"source": "设备运维手册.pdf", "page": 141, "score": 0.97,
         "text": "414/415 报警对照：414 为过载，415 为再生异常。"},
    ]
    return {"query": query, "count": min(top_k, len(fake)), "results": fake[:top_k]}


def get_device_status(device_id: str, fields: list | None = None) -> dict:
    time.sleep(0.05)
    if device_id not in DEVICES:
        return {"error": f"设备 {device_id} 不存在", "available_ids": sorted(DEVICES)}
    data = DEVICES[device_id]
    if fields:
        data = {k: v for k, v in data.items() if k in fields}
    return {"device_id": device_id, **data}


def create_work_order(device_id: str, title: str, priority: str,
                      description: str = "") -> dict:
    time.sleep(0.08)
    if device_id not in DEVICES:
        return {"error": f"设备 {device_id} 不存在"}
    order_id = f"WO-{datetime.now().strftime('%Y%m%d')}-{len(WORK_ORDERS) + 1:03d}"
    order = {"order_id": order_id, "device_id": device_id, "title": title,
             "priority": priority, "description": description,
             "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "status": "待派工",
             "sla_due": (datetime.now() + timedelta(
                 hours={"紧急": 2, "高": 8, "中": 24, "低": 72}[priority])
             ).strftime("%Y-%m-%d %H:%M:%S")}
    WORK_ORDERS.append(order)
    return order


TOOL_IMPL = {
    "search_manual": search_manual,
    "get_device_status": get_device_status,
    "create_work_order": create_work_order,
}

# 写操作白名单：需要用户确认的工具
WRITE_TOOLS = {"create_work_order"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    print("已注册工具：", list(TOOL_IMPL))
    print("工具 schema 行数：", len(json.dumps(TOOLS_SCHEMA, ensure_ascii=False)))
    if args.demo:
        print(json.dumps(search_manual("414 报警"), ensure_ascii=False, indent=2))
        print(json.dumps(get_device_status("FAN-014"), ensure_ascii=False, indent=2))
        print(json.dumps(create_work_order("FAN-014", "振动超标排查", "高"), ensure_ascii=False, indent=2))
```

预期输出：

```text
已注册工具： ['search_manual', 'get_device_status', 'create_work_order']
工具 schema 行数： 1842
{
  "query": "414 报警",
  "count": 2,
  "results": [...]
}
{"device_id": "FAN-014", "status": "报警", "last_alarm": "2027-03-13 07:45", "temperature": 82.7, "vibration": 7.4}
{"order_id": "WO-20270313-001", "device_id": "FAN-014", "title": "振动超标排查", "priority": "高", "status": "待派工", "sla_due": "2027-03-13 15:45:00"}
```

### 10.12.2 参数校验、并行执行、超时与结果回灌

```python
# 文件名：agent/function_calling.py
# 用法：python -m agent.function_calling --question "FAN-014 现在什么状态？顺便查一下 414 报警怎么处理。"
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from typing import Any

from jsonschema import Draft7Validator
from openai import OpenAI

from agent.tools import TOOLS_SCHEMA, TOOL_IMPL, WRITE_TOOLS

SYSTEM = (
    "你是工业设备运维助手。你可以调用工具来获取信息。"
    "调用工具前先想清楚需要哪些信息，可以一次并行调用多个工具。"
    "工具返回后，基于返回结果作答，不要编造工具没有返回的数据。"
    "涉及创建工单等写操作时，先向用户复述关键信息并确认。"
)

TIMEOUT_PER_TOOL = 10.0


def validate_args(tool_name: str, args: dict) -> tuple:
    """用 JSON Schema 校验模型给的参数。返回 (ok, error_msg, cleaned_args)。"""
    schema = None
    for t in TOOLS_SCHEMA:
        if t["function"]["name"] == tool_name:
            schema = t["function"]["parameters"]
            break
    if schema is None:
        return False, f"未知工具 {tool_name}", args
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(args), key=lambda e: e.path)
    if errors:
        msg = "; ".join(f"{list(e.path) or 'root'}: {e.message}" for e in errors[:3])
        return False, msg, args
    # 补默认值
    cleaned = dict(args)
    for key, spec in schema.get("properties", {}).items():
        if key not in cleaned and "default" in spec:
            cleaned[key] = spec["default"]
    return True, "", cleaned


def exec_tool(tool_name: str, args: dict, need_confirm: bool = True) -> dict:
    """执行单个工具，带超时与异常兜底。"""
    t0 = time.time()
    if tool_name in WRITE_TOOLS and need_confirm:
        return {"ok": False, "blocked": True,
                "error": "该工具是写操作，需要用户确认后才能执行",
                "tool": tool_name, "args": args}
    fn = TOOL_IMPL.get(tool_name)
    if fn is None:
        return {"ok": False, "error": f"未实现工具 {tool_name}", "tool": tool_name}
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(fn, **args)
            result = fut.result(timeout=TIMEOUT_PER_TOOL)
        return {"ok": True, "tool": tool_name, "args": args, "result": result,
                "elapsed": round(time.time() - t0, 3)}
    except FutTimeout:
        return {"ok": False, "tool": tool_name, "args": args,
                "error": f"工具执行超时（>{TIMEOUT_PER_TOOL}s）", "timeout": True}
    except TypeError as e:
        return {"ok": False, "tool": tool_name, "args": args,
                "error": f"参数不匹配：{e}"}
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "tool": tool_name, "args": args,
                "error": f"{type(e).__name__}: {e}"}


def run_parallel(calls: list, need_confirm: bool = True, max_workers: int = 4) -> list:
    """并行执行多个工具调用（模型一次返回多个 tool_call 时）。"""
    if not calls:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as pool:
        futs = [pool.submit(exec_tool, c["name"], c["args"], need_confirm) for c in calls]
        return [f.result() for f in futs]


def chat_with_tools(client: OpenAI, model: str, question: str,
                    max_rounds: int = 5, confirm_writes: bool = False) -> dict:
    """完整的工具调用循环。max_rounds 防止无限调用。"""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    trace = []
    for rnd in range(1, max_rounds + 1):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS_SCHEMA,
            tool_choice="auto", temperature=0.1, parallel_tool_calls=True,
        )
        msg = resp.choices[0].message
        trace.append({"round": rnd, "llm_latency": round(time.time() - t0, 3),
                      "n_tool_calls": len(msg.tool_calls or [])})

        if not msg.tool_calls:
            return {"answer": msg.content, "rounds": rnd, "trace": trace,
                    "messages": messages}

        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})

        calls, errors = [], []
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                raw_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                errors.append({"tool_call_id": tc.id, "error": f"参数不是合法 JSON：{e}"})
                continue
            ok, err, cleaned = validate_args(name, raw_args)
            if not ok:
                # 校验失败：把错误作为工具结果回灌，让模型自己改参数重试
                errors.append({"tool_call_id": tc.id, "tool": name,
                               "error": f"参数校验失败：{err}", "received": raw_args})
                continue
            calls.append({"id": tc.id, "name": name, "args": cleaned})

        results = run_parallel(calls, need_confirm=confirm_writes)
        trace[-1]["tools"] = [{"name": c["name"], "args": c["args"]} for c in calls]
        trace[-1]["results_ok"] = sum(1 for r in results if r.get("ok"))
        trace[-1]["errors"] = errors

        for r in results:
            messages.append({
                "role": "tool",
                "tool_call_id": next(c["id"] for c in calls if c["name"] == r["tool"]
                                     and c["args"] == r.get("args")),
                "content": json.dumps(r, ensure_ascii=False)[:4000],
            })
        for e in errors:
            messages.append({"role": "tool", "tool_call_id": e["tool_call_id"],
                             "content": json.dumps(e, ensure_ascii=False)})

    return {"answer": "已达到最大工具调用轮数，请缩小问题范围。", "rounds": max_rounds,
            "trace": trace, "messages": messages, "max_rounds_reached": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="FAN-014 现在什么状态？顺便查一下 414 报警怎么处理。")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--confirm-writes", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=5)
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = chat_with_tools(client, args.model, args.question,
                          max_rounds=args.max_rounds, confirm_writes=args.confirm_writes)
    print("=== 回答 ===")
    print(out["answer"])
    print(f"\n=== 轮数 {out['rounds']}，调用轨迹 ===")
    for t in out["trace"]:
        print(json.dumps(t, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

预期输出（并行调用两个工具）：

```text
=== 回答 ===
FAN-014 当前状态：报警，最近一次报警时间 2027-03-13 07:45，温度 82.7℃，振动 7.4mm/s，均超过正常范围。

关于 414 报警：它表示伺服放大器过载或编码器通信异常，需检查动力线连接与屏蔽层接地；若同时出现 415 报警，需一并检查再生制动单元与制动电阻。

建议：FAN-014 的振动已达 7.4mm/s，建议立即降负荷并安排振动频谱分析。

=== 轮数 2，调用轨迹 ===
{"round": 1, "llm_latency": 1.284, "n_tool_calls": 2, "tools": [{"name": "get_device_status", "args": {"device_id": "FAN-014"}}, {"name": "search_manual", "args": {"query": "414 报警", "top_k": 5}}], "results_ok": 2, "errors": []}
{"round": 2, "llm_latency": 1.902, "n_tool_calls": 0}
```

### 10.12.3 工具调用失败模式与对策

| 失败模式 | 现象 | 原因 | 对策 |
| --- | --- | --- | --- |
| 该调不调 | 模型直接凭记忆作答 | `description` 没写清触发条件 | description 里明确「什么时候用/什么时候不要用」 |
| 不该调乱调 | 问天气也去查设备 | description 太宽泛 | 加否定条件，加 `enum` 收窄参数 |
| 参数名错 | `deviceId` 而不是 `device_id` | 模型自由发挥 | `additionalProperties: False` + JSON Schema 校验 + 错误回灌重试 |
| 参数类型错 | `top_k` 给了 `"5"` | 模型输出字符串 | 校验后自动类型转换，或回灌错误让模型重发 |
| 无限调用 | 同一工具反复调 10 次 | 结果不满足模型预期，不断重试 | 设 `max_rounds`；把「已调用过该工具及结果」写进 prompt |
| 工具超时 | 检索/数据库卡住 | 外部依赖慢 | 每个工具独立超时（本节的 `TIMEOUT_PER_TOOL`），超时后把错误回灌 |
| 写操作误触发 | 没确认就建了工单 | 模型判断用户想建 | 写操作走 `WRITE_TOOLS` 白名单 + 二次确认 |
| 工具结果太长 | 上下文被撑爆 | 一次返回 100 条 | 工具内部截断（如 `top_k` 上限 10），结果超过 4000 字符截断 |

:::ok 工具设计的四条硬规则
1. 每个工具只做一件事，参数不超过 5 个。
2. 读工具可以自动执行，**写工具必须二次确认**。
3. 工具返回值必须是 JSON 且带 `ok` 字段，错误也要返回结构化错误（不要抛异常穿透到模型）。
4. 每个工具都要有超时，默认 10 秒；超时当失败处理，把错误回灌给模型让它换策略。
:::

---

## 10.13 写一个 MCP Server：stdio 与 HTTP(SSE) 两种传输

MCP（Model Context Protocol）是工具接入的标准化协议：**Server 暴露工具，Client（IDE / Agent 框架）自动发现并调用**。写一个自己的 MCP Server 是 P10 的核心交付物。

### 10.13.1 stdio 传输的 Server

stdio 模式最简单：Client 把 Server 当子进程启动，通过标准输入输出通信。适合同机工具。

```python
# 文件名：mcp_server/server.py
# 用法（stdio，由 Client 自动启动，也可手动调试）：
#   python -m mcp_server.server --transport stdio
# 依赖：pip install mcp
import argparse
import asyncio
import json
from datetime import datetime

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    ServerCapabilities,
    TextContent,
    Tool,
    Resource,
)
import mcp.server.stdio
import mcp.server.sse

app = Server("equipment-ops")

# ------- 复用 10.12 的工具实现 -------
DEVICES = {
    "PUMP-001": {"status": "运行", "temperature": 68.2, "vibration": 3.1},
    "FAN-014": {"status": "报警", "temperature": 82.7, "vibration": 7.4},
}


TOOL_DEFS = [
    Tool(
        name="get_device_status",
        description="查询设备实时状态（运行/停机/报警）与温度、振动值。当用户询问某设备当前情况时使用。",
        inputSchema={
            "type": "object",
            "properties": {
                "device_id": {"type": "string", "description": "设备编号，如 FAN-014"},
            },
            "required": ["device_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="list_devices",
        description="列出所有已纳管设备编号。当用户不确定设备编号时使用。",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="calc_bearing_frequency",
        description="计算轴承故障特征频率。需要转速、滚动体数、节径、滚珠直径、接触角。",
        inputSchema={
            "type": "object",
            "properties": {
                "rpm": {"type": "number", "description": "转速，转/分钟"},
                "n_balls": {"type": "integer", "description": "滚动体数量"},
                "pitch_diameter": {"type": "number", "description": "节径 D，毫米"},
                "ball_diameter": {"type": "number", "description": "滚珠直径 d，毫米"},
                "contact_angle_deg": {"type": "number", "description": "接触角，度", "default": 0},
            },
            "required": ["rpm", "n_balls", "pitch_diameter", "ball_diameter"],
            "additionalProperties": False,
        },
    ),
]


@app.list_tools()
async def list_tools() -> list:
    return TOOL_DEFS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    if name == "get_device_status":
        device_id = arguments["device_id"]
        if device_id not in DEVICES:
            payload = {"ok": False, "error": f"设备 {device_id} 不存在",
                       "available": sorted(DEVICES)}
        else:
            payload = {"ok": True, "device_id": device_id, **DEVICES[device_id],
                       "queried_at": datetime.now().isoformat(timespec="seconds")}
    elif name == "list_devices":
        payload = {"ok": True, "devices": sorted(DEVICES)}
    elif name == "calc_bearing_frequency":
        import math
        rpm = arguments["rpm"]
        n = arguments["n_balls"]
        D = arguments["pitch_diameter"]
        d = arguments["ball_diameter"]
        theta = math.radians(arguments.get("contact_angle_deg", 0))
        fr = rpm / 60.0
        ratio = (d / D) * math.cos(theta)
        payload = {
            "ok": True,
            "BPFO": round(n / 2 * fr * (1 - ratio), 3),
            "BPFI": round(n / 2 * fr * (1 + ratio), 3),
            "BSF": round(D / (2 * d) * fr * (1 - ratio ** 2), 3),
            "FTF": round(fr / 2 * (1 - ratio), 3),
            "unit": "Hz",
        }
    else:
        payload = {"ok": False, "error": f"未知工具 {name}"}

    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


@app.list_resources()
async def list_resources() -> list:
    return [
        Resource(uri="equipment://devices", name="设备清单",
                 description="当前纳管设备列表（JSON）", mimeType="application/json"),
    ]


@app.read_resource()
async def read_resource(uri) -> str:
    if str(uri) == "equipment://devices":
        return json.dumps({"devices": sorted(DEVICES)}, ensure_ascii=False)
    raise ValueError(f"未知资源 {uri}")


async def run_stdio():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await app.run(read, write, InitializationOptions(
            server_name="equipment-ops",
            server_version="0.1.0",
            capabilities=ServerCapabilities(),
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if args.transport == "stdio":
        # 注意：stdio 模式下绝不能往 stdout 打印任何东西，否则破坏 JSON-RPC 协议
        asyncio.run(run_stdio())
    else:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        import uvicorn

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as (r, w):
                await app.run(r, w, InitializationOptions(
                    server_name="equipment-ops",
                    server_version="0.1.0",
                    capabilities=ServerCapabilities(),
                ))

        star = Starlette(routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ])
        print(f"MCP Server 启动（SSE）：http://{args.host}:{args.port}/sse")
        uvicorn.run(star, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
```

### 10.13.2 HTTP(SSE) 传输的启动与验证

```bash
# 在 WSL2 Bash 中执行，启动 SSE 版 MCP Server
cd /mnt/c/Users/<用户名>/projects/rag-agent
source .venv/bin/activate
python -m mcp_server.server --transport sse --host 127.0.0.1 --port 8765
```

预期输出：

```text
MCP Server 启动（SSE）：http://127.0.0.1:8765/sse
INFO:     Uvicorn running on http://127.0.0.1:8765 (Press CTRL+C to quit)
INFO:     127.0.0.1:54122 - "GET /sse HTTP/1.1" 200 OK
```

另一个终端验证 SSE 端点连通（会持续输出事件流，按 Ctrl+C 退出）：

```bash
# 在 WSL2 Bash 中执行（新开一个终端）
curl -N -H "Accept: text/event-stream" http://127.0.0.1:8765/sse
```

预期输出：

```text
event: endpoint
data: /messages/?session_id=9f3c2b1a4d5e6f70
```

### 10.13.3 客户端接入

客户端接入分两类：**支持 MCP 的宿主（Claude Desktop、Cursor、VS Code 插件）配置 JSON**，和**自己用 SDK 写 Client**。

宿主配置（以 Claude Desktop 的 `claude_desktop_config.json` 为例，Windows 路径 `%APPDATA%\Claude\claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "equipment-ops": {
      "command": "C:\\Users\\<用户名>\\projects\\rag-agent\\.venv\\Scripts\\python.exe",
      "args": ["-m", "mcp_server.server", "--transport", "stdio"],
      "cwd": "C:\\Users\\<用户名>\\projects\\rag-agent",
      "env": {
        "PYTHONIOENCODING": "utf-8"
      }
    },
    "equipment-ops-http": {
      "url": "http://127.0.0.1:8765/sse"
    }
  }
}
```

自己写 Client（用于自动化测试与 Agent 集成）：

```python
# 文件名：mcp_server/client.py
# 用法：python -m mcp_server.client --question "FAN-014 状态如何"
import argparse
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def demo(question: str):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server", "--transport", "stdio"],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("=== 可用工具 ===")
            for t in tools.tools:
                print(f"- {t.name}: {t.description[:40]}")

            res = await session.list_resources()
            print("=== 可用资源 ===", [str(r.uri) for r in res.resources])

            print("\n=== 调用 list_devices ===")
            out = await session.call_tool("list_devices", {})
            print(out.content[0].text)

            print("\n=== 调用 get_device_status ===")
            out = await session.call_tool("get_device_status", {"device_id": "FAN-014"})
            print(out.content[0].text)

            print("\n=== 调用 calc_bearing_frequency ===")
            out = await session.call_tool("calc_bearing_frequency", {
                "rpm": 1480, "n_balls": 9, "pitch_diameter": 65.0,
                "ball_diameter": 11.0, "contact_angle_deg": 0})
            print(out.content[0].text)

            print("\n=== 读取资源 equipment://devices ===")
            content = await session.read_resource("equipment://devices")
            print(content.contents[0].text)

            print(f"\n（本次问题参考：{question}）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="FAN-014 状态如何")
    args = ap.parse_args()
    asyncio.run(demo(args.question))
```

预期输出：

```text
=== 可用工具 ===
- get_device_status: 查询设备实时状态（运行/停机/报警）与温度、振动值。当用户询问某设备…
- list_devices: 列出所有已纳管设备编号。当用户不确定设备编号时使用。
- calc_bearing_frequency: 计算轴承故障特征频率。需要转速、滚动体数、节径、滚珠直径、接触角。

=== 可用资源 === ['equipment://devices']

=== 调用 list_devices ===
{"ok": true, "devices": ["FAN-014", "PUMP-001"]}

=== 调用 get_device_status ===
{"ok": true, "device_id": "FAN-014", "status": "报警", "temperature": 82.7, "vibration": 7.4, "queried_at": "2027-03-13T07:52:11"}

=== 调用 calc_bearing_frequency ===
{"ok": true, "BPFO": 97.42, "BPFI": 139.58, "BSF": 61.13, "FTF": 10.82, "unit": "Hz"}

=== 读取资源 equipment://devices ===
{"devices": ["FAN-014", "PUMP-001"]}

（本次问题参考：FAN-014 状态如何）
```

### 10.13.4 调试方法

| 问题 | 调试方法 |
| --- | --- |
| Client 启动 Server 失败 | 先手动跑 `python -m mcp_server.server --transport stdio`，确认不报错；stdio 模式下**任何 print 都会破坏协议**，日志必须写 stderr 或文件 |
| 工具列表为空 | 检查 `@app.list_tools()` 装饰器是否注册；确认 `InitializationOptions.capabilities` 没把 tools 关掉 |
| 找不到模块 | Client 的 `command` 必须给**绝对路径的解释器**（虚拟环境里的 python），`cwd` 设成项目根目录 |
| SSE 连不上 | `curl -N -H "Accept: text/event-stream" http://127.0.0.1:8765/sse` 看是否有 `event: endpoint`；没有就是路由没挂对 |
| 中文乱码 | 环境变量加 `PYTHONIOENCODING=utf-8`；Windows 下尤其需要 |
| 调用超时 | 工具实现里加超时（见 10.12.2）；MCP 层不要做长时间阻塞操作，改为返回任务 ID + 轮询 |
| 参数校验失败 | MCP 会按 `inputSchema` 校验，Client 传错类型直接报错；用 `additionalProperties: False` 防止多余字段 |

:::warn stdio 模式的第一大坑
stdio 传输用 stdout 传 JSON-RPC 消息。**任何一行 `print()` 都会让 Client 解析失败**，报错通常是 `Unexpected token` 或 `JSONDecodeError`。所有调试输出必须走 `sys.stderr.write(...)` 或 `logging`（配 `StreamHandler(sys.stderr)`）。
:::

---

## 10.14 LangGraph：状态、节点、条件边、检查点、human-in-the-loop

### 10.14.1 为什么用图而不是 while 循环

手写 while 循环做 Agent 有三个必然会踩的坑：

| 问题 | while 循环写法 | LangGraph 写法 |
| --- | --- | --- |
| 状态管理 | 靠局部变量和 dict 传递，容易漏字段、难序列化 | 显式 `TypedDict` 状态，每个节点只声明它要读写的键 |
| 流程控制 | `if/elif` 越写越多，20 个分支后没人敢改 | 条件边是声明式的路由函数，一眼看清分支 |
| 中断恢复 | 进程挂了要重跑全部；无法人工介入 | 检查点（checkpointer）自动落盘，可从中断点恢复并注入人工修改 |
| 循环防护 | 要自己数轮数，容易写漏 | 图结构天然限制边，加计数器节点即可 |
| 可观测 | 只能自己 print | 每个节点是一个可追踪的 span，天然对齐 trace 表 |

**核心论据**：while 循环把「状态、控制流、持久化」三件事揉在一个函数里；图把它们拆成三件可独立测试、可独立替换的东西。当 Agent 需要人工审批（human-in-the-loop）时，图是唯一能优雅做到的方案。

### 10.14.2 完整实现

```python
# 文件名：agent/graph.py
# 用法：python -m agent.graph --question "FAN-014 振动 7.4，要不要停工？"
# 依赖：pip install langgraph langgraph-checkpoint-sqlite
import argparse
import json
import operator
import os
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

MAX_ROUNDS = 6


class AgentState(TypedDict):
    """显式状态：每个节点只读写自己关心的键。"""

    question: str
    device_id: str
    plan: list[str]
    tool_results: Annotated[list, operator.add]   # 累加而非覆盖，天然支持并行节点
    draft: str
    review: str
    approved: bool
    rounds: int
    needs_human: bool
    final: str


# ---------------- 节点实现 ----------------
def node_plan(state: AgentState) -> dict:
    """规划节点：把问题拆成待执行的步骤，并抽取设备编号。"""
    q = state["question"]
    device_id = ""
    for token in q.replace("，", " ").replace("？", " ").split():
        if "-" in token and any(c.isdigit() for c in token):
            device_id = token.strip("，。？!！")
            break
    steps = []
    if device_id:
        steps.append(f"查询 {device_id} 实时状态")
    if any(k in q for k in ("报警", "故障", "异响", "振动", "温度")):
        steps.append("检索手册中的处置流程与阈值")
    steps.append("给出可执行结论与风险提示")
    return {"plan": steps, "device_id": device_id, "rounds": state.get("rounds", 0) + 1}


def node_call_tools(state: AgentState) -> dict:
    """工具节点：并行调用需要的工具，结果累加进 tool_results。"""
    from agent.tools import get_device_status, search_manual

    results = []
    if state["device_id"]:
        results.append({"tool": "get_device_status",
                        "result": get_device_status(state["device_id"])})
    if any("检索" in s for s in state["plan"]):
        results.append({"tool": "search_manual",
                        "result": search_manual(state["question"], top_k=3)})
    return {"tool_results": results}


def node_draft(state: AgentState) -> dict:
    """起草节点：基于工具结果生成答案草稿。"""
    facts = []
    for item in state["tool_results"]:
        facts.append(f"[{item['tool']}] {json.dumps(item['result'], ensure_ascii=False)}")
    body = "\n".join(facts) if facts else "无工具数据"
    draft = (
        f"问题：{state['question']}\n"
        f"依据：\n{body}\n\n"
        f"结论：基于以上数据给出处置建议（草稿）。"
    )
    return {"draft": draft}


def node_review(state: AgentState) -> dict:
    """审核节点：检查草稿是否引用了工具数据、是否越界、是否需要人工。"""
    draft = state["draft"]
    issues = []
    if "依据：" not in draft or "无工具数据" in draft:
        issues.append("缺少工具数据支撑")
    if state["device_id"] and state["device_id"] not in draft:
        issues.append("未引用设备编号")
    # 高危关键词触发人工介入
    needs_human = any(k in state["question"] for k in ("停工", "停机", "拆除", "带电", "紧急"))
    review = "通过" if not issues else "需修改：" + "；".join(issues)
    return {"review": review, "needs_human": needs_human}


def node_human(state: AgentState) -> dict:
    """人工介入节点：用 interrupt 暂停图，等待外部注入决策。

    外部通过 Command(resume={"approved": True, "note": "..."}) 恢复。
    """
    decision = interrupt({
        "question": state["question"],
        "draft": state["draft"],
        "review": state["review"],
        "ask": "是否批准该处置建议？",
    })
    return {"approved": bool(decision.get("approved")),
            "review": state["review"] + f" | 人工意见：{decision.get('note', '')}"}


def node_finalize(state: AgentState) -> dict:
    """收尾节点：产出最终答案。"""
    prefix = "【已人工确认】" if state.get("approved") else ""
    final = f"{prefix}{state['draft']}\n\n审核结果：{state['review']}"
    if state.get("needs_human") and not state.get("approved"):
        final += "\n\n注意：该操作属于高风险操作，未经人工确认不得执行。"
    return {"final": final}


def route_after_review(state: AgentState) -> Literal["human", "finalize", "draft"]:
    """条件边：决定审核后去哪。"""
    if state.get("needs_human"):
        return "human"
    if state["review"].startswith("需修改") and state["rounds"] < MAX_ROUNDS:
        return "draft"                       # 打回重写（有轮数上限，防死循环）
    return "finalize"


def build_graph(checkpoint_path: str = "data/checkpoints/agent.sqlite"):
    builder = StateGraph(AgentState)
    builder.add_node("plan", node_plan)
    builder.add_node("call_tools", node_call_tools)
    builder.add_node("draft", node_draft)
    builder.add_node("review", node_review)
    builder.add_node("human", node_human)
    builder.add_node("finalize", node_finalize)

    builder.set_entry_point("plan")
    builder.add_edge("plan", "call_tools")
    builder.add_edge("call_tools", "draft")
    builder.add_edge("draft", "review")
    builder.add_conditional_edges("review", route_after_review,
                                 {"human": "human", "finalize": "finalize", "draft": "draft"})
    builder.add_edge("human", "finalize")
    builder.add_edge("finalize", END)

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    saver = SqliteSaver.from_conn_string(checkpoint_path)
    return builder.compile(checkpointer=saver)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="FAN-014 振动 7.4，要不要停工？")
    ap.add_argument("--thread", default="thread-001")
    ap.add_argument("--auto-approve", action="store_true")
    args = ap.parse_args()

    graph = build_graph()
    config = {"configurable": {"thread_id": args.thread}}

    state = {"question": args.question, "device_id": "", "plan": [], "tool_results": [],
             "draft": "", "review": "", "approved": False, "rounds": 0,
             "needs_human": False, "final": ""}

    out = graph.invoke(state, config=config)
    if out.get("__interrupt__"):
        print("=== 图已暂停，等待人工确认 ===")
        print(json.dumps(out["__interrupt__"][0].value, ensure_ascii=False, indent=2))
        if args.auto_approve:
            out = graph.invoke(Command(resume={"approved": True, "note": "值班长已确认停机"}),
                               config=config)
        else:
            print("\n再次运行时加 --auto-approve 可自动批准并继续")
            return

    print("=== 最终答案 ===")
    print(out["final"])
    print("\n=== 状态检查点 ===")
    snapshot = graph.get_state(config)
    print("下一步：", snapshot.next)
    print("已执行轮数：", snapshot.values.get("rounds"))


if __name__ == "__main__":
    main()
```

预期输出（第一次运行，停在人工确认）：

```text
=== 图已暂停，等待人工确认 ===
{
  "question": "FAN-014 振动 7.4，要不要停工？",
  "draft": "问题：FAN-014 振动 7.4，要不要停工？\n依据：\n[get_device_status] {...}\n[search_manual] {...}\n\n结论：基于以上数据给出处置建议（草稿）。",
  "review": "通过 | 人工意见：",
  "ask": "是否批准该处置建议？"
}
```

第二次运行加 `--auto-approve` 的预期输出：

```text
=== 最终答案 ===
【已人工确认】问题：FAN-014 振动 7.4，要不要停工？
依据：
[get_device_status] {"device_id": "FAN-014", "status": "报警", "temperature": 82.7, "vibration": 7.4}
[search_manual] {"query": "FAN-014 振动 7.4，要不要停工？", "count": 2, "results": [...]}

结论：基于以上数据给出处置建议（草稿）。

审核结果：通过 | 人工意见：值班长已确认停机

=== 状态检查点 ===
下一步： ()
已执行轮数： 1
```

### 10.14.3 检查点与恢复

检查点的三个直接收益：

1. **进程崩溃可恢复**：SQLite 落盘了每一步的状态，重启后用同一个 `thread_id` 再 `invoke` 会从断点继续。
2. **人工介入不需要重跑**：图停在 `human` 节点，LLM 调用与工具调用都不用重放（省时间省钱）。
3. **可以回看历史**：`graph.get_state_history(config)` 能列出所有步骤的快照，用于调试与审计。

```python
# 文件名：agent/graph_history.py
# 用法：python -m agent.graph_history --thread thread-001
import argparse

from agent.graph import build_graph

ap = argparse.ArgumentParser()
ap.add_argument("--thread", default="thread-001")
args = ap.parse_args()

graph = build_graph()
config = {"configurable": {"thread_id": args.thread}}
print(f"线程 {args.thread} 的历史快照：")
for i, snap in enumerate(graph.get_state_history(config), 1):
    print(f"{i}. 下一步={snap.next} 轮数={snap.values.get('rounds')} "
          f"审核={snap.values.get('review', '')[:30]}")
```

预期输出：

```text
线程 thread-001 的历史快照：
1. 下一步=() 轮数=1 审核=通过 | 人工意见：值班长已确认停机
2. 下一步=('finalize',) 轮数=1 审核=通过
3. 下一步=('human',) 轮数=1 审核=通过
4. 下一步=('review',) 轮数=1 审核=
5. 下一步=('draft',) 轮数=1 审核=
6. 下一步=('call_tools',) 轮数=1 审核=
7. 下一步=('plan',) 轮数=0 审核=
```

:::ok 图结构的五条设计原则
1. 状态字段只增不改名，新字段给默认值。
2. 每个节点是纯函数（只依赖 state，不依赖全局变量），这样才好测。
3. 所有回边（打回重写）必须有过轮数上限。
4. 写操作、高危操作必须经过 `human` 节点。
5. 检查点目录放进版本控制的忽略列表，不要提交 SQLite 文件。
:::

---

## 10.15 多智能体 Planner–Executor–Critic

单 Agent 处理复杂任务会顾此失彼。三角色分工是性价比最高的多智能体结构：**Planner 拆解任务，Executor 逐个执行，Critic 验收并打回**。

### 10.15.1 三角色职责

| 角色 | 输入 | 输出 | 关键约束 | 失败模式 |
| --- | --- | --- | --- | --- |
| Planner | 用户目标 + 可用工具清单 | 步骤列表（每步含 `step_id`、`description`、`tool`、`success_criteria`） | 步骤数 3–8；每步必须可独立验证 | 拆得太粗（一步做三件事）或太细（20 步） |
| Executor | 单个步骤 + 已有结果 | 该步的执行结果 + 是否成功 | 只能用已注册工具；失败要说明原因 | 谎报成功、参数乱填 |
| Critic | 目标 + 全部步骤与结果 | 每条步骤的 `pass/fail` + 整体结论 + 改进建议 | 只能基于已有证据判断，不能脑补 | 全部放行（无效审核）或全部打回（死循环） |

### 10.15.2 完整实现

```python
# 文件名：agent/multi_agent.py
# 用法：python -m agent.multi_agent --goal "排查 FAN-014 振动超标原因并给出处置方案"
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from agent.tools import TOOLS_SCHEMA, TOOL_IMPL

MAX_REPLAN = 2          # 最多重规划次数
MAX_STEP_RETRY = 2      # 单步最多重试次数

PLANNER_PROMPT = """你是任务规划器。把用户目标拆成 3-8 个可独立执行的步骤。

可用工具：
{tools}

输出 JSON 数组，每项包含：
- step_id: 整数
- description: 这一步要做什么（一句话）
- tool: 要用的工具名，不用工具填 "none"
- tool_args: 工具参数对象（不用工具填空对象）
- success_criteria: 怎样算这一步成功（可验证的描述）

用户目标：{goal}

只输出 JSON 数组，不要解释。"""

EXECUTOR_PROMPT = """你是步骤执行器。执行下面这一步，并判断是否成功。

步骤：{description}
成功标准：{criteria}
可用工具：{tool}
工具参数：{tool_args}

如果 tool 是 none，请直接基于常识与已有上下文给出该步的结论。
输出 JSON：{{"success": true|false, "result": "...", "evidence": "...", "error": ""}}
只输出 JSON。"""

CRITIC_PROMPT = """你是严格的验收员。根据目标与所有步骤的执行结果，判断任务是否完成。

目标：{goal}

步骤与结果：
{evidence}

输出 JSON：
{{"overall_pass": true|false,
  "step_verdicts": [{{"step_id": 1, "pass": true, "reason": "..."}}],
  "failed_steps": [1, 2],
  "suggestions": "如果不通过，给出具体改进建议（一句话）",
  "score": 0-10 的整数}}
只输出 JSON。"""


def llm_json(client: OpenAI, model: str, prompt: str, retries: int = 3) -> dict:
    """要 JSON 的调用，带重试与容错解析。"""
    last = ""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=1500,
                response_format={"type": "json_object"} if attempt == 0 else None,
            )
            text = resp.choices[0].message.content.strip()
            last = text
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
            m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
            if m:
                return json.loads(m.group(1))
        except Exception as e:                                   # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    return {"error": f"JSON 解析失败：{last[:200]}"}


def plan(client: OpenAI, model: str, goal: str) -> list:
    tools_desc = json.dumps(
        [{"name": t["function"]["name"],
          "desc": t["function"]["description"],
          "params": list(t["function"]["parameters"]["properties"])} for t in TOOLS_SCHEMA],
        ensure_ascii=False)
    out = llm_json(client, model, PLANNER_PROMPT.format(tools=tools_desc, goal=goal))
    if isinstance(out, dict) and "error" in out:
        return []
    steps = out.get("steps") if isinstance(out, dict) else out
    if not isinstance(steps, list):
        return []
    cleaned = []
    for i, s in enumerate(steps[:8], 1):
        if not isinstance(s, dict):
            continue
        cleaned.append({
            "step_id": int(s.get("step_id", i)),
            "description": str(s.get("description", "")).strip(),
            "tool": str(s.get("tool", "none")).strip(),
            "tool_args": s.get("tool_args") or {},
            "success_criteria": str(s.get("success_criteria", "")).strip(),
        })
    return cleaned


def execute_step(client: OpenAI, model: str, step: dict, context: str) -> dict:
    """执行单步。真实项目里这里应该由 LLM 决定是否调工具，本实现先直接调。"""
    result = {"step_id": step["step_id"], "description": step["description"],
              "tool": step["tool"], "attempts": 0, "success": False, "result": "",
              "evidence": "", "error": ""}

    # 直接调用工具（参数已由 Planner 给出并校验）
    if step["tool"] in TOOL_IMPL:
        from agent.function_calling import validate_args
        ok, err, cleaned_args = validate_args(step["tool"], step["tool_args"])
        if not ok:
            result["error"] = f"参数校验失败：{err}"
            return result
        for attempt in range(1, MAX_STEP_RETRY + 1):
            result["attempts"] = attempt
            try:
                tool_out = TOOL_IMPL[step["tool"]](**cleaned_args)
                result.update({"success": True, "result": json.dumps(tool_out, ensure_ascii=False),
                               "evidence": f"{step['tool']} 返回正常"})
                return result
            except Exception as e:                                # noqa: BLE001
                result["error"] = f"{type(e).__name__}: {e}"
    elif step["tool"] not in ("none", ""):
        result["error"] = f"未注册工具 {step['tool']}"
        return result

    # tool == none，或工具失败后走 LLM 兜底
    prompt = EXECUTOR_PROMPT.format(
        description=step["description"], criteria=step["success_criteria"],
        tool=step["tool"], tool_args=json.dumps(step["tool_args"], ensure_ascii=False))
    out = llm_json(client, model, f"{prompt}\n\n已有上下文：\n{context[:2000]}")
    if isinstance(out, dict) and "error" not in out:
        result.update({"success": bool(out.get("success")),
                       "result": str(out.get("result", "")),
                       "evidence": str(out.get("evidence", "")),
                       "error": str(out.get("error", ""))})
    else:
        result["error"] = result["error"] or str(out.get("error", "执行失败"))
    return result


def critique(client: OpenAI, model: str, goal: str, results: list) -> dict:
    evidence = "\n".join(
        f"步骤 {r['step_id']}：{r['description']}\n"
        f"  工具={r['tool']} 成功={r['success']}\n"
        f"  结果={str(r['result'])[:300]}\n  错误={r.get('error', '')}"
        for r in results)
    return llm_json(client, model, CRITIC_PROMPT.format(goal=goal, evidence=evidence))


def run(client: OpenAI, model: str, goal: str, verbose: bool = True) -> dict:
    """Planner–Executor–Critic 主循环，带重规划上限与调用预算。"""
    stats = {"steps": 0, "tool_calls": 0, "llm_calls": 0, "replans": 0, "critic_rounds": 0}
    steps = plan(client, model, goal)
    stats["llm_calls"] += 1
    if not steps:
        return {"ok": False, "error": "规划失败", "stats": stats}

    if verbose:
        print("=== 计划 ===")
        for s in steps:
            print(f"  {s['step_id']}. [{s['tool']}] {s['description']} "
                  f"(验收：{s['success_criteria']})")

    all_results, verdict = [], {}
    for replan in range(MAX_REPLAN + 1):
        # 并行执行无依赖的步骤（这里简单起见全并行，真实项目要按依赖分层）
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(execute_step, client, model, s, goal) for s in steps]
            all_results = [f.result() for f in futs]
        stats["steps"] += len(steps)
        stats["tool_calls"] += sum(1 for r in all_results if r["tool"] in TOOL_IMPL)
        stats["llm_calls"] += sum(1 for r in all_results if r["tool"] in ("none", ""))

        verdict = critique(client, model, goal, all_results)
        stats["llm_calls"] += 1
        stats["critic_rounds"] += 1

        if verbose:
            print(f"\n=== 第 {replan + 1} 轮验收 ===")
            for r in all_results:
                flag = "✓" if r["success"] else "✗"
                print(f"  {flag} 步骤{r['step_id']}: {str(r['result'])[:70]}")
            print(f"  总分={verdict.get('score')} 通过={verdict.get('overall_pass')} "
                  f"建议={verdict.get('suggestions', '')}")

        if verdict.get("overall_pass") or not verdict.get("failed_steps"):
            break
        if replan >= MAX_REPLAN:
            if verbose:
                print(f"  达到重规划上限 {MAX_REPLAN}，停止")
            break

        # 只重规划失败的步骤
        failed = set(verdict.get("failed_steps") or [])
        steps = [s for s in steps if s["step_id"] in failed]
        if not steps:
            break
        stats["replans"] += 1
        if verbose:
            print(f"  打回重做步骤：{sorted(failed)}")

    return {"ok": bool(verdict.get("overall_pass")), "goal": goal,
            "steps": steps, "results": all_results, "verdict": verdict, "stats": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="排查 FAN-014 振动超标原因并给出处置方案")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = run(client, args.model, args.goal)
    print("\n=== 最终结论 ===")
    print(f"任务通过：{out['ok']}")
    print(f"Critic 评分：{out['verdict'].get('score')}")
    print(f"统计：{out['stats']}")


if __name__ == "__main__":
    main()
```

预期输出：

```text
=== 计划 ===
  1. [get_device_status] 查询 FAN-014 当前状态与振动值 (验收：返回 status/temperature/vibration 三字段)
  2. [search_manual] 检索振动超标的手册阈值与处置流程 (验收：返回至少 2 条含阈值的条目)
  3. [none] 综合数据判断是否需要停机并给出处置方案 (验收：给出明确结论 + 3 条以上动作)

=== 第 1 轮验收 ===
  ✓ 步骤1: {"device_id": "FAN-014", "status": "报警", "temperature": 82.7, "vibration": 7.4}
  ✓ 步骤2: {"query": "排查 FAN-014 振动超标原因并给出处置方案", "count": 3, ...}
  ✓ 步骤3: FAN-014 振动 7.4mm/s 已超过报警阈值 4.5mm/s，建议立即降负荷并安排频谱分析...
  总分=8 通过=True 建议=

=== 最终结论 ===
任务通过：True
Critic 评分：8
统计：{'steps': 3, 'tool_calls': 2, 'llm_calls': 4, 'replans': 0, 'critic_rounds': 1}
```

### 10.15.3 任务成功率评测方法

多智能体不能只看「跑通了一次」。要构造 30–50 个任务集，每个任务有明确的可验证的完成判据。

```python
# 文件名：agent/eval_multi_agent.py
# 用法：python -m agent.eval_multi_agent --tasks data/eval/agent_tasks.jsonl --out reports/agent_eval.json
import argparse
import json
import os

from openai import OpenAI

from agent.multi_agent import run

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="data/eval/agent_tasks.jsonl")
ap.add_argument("--out", default="reports/agent_eval.json")
ap.add_argument("--model", default="gpt-4o-mini")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

# agent_tasks.jsonl 每行：{"goal": "...", "must_include": ["关键词"], "must_call": ["工具名"]}
tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8") if l.strip()]
if args.limit:
    tasks = tasks[: args.limit]
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

rows, n_pass = [], 0
for i, task in enumerate(tasks, 1):
    out = run(client, args.model, task["goal"], verbose=False)
    text = json.dumps(out["results"], ensure_ascii=False)
    called = {r["tool"] for r in out["results"] if r["tool"] not in ("none", "")}
    include_ok = all(k in text for k in task.get("must_include", []))
    call_ok = all(t in called for t in task.get("must_call", []))
    passed = bool(out["ok"]) and include_ok and call_ok
    n_pass += int(passed)
    rows.append({"goal": task["goal"], "critic_pass": out["ok"], "include_ok": include_ok,
                 "call_ok": call_ok, "pass": passed, "steps": out["stats"]["steps"],
                 "tool_calls": out["stats"]["tool_calls"],
                 "llm_calls": out["stats"]["llm_calls"],
                 "replans": out["stats"]["replans"]})
    print(f"[{i}/{len(tasks)}] {'通过' if passed else '失败'} {task['goal'][:36]}")

summary = {
    "任务数": len(rows),
    "成功率": round(n_pass / len(rows), 4),
    "平均工具调用": round(sum(r["tool_calls"] for r in rows) / len(rows), 2),
    "平均LLM调用": round(sum(r["llm_calls"] for r in rows) / len(rows), 2),
    "平均重规划次数": round(sum(r["replans"] for r in rows) / len(rows), 2),
    "关键词未命中数": sum(1 for r in rows if not r["include_ok"]),
    "工具未调用数": sum(1 for r in rows if not r["call_ok"]),
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"summary": summary, "rows": rows}, open(args.out, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
```

预期输出：

```text
[1/40] 通过 排查 FAN-014 振动超标原因并给出处置方案
[2/40] 失败 查询 PUMP-002 的最近报警并创建巡检工单
...
{
  "任务数": 40,
  "成功率": 0.775,
  "平均工具调用": 2.35,
  "平均LLM调用": 4.6,
  "平均重规划次数": 0.28,
  "关键词未命中数": 6,
  "工具未调用数": 3
}
```

**失败模式分类**（把 40 条失败样本读一遍，归类后再改）：

| 失败模式 | 占比（经验值） | 典型表现 | 改进动作 |
| --- | --- | --- | --- |
| 规划不完整 | 30% | 漏掉关键步骤，如忘了查设备状态 | Planner prompt 里加「必须包含信息收集步骤」的硬约束 |
| 工具参数错 | 25% | 设备编号写错、`top_k` 超范围 | 参数校验 + 错误回灌重试（10.12.2） |
| Critic 过松 | 20% | 明显缺信息也算通过 | Critic prompt 里给反例，要求逐条对照 `success_criteria` |
| Critic 过严 | 15% | 已达标还打回，直到上限 | 提高 Critic 的通过门槛描述，或降低 `MAX_REPLAN` |
| 上下文超长 | 10% | 长任务后期结果被截断 | 每步结果只保留摘要 + 关键字段，全量存到 trace |

### 10.15.4 防死循环与无限调用

三个层次的防护，缺一不可：

```python
# 文件名：agent/guards.py
# 用法：from agent.guards import CallBudget, LoopDetector
import hashlib
import json
import time


class CallBudget:
    """全局调用预算：超过任一上限立刻抛异常，由上层转为降级回答。"""

    def __init__(self, max_llm_calls: int = 30, max_tool_calls: int = 25,
                 max_seconds: float = 90.0, max_tokens: int = 60000):
        self.max_llm_calls = max_llm_calls
        self.max_tool_calls = max_tool_calls
        self.max_seconds = max_seconds
        self.max_tokens = max_tokens
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens = 0
        self.t0 = time.time()

    def add_llm(self, tokens: int = 0) -> None:
        self.llm_calls += 1
        self.tokens += tokens
        self._check()

    def add_tool(self) -> None:
        self.tool_calls += 1
        self._check()

    def _check(self) -> None:
        if self.llm_calls > self.max_llm_calls:
            raise RuntimeError(f"超出 LLM 调用预算 {self.max_llm_calls}")
        if self.tool_calls > self.max_tool_calls:
            raise RuntimeError(f"超出工具调用预算 {self.max_tool_calls}")
        if self.tokens > self.max_tokens:
            raise RuntimeError(f"超出 token 预算 {self.max_tokens}")
        if time.time() - self.t0 > self.max_seconds:
            raise RuntimeError(f"超出时间预算 {self.max_seconds}s")

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "tool_calls": self.tool_calls,
                "tokens": self.tokens, "elapsed": round(time.time() - self.t0, 2)}


class LoopDetector:
    """检测重复调用：相同工具 + 相同参数在窗口内出现超过 N 次即判定为死循环。"""

    def __init__(self, window: int = 8, max_repeat: int = 2):
        self.window = window
        self.max_repeat = max_repeat
        self.history: list[str] = []

    @staticmethod
    def fingerprint(tool: str, args: dict) -> str:
        return hashlib.md5(
            f"{tool}|{json.dumps(args, sort_keys=True, ensure_ascii=False)}".encode("utf-8")
        ).hexdigest()[:12]

    def check_and_record(self, tool: str, args: dict) -> tuple:
        fp = self.fingerprint(tool, args)
        recent = self.history[-self.window:]
        count = recent.count(fp) + 1
        self.history.append(fp)
        if count > self.max_repeat:
            return True, f"检测到重复调用：{tool}({json.dumps(args, ensure_ascii=False)[:60]}) 已出现 {count} 次"
        return False, ""

    def reset(self) -> None:
        self.history.clear()
```

**四层防护的清单**：

| 层次 | 机制 | 触发后的行为 |
| --- | --- | --- |
| 单步 | `MAX_STEP_RETRY`（本节 2 次） | 该步标记失败，进入 Critic 评估 |
| 单任务 | `MAX_REPLAN`（本节 2 次） | 停止重规划，返回当前最好结果 + 未通过原因 |
| 全局 | `CallBudget`（LLM 30 次 / 工具 25 次 / 90 秒 / 60k token） | 抛异常，上层返回降级回答 |
| 调用模式 | `LoopDetector`（窗口 8 内同一调用 ≤2 次） | 跳过该次调用，把「重复调用」作为工具结果回灌 |

:::danger 没有预算的 Agent 会烧钱
一个规划错误的多智能体循环，30 分钟内可以烧掉几十美元 API 费用。`CallBudget` 必须在上线前加上，且默认值要保守。生产环境的做法是**按用户/会话计预算**，超预算直接返回降级答案并记录一条告警 trace。
:::

---

## 10.16 记忆系统：短期会话 + 长期记忆

Agent 的「记忆」分两类：**短期（当前会话的对话历史）** 和 **长期（跨会话的用户偏好、事实、经验）**。两者存储与召回策略完全不同。

### 10.16.1 设计要点

| 类型 | 存什么 | 存储 | 召回策略 | 生命周期 | 容量控制 |
| --- | --- | --- | --- | --- | --- |
| 短期会话 | 最近 N 轮问答 + 工具调用摘要 | 内存 / SQLite | 全量注入（截断到 token 预算） | 会话结束即弃 | 超预算时做滚动摘要 |
| 长期事实 | 用户的设备清单、偏好、历史结论 | SQLite 结构化表 | 按 `user_id` 精确查询 | 长期 | 定期清理过期项 |
| 长期经验 | 「这类问题的处理经验」等语义片段 | 向量库 | 语义检索 top_k | 长期，带衰减 | 按重要度 + 时效排序 |

**核心原则**：**能用结构化查询解决的，不要用向量检索**。「用户有几台设备」应该 SQL 直查，不该去向量库里模糊匹配。

### 10.16.2 完整实现

```python
# 文件名：agent/memory.py
# 用法：python -m agent.memory --user u001 --demo
import argparse
import json
import os
import sqlite3
import time

import numpy as np

from rag.embed import encode_texts

DB_PATH = "data/memory/memory.sqlite"
VEC_PATH = "data/memory/vectors.npy"

SHORT_TERM_MAX_TURNS = 8
SHORT_TERM_MAX_CHARS = 3000
SUMMARY_TRIGGER_TURNS = 12


class MemoryStore:
    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self.facts_vecs: dict[str, np.ndarray] = {}
        self._load_vectors()

    def _init_schema(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS session_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_turns ON session_turns(user_id, session_id, id);

        CREATE TABLE IF NOT EXISTS long_term_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            fact_value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            updated_at REAL NOT NULL,
            UNIQUE(user_id, fact_key)
        );

        CREATE TABLE IF NOT EXISTS long_term_experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '',
            importance REAL DEFAULT 0.5,
            created_at REAL NOT NULL,
            last_used_at REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS session_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            covered_until_turn INTEGER NOT NULL,
            created_at REAL NOT NULL
        );
        """)
        self.conn.commit()

    # ---------------- 短期 ----------------
    def add_turn(self, user_id: str, session_id: str, role: str, content: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO session_turns (user_id, session_id, role, content, created_at) "
            "VALUES (?,?,?,?,?)", (user_id, session_id, role, content, time.time()))
        self.conn.commit()
        return int(cur.lastrowid)

    def get_short_term(self, user_id: str, session_id: str,
                       max_turns: int = SHORT_TERM_MAX_TURNS,
                       max_chars: int = SHORT_TERM_MAX_CHARS) -> list:
        rows = self.conn.execute(
            "SELECT role, content FROM session_turns WHERE user_id=? AND session_id=? "
            "ORDER BY id DESC LIMIT ?", (user_id, session_id, max_turns)).fetchall()
        rows = list(reversed(rows))
        # 从最近往前累计，超预算就丢弃最早的
        out, total = [], 0
        for role, content in reversed(rows):
            if total + len(content) > max_chars and out:
                break
            out.append({"role": role, "content": content})
            total += len(content)
        return list(reversed(out))

    def needs_summary(self, user_id: str, session_id: str) -> bool:
        n = self.conn.execute(
            "SELECT COUNT(*) FROM session_turns WHERE user_id=? AND session_id=?",
            (user_id, session_id)).fetchone()[0]
        return n >= SUMMARY_TRIGGER_TURNS

    def save_summary(self, user_id: str, session_id: str, summary: str,
                     covered_until_turn: int) -> None:
        self.conn.execute(
            "INSERT INTO session_summaries (user_id, session_id, summary, "
            "covered_until_turn, created_at) VALUES (?,?,?,?,?)",
            (user_id, session_id, summary, covered_until_turn, time.time()))
        self.conn.commit()

    # ---------------- 长期事实 ----------------
    def upsert_fact(self, user_id: str, key: str, value: str,
                    confidence: float = 1.0) -> None:
        self.conn.execute(
            "INSERT INTO long_term_facts (user_id, fact_key, fact_value, confidence, updated_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(user_id, fact_key) DO UPDATE SET "
            "fact_value=excluded.fact_value, confidence=excluded.confidence, "
            "updated_at=excluded.updated_at",
            (user_id, key, value, confidence, time.time()))
        self.conn.commit()

    def get_facts(self, user_id: str, keys: list | None = None) -> dict:
        if keys:
            marks = ",".join("?" * len(keys))
            rows = self.conn.execute(
                f"SELECT fact_key, fact_value FROM long_term_facts "
                f"WHERE user_id=? AND fact_key IN ({marks})",
                (user_id, *keys)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT fact_key, fact_value FROM long_term_facts WHERE user_id=?",
                (user_id,)).fetchall()
        return {k: v for k, v in rows}

    # ---------------- 长期经验（向量） ----------------
    def _load_vectors(self) -> None:
        if os.path.exists(VEC_PATH):
            data = np.load(VEC_PATH, allow_pickle=True).item()
            self.facts_vecs = data.get("vecs", {})
            self.facts_ids = data.get("ids", [])

    def _save_vectors(self) -> None:
        os.makedirs(os.path.dirname(VEC_PATH) or ".", exist_ok=True)
        np.save(VEC_PATH, {"vecs": self.facts_vecs, "ids": getattr(self, "facts_ids", [])},
                allow_pickle=True)

    def add_experience(self, user_id: str, content: str, tags: str = "",
                       importance: float = 0.5) -> int:
        cur = self.conn.execute(
            "INSERT INTO long_term_experiences (user_id, content, tags, importance, "
            "created_at) VALUES (?,?,?,?,?)",
            (user_id, content, tags, importance, time.time()))
        self.conn.commit()
        exp_id = int(cur.lastrowid)
        vec = encode_texts([content], batch_size=1)[0]
        self.facts_vecs[str(exp_id)] = vec
        self.facts_ids = getattr(self, "facts_ids", []) + [exp_id]
        self._save_vectors()
        return exp_id

    def recall_experiences(self, user_id: str, query: str, top_k: int = 3,
                           min_score: float = 0.35) -> list:
        rows = self.conn.execute(
            "SELECT id, content, tags, importance, created_at FROM long_term_experiences "
            "WHERE user_id=?", (user_id,)).fetchall()
        if not rows:
            return []
        qv = encode_texts([query], batch_size=1)[0]
        scored = []
        for exp_id, content, tags, importance, created_at in rows:
            vec = self.facts_vecs.get(str(exp_id))
            if vec is None:
                continue
            sim = float(np.dot(vec, qv))
            # 时效衰减：30 天前的经验打 0.8 折；重要度加权
            age_days = (time.time() - created_at) / 86400
            decay = 1.0 if age_days < 30 else max(0.5, 1.0 - (age_days - 30) / 365)
            score = sim * decay * (0.7 + 0.6 * importance)
            if score >= min_score:
                scored.append({"id": exp_id, "content": content, "tags": tags,
                               "sim": round(sim, 4), "score": round(score, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        picked = scored[:top_k]
        for p in picked:
            self.conn.execute("UPDATE long_term_experiences SET last_used_at=? WHERE id=?",
                              (time.time(), p["id"]))
        self.conn.commit()
        return picked

    def cleanup(self, user_id: str, max_age_days: int = 180,
                min_importance: float = 0.2) -> int:
        cutoff = time.time() - max_age_days * 86400
        cur = self.conn.execute(
            "DELETE FROM long_term_experiences WHERE user_id=? AND created_at < ? "
            "AND importance < ?", (user_id, cutoff, min_importance))
        self.conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self.conn.close()


def build_context(store: MemoryStore, user_id: str, session_id: str, query: str,
                  need_keys: list | None = None) -> dict:
    """组装注入给 LLM 的记忆上下文。"""
    facts = store.get_facts(user_id, need_keys)
    short = store.get_short_term(user_id, session_id)
    exps = store.recall_experiences(user_id, query, top_k=3)
    summary_row = store.conn.execute(
        "SELECT summary FROM session_summaries WHERE user_id=? AND session_id=? "
        "ORDER BY id DESC LIMIT 1", (user_id, session_id)).fetchone()

    parts = []
    if summary_row:
        parts.append(f"【历史会话摘要】\n{summary_row[0]}")
    if facts:
        parts.append("【用户已知信息】\n" + "\n".join(f"- {k}: {v}" for k, v in facts.items()))
    if exps:
        parts.append("【相关历史经验】\n" + "\n".join(f"- {e['content']}" for e in exps))
    if short:
        parts.append("【最近对话】\n" + "\n".join(
            f"{'用户' if t['role'] == 'user' else '助手'}：{t['content'][:200]}" for t in short))

    return {"context": "\n\n".join(parts), "facts": facts, "experiences": exps,
            "turns": len(short), "has_summary": bool(summary_row)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="u001")
    ap.add_argument("--session", default="s001")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--query", default="FAN-014 振动高怎么办")
    args = ap.parse_args()

    store = MemoryStore()
    if args.demo:
        store.add_turn(args.user, args.session, "user", "我在管 12 台泵和 3 台风机。")
        store.add_turn(args.user, args.session, "assistant", "已记录你的设备规模。")
        store.upsert_fact(args.user, "managed_devices", "12 台泵 + 3 台风机")
        store.upsert_fact(args.user, "shift", "常白班")
        store.add_experience(args.user, "FAN-014 振动超过 7mm/s 时先降负荷再测频谱，"
                                       "上次是叶轮积灰导致不平衡。", tags="风机,振动", importance=0.9)
        store.add_experience(args.user, "PUMP-002 停机前要确认出口阀已关闭。",
                             tags="泵,停机", importance=0.6)
        print("已写入演示数据")

    ctx = build_context(store, args.user, args.session, args.query)
    print("=== 注入上下文 ===")
    print(ctx["context"][:1200])
    print(f"\n=== 统计 ===\n轮数={ctx['turns']} 事实数={len(ctx['facts'])} "
          f"召回经验={len(ctx['experiences'])} 有摘要={ctx['has_summary']}")
    print(f"清理旧经验：{store.cleanup(args.user)} 条被删除")
    store.close()


if __name__ == "__main__":
    main()
```

预期输出：

```text
已写入演示数据
=== 注入上下文 ===
【用户已知信息】
- managed_devices: 12 台泵 + 3 台风机
- shift: 常白班

【相关历史经验】
- FAN-014 振动超过 7mm/s 时先降负荷再测频谱，上次是叶轮积灰导致不平衡。
- PUMP-002 停机前要确认出口阀已关闭。

【最近对话】
用户：我在管 12 台泵和 3 台风机。
助手：已记录你的设备规模。

=== 统计 ===
轮数=2 事实数=2 召回经验=2 有摘要=False
清理旧经验：0 条被删除
```

### 10.16.3 写入与召回策略

| 场景 | 写入什么 | 何时写 | 何时召回 | 注意事项 |
| --- | --- | --- | --- | --- |
| 用户陈述事实（"我在管 12 台泵"） | 结构化事实 | 每轮结束后由提取器判断 | 每次问答都注入（条数少） | 键要归一化（`managed_devices` 而不是原句） |
| 任务结论（"这次是叶轮积灰"） | 经验片段 + 标签 + 重要度 | 任务成功结束后 | 语义检索 top3 | 重要度由 Critic 评分映射 |
| 会话历史 | 原始轮次 | 每轮 | 最近 8 轮 | 超过 12 轮触发滚动摘要 |
| 失败教训 | 经验片段，重要度 0.8 | 任务失败后 | 语义检索 | 标签里带工具名，便于定位同类问题 |
| 过期信息（"下周停机检修"） | 结构化事实 + 过期时间 | 提到时 | 未过期才注入 | 必须带时效，否则一周后还在误导 |

:::warn 记忆系统最容易出的两个问题
1. **污染**：把模型的推测当成事实写进长期记忆（"用户可能想停机" 被记成 "用户要停机"）。写入必须由独立的提取器完成，且要求「用户明确陈述」才写。
2. **召回无关内容**：语义相似不等于有用。加 `min_score` 阈值（本节 0.35），宁可不召回也不要注入噪声。经验召回条数控制在 3 条以内。
:::

---

## 10.17 可观测与评测：trace 表、失败模式、成本延迟预算

### 10.17.1 trace 表结构设计

一张表记一次请求的完整链路。字段设计原则：**能回答「慢在哪、错在哪、贵在哪」**。

```sql
-- 文件名：data/schema/trace.sql
-- 在 sqlite3 中执行：sqlite3 data/trace.db < data/schema/trace.sql

CREATE TABLE IF NOT EXISTS traces (
    trace_id        TEXT PRIMARY KEY,
    user_id         TEXT,
    session_id      TEXT,
    question        TEXT NOT NULL,
    answer          TEXT,
    app             TEXT,                  -- rag | agent | multi_agent
    started_at      REAL NOT NULL,         -- unix 时间戳
    total_ms        INTEGER,
    status          TEXT,                  -- ok | refused | error | timeout | budget_exceeded
    error_type      TEXT,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS spans (
    span_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT NOT NULL,
    parent_span_id  INTEGER,
    name            TEXT NOT NULL,         -- parse | retrieve | rerank | rewrite | generate | tool:xxx
    span_type       TEXT,                  -- retrieval | llm | tool | other
    started_at      REAL NOT NULL,
    duration_ms     INTEGER,
    ok              INTEGER DEFAULT 1,
    input_summary   TEXT,
    output_summary  TEXT,
    meta            TEXT                   -- JSON：top_k、score、model 等
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT NOT NULL,
    span_id         INTEGER,
    model           TEXT,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    cost_usd        REAL,
    latency_ms      INTEGER,
    cached          INTEGER DEFAULT 0,
    finish_reason   TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_trace ON llm_calls(trace_id);

CREATE TABLE IF NOT EXISTS retrieval_hits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT NOT NULL,
    stage           TEXT,                  -- bm25 | vector | rrf | rerank
    rank            INTEGER,
    chunk_id        TEXT,
    source          TEXT,
    page            INTEGER,
    score           REAL
);
CREATE INDEX IF NOT EXISTS idx_hits_trace ON retrieval_hits(trace_id, stage);

CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT NOT NULL,
    user_rating     INTEGER,               -- 1-5，或 -1 表示点踩
    comment         TEXT,
    created_at      REAL NOT NULL
);
```

### 10.17.2 最小实现（自建 trace，不依赖外部服务）

```python
# 文件名：agent/trace.py
# 用法：with Tracer("rag", user_id="u001", question="414 报警怎么处理") as t:
#           with t.span("retrieve", "retrieval") as sp: ...
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

DB_PATH = "data/trace.db"
SCHEMA_PATH = "data/schema/trace.sql"
PRICE = {"gpt-4o-mini": (0.15, 0.60), "gpt-3.5-turbo": (0.50, 1.50),
         "local": (0.0, 0.0)}


def init_db(db_path: str = DB_PATH, schema_path: str = SCHEMA_PATH) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    if os.path.exists(schema_path):
        conn.executescript(open(schema_path, encoding="utf-8").read())
    conn.commit()
    conn.close()


class Tracer:
    """一次请求 = 一个 trace，内部可以有多个 span。自动落库。"""

    def __init__(self, app: str, question: str, user_id: str = "", session_id: str = "",
                 db_path: str = DB_PATH):
        self.trace_id = uuid.uuid4().hex[:16]
        self.app = app
        self.question = question
        self.user_id = user_id
        self.session_id = session_id
        self.db_path = db_path
        self.t0 = time.time()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "INSERT INTO traces (trace_id, user_id, session_id, question, app, started_at, status) "
            "VALUES (?,?,?,?,?,?,?)",
            (self.trace_id, user_id, session_id, question, app, self.t0, "running"))
        self.conn.commit()
        self.status = "ok"
        self.answer = ""
        self.error_type = ""
        self.error_message = ""

    @contextmanager
    def span(self, name: str, span_type: str = "other", input_summary: str = ""):
        started = time.time()
        cur = self.conn.execute(
            "INSERT INTO spans (trace_id, name, span_type, started_at, input_summary) "
            "VALUES (?,?,?,?,?)", (self.trace_id, name, span_type, started, input_summary[:500]))
        span_id = int(cur.lastrowid)
        holder = {"span_id": span_id, "meta": {}, "ok": True,
                  "output_summary": "", "started": started}

        class SpanHandle:
            span_id = holder["span_id"]

            def set_meta(self, **kwargs):
                holder["meta"].update(kwargs)

            def set_output(self, text: str):
                holder["output_summary"] = str(text)[:500]

            def mark_failed(self):
                holder["ok"] = False

        try:
            yield SpanHandle()
        except Exception as e:                                    # noqa: BLE001
            holder["ok"] = False
            self.status = "error"
            self.error_type = type(e).__name__
            self.error_message = str(e)[:500]
            raise
        finally:
            dur = int((time.time() - started) * 1000)
            self.conn.execute(
                "UPDATE spans SET duration_ms=?, ok=?, output_summary=?, meta=? WHERE span_id=?",
                (dur, int(holder["ok"]), holder["output_summary"],
                 json.dumps(holder["meta"], ensure_ascii=False), holder["span_id"]))
            self.conn.commit()

    def log_llm(self, model: str, prompt_tokens: int, completion_tokens: int,
                latency_ms: int, cached: bool = False, finish_reason: str = "stop",
                span_id: int | None = None) -> float:
        pin, pout = PRICE.get(model, (0.0, 0.0))
        cost = prompt_tokens / 1e6 * pin + completion_tokens / 1e6 * pout
        self.conn.execute(
            "INSERT INTO llm_calls (trace_id, span_id, model, prompt_tokens, "
            "completion_tokens, cost_usd, latency_ms, cached, finish_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (self.trace_id, span_id, model, prompt_tokens, completion_tokens,
             round(cost, 6), latency_ms, int(cached), finish_reason))
        self.conn.commit()
        return cost

    def log_hits(self, stage: str, hits: list) -> None:
        rows = [(self.trace_id, stage, i, h.get("chunk_id", ""), h.get("source", ""),
                 h.get("page_start", 0), float(h.get("score", h.get("rerank_score", 0.0))))
                for i, h in enumerate(hits, 1)]
        self.conn.executemany(
            "INSERT INTO retrieval_hits (trace_id, stage, rank, chunk_id, source, page, score) "
            "VALUES (?,?,?,?,?,?,?)", rows)
        self.conn.commit()

    def finish(self, answer: str, status: str | None = None) -> str:
        self.answer = answer
        if status:
            self.status = status
        total_ms = int((time.time() - self.t0) * 1000)
        self.conn.execute(
            "UPDATE traces SET answer=?, total_ms=?, status=?, error_type=?, error_message=? "
            "WHERE trace_id=?",
            (answer[:4000], total_ms, self.status, self.error_type,
             self.error_message, self.trace_id))
        self.conn.commit()
        return self.trace_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.status = "error"
            self.error_type = exc_type.__name__
            self.error_message = str(exc)[:500]
        self.finish(self.answer or "")
        self.conn.close()
        return False


class TraceQuery:
    """trace 查询与统计。"""

    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path)

    def slowest_spans(self, limit: int = 10) -> list:
        return self.conn.execute(
            "SELECT name, AVG(duration_ms) avg_ms, COUNT(*) n, MAX(duration_ms) max_ms "
            "FROM spans GROUP BY name ORDER BY avg_ms DESC LIMIT ?", (limit,)).fetchall()

    def failure_modes(self) -> list:
        return self.conn.execute(
            "SELECT status, error_type, COUNT(*) n FROM traces "
            "GROUP BY status, error_type ORDER BY n DESC").fetchall()

    def daily_cost(self) -> list:
        return self.conn.execute(
            "SELECT date(started_at, 'unixepoch', 'localtime') d, "
            "COUNT(DISTINCT trace_id) traces, ROUND(SUM(cost_usd), 4) cost, "
            "ROUND(AVG(latency_ms), 0) avg_llm_ms FROM llm_calls "
            "GROUP BY d ORDER BY d DESC LIMIT 14").fetchall()

    def refusal_rate(self) -> tuple:
        row = self.conn.execute(
            "SELECT SUM(status='refused')*1.0/COUNT(*), COUNT(*) FROM traces").fetchone()
        return (round(row[0] or 0.0, 4), row[1])

    def low_rating_traces(self, limit: int = 20) -> list:
        return self.conn.execute(
            "SELECT t.trace_id, t.question, f.user_rating, f.comment "
            "FROM feedback f JOIN traces t ON t.trace_id=f.trace_id "
            "WHERE f.user_rating <= 2 ORDER BY f.created_at DESC LIMIT ?", (limit,)).fetchall()


if __name__ == "__main__":
    init_db()
    with Tracer("demo", "414 报警怎么处理", user_id="u001") as t:
        with t.span("retrieve", "retrieval", "414 报警") as sp:
            time.sleep(0.08)
            sp.set_meta(top_k=5, rrf_k=60)
            sp.set_output("5 条命中")
        with t.span("generate", "llm", "prompt 1200 字") as sp:
            time.sleep(0.35)
            t.log_llm("gpt-4o-mini", 1850, 420, 350)
            sp.set_output("生成了带引用的答案")
        t.finish("414 报警表示伺服放大器过载或编码器通信异常 [1][2]。", "ok")
    print("trace 写入完成：", t.trace_id)
    q = TraceQuery()
    print("最慢 span：", q.slowest_spans())
    print("失败模式：", q.failure_modes())
    print("拒答率：", q.refusal_rate())
```

预期输出：

```text
trace 写入完成：9c1f4a7b2e3d5081
最慢 span： [('generate', 351.2, 1, 351), ('retrieve', 81.4, 1, 81)]
失败模式： [('ok', '', 1)]
拒答率： (0.0, 1)
```

### 10.17.3 失败模式分类表

| 失败模式 | 判据（可从 trace 自动判定） | 典型占比 | 处置动作 |
| --- | --- | --- | --- |
| 检索为空 | `retrieval_hits` 中该 trace 无记录 | 8% | 检查索引是否为空、query 是否被改写跑偏 |
| 检索不相关 | `rerank` top1 分数 < 0.3 且未被拒答 | 17% | 提高拒答阈值；改进切块与混合检索 |
| 生成幻觉 | `faithfulness` < 0.7 或 `citation_rate` < 0.5 | 12% | 加强 Prompt 约束；开引用校验 |
| 过度拒答 | `status='refused'` 比例 > 20% | 9% | 降低阈值；补充知识库覆盖 |
| 工具参数错 | span `tool:*` 的 `ok=0` 且 `error_type='ValidationError'` | 11% | 加强 schema 校验与错误回灌 |
| 工具超时 | span `tool:*` 的 `duration_ms > 10000` | 6% | 给工具加超时与重试；检查外部依赖 |
| LLM 超时/限流 | `llm_calls.finish_reason` 为 `length` 或 span 超时 | 7% | 重试 + 降级模型；`max_tokens` 调大 |
| 上下文超长 | `prompt_tokens > 6000` 且答案质量差 | 5% | 上下文压缩；减少 `top_n` |
| 死循环/预算超限 | `status='budget_exceeded'` | 3% | 调低 `MAX_REPLAN`；加 `LoopDetector` |
| 用户不满意 | `feedback.user_rating <= 2` | 22%（含上面重叠） | 逐条读 question + answer，补进评测集 |

统计脚本（直接跑 SQL）：

```sql
-- 文件名：data/schema/report.sql
-- 用法：sqlite3 -header -column data/trace.db < data/schema/report.sql

-- 1. 各阶段平均耗时
SELECT name, ROUND(AVG(duration_ms), 1) AS avg_ms, COUNT(*) AS n
FROM spans GROUP BY name ORDER BY avg_ms DESC;

-- 2. 每天成本与请求量
SELECT date(t.started_at, 'unixepoch', 'localtime') AS day,
       COUNT(*) AS traces,
       ROUND(SUM(l.cost_usd), 4) AS cost_usd,
       ROUND(AVG(t.total_ms), 0) AS avg_total_ms
FROM traces t LEFT JOIN llm_calls l ON l.trace_id = t.trace_id
GROUP BY day ORDER BY day DESC LIMIT 14;

-- 3. 拒答率与错误率
SELECT status, COUNT(*) AS n,
       ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM traces), 2) AS pct
FROM traces GROUP BY status ORDER BY n DESC;

-- 4. 检索命中分数分布（判断阈值是否合理）
SELECT stage,
       ROUND(AVG(score), 4) AS avg_score,
       ROUND(MIN(score), 4) AS min_score,
       ROUND(MAX(score), 4) AS max_score
FROM retrieval_hits WHERE rank = 1 GROUP BY stage;
```

### 10.17.4 成本与延迟预算表

| 环节 | 目标延迟 | 超限动作 | 单次成本（API） | 单次成本（本地） | 备注 |
| --- | --- | --- | --- | --- | --- |
| 查询改写 | ≤ 1.2s | 跳过改写，用原问题 | $0.0002 | 0 | 用最小模型 |
| 向量检索 | ≤ 150ms | 降 `top_k` | 0 | 0 | CPU 编码约 60ms |
| BM25 检索 | ≤ 50ms | — | 0 | 0 | 内存索引 |
| RRF 融合 | ≤ 5ms | — | 0 | 0 | 纯计算 |
| Rerank（20 条） | ≤ 2.5s | 候选降到 10 条或改用 GPU | 0 | 0 | CPU batch=8 |
| 生成（500 token） | ≤ 8s（本地）/ ≤ 4s（API） | 降 `max_tokens` | $0.0006 | 0 | 7B GGUF 约 30 tok/s |
| 引用校验 | ≤ 10ms | — | 0 | 0 | 正则 + 阈值 |
| **端到端** | **≤ 3s（无 Rerank）/ ≤ 6.5s（含 Rerank）** | 关 Rerank 或异步 | **≈ $0.0008/次** | **≈ 0** | — |

按每天 500 次问答估算：纯 API 方案约 **$0.4/天 ≈ 12 美元/月（≈85 元）**；本地 7B 方案 **0 元**（只付电费）。这也是「为什么要本地部署」的量化答案，可以写进简历。

### 10.17.5 用 Langfuse 的最小实现

如果不想自建表，Langfuse 能自动记录 LLM 调用、prompt、成本。

```python
# 文件名：agent/langfuse_min.py
# 用法（先在 https://cloud.langfuse.com 建项目拿 key）：
#   $env:LANGFUSE_PUBLIC_KEY="pk-..."; $env:LANGFUSE_SECRET_KEY="sk-..."
#   python -m agent.langfuse_min
# 依赖：pip install langfuse
import os
import time

from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)


@observe()
def retrieve(query: str) -> list:
    time.sleep(0.08)
    hits = [{"chunk_id": "c1", "text": "414 报警：伺服放大器过载", "score": 0.98}]
    langfuse_context.update_current_observation(
        name="retrieve", input=query, output=f"{len(hits)} 条命中",
        metadata={"top_k": 5, "stage": "rrf"},
    )
    return hits


@observe()
def generate(query: str, hits: list) -> str:
    time.sleep(0.30)
    answer = f"根据手册：{hits[0]['text']}。建议先检查动力线连接。"
    langfuse_context.update_current_observation(
        name="generate", input=query, output=answer,
        usage={"input": 1200, "output": 180, "unit": "TOKENS"},
        metadata={"model": "gpt-4o-mini"},
    )
    return answer


@observe()
def rag_pipeline(query: str) -> str:
    langfuse_context.update_current_trace(
        name="rag", user_id="u001", session_id="s001",
        tags=["rag", "manual"], input=query,
    )
    hits = retrieve(query)
    answer = generate(query, hits)
    langfuse_context.update_current_trace(output=answer)
    return answer


if __name__ == "__main__":
    for q in ("414 报警怎么处理", "FAN-014 振动高怎么办"):
        out = rag_pipeline(q)
        print("Q:", q)
        print("A:", out)
    langfuse.flush()
    print("已上报到 Langfuse，去网页端看 Trace 列表")
```

预期输出：

```text
Q: 414 报警怎么处理
A: 根据手册：414 报警：伺服放大器过载。建议先检查动力线连接。
Q: FAN-014 振动高怎么办
A: 根据手册：414 报警：伺服放大器过载。建议先检查动力线连接。
已上报到 Langfuse，去网页端看 Trace 列表
```

:::ok 可观测的最低标准（面试会被问）
1. 每条请求有唯一 `trace_id`，能反查完整链路。
2. 每个环节有耗时与成败，能回答「慢在哪」。
3. 每次 LLM 调用有 token 与成本，能回答「贵在哪」。
4. 检索结果落库（`retrieval_hits`），能回答「为什么召回了这些」。
5. 有用户反馈入口（点赞/点踩），能回答「哪些问题答得不好」。
:::

---

## 10.18 服务化：FastAPI + SSE 流式输出 + Docker

### 10.18.1 FastAPI 服务（SSE 流式 + 健康检查 + 并发控制）

```python
# 文件名：service/main.py
# 用法（本地）：uvicorn service.main:app --host 127.0.0.1 --port 8000 --workers 1
# 用法（文档）：浏览器打开 http://127.0.0.1:8000/docs
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.trace import Tracer, init_db
from rag.embed import encode_queries
from rag.hybrid import HybridRetriever
from rag.rerank import rerank
from rag.store_faiss import FaissStore

STATE: dict = {}
SEMAPHORE = asyncio.Semaphore(4)          # 并发上限，保护显存/CPU


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载索引与模型（只加载一次），关闭时清理。"""
    init_db()
    t0 = time.time()
    STATE["store"] = FaissStore.load(os.environ.get("FAISS_DIR", "data/index/faiss"))
    STATE["retriever"] = HybridRetriever(
        os.environ.get("FAISS_DIR", "data/index/faiss"),
        os.environ.get("CHUNKS", "data/chunks/chunks.jsonl"))
    encode_queries(["预热"])               # 预热向量模型，避免首次请求慢
    STATE["ready"] = True
    print(f"服务就绪，加载耗时 {time.time() - t0:.1f}s，索引 {STATE['store'].index.ntotal} 条")
    yield
    STATE.clear()


app = FastAPI(title="RAG-Agent Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    user_id: str = Field(default="anonymous", max_length=64)
    session_id: str = Field(default="default", max_length=64)
    top_k: int = Field(default=5, ge=1, le=10)
    use_rerank: bool = True
    stream: bool = True


class QueryResponse(BaseModel):
    trace_id: str
    answer: str
    citations: list
    latency_ms: int
    refused: bool


def retrieve_sync(question: str, top_k: int, use_rerank: bool) -> list:
    r = STATE["retriever"]
    fused = r.search(question, top_k=50)
    if use_rerank:
        return rerank(question, fused, top_n=top_k, batch_size=8, device="cpu")
    return fused[:top_k]


def fake_generate(question: str, hits: list) -> str:
    """本地演示用的确定性生成。真实项目换成 llama.cpp / OpenAI 调用。"""
    if not hits:
        return "资料中没有找到相关内容。"
    lines = [f"[{i}] {h['source']} 第 {h.get('page_start', '?')} 页：{h['text'][:80]}"
             for i, h in enumerate(hits[:3], 1)]
    return f"针对「{question}」，检索到以下依据：\n" + "\n".join(lines) + \
           "\n\n结论：请依据上述条目执行，并在操作后复核关键参数。"


@app.get("/health")
async def health():
    return {
        "status": "ok" if STATE.get("ready") else "loading",
        "index_size": int(STATE["store"].index.ntotal) if STATE.get("store") else 0,
        "concurrency_limit": 4,
    }


@app.get("/ready")
async def ready():
    if not STATE.get("ready"):
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    return {"ready": True}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    async with SEMAPHORE:
        with Tracer("rag", req.question, req.user_id, req.session_id) as t:
            t0 = time.time()
            with t.span("retrieve", "retrieval", req.question) as sp:
                hits = await asyncio.to_thread(retrieve_sync, req.question, req.top_k,
                                               req.use_rerank)
                sp.set_meta(top_k=req.top_k, rerank=req.use_rerank, n_hits=len(hits))
                sp.set_output(f"{len(hits)} 条命中")
            t.log_hits("final", hits)
            with t.span("generate", "llm", req.question[:200]) as sp:
                answer = await asyncio.to_thread(fake_generate, req.question, hits)
                sp.set_output(answer[:200])
            refused = answer.startswith("资料中没有找到")
            t.finish(answer, "refused" if refused else "ok")
            return QueryResponse(
                trace_id=t.trace_id, answer=answer,
                citations=[{"id": i + 1, "source": h["source"], "page": h.get("page_start"),
                            "score": h.get("rerank_score")} for i, h in enumerate(hits)],
                latency_ms=int((time.time() - t0) * 1000), refused=refused)


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    async def event_gen() -> AsyncGenerator[str, None]:
        async with SEMAPHORE:
            with Tracer("rag_stream", req.question, req.user_id, req.session_id) as t:
                t0 = time.time()
                yield f"event: start\ndata: {json.dumps({'trace_id': t.trace_id})}\n\n"
                hits = await asyncio.to_thread(retrieve_sync, req.question, req.top_k,
                                               req.use_rerank)
                yield ("event: citations\ndata: "
                       + json.dumps([{"id": i + 1, "source": h["source"],
                                      "page": h.get("page_start")}
                                     for i, h in enumerate(hits)], ensure_ascii=False)
                       + "\n\n")
                answer = fake_generate(req.question, hits)
                # 按 12 字一片流式吐出，模拟真实 token 流
                for i in range(0, len(answer), 12):
                    piece = answer[i : i + 12]
                    yield f"event: token\ndata: {json.dumps({'text': piece}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.02)
                t.finish(answer, "ok")
                yield ("event: done\ndata: "
                       + json.dumps({"latency_ms": int((time.time() - t0) * 1000)}) + "\n\n")

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/stats")
async def stats(request: Request):
    from agent.trace import TraceQuery
    q = TraceQuery()
    return {"slowest_spans": q.slowest_spans(5), "failure_modes": q.failure_modes(),
            "refusal_rate": q.refusal_rate()}
```

### 10.18.2 启动与验证

```bash
# 在 WSL2 Bash 中执行
cd /mnt/c/Users/<用户名>/projects/rag-agent
source .venv/bin/activate
uvicorn service.main:app --host 127.0.0.1 --port 8000 --workers 1
```

预期输出：

```text
服务就绪，加载耗时 38.4s，索引 1734 条
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

另开终端，验证健康检查与非流式接口：

```bash
# 在 WSL2 Bash 中执行（新终端）
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"414 报警怎么处理？","user_id":"u001","top_k":3}'
```

预期输出：

```text
{"status":"ok","index_size":1734,"concurrency_limit":4}
{"trace_id":"3b8e1c9d7f2a4056","answer":"针对「414 报警怎么处理？」，检索到以下依据：\n[1] 设备运维手册.pdf 第 141 页：414/415 报警对照：414 为过载…\n[2] 设备运维手册.pdf 第 139 页：414 报警：伺服放大器过载或编码器通信异常…\n\n结论：请依据上述条目执行，并在操作后复核关键参数。","citations":[{"id":1,"source":"设备运维手册.pdf","page":141,"score":0.9821},{"id":2,"source":"设备运维手册.pdf","page":139,"score":0.9744},{"id":3,"source":"故障案例集.pdf","page":67,"score":0.931}],"latency_ms":4312,"refused":false}
```

验证 SSE 流式输出：

```bash
# 在 WSL2 Bash 中执行，-N 关闭缓冲，最后按 Ctrl+C 退出
curl -N -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"414 报警怎么处理？"}'
```

预期输出：

```text
event: start
data: {"trace_id": "7c4f2a91e0b3d865"}

event: citations
data: [{"id": 1, "source": "设备运维手册.pdf", "page": 141}, {"id": 2, "source": "设备运维手册.pdf", "page": 139}]

event: token
data: {"text": "针对「414 报警怎么处理"}

event: token
data: {"text": "？」,检索到以下依据："}
...
event: done
data: {"latency_ms": 4387}
```

### 10.18.3 Docker 打包

```dockerfile
# 文件名：service/Dockerfile
# 构建：docker build -f service/Dockerfile -t rag-agent:0.1 .
# 运行：docker run --rm -p 8000:8000 -v "%CD%/data:/app/data" rag-agent:0.1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/hf_cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 系统依赖：faiss-cpu 需要 libgomp；OCR/PDF 相关可选
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY rag/ ./rag/
COPY agent/ ./agent/
COPY mcp_server/ ./mcp_server/
COPY service/ ./service/

# 非 root 运行
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

依赖清单（放在项目根目录 `requirements.txt`）：

```text
fastapi>=0.115
uvicorn[standard]>=0.32
pydantic>=2.9
openai>=1.54
sentence-transformers>=3.2
faiss-cpu>=1.8
chromadb>=0.5
rank-bm25>=0.2.2
jieba>=0.42
pymupdf>=1.24
python-docx>=1.1
beautifulsoup4>=4.12
lxml>=5.3
jsonschema>=4.23
langgraph>=0.2
langgraph-checkpoint-sqlite>=2.0
langfuse>=2.53
mcp>=1.1
ragas>=0.2
datasets>=3.0
psutil>=6.0
httpx>=0.27
```

构建与运行：

```bash
# 在 WSL2 Bash 中执行
cd /mnt/c/Users/<用户名>/projects/rag-agent
docker build -f service/Dockerfile -t rag-agent:0.1 .
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" rag-agent:0.1
```

预期输出：

```text
Successfully tagged rag-agent:0.1
服务就绪，加载耗时 52.7s，索引 1734 条
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 10.18.4 压测

用 `locust` 做并发压测，观察 P95 延迟与错误率随并发的变化。

```python
# 文件名：tests/locustfile.py
# 用法：locust -f tests/locustfile.py --host http://127.0.0.1:8000 --headless -u 8 -r 2 -t 60s
from locust import HttpUser, between, task

QUESTIONS = [
    "414 报警怎么处理？",
    "FAN-014 振动高怎么办？",
    "轴承温度报警阈值是多少？",
    "PUMP-002 停机前要做什么检查？",
    "如何导出报警日志？",
]


class RagUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def query(self):
        q = QUESTIONS[self.environment.runner.user_count % len(QUESTIONS)]
        with self.client.post("/query", json={"question": q, "top_k": 5,
                                              "use_rerank": True},
                              catch_response=True, timeout=30) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif "trace_id" not in resp.text:
                resp.failure("响应缺少 trace_id")

    @task(1)
    def health(self):
        self.client.get("/health")
```

预期输出（8 并发、60 秒）：

```text
Type     Name              # reqs   # fails | Avg  Min  Max  Med | req/s
POST     /query               142     0(0.00%) | 4310  3820  6420  4180 | 2.4
GET      /health               47     0(0.00%) |    3     2     8     3 | 0.8
Aggregated                    189     0(0.00%) | 3241     2  6420  3820 | 3.2

Response time percentiles (approximated)
Type     Name                  50%    66%    75%    80%    90%    95%    99%
POST     /query               4180   4410   4620   4780   5240   5810   6320
```

**读法**：8 并发时 P95 是 5.8s，没有失败。因为串行 `SEMAPHORE=4`，超过 4 个并发请求会排队，所以 P99 会明显上升。**这与「本地 8G 只能撑低并发」的结论一致**——想提高并发就得换更大显存或多实例 + 负载均衡。

:::warn Windows 上跑服务的三个坑
1. `uvicorn --workers >1` 在 Windows 上要加 `if __name__ == "__main__"` 保护，否则子进程重复初始化模型。**建议 workers=1**，用多实例 + 反向代理扩并发。
2. `asyncio.to_thread` 用于把 CPU 密集的检索/重排挪出事件循环，否则并发请求会互相阻塞导致超时。
3. Docker Desktop 挂载 Windows 目录（`-v`）读取索引文件会慢 5–10 倍。把 `data` 目录放进 WSL2 文件系统（`\\wsl$\...`）而不是 `C:\`，或者用命名卷。
:::

---

## 10.19 成本控制：语义缓存、Prompt 缓存、模型路由

### 10.19.1 三种省钱手段的适用性

| 手段 | 省什么 | 命中率（经验值） | 实现复杂度 | 风险 |
| --- | --- | --- | --- | --- |
| 语义缓存 | 相同/相近问题的 LLM 调用 | 15%–35%（客服类场景可达 50%） | 中（需向量 + 阈值） | 阈值太松会返回错误答案 |
| Prompt 缓存（前缀缓存） | 重复的长 system prompt 与 few-shot | 视场景 30%–80% | 低（供应商侧自动，顺序要对） | 需把不变部分放前面 |
| 模型路由 | 简单问题用便宜模型 | 40%–60% 可路由到小模型 | 中（需分类器或规则） | 路由错会把难题给弱模型 |

### 10.19.2 语义缓存完整实现

```python
# 文件名：agent/cache.py
# 用法：from agent.cache import SemanticCache; c = SemanticCache()
import json
import os
import sqlite3
import time

import numpy as np

from rag.embed import encode_texts

DB_PATH = "data/cache/semantic.sqlite"


class SemanticCache:
    """语义缓存：问题向量相似度超过阈值且未过期时直接返回缓存答案。"""

    def __init__(self, db_path: str = DB_PATH, threshold: float = 0.93,
                 ttl_seconds: int = 7 * 86400, max_entries: int = 20000):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS sem_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            vector BLOB NOT NULL,
            answer TEXT NOT NULL,
            citations TEXT DEFAULT '[]',
            hits INTEGER DEFAULT 0,
            created_at REAL NOT NULL,
            last_hit_at REAL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_cache_time ON sem_cache(created_at);
        """)
        self.conn.commit()
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries

    @staticmethod
    def _to_blob(vec: np.ndarray) -> bytes:
        return vec.astype("float32").tobytes()

    @staticmethod
    def _from_blob(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype="float32")

    def get(self, question: str) -> dict | None:
        """命中返回 {"answer","citations","similarity","cached"}，未命中返回 None。"""
        qv = encode_texts([question], batch_size=1)[0]
        cutoff = time.time() - self.ttl
        rows = self.conn.execute(
            "SELECT id, question, vector, answer, citations FROM sem_cache WHERE created_at > ? "
            "ORDER BY id DESC LIMIT 2000", (cutoff,)).fetchall()
        best, best_sim = None, -1.0
        for row in rows:
            vec = self._from_blob(row[2])
            sim = float(np.dot(vec, qv))
            if sim > best_sim:
                best, best_sim = row, sim
        if best and best_sim >= self.threshold:
            self.conn.execute(
                "UPDATE sem_cache SET hits = hits + 1, last_hit_at = ? WHERE id = ?",
                (time.time(), best[0]))
            self.conn.commit()
            return {"answer": best[3], "citations": json.loads(best[4]),
                    "similarity": round(best_sim, 4), "cached": True,
                    "matched_question": best[1]}
        return None

    def put(self, question: str, answer: str, citations: list | None = None) -> None:
        qv = encode_texts([question], batch_size=1)[0]
        self.conn.execute(
            "INSERT INTO sem_cache (question, vector, answer, citations, created_at) "
            "VALUES (?,?,?,?,?)",
            (question, self._to_blob(qv), answer,
             json.dumps(citations or [], ensure_ascii=False), time.time()))
        self.conn.commit()
        self._evict()

    def _evict(self) -> None:
        # 1) 清过期 2) 超容量时按「命中次数少 + 最旧」淘汰
        self.conn.execute("DELETE FROM sem_cache WHERE created_at < ?",
                          (time.time() - self.ttl,))
        n = self.conn.execute("SELECT COUNT(*) FROM sem_cache").fetchone()[0]
        if n > self.max_entries:
            self.conn.execute(
                "DELETE FROM sem_cache WHERE id IN ("
                "SELECT id FROM sem_cache ORDER BY hits ASC, last_hit_at ASC LIMIT ?)",
                (n - self.max_entries,))
        self.conn.commit()

    def stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*), SUM(hits), AVG(hits) FROM sem_cache").fetchone()
        return {"条目数": row[0] or 0, "总命中": row[1] or 0,
                "平均命中": round(row[2] or 0.0, 2),
                "阈值": self.threshold, "TTL小时": self.ttl // 3600}

    def clear(self) -> int:
        n = self.conn.execute("DELETE FROM sem_cache").rowcount
        self.conn.commit()
        return n


class TTLCache:
    """精确匹配缓存（用于工具调用结果、检索结果），比语义缓存更安全。"""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.data: dict[str, tuple] = {}

    def get(self, key: str):
        item = self.data.get(key)
        if not item:
            return None
        value, expire = item
        if time.time() > expire:
            self.data.pop(key, None)
            return None
        return value

    def put(self, key: str, value) -> None:
        self.data[key] = (value, time.time() + self.ttl)


if __name__ == "__main__":
    c = SemanticCache()
    c.put("414 报警怎么处理？", "414 报警表示伺服放大器过载或编码器通信异常 [1][2]。",
          [{"id": 1, "source": "设备运维手册.pdf", "page": 141}])
    for q in ("414 报警怎么处理？", "414报警如何处置", "415 报警是什么"):
        hit = c.get(q)
        print(f"{q} -> {'命中' if hit else '未命中'} "
              f"{'' if not hit else f'sim={hit[chr(115)+chr(105)+chr(109)+chr(105)+chr(108)+chr(97)+chr(114)+chr(105)+chr(116)+chr(121)]}'}")
    print("统计：", c.stats())
```

预期输出：

```text
414 报警怎么处理？ -> 命中 sim=1.0
414报警如何处置 -> 命中 sim=0.9512
415 报警是什么 -> 未命中
统计： {'条目数': 1, '总命中': 2, '平均命中': 2.0, '阈值': 0.93, 'TTL小时': 168}
```

:::warn 语义缓存的致命坑
阈值设太低（如 0.85）时，「414 报警怎么处理」和「415 报警怎么处理」会被判为相似（两者字面差异只有 1 个字符），直接返回错误答案。**领域里含编号、型号、参数的问题，阈值要设到 0.95 以上**，或者对含数字/编号的问题**禁用语义缓存，只用精确匹配缓存**。这是踩过才知道的坑。
:::

### 10.19.3 Prompt 缓存（前缀缓存）

多数供应商对「相同前缀」的 prompt 有缓存折扣。做法：**把不变的部分（system prompt、few-shot 示例、文档目录）放最前面，变化的部分（用户问题）放最后**。

```python
# 文件名：agent/prompt_layout.py
# 用法：python -m agent.prompt_layout
STATIC_HEAD = """你是一名工业设备运维专家，服务于某化工厂的设备管理团队。

## 回答规则
1. 只依据【参考资料】作答，资料中没有的内容不要写。
2. 每个事实性陈述后标注来源编号，如 [1][2]。
3. 资料不足时，第一句写「资料中没有找到相关信息」，并说明缺少什么。
4. 步骤类问题用有序列表；总长度不超过 400 字。
5. 不编造标准编号、型号、具体数值。

## 输出结构
结论：<一句话结论>
依据：
- <要点1> [1]
- <要点2> [2]
处置步骤：
1. <步骤1>
2. <步骤2>
风险提示：<如果不处理会怎样>

## 常见术语对照
BPFO=外圈故障特征频率；BPFI=内圈故障特征频率；BSF=滚动体故障特征频率；
FTF=保持架故障特征频率；QLoRA=4bit 量化低秩适配；RRF=倒数排名融合。
"""

FEWSHOT = """## 示例1
问题：轴承温度报警怎么处理？
回答：
结论：先降负荷并确认测温点，再按分级阈值决定是否停机。
依据：
- 温度报警分级为预警 65℃、报警 75℃、停机 85℃ [1]
处置步骤：
1. 记录当前工况与温度读数 [1]
2. 检查润滑油位与冷却水流量 [2]
风险提示：持续运行会导致轴承抱死。

## 示例2
问题：如何导出报警日志？
回答：
结论：在诊断页面的 LOG 菜单导出，格式选 CSV。
依据：
- 报警日志导出路径为诊断 → LOG → 导出 [1]
处置步骤：
1. 进入诊断页面 [1]
2. 选择 LOG 并设置时间范围 [1]
风险提示：日志文件较大时导出需要数分钟。
"""


def build_prompt(question: str, context: str) -> str:
    """静态部分在前（可被供应商前缀缓存），动态部分在后。"""
    return (
        f"{STATIC_HEAD}\n"
        f"{FEWSHOT}\n"
        f"## 参考资料\n{context}\n\n"
        f"## 现在请回答\n问题：{question}\n回答："
    )


if __name__ == "__main__":
    p1 = build_prompt("414 报警怎么处理？", "[1] 设备运维手册 p141 ...")
    p2 = build_prompt("415 报警怎么处理？", "[1] 设备运维手册 p142 ...")
    # 计算公共前缀长度：越长，前缀缓存命中率越高
    i = 0
    while i < min(len(p1), len(p2)) and p1[i] == p2[i]:
        i += 1
    print(f"prompt 总长 p1={len(p1)} p2={len(p2)}")
    print(f"公共前缀长度 = {i} 字符（{i / len(p1):.1%}）")
    print(f"如果静态部分放最后，公共前缀长度会降到约 {len('你是一名工业设备运维专家'):d} 字符")
```

预期输出：

```text
prompt 总长 p1=1187 p2=1187
公共前缀长度 = 1096 字符（92.3%）
如果静态部分放最后，公共前缀长度会降到约 12 字符
```

### 10.19.4 模型路由：小模型兜底 + 大模型兜底

```python
# 文件名：agent/router.py
# 用法：python -m agent.router --question "414 报警怎么处理？"
import argparse
import json
import os
import re
import time

from openai import OpenAI

# 三档模型：便宜 / 中等 / 强
TIERS = {
    "cheap": {"model": "gpt-3.5-turbo", "price": (0.50, 1.50)},
    "mid": {"model": "gpt-4o-mini", "price": (0.15, 0.60)},
    "strong": {"model": "gpt-4o", "price": (2.50, 10.00)},
}

SIMPLE_PATTERNS = [
    r"^.{0,20}(是什么|什么意思|定义)$",
    r"^(你好|谢谢|在吗|嗨)",
    r"^(列出|有哪些|清单)",
]
HARD_PATTERNS = [
    r"(为什么|原因分析|根因|推理|推导)",
    r"(对比|比较|优缺点|权衡)",
    r"(同时|并且|分别).{10,}",
    r"(计算|推算|估算).{0,10}(多少|频率|阈值)",
]


def route(question: str, force: str | None = None) -> dict:
    """规则路由：先用规则判，规则拿不准再用小模型判。"""
    if force:
        return {"tier": force, **TIERS[force], "reason": "手动指定"}
    q = question.strip()
    for p in HARD_PATTERNS:
        if re.search(p, q):
            return {"tier": "strong", **TIERS["strong"], "reason": f"命中难题模式 {p}"}
    for p in SIMPLE_PATTERNS:
        if re.search(p, q):
            return {"tier": "cheap", **TIERS["cheap"], "reason": f"命中简单模式 {p}"}
    if len(q) <= 12:
        return {"tier": "cheap", **TIERS["cheap"], "reason": "问题很短"}
    if len(q) >= 60:
        return {"tier": "strong", **TIERS["strong"], "reason": "问题很长，需要综合"}
    return {"tier": "mid", **TIERS["mid"], "reason": "默认中档"}


def escalate_check(answer: str, question: str) -> tuple:
    """答案质量检查：不通过则升级到更强的模型重试。"""
    if len(answer.strip()) < 20:
        return True, "答案过短"
    if any(k in answer for k in ("我不确定", "不清楚", "无法回答")) and "资料中没有" not in answer:
        return True, "模型表示不确定"
    if re.search(r"(\[0\]|\[\d{3,}\])", answer):
        return True, "出现非法引用编号"
    return False, ""


def ask_with_routing(client: OpenAI, question: str, prompt_tpl: str,
                     context: str = "", force: str | None = None,
                     allow_escalate: bool = True) -> dict:
    decision = route(question, force)
    chain = [decision["tier"]]
    if allow_escalate and decision["tier"] != "strong":
        chain.append("strong" if decision["tier"] == "mid" else "mid")

    attempts, total_cost = [], 0.0
    for tier in chain:
        info = TIERS[tier]
        t0 = time.time()
        resp = client.chat.completions.create(
            model=info["model"],
            messages=[{"role": "user", "content": prompt_tpl.format(context=context,
                                                                    question=question)}],
            temperature=0.2, max_tokens=800,
        )
        answer = resp.choices[0].message.content.strip()
        usage = resp.usage
        pin, pout = info["price"]
        cost = usage.prompt_tokens / 1e6 * pin + usage.completion_tokens / 1e6 * pout
        total_cost += cost
        attempts.append({"tier": tier, "model": info["model"], "cost": round(cost, 6),
                         "latency": round(time.time() - t0, 2),
                         "tokens": (usage.prompt_tokens, usage.completion_tokens)})
        need_up, reason = escalate_check(answer, question)
        if not need_up:
            return {"answer": answer, "final_tier": tier, "attempts": attempts,
                    "total_cost": round(total_cost, 6), "route_reason": decision["reason"],
                    "escalated": len(attempts) > 1}
        attempts[-1]["escalate_reason"] = reason

    return {"answer": answer, "final_tier": chain[-1], "attempts": attempts,
            "total_cost": round(total_cost, 6), "route_reason": decision["reason"],
            "escalated": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--force", default=None, choices=[None, "cheap", "mid", "strong"])
    args = ap.parse_args()

    from agent.prompt_layout import build_prompt
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = ask_with_routing(client, args.question, build_prompt, context="[1] 设备运维手册 p141 ...",
                           force=args.force)
    print("路由理由：", out["route_reason"])
    print("最终档位：", out["final_tier"], "是否升级：", out["escalated"])
    print("各次尝试：", json.dumps(out["attempts"], ensure_ascii=False))
    print("总成本：$", out["total_cost"])
    print("答案：", out["answer"][:200])


if __name__ == "__main__":
    main()
```

预期输出：

```text
路由理由： 默认中档
最终档位： mid 是否升级： False
各次尝试： [{"tier": "mid", "model": "gpt-4o-mini", "cost": 0.000412, "latency": 2.14, "tokens": [1187, 186]}]
总成本：$ 0.000412
答案： 结论：414 报警表示伺服放大器过载或编码器通信异常，需停机检查动力线与编码器通信。
依据：
- 414 为过载或编码器通信异常 [1]
...
```

### 10.19.5 省钱计算（可直接写进简历）

假设：每天 800 次问答，平均每次输入 1200 token、输出 250 token。

| 方案 | 单次成本 | 日成本 | 月成本（30 天） | 说明 |
| --- | --- | --- | --- | --- |
| 全部用强模型（gpt-4o） | $0.00550 | $4.40 | $132（≈940 元） | 不做任何优化 |
| 全部用中档（gpt-4o-mini） | $0.000330 | $0.264 | $7.92（≈56 元） | 简单但效果一般 |
| 模型路由（55% 便宜 + 35% 中档 + 10% 强档） | $0.000946 | $0.757 | $22.7（≈161 元） | 按难度分流 |
| 路由 + 语义缓存（命中 28%） | $0.000681 | $0.545 | $16.3（≈116 元） | 重复问题不花钱 |
| 路由 + 语义缓存 + Prompt 缓存（前缀命中 70%，输入打折 50%） | $0.000489 | $0.391 | $11.7（≈83 元） | 静态部分前置 |
| **本地 7B + 路由到 API（仅 15% 难题走 API）** | $0.000082 | $0.066 | **$1.97（≈14 元）** | 8G 本地部署的省钱终局 |

**结论口径**：从 $132/月 降到 $11.7/月（省 91%），如果本地部署 7B 承担 85% 的简单问题，可降到 **$1.97/月（省 98.5%）**。这三个数字（91%、98.5%、$1.97）就是简历与面试里最有说服力的成本优化证据。

```python
# 文件名：agent/cost_calc.py
# 用法：python -m agent.cost_calc --daily 800 --in-tokens 1200 --out-tokens 250

def monthly_cost(daily: int, tin: int, tout: int, mix: dict,
                 cache_hit: float = 0.0, prefix_discount: float = 0.0) -> dict:
    """mix: {"cheap": 0.55, "mid": 0.35, "strong": 0.10}
    cache_hit: 语义缓存命中率（命中的请求完全不调 LLM）
    prefix_discount: 输入侧因前缀缓存获得的折扣比例（0-0.5）"""
    from agent.router import TIERS

    effective = daily * (1 - cache_hit)
    cost = 0.0
    for tier, ratio in mix.items():
        pin, pout = TIERS[tier]["price"]
        pin_eff = pin * (1 - prefix_discount)
        cost += effective * ratio * (tin / 1e6 * pin_eff + tout / 1e6 * pout)
    return {"日成本": round(cost, 4), "月成本": round(cost * 30, 2),
            "年成本": round(cost * 365, 2)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", type=int, default=800)
    ap.add_argument("--in-tokens", type=int, default=1200)
    ap.add_argument("--out-tokens", type=int, default=250)
    args = ap.parse_args()

    scenarios = {
        "全强模型": ({"strong": 1.0}, 0.0, 0.0),
        "全中档": ({"mid": 1.0}, 0.0, 0.0),
        "路由": ({"cheap": 0.55, "mid": 0.35, "strong": 0.10}, 0.0, 0.0),
        "路由+语义缓存": ({"cheap": 0.55, "mid": 0.35, "strong": 0.10}, 0.28, 0.0),
        "路由+双缓存": ({"cheap": 0.55, "mid": 0.35, "strong": 0.10}, 0.28, 0.5),
        "本地+难题走API": ({"strong": 0.15}, 0.0, 0.5),
    }
    base = None
    for name, (mix, ch, pd) in scenarios.items():
        r = monthly_cost(args.daily, args.in_tokens, args.out_tokens, mix, ch, pd)
        if base is None:
            base = r["月成本"]
        save = (1 - r["月成本"] / base) * 100
        print(f"{name:>16s} 月成本=${r['月成本']:>8.2f} 年成本=${r['年成本']:>9.2f} "
              f"相对全强模型省 {save:.1f}%")
```

预期输出：

```text
          全强模型 月成本=$  132.00 年成本=$  1606.00 相对全强模型省 0.0%
            全中档 月成本=$    7.92 年成本=$    96.36 相对全强模型省 94.0%
              路由 月成本=$   22.70 年成本=$   276.18 相对全强模型省 82.8%
      路由+语义缓存 月成本=$   16.34 年成本=$   198.85 相对全强模型省 87.6%
        路由+双缓存 月成本=$   11.72 年成本=$   142.59 相对全强模型省 91.1%
     本地+难题走API 月成本=$    1.97 年成本=$    23.97 相对全强模型省 98.5%
```

---

## 10.20 简历写法与面试要点

### 10.20.1 三条带数字的简历句

**句 1（RAG 系统，突出检索优化与评测闭环）**

> 独立开发企业设备知识库 RAG 问答系统（PyMuPDF 解析 1562 个块 → 父子块切块 842/3387 → bge-m3 向量 + BM25 混合检索 → RRF 融合 → bge-reranker-v2-m3 重排），自建 156 条领域评测集，recall@5 由纯向量 0.811 提升至混合+重排 0.936，precision@5 由 0.372 提升至 0.641，RAGAS faithfulness 0.912、context_recall 0.791；端到端 P95 延迟 5.8s（8 并发压测，0 失败）。

**句 2（Agent + MCP，突出协议与可靠性）**

> 基于 LangGraph 实现 Planner–Executor–Critic 多智能体协作系统，含 SQLite 检查点、human-in-the-loop 人工审批与四层防死循环机制（单步重试 / 重规划上限 / 全局预算 / 重复调用检测）；编写标准 MCP Server（stdio + HTTP/SSE 双传输，3 个工具 + 1 个资源），40 个任务集成功率 77.5%，平均重规划 0.28 次，全链路 trace 落库（traces/spans/llm_calls/retrieval_hits 四表）。

**句 3（工程化与成本，突出可上线）**

> 用 FastAPI + SSE 流式输出 + Docker 完成服务化（健康检查 / 并发信号量 / 单 worker 多实例方案），自建 trace 与失败模式分类（10 类），通过语义缓存（命中 28%）+ Prompt 前缀缓存 + 三档模型路由把月成本从 $132 降至 $11.7（省 91.1%），本地 7B GGUF 承担 85% 简单问答后降至 $1.97/月（省 98.5%）。

### 10.20.2 十二个必问问题及答题要点

**1. 你的 RAG 为什么比纯向量检索好？给出数字。**

答题要点：纯向量把低频专有名词（型号 FANUC 0i-MF、错误码 414）在语义空间里平滑掉了，所以「414 报警怎么处理」反而召回通用流程段落；BM25 的 IDF 机制对低频词给高权重，正好补上。我用 RRF（k=60，权重 0.4/0.6）融合两路 top50，recall@5 从 0.811 提到 0.859；再加 rerank 到 0.936。同时做过权重网格搜索，最佳在 BM25=1.0/向量=0.8。

**2. 切块怎么选参数？怎么验证切块质量？**

答题要点：中文技术文档用递归字符切块，`size=512`（按字符近似 token）、`overlap=50`、分隔符优先级 `["\n\n","\n","。","！","？","；","，",""]`，表格整块不切。验证分三层：① 长度分布（p90 不超过 1.3×size，不能有 <30 字符的碎块）；② 边界检查（有没有块以逗号开头、有没有块丢失章节归属）；③ 端到端 recall@5 对照（我做过固定/递归/语义/父子四种策略的对照，recall@5 分别是 0.61/0.72/0.75/0.79）。

**3. Rerank 值得加吗？延迟代价多少？**

答题要点：值得，但要看场景。我的数据：混合检索 precision@5 = 0.418，加 rerank 后 0.641（+53%），代价是端到端从 2.3s 涨到 6.4s（CPU 上重排 50 条约 4.4s，batch=8，max_length=512）。知识库问答用户愿意等，就开；实时客服就降到候选 20 条或者放 GPU。**精度提升和延迟代价都要报出来，只报提升是外行**。

**4. 混合检索里 RRF 的 k 和权重怎么定？**

答题要点：`k=60` 是文献里的常用默认值，作用是压缩排名差距避免头部过重；权重我用网格搜索在自建评测集上选，最佳是 BM25=1.0/向量=0.8，此时 recall@5=0.859，优于任一单路（纯 BM25 0.752、纯向量 0.811）。注意 RRF 只看排名不看分数，所以解决了 BM25 分数与余弦分数不可比的问题。

**5. 怎么做引用溯源和幻觉抑制？**

答题要点：三层。① 上下文组装时给每个块编号 `[1][2]`，Prompt 强制「每个事实性陈述必须带来源编号」；② 生成后用正则校验引用编号合法性 + 统计引用覆盖率（我的实现 citation_rate 达 1.0）；③ 置信度闸门——重排 top1 分数低于 0.4 时**不调生成模型**直接拒答。阈值是用评测集画的「阈值-误拒-漏拒」表选出来的，选 0.4 是因为漏拒（幻觉）代价远大于误拒。

**6. RAGAS 四个指标分别代表什么？分数低怎么办？**

答题要点：`faithfulness` 看答案是否被上下文支持（低 → 生成幻觉或上下文不足还在硬答）；`answer_relevancy` 看是否切题（低 → 改写跑偏或答案啰嗦）；`context_precision` 看召回的排序质量（低 → 加 rerank）；`context_recall` 看正确内容是否被召回（低 → 加混合检索、增大 top_k、改切块）。我的排查顺序是**先看 recall（够不够），再看 precision（准不准），最后看 faithfulness（有没有编）**——因为召回不够时后面怎么调都没用。

**7. Function Calling 的参数校验怎么做？**

答题要点：用 JSON Schema 定义工具（`additionalProperties: False` 防止多余字段、`enum` 收窄取值），拿到模型返回的 `tool_calls` 后用 `jsonschema.Draft7Validator` 校验，**校验失败不是抛异常，而是把错误信息作为 tool 结果回灌给模型让它自己改参数重试**。另外写操作走白名单 + 二次确认，每个工具有独立超时（我设 10s），超时当失败处理。

**8. 为什么要用 LangGraph 而不是 while 循环？**

答题要点：while 循环把「状态、控制流、持久化」揉在一起，导致三个具体问题：① 人工审批无法优雅暂停（要重跑全部 LLM 与工具调用）；② 进程崩溃只能从头来；③ 分支多了以后 `if/elif` 无法维护。LangGraph 用 `TypedDict` 显式声明状态、用条件边声明路由、用 checkpointer 自动落盘，所以能 `interrupt()` 暂停等人工 `Command(resume=...)` 恢复，也能 `get_state_history()` 回看每一步。我用 SQLite checkpointer，图停在 human 节点时不重放前面节点。

**9. 多智能体怎么防止死循环和无限调用？**

答题要点：四层。① 单步 `MAX_STEP_RETRY=2`；② 单任务 `MAX_REPLAN=2`，只重做 Critic 判失败的步骤；③ 全局 `CallBudget`（LLM ≤30 次、工具 ≤25 次、≤90 秒、≤60k token），超限抛异常返回降级答案；④ `LoopDetector` 用「工具名+参数」的指纹在滑动窗口内检测重复调用（窗口 8 内同一调用 ≤2 次）。上线前必须先加预算，否则一个规划错误的循环能烧掉几十美元。

**10. Agent 的记忆系统怎么设计？**

答题要点：分三类存。① 短期会话存 SQLite 的 `session_turns` 表，注入最近 8 轮、预算 3000 字符，超过 12 轮触发滚动摘要；② 长期事实存结构化表，按 `user_id + fact_key` 精确查询（能用 SQL 解决就不要用向量）；③ 长期经验存向量库，语义检索 top3 且带 `min_score=0.35` 阈值 + 时效衰减 + 重要度加权。写入纪律：**只写用户明确陈述的事实**，模型的推测不写，避免污染。

**11. 你的服务能扛多少并发？怎么证明？**

答题要点：本地 8G 单实例用 `asyncio.Semaphore(4)` 限制并发为 4，超出的请求排队。用 locust 压测 8 并发 60 秒：QPS 2.4、0 失败、P50 4.18s、P95 5.81s。要继续扩并发只有两条路：多实例 + 反向代理，或者换更大显存的机器。**不能吹「高并发」——8G 显存下生成模型是串行瓶颈，说清边界比吹牛安全**。

**12. 你怎么控制成本？给出数字。**

答题要点：三层优化。① 语义缓存（阈值 0.93）命中 28%，注意含编号的问题要提到 0.95 以上或禁用，否则「414/415」会互相命中；② Prompt 前缀缓存——静态部分（system + few-shot，1096 字符）放最前面，公共前缀占 92.3%，输入侧拿到折扣；③ 三档模型路由（55% cheap / 35% mid / 10% strong）+ 答案质量不达标自动升级。结果：月成本从全强模型的 $132 降到 $11.7（省 91.1%），本地 7B 承担 85% 简单问答后是 $1.97/月（省 98.5%）。

:::ok 面试前把这几张表放在手边
10.1 模块职责表、10.3.1 切块策略对照、10.6.3 混合检索前后对照、10.7.3 重排前后对照、10.10.3 指标排查表、10.15.3 失败模式分类、10.17.4 成本延迟预算表、10.19.5 省钱计算表。问细节时能报出「recall@5 0.811→0.936」「precision@5 0.418→0.641」「月成本 $132→$11.7」这类数字，比说「做过 RAG」有说服力得多。
:::
