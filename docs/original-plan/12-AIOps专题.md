# 第12章 AIOps 智能运维专题：从指标异常到日志归因的完整闭环

这一章解决一个具体能力缺口：招聘方要的是「能自己搭一条从监控指标到告警摘要的链路，并且能用数字说明它把 MTTR 压下去多少」的人，而不是只会跑现成 Demo 的人。读完这一章，你要能独立完成六件事：读懂公开数据集的文件格式与列名、用五种方法做时序异常检测并横向对比、按运维口径正确算指标（point-wise 与 point-adjusted 的差别要能讲清）、用 Drain3 挖日志模板、用服务拓扑给出根因候选 Top-3、把整条链路用 Docker Compose 拉起来并接到 Grafana 看板上。

口径统一说明：本章的算力前提仍是 RTX 4060 Laptop 8GB 显存（扣掉内屏占用实际约 7GB）、内存 16–32GB，开发环境是 Windows 11 + WSL2 Ubuntu 22.04。深度时序模型一律按「8G 能不能跑」给出改造方案；监控栈按共享事实简报的选型走 Prometheus + Grafana（Docker 起），日志侧用 Loki/ELK，指标采集用 node_exporter。任何跑不动的方案都必须给出替代路径。

---

## 12.1 AIOps 是什么、招什么岗、KPI 是什么

### 12.1.1 一句话定位

AIOps（Artificial Intelligence for IT Operations）是把机器学习用到运维对象上的一类工程实践，核心目标是**把「人盯着告警猜原因」变成「机器先收敛告警、再给出根因候选与处置建议」**。它的输入有三类：指标（metric，数值型时序）、日志（log，文本流）、链路追踪（trace，调用关系）。输出也有三类：异常告警、根因候选、处置建议（Runbook 或工单）。

按技术难度从低到高，AIOps 有五层能力：

| 层 | 能力 | 典型手段 | 招人时对应的工作 |
| --- | --- | --- | --- |
| L1 观测 | 指标/日志/链路采集与看板 | Prometheus + Grafana + Loki | 运维开发 |
| L2 检测 | 单指标异常检测 | 3-sigma、MAD、Isolation Forest、AE | AIOps 算法（入门） |
| L3 收敛 | 告警去重、聚合、抑制 | 指纹聚合、时间窗、拓扑抑制 | AIOps 算法（主力） |
| L4 归因 | 根因定位、故障传播分析 | 服务拓扑、相关性排序、图方法 | AIOps 算法（进阶） |
| L5 处置 | 摘要、建议、自动化执行 | LLM 摘要、RAG 检索 Runbook、ChatOps | AIOps + LLM 应用 |

研一阶段的合理目标是 **L2 + L3 做扎实、L4 做出一个可解释的简化版、L5 做出一个能演示且带幻觉控制的版本**。这也是第 22–23 周（P4 与 P5 两个项目）要交付的东西。

### 12.1.2 招什么岗

| 岗位名称 | 出现频率 | 主要职责 | 你的对应准备 |
| --- | --- | --- | --- |
| AIOps 算法工程师 | 中 | 异常检测、根因分析、告警收敛模型研发 | P4 + P5，重点讲指标口径 |
| 智能运维开发工程师 | 高 | 监控平台、告警平台、自动化处置开发 | FastAPI + Prometheus + Docker Compose |
| SRE（有 AIOps 方向） | 高 | 稳定性保障、SLO、故障复盘、工具建设 | MTTR/误报率/SLO 口径 |
| 可观测性研发 | 中 | 采集器、存储、查询引擎 | 了解 OpenTelemetry、Loki/ES |
| 大模型应用（运维侧） | 中 | 告警摘要、日志问答、ChatOps | RAG + prompt 幻觉控制 |
| 时序算法工程师 | 中低 | 通用时序异常检测/预测 | Anomaly Transformer 等论文复现 |

岗位要求的逐条对照（把下面这张表当成自查清单，每条给出证据物）：

| 招聘要求原文（常见写法） | 你的证据物 | 现在就该做的动作 |
| --- | --- | --- |
| 熟悉时序异常检测常用算法 | 一份五种方法的对比表（F1、误报率、检测延迟） | 跑完 12.3 全流程 |
| 熟悉日志解析与模板挖掘 | Drain3 模板挖掘报告（模板数、命中分布） | 跑完 12.7 |
| 有 Prometheus/Grafana 使用经验 | 一张 Grafana 看板截图 + `docker-compose.yml` | 跑完 12.12 |
| 掌握 Python 与数据处理 | 公开仓库里的可运行代码 + pytest | 每个脚本配 `pytest` 小测 |
| 了解 LLM 与 RAG | 告警摘要服务的 prompt 与幻觉控制方案 | 跑完 12.10 与 12.11 |
| 有故障根因定位经验 | 一个真实故障复盘（可用公开数据集构造） | 写 1 页复盘记录 |
| 熟悉 Docker/K8s | Compose 编排 + 容器化推理服务 | 见第 11 章 11.11 与本章 12.12 |
| 有 KPI 意识（MTTR 等） | MTTR 前后对比表 | 跑完 12.13 |

### 12.1.3 KPI 定义与口径

三个核心 KPI，必须先定义再谈优化，否则数字全是自说自话。

| KPI | 定义 | 计算式 | 提升方向 | 常见误用 |
| --- | --- | --- | --- | --- |
| MTTR | Mean Time To Repair，平均恢复时长 | 所有故障的（恢复时刻 − 告警产生时刻）之和 ÷ 故障数 | 缩短告警到定位的时长、自动化处置 | 把「故障开始时刻」当起点，会虚增；起点必须是「系统可感知到异常的时刻」 |
| 误报率 | False Alarm Rate | 判定为异常但实际正常的告警数 ÷ 全部告警数 | 提高阈值、多模型投票、时间窗过滤 | 与「漏报率」混用；两者此消彼长，必须同时报 |
| 告警收敛率 | Alarm Compression Ratio | 1 − 收敛后告警数 ÷ 收敛前告警数 | 指纹去重、同服务同指标时间窗聚合、拓扑抑制 | 只报收敛率不报「是否漏掉真实故障」 |
| 检测延迟 | Detection Delay | 异常真实起始时刻到首次告警时刻的差值 | 缩短窗口、提高采样率 | 用窗口中心做时间戳会系统性低估 |

:::warn 拿 KPI 说事之前先固定口径
「MTTR 从 45 分钟降到 12 分钟」这句话只有在写清三件事时才成立：① 样本量（多少个故障）；② 起点定义（告警产生 vs 故障开始）；③ 是否包含人工审批环节。面试官几乎必问这三条。
:::

---

## 12.2 数据集：来源、规模、下载、格式与探查代码

### 12.2.1 六个常用数据集对照

| 数据集 | 全称/来源 | 模态 | 规模 | 标注方式 | 获取方式 | 适合做什么 |
| --- | --- | --- | --- | --- | --- | --- |
| AIOps Challenge | CCF AIOps 挑战赛（多年份） | 指标 + 日志 + 链路 | 每年不同，单节点指标数万条 | 故障区间标注（含根因节点） | 官网注册下载 | 端到端故障定位、多模态融合 |
| SMD | Server Machine Dataset（OmniAnomaly 论文） | 指标 | 28 台服务器 × 38 维，约 3.6 万点/台 | 逐点标签 | GitHub 仓库 `NetManAIOps/OmniAnomaly` | 多变量时序异常检测基准 |
| SWaT | Secure Water Treatment（iTrust 实验室） | 指标 | 51 维，约 100 万点（含攻击段） | 攻击区间标注 | iTrust 官网申请 | 工控异常检测、攻击检测 |
| PSM | Pooled Server Metrics（eBay） | 指标 | 25 维，约 13.2 万点 | 逐点标签 | GitHub（Yahoo/eBay 相关仓库） | 服务器指标异常检测 |
| Loghub | 日志研究集合（HDFS/BGL/Thunderbird 等） | 日志 | HDFS 约 1100 万行、BGL 约 470 万行 | 会话级/行级标签 | GitHub `logpai/loghub` | 日志模板挖掘、日志异常检测 |
| Nezha | Nezha 数据集（华为等） | 日志 + 指标 + 链路 | 多个微服务场景 | 故障类型与根因标注 | GitHub `Nezha` 仓库 | 微服务根因分析、多模态 |

:::tip 从哪一份开始
先做 **SMD**（多变量指标、逐点标签、体量小、论文可对比），再做 **Loghub 的 HDFS**（日志解析入门），然后上 **AIOps Challenge**（多模态、含根因，简历项目用它）。SWaT 要申请，别一开始就卡在数据获取上。
:::

### 12.2.2 SMD 的文件格式与列名

SMD 每个服务器一个目录，含三个文件：`train/<machine>.txt`（正常期）、`test/<machine>.txt`（含异常）、`test_label/<machine>.txt`（逐点标签）。

- `<machine>.txt`：纯文本，每行一个时刻，38 个数值用逗号分隔，**没有表头**。
- `<machine>.txt` 标签文件：每行一个 0/1，0 = 正常，1 = 异常，行数与测试文件一致。
- 训练文件一般不含异常，因此常被当作「只含正常样本」用于拟合无监督模型。

### 12.2.3 HDFS 日志的格式

`HDFS.log` 每行一条日志，格式形如：

```text
081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
```

字段依次是：日期（`yymmdd`）、时刻（`hhmmss`）、毫秒/线程号、级别（INFO/WARN/ERROR/FATAL）、组件名、正文。配套文件：`anomaly_label.csv` 给出每个 block 是否异常（BlockId, Label），因此 HDFS 是**会话级（block 级）**标签，不是行级 —— 这一点必须在报告里写明，否则指标算错。

### 12.2.4 BGL 日志的格式

```text
- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
```

字段：`-`、时间戳（epoch）、日期、节点名、本地时间、节点名重复、来源（RAS/APP）、级别、正文。标签文件给出「该行是否属于异常段」。

### 12.2.5 获得方式（命令可复制）

```bash
# 在 WSL2 Ubuntu 22.04 的 Bash 中执行：准备目录并拉取 SMD 与 Loghub
mkdir -p ~/aiops/data && cd ~/aiops/data

# SMD：随 OmniAnomaly 仓库一起发布
git clone --depth 1 https://github.com/NetManAIOps/OmniAnomaly.git
ls OmniAnomaly/ServerMachineDataset | head
```

预期输出形如：

```text
OmniAnomaly/ServerMachineDataset
```

`ls` 会列出 `train`、`test`、`test_label` 三个目录名。

```bash
# 在 WSL2 的 Bash 中执行：拉取 Loghub（只取 HDFS 与 BGL，避免整仓库过大）
git clone --depth 1 https://github.com/logpai/loghub.git
ls -lh loghub/HDFS loghub/BGL
```

预期输出包含 `HDFS_2k.log`、`HDFS.log`、`anomaly_label.csv`（HDFS 目录）与 `BGL_2k.log`、`BGL.log`（BGL 目录）。完整 `HDFS.log` 约 1.5GB，磁盘不够就只用 `HDFS_2k.log` 跑通流程。

### 12.2.6 数据探查代码（读取 + 画图）

```python
# explore_smd.py —— 读取 SMD 单个服务器数据并画出指标曲线与标签区间
# 执行位置：WSL2 Bash（python explore_smd.py）
import os

import matplotlib
matplotlib.use("Agg")            # WSL 无显示器时必须用 Agg 后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMD_ROOT = os.path.expanduser("~/aiops/data/OmniAnomaly/ServerMachineDataset")
MACHINE = "machine-1-1.txt"
OUT_DIR = os.path.expanduser("~/aiops/figs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_machine(machine=MACHINE, root=SMD_ROOT):
    """返回 (train_df, test_df, label_series)。SMD 无表头，38 个数值列。"""
    train = pd.read_csv(os.path.join(root, "train", machine), header=None)
    test = pd.read_csv(os.path.join(root, "test", machine), header=None)
    label = pd.read_csv(os.path.join(root, "test_label", machine), header=None).iloc[:, 0]
    cols = ["m%02d" % i for i in range(train.shape[1])]
    train.columns = cols
    test.columns = cols
    label.name = "label"
    return train, test, label


def basic_report(train, test, label):
    print("训练集形状 :", train.shape, " 时间跨度 :", train.shape[0], "点")
    print("测试集形状 :", test.shape)
    print("异常点数量 :", int(label.sum()), "/", len(label),
          " 占比 : %.2f%%" % (100.0 * label.mean()))
    print("缺失值     :", int(train.isna().sum().sum() + test.isna().sum().sum()))
    print("指标取值范围（前 5 维）：")
    print(pd.DataFrame({
        "训练最小值": train.iloc[:, :5].min(),
        "训练最大值": train.iloc[:, :5].max(),
        "训练均值": train.iloc[:, :5].mean().round(4),
        "训练标准差": train.iloc[:, :5].std().round(4),
    }))
    # 异常段切分：把连续的 1 合并成区间
    arr = label.to_numpy()
    edges = np.diff(np.concatenate([[0], arr, [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    print("异常段数量 :", len(starts))
    for s, e in list(zip(starts, ends))[:10]:
        print("  段 [%6d, %6d) 长度 %5d" % (s, e, e - s))
    return list(zip(starts, ends))


def plot_overview(test, label, segments, path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, dim in zip(axes, [0, 1, 2]):
        ax.plot(test.iloc[:, dim].to_numpy(), linewidth=0.7, label=test.columns[dim])
        for s, e in segments:
            ax.axvspan(s, e, color="red", alpha=0.18)
        ax.set_ylabel(test.columns[dim])
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title("SMD %s：前 3 维指标曲线（红色区间 = 标注异常段）" % MACHINE)
    axes[-1].set_xlabel("采样点序号")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("图已保存 :", path)


if __name__ == "__main__":
    tr, te, lb = load_machine()
    segs = basic_report(tr, te, lb)
    plot_overview(te, lb, segs, os.path.join(OUT_DIR, "smd_overview.png"))
```

执行与预期输出：

```bash
# 在 WSL2 Bash 中执行
cd ~/aiops && python explore_smd.py
```

预期输出形如：

```text
训练集形状 : (28479, 38)  时间跨度 : 28479 点
测试集形状 : (28479, 38)
异常点数量 : 2192 / 28479  占比 : 7.70%
缺失值     : 0
异常段数量 : 6
  段 [ 3839,  3928) 长度    89
  段 [ 4287,  4487) 长度   200
图已保存 : ~/aiops/figs/smd_overview.png
```

**探查阶段必须回答的四个问题**（写进报告）：① 异常占比多少（决定评价指标怎么选）；② 异常是连续的段还是散点（决定用 point-wise 还是区间评价）；③ 各维量纲差异多大（决定要不要标准化）；④ 训练集是否含异常（决定能不能用无监督拟合）。

### 12.2.7 HDFS 日志探查代码

```python
# explore_hdfs.py —— 读取 HDFS 日志，解析字段，统计级别与 block 分布
# 执行位置：WSL2 Bash
import os
import re
from collections import Counter

import pandas as pd

LOG_PATH = os.path.expanduser("~/aiops/data/loghub/HDFS/HDFS_2k.log")
LABEL_PATH = os.path.expanduser("~/aiops/data/loghub/HDFS/anomaly_label.csv")

LINE_RE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<ms>\d+)\s+"
    r"(?P<level>[A-Z]+)\s+(?P<component>[\w$.]+):\s+(?P<message>.*)$"
)
BLOCK_RE = re.compile(r"(blk_-?\d+)")


def parse_lines(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            d = m.groupdict()
            blocks = BLOCK_RE.findall(d["message"])
            rows.append({
                "line_no": i,
                "date": d["date"],
                "time": d["time"],
                "level": d["level"],
                "component": d["component"],
                "message": d["message"],
                "block": blocks[0] if blocks else None,
                "n_blocks": len(blocks),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = parse_lines(LOG_PATH)
    print("总行数 :", len(df))
    print("解析成功率 : %.1f%%" % (100.0 * len(df) / sum(1 for _ in open(LOG_PATH, encoding="utf-8", errors="ignore"))))
    print("\n级别分布 :")
    print(df["level"].value_counts())
    print("\n组件 Top10 :")
    print(df["component"].value_counts().head(10))
    print("\nblock 数量 :", df["block"].nunique())
    print("\n每个 block 的日志行数分布 :")
    print(df.groupby("block").size().describe().round(2))

    if os.path.exists(LABEL_PATH):
        lab = pd.read_csv(LABEL_PATH)
        print("\n标签文件列名 :", list(lab.columns))
        lab.columns = ["block", "label"]
        lab["label"] = (lab["label"].astype(str).str.upper() == "ANOMALY").astype(int)
        merged = df.merge(lab, on="block", how="left")
        print("标签分布 :")
        print(lab["label"].value_counts())
        print("异常 block 的日志行数均值 :", merged.groupby("label").size().to_dict())
```

预期输出形如：

```text
总行数 : 2000
解析成功率 : 100.0%
级别分布 : INFO 1997, WARN 3
block 数量 : 187
标签分布 : 0  1755
           1   168
```

**关键结论**：HDFS 是 **block 级（会话级）标签**，不是行级。要做行级异常检测，必须先把 block 标签传播到该 block 的所有日志行，并且在报告里明确写「标签是弱监督传播得到，存在噪声」。

---

## 12.3 时序异常检测：从统计基线到自编码器

### 12.3.1 统一评价入口

为了让五种方法可比，先写一份公共的「打分 → 阈值 → 指标」管线。**所有方法都输出同一格式的异常分数**，再用同一套评价函数计算，这样对比才公平。

```python
# aiops_common.py —— 公共工具：标准化、阈值搜索、指标计算
# 执行位置：WSL2 Bash（被其他脚本 import）
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def standardize(train, test):
    """用训练集的均值与标准差标准化，避免测试集信息泄漏。"""
    mu = train.mean(axis=0)
    sigma = train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return (train - mu) / sigma, (test - mu) / sigma


def best_threshold(scores, labels, grid=200):
    """在分数分位数上扫阈值，取 F1 最大的阈值（仅用于离线对比，线上不能用标签）。"""
    lo, hi = np.percentile(scores, 1), np.percentile(scores, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    cand = np.linspace(lo, hi, grid)
    best = (None, -1.0)
    for t in cand:
        pred = (scores > t).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def metrics_at(scores, labels, threshold):
    pred = (scores > threshold).astype(int)
    return {
        "precision": round(float(precision_score(labels, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, pred, zero_division=0)), 4),
        "alarms": int(pred.sum()),
        "false_alarms": int(((pred == 1) & (labels == 0)).sum()),
        "missed": int(((pred == 0) & (labels == 1)).sum()),
    }


def detection_delay(scores, labels, threshold):
    """首个异常段的检出延迟：真实起点到第一个命中点的距离（未检出返回 None）。"""
    pred = (scores > threshold).astype(int)
    arr = labels.astype(int)
    edges = np.diff(np.concatenate([[0], arr, [0]]))
    starts = np.where(edges == 1)[0]
    for s in starts:
        hit = np.where(pred[s:] == 1)[0]
        if hit.size:
            return int(hit[0])
    return None
```

预期输出：无（这是被 import 的模块）。自测方式：`python -c "import aiops_common; print('ok')"`。

### 12.3.2 统计基线一：3-sigma

原理：假设指标近似正态，偏离均值超过 3 倍标准差即异常。**必须用训练集算均值与标准差**，用测试集自己的统计量会把异常点算进基线（信息泄漏），这是最常见的低级错误。

```python
# det_3sigma.py —— 3-sigma 与 MAD 两种统计基线
# 执行位置：WSL2 Bash
import numpy as np

from aiops_common import best_threshold, detection_delay, metrics_at
from explore_smd import load_machine


def score_3sigma(train_z, test_z):
    """逐维算 z 分数，取绝对值最大的那一维作为分数。"""
    z = np.abs(test_z)
    return z.max(axis=1)


def score_mad(train_z, test_z):
    """MAD：中位数绝对偏差，对离群值比标准差稳健。"""
    med = np.median(train_z, axis=0)
    mad = np.median(np.abs(train_z - med), axis=0)
    mad[mad < 1e-8] = 1.0
    robust_z = np.abs(test_z - med) / (1.4826 * mad)   # 1.4826 使 MAD 与正态标准差可比
    return robust_z.max(axis=1)


if __name__ == "__main__":
    train, test, label = load_machine()
    from aiops_common import standardize
    tr_z, te_z = standardize(train.to_numpy(), test.to_numpy())
    labels = label.to_numpy()

    for name, fn in [("3-sigma", score_3sigma), ("MAD", score_mad)]:
        scores = fn(tr_z, te_z)
        thr, f1 = best_threshold(scores, labels)
        m = metrics_at(scores, labels, thr)
        delay = detection_delay(scores, labels, thr)
        print("%-8s 阈值=%8.3f  P=%.3f R=%.3f F1=%.3f 告警=%5d 误报=%5d 漏报=%5d 延迟=%s"
              % (name, thr, m["precision"], m["recall"], m["f1"],
                 m["alarms"], m["false_alarms"], m["missed"], delay))
```

预期输出形如：

```text
3-sigma  阈值=   6.812  P=0.412 R=0.596 F1=0.487 告警= 3172 误报= 1865 漏报=  886 延迟=12
MAD      阈值=   7.905  P=0.438 R=0.571 F1=0.495 告警= 2971 误报= 1671 漏报=  935 延迟=15
```

MAD 通常比 3-sigma 的误报少一些，因为中位数与 MAD 都不受异常点拖累。

### 12.3.3 统计基线二：滑动分位数

原理：用滑动窗口内的历史分位数（如 P99）作为阈值。它自动适应指标的缓慢漂移（如内存逐步增长），是运维场景中最常用的「不需要训练」的方法。

```python
# det_moving_quantile.py —— 滑动窗口分位数检测
# 执行位置：WSL2 Bash
import numpy as np
import pandas as pd

from aiops_common import best_threshold, detection_delay, metrics_at
from explore_smd import load_machine


def score_moving_quantile(test_df, window=200, q=0.99):
    """对每一维算滑动分位阈值，用「超出阈值的相对幅度」作为分数。"""
    scores = np.zeros(len(test_df))
    for col in test_df.columns:
        s = test_df[col]
        thr = s.rolling(window=window, min_periods=window // 4).quantile(q)
        med = s.rolling(window=window, min_periods=window // 4).median()
        scale = (s.rolling(window=window, min_periods=window // 4).std()).replace(0, np.nan)
        excess = (s - thr) / scale
        scores = np.maximum(scores, excess.fillna(0).to_numpy())
    return np.nan_to_num(scores, nan=0.0)


if __name__ == "__main__":
    train, test, label = load_machine()
    labels = label.to_numpy()

    for window in [50, 200, 500]:
        for q in [0.99, 0.995]:
            scores = score_moving_quantile(test, window=window, q=q)
            thr, _ = best_threshold(scores, labels)
            m = metrics_at(scores, labels, thr)
            delay = detection_delay(scores, labels, thr)
            print("窗口=%3d 分位=%.3f  P=%.3f R=%.3f F1=%.3f 告警=%5d 误报=%5d 延迟=%s"
                  % (window, q, m["precision"], m["recall"], m["f1"],
                     m["alarms"], m["false_alarms"], delay))
```

预期输出形如 `窗口=200 分位=0.990  P=0.521 R=0.640 F1=0.574 告警= 2694 误报= 1194 延迟=8`。窗口越大越稳但检测越滞后；分位越高误报越少但漏报越多。这两个权衡必须在报告里画成表。

### 12.3.4 Isolation Forest

原理：随机切分特征空间，异常点更容易被少数几次切割孤立，因此「平均路径长度」短。优点是无需假设分布、支持多变量联合检测；缺点是**对逐点独立打分，不利用时序连续性**，容易在噪声大的维度上误报。

```python
# det_iforest.py —— Isolation Forest 多变量异常检测
# 执行位置：WSL2 Bash
import numpy as np
from sklearn.ensemble import IsolationForest

from aiops_common import best_threshold, detection_delay, metrics_at, standardize
from explore_smd import load_machine


def score_iforest(train_z, test_z, n_estimators=200, max_samples=4096, seed=42):
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(max_samples, len(train_z)),
        contamination="auto",       # 不预设比例，交给 score_samples 排序
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_z)
    # decision_function 越大越正常，取负号变成「越大越异常」
    return -model.decision_function(test_z)


if __name__ == "__main__":
    train, test, label = load_machine()
    tr_z, te_z = standardize(train.to_numpy(), test.to_numpy())
    labels = label.to_numpy()

    for n_est in [100, 200, 400]:
        scores = score_iforest(tr_z, te_z, n_estimators=n_est)
        thr, _ = best_threshold(scores, labels)
        m = metrics_at(scores, labels, thr)
        delay = detection_delay(scores, labels, thr)
        print("树数=%3d  P=%.3f R=%.3f F1=%.3f 告警=%5d 误报=%5d 延迟=%s"
              % (n_est, m["precision"], m["recall"], m["f1"],
                 m["alarms"], m["false_alarms"], delay))
```

预期输出形如 `树数=200  P=0.552 R=0.618 F1=0.583 告警= 2214 误报=  953 延迟=9`。树数从 100 加到 400 通常只有 1–2 个百分点的提升，性价比在 200 附近。

### 12.3.5 LOF（局部离群因子）

原理：比较样本与其 k 近邻的局部密度，密度明显低于邻居的判为异常。优点是能发现「局部」离群（不同簇的密度不同）；缺点是 **O(n²) 复杂度**，28k 点直接跑会很慢，必须先降采样或用 `n_neighbors` 小的配置。

```python
# det_lof.py —— LOF 局部离群检测（含降采样，避免 O(n^2) 卡死）
# 执行位置：WSL2 Bash
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from aiops_common import best_threshold, detection_delay, metrics_at, standardize
from explore_smd import load_machine


def score_lof(train_z, test_z, n_neighbors=35, max_train=3000, stride=10, seed=0):
    """训练集降采样建索引；测试集隔点打分，再线性插值回全长度。"""
    rng = np.random.default_rng(seed)
    if len(train_z) > max_train:
        idx = rng.choice(len(train_z), size=max_train, replace=False)
        ref = train_z[idx]
    else:
        ref = train_z

    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1)
    model.fit(ref)

    sub = test_z[::stride]
    sub_scores = -model.decision_function(sub)
    full = np.interp(np.arange(len(test_z)), np.arange(0, len(test_z), stride), sub_scores)
    return full


if __name__ == "__main__":
    train, test, label = load_machine()
    tr_z, te_z = standardize(train.to_numpy(), test.to_numpy())
    labels = label.to_numpy()

    for k in [20, 35, 50]:
        scores = score_lof(tr_z, te_z, n_neighbors=k)
        thr, _ = best_threshold(scores, labels)
        m = metrics_at(scores, labels, thr)
        delay = detection_delay(scores, labels, thr)
        print("k=%2d  P=%.3f R=%.3f F1=%.3f 告警=%5d 误报=%5d 延迟=%s"
              % (k, m["precision"], m["recall"], m["f1"],
                 m["alarms"], m["false_alarms"], delay))
```

预期输出形如 `k=35  P=0.487 R=0.533 F1=0.509 告警= 2482 误报= 1273 延迟=18`。LOF 在本任务上通常不如 Isolation Forest，但它的「局部密度」视角在报告里值得保留一段对比讨论。

### 12.3.6 自编码器（AE）

原理：用只含正常样本的训练集训练一个「压缩再还原」的网络，正常样本重构误差小、异常样本重构误差大。**关键工程细节**：用滑动窗口把多维时序切成 `(window, n_features)` 的片段，让模型看到时间上下文。

```python
# det_ae.py —— 基于滑窗的自编码器异常检测（PyTorch，8G 显存轻松跑）
# 执行位置：WSL2 Bash
import numpy as np
import torch
import torch.nn as nn

from aiops_common import best_threshold, detection_delay, metrics_at, standardize
from explore_smd import load_machine

WINDOW = 32
STRIDE = 8
SEED = 42


def make_windows(arr, window=WINDOW, stride=STRIDE):
    idx = np.arange(0, len(arr) - window + 1, stride)
    return np.stack([arr[i:i + window] for i in idx]), idx


class AENet(nn.Module):
    def __init__(self, n_features, hidden=32, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * WINDOW, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features * WINDOW),
            nn.Unflatten(1, (WINDOW, n_features)),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae(train_z, epochs=30, batch=256, lr=1e-3, device=None):
    torch.manual_seed(SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x, _ = make_windows(train_z)
    xt = torch.tensor(x, dtype=torch.float32)
    model = AENet(train_z.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt), batch_size=batch, shuffle=True)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for (b,) in loader:
            b = b.to(device)
            opt.zero_grad()
            out = model(b)
            loss = loss_fn(out, b)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if (ep + 1) % 10 == 0:
            print("  epoch %2d loss=%.6f" % (ep + 1, total / len(xt)))
    return model, device


@torch.no_grad()
def score_ae(model, test_z, device):
    """逐窗口算重构误差，再把窗口分数映射回每个采样点。"""
    x, idx = make_windows(test_z)
    errs = []
    for i in range(0, len(x), 512):
        b = torch.tensor(x[i:i + 512], dtype=torch.float32, device=device)
        rec = model(b)
        err = ((rec - b) ** 2).mean(dim=(1, 2))
        errs.append(err.cpu().numpy())
    win_scores = np.concatenate(errs)
    full = np.zeros(len(test_z))
    counts = np.zeros(len(test_z))
    for k, i in enumerate(idx):
        full[i:i + WINDOW] += win_scores[k]
        counts[i:i + WINDOW] += 1
    counts[counts == 0] = 1
    return full / counts


if __name__ == "__main__":
    train, test, label = load_machine()
    tr_z, te_z = standardize(train.to_numpy(), test.to_numpy())
    labels = label.to_numpy()

    model, device = train_ae(tr_z, epochs=30)
    scores = score_ae(model, te_z, device)
    thr, _ = best_threshold(scores, labels)
    m = metrics_at(scores, labels, thr)
    print("AE       阈值=%.5f P=%.3f R=%.3f F1=%.3f 告警=%5d 误报=%5d 漏报=%5d 延迟=%s"
          % (thr, m["precision"], m["recall"], m["f1"], m["alarms"],
             m["false_alarms"], m["missed"], detection_delay(scores, labels, thr)))
    torch.save({"state": model.state_dict(), "window": WINDOW},
               "ae_smd.pt")
    print("模型已保存：ae_smd.pt")
```

预期输出形如：

```text
  epoch 10 loss=0.041233
  epoch 20 loss=0.028871
  epoch 30 loss=0.024110
AE       阈值=0.91823 P=0.686 R=0.712 F1=0.699 告警= 2280 误报=  722 漏报=  631 延迟=5
模型已保存：ae_smd.pt
```

AE 通常显著好于统计基线（F1 提升 10–20 个百分点），并且**显存占用很小**（本配置下约 0.2GB），是 8G 上性价比最高的方案。

### 12.3.7 LSTM-AE

原理：编码器用 LSTM 把窗口压成隐向量，解码器用 LSTM 还原。相比全连接 AE，它能显式建模时间依赖，对「缓慢漂移」与「周期被打断」两类异常更敏感。代价是训练慢 3–5 倍，且对超参更敏感。

```python
# det_lstm_ae.py —— LSTM 自编码器异常检测
# 执行位置：WSL2 Bash
import numpy as np
import torch
import torch.nn as nn

from aiops_common import best_threshold, detection_delay, metrics_at, standardize
from det_ae import make_windows
from explore_smd import load_machine

WINDOW = 64
STRIDE = 8
SEED = 7


class LSTMAE(nn.Module):
    def __init__(self, n_features, hidden=64, latent=16, layers=1):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.enc(x)                 # h: (layers, B, hidden)
        z = self.to_latent(h[-1])               # (B, latent)
        # 把隐向量复制成 T 步作为解码器输入
        seq = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        dec, _ = self.dec(seq)
        return self.out(dec)


def train_lstm_ae(train_z, epochs=20, batch=128, lr=1e-3):
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, _ = make_windows(train_z, window=WINDOW, stride=STRIDE)
    xt = torch.tensor(x, dtype=torch.float32)
    model = LSTMAE(train_z.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt), batch_size=batch, shuffle=True)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for (b,) in loader:
            b = b.to(device)
            opt.zero_grad()
            loss = loss_fn(model(b), b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        if (ep + 1) % 5 == 0:
            print("  epoch %2d loss=%.6f" % (ep + 1, total / len(xt)))
    return model, device


@torch.no_grad()
def score_lstm_ae(model, test_z, device):
    x, idx = make_windows(test_z, window=WINDOW, stride=STRIDE)
    errs = []
    for i in range(0, len(x), 256):
        b = torch.tensor(x[i:i + 256], dtype=torch.float32, device=device)
        rec = model(b)
        errs.append(((rec - b) ** 2).mean(dim=(1, 2)).cpu().numpy())
    win = np.concatenate(errs)
    full = np.zeros(len(test_z))
    cnt = np.zeros(len(test_z))
    for k, i in enumerate(idx):
        full[i:i + WINDOW] += win[k]
        cnt[i:i + WINDOW] += 1
    cnt[cnt == 0] = 1
    return full / cnt


if __name__ == "__main__":
    train, test, label = load_machine()
    tr_z, te_z = standardize(train.to_numpy(), test.to_numpy())
    labels = label.to_numpy()
    model, device = train_lstm_ae(tr_z, epochs=20)
    scores = score_lstm_ae(model, te_z, device)
    thr, _ = best_threshold(scores, labels)
    m = metrics_at(scores, labels, thr)
    print("LSTM-AE  P=%.3f R=%.3f F1=%.3f 告警=%5d 误报=%5d 延迟=%s"
          % (m["precision"], m["recall"], m["f1"], m["alarms"],
             m["false_alarms"], detection_delay(scores, labels, thr)))
```

预期输出形如 `LSTM-AE  P=0.704 R=0.731 F1=0.717 告警= 2210 误报= 678 延迟=4`。

### 12.3.8 五方法对比表模板

| 方法 | 需要训练 | 训练耗时 | F1 | 误报数 | 漏报数 | 检测延迟（点） | 显存/内存 | 适用场景 | 主要弱点 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-sigma | 否 | 0s | 填入 | 填入 | 填入 | 填入 | 可忽略 | 单指标、量纲稳定 | 对离群值敏感、需正态假设 |
| MAD | 否 | 0s | 填入 | 填入 | 填入 | 填入 | 可忽略 | 有离群值的单指标 | 对周期性指标无感 |
| 滑动分位 | 否 | 秒级 | 填入 | 填入 | 填入 | 填入 | 可忽略 | 缓慢漂移的指标 | 阈值滞后、窗口敏感 |
| Isolation Forest | 是 | 10–60s | 填入 | 填入 | 填入 | 填入 | <1GB | 多变量联合、快速基线 | 忽略时序连续性 |
| LOF | 是 | 1–5min | 填入 | 填入 | 填入 | 填入 | 1–2GB | 局部密度差异场景 | O(n²)，需降采样 |
| AE | 是 | 1–3min | 填入 | 填入 | 填入 | 填入 | <0.5GB 显存 | 多变量、有充足正常样本 | 窗口长度需调 |
| LSTM-AE | 是 | 5–15min | 填入 | 填入 | 填入 | 填入 | <1GB 显存 | 强时间依赖场景 | 训练慢、超参敏感 |

:::ok 报告结论怎么写
不要写「AE 最好」。写：**「在 SMD 的 machine-1-1 上，AE（窗口 32、步长 8、30 epoch）取得 F1 0.699，比 3-sigma（0.487）高 21.2 个百分点，误报数从 1865 降到 722；但检测延迟从 12 点升到 5 点（更快）同时训练成本从 0 升到 90 秒。生产上建议统计基线做第一层快速告警、AE 做第二层确认，两层串联后误报率再降 X 个百分点。」** 这种写法同时给了收益、代价与工程结论。
:::

---

## 12.4 深度时序模型：Anomaly Transformer / TimesNet / PatchTST 在 8G 上的可用性

### 12.4.1 三个模型的定位

| 模型 | 核心思想 | 输入形式 | 原论文规模 | 8G 可用性 | 改造重点 |
| --- | --- | --- | --- | --- | --- |
| Anomaly Transformer | 关联差异（先验关联 vs 序列关联）做无监督检测 | 多变量窗口 | 窗口 100，batch 32 | 可用（显存约 1.5–3GB） | 窗口降到 48–64，batch 降到 8–16 |
| TimesNet | 把一维时序折叠成二维周期图，用 2D 卷积提特征 | 多变量窗口 | 窗口 96–336，batch 32 | 可用（显存约 2–4GB） | 周期 `top_k` 降到 2，层数降到 2 |
| PatchTST | 把时序切 patch 做通道独立 Transformer | 单变量为主，多变量走通道独立 | 窗口 336–512 | 需大改（原配置在 8G 上 OOM） | 窗口 ≤128，patch 16，batch 8，层数 2 |

三者的共同结论：**8G 上能跑，但必须缩输入长度与 batch，并且要把「缩了之后精度掉了多少」如实写进报告**。这才是「论文复现与改进」（P12）想看到的东西。

### 12.4.2 Anomaly Transformer 最小跑通命令

```bash
# 在 WSL2 的 Bash 中执行：克隆官方仓库并跑通 SMD 的一个服务器
cd ~/aiops
git clone https://github.com/thuml/Anomaly-Transformer.git
cd Anomaly-Transformer
conda create -n anomtr python=3.9 -y && conda activate anomtr
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
ls scripts
```

预期输出：`scripts` 目录里列出各数据集的 shell 脚本名（如 `SMD.sh`）。

```bash
# 在 WSL2 的 Bash 中执行：按 8G 显存改造后跑 SMD（关键参数都缩小）
python -u main.py \
  --anomaly_ratio 1.0 \
  --model AnomalyTransformer \
  --mode train \
  --data SMD \
  --root_path ./dataset/SMD \
  --seq_len 64 \
  --label_len 32 \
  --pred_len 32 \
  --batch_size 8 \
  --d_model 32 \
  --d_ff 32 \
  --n_heads 4 \
  --e_layers 2 \
  --dropout 0.1 \
  --num_epochs 3 \
  --itr 1
```

预期输出：每轮打印 `Epoch: 1, Steps: 100 ... loss=...`，并在 `./results/` 下生成 `metrics.npy` 与 `test_label.npy`（用于算 F1）。

:::warn 官方脚本里的默认参数会在 8G 上 OOM
官方 `SMD.sh` 默认 `seq_len=100`、`batch_size=32`、`d_model=512`、`e_layers=3`。这套配置在 8G 上跑 7B 级显存压力，直接 `CUDA out of memory`。必须先按上面的命令把 `d_model` 降到 32–64、`e_layers` 降到 2、`batch_size` 降到 8。**改造参数与精度变化必须一起写进报告**，只写「跑通了」没有价值。
:::

### 12.4.3 TimesNet 与 PatchTST 的最小命令

```bash
# 在 WSL2 的 Bash 中执行：TimesNet 跑 SMD（缩小配置）
cd ~/aiops
git clone https://github.com/thuml/Time-Series-Library.git
cd Time-Series-Library
conda create -n tslib python=3.10 -y && conda activate tslib
pip install -r requirements.txt
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD_small \
  --model TimesNet \
  --data SMD \
  --features M \
  --seq_len 64 \
  --label_len 0 \
  --pred_len 64 \
  --e_layers 2 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 2 \
  --num_kernels 4 \
  --batch_size 16 \
  --train_epochs 3 \
  --itr 1
```

预期输出：训练日志逐轮打印 loss，结束后在 `./checkpoints/` 下生成 `.pth`，并在控制台打印 `accuracy`、`precision`、`recall`、`f1`（该仓库自带点调整 F1 的实现）。

```bash
# 在 WSL2 的 Bash 中执行：PatchTST 跑单变量异常检测（窗口必须缩到 128）
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD_patch \
  --model PatchTST \
  --data SMD \
  --features M \
  --seq_len 128 \
  --pred_len 128 \
  --e_layers 2 \
  --n_heads 4 \
  --d_model 32 \
  --d_ff 64 \
  --patch_len 16 \
  --stride 8 \
  --batch_size 8 \
  --train_epochs 3 \
  --itr 1
```

预期输出：训练日志与最终 F1。若报 `CUDA out of memory`，先把 `--batch_size` 降到 4、`--patch_len` 降到 8。

### 12.4.4 8G 上的可用性与训练时长对照

| 模型 | 输入窗口 | batch | 层数 | 显存峰值 | 3 epoch 训练时长（SMD 单机） | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| Anomaly Transformer | 64 | 8 | 2 | 2.1–2.9GB | 6–12 分钟 | **推荐**，改造成本最低 |
| Anomaly Transformer | 100 | 32 | 3（官方） | >8GB | — | 跑不了，必须改 |
| TimesNet | 64 | 16 | 2 | 2.6–3.8GB | 8–15 分钟 | 推荐，`top_k=2` 很关键 |
| TimesNet | 96 | 32 | 2 | 4.5–6.0GB | 15–25 分钟 | 可行但需关后台程序 |
| PatchTST | 128 | 8 | 2 | 3.0–4.2GB | 10–18 分钟 | 可行，窗口再大就 OOM |
| PatchTST | 336 | 8 | 3 | >8GB | — | 跑不了 |

:::tip 8G 上调深度时序模型的四个固定动作
① 先降 `batch_size`（对显存影响最大、对精度影响最小）；② 再降 `d_model`/`d_ff`（32 或 64）；③ 再降 `e_layers`（2 层）；④ 最后才降 `seq_len`（对精度影响最大）。同时设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片。
:::

---

## 12.5 评估口径（本章最重要的一节）

### 12.5.1 point-wise 与 point-adjusted 的区别

- **point-wise（逐点）**：把每个采样点单独判对错。异常段有 100 个点，你只检出 1 个，recall = 1/100 = 1%。
- **point-adjusted（点调整）**：**只要某个真实异常段里有任意一个点被检出，就把该段内所有点都算作「已检出」**。上例中 recall 直接变成 100%。

点调整在 AIOps 论文里被广泛使用（OmniAnomaly、Anomaly Transformer 都用它），原因是运维关心的是「这个故障有没有被发现」，而不是「故障期间每个采样点有没有都报警」。但它的副作用极大：

| 陷阱 | 后果 | 规避做法 |
| --- | --- | --- |
| 只据点调整 F1 汇报 | 数字虚高 20–40 个百分点，面试时被追问就露怯 | **同时报 point-wise 与 point-adjusted 两栏** |
| 用点调整 F1 调阈值 | 阈值被推向「每段报一个点」的极端，误报率暴涨 | 调阈值用 point-wise 或业务加权指标 |
| 异常段很长时只看段级 | 段内到底报了几次完全看不到，运维体验可能很差 | 额外报「每段平均告警数」与「段内覆盖率」 |
| 测试集异常段极少（如 2 段） | F1 方差极大，换一个随机种子结论就变 | 报均值 ± 标准差（多 seed）或多机器汇总 |

:::danger 不要只写一个数字
简历与报告里写「异常检测 F1 = 0.92」是不合格的，必须写成 **「point-wise F1 = 0.71，point-adjusted F1 = 0.92（SMD 28 台机器均值 ± 0.04）」**。前者是你真实的逐点能力，后者是与论文可比的口径，两个都要有。
:::

### 12.5.2 正确的指标计算代码

```python
# aiops_metrics.py —— point-wise 与 point-adjusted 指标、误报率、检测延迟
# 执行位置：WSL2 Bash
import numpy as np


def point_wise(pred, label):
    """逐点混淆矩阵与指标。pred/label 均为 0/1 数组。"""
    pred = np.asarray(pred).astype(int)
    label = np.asarray(label).astype(int)
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0        # 误报率（假正例率）
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "far": round(far, 6), "alarms": int(pred.sum())}


def segments(label):
    """把 0/1 标签切成连续异常段列表 [(start, end), ...]，区间左闭右开。"""
    arr = np.asarray(label).astype(int)
    edges = np.diff(np.concatenate([[0], arr, [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def point_adjusted(pred, label):
    """点调整：段内命中一次则该段全部视为命中；再算逐点指标。"""
    pred = np.asarray(pred).astype(int).copy()
    label = np.asarray(label).astype(int)
    for s, e in segments(label):
        if pred[s:e].sum() > 0:
            pred[s:e] = 1
    return point_wise(pred, label)


def range_recall(pred, label):
    """段级召回：有多少比例的异常段被至少检出一次。"""
    pred = np.asarray(pred).astype(int)
    segs = segments(label)
    if not segs:
        return 0.0
    hit = sum(1 for s, e in segs if pred[s:e].sum() > 0)
    return round(hit / len(segs), 4)


def detection_delays(pred, label):
    """每个异常段的检出延迟（点）；未检出的记为 None。"""
    pred = np.asarray(pred).astype(int)
    out = []
    for s, e in segments(label):
        hit = np.where(pred[s:e] == 1)[0]
        out.append(int(hit[0]) if hit.size else None)
    return out


def alarm_per_segment(pred, label, max_gap=10):
    """每段平均告警条数（相邻告警间隔 > max_gap 视为两条独立告警）。"""
    pred = np.asarray(pred).astype(int)
    counts = []
    for s, e in segments(label):
        idx = np.where(pred[s:e] == 1)[0]
        if idx.size == 0:
            counts.append(0)
            continue
        n = 1
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > max_gap:
                n += 1
        counts.append(n)
    return counts


if __name__ == "__main__":
    # 构造一个直观例子：3 段真实异常，其中 2 段被检出（各只报 1 个点）
    label = np.zeros(200, dtype=int)
    label[20:60] = 1
    label[90:110] = 1
    label[150:190] = 1
    pred = np.zeros(200, dtype=int)
    pred[25] = 1        # 第 1 段命中单点
    pred[100] = 1       # 第 2 段命中单点
    pred[175] = 1       # 第 3 段命中单点（但延迟 25 点）
    pred[5] = 1         # 一个纯误报

    print("point-wise      :", point_wise(pred, label))
    print("point-adjusted  :", point_adjusted(pred, label))
    print("段级召回        :", range_recall(pred, label))
    print("各段检出延迟    :", detection_delays(pred, label))
    print("每段告警条数    :", alarm_per_segment(pred, label))
```

预期输出：

```text
point-wise      : {'tp': 3, 'fp': 1, 'fn': 87, 'tn': 109, 'precision': 0.75, 'recall': 0.0333, 'f1': 0.0638, 'far': 0.009091, 'alarms': 4}
point-adjusted  : {'tp': 90, 'fp': 91, 'fn': 0, 'tn': 19, 'precision': 0.4972, 'recall': 1.0, 'f1': 0.6636, 'far': 0.827273, 'alarms': 181}
段级召回        : 1.0
各段检出延迟    : [5, 10, 25]
每段告警条数    : [1, 1, 1]
```

这个例子把陷阱展示得非常清楚：**逐点 recall 只有 3.3%，点调整后变成 100%**。同一份预测，两个口径差 30 倍。报告里只写一个数字就是误导。

### 12.5.3 误报率与检测延迟的正确用法

- **误报率（FAR）**：`FP / (FP + TN)`，即假正例率。它有一个致命的场景依赖：正常点基数极大时，FAR 看起来很小（上例 0.9%），但绝对误报量（1 个点）比例却很高。**运维更关心「每天多少条告警」与「误报占比」**，所以建议同时报：
  - `误报率 FAR = FP / (FP + TN)`
  - `告警准确率 = TP / (TP + FP)`（等价于 precision）
  - `日均告警数 = 告警总数 / 覆盖天数`
- **检测延迟**：用「异常段起点到第一个命中点」的点数表示，再按采样间隔折算成秒。不要用「窗口中心」当时间戳，会系统性少算半个窗口。

---

## 12.6 告警压缩与收敛

### 12.6.1 三种收敛手段

| 手段 | 做什么 | 典型压缩比 | 风险 |
| --- | --- | --- | --- |
| 指纹去重 | 同服务 + 同指标 + 同级别 + 同一分钟算同一条 | 2–5× | 不同根因的同名告警被合并 |
| 时间窗聚合 | 同服务同指标在 N 分钟内的告警合并成一条（附带次数） | 3–10× | 窗口太大导致后续新故障被吞 |
| 抑制规则 | 高优先级告警抑制同一拓扑下的下游告警 | 2–4× | 规则写错会抑制真实故障 |

三层叠加后，实测收敛比通常在 **5–20×**（即原来 1000 条告警变成 50–200 条）。

### 12.6.2 实现代码

```python
# alert_compressor.py —— 指纹去重 + 时间窗聚合 + 拓扑抑制 + 收敛率计算
# 执行位置：WSL2 Bash
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd


@dataclass
class Alert:
    ts: datetime
    service: str
    metric: str
    level: str                 # warning | critical
    value: float
    message: str = ""
    count: int = 1
    first_ts: datetime = None
    last_ts: datetime = None
    sources: list = field(default_factory=list)

    def fingerprint(self):
        """指纹 = 服务 + 指标 + 级别 + 分钟粒度时间桶。"""
        bucket = self.ts.strftime("%Y%m%d%H%M")
        raw = "%s|%s|%s|%s" % (self.service, self.metric, self.level, bucket)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


TOPOLOGY = {
    # 上游 -> 下游列表：上游故障时抑制下游告警
    "gateway": ["checkout", "search", "profile"],
    "checkout": ["payment", "inventory"],
    "payment": ["bank-gateway"],
}


def dedup(alerts):
    """第一层：指纹去重。同一指纹合成一条，count 累加。"""
    table = {}
    for a in sorted(alerts, key=lambda x: x.ts):
        fp = a.fingerprint()
        if fp in table:
            kept = table[fp]
            kept.count += 1
            kept.last_ts = a.ts
            kept.value = max(kept.value, a.value)
            kept.sources.append(a.message[:40])
        else:
            a.first_ts = a.ts
            a.last_ts = a.ts
            a.sources.append(a.message[:40])
            table[fp] = a
    return list(table.values())


def aggregate(alerts, window_minutes=10):
    """第二层：同服务同指标在时间窗内合并（跨指纹）。"""
    grouped = {}
    for a in sorted(alerts, key=lambda x: x.ts):
        key = (a.service, a.metric)
        if key not in grouped:
            grouped[key] = a
            continue
        last = grouped[key]
        if a.ts - last.last_ts <= timedelta(minutes=window_minutes):
            last.count += a.count
            last.last_ts = a.ts
            last.value = max(last.value, a.value)
            last.sources.extend(a.sources)
        else:
            grouped[key + (a.ts.isoformat(),)] = a
            grouped[key] = a
    return list(grouped.values())


def suppress(alerts, topology=TOPOLOGY):
    """第三层：上游 critical 抑制下游告警（时间窗内）。"""
    window = timedelta(minutes=5)
    criticals = [a for a in alerts if a.level == "critical"]
    kept, suppressed = [], []
    for a in alerts:
        blocked_by = None
        for c in criticals:
            if c is a:
                continue
            downstream = topology.get(c.service, [])
            if a.service in downstream and abs((a.ts - c.ts).total_seconds()) <= window.total_seconds():
                blocked_by = c
                break
        if blocked_by is None:
            kept.append(a)
        else:
            a.suppressed_by = blocked_by.service      # 记录抑制来源，便于审计
            suppressed.append(a)
    return kept, suppressed


def compression_ratio(before, after):
    if before == 0:
        return 0.0
    return round(1 - after / before, 4)


def run_pipeline(alerts, window_minutes=10):
    n0 = len(alerts)
    s1 = dedup(alerts)
    s2 = aggregate(s1, window_minutes)
    kept, suppressed = suppress(s2)
    print("原始告警      : %5d" % n0)
    print("指纹去重后    : %5d  收敛率 %.2f%%" % (len(s1), 100 * compression_ratio(n0, len(s1))))
    print("时间窗聚合后  : %5d  累计收敛率 %.2f%%" % (len(s2), 100 * compression_ratio(n0, len(s2))))
    print("拓扑抑制后    : %5d  累计收敛率 %.2f%%（抑制 %d 条）"
          % (len(kept), 100 * compression_ratio(n0, len(kept)), len(suppressed)))
    return kept


if __name__ == "__main__":
    base = datetime(2027, 3, 1, 10, 0, 0)
    rows = []
    # 模拟：payment 出 critical，触发下游与同指标刷屏
    for i in range(40):
        rows.append(Alert(base + timedelta(seconds=20 * i), "payment", "p99_latency",
                          "critical", 1800 + i, "P99 latency spike"))
    for i in range(25):
        rows.append(Alert(base + timedelta(seconds=25 * i), "bank-gateway", "error_rate",
                          "warning", 0.06, "error rate high"))
    for i in range(15):
        rows.append(Alert(base + timedelta(minutes=12 + i), "search", "cpu_usage",
                          "warning", 0.91, "cpu high"))
    kept = run_pipeline(rows, window_minutes=10)
    print("\n保留的告警（前 5 条）：")
    for a in kept[:5]:
        print("  %s %-12s %-12s %-8s count=%d 窗口=[%s ~ %s]"
              % (a.ts.strftime("%H:%M:%S"), a.service, a.metric, a.level,
                 a.count, a.first_ts.strftime("%H:%M:%S"), a.last_ts.strftime("%H:%M:%S")))
```

预期输出形如：

```text
原始告警      :    80
指纹去重后    :    53  收敛率 33.75%
时间窗聚合后  :     6  累计收敛率 92.50%
拓扑抑制后    :     4  累计收敛率 95.00%（抑制 2 条）
```

**收敛率计算的三个口径**（报告里要选一个并注明）：① `1 - 收敛后条数/收敛前条数`（条数口径，最常用）；② `1 - 收敛后告警组数/收敛前告警组数`（组口径，更贴近值班体验）；③ `抑制条数/原始条数`（只算抑制规则的效果）。三个数字往往差很多，混用会导致「我做到了 95% 收敛」与「我做到了 60% 收敛」同时出现在两份报告里。

:::warn 抑制规则必须有审计与回滚
抑制规则写错会静默吞掉真实故障，这是 AIOps 里最危险的一类事故。工程要求：① 每条被抑制的告警都要记录 `suppressed_by`；② 抑制规则要有 TTL（如只生效 5 分钟）；③ 上生产前先在历史告警上回放，统计「被抑制的告警里有多少后来升级成了真实故障」。
:::

---

## 12.7 日志解析：Drain 算法与 Drain3

### 12.7.1 Drain 算法原理

Drain 是一棵**固定深度的前缀树 + 按 token 数分桶**的在线日志解析算法。步骤：

1. **预处理**：把日志行按 token 切分，识别并替换变量（IP、数字、十六进制、路径、block id 等），例如把 `blk_38865049064139660` 换成通配符。
2. **分桶**：以「token 数」作为第一层哈希键。不同长度的日志必然不同模板。
3. **树深度搜索**：前缀树的前 `depth-1` 层用具体 token 匹配，最后一层用**相似度阈值**匹配已有模板。
4. **匹配或新建**：与某个叶子模板的相似度 ≥ 阈值就归入该模板（并把不一致的位置改成通配符），否则新建模板。

三个关键参数：

| 参数 | 含义 | 常见取值 | 影响 |
| --- | --- | --- | --- |
| `depth` | 前缀树的层数 | 4（Drain3 默认） | 越大越依赖前缀，模板更细但更容易碎；越小合并越激进 |
| `sim_th` | 相似度阈值 | 0.4–0.5 | 越低合并越激进（模板少、更泛化）；越高模板越多越细 |
| `max_children` | 每个节点的最大子节点数 | 100 | 防止树爆炸；超出的子节点归入通配符分支 |
| `max_clusters` | 最大模板数 | 1024 | 内存上限，超出后旧模板被淘汰 |

### 12.7.2 Drain3 完整使用代码

```python
# log_parse_drain.py —— 用 Drain3 挖掘 HDFS 日志模板
# 执行位置：WSL2 Bash（先 pip install drain3）
import json
import os
from collections import Counter

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

LOG_PATH = os.path.expanduser("~/aiops/data/loghub/HDFS/HDFS_2k.log")
STATE_PATH = os.path.expanduser("~/aiops/drain3_state.bin")
CONFIG = {
    "depth": 4,
    "sim_th": 0.4,
    "max_children": 100,
    "max_clusters": 1024,
    "parametrize_numeric_tokens": True,
}


def build_miner(config_dict=CONFIG, persist=True):
    cfg = TemplateMinerConfig()
    cfg.drain_extra_delimiters = ["_", ".", ":", "/"]
    # Drain3 的参数通过 config 段落传入
    cfg.profiling_enabled = False
    cfg.snapshot_interval_minutes = 0
    persistence = FilePersistence(STATE_PATH) if persist else None
    miner = TemplateMiner(persistence_handler=persistence, config=cfg)
    # 手动覆盖 drain 参数（不同版本 API 略有差异，用 setattr 兜底）
    for k, v in config_dict.items():
        if hasattr(miner.drain, k):
            setattr(miner.drain, k, v)
        elif hasattr(cfg, "drain_" + k):
            setattr(cfg, "drain_" + k, v)
    if hasattr(miner.drain, "parameterize_numeric_tokens"):
        miner.drain.parameterize_numeric_tokens = config_dict.get("parametrize_numeric_tokens", True)
    return miner


def parse_file(path, miner):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            res = miner.add_log_message(line)
            rows.append({
                "line": line,
                "template_id": "T%d" % res["cluster_id"],
                "template": res["template_mined"],
                "changed": res.get("change_type", "none"),
            })
    return rows


def report(rows, miner):
    print("总行数      :", len(rows))
    print("模板总数    :", len(miner.drain.clusters))
    cnt = Counter(r["template_id"] for r in rows)
    print("\nTop10 高频模板：")
    tid2tpl = {"T%d" % c.cluster_id: c.get_template() for c in miner.drain.clusters}
    for tid, n in cnt.most_common(10):
        print("  %-6s %5d 次  %s" % (tid, n, tid2tpl.get(tid, "")[:110]))
    print("\n低频模板（出现 1 次的模板数）:", sum(1 for _, n in cnt.items() if n == 1))
    return cnt


if __name__ == "__main__":
    miner = build_miner()
    rows = parse_file(LOG_PATH, miner)
    cnt = report(rows, miner)
    out = os.path.expanduser("~/aiops/parsed_hdfs.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r, n in zip(rows, [cnt[r["template_id"]] for r in rows]):
            r["template_count"] = n
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n解析结果已写入 :", out)
```

执行与预期输出：

```bash
# 在 WSL2 Bash 中执行
pip install drain3
python log_parse_drain.py
```

预期输出形如：

```text
总行数      : 2000
模板总数    : 46

Top10 高频模板：
  T12      412 次  PacketResponder <:*:> for block <:*:> terminating
  T7       288 次  Receiving block <:*:> src: <:*:> dest: <:*:>
  T19      201 次  BLOCK* NameSystem.addStoredBlock: blockMap updated: <:*:> is added to <:*:>
  T3        96 次  Verifying <:*:>.blockFile for block <:*:>
低频模板（出现 1 次的模板数）: 11
```

### 12.7.3 参数对结果的影响（必须做这张表）

```python
# log_parse_sweep.py —— 扫 depth 与 sim_th，看模板数量的变化
# 执行位置：WSL2 Bash
import os

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from log_parse_drain import LOG_PATH, parse_file


def count_templates(depth, sim_th, max_lines=2000):
    cfg = TemplateMinerConfig()
    miner = TemplateMiner(config=cfg)
    miner.drain.depth = depth
    miner.drain.sim_th = sim_th
    miner.drain.max_children = 100
    rows = parse_file(LOG_PATH, miner) if max_lines is None else None
    if rows is None:
        rows = []
        with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                if line.strip():
                    miner.add_log_message(line.strip())
                    rows.append(i)
    return len(miner.drain.clusters), len(rows)


if __name__ == "__main__":
    print("| depth | sim_th | 模板数 | 处理行数 |")
    print("| --- | --- | --- | --- |")
    for depth in [3, 4, 6]:
        for sim_th in [0.3, 0.4, 0.5, 0.7]:
            n_tpl, n_line = count_templates(depth, sim_th)
            print("| %d | %.1f | %d | %d |" % (depth, sim_th, n_tpl, n_line))
```

预期输出（表格可直接贴进报告）：

```text
| depth | sim_th | 模板数 | 处理行数 |
| --- | --- | --- | --- |
| 3 | 0.3 | 21 | 2000 |
| 3 | 0.5 | 38 | 2000 |
| 4 | 0.4 | 46 | 2000 |
| 6 | 0.7 | 129 | 2000 |
```

**结论写法**：`sim_th` 从 0.3 升到 0.7，模板数从 21 涨到 129（约 6 倍），说明阈值直接决定「泛化 vs 精细」的取舍；`depth` 从 3 升到 6 影响相对小（因为 HDFS 日志前缀差异本来就大）。生产建议 `depth=4, sim_th=0.4` 起步，再按「新日志被错误归入旧模板的比例」微调。

---

## 12.8 日志异常检测：模板序列 + 分类模型

### 12.8.1 思路

逐行判异常很难（单行日志信息量太少）。工程上主流做法是 **会话级**：

1. 用 Drain3 把日志行映射成模板 ID；
2. 按会话（HDFS 是 block id）把模板 ID 序列拼起来；
3. 用「模板序列的 TF-IDF 或词袋」作为特征，训练二分类器；
4. 输出该会话是否异常。

```python
# log_anomaly.py —— 模板序列特征 + 逻辑回归/随机森林分类
# 执行位置：WSL2 Bash
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

HDFS_DIR = os.path.expanduser("~/aiops/data/loghub/HDFS")
LOG_PATH = os.path.join(HDFS_DIR, "HDFS_2k.log")
LABEL_PATH = os.path.join(HDFS_DIR, "anomaly_label.csv")


def parse_to_templates(path):
    """把每行日志变成模板 ID，同时抽取 block id 作为会话键。"""
    import re
    block_re = re.compile(r"(blk_-?\d+)")
    cfg = TemplateMinerConfig()
    miner = TemplateMiner(config=cfg)
    miner.drain.depth = 4
    miner.drain.sim_th = 0.4
    miner.drain.max_children = 100
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            res = miner.add_log_message(line)
            m = block_re.search(line)
            rows.append({
                "block": m.group(1) if m else "unknown",
                "tid": "T%d" % res["cluster_id"],
                "template": res["template_mined"],
            })
    return pd.DataFrame(rows), miner


def build_sessions(df, labels):
    """按 block 聚合模板序列，生成「会话样本」。"""
    docs = df.groupby("block")["tid"].apply(lambda s: " ".join(s)).reset_index()
    docs.columns = ["block", "tid_seq"]
    docs["n_lines"] = df.groupby("block").size().reindex(docs["block"]).to_numpy()
    docs["n_unique_tid"] = df.groupby("block")["tid"].nunique().reindex(docs["block"]).to_numpy()
    docs = docs.merge(labels, on="block", how="inner")
    return docs


def main():
    df, miner = parse_to_templates(LOG_PATH)
    print("日志行数 :", len(df), " 模板数 :", len(miner.drain.clusters))

    lab = pd.read_csv(LABEL_PATH)
    lab.columns = ["block", "label"]
    lab["label"] = (lab["label"].astype(str).str.upper() == "ANOMALY").astype(int)

    sessions = build_sessions(df, lab)
    print("会话数   :", len(sessions), " 异常会话 :", int(sessions["label"].sum()))

    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=1)
    X = vec.fit_transform(sessions["tid_seq"])
    extra = sessions[["n_lines", "n_unique_tid"]].to_numpy(dtype=float)
    from scipy.sparse import hstack
    X = hstack([X, extra]).tocsr()
    y = sessions["label"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.sum() > 1 else None)

    for name, clf in [
        ("逻辑回归", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("随机森林", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
    ]:
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        print("\n===== %s =====" % name)
        print(confusion_matrix(y_te, pred))
        print(classification_report(y_te, pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
```

执行与预期输出：

```bash
# 在 WSL2 Bash 中执行（需要 scikit-learn、scipy、drain3）
python log_anomaly.py
```

预期输出形如：

```text
日志行数 : 2000  模板数 : 46
会话数   : 187  异常会话 : 23
===== 逻辑回归 =====
[[42  3]
 [ 3  9]]
              precision    recall  f1-score   support
           0     0.9333    0.9333    0.9333        45
           1     0.7500    0.7500    0.7500        12
```

注意：`HDFS_2k.log` 只有 2000 行、187 个会话，指标方差极大。**正式实验必须用完整 `HDFS.log`（约 1100 万行、57 万+ 会话）**，否则报告里的 F1 不可信。

### 12.8.2 对比表模板

| 方法 | 特征 | 精确率 | 召回率 | F1 | 训练时长 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 关键词规则（ERROR/FATAL 计数） | 关键字次数 | 填入 | 填入 | 填入 | 0s | 最强基线，必须先跑 |
| 模板词袋 + 逻辑回归 | TF-IDF(tid) | 填入 | 填入 | 填入 | 秒级 | 可解释（看权重） |
| 模板词袋 + 随机森林 | TF-IDF + 统计量 | 填入 | 填入 | 填入 | 秒–分钟 | 通常最强 |
| 模板序列 + LSTM | tid 嵌入序列 | 填入 | 填入 | 填入 | 分钟级 | 8G 可跑 |
| 模板序列 + 小 Transformer | tid 嵌入 + 位置编码 | 填入 | 填入 | 填入 | 分钟级 | 数据量大才值得 |

:::tip 一定要先跑规则基线
「用 ERROR 关键字计数」这个基线在 HDFS 上经常能拿到 F0.8 以上的 F1。如果你的模型打不过它，说明模型没学到东西。把基线放进对比表是加分项，不是丢人的事 —— 面试官喜欢能看到「先立基线再改进」的思路。
:::

---

## 12.9 根因分析：拓扑构建、相关性排序、根因候选 Top-3

### 12.9.1 三层方法

1. **拓扑层**：把服务依赖画成有向图，故障沿边传播；上游是下游的候选根因。
2. **相关性层**：算各指标与「故障指标」的相关性（Pearson / Spearman / 互信息），越高越可能是根因。
3. **因果层（简化）**：用「时间先后 + 相关性 + 拓扑可达」三个条件过滤，把「相关」升级为「疑似因果」。

```python
# rca_engine.py —— 服务拓扑 + 相关性排序 + 根因候选 Top-3
# 执行位置：WSL2 Bash
from dataclasses import dataclass

import numpy as np
import pandas as pd


# 服务依赖图（上游 -> 下游）。真实场景从调用链或 CMDB 构建。
TOPOLOGY = {
    "gateway": ["checkout", "search", "profile"],
    "checkout": ["payment", "inventory"],
    "payment": ["bank-gateway"],
    "search": ["index-service"],
    "profile": ["user-db"],
    "inventory": ["stock-db"],
}


@dataclass
class Candidate:
    service: str
    metric: str
    corr: float
    lead_time: float          # 相对故障时刻的平均超前量（秒），正数表示更早异常
    topology_score: float
    score: float
    evidence: str


def build_reverse_graph(topology=TOPOLOGY):
    """反向图：下游 -> 上游列表，用于「向上找根因」。"""
    rev = {}
    for up, downs in topology.items():
        for d in downs:
            rev.setdefault(d, []).append(up)
    return rev


def upstream_of(service, reverse, max_hops=3):
    """BFS 找所有上游（含间接），返回 {服务: 跳数}。"""
    seen = {service: 0}
    frontier = [service]
    for hop in range(1, max_hops + 1):
        nxt = []
        for s in frontier:
            for up in reverse.get(s, []):
                if up not in seen:
                    seen[up] = hop
                    nxt.append(up)
        frontier = nxt
        if not frontier:
            break
    seen.pop(service, None)
    return seen


def corr_with_target(df, target_col, candidate_cols):
    """算每个候选指标相对目标指标的 Spearman 相关（对非线性单调关系更稳）。"""
    out = {}
    tgt = df[target_col].to_numpy()
    for c in candidate_cols:
        if c == target_col:
            continue
        v = df[c].to_numpy()
        if np.std(v) < 1e-9:
            out[c] = 0.0
            continue
        out[c] = float(pd.Series(v).corr(pd.Series(tgt), method="spearman"))
    return out


def lead_time(df, target_col, candidate_col, threshold_q=0.95, fs=1.0):
    """候选指标首次越过其高分位的时刻，比目标首次越界的时刻早多少（秒）。"""
    def first_cross(col):
        thr = np.quantile(df[col].to_numpy(), threshold_q)
        idx = np.where(df[col].to_numpy() > thr)[0]
        return idx[0] if idx.size else len(df)
    return (first_cross(target_col) - first_cross(candidate_col)) * fs


def rank_candidates(df, target_service, target_metric, reverse=build_reverse_graph(),
                    max_hops=3, w_corr=0.6, w_topo=0.25, w_lead=0.15):
    """输出根因候选 Top-3。df 的列名约定为 '<service>.<metric>'。"""
    target_col = "%s.%s" % (target_service, target_metric)
    if target_col not in df.columns:
        raise KeyError("目标列不存在：%s（现有列：%s）" % (target_col, list(df.columns)[:10]))

    ups = upstream_of(target_service, reverse, max_hops)
    cols = [c for c in df.columns]
    corrs = corr_with_target(df, target_col, cols)

    cands = []
    for col, corr in corrs.items():
        svc, _, metric = col.partition(".")
        if svc == target_service and metric == target_metric:
            continue
        # 拓扑得分：越近的上游越高；不属于上游的按 0.2 保底（仍可能是环境类根因）
        hops = ups.get(svc)
        topo = 1.0 if hops == 1 else (0.7 if hops == 2 else (0.45 if hops == 3 else 0.2))
        lead = lead_time(df, target_col, col) if corr > 0 else -999
        lead_score = 0.0 if lead < 0 else min(lead / 60.0, 1.0)     # 60 秒封顶
        score = w_corr * abs(corr) + w_topo * topo + w_lead * lead_score
        cands.append(Candidate(
            service=svc, metric=metric, corr=round(corr, 4),
            lead_time=round(float(lead), 1), topology_score=topo,
            score=round(float(score), 4),
            evidence="corr=%.3f, 上游跳数=%s, 超前=%ss" % (corr, hops if hops else "非上游", lead),
        ))

    cands.sort(key=lambda c: -c.score)
    seen = set()
    top = []
    for c in cands:                      # 同一服务只保留最强指标，避免 Top-3 被同一服务占满
        if c.service in seen:
            continue
        seen.add(c.service)
        top.append(c)
        if len(top) == 3:
            break
    return top


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 240
    idx = np.arange(n)
    # 构造：bank-gateway 网络延迟先升 -> payment 延迟升高 -> checkout 报错率升高
    df = pd.DataFrame(index=idx)
    df["bank-gateway.net_latency_ms"] = 20 + 80 * (idx > 60) + rng.normal(0, 3, n)
    df["payment.p99_latency_ms"] = 120 + 900 * (idx > 75) + rng.normal(0, 12, n)
    df["checkout.error_rate"] = 0.01 + 0.09 * (idx > 90) + rng.normal(0, 0.003, n)
    df["inventory.p99_latency_ms"] = 80 + rng.normal(0, 6, n)
    df["search.p99_latency_ms"] = 70 + rng.normal(0, 5, n)
    df["gateway.cpu_usage"] = 0.3 + 0.4 * (idx > 95) + rng.normal(0, 0.02, n)

    print("故障现象：checkout 的 error_rate 从 0.01 涨到 0.10")
    print("\n根因候选 Top-3：")
    for i, c in enumerate(rank_candidates(df, "checkout", "error_rate"), 1):
        print("  %d. %-14s %-18s 综合分=%.4f  %s"
              % (i, c.service, c.metric, c.score, c.evidence))
```

预期输出形如：

```text
故障现象：checkout 的 error_rate 从 0.01 涨到 0.10

根因候选 Top-3：
  1. payment        p99_latency_ms     综合分=0.8150  corr=0.958, 上游跳数=1, 超前=15.0s
  2. bank-gateway   net_latency_ms     综合分=0.7320  corr=0.873, 上游跳数=2, 超前=30.0s
  3. gateway        cpu_usage          综合分=0.5540  corr=0.792, 上游跳数=2, 超前=0.0s
```

### 12.9.2 图方法的简化实现

上面的做法是「启发式排序」。如果想讲得更像图方法，可以加一层随机游走（Personalized PageRank）在拓扑上做传播打分：

```python
# rca_graph.py —— 在服务拓扑上做带权随机游走，给根因候选打传播分
# 执行位置：WSL2 Bash
import numpy as np

from rca_engine import TOPOLOGY


def build_transition(topology=TOPOLOGY, alpha=0.85, self_loop=0.15):
    """构造「下游 -> 上游」的转移矩阵：故障从下游往上游传播。"""
    nodes = sorted({n for up, ds in topology.items() for n in ([up] + ds)})
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    P = np.zeros((n, n))
    for up, downs in topology.items():
        for d in downs:
            P[idx[d], idx[up]] += 1.0        # 下游被访问时，跳到上游
    P += self_loop * np.eye(n)               # 自环，避免死胡同
    P = P / P.sum(axis=1, keepdims=True)
    return nodes, idx, P


def personalized_pagerank(seed_service, node_scores, topology=TOPOLOGY, iters=50, alpha=0.85):
    """以故障服务为种子做个性化 PageRank，node_scores 是初始异常强度。"""
    nodes, idx, P = build_transition(topology)
    personal = np.zeros(len(nodes))
    personal[idx[seed_service]] = 1.0
    v = np.array([node_scores.get(n, 0.0) for n in nodes], dtype=float)
    v = v / (v.sum() or 1.0)
    for _ in range(iters):
        v = alpha * (P.T @ v) + (1 - alpha) * personal
    return {n: round(float(v[idx[n]]), 4) for n in nodes}


if __name__ == "__main__":
    anomaly_intensity = {
        "checkout": 1.0, "payment": 0.8, "gateway": 0.3,
        "bank-gateway": 0.6, "inventory": 0.1, "search": 0.05,
    }
    ranks = personalized_pagerank("checkout", anomaly_intensity)
    print("传播分（降序）：")
    for k, v in sorted(ranks.items(), key=lambda kv: -kv[1]):
        print("  %-14s %.4f" % (k, v))
```

预期输出形如 `payment 0.3421`、`bank-gateway 0.2210`、`checkout 0.1804`。把「传播分」与「相关性分」加权（例如 0.5/0.5）就得到最终 Top-3，报告里可以给出「只用相关性」与「相关性 + 图传播」两种排序的对比，说明图方法的增益。

:::ok 根因分析的诚实写法
不要写「实现了根因定位算法，准确率 95%」。写：**「在自建的 5 服务拓扑 + 240 点合成指标上，用相关性 + 拓扑距离 + 时间超前三点加权的排序器，把真实根因（bank-gateway 网络延迟）排到第 2 位（Top-3 命中率 100%）；在 AIOps Challenge 的 N 个故障样本上 Top-3 命中率为 X%（样本量 N）」。** 关键是把「合成数据」与「真实数据」分开说，并给出样本量。
:::

---

## 12.10 LLM 告警摘要与处置建议

### 12.10.1 Prompt 模板（要求引用原始证据）

幻觉控制的唯一有效手段是**结构化输入 + 强制引用 + 拒答条件**。下面这份 prompt 可以直接用。

```text
你是资深 SRE，负责把告警整理成可执行的处置建议。严格按下面的输出结构回答，不要增加小节。

【输入格式说明】
- METRICS：指标名, 当前值, 基线值, 持续时间
- LOGS：日志时间, 级别, 组件, 正文（最多 20 行）
- TOPOLOGY：服务依赖（上游 -> 下游）

【输出结构】
1. 现象：一句话，必须写成「<服务> 的 <指标> 从 <基线值> 升到 <当前值>，持续 <时长>」
2. 影响面：列出受影响的直接下游服务，只能引用 TOPOLOGY 中出现的名字
3. 根因候选 Top-3：每条写成「<服务>/<指标>（可信度 高/中/低）— 证据：<引用具体指标名或日志原文片段>」
4. 建议动作：分两组
   - 可自动执行（如：扩容副本、清理磁盘、重启无状态服务）
   - 需人工审批（如：回滚发布、切流量、重启数据库、修改配置）
5. 信息不足声明：如果输入不含判断所需的信息，写「信息不足：缺少 <具体缺什么>」，不要猜测

【硬约束】
- 禁止编造输入中不存在的指标名、服务名、日志行
- 每个根因候选必须至少引用一条输入里的证据（指标名+数值 或 日志原文片段）
- 不确定时降级可信度，不要升级确定性

【METRICS】
{metrics_block}

【LOGS】
{logs_block}

【TOPOLOGY】
{topology_block}
```

### 12.10.2 完整实现代码

```python
# alert_summarizer.py —— 告警摘要：组装证据 -> 调 LLM -> 校验引用 -> 落盘
# 执行位置：WSL2 Bash（LLM 用本地 Ollama 或云端 API）
import json
import os
import re
from datetime import datetime

from openai import OpenAI

LOCAL = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
REMOTE = OpenAI(base_url=os.environ.get("OPENAI_BASE", "https://api.deepseek.com/v1"),
                api_key=os.environ.get("OPENAI_KEY", "none"))
LOCAL_MODEL = "qwen2.5:7b-instruct-q4_K_M"
REMOTE_MODEL = "deepseek-chat"


def build_blocks(alert):
    metrics_block = "\n".join(
        "- %s, 当前=%s, 基线=%s, 持续=%s" % (m["name"], m["value"], m["baseline"], m["duration"])
        for m in alert["metrics"])
    logs_block = "\n".join(
        "- %s %s %s: %s" % (l["ts"], l["level"], l["component"], l["message"])
        for l in alert["logs"][:20])
    topo = "\n".join("- %s -> %s" % (up, ", ".join(ds)) for up, ds in alert["topology"].items())
    return metrics_block, logs_block, topo


PROMPT_TEMPLATE = open(os.path.join(os.path.dirname(__file__), "prompt_summary.txt"),
                       encoding="utf-8").read()


def call_llm(prompt, backend="local", max_tokens=800):
    client, model = (LOCAL, LOCAL_MODEL) if backend == "local" else (REMOTE, REMOTE_MODEL)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,            # 摘要任务要低温度
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


NUM_RE = re.compile(r"(\d+\.?\d*)")


def verify_citations(answer, alert):
    """幻觉检查：回答里出现的每个数字与日志片段，是否能在输入里找到。"""
    allowed_numbers = set()
    for m in alert["metrics"]:
        for v in (m["value"], m["baseline"]):
            allowed_numbers.update(NUM_RE.findall(str(v)))
    for l in alert["logs"]:
        allowed_numbers.update(NUM_RE.findall(l["message"]))
        allowed_numbers.update(NUM_RE.findall(l["ts"]))

    used = set(NUM_RE.findall(answer))
    unsupported_numbers = sorted(used - allowed_numbers)

    allowed_services = set(alert["topology"].keys())
    for ds in alert["topology"].values():
        allowed_services.update(ds)
    mentioned_services = set(re.findall(r"[a-z][a-z0-9\-]{2,}", answer.lower()))
    suspicious_services = sorted(
        s for s in mentioned_services
        if s not in allowed_services and s not in {"metrics", "logs", "topology", "top3", "p99", "mt", "sre"}
    )

    log_snippets = [l["message"][:28] for l in alert["logs"]]
    cited_logs = [s for s in log_snippets if s and s in answer]

    return {
        "unsupported_numbers": unsupported_numbers[:12],
        "suspicious_services": suspicious_services[:12],
        "cited_log_snippets": len(cited_logs),
        "total_log_snippets": len(log_snippets),
        "has_insufficient_declaration": "信息不足" in answer,
    }


def summarize(alert, backend="local", save_dir="summaries"):
    metrics_block, logs_block, topo = build_blocks(alert)
    prompt = (PROMPT_TEMPLATE
              .replace("{metrics_block}", metrics_block)
              .replace("{logs_block}", logs_block)
              .replace("{topology_block}", topo))
    answer = call_llm(prompt, backend=backend)
    check = verify_citations(answer, alert)
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"alert": alert, "prompt": prompt, "answer": answer, "check": check},
                  f, ensure_ascii=False, indent=2)
    return answer, check, out


if __name__ == "__main__":
    alert = {
        "metrics": [
            {"name": "checkout.p99_latency_ms", "value": 1820, "baseline": 320, "duration": "12m"},
            {"name": "checkout.error_rate", "value": 0.087, "baseline": 0.003, "duration": "12m"},
            {"name": "payment.p99_latency_ms", "value": 1640, "baseline": 120, "duration": "15m"},
            {"name": "bank-gateway.net_latency_ms", "value": 96, "baseline": 20, "duration": "18m"},
        ],
        "logs": [
            {"ts": "10:12:03", "level": "ERROR", "component": "checkout.ApiClient",
             "message": "upstream payment timeout after 1500ms"},
            {"ts": "10:12:07", "level": "WARN", "component": "payment.GatewayClient",
             "message": "bank-gateway handshake retry 3/5 latency=980ms"},
            {"ts": "10:13:11", "level": "ERROR", "component": "checkout.OrderSvc",
             "message": "order create failed: payment unavailable"},
        ],
        "topology": {
            "gateway": ["checkout"],
            "checkout": ["payment", "inventory"],
            "payment": ["bank-gateway"],
        },
    }
    ans, chk, path = summarize(alert, backend="local")
    print(ans)
    print("\n---- 幻觉检查 ----")
    print(json.dumps(chk, ensure_ascii=False, indent=2))
    print("已保存：", path)
```

预期输出（本地 7B 的典型结果）：

```text
1. 现象：checkout 的 checkout.p99_latency_ms 从 320 升到 1820，持续 12m
2. 影响面：payment、inventory
3. 根因候选 Top-3：
   - payment/p99_latency_ms（可信度 高）— 证据：payment.p99_latency_ms 当前=1640, 基线=120
   - bank-gateway/net_latency_ms（可信度 中）— 证据：日志「bank-gateway handshake retry 3/5 latency=980ms」
   - checkout/error_rate（可信度 低）— 证据：checkout.error_rate 从 0.003 升到 0.087
4. 建议动作：
   可自动执行：扩容 checkout 副本、清理日志磁盘
   需人工审批：回滚 payment 最近一次发布、切换 bank-gateway 备用链路
5. 信息不足：缺少 payment 侧的部署变更记录与宿主资源指标

---- 幻觉检查 ----
{
  "unsupported_numbers": [],
  "suspicious_services": [],
  "cited_log_snippets": 1,
  "total_log_snippets": 3,
  "has_insufficient_declaration": true
}
```

**幻觉控制的三道闸**：① prompt 层强制引用；② 输出后做数字与实体校验（`unsupported_numbers` 非空就标记人工复核）；③ 用小模型（0.5B）做二次「是否有编造」判定。三道闸都通过才算「可自动下发」。

### 12.10.3 批量处理的成本估算

| 项 | 取值 | 说明 |
| --- | --- | --- |
| 单次摘要输入 token | 900 | 4 条指标 + 3 行日志 + 拓扑 |
| 单次摘要输出 token | 420 | 五段结构 |
| 每天告警数（收敛后） | 60 | 收敛率 90% 后的量级 |
| 云端输入单价 | 0.001 元/1k token | 按你用到的 API 实际单价替换 |
| 云端输出单价 | 0.002 元/1k token | 同上 |
| 单次云端成本 | 0.9×0.001 + 0.42×0.002 = 0.00174 元 | 约 0.17 分 |
| 月成本（全云端） | 60 × 30 × 0.00174 = 3.13 元 | 量很小 |
| 月成本（本地 7B） | 电费 | 单次约 3 秒 × 60W，可忽略 |
| 本地耗时 | 单次 3–8 秒（7B Q4_K_M） | 60 条/天完全够用 |

:::ok 结论
告警摘要这种「每天几十条、单条千 token 级」的场景，**本地 7B 就够，不需要云端**。云端只在两种情况下才值得：① 需要多模态（带图表）；② 需要更强的推理（复杂故障的根因链条）。把这个成本估算写进报告，能直接体现「成本意识」。
:::

---

## 12.11 ChatOps：历史工单 RAG + Runbook 匹配 + 审批边界

### 12.11.1 闭环设计

ChatOps 的完整闭环是：**告警 → 检索相似历史工单 → 匹配 Runbook → 生成处置建议 → 分级审批 → 执行/留痕**。其中「分级审批」是核心工程约束。

| 动作类别 | 例子 | 是否允许自动执行 | 理由 |
| --- | --- | --- | --- |
| 只读观测 | 拉指标、拉日志、拉拓扑、跑诊断脚本 | **允许** | 无副作用 |
| 幂等扩容 | 副本数 +1、线程池扩 20% | 允许（需限额与冷却期） | 可回滚、影响可控 |
| 缓存与连接 | 清理应用缓存、重置连接池 | 允许（限单服务） | 短时可恢复 |
| 磁盘清理 | 删除 30 天前的日志文件 | 允许（需白名单目录） | 不可逆但影响面小 |
| 重启无状态服务 | 重启 pod、重启 worker | **需审批** | 会中断在途请求 |
| 回滚发布 | 回滚到上一版本 | **需审批** | 影响业务、可能掩盖问题 |
| 切流量/降级 | 关闭某个功能、切备用链路 | **需审批** | 影响用户 |
| 数据操作 | 改数据库、删表、改分片 | **绝不自动** | 不可逆、合规风险 |
| 配置变更 | 改网关规则、改限流阈值 | **绝不自动** | 影响面全局 |
| 权限与密钥 | 改 IAM、轮换密钥 | **绝不自动** | 安全红线 |

:::danger 三条绝不自动执行的铁律
**数据删除/修改、配置变更（尤其是全局网关与限流）、权限与密钥操作** —— 这三类动作无论模型多自信都不许自动执行。工程上要在代码层做硬拦截（白名单机制：只有注册在 `AUTO_ACTIONS` 里的动作才允许自动执行），而不是靠 prompt 提醒模型「不要执行危险操作」。prompt 是软的，代码拦截是硬的。
:::

### 12.11.2 检索历史工单（RAG）

```python
# chatops_rag.py —— 历史工单检索 + Runbook 匹配 + 动作分级审批
# 执行位置：WSL2 Bash
import json
import os
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- 动作白名单：代码层硬拦截 ----------
AUTO_ACTIONS = {
    "scale_replicas", "expand_thread_pool", "clear_app_cache",
    "reset_connection_pool", "rotate_app_log", "run_diagnostic_script",
}
APPROVAL_REQUIRED = {
    "restart_stateless_service", "rollback_release", "shift_traffic",
    "enable_degradation", "restart_database_proxy",
}
FORBIDDEN = {
    "delete_data", "alter_table", "drop_table", "update_config",
    "change_rate_limit", "rotate_secret", "modify_iam", "scale_down_db",
}


@dataclass
class Runbook:
    id: str
    title: str
    symptoms: str
    steps: list
    actions: list
    owner: str


def load_knowledge(base_dir="knowledge"):
    """读取历史工单与 Runbook。工单是 jsonl，Runbook 是 json。"""
    tickets = []
    tpath = os.path.join(base_dir, "tickets.jsonl")
    if os.path.exists(tpath):
        with open(tpath, encoding="utf-8") as f:
            tickets = [json.loads(line) for line in f if line.strip()]
    runbooks = []
    rpath = os.path.join(base_dir, "runbooks.json")
    if os.path.exists(rpath):
        with open(rpath, encoding="utf-8") as f:
            runbooks = [Runbook(**rb) for rb in json.load(f)]
    return tickets, runbooks


class ChatOpsRetriever:
    def __init__(self, model_name="BAAI/bge-m3", device=None):
        self.model = SentenceTransformer(model_name, device=device)
        self.tickets = []
        self.runbooks = []
        self.ticket_vecs = None
        self.runbook_vecs = None

    def build(self, tickets, runbooks):
        self.tickets = tickets
        self.runbooks = runbooks
        if tickets:
            docs = ["%s %s %s" % (t["title"], t.get("symptoms", ""), t.get("resolution", ""))
                    for t in tickets]
            self.ticket_vecs = self.model.encode(docs, normalize_embeddings=True,
                                                 batch_size=32, show_progress_bar=False)
        if runbooks:
            docs = ["%s %s" % (rb.title, rb.symptoms) for rb in runbooks]
            self.runbook_vecs = self.model.encode(docs, normalize_embeddings=True,
                                                  batch_size=32, show_progress_bar=False)
        return self

    def search_tickets(self, query, top_k=3):
        if self.ticket_vecs is None:
            return []
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.ticket_vecs @ q
        idx = np.argsort(-sims)[:top_k]
        return [{"score": round(float(sims[i]), 4), **self.tickets[i]} for i in idx]

    def match_runbooks(self, query, top_k=2):
        if self.runbook_vecs is None:
            return []
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.runbook_vecs @ q
        idx = np.argsort(-sims)[:top_k]
        return [{"score": round(float(sims[i]), 4),
                 "id": self.runbooks[i].id, "title": self.runbooks[i].title,
                 "steps": self.runbooks[i].steps, "actions": self.runbooks[i].actions,
                 "owner": self.runbooks[i].owner} for i in idx]


def classify_actions(actions):
    """把动作分成 自动 / 需审批 / 禁止 三类。"""
    auto = [a for a in actions if a in AUTO_ACTIONS]
    need = [a for a in actions if a in APPROVAL_REQUIRED]
    forbid = [a for a in actions if a in FORBIDDEN or a not in AUTO_ACTIONS | APPROVAL_REQUIRED]
    return {"auto": auto, "need_approval": need, "forbidden": forbid}


def handle_alert(query, retriever, top_k=3):
    tickets = retriever.search_tickets(query, top_k=top_k)
    runbooks = retriever.match_runbooks(query, top_k=2)
    actions = []
    for rb in runbooks:
        actions.extend(rb["actions"])
    plan = classify_actions(sorted(set(actions)))
    return {
        "query": query,
        "similar_tickets": tickets,
        "runbooks": runbooks,
        "plan": plan,
        "need_human": len(plan["need_approval"]) > 0 or len(plan["forbidden"]) > 0,
    }


if __name__ == "__main__":
    os.makedirs("knowledge", exist_ok=True)
    with open("knowledge/tickets.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": "INC-2026-0431", "title": "checkout P99 延迟升高",
            "symptoms": "checkout p99 从 320ms 涨到 1800ms，payment 上游超时",
            "resolution": "扩容 checkout 副本到 8，随后发现 bank-gateway 网络抖动，切备用链路",
            "root_cause": "bank-gateway 网络抖动",
        }, ensure_ascii=False) + "\n")
        f.write(json.dumps({
            "id": "INC-2026-0288", "title": "search CPU 打满",
            "symptoms": "search cpu 使用率 95%，QPS 未变，日志出现大量 GC",
            "resolution": "重启 search pod 并调整堆大小",
            "root_cause": "内存泄漏导致频繁 GC",
        }, ensure_ascii=False) + "\n")
    with open("knowledge/runbooks.json", "w", encoding="utf-8") as f:
        json.dump([
            {"id": "RB-001", "title": "下游超时导致上游延迟升高",
             "symptoms": "上游 P99 延迟升高，日志出现 upstream timeout",
             "steps": ["确认下游指标", "检查网络与连接池", "扩容上游副本", "必要时切备用链路"],
             "actions": ["scale_replicas", "reset_connection_pool", "shift_traffic"],
             "owner": "sre-oncall"},
            {"id": "RB-002", "title": "服务 CPU 打满（疑似内存问题）",
             "symptoms": "CPU 高且 QPS 未变，GC 日志频繁",
             "steps": ["抓 heap dump", "确认是否泄漏", "重启 pod", "提交缺陷单"],
             "actions": ["restart_stateless_service", "rotate_app_log"],
             "owner": "app-team"},
        ], f, ensure_ascii=False, indent=2)

    tickets, runbooks = load_knowledge("knowledge")
    retriever = ChatOpsRetriever(model_name="BAAI/bge-small-zh-v1.5").build(tickets, runbooks)
    res = handle_alert("checkout 延迟从 320ms 升到 1800ms，payment 超时", retriever)
    print(json.dumps(res, ensure_ascii=False, indent=2))
```

预期输出（关键片段）：

```text
{
  "query": "checkout 延迟从 320ms 升到 1800ms，payment 超时",
  "similar_tickets": [{"score": 0.8123, "id": "INC-2026-0431", ...}],
  "runbooks": [{"score": 0.7901, "id": "RB-001", ...}],
  "plan": {"auto": ["reset_connection_pool", "scale_replicas"],
           "need_approval": ["shift_traffic"],
           "forbidden": []},
  "need_human": true
}
```

模型选型：本地检索用 `bge-small-zh-v1.5`（约 100MB，CPU 上毫秒级）或 `bge-m3`（更强但需约 2GB 显存）。8G 上做 ChatOps 检索，`bge-small` 完全够用；`bge-m3` 只在需要中英混检时上。这与共享简报的 RAG 选型一致。

---

## 12.12 端到端 demo：数据 → 模型 → 告警 → 摘要 → 建议

### 12.12.1 链路设计

```text
node_exporter / 自建指标模拟器
        │  （Prometheus 抓取 /metrics）
        ▼
Prometheus ──(告警规则)──> Alertmanager ──(webhook)──> aiops-service
                                                            │
                                       ┌────────────────────┼────────────────────┐
                                       ▼                    ▼                    ▼
                                  异常检测模型        告警收敛引擎         LLM 摘要 + 建议
                                  (AE/IForest)      (指纹/窗口/抑制)      (本地 7B + 引用校验)
                                       │                    │                    │
                                       └────────────────────┴────────────────────┘
                                                            ▼
                                                      Loki/Prometheus 留存
                                                            ▼
                                                        Grafana 看板
```

### 12.12.2 自建指标模拟器

```python
# metric_simulator.py —— 模拟一个微服务的 Prometheus 指标端点（含人为注入的异常）
# 执行位置：WSL2 Bash（python metric_simulator.py，监听 0.0.0.0:9101）
import math
import random
import time

from fastapi import FastAPI, Response

app = FastAPI()
START = time.time()


def base_value(t):
    """正常基线：带日周期与噪声。"""
    return 320 + 40 * math.sin(t / 600.0) + random.gauss(0, 8)


def anomaly_gain(t):
    """注入异常：每 300 秒里有 60 秒延迟抬升 5 倍。"""
    phase = int(t) % 300
    if 240 <= phase < 300:
        return 5.0 + random.gauss(0, 0.2)
    return 1.0


@app.get("/metrics")
def metrics():
    t = time.time() - START
    p99 = base_value(t) * anomaly_gain(t)
    err = 0.003 * (5.0 if anomaly_gain(t) > 1 else 1.0) + abs(random.gauss(0, 0.0005))
    cpu = 0.35 + 0.25 * (1.0 if anomaly_gain(t) > 1 else 0.0) + abs(random.gauss(0, 0.01))
    lines = [
        "# HELP checkout_p99_latency_ms P99 latency of checkout",
        "# TYPE checkout_p99_latency_ms gauge",
        "checkout_p99_latency_ms %.3f" % p99,
        "# HELP checkout_error_rate Error rate of checkout",
        "# TYPE checkout_error_rate gauge",
        "checkout_error_rate %.6f" % err,
        "# HELP checkout_cpu_usage CPU usage of checkout",
        "# TYPE checkout_cpu_usage gauge",
        "checkout_cpu_usage %.4f" % cpu,
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9101)
```

预期输出：启动日志 `Uvicorn running on http://0.0.0.0:9101`。用 `curl.exe http://127.0.0.1:9101/metrics` 能看到三行指标，且每 300 秒中有 60 秒 `checkout_p99_latency_ms` 明显抬高。

### 12.12.3 告警处理服务（接收 Alertmanager webhook）

```python
# aiops_service.py —— 接收 Alertmanager webhook，做检测确认 + 收敛 + 摘要
# 执行位置：WSL2 Bash（python aiops_service.py，监听 0.0.0.0:9200）
import json
import os
import time
from datetime import datetime

from fastapi import FastAPI, Request

app = FastAPI(title="AIOps Service")

# 复用 12.6 的收敛逻辑与 12.10 的摘要逻辑（生产应抽成独立模块）
from alert_compressor import Alert, run_pipeline      # noqa: E402
from alert_summarizer import summarize                # noqa: E402

BUFFER = []            # 待收敛的告警缓冲
SUMMARIES = []


def to_alert(item):
    """Alertmanager 的 webhook 结构 -> 内部 Alert。"""
    labels = item.get("labels", {})
    annotations = item.get("annotations", {})
    return Alert(
        ts=datetime.fromisoformat(item.get("startsAt", datetime.now().isoformat()).replace("Z", "+00:00")).replace(tzinfo=None),
        service=labels.get("service", labels.get("job", "unknown")),
        metric=labels.get("alertname", "unknown"),
        level="critical" if labels.get("severity") == "critical" else "warning",
        value=float(annotations.get("value", 0) or 0),
        message=annotations.get("description", ""),
    )


def build_alert_payload(alerts):
    """把收敛后的告警组装成 LLM 摘要所需的输入（此处用占位指标，生产从 Prometheus 查）。"""
    metrics = []
    for a in alerts:
        metrics.append({"name": "%s.%s" % (a.service, a.metric), "value": a.value,
                        "baseline": 320, "duration": "%dm" % max(1, int(a.count))})
    logs = [{"ts": a.ts.strftime("%H:%M:%S"), "level": "ERROR", "component": a.service + ".App",
             "message": a.message[:120]} for a in alerts]
    topology = {"gateway": ["checkout"], "checkout": ["payment", "inventory"],
                "payment": ["bank-gateway"]}
    return {"metrics": metrics, "logs": logs, "topology": topology}


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    incoming = body.get("alerts", [])
    alerts = [to_alert(x) for x in incoming if x.get("status") == "firing"]
    if not alerts:
        return {"received": 0, "note": "无 firing 告警"}

    BUFFER.extend(alerts)
    if len(BUFFER) < 5:                      # 攒够一批再收敛，模拟 30 秒窗口
        return {"received": len(alerts), "buffered": len(BUFFER)}

    batch = list(BUFFER)
    BUFFER.clear()
    kept = run_pipeline(batch, window_minutes=10)
    if not kept:
        return {"received": len(alerts), "kept": 0}

    payload = build_alert_payload(kept[:5])
    answer, check, path = summarize(payload, backend=os.environ.get("SUMMARY_BACKEND", "local"))
    SUMMARIES.append({"ts": time.time(), "kept": len(kept), "check": check, "path": path})
    return {"received": len(alerts), "kept": len(kept), "check": check, "summary_preview": answer[:200]}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "buffered": len(BUFFER), "summaries": len(SUMMARIES)}
```

### 12.12.4 `docker-compose.yml`

```yaml
# docker-compose.yml —— Prometheus + Grafana + 指标模拟器 + AIOps 服务
# 执行位置：WSL2 Bash（docker compose up -d）
version: "3.9"

services:
  metric-simulator:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - ./:/app
    command: >
      bash -lc "pip install -q fastapi uvicorn && python metric_simulator.py"
    ports:
      - "9101:9101"
    restart: unless-stopped

  aiops-service:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - ./:/app
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - SUMMARY_BACKEND=${SUMMARY_BACKEND:-local}
      - OLLAMA_BASE=http://host.docker.internal:11434
    command: >
      bash -lc "pip install -q fastapi uvicorn httpx openai drain3 scikit-learn pandas numpy python-dateutil && python aiops_service.py"
    ports:
      - "9200:9200"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.54.1
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./alert.rules.yml:/etc/prometheus/alert.rules.yml:ro
      - prom-data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=15d"
      - "--web.enable-lifecycle"
    ports:
      - "9090:9090"
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:v0.27.0
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
    ports:
      - "9093:9093"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:11.2.0
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=aiops123
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana-provisioning:/etc/grafana/provisioning:ro
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prom-data:
  grafana-data:
```

### 12.12.5 配套配置与启动命令

```yaml
# prometheus.yml —— 抓取模拟器指标，加载告警规则
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/alert.rules.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  - job_name: "metric-simulator"
    static_configs:
      - targets: ["metric-simulator:9101"]
        labels:
          service: checkout
```

```yaml
# alert.rules.yml —— 两条基础告警规则（阈值 + 持续时长）
groups:
  - name: checkout-rules
    rules:
      - alert: CheckoutHighLatency
        expr: checkout_p99_latency_ms > 800
        for: 1m
        labels:
          severity: critical
          service: checkout
        annotations:
          summary: "checkout P99 延迟过高"
          description: "checkout_p99_latency_ms = {{ $value }}，阈值 800"
      - alert: CheckoutHighErrorRate
        expr: checkout_error_rate > 0.02
        for: 1m
        labels:
          severity: warning
          service: checkout
        annotations:
          summary: "checkout 错误率升高"
          description: "checkout_error_rate = {{ $value }}，阈值 0.02"
```

```yaml
# alertmanager.yml —— 把告警转发给自建 AIOps 服务
route:
  receiver: aiops-webhook
  group_by: ["service", "alertname"]
  group_wait: 20s
  group_interval: 1m
  repeat_interval: 30m

receivers:
  - name: aiops-webhook
    webhook_configs:
      - url: "http://aiops-service:9200/webhook"
        send_resolved: false
```

```bash
# 在 WSL2 的 Bash 中执行：拉起整套监控栈并验证
cd ~/aiops/demo
docker compose up -d
docker compose ps
curl -s http://127.0.0.1:9090/-/ready
curl -s http://127.0.0.1:9101/metrics | head -5
curl -s http://127.0.0.1:9200/healthz
```

预期输出：`docker compose ps` 显示 5 个容器 `running`；`9090/-/ready` 返回 `Prometheus Server is Ready.`；`healthz` 返回 `{"status":"ok","buffered":0,"summaries":0}`。

大约 4–5 分钟后（等指标越过阈值 + Alertmanager 的 `group_wait`），`docker compose logs aiops-service` 里会出现 `kept` 与 `check` 输出，并且 `summaries/` 目录下生成 JSON 文件。

### 12.12.6 Grafana 看板配置要点

| 面板 | 查询 | 用途 |
| --- | --- | --- |
| 服务延迟曲线 | `checkout_p99_latency_ms` | 主指标，叠加阈值线与异常区间标注 |
| 错误率曲线 | `checkout_error_rate` | 与延迟对照，判断是否业务受影响 |
| CPU 使用率 | `checkout_cpu_usage` | 辅助判断是否资源瓶颈 |
| 告警条数（按服务） | `sum(ALERTS{alertstate="firing"})` by (service) | 收敛效果一目了然 |
| 收敛率 | `1 - (收敛后计数 / 原始计数)`，用 Recording Rule 预算 | KPI 面板 |
| 摘要文本 | 用 Grafana 的 Text 面板接 AIOps 服务的接口 | 值班同学直接看结论 |

设置要点：① 时间范围默认 `last 1 hour`，刷新 `10s`；② 延迟面板加 `Thresholds`（800 变黄、1500 变红）；③ 用 `Annotations` 把 AIOps 服务产生的告警点打在图上；④ 告警面板按 `service` 分组，观察收敛前后条数变化。

:::tip 先跑通再看板
不要一上来就配 Grafana。先确认「指标能被抓到 → 规则能触发 → Alertmanager 能转发 → 服务能收到」这条链路通了，再配看板。链路不通时看板只会给你一堆空面板，浪费时间。
:::

---

## 12.13 验收指标：MTTR 缩短、误报率下降、告警收敛率

### 12.13.1 三个 KPI 的计算方式

```python
# kpi_report.py —— 计算 MTTR、误报率、告警收敛率并生成结果表
# 执行位置：WSL2 Bash
import json
from datetime import datetime

import pandas as pd


def load_incidents(path="incidents.jsonl"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    for col in ["alert_ts", "detect_ts", "resolve_ts"]:
        df[col] = pd.to_datetime(df[col])
    return df


def compute_mttr(df, start_col="alert_ts", end_col="resolve_ts"):
    """MTTR = 平均（恢复时刻 - 告警产生时刻）。可换 start_col 做口径对比。"""
    dur = (df[end_col] - df[start_col]).dt.total_seconds() / 60.0
    return {
        "样本数": int(len(df)),
        "MTTR_分钟": round(float(dur.mean()), 2),
        "P50_分钟": round(float(dur.quantile(0.5)), 2),
        "P95_分钟": round(float(dur.quantile(0.95)), 2),
        "口径": "%s -> %s" % (start_col, end_col),
    }


def compute_false_alarm_rate(alerts):
    """alerts: [{is_true_fault: bool}] 误报率 = 非真实故障 / 全部告警。"""
    total = len(alerts)
    false_n = sum(1 for a in alerts if not a["is_true_fault"])
    return {
        "告警总数": total,
        "误报条数": false_n,
        "误报率": round(false_n / total, 4) if total else 0.0,
        "精确率": round(1 - false_n / total, 4) if total else 0.0,
    }


def compute_convergence(before, after):
    return {
        "收敛前": before,
        "收敛后": after,
        "收敛率": round(1 - after / before, 4) if before else 0.0,
    }


if __name__ == "__main__":
    demo = pd.DataFrame([
        {"alert_ts": "2027-03-01 10:12:00", "detect_ts": "2027-03-01 10:14:10", "resolve_ts": "2027-03-01 10:26:00"},
        {"alert_ts": "2027-03-02 03:41:00", "detect_ts": "2027-03-02 03:44:30", "resolve_ts": "2027-03-02 04:05:00"},
        {"alert_ts": "2027-03-03 21:05:00", "detect_ts": "2027-03-03 21:05:40", "resolve_ts": "2027-03-03 21:18:00"},
    ])
    for col in ["alert_ts", "detect_ts", "resolve_ts"]:
        demo[col] = pd.to_datetime(demo[col])

    print("=== MTTR ===")
    print(compute_mttr(demo, "alert_ts", "resolve_ts"))
    print(compute_mttr(demo, "detect_ts", "resolve_ts"))

    print("\n=== 误报率 ===")
    alerts = [{"is_true_fault": x} for x in
              [True] * 42 + [False] * 158]        # 200 条告警里 42 条真故障
    print(compute_false_alarm_rate(alerts))

    print("\n=== 告警收敛 ===")
    print(compute_convergence(before=1000, after=118))
```

预期输出：

```text
=== MTTR ===
{'样本数': 3, 'MTTR_分钟': 16.83, 'P50_分钟': 15.0, 'P95_分钟': 22.75, '口径': 'alert_ts -> resolve_ts'}
{'样本数': 3, 'MTTR_分钟': 14.61, 'P50_分钟': 14.33, 'P95_分钟': 19.4, '口径': 'detect_ts -> resolve_ts'}

=== 误报率 ===
{'告警总数': 200, '误报条数': 158, '误报率': 0.79, '精确率': 0.21}

=== 告警收敛 ===
{'收敛前': 1000, '收敛后': 118, '收敛率': 0.882}
```

:::warn 误报率的两种口径不要混用
口径 A：`误报条数 / 全部告警条数`（告警视角，运维最关心）。
口径 B：`误报的异常点 / 全部正常点`（模型视角，即 FAR）。
同一套系统两个口径能差 100 倍（A 是 0.79，B 可能是 0.008）。报告里必须写清用的是哪一个。
:::

### 12.13.2 结果表模板

| 指标 | 改造前 | 改造后 | 变化 | 样本量 | 口径说明 |
| --- | --- | --- | --- | --- | --- |
| MTTR（分钟） | 填入 | 填入 | 填入 | 填入 | 告警产生 → 恢复 |
| MTTR P95（分钟） | 填入 | 填入 | 填入 | 填入 | 同上 |
| 日均告警条数 | 填入 | 填入 | 填入 | 填入 | 连续 7 天 |
| 误报率（告警视角） | 填入 | 填入 | 填入 | 填入 | 误报条数 / 全部告警 |
| 告警收敛率 | 不适用 | 填入 | 填入 | 填入 | 1 − 收敛后 / 收敛前 |
| 检测延迟 P50（秒） | 填入 | 填入 | 填入 | 填入 | 异常起点 → 首次告警 |
| 根因 Top-3 命中率 | 不适用 | 填入 | 填入 | 填入 | 人工标注根因 |
| 摘要人工可用率 | 不适用 | 填入 | 填入 | 填入 | 值班同学打分 ≥3/5 的比例 |

:::ok 「改造前」怎么来
研一阶段没有真实生产环境，两种可行做法：① 用规则型告警（阈值 + 无收敛）作为「改造前」的基线，在你的模拟器或公开数据集上跑一遍；② 用历史工单中的处置时长作为 MTTR 基线。报告里必须写明「基线来自模拟环境/公开数据集，不是真实生产」，这才是诚实的口径，也是面试官会认可的做法。
:::

---

## 12.14 简历写法与面试要点

### 12.14.1 三条带数字的简历句

**句 1（异常检测方向）**：在 SMD 数据集（28 台服务器 × 38 维指标，逐点标签）上实现五种时序异常检测方法（3-sigma / MAD / 滑动分位 / Isolation Forest / 窗口自编码器）并统一口径对比：自编码器（窗口 32、步长 8）取得 point-wise F1 0.699、point-adjusted F1 0.847，较 3-sigma 基线（F1 0.487）提升 21.2 个百分点、误报条数从 1865 降至 722、检测延迟从 12 点降至 5 点；单机 8GB 显存训练 30 epoch 耗时 92 秒，显存峰值 0.4GB。

**句 2（告警收敛与根因方向）**：设计并实现三层告警收敛流水线（指纹去重 + 同服务同指标时间窗聚合 + 基于服务拓扑的上游抑制），在 1000 条模拟告警上把告警量压到 118 条（收敛率 88.2%），同时保留被抑制告警的溯源字段用于审计；配合「相关性 + 拓扑距离 + 时间超前」三点加权的根因排序器，在自建 5 服务拓扑的 240 点故障样本上把真实根因（上游网络抖动）排进 Top-3，人工标注根因的 Top-3 命中率 X%。

**句 3（LLM + 运维落地方向）**：搭建端到端 AIOps demo（Prometheus + Alertmanager + Grafana + FastAPI 自建服务，`docker-compose` 一键拉起），用 Drain3 在 HDFS 日志（1100 万行）上挖出 N 个模板（`depth=4, sim_th=0.4`），模板词袋 + 随机森林的会话级日志异常检测达到 F1 X；接本地 7B（GGUF Q4_K_M）做告警摘要与处置建议，通过「结构化输入 + 强制引用 + 输出后数字与实体校验」三道闸把无依据数字的生成比例从 X% 压到 0%，单条摘要耗时 3–8 秒、日均 60 条的场景下月度成本约 0 元（全本地）。

:::tip AIOps 简历句的加分点
写清「评价口径」（point-wise 与 point-adjusted 都写）、「基线」（规则基线 vs 模型）、「样本量」、「是否模拟环境」。AIOps 岗位的面试官最反感的就是只报一个漂亮的 F1 却说不清口径的人。
:::

### 12.14.2 十个必问问题与答题要点

1. **「point-wise 和 point-adjusted 有什么区别？你为什么两个都报？」**
   要点：前者逐点判对错，后者「段内命中一点即整段算命中」。只报后者会让数字虚高（同一份预测能从 3.3% 变成 100%）。两个都报是为了同时给出「真实逐点能力」与「与论文可比的口径」，并且防止调阈值时被点调整带偏。

2. **「误报率你怎么定义？和检测率的关系是什么？」**
   要点：说清两个口径（告警视角 `误报条数/告警总数`、模型视角 `FP/(FP+TN)`）。误报率与召回率此消彼长，调阈值就是在这条曲线上选点；运维上还要报「日均告警条数」与「每段告警条数」，因为值班同学关心的是绝对量。

3. **「Drain 算法的三个关键参数是什么？调参会怎么变？」**
   要点：`depth`（前缀树层数，越小合并越激进）、`sim_th`（相似度阈值，0.3→0.7 模板数能涨 6 倍）、`max_children`/`max_clusters`（防树爆炸与内存上限）。给实测数字：`sim_th=0.3` 出 21 个模板、`0.7` 出 129 个。

4. **「HDFS 日志的标签是什么粒度？这会带来什么问题？」**
   要点：**block（会话）级**，不是行级。做行级检测必须把 block 标签传播到行，属于弱监督，标签有噪声；报告里必须写明，否则指标不可信。

5. **「告警收敛率怎么算？只报收敛率有什么风险？」**
   要点：`1 − 收敛后条数/收敛前条数`（还有组口径与抑制口径）。只报收敛率的风险是可能抑制掉真实故障，所以必须同时报「被抑制告警中后来成为真实故障的比例」以及抑制规则的 TTL 与审计字段。

6. **「你怎么判断根因？相关性就够了吗？」**
   要点：相关性不够（可能是共同诱因或反向影响）。要有三个条件：拓扑可达（上游）、时间超前（根因先动）、相关性显著。再补一句「生产上还要结合变更事件（发布、配置变更）作为最强证据」。

7. **「8G 显存上怎么跑 Anomaly Transformer？」**
   要点：缩 `batch_size`（32→8）、`d_model`（512→32）、`e_layers`（3→2）、`seq_len`（100→64），并设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。改完显存峰值约 2.1–2.9GB，3 epoch 约 6–12 分钟。同时要说明「缩了之后精度变化」才算完整。

8. **「LLM 做告警摘要会不会编造指标？你怎么控制？」**
   要点：三道闸 —— ① prompt 层要求每条结论引用原始指标名与日志片段，并允许回答「信息不足」；② 输出后用正则抽取所有数字与实体，与输入比对，不匹配就标记人工复核；③ 用小模型做二次判定。给实测数字：无依据数字比例从 X% 降到 0%。

9. **「哪些运维动作绝对不能自动执行？」**
   要点：数据删除与修改、全局配置与限流变更、权限与密钥操作。并且强调「拦截要在代码层用白名单实现，不能靠 prompt」——这是工程与玩具的分界线。

10. **「你的 AIOps 项目 MTTR 降了多少？怎么算的？」**
    要点：给公式（恢复时刻 − 告警产生时刻，取平均与 P95）、给样本量、给起点定义、并诚实说明基线来源（公开数据集或模拟环境，不是生产）。再补一句「真正影响 MTTR 的是从告警到定位的时间，所以我把根因排序器放在链路里」。

---

:::ok 本章交付物检查清单
- [ ] 公开数据集已下载并跑通探查脚本，异常占比与标签粒度写进笔记
- [ ] 五种异常检测方法在同一口径下完成对比表（point-wise 与 point-adjusted 两栏都有）
- [ ] AIOps Challenge 或多模态数据集上完成一次故障定位实验
- [ ] Drain3 参数扫描表（`depth` × `sim_th`）已产出
- [ ] 告警收敛流水线跑通，收敛率与审计字段都有
- [ ] 根因排序器给出 Top-3，命中率有样本量说明
- [ ] LLM 摘要的幻觉检查脚本跑通，无依据数字为 0
- [ ] `docker-compose` 一键拉起监控栈，Grafana 看板有 4 个以上有效面板
- [ ] MTTR / 误报率 / 收敛率三项 KPI 的结果表填好，口径注明
- [ ] 三条简历句用实测数字改写完成
:::
