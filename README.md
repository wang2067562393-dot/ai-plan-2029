# ai-plan-2029 · AI 转型系统训练日志

> 研一在读（人工智能方向）的系统补强计划：从工程地基到深度学习，每周一个可运行、可验证的交付物。
> 原则：**每周代码必须跑通，每个结果必须可复现，每次环境问题都沉淀成脚本。**

![ci](https://github.com/wang2067562393-dot/ai-plan-2029/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![uv](https://img.shields.io/badge/uv-依赖管理-26120B?logo=astral&logoColor=white)
![status](https://img.shields.io/badge/进度-W3进行中-2EA44E)

## 这是什么

一份公开的、按周推进的 AI 学习工程仓库。不堆书单，只记录真正跑过的代码、踩过的环境坑和可复现的实验结果。

## 目录结构

```
ai-plan-2029/
├── scripts/              # 每周练习脚本
│   ├── bench_gpu.py            # GPU 显存预算与精度吞吐基准
│   ├── w02_get_data.py         # 多镜像降级的数据下载器
│   ├── w02_pandas_groupby.py   # Pandas 分组聚合练习
│   └── w03_survival_analysis.py# 纯函数化的生还率分析（可测试）
├── tests/                # 单元测试（GitHub Actions 自动运行）
│   └── test_w03.py
├── outputs/              # 实验结果（入库的小 JSON）
│   └── w01_bench.json
├── src/ai_plan_2029/     # 可安装包骨架（uv + ruff + mypy + pytest + pre-commit）
├── w02_offline.sh        # 一键离线化数据获取（多源重试 + 列名校验 + 二次运行秒退）
└── pyproject.toml
```

## 环境复现

```bash
# 使用 uv 管理环境（Python 3.11+）
uv sync

# GPU 基准：实测本机 CUDA / 显存 / 三种精度吞吐
uv run python scripts/bench_gpu.py
# 结果写入 outputs/w01_bench.json

# W2：离线获取 titanic 数据后跑分组聚合
bash w02_offline.sh
uv run python scripts/w02_pandas_groupby.py

# W3：纯函数化分析（IO 与逻辑分离），生成 Markdown 摘要
uv run python scripts/w03_survival_analysis.py

# 单元测试（CI 同一条命令）
python -m unittest discover -s tests -v
```

## 实验记录

### W1 · GPU 基准（RTX 4060 Laptop, 8 GB）

| 精度 | 4096³ 矩阵乘耗时 | 实测吞吐 |
|---|---|---|
| FP32 | 10.01 ms | 13.72 TFLOPS |
| FP16 | 4.34 ms | 31.68 TFLOPS |
| BF16 | 4.33 ms | 31.71 TFLOPS |

**结论**：本机开启 TF32/低精度后吞吐约为 FP32 的 2.3 倍；8 GB 显存扣除系统占用约可用 6.7 GB，后续训练按此预算设计 batch size。

### W2 · 数据工程地基（进行中）

- Pandas 分组聚合 / 多级索引（Titanic）；
- 工程化的离线数据获取：直连失败时按 5 个镜像源降级，SHA 之外做**列名校验防拦截页污染**，重复运行直接秒退，数据目录自动写入 `.gitignore`。

### W3 · 可测试的分析代码

- 把"W2 的打印式分析"重构成**纯函数**（`survival_by` / `build_markdown`），IO 与统计逻辑分离；
- 用合成小数据写单元测试（分组生还率、多维分组、Markdown 渲染），不依赖网络和真实数据文件；
- GitHub Actions 在每次 push 时自动跑测试。

## 后续路线

- [x] W1–W3：GPU 基准、数据工程离线化、可测试的 Pandas 分析
- [ ] W4：Matplotlib 可复现图表（固定随机种子与样式）
- [ ] W5–W8：PyTorch 从零手搓 MLP / CNN（CIFAR-10），不看教程能复现训练循环
- [ ] W9+：Transformer 手写 → 微调与推理实践（LoRA、量化、投机解码）

## 工具链约定

- 依赖与环境：`uv` 锁定（`uv.lock` 入库）；
- 代码质量：`ruff` + `mypy` + `pre-commit`；
- 实验结果：机器可读 JSON 入库，大文件与原始数据不入库；
- 提交信息：Conventional Commits（`feat(w2): ...`）。
