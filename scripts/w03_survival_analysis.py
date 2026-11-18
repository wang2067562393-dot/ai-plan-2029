#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""w03_survival_analysis.py —— W3：把分组统计写成可测试的纯函数 + 产出 Markdown 摘要。

数据仍走 W2 的离线缓存（data/raw/titanic.csv），分析逻辑与 IO 分离：
- survival_by(...)    纯函数，单元测试直接喂小 DataFrame，不联网；
- build_markdown(...) 把透视表渲染成 Markdown；
- main() 负责读数据、落盘 outputs/。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w02_pandas_groupby import load_titanic  # noqa: E402


def survival_by(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """按给定列聚合生还率与人数，按生还率降序。"""
    g = df.groupby(by)["Survived"].agg(survival_rate="mean", count="count")
    return g.sort_values("survival_rate", ascending=False).reset_index()


def build_markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in table.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.3f}" if c == "survival_rate" else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load_titanic()
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)

    sections = []
    for by, title in [(["Pclass"], "各舱位生还率"), (["Sex"], "各性别生还率"), (["Pclass", "Sex"], "舱位×性别")]:
        table = survival_by(df, by)
        sections.append(f"## {title}\n\n" + build_markdown(table))

    report = "# Titanic 生还率分析（W3）\n\n" + "\n".join(sections)
    dest = out_dir / "w03_summary.md"
    dest.write_text(report, encoding="utf-8")
    print(report)
    print(f"[ok] 已写入 {dest}")


if __name__ == "__main__":
    main()
