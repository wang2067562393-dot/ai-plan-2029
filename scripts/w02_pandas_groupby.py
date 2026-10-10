#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""w02_pandas_groupby.py —— W2 D2：Pandas 分组聚合与多级索引。

与课表原文的唯一差别（网络加固）：
    原文：  df = pd.read_csv("https://raw.githubusercontent.com/.../titanic.csv")
    现在：  先读 data/raw/titanic.csv；没有就调用 w02_get_data.py 的逻辑下载并缓存。

原因：raw.githubusercontent.com 在不少链路上于 TLS 握手阶段被 RST，
`pd.read_csv(URL)` 没有重试、没有镜像、没有缓存，一断就整段报错。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# 允许 `uv run python scripts/w02_pandas_groupby.py` 时也能 import 同目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from w02_get_data import RAW_DIR, build_opener, fetch_one  # noqa: E402

CACHE = RAW_DIR / "titanic.csv"


def load_titanic() -> pd.DataFrame:
    if CACHE.exists() and CACHE.stat().st_size > 64:
        print(f"[data] 使用本地缓存 {CACHE}")
        return pd.read_csv(CACHE)

    print("[data] 本地无缓存，尝试联网获取（GitHub 直连 + 多镜像降级）")
    fetch_one("titanic.csv", build_opener())

    if not CACHE.exists():
        raise SystemExit(
            f"无法取得 {CACHE}。请按上面打印的手工方案下载后放到该路径，再重跑本脚本。"
        )
    return pd.read_csv(CACHE)


def main() -> None:
    df = load_titanic()
    print(f"shape = {df.shape}\n")

    # ---- 原文的练习内容：分组聚合 + 多级索引 ----
    print("按舱位与性别：生还率与人数")
    print(df.groupby(["Pclass", "Sex"])["Survived"].agg(["mean", "count"]))

    print("\n各舱位生还率（unstack 成多级列索引）")
    print(df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack())

    print("\n缺失值最多的 5 列")
    print(df.isna().sum().sort_values(ascending=False).head())


if __name__ == "__main__":
    main()
