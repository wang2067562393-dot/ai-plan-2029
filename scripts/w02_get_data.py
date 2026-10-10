#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""w02_get_data.py —— 把 W2 数据集落到 <项目根>/data/raw/，避开 GitHub 直连封锁。

【为什么需要它】
raw.githubusercontent.com 在不少链路上于 TLS 握手 (ClientHello/SNI) 阶段被 RST，
`pd.read_csv("https://raw.githubusercontent.com/...")` 与裸 urllib 都必然失败；
而且 pandas/urllib 既没有重试也没有缓存。所以正确姿势是：
「**本地文件优先，联网只是兜底**」。

【本脚本的路径锚定（修掉了上一版的 cwd 依赖 bug）】
不再用相对的 "data/raw"，而是以 __file__ 反推项目根：
    <root>/scripts/w02_get_data.py  ->  <root>/data/raw/
无论你在家目录还是 scripts/ 里执行，都落在同一个地方。

【获取顺序】
  0) <root>/data/raw/<name> 已存在 -> 直接跳过，永不联网（这就是"一劳永逸"）
  1) 若设置了 HTTPS_PROXY/ALL_PROXY -> 走代理
  2) 直连 GitHub raw（3 次指数退避）
  3) 依次尝试加速镜像，每个都校验状态码 + Content-Type + 内容首行
  4) 全失败 -> 打印三条手工方案（含 Windows 侧代理下载命令）后退出

【用法】
    uv run python scripts/w02_get_data.py            # 取 titanic.csv
    uv run python scripts/w02_get_data.py --all      # titanic/iris/tips
    HTTPS_PROXY=http://127.0.0.1:7890 uv run python scripts/w02_get_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "pandas-dev/pandas"
BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

# --- 内容特征：用来一眼识破"看起来 200 实际是拦截页"的响应 ---
EXPECTED_HEADERS = {
    "titanic.csv": b"PassengerId",
    "iris.csv": b"sepal_length",
    "tips.csv": b"total_bill",
}

# GitHub 加速前缀（把完整 raw URL 拼在其后即可）
GITHUB_PROXIES = [
    "https://ghfast.top/",
    "https://gh-proxy.com/",
    "https://ghproxy.net/",
    "https://raw.gitmirror.com/",
]

# CDN 模板（用 format 填 repo/branch/path）
MIRROR_TEMPLATES = [
    "https://cdn.jsdelivr.net/gh/{repo}@{branch}/{path}",
    "https://gcore.jsdelivr.net/gh/{repo}@{branch}/{path}",
    "https://fastly.jsdelivr.net/gh/{repo}@{branch}/{path}",
]

DATASETS = {
    "titanic.csv": f"{RAW_BASE}/doc/data/titanic.csv",
    "iris.csv": f"{RAW_BASE}/doc/data/iris.csv",
    "tips.csv": f"{RAW_BASE}/doc/data/tips.csv",
}

TIMEOUT = 25
ATTEMPTS = 3
MIN_BYTES = 64


def project_root(start: Path | None = None) -> Path:
    """从脚本位置向上找项目根（含 pyproject.toml / .git 的目录）。"""
    here = (start or Path(__file__)).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return here.parent


RAW_DIR = project_root() / "data" / "raw"


def targets_for(filename: str) -> list[str]:
    path = f"doc/data/{filename}"
    raw_url = f"{RAW_BASE}/{path}"
    urls = [raw_url]
    urls += [t.format(repo=REPO, branch=BRANCH, path=path) for t in MIRROR_TEMPLATES]
    urls += [prefix + raw_url for prefix in GITHUB_PROXIES]
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def build_opener() -> urllib.request.OpenerDirector:
    """urllib 只认环境变量代理：不读 Windows 注册表，也不读 Linux 系统代理设置。"""
    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("ALL_PROXY")
        or os.environ.get("all_proxy")
    )
    if proxy:
        print(f"[net] 使用代理: {proxy}")
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    print("[net] 未设置 *_PROXY -> 直连（若被 RST，见失败后打印的手工方案）")
    return urllib.request.build_opener()


def verify_payload(
    payload: bytes, filename: str, content_type: str
) -> tuple[bool, str]:
    if len(payload) < MIN_BYTES:
        return False, f"内容过短（{len(payload)} bytes）"
    if "html" in content_type.lower() or "json" in content_type.lower():
        return False, f"Content-Type 是 {content_type!r}（拦截页/错误页）"
    head = payload[:400].lstrip().lower()
    if (
        head.startswith(b"<!doctype")
        or head.startswith(b"<html")
        or head.startswith(b"{")
    ):
        return False, "响应体是 HTML/JSON，不是数据文件"
    expected = EXPECTED_HEADERS.get(filename)
    if expected and expected.lower() not in payload[:400].lower():
        return False, f"首行缺少预期列名 {expected!r}"
    return True, "ok"


def download(url: str, dest: Path, opener: urllib.request.OpenerDirector) -> bool:
    """下载单个 URL：原子落盘 + 三次退避重试；任何异常只记日志不外抛。"""
    dest.parent.mkdir(parents=True, exist_ok=True)  # <- 修掉上一版的 FileNotFoundError
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "curl/8.5.0", "Accept": "*/*"}
            )
            with opener.open(req, timeout=TIMEOUT) as resp:  # noqa: S310
                payload = resp.read()
                content_type = resp.headers.get("Content-Type", "")
            ok, why = verify_payload(payload, dest.name, content_type)
            if not ok:
                print(f"    ! 校验不通过（{why}）-> 换下一个源")
                return False
            tmp.write_bytes(payload)
            tmp.replace(dest)
            digest = hashlib.sha256(payload).hexdigest()[:12]
            print(f"    OK {len(payload):,} bytes  sha256:{digest}  <- {url}")
            return True
        except Exception as exc:  # noqa: BLE001  网络异常种类多，统一降级
            print(f"    第{attempt}/{ATTEMPTS}次失败: {type(exc).__name__}: {exc}")
            if attempt < ATTEMPTS:
                time.sleep(2**attempt)
    return False


def manual_help(filename: str, dest: Path) -> None:
    print(f"\n[FAIL] {filename} 的直连与所有镜像均不可用。三条一劳永逸的手工方案：")
    print(f"  落盘目标：{dest}\n")
    print("  (1) 有代理：导出环境变量后重跑本脚本")
    print("        export HTTPS_PROXY=http://127.0.0.1:7890   # 换成你自己的端口")
    print("        uv run python scripts/w02_get_data.py\n")
    print("  (2) Windows 侧已有代理，直接下载再上传（WSL 可直接读 /mnt/c/...）：")
    print("        Invoke-WebRequest -Proxy http://127.0.0.1:7897 \\")
    print(
        f'          -Uri {DATASETS[filename]} -OutFile "$env:USERPROFILE\\Desktop\\titanic.csv"'
    )
    print(f"        # 然后 scp 到服务器 {dest}\n")
    print("  (3) 浏览器点下面的链接保存：")
    for url in [DATASETS[filename], f"https://ghfast.top/{DATASETS[filename]}"]:
        print(f"        {url}")


def fetch_one(
    filename: str,
    opener: urllib.request.OpenerDirector,
    raw_dir: Path = RAW_DIR,
) -> bool:
    if filename not in DATASETS:
        print(f"[skip] 未知数据集 {filename}（可选 {list(DATASETS)}）")
        return False

    dest = raw_dir / filename
    if dest.exists() and dest.stat().st_size >= MIN_BYTES:
        print(f"[skip] {dest} 已存在（{dest.stat().st_size:,} bytes），不再联网")
        return True

    print(f"\n[get ] {filename}  ->  {dest}")
    for url in targets_for(filename):
        if download(url, dest, opener):
            return True

    manual_help(filename, dest)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="下载 W2 数据集（锚定项目根，多镜像降级）"
    )
    parser.add_argument("names", nargs="*", help="文件名，默认 titanic.csv")
    parser.add_argument("--all", action="store_true", help="下载 titanic/iris/tips")
    parser.add_argument(
        "--dir", default=None, help="自定义落盘目录（默认 <root>/data/raw）"
    )
    args = parser.parse_args()

    raw_dir = Path(args.dir).expanduser().resolve() if args.dir else RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[path] 项目根 = {project_root()}\n[path] 数据目录 = {raw_dir}")

    names = list(DATASETS) if args.all else (args.names or ["titanic.csv"])
    opener = build_opener()
    results = {name: fetch_one(name, opener, raw_dir) for name in names}

    print("\n=== 结果 ===")
    ok_all = True
    for name, ok in results.items():
        ok_all &= ok
        print(f"  {'OK  ' if ok else 'FAIL'} {raw_dir / name}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
