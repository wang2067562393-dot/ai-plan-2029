#!/usr/bin/env bash
# w02_offline.sh —— 一次跑完，把 W2 的 titanic.csv 落到 <项目根>/data/raw/，之后永久离线可用。
#
#   chmod +x w02_offline.sh
#   ./w02_offline.sh                                    # 自动找项目根
#   ./w02_offline.sh ~/projects/ai-plan-2029             # 指定项目根
#   HTTPS_PROXY=http://127.0.0.1:7890 ./w02_offline.sh   # 有代理就走代理
#
# 特点：多个源逐个试 + 校验内容 + 校验列名 + 二次运行秒退 + 自动写入 .gitignore。
set -euo pipefail

PROJECT_ROOT="${1:-}"
if [[ -z "$PROJECT_ROOT" ]]; then
  d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [[ "$d" != "/" && ! -f "$d/pyproject.toml" && ! -d "$d/.git" ]]; do d="$(dirname "$d")"; done
  PROJECT_ROOT="$d"
fi
RAW_DIR="$PROJECT_ROOT/data/raw"
DEST="$RAW_DIR/titanic.csv"
RAW="https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"

echo "[path] 项目根   = $PROJECT_ROOT"
echo "[path] 目标文件 = $DEST"

if [[ -s "$DEST" ]] && head -c 400 "$DEST" | grep -q 'PassengerId'; then
  echo "[skip] 已存在且内容正确（$(stat -c%s "$DEST") bytes），无需联网。"
  exit 0
fi

command -v curl >/dev/null || { echo "缺少 curl，请先 apt install curl"; exit 1; }
mkdir -p "$RAW_DIR"

PROXY_ARGS=()
if [[ -n "${HTTPS_PROXY:-}" ]]; then
  PROXY_ARGS=(--proxy "$HTTPS_PROXY")
  echo "[net] 使用代理 $HTTPS_PROXY"
fi

CANDIDATES=(
  "$RAW"
  "https://ghfast.top/$RAW"
  "https://gh-proxy.com/$RAW"
  "https://ghproxy.net/$RAW"
  "https://cdn.jsdelivr.net/gh/pandas-dev/pandas@main/doc/data/titanic.csv"
  "https://gcore.jsdelivr.net/gh/pandas-dev/pandas@main/doc/data/titanic.csv"
)

TMP="$DEST.part"
ok=0
for url in "${CANDIDATES[@]}"; do
  echo "[get ] $url"
  if curl -fsSL --connect-timeout 10 --retry 3 --retry-delay 2 \
        -A 'curl/8.5.0' "${PROXY_ARGS[@]}" -o "$TMP" "$url" \
     && [[ $(wc -c < "$TMP") -gt 1000 ]] \
     && head -c 200 "$TMP" | grep -q 'PassengerId'; then
    mv "$TMP" "$DEST"
    echo "[ok  ] 已保存 $(wc -c < "$DEST") bytes -> $DEST"
    ok=1
    break
  fi
  rm -f "$TMP"
done

if [[ "$ok" -ne 1 ]]; then
  echo "[fail] 所有源都不可用。两条手工方案：" >&2
  echo "  1) 有代理：HTTPS_PROXY=http://127.0.0.1:7890 $0 \"$PROJECT_ROOT\"" >&2
  echo "  2) Windows 侧下载后传上来（PowerShell）：" >&2
  echo "       Invoke-WebRequest -Proxy http://127.0.0.1:7897 -Uri $RAW -OutFile \"\$env:USERPROFILE\\Desktop\\titanic.csv\"" >&2
  echo "       scp ~/Desktop/titanic.csv  <user>@<host>:$DEST" >&2
  exit 1
fi

# 校验列名，防止把拦截页当成数据（python3 / python / uv run python 依次兜底）
PY=""
for cand in python3 python; do
  command -v "$cand" >/dev/null 2>&1 && { PY="$cand"; break; }
done
if [[ -z "$PY" ]] && command -v uv >/dev/null 2>&1; then
  PY="uv run --no-project python"
fi
if [[ -z "$PY" ]]; then
  echo "[warn] 没有可用的 python，跳过列名校验（文件已下载，可自行 head -1 $DEST 检查）"
else
  $PY - "$DEST" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as fh:
    cols = next(csv.reader(fh))

need = {"PassengerId", "Survived", "Pclass", "Sex", "Age", "Fare"}
missing = need - set(cols)
if missing:
    raise SystemExit(f"[fail] 缺少列 {missing}，文件可能不是 titanic.csv")
print(f"[check] {len(cols)} 列 OK: {', '.join(cols[:6])} ...")
PY
fi

# 数据不入库（与课表 W1 的 .gitignore 约定一致）
GITIGNORE="$PROJECT_ROOT/.gitignore"
touch "$GITIGNORE"
grep -qxF 'data/raw/' "$GITIGNORE" || { echo 'data/raw/' >> "$GITIGNORE"; echo "[git ] 已把 data/raw/ 写入 $GITIGNORE"; }

echo
echo "完成。以后直接跑："
echo "  cd $PROJECT_ROOT && uv run python scripts/w02_pandas_groupby.py"
