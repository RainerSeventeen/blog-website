#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if ! command -v pnpm >/dev/null 2>&1; then
  echo "pnpm 未安装，请先安装 pnpm。"
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "安装依赖..."
  pnpm install
fi

echo "启动 Astro 开发服务器..."
exec pnpm dev --host
