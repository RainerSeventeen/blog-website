#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_ROOT="${WEB_ROOT:-/var/www}"

cd "$ROOT_DIR"

echo "==> 拉取最新代码"
git pull origin main

echo "==> 安装依赖"
pnpm install --frozen-lockfile

echo "==> 构建所有子站"
pnpm build:all

echo "==> 同步静态文件"
rsync -a --delete packages/home/dist/    "$WEB_ROOT/home/"
rsync -a --delete packages/note/dist/    "$WEB_ROOT/note/"

# project 站就绪后取消注释
# rsync -a --delete packages/project/dist/ "$WEB_ROOT/project/"

echo "==> 部署完成"
