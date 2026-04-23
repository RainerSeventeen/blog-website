#!/usr/bin/env bash

# 本地生产预览入口（build → preview）
# 用法: ./scripts/test.sh [site...]
#   ./scripts/test.sh          → 构建并启动所有已就绪的站
#   ./scripts/test.sh home     → 只构建并启动 home
#   ./scripts/test.sh home note → 构建并启动 home 和 note

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDS=()

# ── 站点定义（与 dev.sh 保持一致）────────────────────────────
# 格式: "name|port|subdomain|ready"
declare -a SITES=(
  "home|4320|xxx.com|1"
  "note|4321|note.xxx.com|1"
  "project|4322|project.xxx.com|0"
  "lab|4323|lab.xxx.com|0"
)

# ── 工具函数 ──────────────────────────────────────────────
COLOR_RESET="\033[0m"
COLOR_BOLD="\033[1m"
COLOR_GREEN="\033[32m"
COLOR_YELLOW="\033[33m"
COLOR_CYAN="\033[36m"
COLOR_DIM="\033[2m"
COLOR_RED="\033[31m"

cleanup() {
  echo ""
  echo -e "${COLOR_DIM}正在关闭所有预览服务器...${COLOR_RESET}"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo -e "${COLOR_DIM}已退出。${COLOR_RESET}"
}
trap cleanup SIGINT SIGTERM EXIT

# ── 解析目标站点 ───────────────────────────────────────────
TARGETS=("$@")

should_run() {
  local name="$1"
  local ready="$2"
  [[ "$ready" -ne 1 ]] && return 1
  if [[ "${#TARGETS[@]}" -eq 0 ]]; then
    return 0
  fi
  for t in "${TARGETS[@]}"; do
    [[ "$t" == "$name" ]] && return 0
  done
  return 1
}

# ── 打印站点总览 ───────────────────────────────────────────
echo ""
echo -e "${COLOR_BOLD}  本地生产预览环境${COLOR_RESET}"
echo -e "  ─────────────────────────────────────"

for site in "${SITES[@]}"; do
  IFS='|' read -r name port subdomain ready <<< "$site"
  local_url="http://localhost:${port}"
  if [[ "$ready" -eq 1 ]]; then
    echo -e "  ${COLOR_GREEN}●${COLOR_RESET} ${COLOR_BOLD}${name}${COLOR_RESET}    ${COLOR_CYAN}${local_url}${COLOR_RESET}  ${COLOR_DIM}(生产: ${subdomain})${COLOR_RESET}"
  else
    echo -e "  ${COLOR_DIM}○ ${name}    ${local_url}  (准备中)${COLOR_RESET}"
  fi
done
echo -e "  ─────────────────────────────────────"
echo ""

# ── 构建阶段（串行，保证日志清晰）────────────────────────────
BUILD_TARGETS=()
for site in "${SITES[@]}"; do
  IFS='|' read -r name port subdomain ready <<< "$site"
  if ! should_run "$name" "$ready"; then
    continue
  fi
  pkg_dir="$ROOT_DIR/packages/$name"
  if [[ ! -d "$pkg_dir" ]]; then
    echo -e "${COLOR_YELLOW}  警告: packages/${name} 不存在，跳过${COLOR_RESET}"
    continue
  fi
  BUILD_TARGETS+=("$name")
done

if [[ "${#BUILD_TARGETS[@]}" -eq 0 ]]; then
  echo -e "${COLOR_YELLOW}  没有可构建的站点。${COLOR_RESET}"
  echo -e "  用法: $0 [site...]  例如: $0 note"
  exit 1
fi

for name in "${BUILD_TARGETS[@]}"; do
  pkg_dir="$ROOT_DIR/packages/$name"
  echo -e "  ${COLOR_BOLD}[构建]${COLOR_RESET} ${name}..."
  if ! (cd "$pkg_dir" && pnpm build 2>&1 | sed "s/^/    /"); then
    echo -e "  ${COLOR_RED}✗ ${name} 构建失败，中止。${COLOR_RESET}"
    exit 1
  fi
  echo -e "  ${COLOR_GREEN}✓ ${name} 构建完成${COLOR_RESET}"
  echo ""
done

# ── 启动阶段（并行后台）──────────────────────────────────────
for name in "${BUILD_TARGETS[@]}"; do
  pkg_dir="$ROOT_DIR/packages/$name"
  IFS='|' read -r _n port _s _r <<< "$(printf '%s\n' "${SITES[@]}" | grep "^${name}|")"
  echo -e "  ${COLOR_GREEN}启动${COLOR_RESET} ${COLOR_BOLD}${name}${COLOR_RESET} → http://localhost:${port}"
  (cd "$pkg_dir" && pnpm preview 2>&1 | sed "s/^/  [${name}] /") &
  PIDS+=($!)
done

echo ""
echo -e "  ${COLOR_DIM}按 Ctrl+C 停止所有服务器${COLOR_RESET}"
echo ""

# ── 等待所有子进程 ─────────────────────────────────────────
wait
