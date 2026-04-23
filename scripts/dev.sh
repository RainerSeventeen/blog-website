#!/usr/bin/env bash

# 多站本地开发入口
# 用法: ./scripts/dev.sh [site...]
#   ./scripts/dev.sh          → 启动所有已就绪的站
#   ./scripts/dev.sh note     → 只启动 note
#   ./scripts/dev.sh home note → 启动 home 和 note

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDS=()

# ── 站点定义 ──────────────────────────────────────────────
# 格式: "name|port|subdomain|ready"
# ready=1 表示已实现，ready=0 表示占位（仅显示，不启动）
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

cleanup() {
  echo ""
  echo -e "${COLOR_DIM}正在关闭所有开发服务器...${COLOR_RESET}"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo -e "${COLOR_DIM}已退出。${COLOR_RESET}"
}
trap cleanup SIGINT SIGTERM EXIT

# ── 解析目标站点 ───────────────────────────────────────────
TARGETS=("$@")

should_start() {
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
echo -e "${COLOR_BOLD}  本地多站开发环境${COLOR_RESET}"
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

# ── 启动各站 ──────────────────────────────────────────────
STARTED=0
for site in "${SITES[@]}"; do
  IFS='|' read -r name port subdomain ready <<< "$site"

  if ! should_start "$name" "$ready"; then
    continue
  fi

  pkg_dir="$ROOT_DIR/packages/$name"
  if [[ ! -d "$pkg_dir" ]]; then
    echo -e "${COLOR_YELLOW}  警告: packages/${name} 不存在，跳过${COLOR_RESET}"
    continue
  fi

  echo -e "  ${COLOR_GREEN}启动${COLOR_RESET} ${COLOR_BOLD}${name}${COLOR_RESET} → http://localhost:${port}"
  (cd "$pkg_dir" && pnpm dev 2>&1 | sed "s/^/  [${name}] /") &
  PIDS+=($!)
  STARTED=$((STARTED + 1))
done

if [[ "$STARTED" -eq 0 ]]; then
  echo -e "${COLOR_YELLOW}  没有可启动的站点。${COLOR_RESET}"
  echo -e "  用法: $0 [site...]  例如: $0 note"
  exit 1
fi

echo ""
echo -e "  ${COLOR_DIM}按 Ctrl+C 停止所有服务器${COLOR_RESET}"
echo ""

# ── 等待所有子进程 ─────────────────────────────────────────
wait
