#!/usr/bin/env bash

# Unified environment entrypoint for all local scripts and package commands.
# Source this file before starting Astro, build, preview, or deploy commands.

if [[ -n "${BLOG_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

ENV_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOG_ENV_FILE="${BLOG_ENV_FILE:-$ENV_ROOT_DIR/.env}"

if [[ -f "$BLOG_ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$BLOG_ENV_FILE"
  set +a
fi

export BLOG_ENV_LOADED=1
export BLOG_ENV_FILE
