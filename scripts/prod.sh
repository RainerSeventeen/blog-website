#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-4321}"

if ! command -v pnpm >/dev/null 2>&1; then
	echo "Error: pnpm is not installed or not in PATH." >&2
	exit 1
fi

cd "$ROOT_DIR"

echo "==> Syncing docs"
pnpm sync:docs

echo "==> Building site"
pnpm astro build

echo "==> Starting Astro preview on http://${HOST}:${PORT}"
exec pnpm astro preview --host "$HOST" --port "$PORT"
