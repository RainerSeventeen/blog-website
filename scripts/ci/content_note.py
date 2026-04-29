#!/usr/bin/env python3
"""
content/note Markdown 内容检查器
- 检查代码块语言标识符是否被 Shiki 支持
- 检查 KaTeX 数学公式是否能成功渲染

运行方式（从仓库根目录）：python3 scripts/ci/content_note.py
"""

import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CONTENT_DIR = ROOT / "content" / "note"
NOTE_PKG_DIR = ROOT / "packages" / "note"
PNPM_STORE = ROOT / "node_modules" / ".pnpm"

# astro.config.ts expressiveCode.shiki.langAlias 的 key（变更时手动同步）
LANG_ALIASES: set[str] = {"C", "C++", "Cpp", "CPP", "c++", "cplusplus", "Bash", "BASH", "text", "Text", "TEXT"}


# ── Shiki 语言列表 ──────────────────────────────────────────────────────────────

def get_shiki_valid_langs() -> set[str]:
    """
    从 pnpm 虚拟存储中直接解析 shiki 的 langs-bundle-full-*.mjs 文件，
    提取所有合法语言 ID 和别名，再并入 astro.config.ts 里的自定义别名。
    """
    pattern = str(PNPM_STORE / "shiki@*" / "node_modules" / "shiki" / "dist" / "langs-bundle-full-*.mjs")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print("[ERROR] 未找到 shiki langs-bundle-full 文件，请确认已运行 pnpm install", file=sys.stderr)
        sys.exit(2)

    # 选最新版本（按文件名排序取最后）
    content = Path(matches[-1]).read_text(encoding="utf-8")

    lang_ids = set(re.findall(r'"id":\s*"([^"]+)"', content))
    # 提取 aliases 数组中的字符串值
    for alias_block in re.findall(r'"aliases":\s*\[([^\]]*)\]', content):
        for alias in re.findall(r'"([^"]+)"', alias_block):
            lang_ids.add(alias)

    return lang_ids | LANG_ALIASES


# ── KaTeX 批量验证 ──────────────────────────────────────────────────────────────

def validate_katex_batch(items: list[dict]) -> list[str | None]:
    """
    items: [{"expr": str, "display": bool}, ...]
    返回 [error_msg_or_None, ...] 与 items 一一对应。
    通过 Node.js 调用 katex（pnpm store 绝对路径），批量验证。
    """
    if not items:
        return []

    # 找 katex 包路径（packages/note/node_modules/katex 是直接依赖）
    katex_path = NOTE_PKG_DIR / "node_modules" / "katex"
    if not katex_path.exists():
        print(f"[ERROR] 未找到 katex：{katex_path}", file=sys.stderr)
        sys.exit(2)
    katex_index = katex_path / "dist" / "katex.mjs"
    if not katex_index.exists():
        katex_index = katex_path / "dist" / "katex.js"

    payload = [{"expr": it["expr"], "display": it["display"]} for it in items]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        data_path = tmp.name

    node_code = (
        f"import katex from {json.dumps(str(katex_index))};\n"
        "import { readFileSync } from 'fs';\n"
        f"const items = JSON.parse(readFileSync({json.dumps(data_path)}, 'utf8'));\n"
        "const results = items.map(item => {\n"
        "  try {\n"
        "    katex.renderToString(item.expr, { throwOnError: true, strict: false, displayMode: item.display });\n"
        "    return null;\n"
        "  } catch (e) {\n"
        "    return e.message;\n"
        "  }\n"
        "});\n"
        "process.stdout.write(JSON.stringify(results));\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".mjs", delete=False, encoding="utf-8") as tmp:
        tmp.write(node_code)
        script_path = tmp.name

    try:
        result = subprocess.run(
            ["node", script_path],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"[ERROR] KaTeX 验证进程异常:\n{result.stderr}", file=sys.stderr)
            sys.exit(2)
        return json.loads(result.stdout)
    finally:
        os.unlink(data_path)
        os.unlink(script_path)


# ── Markdown 解析 ───────────────────────────────────────────────────────────────

def parse_file(path: Path) -> tuple[list[tuple[int, str]], list[dict]]:
    """
    返回 (shiki_candidates, katex_items)
    shiki_candidates: [(lineno, lang), ...]
    katex_items:      [{"line_no": int, "expr": str, "display": bool}, ...]
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    shiki_candidates: list[tuple[int, str]] = []
    katex_items: list[dict] = []

    in_code_block = False
    block_math_start: int | None = None
    block_math_lines: list[str] = []

    for i, line in enumerate(lines):
        lineno = i + 1
        stripped = line.strip()

        # ── 代码块边界 ──
        if line.startswith("```"):
            if not in_code_block:
                in_code_block = True
                lang_part = line[3:].strip()
                lang = lang_part.split()[0] if lang_part else ""
                if lang:
                    shiki_candidates.append((lineno, lang))
            else:
                in_code_block = False
            continue

        if in_code_block:
            continue

        # ── 块级公式 $$ ... $$ ──
        if block_math_start is None and stripped.startswith("$$"):
            inner = stripped[2:]
            if inner.endswith("$$") and len(inner) >= 2:
                expr = inner[:-2].strip()
                if expr:
                    katex_items.append({"line_no": lineno, "expr": expr, "display": True})
            elif inner == "":
                block_math_start = lineno
                block_math_lines = []
            else:
                block_math_start = lineno
                block_math_lines = [inner]
            continue

        if block_math_start is not None:
            if stripped == "$$":
                expr = "\n".join(block_math_lines)
                katex_items.append({"line_no": block_math_start, "expr": expr, "display": True})
                block_math_start = None
                block_math_lines = []
            else:
                block_math_lines.append(line)
            continue

        # ── 行内公式 $...$ ──
        _extract_inline_math(line, lineno, katex_items)

    return shiki_candidates, katex_items


def _extract_inline_math(line: str, lineno: int, out: list[dict]):
    """从一行中提取所有行内 $...$ 公式（跳过 $$...$$、行内代码）。"""
    line_masked = re.sub(r"`[^`]+`", lambda m: " " * len(m.group()), line)

    pos = 0
    while pos < len(line_masked):
        idx = line_masked.find("$", pos)
        if idx == -1:
            break
        if line_masked[idx : idx + 2] == "$$":
            pos = idx + 2
            continue
        close = line_masked.find("$", idx + 1)
        if close == -1:
            break
        if line_masked[close : close + 2] == "$$":
            pos = close + 2
            continue
        expr = line[idx + 1 : close]
        if expr.strip():
            out.append({"line_no": lineno, "expr": expr, "display": False})
        pos = close + 1


# ── 主流程 ──────────────────────────────────────────────────────────────────────

def main():
    print("正在获取 Shiki 语言列表...", flush=True)
    valid_langs = get_shiki_valid_langs()
    print(f"  支持 {len(valid_langs)} 种语言标识符（含别名）\n", flush=True)

    md_files = sorted(CONTENT_DIR.rglob("*.md")) + sorted(CONTENT_DIR.rglob("*.mdx"))
    if not md_files:
        print("未找到任何 Markdown 文件。")
        return

    all_errors: list[str] = []
    all_katex_items: list[dict] = []

    for fpath in md_files:
        rel = fpath.relative_to(ROOT)
        shiki_candidates, katex_items = parse_file(fpath)

        for lineno, lang in shiki_candidates:
            if lang not in valid_langs:
                all_errors.append(f"[SHIKI] {rel}:{lineno}  unknown lang: `{lang}`")

        for item in katex_items:
            item["file"] = str(rel)
            all_katex_items.append(item)

    if all_katex_items:
        print(f"正在验证 {len(all_katex_items)} 个数学公式...", flush=True)
        katex_results = validate_katex_batch(all_katex_items)
        for item, err in zip(all_katex_items, katex_results):
            if err:
                short_err = err.splitlines()[0]
                all_errors.append(f"[KATEX] {item['file']}:{item['line_no']}  {short_err}")

    print("─" * 50)
    if all_errors:
        for e in all_errors:
            print(e)
        print("─" * 50)
        print(f"✗ {len(all_errors)} 个错误")
        sys.exit(1)
    else:
        print(f"✓ 全部通过（{len(md_files)} 个文件，{len(all_katex_items)} 个公式）")


if __name__ == "__main__":
    main()
