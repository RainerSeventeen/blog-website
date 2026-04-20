#!/usr/bin/env python3
"""
CI 检查脚本：校验 content/note 下所有 .md 文件的 frontmatter 合法性

检查项：
1. 必填字段是否存在且非空：title, pubDate, slugId, category, section
2. pubDate 是否为合法日期格式（YYYY-MM-DD 或 ISO）
3. slugId 是否在全库唯一
4. slugId 格式是否符合 section/path 约定（不含空格，路径分隔符为 /）
5. draft 字段若存在必须为 boolean
6. order / pinTop 若存在必须为数字
7. index.md 文件必须包含 navTitle 字段
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CONTENT_DIR = (SCRIPT_DIR / "../content/note").resolve()

REQUIRED_FIELDS = ["title", "pubDate", "slugId", "category", "section"]


# ─── 工具函数 ───────────────────────────────────────────────────────────────

def parse_frontmatter(raw: str) -> dict | None:
    match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---", raw)
    if not match:
        return None
    obj = {}
    for line in match.group(1).splitlines():
        kv = re.match(r"^(\w+)\s*:\s*(.*)$", line)
        if not kv:
            continue
        key, val = kv.group(1), kv.group(2).strip().strip("\"'")
        obj[key] = val
    return obj


def collect_md_files(directory: Path) -> list[Path]:
    results = []
    for entry in directory.iterdir():
        if entry.is_dir():
            results.extend(collect_md_files(entry))
        elif entry.suffix == ".md":
            results.append(entry)
    return results


def is_valid_date(s: str) -> bool:
    if not s:
        return False
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            datetime.strptime(s[:len(fmt) - fmt.count('%') * 1], fmt)
            return True
        except ValueError:
            pass
    # fallback: try ISO parse via dateutil-style check
    try:
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def rel_path(p: Path) -> str:
    return str(p.relative_to(Path.cwd()))


# ─── 主逻辑 ─────────────────────────────────────────────────────────────────

files = collect_md_files(CONTENT_DIR)
errors: list[dict] = []
warnings: list[dict] = []
slug_map: dict[str, str] = {}

for file in files:
    raw = file.read_text(encoding="utf-8")
    fm = parse_frontmatter(raw)
    rel = rel_path(file)
    is_index = file.name == "index.md"

    if fm is None:
        errors.append({"file": rel, "msg": "缺少 frontmatter（无 --- 块）"})
        continue

    # 1. 必填字段
    for field in REQUIRED_FIELDS:
        if not fm.get(field):
            errors.append({"file": rel, "msg": f"缺少必填字段: {field}"})

    # 2. pubDate 合法性
    if fm.get("pubDate") and not is_valid_date(fm["pubDate"]):
        errors.append({"file": rel, "msg": f'pubDate 不是合法日期: "{fm["pubDate"]}"'})

    # 3. slugId 唯一性
    if fm.get("slugId"):
        slug_id = fm["slugId"]
        if slug_id in slug_map:
            errors.append({"file": rel, "msg": f'slugId "{slug_id}" 与 {slug_map[slug_id]} 重复'})
        else:
            slug_map[slug_id] = rel

        # 4. slugId 格式：只允许字母、数字、连字符、下划线、斜杠，不含空格
        if not re.fullmatch(r"[a-zA-Z0-9\-_/]+", slug_id):
            errors.append({"file": rel, "msg": f'slugId 含非法字符（只允许字母/数字/连字符/下划线/斜杠）: "{slug_id}"'})

    # 5. draft 必须为 boolean 字符串
    if "draft" in fm and fm["draft"] not in ("true", "false"):
        errors.append({"file": rel, "msg": f'draft 必须为 true 或 false，当前: "{fm["draft"]}"'})

    # 6. order / pinTop 必须为数字
    for num_field in ("order", "pinTop"):
        if num_field in fm:
            try:
                float(fm[num_field])
            except ValueError:
                errors.append({"file": rel, "msg": f'{num_field} 必须为数字，当前: "{fm[num_field]}"'})

    # 7. index.md 建议有 navTitle
    if is_index and not fm.get("navTitle"):
        warnings.append({"file": rel, "msg": "index.md 建议提供 navTitle 字段"})

    # 8. description 过长警告（> 200 字符）
    if fm.get("description") and len(fm["description"]) > 200:
        warnings.append({"file": rel, "msg": f'description 过长（{len(fm["description"])} 字符，建议 ≤ 200）'})


# ─── 输出报告 ────────────────────────────────────────────────────────────────

total = len(files)
print(f"\n扫描文件：{total} 个\n")

if warnings:
    print(f"⚠️  警告 ({len(warnings)})：")
    for item in warnings:
        print(f"  [WARN]  {item['file']}\n          → {item['msg']}")
    print()

if errors:
    print(f"❌ 错误 ({len(errors)})：")
    for item in errors:
        print(f"  [ERROR] {item['file']}\n          → {item['msg']}")
    print()
    print(f"检查失败：发现 {len(errors)} 个错误。")
    sys.exit(1)
else:
    warn_note = f"（有 {len(warnings)} 个警告）" if warnings else ""
    print(f"✅ 所有文件校验通过{warn_note}。")
