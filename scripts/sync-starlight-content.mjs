import { cp, lstat, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { glob } from "node:fs/promises";
import matter from "gray-matter";

const rootDir = path.dirname(fileURLToPath(new URL("../package.json", import.meta.url)));
const sourceDir = path.join(rootDir, "content", "note");
const targetDir = path.join(rootDir, "src", "content", "docs");

// Step 1: 删除并重建 docs 目录
await mkdir(path.dirname(targetDir), { recursive: true });

try {
	const stat = await lstat(targetDir);
	if (stat.isSymbolicLink() || stat.isDirectory()) {
		await rm(targetDir, { recursive: true, force: true });
	}
} catch {}

await cp(sourceDir, targetDir, { recursive: true });

// Step 2: 遍历所有 .md/.mdx 文件，注入 sidebar.label 和 sidebar.order
const files = await Array.fromAsync(
	glob("**/*.{md,mdx}", { cwd: targetDir })
);

for (const relPath of files) {
	const filePath = path.join(targetDir, relPath);
	const raw = await readFile(filePath, "utf-8");

	let parsed;
	try {
		parsed = matter(raw);
	} catch {
		continue;
	}

	const { data, content } = parsed;
	let changed = false;

	// 注入 sidebar.label（来自 navTitle）
	if (data.navTitle) {
		if (!data.sidebar) data.sidebar = {};
		if (data.sidebar.label !== data.navTitle) {
			data.sidebar.label = data.navTitle;
			changed = true;
		}
	}

	// 注入 sidebar.order（来自 order）
	if (typeof data.order === "number") {
		if (!data.sidebar) data.sidebar = {};
		if (data.sidebar.order !== data.order) {
			data.sidebar.order = data.order;
			changed = true;
		}
	}

	if (changed) {
		await writeFile(filePath, matter.stringify(content, data), "utf-8");
	}
}

console.log(`[sync] Synced ${files.length} files from content/note → src/content/docs`);
