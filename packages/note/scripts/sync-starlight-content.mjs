import { cp, lstat, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { glob } from "node:fs/promises";
import matter from "gray-matter";

// packages/note 包目录
const pkgDir = path.dirname(fileURLToPath(new URL("../package.json", import.meta.url)));
// workspace 根目录（monorepo 根）
const workspaceDir = path.resolve(pkgDir, "../..");
const sourceDir = path.join(workspaceDir, "content", "note");
const targetDir = path.join(pkgDir, "src", "content", "docs");

// Step 1: 删除并重建 docs 目录
await mkdir(path.dirname(targetDir), { recursive: true });

try {
	const stat = await lstat(targetDir);
	if (stat.isSymbolicLink() || stat.isDirectory()) {
		await rm(targetDir, { recursive: true, force: true });
	}
} catch {}

await cp(sourceDir, targetDir, { recursive: true });
await rm(path.join(targetDir, "_navigation.ts"), { force: true });
await rm(path.join(targetDir, "_navigation.json"), { force: true });

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

	// 注入 title（若缺失，从内容中第一个 # 标题提取；再没有则 fallback 到文件名）
	if (!data.title) {
		const match = content.match(/^#\s+(.+)$/m);
		if (match) {
			data.title = match[1].trim();
			changed = true;
		} else {
			const basename = path.basename(filePath, path.extname(filePath));
			data.title = basename;
			changed = true;
			console.warn(`[sync] WARNING: no title or h1 found, using filename as title: ${relPath}`);
		}
	}

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
