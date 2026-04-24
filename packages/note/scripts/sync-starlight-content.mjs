import { cp, lstat, mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
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

async function directoryHasIndex(dir) {
	const entries = await readdir(dir, { withFileTypes: true });
	return entries.some(
		(entry) =>
			entry.isFile() &&
			(entry.name.toLowerCase() === "index.md" || entry.name.toLowerCase() === "index.mdx")
	);
}

async function directoryHasContent(dir) {
	const entries = await readdir(dir, { withFileTypes: true });
	for (const entry of entries) {
		const entryPath = path.join(dir, entry.name);
		if (entry.isDirectory() && (await directoryHasContent(entryPath))) return true;
		if (entry.isFile() && /\.(md|mdx)$/i.test(entry.name) && !/^index\.mdx?$/i.test(entry.name)) {
			return true;
		}
	}
	return false;
}

async function createFallbackDirectoryIndexes(dir = targetDir) {
	const entries = await readdir(dir, { withFileTypes: true });
	for (const entry of entries) {
		if (entry.isDirectory()) {
			await createFallbackDirectoryIndexes(path.join(dir, entry.name));
		}
	}

	if (dir === targetDir) return;
	if (await directoryHasIndex(dir)) return;
	if (!(await directoryHasContent(dir))) return;

	const relPath = path.relative(targetDir, dir);
	const title = relPath.split(path.sep).at(-1) ?? relPath;
	const fallbackIndex = matter.stringify('import DirectoryPage from "@components/DirectoryPage.astro";\n\n<DirectoryPage />\n', {
		title,
		description: "暂无简介",
		directorySummary: "暂无简介",
		generatedDirectoryPage: true,
		lastUpdated: false,
		tableOfContents: false,
	});
	await writeFile(path.join(dir, "index.mdx"), fallbackIndex, "utf-8");
}

await createFallbackDirectoryIndexes();

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
