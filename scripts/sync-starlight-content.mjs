import { cp, lstat, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.dirname(fileURLToPath(new URL("../package.json", import.meta.url)));
const sourceDir = path.join(rootDir, "content", "note");
const targetDir = path.join(rootDir, "src", "content", "docs");

await mkdir(path.dirname(targetDir), { recursive: true });

try {
	const stat = await lstat(targetDir);
	if (stat.isSymbolicLink() || stat.isDirectory()) {
		await rm(targetDir, { recursive: true, force: true });
	}
} catch {}

await cp(sourceDir, targetDir, { recursive: true });
