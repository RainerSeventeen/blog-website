import { execFileSync } from "node:child_process";
import matter from "gray-matter";
import { getSiteConfig } from "../../../../content/site-config";

export interface RecentNotePreview {
  title: string;
  href: string;
  category: string;
  published: string;
  updated: string;
  description: string;
}

const NOTE_SITE_ORIGIN = getSiteConfig("note").href;

function getWorkspaceDir(): string {
  try {
    return execFileSync("git", ["rev-parse", "--show-toplevel"], {
      encoding: "utf-8",
    }).trim();
  } catch {
    return process.cwd();
  }
}

const workspaceDir = getWorkspaceDir();

// Vite 在构建期解析 glob，将文件内容以字符串形式打包进 bundle。
// SSR 运行时直接从内存读取，不依赖运行时文件系统路径。
const rawNoteFiles = import.meta.glob("../../../../content/note/**/*.{md,mdx}", {
  eager: true,
  query: "?raw",
  import: "default",
}) as Record<string, string>;

function shouldSkipFile(filePath: string) {
  const basename = filePath.split("/").pop()?.toLowerCase() ?? "";
  return (
    basename === "index.md" ||
    basename === "index.mdx" ||
    basename === "404.md" ||
    basename === "404.mdx" ||
    basename === "landing.md" ||
    basename === "landing.mdx"
  );
}

function sanitizeText(value: unknown) {
  if (typeof value !== "string") return "";
  return value
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^>\s*/gm, "")
    .replace(/\s+/g, " ")
    .trim();
}

function buildNoteHref(globKey: string) {
  // globKey 形如: ../../../../content/note/deep-learning/papers/lora.md
  const match = globKey.match(/content\/note\/(.+)$/);
  if (!match) return NOTE_SITE_ORIGIN;
  const withoutExtension = match[1].replace(/\.(md|mdx)$/i, "");
  const normalizedOrigin =
    NOTE_SITE_ORIGIN === "/"
      ? ""
      : NOTE_SITE_ORIGIN.replace(/\/+$/, "");
  return `${normalizedOrigin}/${withoutExtension}/`;
}

function getGitCommitDates(globKey: string) {
  const match = globKey.match(/content\/note\/(.+)$/);
  if (!match) return undefined;

  try {
    const stdout = execFileSync(
      "git",
      ["log", "--follow", "--format=%cI", "--", `content/note/${match[1]}`],
      { cwd: workspaceDir, encoding: "utf-8" }
    );
    const dates = stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    const published = dates.at(-1);
    const updated = dates.at(0);
    if (!published || !updated) return undefined;
    return { published, updated };
  } catch {
    return undefined;
  }
}

export function getRecentNotes(limit = 4): RecentNotePreview[] {
  const notes: RecentNotePreview[] = [];

  for (const [filePath, rawContent] of Object.entries(rawNoteFiles)) {
    if (shouldSkipFile(filePath)) continue;
    if (typeof rawContent !== "string") continue;

    const { data } = matter(rawContent);
    const gitDates = getGitCommitDates(filePath);

    if (data.draft === true || !gitDates) continue;

    const updated = new Date(gitDates.updated);
    const published = new Date(gitDates.published);
    if (Number.isNaN(updated.getTime()) || Number.isNaN(published.getTime())) continue;

    notes.push({
      title: sanitizeText(data.title) || "Untitled",
      href: buildNoteHref(filePath),
      category: sanitizeText(data.category) || "note",
      published: published.toISOString(),
      updated: updated.toISOString(),
      description: sanitizeText(data.description),
    });
  }

  return notes
    .sort((a, b) => new Date(b.updated).getTime() - new Date(a.updated).getTime())
    .slice(0, limit);
}
