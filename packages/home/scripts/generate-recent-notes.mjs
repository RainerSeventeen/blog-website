import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import matter from "gray-matter";
import { getNoteId } from "../../../content/note-routing.mjs";

function shouldSkipFile(filePath) {
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

function sanitizeText(value) {
  if (typeof value !== "string") return "";
  return value
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^>\s*/gm, "")
    .replace(/\s+/g, " ")
    .trim();
}

function extractH1Title(rawContent) {
  const match = rawContent.match(/^#\s+(.+)$/m);
  if (!match) return "";
  return sanitizeText(match[1]);
}

function buildNoteHref(globKey, data) {
  // 路径形如 content/note/deep-learning/papers/lora.md。
  const match = globKey.match(/content\/note\/(.+)$/);
  if (!match) return "/";
  const id = getNoteId({ entry: match[1], data });
  return id === "index" ? "/" : `/${id}/`;
}

function getGitCommitDates(globKey, workspaceDir) {
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

export function collectRecentNotes(workspaceDir) {
  const notes = [];
  const sourceDir = path.join(workspaceDir, "content/note");
  const rawNoteFiles = Object.fromEntries(
    readdirSync(sourceDir, { recursive: true }).filter((file) => /\.(md|mdx)$/i.test(file))
      .map((file) => [`content/note/${file.split(path.sep).join("/")}`, readFileSync(path.join(sourceDir, file), "utf8")])
  );

  for (const [filePath, rawContent] of Object.entries(rawNoteFiles)) {
    if (shouldSkipFile(filePath)) continue;
    if (typeof rawContent !== "string") continue;

    const { data } = matter(rawContent);
    if (data.draft === true) continue;
    const gitDates = getGitCommitDates(filePath, workspaceDir);

    if (!gitDates) continue;

    const updated = new Date(gitDates.updated);
    const published = new Date(gitDates.published);
    if (Number.isNaN(updated.getTime()) || Number.isNaN(published.getTime())) continue;

    notes.push({
      title: sanitizeText(data.title) || extractH1Title(rawContent) || "Untitled",
      href: buildNoteHref(filePath, data),
      category: sanitizeText(data.category) || "note",
      published: published.toISOString(),
      updated: updated.toISOString(),
      description: sanitizeText(data.description),
    });
  }

  return notes
    .sort((a, b) => new Date(b.updated).getTime() - new Date(a.updated).getTime());
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const workspaceDir = fileURLToPath(new URL("../../../", import.meta.url));
  const output = fileURLToPath(new URL("../src/generated/recent-notes.json", import.meta.url));
  const notes = collectRecentNotes(workspaceDir);
  mkdirSync(path.dirname(output), { recursive: true });
  writeFileSync(output, JSON.stringify(notes, null, 2) + "\n");
  console.log(`[recent-notes] Generated ${notes.length} article summaries`);
}
