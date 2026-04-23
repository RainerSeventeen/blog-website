import { glob, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import matter from "gray-matter";

export interface RecentNotePreview {
  title: string;
  href: string;
  category: string;
  pubDate: string;
  description: string;
}

const NOTE_SITE_ORIGIN = "https://note.rainerseventeen.com";
const noteContentDir = fileURLToPath(
  new URL("../../../../content/note/", import.meta.url),
);

function shouldSkipFile(relativePath: string) {
  const basename = path.basename(relativePath).toLowerCase();
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

function buildNoteHref(relativePath: string) {
  const withoutExtension = relativePath.replace(/\.(md|mdx)$/i, "");
  const normalizedPath = withoutExtension.split(path.sep).join("/");
  return `${NOTE_SITE_ORIGIN}/${normalizedPath}/`;
}

export async function getRecentNotes(limit = 4): Promise<RecentNotePreview[]> {
  const files = await Array.fromAsync(
    glob("**/*.{md,mdx}", { cwd: noteContentDir }),
  );

  const notes = await Promise.all(
    files.map(async (relativePath) => {
      if (shouldSkipFile(relativePath)) return null;

      const absolutePath = path.join(noteContentDir, relativePath);
      const raw = await readFile(absolutePath, "utf-8");
      const { data } = matter(raw);

      if (data.draft === true || !data.pubDate) return null;

      const pubDate = new Date(String(data.pubDate));
      if (Number.isNaN(pubDate.getTime())) return null;

      return {
        title: sanitizeText(data.title) || "Untitled",
        href: buildNoteHref(relativePath),
        category: sanitizeText(data.category) || "note",
        pubDate: pubDate.toISOString(),
        description: sanitizeText(data.description),
      } satisfies RecentNotePreview;
    }),
  );

  return notes
    .filter((note): note is RecentNotePreview => note !== null)
    .sort((a, b) => new Date(b.pubDate).getTime() - new Date(a.pubDate).getTime())
    .slice(0, limit);
}
