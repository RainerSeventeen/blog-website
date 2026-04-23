import matter from "gray-matter";

export interface RecentNotePreview {
  title: string;
  href: string;
  category: string;
  pubDate: string;
  description: string;
}

const NOTE_SITE_ORIGIN = "https://note.rainerseventeen.com";

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
  if (!match) return `${NOTE_SITE_ORIGIN}/`;
  const withoutExtension = match[1].replace(/\.(md|mdx)$/i, "");
  return `${NOTE_SITE_ORIGIN}/${withoutExtension}/`;
}

export function getRecentNotes(limit = 4): RecentNotePreview[] {
  const notes: RecentNotePreview[] = [];

  for (const [filePath, rawContent] of Object.entries(rawNoteFiles)) {
    if (shouldSkipFile(filePath)) continue;
    if (typeof rawContent !== "string") continue;

    const { data } = matter(rawContent);

    if (data.draft === true || !data.pubDate) continue;

    const pubDate = new Date(String(data.pubDate));
    if (Number.isNaN(pubDate.getTime())) continue;

    notes.push({
      title: sanitizeText(data.title) || "Untitled",
      href: buildNoteHref(filePath),
      category: sanitizeText(data.category) || "note",
      pubDate: pubDate.toISOString(),
      description: sanitizeText(data.description),
    });
  }

  return notes
    .sort((a, b) => new Date(b.pubDate).getTime() - new Date(a.pubDate).getTime())
    .slice(0, limit);
}
