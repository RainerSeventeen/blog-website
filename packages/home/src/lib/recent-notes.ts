import notes from "../generated/recent-notes.json";
import { getSiteConfig } from "../../../../content/site-config";

export interface RecentNotePreview {
  title: string;
  href: string;
  category: string;
  published: string;
  updated: string;
  description: string;
}

// 摘要和 Git 日期在构建前生成，访问首页时只读取打包后的数据。
export function getRecentNotes(limit = 4): RecentNotePreview[] {
  const origin = getSiteConfig("note").href.replace(/\/+$/, "");
  return notes.slice(0, limit).map((note) => ({ ...note, href: `${origin}${note.href}` }));
}
