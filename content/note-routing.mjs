import { slug } from "github-slugger";

/** 两站共用的文章 ID，保持 Astro 默认路径规则并支持 frontmatter.slug。 */
export function getNoteId({ entry, data = {} }) {
  if (data.slug) return data.slug;
  return entry
    .replace(/\\/g, "/")
    .replace(/\.[^/.]+$/, "")
    .split("/")
    .map((segment) => slug(segment))
    .join("/")
    .replace(/\/index$/, "");
}
