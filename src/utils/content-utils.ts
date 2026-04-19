import { getCollection, type CollectionEntry } from "astro:content";

export type BlogEntry = CollectionEntry<"blog">;

export type SectionSummary = {
  name: string;
  count: number;
  href: string;
};

export type BreadcrumbItem = {
  label: string;
  href: string;
};

const defaultFilter = ({ data }: BlogEntry) => {
  return import.meta.env.PROD ? data.draft !== true : true;
};

const defaultSort = (a: BlogEntry, b: BlogEntry) => {
  return b.data.pubDate.valueOf() - a.data.pubDate.valueOf();
};

export async function getBlogEntries(
  filter?: (entry: BlogEntry) => boolean | undefined,
  sort?: (a: BlogEntry, b: BlogEntry) => number,
): Promise<BlogEntry[]> {
  const entries = await getCollection("blog", filter || defaultFilter);
  return entries.sort(sort || defaultSort);
}

export function getEntrySection(entry: BlogEntry): string {
  return entry.data.section;
}

export function getEntrySegments(entry: BlogEntry): string[] {
  return entry.id.split("/").filter(Boolean);
}

export function getFolderPath(entry: BlogEntry): string {
  return entry.id.split("/").slice(0, -1).join("/");
}

export function getCategoryLabel(entry: BlogEntry): string {
  return entry.data.category;
}

export async function getSectionSummaries(): Promise<SectionSummary[]> {
  const entries = await getBlogEntries();
  const counts = new Map<string, number>();

  for (const entry of entries) {
    const section = getEntrySection(entry);
    counts.set(section, (counts.get(section) || 0) + 1);
  }

  return [...counts.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([name, count]) => ({
      name,
      count,
      href: `/archives/#section-${name}`,
    }));
}

export async function getSiblingEntries(entry: BlogEntry): Promise<BlogEntry[]> {
  const folderPath = getFolderPath(entry);
  const entries = await getBlogEntries();
  return entries.filter((candidate) => getFolderPath(candidate) === folderPath);
}

export async function getTopLevelEntries(): Promise<SectionSummary[]> {
  return getSectionSummaries();
}

export function getBreadcrumbs(entry: BlogEntry): BreadcrumbItem[] {
  const segments = getEntrySegments(entry);
  const items: BreadcrumbItem[] = [{ label: "Blog", href: "/blog/" }];

  if (segments[0]) {
    items.push({
      label: segments[0],
      href: `/archives/#section-${segments[0]}`,
    });
  }

  if (segments.length > 2) {
    const category = segments[segments.length - 2];
    items.push({
      label: category,
      href: `/archives/#category-${category}`,
    });
  }

  items.push({
    label: entry.data.title,
    href: `/blog/${entry.id}/`,
  });

  return items;
}

export function groupEntriesByYear(entries: BlogEntry[]) {
  return entries.reduce<Record<string, BlogEntry[]>>((acc, entry) => {
    const year = String(entry.data.pubDate.getFullYear());
    acc[year] ||= [];
    acc[year].push(entry);
    return acc;
  }, {});
}

export function groupEntriesBySection(entries: BlogEntry[]) {
  return entries.reduce<Record<string, BlogEntry[]>>((acc, entry) => {
    const section = entry.data.section;
    acc[section] ||= [];
    acc[section].push(entry);
    return acc;
  }, {});
}
