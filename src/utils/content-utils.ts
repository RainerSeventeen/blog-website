import { getCollection, render, type CollectionEntry } from "astro:content";

export type BlogEntry = CollectionEntry<"note">;

export type SectionSummary = {
  name: string;
  count: number;
  href: string;
};

export type BreadcrumbItem = {
  label: string;
  href?: string;
};

export type ContentNodeKind = "folder" | "article";

export type ArticleNode = {
  kind: "article";
  path: string;
  slug: string;
  title: string;
  parentPath: string;
  depth: number;
  entry: BlogEntry;
};

export type FolderNode = {
  kind: "folder";
  path: string;
  slug: string;
  title: string;
  parentPath: string | null;
  depth: number;
  folders: FolderNode[];
  articles: ArticleNode[];
  indexEntry?: BlogEntry;
  articleCount: number;
  articleCountRecursive: number;
};

export type ContentTree = {
  root: FolderNode;
  foldersByPath: Map<string, FolderNode>;
  articlesByPath: Map<string, ArticleNode>;
};

type RenderedBlogEntry = Awaited<ReturnType<typeof render>>;

export type ArticleReading = {
  words: number;
  minutes: number;
};

export type ArticleTopicEntry = {
  id: string;
  title: string;
};

export type ParentLink = {
  href: string;
  label: string;
};

export type DirectoryStat = {
  label: string;
  value: string;
};

export type DirectoryCollectionItem = {
  kind: ContentNodeKind;
  href: string;
  title: string;
  meta?: string;
  description?: string;
};

export type DirectoryCollectionGroup = {
  id: "folders" | "articles";
  title: string;
  items: DirectoryCollectionItem[];
};

export type DirectoryNotes = {
  pubDate: Date;
  Content: RenderedBlogEntry["Content"];
};

export type DirectoryPageModel = {
  kind: "directory";
  path: string;
  pageTitle: string;
  pageDescription: string;
  breadcrumbs: BreadcrumbItem[];
  title: string;
  description: string;
  stats: DirectoryStat[];
  parentLink: ParentLink | null;
  groups: DirectoryCollectionGroup[];
  notes: DirectoryNotes | null;
};

export type ArticlePageModel = {
  kind: "article";
  path: string;
  pageTitle: string;
  pageDescription: string;
  breadcrumbs: BreadcrumbItem[];
  entry: BlogEntry;
  Content: RenderedBlogEntry["Content"];
  headings: RenderedBlogEntry["headings"];
  reading: ArticleReading;
  image: string;
  prev: BlogEntry | null;
  next: BlogEntry | null;
  topicEntries: ArticleTopicEntry[];
  currentEntryId: string;
  parentFolderLink: ParentLink | null;
};

const defaultFilter = ({ data }: BlogEntry) => {
  return import.meta.env.PROD ? data.draft !== true : true;
};

const defaultSort = (a: BlogEntry, b: BlogEntry) => {
  return b.data.pubDate.valueOf() - a.data.pubDate.valueOf();
};

let contentTreePromise: Promise<ContentTree> | null = null;
let blogEntriesPromise: Promise<BlogEntry[]> | null = null;

async function getDefaultBlogEntries(): Promise<BlogEntry[]> {
  if (!blogEntriesPromise) {
    blogEntriesPromise = getCollection("note", defaultFilter).then((entries) =>
      [...entries].sort(defaultSort),
    );
  }

  return blogEntriesPromise;
}

export async function getBlogEntries(
  filter?: (entry: BlogEntry) => boolean | undefined,
  sort?: (a: BlogEntry, b: BlogEntry) => number,
): Promise<BlogEntry[]> {
  if (!filter && !sort) {
    return [...(await getDefaultBlogEntries())];
  }

  const entries = await getCollection("note", filter || defaultFilter);
  return [...entries].sort(sort || defaultSort);
}

export function normalizePath(path: string): string {
  return path
    .trim()
    .split("/")
    .filter(Boolean)
    .join("/");
}

export function splitPath(path: string): string[] {
  const normalizedPath = normalizePath(path);
  return normalizedPath ? normalizedPath.split("/") : [];
}

export function joinPath(...segments: string[]): string {
  return normalizePath(segments.flatMap((segment) => splitPath(segment)).join("/"));
}

export function getParentPath(path: string): string | null {
  const segments = splitPath(path);
  if (segments.length === 0) {
    return null;
  }

  return joinPath(...segments.slice(0, -1));
}

export function getAncestorPaths(path: string): string[] {
  const segments = splitPath(path);
  return segments.map((_, index) => joinPath(...segments.slice(0, index + 1)));
}

export function getPathSlug(path: string): string {
  const segments = splitPath(path);
  return segments.at(-1) || "";
}

export function getBlogPath(path: string): string {
  const normalizedPath = normalizePath(path);
  return normalizedPath ? `/blog/${normalizedPath}/` : "/blog/";
}

export function getEntrySection(entry: BlogEntry): string {
  return entry.data.section;
}

function getEntryPathId(entry: BlogEntry): string {
  const normalizedPath = normalizePath(entry.data.slugId || entry.id);

  if (!normalizedPath) {
    return normalizedPath;
  }

  return normalizedPath;
}

export function getEntrySegments(entry: BlogEntry): string[] {
  return splitPath(getEntryPathId(entry));
}

export function getCategoryLabel(entry: BlogEntry): string {
  return entry.data.category;
}

export function isIndexEntry(entry: BlogEntry): boolean {
  return getPathSlug(getEntryPathId(entry)).toLowerCase() === "index";
}

export function getArticlePath(entry: BlogEntry): string {
  if (isIndexEntry(entry)) {
    throw new Error(`Index entry "${getEntryPathId(entry)}" does not map to an article path.`);
  }

  const articlePath = getEntryPathId(entry);
  if (!articlePath) {
    throw new Error(`Entry "${entry.id}" has an invalid empty article path.`);
  }

  return articlePath;
}

export function getFolderPathFromEntry(entry: BlogEntry): string {
  const normalizedId = getEntryPathId(entry);
  return getParentPath(normalizedId) ?? "";
}

export function getFolderPath(entry: BlogEntry): string {
  return getFolderPathFromEntry(entry);
}

function formatPathLabel(value: string): string {
  return value
    .split(/[-_/]/)
    .filter(Boolean)
    .map((segment) => {
      if (segment.length <= 4 && /^[a-z0-9]+$/i.test(segment)) {
        return segment.toUpperCase();
      }

      return segment.charAt(0).toUpperCase() + segment.slice(1);
    })
    .join(" ");
}

function getEntryNavTitle(entry: BlogEntry): string {
  return entry.data.navTitle?.trim() || entry.data.title;
}

function getFolderDisplayTitle(folder: FolderNode): string {
  if (folder.indexEntry) {
    return getEntryNavTitle(folder.indexEntry);
  }

  return folder.slug ? formatPathLabel(folder.slug) : "内容";
}

function createFolderNode(path: string): FolderNode {
  const normalizedPath = normalizePath(path);

  return {
    kind: "folder",
    path: normalizedPath,
    slug: getPathSlug(normalizedPath),
    title: normalizedPath ? formatPathLabel(getPathSlug(normalizedPath)) : "内容",
    parentPath: getParentPath(normalizedPath),
    depth: splitPath(normalizedPath).length,
    folders: [],
    articles: [],
    articleCount: 0,
    articleCountRecursive: 0,
  };
}

function compareOrder(a?: number, b?: number) {
  const safeA = typeof a === "number" ? a : Number.POSITIVE_INFINITY;
  const safeB = typeof b === "number" ? b : Number.POSITIVE_INFINITY;

  if (safeA !== safeB) {
    return safeA - safeB;
  }

  return 0;
}

function compareFolderNodes(a: FolderNode, b: FolderNode) {
  const orderComparison = compareOrder(a.indexEntry?.data.order, b.indexEntry?.data.order);
  if (orderComparison !== 0) {
    return orderComparison;
  }

  return getFolderDisplayTitle(a).localeCompare(getFolderDisplayTitle(b), "zh-CN", {
    numeric: true,
    sensitivity: "base",
  });
}

function compareArticleNodes(a: ArticleNode, b: ArticleNode) {
  const orderComparison = compareOrder(a.entry.data.order, b.entry.data.order);
  if (orderComparison !== 0) {
    return orderComparison;
  }

  return a.title.localeCompare(b.title, "zh-CN", {
    numeric: true,
    sensitivity: "base",
  });
}

export function buildContentTree(entries: BlogEntry[]): ContentTree {
  const root = createFolderNode("");
  const foldersByPath = new Map<string, FolderNode>([["", root]]);
  const articlesByPath = new Map<string, ArticleNode>();

  const ensureFolderNode = (path: string): FolderNode => {
    const normalizedPath = normalizePath(path);
    const existingFolder = foldersByPath.get(normalizedPath);

    if (existingFolder) {
      return existingFolder;
    }

    if (normalizedPath && articlesByPath.has(normalizedPath)) {
      throw new Error(
        `Path conflict: folder "${normalizedPath}" conflicts with article "${normalizedPath}".`,
      );
    }

    const parentPath = getParentPath(normalizedPath);
    const parentFolder = parentPath === null ? root : ensureFolderNode(parentPath);
    const folder = createFolderNode(normalizedPath);

    parentFolder.folders.push(folder);
    foldersByPath.set(normalizedPath, folder);

    return folder;
  };

  for (const entry of entries) {
    if (isIndexEntry(entry)) {
      const folderPath = normalizePath(getParentPath(getEntryPathId(entry)) ?? "");
      const folder = ensureFolderNode(folderPath);

      if (folder.indexEntry) {
        throw new Error(`Duplicate folder index entry detected at "${folderPath}".`);
      }

      if (folderPath && articlesByPath.has(folderPath)) {
        throw new Error(
          `Path conflict: folder "${folderPath}" conflicts with article "${folderPath}".`,
        );
      }

      folder.indexEntry = entry;
      folder.title = getFolderDisplayTitle(folder);
      continue;
    }

    const articlePath = getArticlePath(entry);

    if (articlesByPath.has(articlePath)) {
      throw new Error(`Duplicate article path detected at "${articlePath}".`);
    }

    if (foldersByPath.has(articlePath)) {
      throw new Error(
        `Path conflict: article "${articlePath}" conflicts with folder "${articlePath}".`,
      );
    }

    const parentPath = getFolderPathFromEntry(entry);
    const parentFolder = ensureFolderNode(parentPath);
    const articleNode: ArticleNode = {
      kind: "article",
      path: articlePath,
      slug: getPathSlug(articlePath),
      title: getEntryNavTitle(entry),
      parentPath,
      depth: splitPath(articlePath).length,
      entry,
    };

    parentFolder.articles.push(articleNode);
    articlesByPath.set(articlePath, articleNode);
  }

  const finalizeFolder = (folder: FolderNode): number => {
    folder.folders.forEach((childFolder) => {
      childFolder.title = getFolderDisplayTitle(childFolder);
    });

    folder.folders.sort(compareFolderNodes);
    folder.articles.sort(compareArticleNodes);
    folder.articleCount = folder.articles.length;

    folder.articleCountRecursive = folder.articles.length;
    for (const childFolder of folder.folders) {
      folder.articleCountRecursive += finalizeFolder(childFolder);
    }

    return folder.articleCountRecursive;
  };

  root.title = getFolderDisplayTitle(root);
  finalizeFolder(root);

  return {
    root,
    foldersByPath,
    articlesByPath,
  };
}

export async function getContentTree(): Promise<ContentTree> {
  if (!contentTreePromise) {
    contentTreePromise = getDefaultBlogEntries().then((entries) => buildContentTree(entries));
  }

  return contentTreePromise;
}

export function getFolderNode(tree: ContentTree, path: string): FolderNode | undefined {
  return tree.foldersByPath.get(normalizePath(path));
}

export function getArticleNode(tree: ContentTree, path: string): ArticleNode | undefined {
  return tree.articlesByPath.get(normalizePath(path));
}

export function getFolderChildren(tree: ContentTree, path: string) {
  const folder = getFolderNode(tree, path);
  if (!folder) {
    return undefined;
  }

  return {
    folders: folder.folders,
    articles: folder.articles,
  };
}

type BreadcrumbOptions = {
  includeCurrentLink?: boolean;
};

export function getBreadcrumbsForPath(
  tree: ContentTree,
  path: string,
  options: BreadcrumbOptions = {},
): BreadcrumbItem[] {
  const normalizedPath = normalizePath(path);
  const { includeCurrentLink = false } = options;
  const breadcrumbs: BreadcrumbItem[] = [{ label: "知识库", href: "/archives/" }];

  for (const ancestorPath of getAncestorPaths(normalizedPath)) {
    const folder = getFolderNode(tree, ancestorPath);
    const isCurrent = ancestorPath === normalizedPath;

    breadcrumbs.push({
      label: folder?.title || formatPathLabel(getPathSlug(ancestorPath)),
      href: isCurrent && !includeCurrentLink ? undefined : getBlogPath(ancestorPath),
    });
  }

  return breadcrumbs;
}

function trimOptionalText(value?: string) {
  const normalizedValue = value?.trim();
  return normalizedValue ? normalizedValue : undefined;
}

function buildDirectoryFallbackDescription(folder: FolderNode) {
  if (folder.folders.length > 0 && folder.articles.length > 0) {
    return "选择一个文件夹进行访问，或点击阅读一篇文章。";
  }

  if (folder.folders.length > 0) {
    return "选择一个子目录进行访问。";
  }

  if (folder.articles.length > 0) {
    return "选择一个文章阅读。";
  }

  return "该目录下暂时还没有内容。";
}

function buildDirectoryCollectionGroups(folder: FolderNode): DirectoryCollectionGroup[] {
  const folderItems: DirectoryCollectionItem[] = folder.folders.map((childFolder) => ({
    kind: "folder",
    href: getBlogPath(childFolder.path),
    title: childFolder.title,
    meta: `${childFolder.folders.length} 个子目录 · ${childFolder.articleCount} 篇直接文章 · 共 ${childFolder.articleCountRecursive} 篇内容`,
    description: trimOptionalText(childFolder.indexEntry?.data.description),
  }));

  const articleItems: DirectoryCollectionItem[] = folder.articles.map((article) => ({
    kind: "article",
    href: getBlogPath(article.path),
    title: article.title,
    description: trimOptionalText(article.entry.data.description),
  }));

  return [
    ...(folderItems.length > 0
      ? [{ id: "folders" as const, title: "子目录", items: folderItems }]
      : []),
    ...(articleItems.length > 0
      ? [{ id: "articles" as const, title: "文章", items: articleItems }]
      : []),
  ];
}

function getArticleReading(renderedEntry: RenderedBlogEntry): ArticleReading {
  const frontmatter = renderedEntry.remarkPluginFrontmatter as
    | { words?: number; minutes?: number }
    | undefined;

  return {
    words: frontmatter?.words || 0,
    minutes: frontmatter?.minutes || 0,
  };
}

export async function buildDirectoryPageModel(
  tree: ContentTree,
  path: string,
  siteTitle: string,
): Promise<DirectoryPageModel> {
  const folder = getFolderNode(tree, path);

  if (!folder) {
    throw new Error(`Folder path "${path}" was not found in the content tree.`);
  }

  const description = trimOptionalText(folder.indexEntry?.data.description) || buildDirectoryFallbackDescription(folder);
  const hasNotes = Boolean(folder.indexEntry?.body.trim());
  const renderedNotes = hasNotes && folder.indexEntry ? await render(folder.indexEntry) : null;
  const parentFolder = folder.parentPath ? getFolderNode(tree, folder.parentPath) : null;

  return {
    kind: "directory",
    path: folder.path,
    pageTitle: `${folder.title} - ${siteTitle}`,
    pageDescription: description,
    breadcrumbs: getBreadcrumbsForPath(tree, folder.path),
    title: folder.title,
    description,
    stats: [
      { label: "子目录", value: String(folder.folders.length) },
      { label: "文章", value: String(folder.articles.length) },
      { label: "总内容", value: String(folder.articleCountRecursive) },
    ],
    parentLink:
      folder.parentPath !== null
        ? {
            href: folder.parentPath ? getBlogPath(folder.parentPath) : "/archives/",
            label: parentFolder ? getFolderDisplayTitle(parentFolder) : "知识库",
          }
        : null,
    groups: buildDirectoryCollectionGroups(folder),
    notes:
      folder.indexEntry && renderedNotes
        ? {
            pubDate: folder.indexEntry.data.pubDate,
            Content: renderedNotes.Content,
          }
        : null,
  };
}

export async function buildArticlePageModel(
  tree: ContentTree,
  path: string,
  siteTitle: string,
): Promise<ArticlePageModel> {
  const articleNode = getArticleNode(tree, path);

  if (!articleNode) {
    throw new Error(`Article path "${path}" was not found in the content tree.`);
  }

  const entry = articleNode.entry;
  const renderedArticle = await render(entry);
  const parentFolder = getFolderNode(tree, articleNode.parentPath);
  const articleEntries = (await getBlogEntries()).filter((candidate) => !isIndexEntry(candidate));
  const articleIndex = articleEntries.findIndex(
    (candidate) => getArticlePath(candidate) === articleNode.path,
  );

  return {
    kind: "article",
    path: articleNode.path,
    pageTitle: `${entry.data.title} - ${siteTitle}`,
    pageDescription: entry.data.description,
    breadcrumbs: getBreadcrumbsForPath(tree, articleNode.parentPath, {
      includeCurrentLink: true,
    }),
    entry,
    Content: renderedArticle.Content,
    headings: renderedArticle.headings,
    reading: getArticleReading(renderedArticle),
    image: entry.data.image,
    prev: articleIndex > 0 ? articleEntries[articleIndex - 1] : null,
    next:
      articleIndex >= 0 && articleIndex < articleEntries.length - 1
        ? articleEntries[articleIndex + 1]
        : null,
    topicEntries:
      parentFolder?.articles.map((article) => ({
        id: article.path,
        title: article.title,
      })) || [],
    currentEntryId: articleNode.path,
    parentFolderLink: parentFolder?.path
      ? {
          href: getBlogPath(parentFolder.path),
          label: parentFolder.title,
        }
      : null,
  };
}

export async function getSectionSummaries(): Promise<SectionSummary[]> {
  const tree = await getContentTree();

  return tree.root.folders.map((folder) => ({
    name: folder.title,
    count: folder.articleCountRecursive,
    href: getBlogPath(folder.path),
  }));
}

export async function getSiblingEntries(entry: BlogEntry): Promise<BlogEntry[]> {
  if (isIndexEntry(entry)) {
    return [];
  }

  const tree = await getContentTree();
  const articleNode = getArticleNode(tree, getArticlePath(entry));
  if (!articleNode) {
    return [];
  }

  const folder = getFolderNode(tree, articleNode.parentPath);
  return folder?.articles.map((article) => article.entry) || [];
}

export async function getTopLevelEntries(): Promise<SectionSummary[]> {
  return getSectionSummaries();
}

export async function getBreadcrumbs(entry: BlogEntry): Promise<BreadcrumbItem[]> {
  const tree = await getContentTree();
  return getBreadcrumbsForPath(tree, getFolderPathFromEntry(entry));
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
