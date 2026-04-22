import path from "node:path";
import { getCollection } from "astro:content";
import { getPathLabel, getPathNavigationMeta } from "./navigation-metadata";
import { topLevelNoteNavigationSegments } from "@content/note/_navigation";

export interface Ancestor {
	slug: string;
	title: string;
}

export interface ArticleNode {
	type: "article";
	slug: string;
	title: string;
	sourceExt?: string;
	navTitle?: string;
	order?: number;
	pubDate?: Date;
	description?: string;
	ancestors: Ancestor[];
}

export interface FolderNode {
	type: "folder";
	slug: string; // e.g. "deep-learning/training"
	title: string; // navTitle || dirname
	children: (FolderNode | ArticleNode)[];
	articleCount: number; // recursive total
	order?: number;
	ancestors: Ancestor[];
	description?: string;
	directorySummary?: string;
	shortLabel?: string;
	meta?: string;
	topNav?: boolean;
	recentArticles: ArticleNode[];
	lastUpdated?: Date;
}

export interface ContentTree {
	foldersByPath: Map<string, FolderNode>;
	articlesByPath: Map<string, ArticleNode>;
	roots: FolderNode[];
}

function segmentToTitle(segment: string, slugPath?: string): string {
	return getPathLabel(slugPath ?? segment);
}

export interface DirectoryPageData {
	folder: FolderNode;
	childFolders: FolderNode[];
	directArticles: ArticleNode[];
	recentArticles: ArticleNode[];
	lastUpdated?: Date;
	childFolderCount: number;
}

export function normalizeDirectorySlug(routeId: string): string {
	if (routeId === "index") return "";
	return routeId.endsWith("/index") ? routeId.slice(0, -6) : routeId;
}

/** Sort children: by order (asc) first, then by pubDate (desc) for articles, then by title */
function sortChildren(children: (FolderNode | ArticleNode)[]): (FolderNode | ArticleNode)[] {
	return [...children].sort((a, b) => {
		const aOrder = a.order ?? Infinity;
		const bOrder = b.order ?? Infinity;
		if (aOrder !== bOrder) return aOrder - bOrder;

		// articles: sort by pubDate desc
		if (a.type === "article" && b.type === "article") {
			const aDate = a.pubDate?.getTime() ?? 0;
			const bDate = b.pubDate?.getTime() ?? 0;
			if (aDate !== bDate) return bDate - aDate;
		}

		return a.title.localeCompare(b.title);
	});
}

/** Collect all articles recursively from a folder node */
function collectArticles(node: FolderNode): ArticleNode[] {
	const articles: ArticleNode[] = [];
	for (const child of node.children) {
		if (child.type === "article") {
			articles.push(child);
		} else {
			articles.push(...collectArticles(child));
		}
	}
	return articles;
}

function buildAncestorsFromSlug(
	slug: string,
	foldersByPath: Map<string, FolderNode>,
	includeSelf = false
): Ancestor[] {
	if (!slug) return [];

	const segments = slug.split("/");
	const maxIndex = includeSelf ? segments.length : segments.length - 1;
	const ancestors: Ancestor[] = [];

	for (let i = 0; i < maxIndex; i++) {
		const ancestorSlug = segments.slice(0, i + 1).join("/");
		const ancestorFolder = foldersByPath.get(ancestorSlug);
		ancestors.push({
			slug: ancestorSlug,
			title: ancestorFolder?.title ?? getPathLabel(ancestorSlug),
		});
	}

	return ancestors;
}

function applyFolderMetadata(
	folder: FolderNode,
	slugPath: string,
	overrides?: {
		title?: string;
		description?: string;
		directorySummary?: string;
		order?: number;
	}
): void {
	const meta = getPathNavigationMeta(slugPath);

	folder.title = meta?.label ?? overrides?.title ?? folder.title;
	folder.shortLabel = meta?.shortLabel ?? folder.shortLabel;
	folder.description = meta?.description ?? overrides?.description ?? folder.description;
	folder.directorySummary = meta?.directorySummary ?? overrides?.directorySummary ?? folder.directorySummary;
	folder.meta = meta?.meta ?? folder.meta;
	folder.order = meta?.order ?? overrides?.order ?? folder.order;
	folder.topNav = meta?.topNav ?? folder.topNav;
}

let _cache: ContentTree | null = null;

export async function buildContentTree(): Promise<ContentTree> {
	if (_cache) return _cache;

	const entries = await getCollection("docs");
	const rootIndexEntry = entries.find(
		(entry) =>
			normalizeDirectorySlug(entry.id) === "" &&
			/(?:^|\/)index\.mdx?$/.test(entry.filePath ?? "")
	);

	const foldersByPath = new Map<string, FolderNode>();
	const articlesByPath = new Map<string, ArticleNode>();
	const rootFolder: FolderNode = {
		type: "folder",
		slug: "",
		title: String(rootIndexEntry?.data.navTitle ?? rootIndexEntry?.data.title ?? "Note"),
		children: [],
		articleCount: 0,
		ancestors: [],
		description: rootIndexEntry?.data.description as string | undefined,
		directorySummary: rootIndexEntry?.data.directorySummary as string | undefined,
		recentArticles: [],
	};
	foldersByPath.set("", rootFolder);

	// Create folder nodes for every path segment
	function ensureFolder(slugPath: string, ancestors: Ancestor[]): FolderNode {
		if (foldersByPath.has(slugPath)) {
			const folder = foldersByPath.get(slugPath)!;
			applyFolderMetadata(folder, slugPath);
			return folder;
		}

		const segments = slugPath.split("/");
		const segment = segments[segments.length - 1];

		const node: FolderNode = {
			type: "folder",
			slug: slugPath,
			title: segmentToTitle(segment, slugPath),
			children: [],
			articleCount: 0,
			ancestors,
			recentArticles: [],
		};
		applyFolderMetadata(node, slugPath);
		foldersByPath.set(slugPath, node);
		return node;
	}

	// First pass: build folder hierarchy and article nodes
	for (const entry of entries) {
		// entry.id is like "deep-learning/training/lora-paper" (no .md extension in Starlight)
		const id = entry.id; // e.g. "deep-learning/training/lora-paper"
		const segments = id.split("/");
		const isDirectoryIndex = /(?:^|\/)index\.mdx?$/.test(entry.filePath ?? "");

		// Skip root-level files (404, index)
		if (segments.length === 1) continue;

		if (isDirectoryIndex) {
			const folderSlug = id;
			const folderAncestors = buildAncestorsFromSlug(folderSlug, foldersByPath);
			const folder = ensureFolder(folderSlug, folderAncestors);
			applyFolderMetadata(folder, folderSlug, {
				title: (entry.data.navTitle ?? entry.data.title) as string,
				description: entry.data.description as string | undefined,
				directorySummary: entry.data.directorySummary as string | undefined,
				order: entry.data.order as number | undefined,
			});
			continue;
		}

		// Build ancestor chain for this entry
		const ancestors: Ancestor[] = [];
		for (let i = 0; i < segments.length - 1; i++) {
			const folderSlug = segments.slice(0, i + 1).join("/");
			const folderAncestors: Ancestor[] = segments.slice(0, i).map((_, j) => ({
				slug: segments.slice(0, j + 1).join("/"),
				title: segmentToTitle(segments[j], segments.slice(0, j + 1).join("/")),
			}));
			const folder = ensureFolder(folderSlug, folderAncestors);
			ancestors.push({ slug: folder.slug, title: folder.title });
		}

		// Create article node
		const articleTitle = (entry.data.navTitle ?? entry.data.title) as string;
		const articleNode: ArticleNode = {
				type: "article",
				slug: id,
			title: articleTitle,
			sourceExt: path.extname(entry.filePath ?? "") || undefined,
			navTitle: entry.data.navTitle as string | undefined,
			order: entry.data.order as number | undefined,
			pubDate: entry.data.pubDate as Date | undefined,
			description: entry.data.description as string | undefined,
			ancestors,
		};
		articlesByPath.set(id, articleNode);

		// Add article to its immediate parent folder
		const parentSlug = segments.slice(0, -1).join("/");
		const parentFolder = ensureFolder(parentSlug, ancestors.slice(0, -1));
		if (!parentFolder.children.find((c) => c.slug === id)) {
			parentFolder.children.push(articleNode);
		}
	}

	// Second pass: build folder hierarchy (folders as children of parent folders)
	for (const [slug, folder] of foldersByPath) {
		const segments = slug.split("/");
		if (segments.length <= 1) continue; // top-level folders have no parent folder

		const parentSlug = segments.slice(0, -1).join("/");
		if (foldersByPath.has(parentSlug)) {
			const parent = foldersByPath.get(parentSlug)!;
			if (!parent.children.find((c) => c.slug === slug)) {
				parent.children.push(folder);
			}
		}
	}

	// Third pass: sort children and compute articleCount + recentArticles
	for (const folder of foldersByPath.values()) {
		folder.children = sortChildren(folder.children);
		const allArticles = collectArticles(folder);
		folder.articleCount = allArticles.length;
		const recentArticles = [...allArticles].sort(
			(a, b) => (b.pubDate?.getTime() ?? 0) - (a.pubDate?.getTime() ?? 0)
		);
		folder.recentArticles = recentArticles.slice(0, 5);
		folder.lastUpdated = recentArticles[0]?.pubDate;
	}

	for (const folder of foldersByPath.values()) {
		folder.ancestors = buildAncestorsFromSlug(folder.slug, foldersByPath);
	}

	for (const article of articlesByPath.values()) {
		article.ancestors = buildAncestorsFromSlug(article.slug, foldersByPath);
	}

	// Identify roots (top-level folders = 1 segment slug)
	const roots: FolderNode[] = [];
	const rootOrder = topLevelNoteNavigationSegments;
	for (const name of rootOrder) {
		if (foldersByPath.has(name)) {
			roots.push(foldersByPath.get(name)!);
		}
	}
	// Any remaining top-level folders not in the order list
	for (const [slug, folder] of foldersByPath) {
		if (slug !== "" && !slug.includes("/") && !rootOrder.includes(slug)) {
			roots.push(folder);
		}
	}
	rootFolder.children = sortChildren(roots);
	rootFolder.articleCount = roots.reduce((count, folder) => count + folder.articleCount, 0);
	rootFolder.recentArticles = roots
		.flatMap((folder) => folder.recentArticles)
		.sort((a, b) => (b.pubDate?.getTime() ?? 0) - (a.pubDate?.getTime() ?? 0))
		.slice(0, 5);
	rootFolder.lastUpdated = rootFolder.recentArticles[0]?.pubDate;

	_cache = { foldersByPath, articlesByPath, roots };
	return _cache;
}

export async function getDirectoryPageData(
	routeId: string
): Promise<DirectoryPageData | undefined> {
	const tree = await buildContentTree();
	const folderSlug = normalizeDirectorySlug(routeId);
	const folder = tree.foldersByPath.get(folderSlug);
	if (!folder) return undefined;

	const childFolders = folder.children.filter((child): child is FolderNode => child.type === "folder");
	const directArticles = folder.children.filter(
		(child): child is ArticleNode => child.type === "article"
	);

	return {
		folder,
		childFolders,
		directArticles,
		recentArticles: folder.recentArticles,
		lastUpdated: folder.lastUpdated,
		childFolderCount: childFolders.length,
	};
}
