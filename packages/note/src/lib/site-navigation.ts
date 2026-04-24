import { buildContentTree, type ArticleNode, type FolderNode } from "./content-tree";
import { getPathLabel } from "./navigation-metadata";
import { topLevelNoteNavigationSegments } from "@content/note/_navigation";

export type TopDomain = "project" | "note" | "lab";

export interface NavLinkPreview {
	label: string;
	href: string;
	description?: string;
	meta?: string;
}

export interface NavArticlePreview extends NavLinkPreview {
	pubDate?: Date;
	sectionKey: string;
	sectionLabel: string;
}

export interface NavSectionPreview {
	key: string;
	label: string;
	shortLabel: string;
	href: string;
	description: string;
	meta?: string;
	order: number;
	topNav: boolean;
	articleCount: number;
	featuredLinks: NavLinkPreview[];
	recentArticles: NavArticlePreview[];
}

export interface NavFlyoutGroup {
	key: string;
	title: string;
	description?: string;
	links: NavLinkPreview[];
}

export interface NavFlyoutModel {
	eyebrow: string;
	title: string;
	description: string;
	elevatedGroup: NavFlyoutGroup;
	primaryGroups: NavFlyoutGroup[];
	secondaryGroups: NavFlyoutGroup[];
}

export interface TopDomainNavItem {
	key: TopDomain;
	label: string;
	description: string;
	href?: string;
	available: boolean;
	ctaLabel: string;
	ctaHref?: string;
	status?: string;
	sections: NavSectionPreview[];
	recentArticles: NavArticlePreview[];
	flyout?: NavFlyoutModel;
}

export interface RouteNavigationContext {
	domain: TopDomain | null;
	sectionKey: string | null;
}

export interface SiteNavigationModel {
	items: TopDomainNavItem[];
	noteSections: NavSectionPreview[];
	primarySections: NavSectionPreview[];
	overflowSections: NavSectionPreview[];
}

export const MAX_PRIMARY_NAV_SECTIONS = 4;

function resolveSiteUrl(envVar: string | undefined, fallback: string): string {
	return envVar?.trim() || fallback;
}

export const SITE_URLS = {
	main: resolveSiteUrl(import.meta.env.PUBLIC_MAIN_SITE_URL, "/"),
	note: resolveSiteUrl(import.meta.env.PUBLIC_NOTE_SITE_URL, "/"),
	project: resolveSiteUrl(import.meta.env.PUBLIC_PROJECT_SITE_URL, "/project/"),
	lab: resolveSiteUrl(import.meta.env.PUBLIC_LAB_SITE_URL, "/lab/"),
} as const;

const DOMAIN_META: Record<
	TopDomain,
	{
		label: string;
		description: string;
		href?: string;
		available: boolean;
		ctaLabel: string;
		ctaHref?: string;
		status?: string;
	}
> = {
	project: {
		label: "Project",
		description: "项目记录、构建日志与方案复盘。",
		href: SITE_URLS.project,
		available: false,
		ctaLabel: "准备中",
		ctaHref: SITE_URLS.project,
		status: "项目域即将开放",
	},
	note: {
		label: "Note",
		description: "当前知识库主内容，按主题分流并进入具体知识树。",
		href: SITE_URLS.note,
		available: true,
		ctaLabel: "进入知识库首页",
		ctaHref: SITE_URLS.note,
	},
	lab: {
		label: "Lab",
		description: "实验、草稿、Demo 与探索性内容。",
		href: SITE_URLS.lab,
		available: false,
		ctaLabel: "准备中",
		ctaHref: SITE_URLS.lab,
		status: "实验域即将开放",
	},
};

export const NOTE_DOMAIN_ROOT = {
	title: DOMAIN_META.note.label,
	href: DOMAIN_META.note.href ?? "/",
} as const;

export const SECTION_DOMAIN_MAP: Record<string, TopDomain> = Object.fromEntries(
	topLevelNoteNavigationSegments.map((segment) => [segment, "note" as const])
);

let navigationCache: SiteNavigationModel | null = null;

function slugToHref(slug: string): string {
	return slug ? `/${slug}/` : "/";
}

function formatArticleMeta(article: ArticleNode): string | undefined {
	if (!article.pubDate) return undefined;

	return new Intl.DateTimeFormat("zh-CN", {
		month: "short",
		day: "numeric",
	}).format(article.pubDate);
}

function toLinkPreview(
	node: FolderNode | ArticleNode,
	override?: {
		label?: string;
		description?: string;
	}
): NavLinkPreview {
	return {
		label: override?.label ?? node.title,
		href: slugToHref(node.slug),
		description: override?.description ?? node.description ?? (node.type === "folder" ? `浏览 ${node.title} 目录` : undefined),
	};
}

function toArticlePreview(article: ArticleNode): NavArticlePreview {
	const sectionKey = article.slug.split("/")[0] ?? "";
	return {
		label: article.title,
		href: slugToHref(article.slug),
		description: article.description,
		meta: formatArticleMeta(article),
		pubDate: article.pubDate,
		sectionKey,
		sectionLabel: getSectionLabel(sectionKey),
	};
}

function compareSections(a: NavSectionPreview, b: NavSectionPreview): number {
	return a.order - b.order || a.label.localeCompare(b.label, "zh-CN");
}

function buildNoteSectionPreview(root: FolderNode): NavSectionPreview {
	const structureNodes = root.children.filter((child): child is FolderNode => child.type === "folder");
	const previewNodes = (structureNodes.length > 0 ? structureNodes : root.children).slice(0, 6);
	const description = root.description ?? "查看该分区下的全部内容。";

	return {
		key: root.slug,
		label: root.title,
		shortLabel: root.shortLabel ?? root.title,
		href: slugToHref(root.slug),
		description,
		meta: root.meta,
		order: root.order ?? Number.MAX_SAFE_INTEGER,
		topNav: root.topNav ?? true,
		articleCount: root.articleCount,
		featuredLinks: previewNodes.map((child) => toLinkPreview(child)),
		recentArticles: root.recentArticles.slice(0, 4).map(toArticlePreview),
	};
}

function buildSectionBrowseGroups(sections: NavSectionPreview[]): NavFlyoutGroup[] {
	return sections.map((section) => ({
		key: section.key,
		title: section.label,
		description: section.description,
		links: [
			{
				label: `浏览 ${section.label}`,
				href: section.href,
				description: section.description,
				meta: `${section.articleCount} 篇`,
			},
			...section.featuredLinks.slice(0, 3),
		],
	}));
}

function buildNoteFlyout(
	sections: NavSectionPreview[],
	recentArticles: NavArticlePreview[]
): NavFlyoutModel {
	return {
		eyebrow: "Knowledge Domain",
		title: "Note",
		description: "先从知识库主入口进入，再按主题切换到具体分区，最后深入到目录树和文章。",
		elevatedGroup: {
			key: "explore-note",
			title: "Explore Note",
			description: "优先进入最常用的主入口。",
			links: [
				{
					label: "进入知识库首页",
					href: "/",
					description: "从首页查看最近整理、索引入口与全局结构。",
				},
				...sections.map((section) => ({
					label: section.label,
					href: section.href,
					description: section.description,
					meta: `${section.articleCount} 篇`,
				})),
			],
		},
		primaryGroups: buildSectionBrowseGroups(sections),
		secondaryGroups: [
			{
				key: "note-recent",
				title: "More from Note",
				description: "最近更新与继续阅读入口。",
				links: recentArticles.slice(0, 4).map((article) => ({
					label: article.label,
					href: article.href,
					description: article.sectionLabel,
					meta: article.meta,
				})),
			},
			{
				key: "note-sections",
				title: "Browse All Topics",
				description: "快速跳回各主题首页。",
				links: sections.map((section) => ({
					label: section.label,
					href: section.href,
					description: section.description,
					meta: `${section.articleCount} 篇`,
				})),
			},
		],
	};
}

export async function getSiteNavigation(): Promise<SiteNavigationModel> {
	if (navigationCache) return navigationCache;

	const tree = await buildContentTree();
	const noteSections = tree.roots
		.filter((root) => SECTION_DOMAIN_MAP[root.slug] === "note")
		.map(buildNoteSectionPreview)
		.sort(compareSections);
	const topNavSections = noteSections.filter((section) => section.topNav);
	const primarySections = topNavSections.slice(0, MAX_PRIMARY_NAV_SECTIONS);
	const overflowSections = topNavSections.slice(MAX_PRIMARY_NAV_SECTIONS);

	const noteRecentArticles = Array.from(
		noteSections
			.flatMap((section) => section.recentArticles)
			.reduce((acc, article) => acc.set(article.href, article), new Map<string, NavArticlePreview>())
			.values()
	)
		.sort((a, b) => (b.pubDate?.getTime() ?? 0) - (a.pubDate?.getTime() ?? 0))
		.slice(0, 6);

	navigationCache = {
		items: [
			{
				key: "project",
				label: DOMAIN_META.project.label,
				description: DOMAIN_META.project.description,
				href: DOMAIN_META.project.href,
				available: DOMAIN_META.project.available,
				ctaLabel: DOMAIN_META.project.ctaLabel,
				ctaHref: DOMAIN_META.project.ctaHref,
				status: DOMAIN_META.project.status,
				sections: [],
				recentArticles: [],
			},
			{
				key: "note",
				label: DOMAIN_META.note.label,
				description: DOMAIN_META.note.description,
				href: DOMAIN_META.note.href,
				available: DOMAIN_META.note.available,
				ctaLabel: DOMAIN_META.note.ctaLabel,
				ctaHref: DOMAIN_META.note.ctaHref,
				sections: noteSections,
				recentArticles: noteRecentArticles,
				flyout: buildNoteFlyout(noteSections, noteRecentArticles),
			},
			{
				key: "lab",
				label: DOMAIN_META.lab.label,
				description: DOMAIN_META.lab.description,
				href: DOMAIN_META.lab.href,
				available: DOMAIN_META.lab.available,
				ctaLabel: DOMAIN_META.lab.ctaLabel,
				ctaHref: DOMAIN_META.lab.ctaHref,
				status: DOMAIN_META.lab.status,
				sections: [],
				recentArticles: [],
			},
		],
		noteSections,
		primarySections,
		overflowSections,
	};

	return navigationCache;
}

export function getSectionLabel(sectionKey: string): string {
	return getPathLabel(sectionKey, sectionKey);
}

export function isNoteHomeRouteId(routeId: string): boolean {
	return routeId === "" || routeId === "index";
}

export function inferRouteNavigationContext(routeId: string): RouteNavigationContext {
	if (isNoteHomeRouteId(routeId)) {
		return { domain: "note", sectionKey: null };
	}

	if (routeId === "404") {
		return { domain: null, sectionKey: null };
	}

	const sectionKey = routeId.split("/")[0] ?? null;
	if (!sectionKey) {
		return { domain: null, sectionKey: null };
	}

	return {
		domain: SECTION_DOMAIN_MAP[sectionKey] ?? null,
		sectionKey,
	};
}
