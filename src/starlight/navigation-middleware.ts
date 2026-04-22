import config from "virtual:starlight/user-config";
import { defineRouteMiddleware, type StarlightRouteData } from "@astrojs/starlight/route-data";
import { buildContentTree, getDirectoryPageData } from "../lib/content-tree";
import { getSiteNavigation, inferRouteNavigationContext } from "../lib/site-navigation";

type SidebarEntry = StarlightRouteData["sidebar"][number];
type SidebarLink = Extract<SidebarEntry, { type: "link" }>;
type PrevNextLinkConfig = StarlightRouteData["entry"]["data"]["prev"];
type TocConfig = NonNullable<StarlightRouteData["toc"]>;
const PAGE_TITLE_ID = "_top";

async function buildDirectoryHrefs(): Promise<Set<string>> {
	const tree = await buildContentTree();
	const hrefs = new Set<string>();
	for (const slug of tree.foldersByPath.keys()) {
		hrefs.add(`/${slug}/`);
	}
	return hrefs;
}

function getDirectoryPageToc(
	overviewLabel: string,
	hasStructure: boolean
): TocConfig {
	const defaultConfig = { minHeadingLevel: 2, maxHeadingLevel: 3 };
	const tocConfig =
		typeof config.tableOfContents === "object" ? config.tableOfContents : defaultConfig;

	return {
		...tocConfig,
		items: [
			{
				depth: 2,
				slug: PAGE_TITLE_ID,
				text: overviewLabel,
				children: [],
			},
			...(hasStructure
				? [
						{
							depth: 2,
							slug: "directory-structure",
							text: "内容结构",
							children: [],
						},
					]
				: []),
		],
	};
}

async function getNoteHomePageToc(): Promise<TocConfig> {
	const navigation = await getSiteNavigation();
	const defaultConfig = { minHeadingLevel: 2, maxHeadingLevel: 3 };
	const tocConfig =
		typeof config.tableOfContents === "object" ? config.tableOfContents : defaultConfig;

	return {
		...tocConfig,
		items: [
			{
				depth: 2,
				slug: PAGE_TITLE_ID,
				text: "概览",
				children: [],
			},
			...navigation.noteSections.map((section) => ({
				depth: 2,
				slug: `note-section-${section.key}`,
				text: section.label,
				children: [],
			})),
		],
	};
}

function normalizePath(path: string): string {
	return path.endsWith("/") ? path.slice(0, -1) || "/" : path;
}

function isLinkEntry(entry: SidebarEntry): entry is Extract<SidebarEntry, { type: "link" }> {
	return entry.type === "link";
}

function entryBelongsToSection(entry: SidebarEntry, sectionKey: string): boolean {
	if (isLinkEntry(entry)) {
		const prefix = `/${sectionKey}`;
		return normalizePath(entry.href) === prefix || entry.href.startsWith(`${prefix}/`);
	}

	return entry.entries.some((child) => entryBelongsToSection(child, sectionKey));
}

function pruneSidebarEntry(entry: SidebarEntry, sectionKey: string): SidebarEntry | undefined {
	if (isLinkEntry(entry)) {
		return entryBelongsToSection(entry, sectionKey) ? entry : undefined;
	}

	const entries = entry.entries
		.map((child) => pruneSidebarEntry(child, sectionKey))
		.filter((child): child is SidebarEntry => Boolean(child));

	if (entries.length === 0) return undefined;
	return { ...entry, entries };
}

function unwrapSingleTopLevelGroup(entries: SidebarEntry[]): SidebarEntry[] {
	if (entries.length === 1 && entries[0].type === "group") {
		return entries[0].entries;
	}

	return entries;
}

function flattenSidebar(entries: SidebarEntry[], skip?: Set<string>): SidebarLink[] {
	return entries.flatMap((entry) => {
		if (entry.type === "group") return flattenSidebar(entry.entries, skip);
		return skip?.has(entry.href) ? [] : [entry];
	});
}

function applyPrevNextLinkConfig(
	link: SidebarLink | undefined,
	paginationEnabled: boolean,
	linkConfig: PrevNextLinkConfig
): SidebarLink | undefined {
	if (linkConfig === false) return undefined;
	if (linkConfig === true) return link;

	if (typeof linkConfig === "string" && link) {
		return { ...link, label: linkConfig };
	}

	if (linkConfig && typeof linkConfig === "object") {
		if (link) {
			return {
				...link,
				label: linkConfig.label ?? link.label,
				href: linkConfig.link ?? link.href,
				attrs: {},
			};
		}

		if (linkConfig.link && linkConfig.label) {
			return {
				type: "link",
				label: linkConfig.label,
				href: linkConfig.link,
				badge: undefined,
				isCurrent: false,
				attrs: {},
			};
		}
	}

	return paginationEnabled ? link : undefined;
}

function computePagination(
	entries: SidebarEntry[],
	paginationEnabled: boolean,
	linkConfig: {
		prev?: PrevNextLinkConfig;
		next?: PrevNextLinkConfig;
	},
	directoryHrefs?: Set<string>
): StarlightRouteData["pagination"] {
	const flatLinks = flattenSidebar(entries, directoryHrefs);
	const currentIndex = flatLinks.findIndex((entry) => entry.isCurrent);
	const prev = applyPrevNextLinkConfig(
		flatLinks[currentIndex - 1],
		paginationEnabled,
		linkConfig.prev
	);
	const next = applyPrevNextLinkConfig(
		currentIndex > -1 ? flatLinks[currentIndex + 1] : undefined,
		paginationEnabled,
		linkConfig.next
	);
	return { prev, next };
}

export const onRequest = defineRouteMiddleware(async (context, next) => {
	const route = context.locals.starlightRoute;
	const { domain, sectionKey } = inferRouteNavigationContext(route.entry.id);
	const directory = await getDirectoryPageData(route.entry.id);
	const directoryHrefs = await buildDirectoryHrefs();

	if (route.entry.id === "index") {
		route.toc = await getNoteHomePageToc();
	}

	if (!route.hasSidebar || domain !== "note" || !sectionKey) {
		await next();
		return;
	}

	const filteredSidebar = unwrapSingleTopLevelGroup(
		route.sidebar
			.map((entry) => pruneSidebarEntry(entry, sectionKey))
			.filter((entry): entry is SidebarEntry => Boolean(entry))
	);

	if (filteredSidebar.length > 0) {
		route.sidebar = filteredSidebar;
		route.pagination = computePagination(filteredSidebar, config.pagination, {
			prev: route.entry.data.prev,
			next: route.entry.data.next,
		}, directoryHrefs);
	}

	if (directory) {
		route.toc = getDirectoryPageToc(
			context.locals.t("tableOfContents.overview"),
			directory.folder.children.length > 0
		);
	}

	await next();
});
