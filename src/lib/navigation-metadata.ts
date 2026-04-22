import navigationMeta from "../../content/note/_navigation.json";
import { getRouteLabel } from "./route-labels";

export interface PathNavigationMeta {
	label?: string;
	shortLabel?: string;
	description?: string;
	directorySummary?: string;
	meta?: string;
	order?: number;
	topNav?: boolean;
}

interface NavigationMetaFile {
	nodes?: Record<string, PathNavigationMeta>;
}

const PATH_NAVIGATION_META = navigationMeta as NavigationMetaFile;

export function normalizeNavigationPath(path: string): string {
	const normalized = path.replace(/^\/+|\/+$/g, "");
	if (normalized === "index") return "";
	return normalized.endsWith("/index") ? normalized.slice(0, -6) : normalized;
}

export function getPathNavigationMeta(path: string): PathNavigationMeta | undefined {
	return PATH_NAVIGATION_META.nodes?.[normalizeNavigationPath(path)];
}

export function getPathLabel(path: string, fallback?: string): string {
	const normalized = normalizeNavigationPath(path);
	return getPathNavigationMeta(normalized)?.label ?? fallback ?? getRouteLabel(normalized);
}

export function getPathDescription(path: string, fallback?: string): string | undefined {
	return getPathNavigationMeta(path)?.description ?? fallback;
}
