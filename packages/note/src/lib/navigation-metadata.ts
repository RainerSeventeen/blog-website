import {
	getFallbackNavigationLabel,
	pathNavigationMeta,
	type PathNavigationMeta,
} from "@content/note/_navigation";

export function normalizeNavigationPath(path: string): string {
	const normalized = path.replace(/^\/+|\/+$/g, "");
	if (normalized === "index") return "";
	return normalized.endsWith("/index") ? normalized.slice(0, -6) : normalized;
}

export function getPathNavigationMeta(path: string): PathNavigationMeta | undefined {
	return pathNavigationMeta[normalizeNavigationPath(path)];
}

export function getPathLabel(path: string, fallback?: string): string {
	const normalized = normalizeNavigationPath(path);
	return getPathNavigationMeta(normalized)?.label ?? fallback ?? getFallbackNavigationLabel(normalized);
}

export function getPathDescription(path: string, fallback?: string): string | undefined {
	return getPathNavigationMeta(path)?.description ?? fallback;
}
