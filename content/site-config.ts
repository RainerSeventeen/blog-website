export type TopDomain = "project" | "note" | "lab";
export type SiteKey = "main" | TopDomain;

type PublicSiteEnvKey =
	| "PUBLIC_MAIN_SITE_URL"
	| "PUBLIC_NOTE_SITE_URL"
	| "PUBLIC_PROJECT_SITE_URL"
	| "PUBLIC_LAB_SITE_URL";

export type SiteEnv = Partial<Record<PublicSiteEnvKey, string>>;

interface BaseSiteConfig {
	label: string;
	siteTitle: string;
	description: string;
	cardDescription: string;
	domainLabel: string;
	eyebrow: string;
	defaultHref: string;
	available: boolean;
	homeCtaLabel: string;
	navCtaLabel: string;
	status?: string;
}

export interface SiteConfig extends Omit<BaseSiteConfig, "defaultHref"> {
	key: SiteKey;
	href: string;
}

const SITE_CONFIG_REGISTRY: Record<SiteKey, BaseSiteConfig> = {
	main: {
		label: "Home",
		siteTitle: "主站",
		description: "个人主站首页，负责内容导航、站点说明与分站入口聚合。",
		cardDescription: "内容导航与总览",
		domainLabel: "rainerseventeen.com",
		eyebrow: "Home Base",
		defaultHref: "/",
		available: true,
		homeCtaLabel: "返回主站",
		navCtaLabel: "返回主站首页",
	},
	note: {
		label: "Note",
		siteTitle: "笔记站",
		description: "当前知识库主内容，按主题分流并进入具体知识树。",
		cardDescription: "内容与知识",
		domainLabel: "note.rainerseventeen.com",
		eyebrow: "Knowledge Base",
		defaultHref: "/",
		available: true,
		homeCtaLabel: "前往笔记站",
		navCtaLabel: "进入知识库首页",
	},
	project: {
		label: "Project",
		siteTitle: "项目站",
		description: "项目记录、构建日志与方案复盘。",
		cardDescription: "构建与作品",
		domainLabel: "project.rainerseventeen.com",
		eyebrow: "Build Log",
		defaultHref: "/project/",
		available: false,
		homeCtaLabel: "前往项目站",
		navCtaLabel: "准备中",
		status: "项目域即将开放",
	},
	lab: {
		label: "Lab",
		siteTitle: "实验站",
		description: "实验、草稿、Demo 与探索性内容。",
		cardDescription: "实验与探索",
		domainLabel: "lab.rainerseventeen.com",
		eyebrow: "Experiments",
		defaultHref: "/lab/",
		available: false,
		homeCtaLabel: "前往实验站",
		navCtaLabel: "准备中",
		status: "实验域即将开放",
	},
};

function resolveSiteUrl(envVar: string | undefined, fallback: string): string {
	return envVar?.trim() || fallback;
}

export function getSiteUrls(env: SiteEnv = import.meta.env): Record<SiteKey, string> {
	return {
		main: resolveSiteUrl(env.PUBLIC_MAIN_SITE_URL, SITE_CONFIG_REGISTRY.main.defaultHref),
		note: resolveSiteUrl(env.PUBLIC_NOTE_SITE_URL, SITE_CONFIG_REGISTRY.note.defaultHref),
		project: resolveSiteUrl(env.PUBLIC_PROJECT_SITE_URL, SITE_CONFIG_REGISTRY.project.defaultHref),
		lab: resolveSiteUrl(env.PUBLIC_LAB_SITE_URL, SITE_CONFIG_REGISTRY.lab.defaultHref),
	};
}

export function getSiteConfig<K extends SiteKey>(
	key: K,
	env: SiteEnv = import.meta.env
): SiteConfig & { key: K } {
	const { defaultHref: _defaultHref, ...config } = SITE_CONFIG_REGISTRY[key];
	return {
		key,
		...config,
		href: getSiteUrls(env)[key],
	};
}

export function isExternalHref(href: string): boolean {
	return /^https?:\/\//.test(href);
}
