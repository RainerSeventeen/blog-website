export interface PathNavigationMeta {
	label?: string;
	shortLabel?: string;
	description?: string;
	directorySummary?: string;
	meta?: string;
	order?: number;
	topNav?: boolean;
}

export interface NoteNavigationNode extends PathNavigationMeta {
	segment: string;
	children?: readonly NoteNavigationNode[];
}

function compareByOrder(a: PathNavigationMeta, b: PathNavigationMeta): number {
	return (a.order ?? Number.MAX_SAFE_INTEGER) - (b.order ?? Number.MAX_SAFE_INTEGER);
}

function segmentToTitle(segment: string): string {
	return segment
		.replace(/-/g, " ")
		.replace(/\b\w/g, (char) => char.toUpperCase());
}

export const noteNavigationTree = [
	{
		segment: "deep-learning",
		label: "深度学习",
		shortLabel: "深度学习",
		description: "基础、论文、训练与面试知识的系统整理",
		directorySummary: "基础概念、论文脉络、训练方法和面试题解。",
		meta: "从核心概念到论文脉络，再到面试与工程问题",
		order: 10,
		topNav: true,
		children: [
			{
				segment: "foundations",
				label: "基础概念",
				description: "深度学习基础知识",
			},
			{
				segment: "papers",
				label: "论文精读",
				description: "Transformer、多模态与训练方法的论文整理",
				directorySummary: "按研究方向整理论文阅读笔记，方便先看主题再进入具体论文。",
				children: [
					{
						segment: "transformers",
						label: "Transformer",
					},
					{
						segment: "training",
						label: "训练方法",
					},
					{
						segment: "multimodal",
						label: "多模态",
					},
				],
			},
			{
				segment: "interview",
				label: "面试题",
				description: "覆盖深度学习、工程、RAG 与 Python 一些八股文",
			},
		],
	},
	{
		segment: "code-algorithm",
		label: "代码算法",
		shortLabel: "代码算法",
		description: "编程语言、框架、自动化与科学计算",
		directorySummary: "面向工程实践的编程语言，工程库框架使用说明，算法题相关内容。",
		meta: "面向工程实践的语言、框架、算法与 AI 系统能力",
		order: 20,
		topNav: true,
		children: [
			{
				segment: "algorithm",
				label: "算法刷题",
				description: "LeetCode 常见题目以及一些笔记",
				children: [
					{
						segment: "Hot100",
						label: "LeetCode Hot100",
					},
					{
						segment: "代码随想录",
						label: "代码随想录",
					},
				],
			},
			{
				segment: "api",
				label: "API 速查",
				description: "一些库的常用 API 笔记，用于查询",
			},
			{
				segment: "language",
				label: "编程语言",
				description: "基于编程语言相关的笔记",
			},
			{
				segment: "lib",
				label: "框架库",
				description: "第三方框架或库的使用笔记",
			},
		],
	},
	{
		segment: "minds",
		label: "随想记录",
		shortLabel: "随想",
		description: "想到什么写什么，和知识无关",
		directorySummary: "记录实习经历、日常感想与阶段性回顾。",
		meta: "记录实习经历、日常感想与阶段性回顾",
		order: 40,
		topNav: true,
		children: [
			{
				segment: "internship",
				label: "实习总结",
				description: "实习经历复盘",
			},
		],
	},
	{
		segment: "tools",
		label: "工具使用",
		shortLabel: "工具",
		description: "容器、环境、基础设施与工作流实践",
		directorySummary: "开发常见工具使用的笔记。",
		meta: "围绕开发环境、部署链路与协作流程的工具使用记录",
		order: 30,
		topNav: true,
		children: [
			{
				segment: "common",
				label: "常用工具",
				description: "日常开发工具的使用指南",
			},
			{
				segment: "others",
				label: "其他工具",
				description: "其他补充工具记录",
			},
			{
				segment: "workflow",
				label: "工作流",
				description: "工作流与非工具使用说明等",
			},
		],
	},
] satisfies readonly NoteNavigationNode[];

function flattenNavigationTree(
	nodes: readonly NoteNavigationNode[],
	parentPath = "",
	accumulator: Record<string, PathNavigationMeta> = {}
): Record<string, PathNavigationMeta> {
	for (const node of nodes) {
		const path = parentPath ? `${parentPath}/${node.segment}` : node.segment;
		if (accumulator[path]) {
			throw new Error(`Duplicate navigation path: ${path}`);
		}

		const { segment: _segment, children, ...meta } = node;
		accumulator[path] = meta;

		if (children?.length) {
			flattenNavigationTree(children, path, accumulator);
		}
	}

	return accumulator;
}

export const topLevelNoteNavigation = [...noteNavigationTree].sort(compareByOrder);
export const topLevelNoteNavigationSegments = topLevelNoteNavigation.map((node) => node.segment);
export const pathNavigationMeta = flattenNavigationTree(noteNavigationTree);

export function getTopLevelNoteNavigation() {
	return topLevelNoteNavigation;
}

export function getFallbackNavigationLabel(path: string): string {
	const lastSegment = path.split("/").at(-1) ?? path;
	return segmentToTitle(lastSegment);
}
