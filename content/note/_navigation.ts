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
		description: "基础、论文、训练与面试知识的系统整理。",
		directorySummary: "基础概念、论文脉络、训练方法和面试题解都从这里进入。",
		meta: "从核心概念到论文脉络，再到面试与工程问题。",
		order: 10,
		topNav: true,
		children: [
			{
				segment: "foundations",
				label: "基础概念",
				description: "Loss、Embedding、Normalization 基础知识。",
				directorySummary: "从基础模块与核心概念入手，快速建立深度学习知识底座。",
			},
			{
				segment: "papers",
				label: "论文精读",
				description: "Transformer、多模态与训练方法的论文拆解与脉络整理。",
				directorySummary: "按研究方向整理论文阅读笔记，方便先看主题再进入具体论文。",
				children: [
					{
						segment: "transformers",
						label: "Transformer",
						directorySummary: "从注意力机制到经典模型，按 Transformer 主线组织论文阅读入口。",
					},
					{
						segment: "training",
						label: "训练方法",
						directorySummary: "聚焦训练策略和微调方法，适合和工程实践一起对照阅读。",
					},
					{
						segment: "multimodal",
						label: "多模态",
						directorySummary: "这一组收录多模态模型的关键论文，适合按模型路线串联阅读。",
					},
				],
			},
			{
				segment: "interview",
				label: "面试题库",
				description: "覆盖深度学习、工程、RAG 与 Python 的面试问题与答案。",
				directorySummary: "先看问题总览，再进入题解目录按专题深入准备。",
				children: [
					{
						segment: "answer",
						label: "题解",
						directorySummary: "题解按专题分组，适合按模块复习或针对性补齐薄弱环节。",
						children: [
							{
								segment: "01-engineering",
								label: "工程基础",
								directorySummary: "聚焦工程能力与系统实现，是面试中最容易追问细节的一组内容。",
							},
							{
								segment: "02-ml-dl",
								label: "机器学习与深度学习",
								directorySummary: "这一组覆盖机器学习和深度学习基础，是大多数面试的核心板块。",
							},
							{
								segment: "03-transformer-llm",
								label: "Transformer 与 LLM",
								directorySummary: "适合系统回顾 Transformer 结构、注意力机制以及 LLM 关键概念。",
							},
							{
								segment: "04-training",
								label: "训练与对齐",
								directorySummary: "重点覆盖训练流程、优化策略与模型对齐，是近几年高频考点。",
							},
							{
								segment: "05-models",
								label: "模型专题",
								directorySummary: "这一组适合按模型专题横向对比，补足特定方向的知识空白。",
							},
							{
								segment: "06-rag-agent",
								label: "RAG 与 Agent",
								directorySummary: "聚焦大模型应用侧能力，适合准备系统设计和应用落地相关问题。",
							},
							{
								segment: "python",
								label: "Python",
								directorySummary: "适合单独回顾 Python 语法、对象模型和常见标准库用法。",
							},
						],
					},
				],
			},
		],
	},
	{
		segment: "code-algorithm",
		label: "代码算法",
		shortLabel: "代码算法",
		description: "编程语言、框架、自动化与科学计算。",
		directorySummary: "面向工程实践的语言、框架、算法与 AI 系统能力都从这里进入。",
		meta: "面向工程实践的语言、框架、算法与 AI 系统能力。",
		order: 20,
		topNav: true,
		children: [
			{
				segment: "programming",
				label: "编程语言",
				description: "Python、C++ 与常用接口的速查和题解沉淀。",
				directorySummary: "以语言和基础接口为核心，适合快速查阅常见 API 与代码片段。",
			},
			{
				segment: "frameworks",
				label: "框架实践",
				description: "PyTorch 等开发框架的 API 整理与使用经验。",
				directorySummary: "这里主要整理框架 API 和记忆成本高、但高频使用的工程细节。",
			},
			{
				segment: "scientific-computing",
				label: "科学计算",
				description: "NumPy 等科学计算工具的使用方式与常见模式。",
				directorySummary: "聚焦科学计算工具链，方便按主题查找矩阵、数组和数值计算相关内容。",
			},
			{
				segment: "automation",
				label: "自动化",
				description: "Paper Tracker、脚本工具与流程自动化实践。",
				directorySummary: "这一组收录自动化脚本、流程编排和信息获取相关实践内容。",
			},
			{
				segment: "learning-systems",
				label: "学习系统",
				description: "LeetCode、刷题与知识训练系统的组织方式。",
				directorySummary: "关注学习方法、训练闭环和长期积累系统的组织方式。",
			},
			{
				segment: "ai-systems",
				label: "AI 系统",
				description: "RAG、多模态与智能应用系统的项目实践。",
				directorySummary: "适合按系统视角查看多模态、检索增强和应用落地相关内容。",
			},
		],
	},
	{
		segment: "minds",
		label: "随想记录",
		shortLabel: "随想",
		description: "实习总结、碎碎念与阶段性思考的收录。",
		directorySummary: "记录实习经历、日常感想与阶段性回顾，按时间线自然积累。",
		meta: "实习总结与碎碎念的长期积累。",
		order: 40,
		topNav: true,
	},
	{
		segment: "tools",
		label: "工具使用",
		shortLabel: "工具",
		description: "容器、环境、基础设施与工作流实践。",
		directorySummary: "围绕开发环境、部署链路和日常协作流程整理的工具使用记录。",
		meta: "围绕开发环境、部署链路与协作流程的工具使用记录。",
		order: 30,
		topNav: true,
		children: [
			{
				segment: "environments",
				label: "环境配置",
				description: "Conda、Python 环境与依赖管理的常用方案。",
				directorySummary: "聚焦开发环境和依赖管理，适合快速查找环境搭建与维护方式。",
			},
			{
				segment: "containers",
				label: "容器与镜像",
				description: "Docker、镜像构建、容器运行与 Compose 使用。",
				directorySummary: "从基础命令到镜像构建与运行方式，这里按容器主题集中整理。",
			},
			{
				segment: "infrastructure",
				label: "服务器与部署",
				description: "服务器使用、部署流程与基础设施相关实践。",
				directorySummary: "聚焦服务端环境、部署流程与基础设施使用中的常见操作和经验。",
			},
			{
				segment: "workflow",
				label: "工作流",
				description: "Vibe Coding、自动化与日常开发协作流程。",
				directorySummary: "适合回顾日常开发中的协作方法、自动化流程和效率工具组合。",
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
