const ROUTE_LABELS: Record<string, string> = {
	"deep-learning": "深度学习",
	"code-algorithm": "代码算法",
	tools: "工具使用",
	foundations: "基础概念",
	papers: "论文精读",
	multimodal: "多模态",
	transformers: "Transformer",
	training: "训练方法",
	interview: "面试题库",
	answer: "题解",
	"01-engineering": "工程基础",
	"02-ml-dl": "机器学习与深度学习",
	"03-transformer-llm": "Transformer 与 LLM",
	"04-training": "训练与对齐",
	"05-models": "模型专题",
	"06-rag-agent": "RAG 与 Agent",
	python: "Python",
	programming: "编程语言",
	frameworks: "框架实践",
	"scientific-computing": "科学计算",
	automation: "自动化",
	"learning-systems": "学习系统",
	"ai-systems": "AI 系统",
	environments: "环境配置",
	containers: "容器与镜像",
	infrastructure: "服务器与部署",
	workflow: "工作流",
};

function segmentToTitle(segment: string): string {
	return segment
		.replace(/-/g, " ")
		.replace(/\b\w/g, (char) => char.toUpperCase());
}

export function getRouteLabel(slug: string, fallback?: string): string {
	const exactMatch = ROUTE_LABELS[slug];
	if (exactMatch) return exactMatch;

	const lastSegment = slug.split("/").at(-1) ?? slug;
	return ROUTE_LABELS[lastSegment] ?? fallback ?? segmentToTitle(lastSegment);
}
