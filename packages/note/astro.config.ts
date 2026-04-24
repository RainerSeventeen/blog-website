import { fileURLToPath } from "node:url";
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { getTopLevelNoteNavigation } from "../../content/note/_navigation";

const noteSidebar = getTopLevelNoteNavigation()
	.filter((node) => node.topNav !== false)
	.map((node) => ({
		label: node.label ?? node.segment,
		autogenerate: { directory: node.segment },
	}));

type MarkdownNode = {
	type: string;
	value?: string;
	children?: MarkdownNode[];
	position?: {
		start?: { offset?: number };
		end?: { offset?: number };
	};
	data?: unknown;
};

function remarkStandaloneDoubleDollarMath() {
	return (tree: MarkdownNode, file: { value?: unknown }) => {
		const source = typeof file.value === "string" ? file.value : "";

		function visit(node: MarkdownNode) {
			if (!node.children) return;

			for (let index = 0; index < node.children.length; index += 1) {
				const child = node.children[index];
				if (child.type === "paragraph" && child.children?.length === 1) {
					const inlineMath = child.children[0];
					const start = child.position?.start?.offset;
					const end = child.position?.end?.offset;
					const raw = typeof start === "number" && typeof end === "number" ? source.slice(start, end).trim() : "";

					if (inlineMath.type === "inlineMath" && /^\$\$[\s\S]*\$\$$/.test(raw)) {
						node.children[index] = {
							type: "math",
							value: inlineMath.value,
							position: child.position,
							data: {
								hName: "pre",
								hChildren: [
									{
										type: "element",
										tagName: "code",
										properties: { className: ["language-math", "math-display"] },
										children: [{ type: "text", value: inlineMath.value }],
									},
								],
							},
						};
						continue;
					}
				}

				visit(child);
			}
		}

		visit(tree);
	};
}

export default defineConfig({
	markdown: {
		remarkPlugins: [[remarkMath, { singleDollarTextMath: true }], remarkStandaloneDoubleDollarMath],
		rehypePlugins: [[rehypeKatex, { strict: false, throwOnError: false }]],
	},
	vite: {
		resolve: {
			alias: {
				"@components": fileURLToPath(new URL("./src/components", import.meta.url)),
				"@content": fileURLToPath(new URL("../../content", import.meta.url)),
			},
		},
	},
	integrations: [
		starlight({
			title: "RainerSeventeen | Note",
			description: "以 content/note 为唯一内容源的个人知识库。",
			logo: {
				light: "/src/assets/rainer-rain.svg",
				dark: "/src/assets/rainer-rain.svg",
				alt: "RainerSeventeen rain cloud icon",
				replacesTitle: false,
			},
			lastUpdated: true,
			editLink: {
				baseUrl: "https://github.com/RainerSeventeen/blog-website-new/edit/main/",
			},
			social: [
				{
					icon: "github",
					label: "GitHub",
					href: "https://github.com/RainerSeventeen/blog-website-new",
				},
			],
			pagefind: true,
			customCss: ["./src/assets/landing.css"],
			expressiveCode: {
				shiki: {
					langAlias: {
						C: "c",
						"C++": "cpp",
						Cpp: "cpp",
						CPP: "cpp",
						"c++": "cpp",
						cplusplus: "cpp",
						Bash: "bash",
						BASH: "bash",
					},
				},
			},
			routeMiddleware: "./src/starlight/navigation-middleware.ts",
			sidebar: noteSidebar,
			components: {
				Header: "./src/components/CustomHeader.astro",
				SiteTitle: "./src/components/CustomSiteTitle.astro",
				PageTitle: "./src/components/CustomPageTitle.astro",
				Sidebar: "./src/components/CustomSidebar.astro",
			},
		}),
	],
});
