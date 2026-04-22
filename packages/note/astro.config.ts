import { fileURLToPath } from "node:url";
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import { getTopLevelNoteNavigation } from "../../content/note/_navigation";

const noteSidebar = getTopLevelNoteNavigation()
	.filter((node) => node.topNav !== false)
	.map((node) => ({
		label: node.label ?? node.segment,
		autogenerate: { directory: node.segment },
	}));

export default defineConfig({
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
			title: "RainerSeventeen",
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
			routeMiddleware: "./src/starlight/navigation-middleware.ts",
			sidebar: noteSidebar,
			components: {
				Header: "./src/components/CustomHeader.astro",
				PageTitle: "./src/components/CustomPageTitle.astro",
				Sidebar: "./src/components/CustomSidebar.astro",
			},
		}),
	],
});
