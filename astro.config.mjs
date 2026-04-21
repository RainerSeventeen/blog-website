// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
	integrations: [
		starlight({
			title: "Rainer Notes",
			description: "以 content/note 为唯一内容源的个人知识库。",
			logo: {
				light: "/src/assets/logo-light.svg",
				dark: "/src/assets/logo-dark.svg",
				replacesTitle: true,
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
			customCss: ["./src/assets/landing.css"],
			sidebar: [
				{
					label: "深度学习",
					autogenerate: { directory: "deep-learning" },
				},
				{
					label: "代码算法",
					autogenerate: { directory: "code-algorithm" },
				},
				{
					label: "工具使用",
					autogenerate: { directory: "tools" },
				},
			],
			components: {
				PageTitle: "./src/components/CustomPageTitle.astro",
			},
		}),
	],
});
