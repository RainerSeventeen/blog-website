// @ts-check
import { existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const docsDir = path.join(rootDir, "src", "content", "docs");
const fallbackDocsDir = path.join(rootDir, "content", "note");

function getTopLevelDocDirectories() {
	const contentDir = existsSync(docsDir) ? docsDir : fallbackDocsDir;

	if (!existsSync(contentDir)) {
		return [];
	}

	return readdirSync(contentDir, { withFileTypes: true })
		.filter((entry) => entry.isDirectory())
		.map((entry) => entry.name)
		.sort((a, b) => a.localeCompare(b));
}

const sidebarDirectories = getTopLevelDocDirectories();

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
			sidebar: sidebarDirectories.map((directory) => ({
				label: directory,
				autogenerate: { directory },
			})),
		}),
	],
});
