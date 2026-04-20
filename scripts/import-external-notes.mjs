import fs from "node:fs/promises";
import path from "node:path";

const outputRoot = "/Users/rainer/MyFiles/Code/blog-website-new/src/content/blog";

const sources = [
  {
    title: "工具使用手册",
    description: "沉淀常用开发工具、环境配置与实际工作流，方便后续快速查找和复用。",
    outputDir: "tools-manual",
    order: 40,
    sourceRoot: "/Users/rainer/MyFiles/Blog/工具使用手册",
    files: [
      { sourceName: "Conda 使用指南.md", slug: "conda-guide" },
      { sourceName: "Docker 使用指南.md", slug: "docker-guide" },
      { sourceName: "VibeCoding 经验分享.md", slug: "vibe-coding-workflow" },
      { sourceName: "文献管理规则.md", slug: "paper-notes-rules" },
      { sourceName: "服务器使用指南.md", slug: "server-guide" },
    ],
  },
  {
    title: "代码基础",
    description: "收纳常用语言、框架和数据结构的基础 API 与实现细节，作为日常编码速查表。",
    outputDir: "code-basics",
    order: 50,
    sourceRoot: "/Users/rainer/MyFiles/Blog/代码基础",
    files: [
      { sourceName: "API of Numpy.md", slug: "numpy-api" },
      { sourceName: "API of PyTorch.md", slug: "pytorch-api" },
      { sourceName: "API of Python in LeetCode.md", slug: "python-leetcode-api" },
      { sourceName: "C++标准容器.md", slug: "cpp-stl" },
    ],
  },
  {
    title: "项目相关",
    description: "集中整理个人项目的设计思路、实现路径与阶段性总结，方便回顾与对外展示。",
    outputDir: "project-notes",
    order: 60,
    sourceRoot: "/Users/rainer/MyFiles/Blog/项目相关",
    files: [
      { sourceName: "LeetCodeSolution 简介.md", slug: "leetcode-solution-intro" },
      { sourceName: "PaperTracker 简介.md", slug: "paper-tracker-intro" },
      { sourceName: "多模态RAG 项目.md", slug: "multimodal-rag-project" },
    ],
  },
];

function escapeFrontmatter(value) {
  return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function trimOptionalText(value) {
  const normalized = value?.trim();
  return normalized ? normalized : undefined;
}

function detectTitle(body, fallback) {
  const match = body.match(/^#\s+(.+)$/m);
  return trimOptionalText(match?.[1]) || fallback;
}

function detectDescription(body, fallback) {
  const cleaned = body
    .split("\n")
    .map((line) => line.trim())
    .filter(
      (line) =>
        line &&
        !line.startsWith("#") &&
        !line.startsWith("```") &&
        !line.startsWith("|") &&
        !line.startsWith("![]"),
    )
    .map((line) => line.replace(/^>\s*/, "").replace(/^[-*+]\s*/, "").replace(/`/g, ""))
    .find((line) => line.length > 16);

  return (cleaned || fallback).slice(0, 120);
}

function createFrontmatter(fields) {
  return [
    "---",
    ...fields,
    "---",
    "",
  ].join("\n");
}

async function writeSectionIndex(source) {
  const outputDir = path.join(outputRoot, source.outputDir);
  const outputPath = path.join(outputDir, "index.md");
  const now = new Date().toISOString().slice(0, 10);
  const body = [
    `# ${source.title}`,
    "",
    source.description,
    "",
    "这个分类页会自动收纳对应源目录下的笔记，方便从站点中统一浏览。",
    "",
  ].join("\n");

  const frontmatter = createFrontmatter([
    `title: "${escapeFrontmatter(source.title)}"`,
    `navTitle: "${escapeFrontmatter(source.title)}"`,
    `description: "${escapeFrontmatter(source.description)}"`,
    `pubDate: ${now}`,
    "draft: false",
    `slugId: "${source.outputDir}/index"`,
    `category: "${source.outputDir}"`,
    `section: "${source.outputDir}"`,
    `sourcePath: "${escapeFrontmatter(source.sourceRoot)}"`,
    `order: ${source.order}`,
  ]);

  await fs.mkdir(outputDir, { recursive: true });
  await fs.writeFile(outputPath, `${frontmatter}${body}`);
}

async function writeArticle(source, fileConfig) {
  const sourcePath = path.join(source.sourceRoot, fileConfig.sourceName);
  const outputPath = path.join(outputRoot, source.outputDir, `${fileConfig.slug}.md`);
  const raw = await fs.readFile(sourcePath, "utf8");
  const stats = await fs.stat(sourcePath);
  const title = detectTitle(raw, fileConfig.sourceName.replace(/\.md$/, ""));
  const description = detectDescription(raw, `${title} 笔记`);
  const pubDate = stats.mtime.toISOString().slice(0, 10);

  const frontmatter = createFrontmatter([
    `title: "${escapeFrontmatter(title)}"`,
    `description: "${escapeFrontmatter(description)}"`,
    `pubDate: ${pubDate}`,
    "draft: false",
    `slugId: "${source.outputDir}/${fileConfig.slug}"`,
    `category: "${source.outputDir}"`,
    `section: "${source.outputDir}"`,
    `sourcePath: "${escapeFrontmatter(path.relative("/Users/rainer/MyFiles/Blog", sourcePath).replace(/\\/g, "/"))}"`,
  ]);

  await fs.writeFile(outputPath, `${frontmatter}${raw.trimStart()}\n`);
}

async function main() {
  for (const source of sources) {
    const outputDir = path.join(outputRoot, source.outputDir);
    await fs.rm(outputDir, { recursive: true, force: true });
    await writeSectionIndex(source);

    for (const fileConfig of source.files) {
      await writeArticle(source, fileConfig);
    }
  }

  console.log(`Imported ${sources.length} external note sections.`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
