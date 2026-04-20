# AGENTS.md

## 项目定位

- 这是一个基于 `Astro 5` 的静态知识库站点，内容主题集中在 AI、深度学习、论文笔记、工具手册和项目复盘。
- UI 层以 `Astro + TypeScript + Tailwind CSS 4` 为主，局部交互使用 `Svelte`。
- 内容源主要位于 `src/content/blog`，由 `astro:content` 管理，并在构建期生成静态页面和 Pagefind 搜索索引。

## 常用命令

- `pnpm dev`：启动本地开发。
- `pnpm build`：执行 Astro 构建，并对 `dist` 运行 `pagefind --site dist`。
- `pnpm preview`：预览构建产物。
- `pnpm sync:playbook`：从外部 `deep-learning-playbook` 仓库导入 Markdown。
- `pnpm sync:external`：从外部笔记目录导入工具手册、代码基础、项目相关内容。

## 目录与模块

### 1. 路由入口

- `src/pages/index.astro`
  首页。通过 `getContentTree()` 和 `getBlogEntries()` 组装专题入口与最近更新。
- `src/pages/[...page].astro`
  分页列表页。基于 `paginate()` 对文章列表做分页。
- `src/pages/archives.astro`
  根目录页。展示顶层专题。
- `src/pages/blog/[...id].astro`
  统一承载“目录页”和“文章页”两类内容；先 `getStaticPaths()`，再交给 `buildDirectoryPageModel()` / `buildArticlePageModel()`。
- `src/pages/rss.xml.ts`
  RSS 输出。

### 2. 内容模型

- `src/content.config.ts`
  定义唯一的 `blog` collection，内容加载路径是 `src/content/blog/**/*.md`。
- 每篇内容要求至少提供这些 frontmatter：
  `title`、`pubDate`、`slugId`、`category`、`section`。
- 约定：
  - `index.md` 表示目录说明页，不当作普通文章。
  - `slugId` 决定站点路径，必须和内容在知识树中的目标位置一致。
  - `section` 表示顶层专题，例如 `deep-learning-playbook`、`tools-manual`。

### 3. 核心编排层

- `src/utils/content-utils.ts` 是站点的核心服务层。
- 这里负责：
  - 读取和排序文章；
  - 构建内容树 `ContentTree`；
  - 区分目录节点与文章节点；
  - 生成面包屑、目录页模型、文章页模型；
  - 处理上一篇/下一篇、目录统计、专题内导航等派生数据。
- 对站点结构、路径、目录展示逻辑的修改，优先从这里理解和下手。

### 4. 布局与展示层

- `src/layouts/Layout.astro`
  HTML 壳、主题初始化、全局样式和 AOS 初始化。
- `src/layouts/MainPageLayout.astro`
  站点主布局，统一挂载 `Header`、`Footer`、搜索模态。
- `src/components/DirectoryPage.astro`
  目录页展示。
- `src/components/ArticlePage.astro`
  文章页展示，挂载 TOC、上一篇/下一篇、阅读信息。
- `src/components/misc/Search.astro`
  搜索弹窗。开发环境展示假数据，占位用途；真实搜索依赖构建后的 Pagefind 索引。

### 5. Markdown / 内容渲染链

- `astro.config.mjs` 配置了 Markdown 渲染管线。
- 主要插件：
  - `remark-reading-time`：生成阅读字数和阅读时间。
  - `remark-typst`：渲染 `typst` 代码块。
  - `remark-directive-rehype` + `rehype-components`：支持 GitHub 卡片、音乐卡片、引用块、提示框等自定义指令。
  - `rehype-katex` + `remark-math`：公式渲染。
  - `rehype-figure-plugin`：图片/figure 增强。
  - `remark-combined`：自定义内联语法处理。

### 6. 内容同步脚本

- `scripts/import-playbook.mjs`
  将外部 Markdown 仓库内容导入到 `src/content/blog/deep-learning-playbook`。
- `scripts/import-external-notes.mjs`
  将外部笔记目录导入到 `tools-manual`、`code-basics`、`project-notes`。
- 这两个脚本都依赖硬编码的本机绝对路径；如果外部目录不存在，不要贸然运行。

## 修改约定

- 优先修改 `src/**`，不要手动编辑 `dist/**`。
- `reference/**` 是参考实现，不是当前站点的运行主目录；只有在明确做对照迁移时才修改。
- 调整内容结构时，务必同时检查：
  - `slugId` 是否唯一；
  - 目录和文章路径是否冲突；
  - `index.md` 是否仍然正确表示目录说明。
- 调整目录页/文章页逻辑时，注意 `content-utils.ts` 里的树构建和面包屑逻辑是否同步。
- 调整搜索相关逻辑时，记住开发环境没有真实 Pagefind 索引；需要 `pnpm build` 后再验证真实效果。
- 调整 Markdown 插件时，优先验证构建流程，因为这类改动通常在编译期暴露问题。

## 建议验证

- 一般页面或样式修改后：运行 `pnpm build`。
- 涉及内容路由、frontmatter、Markdown 插件、搜索、分页时：必须运行 `pnpm build`。

## 当前仓库链接状态

- `AGENTS.md` 是规范指令文件。
- `CLAUDE.md` 应保持为指向 `AGENTS.md` 的符号链接。
- `.claude/skills` 应保持为指向 `.codex/skills` 的符号链接。
