# AGENTS.md

AI 助手在此仓库中工作时的指导说明。

## 项目概览

这是一个基于 **Astro + pnpm workspaces** 的个人博客多站项目，包含：

- `packages/home` — 主站（Astro + Tailwind CSS v4，静态输出）
- `packages/note` — 笔记站（Astro Starlight，支持 KaTeX 数学公式）
- `content/` — 所有站点共享的内容源，**note 内容只在此处维护**

## 关键规则

### 内容修改
- 笔记内容统一修改 `content/note/` 目录下的 Markdown/MDX 文件，**不要**直接修改 `packages/note/src/content/`（该目录由脚本同步生成）
- 站点元信息（标题、描述、域名）统一在 `content/site-config.ts` 中修改

### 导航
- note 站侧边栏导航由 `content/note/_navigation.ts` 控制，增删顶级目录时需同步更新此文件

### 新增站点
- 在 `packages/` 下新建包目录
- 在 `content/site-config.ts` 的 `SITE_CONFIG_REGISTRY` 中添加配置
- 在 `scripts/dev.sh` 的 `SITES` 数组中添加条目

## 命令参考

```bash
pnpm dev              # 启动所有已就绪站点
pnpm dev:note         # 只启动 note（port 4321）
pnpm dev:home         # 只启动 home（port 4320）
pnpm build:all        # 构建所有包
pnpm build:note
pnpm build:home
```

## 技术约束

- Node.js 包管理器：**pnpm**（不要使用 npm/yarn）
- Astro 版本：v6
- Tailwind CSS：v4（home 站，配置在 `vite.plugins` 中，不使用 PostCSS 插件）
- TypeScript 路径别名：`@components` → `packages/note/src/components`，`@content` → `content/`

## 不要做的事

- 不要提交 `node_modules/` 或 `dist/` 目录
- 不要修改 `pnpm-lock.yaml`（除非确实在更新依赖）
- 不要在 `packages/note/src/content/` 中直接创建或编辑内容文件
