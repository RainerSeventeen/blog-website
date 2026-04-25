# RainerSeventeen Blog 

一个博客网站，基于 Astro 构建的多站架构，目前包含主站与笔记站，项目站和实验站正在规划中。

具体的部署效果参见 [网站](https://rainerseventeen.cn/)

## 站点一览

| 站点 | 包路径 | 生产域名 | 状态 |
|------|--------|----------|------|
| Home | `packages/home` | rainerseventeen.cn | 已上线 |
| Note | `packages/note` | note.rainerseventeen.cn | 已上线 |
| Project | `packages/project` | project.rainerseventeen.cn | 规划中 |
| Lab | `packages/lab` | lab.rainerseventeen.cn | 规划中 |

- **home** — 个人主站，负责内容导航与各分站入口聚合
- **note** — 知识库，基于 Astro Starlight，支持 KaTeX 数学公式，内容统一维护于 `content/note/`

## 技术栈

- [Astro](https://astro.build/) v6
- [Astro Starlight](https://starlight.astro.build/)（note 站）
- [Tailwind CSS](https://tailwindcss.com/) v4（home 站）
- [pnpm workspaces](https://pnpm.io/workspaces) 多包管理

## 快速开始

```bash
# 安装依赖
pnpm install

# 启动所有已就绪的站点
pnpm dev

# 只启动某个站点
pnpm dev:note
pnpm dev:home

# 构建
pnpm build:all
pnpm build:note
pnpm build:home
```

本地访问地址：

- home → http://localhost:4320
- note → http://localhost:4321

## 目录结构

```
.
├── content/              # 共享内容源
│   ├── note/             # 笔记内容（Markdown/MDX）
│   └── site-config.ts    # 各站点元信息配置
├── packages/
│   ├── home/             # 主站 Astro 应用
│   └── note/             # 笔记站 Astro + Starlight 应用
├── scripts/              # 开发脚本
│   ├── dev.sh            # 多站并行启动入口
│   └── env.sh            # 环境变量注入
└── pnpm-workspace.yaml
```

## 内容管理

笔记内容统一存放于 `content/note/`，note 站在每次构建或启动开发服务器时通过 `sync:docs` 脚本自动同步。侧边栏导航结构由 `content/note/_navigation.ts` 定义。

## 环境变量

各站点 URL 通过环境变量注入，在 `scripts/env.sh` 中配置：

| 变量 | 说明 |
|------|------|
| `PUBLIC_MAIN_SITE_URL` | 主站地址 |
| `PUBLIC_NOTE_SITE_URL` | 笔记站地址 |
| `PUBLIC_PROJECT_SITE_URL` | 项目站地址 |
| `PUBLIC_LAB_SITE_URL` | 实验站地址 |

## 致谢

本项目在开发过程中参考了以下优秀的开源模板，感谢作者的工作：

- [Vncntvx/LandingPage](https://github.com/Vncntvx/LandingPage) — 基于 Astro + Tailwind CSS 的极简个人主页模板
- [Motues/Momo](https://github.com/Motues/Momo) — 基于 Astro 的极简博客模板

## 许可证

[MIT](./LICENSE)
