import type { ImageMetadata } from "astro";
import labArtwork from "../images/destinations/lab.svg";
import noteArtwork from "../images/destinations/note.svg";
import projectArtwork from "../images/destinations/project.svg";

export const profile = {
  avatar: "http://oss.rainerseventeen.cn/blog/basic/personal_pic_colorful.jpg",
  name: "RainerSeventeen",
  motto: "Some things were just meant to be.",
  description: "欢迎来到 RainerSeventeen 主站！",
  links: [
    {
      label: "GitHub",
      href: "https://github.com/RainerSeventeen",
      icon: "github" as const,
    },
    {
      label: "Email",
      href: "mailto:rainerseventeeen@outlook.com",
      icon: "email" as const,
    },
    {
      label: "RSS",
      href: "/rss.xml",
      icon: "rss" as const,
    },
  ],
};

export interface FeaturedRepo {
  owner: string;
  name: string;
  description: string;
}

export interface AboutPageMarkdownSection {
  title: string;
  paragraphs: string[];
  bullets?: string[];
  cards?: AboutPageLinkCard[];
}

export interface AboutPageLinkCard {
  title: string;
  href: string;
  description: string;
  ctaLabel: string;
  domainLabel: string;
  eyebrow: string;
  icon: "steam";
  external: boolean;
}

export interface Destination {
  label: string;
  href: string;
  description: string;
  ctaLabel: string;
  external: boolean;
  domainLabel: string;
  eyebrow: string;
  artwork: ImageMetadata;
  artworkAlt: string;
  accentClass: string;
}

export const destinations: Destination[] = [
  {
    label: "Note",
    href: "https://note.rainerseventeen.com",
    description: "内容与知识",
    ctaLabel: "前往笔记站",
    external: true,
    domainLabel: "note.rainerseventeen.com",
    eyebrow: "Knowledge Base",
    artwork: noteArtwork,
    artworkAlt: "Note 站入口插画",
    accentClass:
      "from-indigo-500/12 via-white to-violet-500/10 dark:from-indigo-400/12 dark:via-slate-900 dark:to-violet-400/10",
  },
  {
    label: "Project",
    href: "https://project.rainerseventeen.com",
    description: "构建与作品",
    ctaLabel: "前往项目站",
    external: true,
    domainLabel: "project.rainerseventeen.com",
    eyebrow: "Build Log",
    artwork: projectArtwork,
    artworkAlt: "Project 站入口插画",
    accentClass:
      "from-sky-500/12 via-white to-cyan-500/10 dark:from-sky-400/12 dark:via-slate-900 dark:to-cyan-400/10",
  },
  {
    label: "Lab",
    href: "https://lab.rainerseventeen.com",
    description: "实验与探索",
    ctaLabel: "前往实验站",
    external: true,
    domainLabel: "lab.rainerseventeen.com",
    eyebrow: "Experiments",
    artwork: labArtwork,
    artworkAlt: "Lab 站入口插画",
    accentClass:
      "from-emerald-500/12 via-white to-teal-500/10 dark:from-emerald-400/12 dark:via-slate-900 dark:to-teal-400/10",
  },
];

export const featuredRepos: FeaturedRepo[] = [
  {
    owner: "RainerSeventeen",
    name: "paper-tracker",
    description: "把论文检索、去重、总结和多格式输出串起来的自动化工具。",
  },
  {
    owner: "RainerSeventeen",
    name: "blog-website-new",
    description: "当前这套个人站点的 monorepo，统一承载主站、笔记和后续扩展。",
  },
  {
    owner: "RainerSeventeen",
    name: "dive-into-deep-learning",
    description: "围绕深度学习基础与实现的代码仓库，用来沉淀训练细节和实验片段。",
  },
  {
    owner: "RainerSeventeen",
    name: "MultiRAG-Doc",
    description: "一个多模态结合 RAG 技术的论文检索项目。",
  },
];

export const aboutPage = {
  heroTitle: "关于",
  heroDescription:
    "Some things were just meant to be.",
  teaser:
    "学点新东西，随手写写笔记，也捣鼓点稀奇古怪的小项目～",
  summary: [
    "总之，让我们开始吧！",
  ],
  markdownSections: [
    {
      title: "关于作者",
      paragraphs: [
        "目前大四，天天缩在寝室阴暗爬行，有时候捣鼓好玩的东西，有时候也学感兴趣的课程，有时候单纯在玩。",
      ],
      bullets: [
        "接着学深度学习中 (已被八股文塞满)",
        "随缘刷刷 LeetCode 算法 (写不动了已经)",
        "看心情维护 GitHub 项目 (虽然好像无人在意咱的 Repo ...)",

      ],
    },
    {
      title: "好好学习",
      paragraphs: [
        "本科花了点时间搞了一下绩点，但是实际上 GPA 也不是很出众，论文也没有发，感觉是没有在一开始明确路线导致的。",
        "不过倒也算不上一事无成，在研究生阶段就早早定好方向，然后对着这一个方向持续深入学习吧。",
      ],
      bullets: [
      ],
    },
    {
      title: "多多折腾",
      paragraphs: [
        "一直有一些稀奇古怪的想法，之前碍于时间成本也没有落地。不过 Agent 时代到来了，让执行变得轻松得多，因此有机会把一些想法落地。这个网站也是折腾的产物之一。",
        "本来是网站是计划直接套模板的，上一版的博客也是，但是总想着利用服务器做一些自动化或者 API 等接口，所以说纯模板没法满足功能。于是就和 Agent 聊天“聊”出来了这个网站。",
        "后续会计划在 Lab 分站中搞点新东西上去，然后把 Project 分站作为已经成熟的实现的展示。",

      ],
      bullets: [
      ],
    },
    {
      title: "稍稍玩玩",
      paragraphs: [
        "叽里咕噜说什么呢，先玩会游戏再工作～",
        "没错啊我就是个臭打游戏的，涉猎略微有些杂，基本所有游戏都是 Steam 上滴。",
        "不过，如果想加好友请务必通过其他方式通知我，一般陌生好友我全部拒绝 (没办法全是忽悠饰品的)"
      ],
      cards: [
        {
          title: "Steam",
          href: "https://steamcommunity.com/profiles/76561199032761649/",
          description: "Steam 个人主页；想加好友的话，记得先通过其他渠道打个招呼。",
          ctaLabel: "打开 Steam 档案",
          domainLabel: "steamcommunity.com",
          eyebrow: "Game Shelf",
          icon: "steam",
          external: true,
        },
      ],
      bullets: [
      ],
    },
  ] satisfies AboutPageMarkdownSection[],
} as const;
