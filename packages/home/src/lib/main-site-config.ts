export const profile = {
  avatar: "http://oss.rainerseventeen.cn/blog/basic/personal_pic_colorful.jpg",
  name: "RainerSeventeen",
  motto: "Some things were just meant to be.",
  description: "要做的事情还有很多，要学的东西还有很多，让我们开始吧。",
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
  name: string;
  href: string;
  description: string;
  owner: string;
  stars: number;
  forks: number;
  license: string;
  language: string;
}

export interface AboutPageMarkdownSection {
  title: string;
  paragraphs: string[];
  bullets?: string[];
}

export const destinations = [
  {
    label: "Note",
    href: "https://note.rainerseventeen.com",
    description: "内容与知识",
    ctaLabel: "前往笔记站",
    external: true,
  },
  {
    label: "Project",
    href: "https://project.rainerseventeen.com",
    description: "构建与作品",
    ctaLabel: "前往项目站",
    external: true,
  },
  {
    label: "Lab",
    href: "https://lab.rainerseventeen.com",
    description: "实验与探索",
    ctaLabel: "前往实验站",
    external: true,
  },
];

export const featuredRepos: FeaturedRepo[] = [
  {
    name: "paper-tracker",
    href: "https://github.com/RainerSeventeen/paper-tracker",
    description: "把论文检索、去重、总结和多格式输出串起来的自动化工具。",
    owner: "RainerSeventeen",
    stars: 43,
    forks: 13,
    license: "MIT",
    language: "Python",
  },
  {
    name: "blog-website-new",
    href: "https://github.com/RainerSeventeen/blog-website-new",
    description: "当前这套个人站点的 monorepo，统一承载主站、笔记和后续扩展。",
    owner: "RainerSeventeen",
    stars: 0,
    forks: 0,
    license: "No License",
    language: "Astro",
  },
  {
    name: "dive-into-deep-learning",
    href: "https://github.com/RainerSeventeen/dive-into-deep-learning",
    description: "围绕深度学习基础与实现的代码仓库，用来沉淀训练细节和实验片段。",
    owner: "RainerSeventeen",
    stars: 0,
    forks: 0,
    license: "No License",
    language: "Jupyter Notebook",
  },
  {
    name: "leetcode-solutions",
    href: "https://github.com/rainerWJY/leetcode-solutions",
    description: "以 Markdown 和自动化脚本为核心的题解知识库，强调回顾和检索能力。",
    owner: "rainerWJY",
    stars: 0,
    forks: 0,
    license: "No License",
    language: "Markdown",
  },
];

export const aboutPage = {
  heroTitle: "关于",
  heroDescription:
    "我把个人网站当作一个持续生长的工作台：一边学习、一边实现、一边把过程整理成可以复用的知识。",
  teaser:
    "学点新东西，随手写写笔记，也捣鼓点稀奇古怪的小项目～",
  summary: [
    "欢迎来到我的个人网站！",
  ],
  markdownSections: [
    {
      title: "好好学习",
      paragraphs: [
        "学习对我来说不是把材料看完，而是把一个问题反复拆开，直到它能被讲清楚、写清楚、做出来。相比只记结论，我更在意为什么这样设计、为什么这样训练、为什么这样组织系统。",
        "最近关注得更多的是深度学习基础、模型结构理解，以及把零散知识重新整理成可检索、可复盘的内容体系。写笔记、做实验、补实现，基本都围绕这一件事展开。",
      ],
      bullets: [
        "把知识点从“看过”推进到“能解释、能复现、能复用”",
        "持续整理 Note 站内容，让学习记录本身也能成为工具",
        "把抽象概念尽量落到代码、实验和真实问题上",
      ],
    },
    {
      title: "多多折腾",
      paragraphs: [
        "光学不做，很容易停留在“好像懂了”。所以我会不停折腾项目、站点、脚本和自动化流程，把一个个想法接成可以运行的东西，再看看它们哪里脆、哪里丑、哪里还能继续长。",
        "这个博客、笔记站和一些实验项目，本质上都是同一件事的不同切面：一边做，一边修，一边把经验沉淀成下一次还能继续用的积木。",
      ],
      bullets: [
        "把写作、归档、检索和开发流程尽量串起来",
        "优先做可维护、可扩展、能反复迭代的小系统",
        "用真实项目检验想法，而不是只停留在设想里",
      ],
    },
    {
      title: "稍稍玩玩",
      paragraphs: [
        "我也想给自己留一点不那么紧绷的空间。这里的“玩”不是完全脱离正事，而是允许自己因为好奇去试一些小东西，比如一个新页面写法、一段顺手的自动化，或者一个暂时没有明确用途的小实验。",
        "很多真正有意思的东西，往往就是从这些看起来不那么正式的小折腾里长出来的。比起把一切都做成功能列表，我更希望这个站点还保留一点探索感和个人气味。",
      ],
      bullets: [
        "给好奇心留位置，也给长期投入留弹性",
        "允许自己做一些暂时不功利但足够有趣的尝试",
        "让网站不只是展示结果，也保留探索过程",
      ],
    },
  ] satisfies AboutPageMarkdownSection[],
} as const;
