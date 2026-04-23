export const profile = {
  avatar: "http://oss.rainerseventeen.cn/blog/basic/personal_pic_colorful.jpg",
  name: "Rainer",
  motto: "一个普通的 EE 学生。",
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
