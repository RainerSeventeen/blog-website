import type { ProfileConfig, SiteConfig } from "./types/config";

export const siteConfig: SiteConfig = {
  title: "Deep Learning Playbook",
  subTitle: "AI 笔记、论文精读与面试整理",
  favicon: "/favicon/favicon.ico",
  pageSize: 6,
  toc: {
    enable: true,
    depth: 3,
  },
  search: {
    enable: true,
  },
};

export const profileConfig: ProfileConfig = {
  avatar: "assets/Motues.jpg",
  name: "Rainer",
  description:
    "面向深度学习 / AI 算法岗位准备的知识库，按基础原理、论文精读、面试问题与课程笔记组织。",
  indexPage: "/",
  startYear: 2024,
  links: [
    {
      name: "GitHub",
      url: "https://github.com",
      icon: "fa6-brands:github",
      label: "GitHub",
    },
    {
      name: "RSS",
      url: "/rss.xml",
      icon: "simple-icons:rss",
      label: "RSS",
    },
  ],
};
