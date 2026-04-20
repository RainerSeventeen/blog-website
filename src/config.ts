import type { ProfileConfig, SiteConfig } from "./types/config";

export const siteConfig: SiteConfig = {
  title: "RainerSeventeen",
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
  crossSiteLinks: {
    mainSite: "#",         // TODO: 主站域名，如 https://xxx.com
    noteSite: "/",         // 笔记站，开发时指向本站根路径
    projectSite: "#",      // TODO: 项目站域名，如 https://project.xxx.com
  },
};

export const profileConfig: ProfileConfig = {
  avatar: "http://oss.rainerseventeen.cn/blog/basic/personal_pic_colorful.jpg",
  name: "Rainer",
  motto: "一个普通的 EE 学生",
  description:
    "要做的事情还有很多，要学的东西还有很多，让我们开始吧。",
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
      name: "Email",
      url: "mailto:rainerseventeeen@outlook.com",
      icon: "fa6-solid:envelope",
      label: "Email",
    },
    {
      name: "RSS",
      url: "/rss.xml",
      icon: "simple-icons:rss",
      label: "RSS",
    },
  ],
};
