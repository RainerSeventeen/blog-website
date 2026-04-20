export type SiteConfig = {
    title: string;
    subTitle: string;
    favicon: string;
    pageSize: number;
    toc: {
        enable: boolean;
        depth: number;
    };
    search: {
        enable: boolean;
    };
    crossSiteLinks: {
        mainSite: string;
        noteSite: string;
        projectSite: string;
    };
}

export type ProfileConfig = {
    avatar: string;
    name: string;
    motto: string;
    description: string;
    indexPage: string;
    startYear: number;
    links: {
        name: string;
        url: string;
        icon: string;
        label: string;
    }[];
}
