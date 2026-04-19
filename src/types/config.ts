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
}

export type ProfileConfig = {
    avatar: string;
    name: string;
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
