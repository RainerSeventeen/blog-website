import rss from "@astrojs/rss";
import { getBlogEntries } from "@utils/content-utils";

export async function GET(context) {
  const posts = await getBlogEntries();

  return rss({
    title: "RainerSeventeen",
    description: "学习笔记、论文精读与生活随笔",
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.pubDate,
      description: post.data.description,
      link: `/blog/${post.id}/`,
    })),
  });
}
