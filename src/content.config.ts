import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const blogCollection = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/blog" }),
  schema: z.object({
    title: z.string(),
    pubDate: z.date(),
    draft: z.boolean().optional().default(false),
    description: z.string().optional().default(""),
    image: z.string().optional().default(""),
    slugId: z.string(),
    category: z.string(),
    section: z.string(),
    sourcePath: z.string().optional(),
    paper: z.string().optional(),
    navTitle: z.string().optional(),
    order: z.number().optional(),
    pinTop: z.number().optional().default(0),
  }),
});

export const collections = {
  blog: blogCollection,
};
