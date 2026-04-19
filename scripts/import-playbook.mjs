import fs from "node:fs/promises";
import path from "node:path";

const sourceRoot = "/Users/rainer/MyFiles/Code/deep-learning-playbook";
const outputRoot = "/Users/rainer/MyFiles/Code/blog-website-new/src/content/blog";
const allowedRoots = new Set(["foundations", "papers", "interview", "courses"]);

function slugifySegment(segment) {
  return segment
    .replace(/\.[^.]+$/, "")
    .trim()
    .toLowerCase()
    .replace(/[_\s]+/g, "-")
    .replace(/&/g, "and")
    .replace(/[^a-z0-9-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

function normalizeOutputSegments(relativeNoExt) {
  const segments = relativeNoExt.split(path.sep).map(slugifySegment);
  const lastSegment = segments.at(-1);

  if (lastSegment === "readme" || lastSegment === "index") {
    segments[segments.length - 1] = "index";
  }

  return segments;
}

function parseFrontmatter(raw) {
  if (!raw.startsWith("---\n")) {
    return { data: {}, body: raw };
  }

  const endIndex = raw.indexOf("\n---\n", 4);
  if (endIndex === -1) {
    return { data: {}, body: raw };
  }

  const yamlBlock = raw.slice(4, endIndex).trim();
  const body = raw.slice(endIndex + 5);
  const data = {};

  for (const line of yamlBlock.split("\n")) {
    const separatorIndex = line.indexOf(":");
    if (separatorIndex === -1) {
      continue;
    }

    const key = line.slice(0, separatorIndex).trim();
    const value = line.slice(separatorIndex + 1).trim().replace(/^"(.*)"$/, "$1");
    data[key] = value;
  }

  return { data, body };
}

function detectTitle(body, fallback) {
  const match = body.match(/^#\s+(.+)$/m);
  return (match?.[1] || fallback).trim();
}

function detectDescription(body, fallback) {
  const cleaned = body
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#") && !line.startsWith("```") && !line.startsWith("|"))
    .map((line) => line.replace(/^>\s*/, "").replace(/^[-*+]\s*/, "").replace(/`/g, ""))
    .find((line) => line.length > 12);

  return (cleaned || `${fallback} 笔记`).slice(0, 120);
}

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walk(fullPath)));
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      files.push(fullPath);
    }
  }

  return files;
}

async function main() {
  const allFiles = await walk(sourceRoot);
  const sourceFiles = allFiles.filter((file) => allowedRoots.has(path.relative(sourceRoot, file).split(path.sep)[0]));

  await fs.rm(outputRoot, { recursive: true, force: true });
  await fs.mkdir(outputRoot, { recursive: true });

  let imported = 0;
  let skipped = 0;
  const seenOutputPaths = new Set();

  for (const file of sourceFiles) {
    const relativePath = path.relative(sourceRoot, file);
    const stats = await fs.stat(file);

    if (stats.size === 0) {
      skipped += 1;
      continue;
    }

    const raw = await fs.readFile(file, "utf8");
    const { data, body } = parseFrontmatter(raw);

    const relativeNoExt = relativePath.replace(/\.md$/, "");
    const outputSegments = normalizeOutputSegments(relativeNoExt);

    if (outputSegments.some((segment) => !segment)) {
      throw new Error(`Invalid output path generated from "${relativePath}"`);
    }

    const outputPath = path.join(outputRoot, `${outputSegments.join("/")}.md`);

    if (seenOutputPaths.has(outputPath)) {
      throw new Error(`Output path conflict detected: "${outputPath}"`);
    }

    seenOutputPaths.add(outputPath);
    const slugId = outputSegments.join("/");
    const section = outputSegments[0];
    const category =
      outputSegments.length > 1 ? outputSegments[outputSegments.length - 2] : outputSegments[0];
    const fallbackTitle = path.basename(relativeNoExt);
    const title = detectTitle(body, fallbackTitle);
    const description = detectDescription(body, title).replace(/"/g, '\\"');
    const pubDate = stats.mtime.toISOString().slice(0, 10);
    const titleValue = title.replace(/"/g, '\\"');
    const sourceValue = relativePath.replace(/\\/g, "/");

    const frontmatter = [
      "---",
      `title: "${titleValue}"`,
      `description: "${description}"`,
      `pubDate: ${pubDate}`,
      "draft: false",
      `slugId: "${slugId}"`,
      `category: "${category}"`,
      `section: "${section}"`,
      `sourcePath: "${sourceValue}"`,
      data.paper ? `paper: "${String(data.paper).replace(/"/g, '\\"')}"` : null,
      "---",
      "",
    ]
      .filter(Boolean)
      .join("\n");

    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, `${frontmatter}${body.trimStart()}\n`);
    imported += 1;
  }

  console.log(`Imported ${imported} files to ${outputRoot}`);
  if (skipped > 0) {
    console.log(`Skipped ${skipped} empty files`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
