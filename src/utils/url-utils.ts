function joinUrl(...parts: string[]): string {
  return parts.join("/").replace(/\/+/g, "/");
}

export function baseUrl(path: string) {
  return joinUrl("", import.meta.env.BASE_URL, path);
}

export function blogCoverUrl(contentPath: string, blogName: string): string {
  if (!contentPath) {
    return "";
  }

  if (contentPath.startsWith("http") || contentPath.startsWith("/")) {
    return contentPath;
  }

  const normalizedPath = contentPath.startsWith("./")
    ? contentPath.slice(2)
    : contentPath;

  return joinUrl("content/note", blogName, normalizedPath);
}
